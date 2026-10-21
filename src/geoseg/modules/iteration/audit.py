"""Objective audits for iterative segmentation.

The audit is the gate that makes the iteration loop repeatable: it detects
under-segmented labels (one label containing two distinct solid colours),
text/background labels, fragmentation, and layer-count mismatch.  A visual
agent can add findings later; these checks provide the stable baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from skimage.color import lab2rgb, rgb2lab
from skimage.measure import label as measure_label
from sklearn.cluster import KMeans


from geoseg.modules.iteration.cleanup import text_pixel_mask


Action = Literal[
    "accept",
    "increase_layers",
    "switch_engine",
    "remove_text_labels",
    "clean_text_pixels",
    "filter_small_components",
    "split_under_segmented",
]


@dataclass
class LabelAudit:
    label: int
    area: int
    area_fraction: float
    n_components: int
    small_components: int
    dominant_colors: list[dict]
    two_color_separation: float
    minority_fraction: float
    white_fraction: float
    black_fraction: float
    flags: list[str] = field(default_factory=list)


@dataclass
class AuditReport:
    n_labels: int
    labels: list[LabelAudit]
    under_segmented_labels: list[int]
    text_labels: list[int]
    fragmented_labels: list[int]
    layer_count_issue: str | None
    score: float
    accepted: bool
    suggested_action: Action
    text_pixel_fraction: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_labels": self.n_labels,
            "under_segmented_labels": self.under_segmented_labels,
            "text_labels": self.text_labels,
            "fragmented_labels": self.fragmented_labels,
            "layer_count_issue": self.layer_count_issue,
            "score": round(self.score, 4),
            "accepted": self.accepted,
            "suggested_action": self.suggested_action,
            "text_pixel_fraction": round(self.text_pixel_fraction, 4),
            "notes": self.notes,
            "labels": [
                {
                    "label": item.label,
                    "area": item.area,
                    "area_fraction": round(item.area_fraction, 4),
                    "n_components": item.n_components,
                    "small_components": item.small_components,
                    "dominant_colors": item.dominant_colors,
                    "two_color_separation": round(item.two_color_separation, 2),
                    "minority_fraction": round(item.minority_fraction, 3),
                    "white_fraction": round(item.white_fraction, 3),
                    "black_fraction": round(item.black_fraction, 3),
                    "flags": item.flags,
                }
                for item in self.labels
            ],
        }


@dataclass
class AuditConfig:
    under_seg_delta_e: float = 10.0
    under_seg_minority_fraction: float = 0.15
    min_label_area_fraction: float = 0.005
    text_white_fraction: float = 0.40
    text_black_fraction: float = 0.40
    small_component_pixels: int = 20
    fragmented_small_components: int = 4
    expected_min_layers: int | None = None
    expected_max_layers: int | None = None
    max_colors_per_label: int = 2
    text_pixel_fraction_threshold: float = 0.005


def _dominant_colors(pixels_rgb: np.ndarray, max_colors: int = 2) -> tuple[list[dict], float, float]:
    """Cluster one label's pixels and return colours, separation, minority fraction."""
    if len(pixels_rgb) < 20:
        mean = pixels_rgb.mean(axis=0) if len(pixels_rgb) else np.zeros(3)
        return (
            [
                {
                    "rgb": [int(round(v)) for v in mean],
                    "fraction": 1.0,
                }
            ],
            0.0,
            0.0,
        )
    rgb = pixels_rgb.astype(np.uint8)
    bins = (
        (rgb[:, 0] >> 3).astype(np.int32) * 1024
        + (rgb[:, 1] >> 3).astype(np.int32) * 32
        + (rgb[:, 2] >> 3).astype(np.int32)
    )
    counts = np.bincount(bins, minlength=32768)
    present = counts > 0
    ids = np.flatnonzero(present)
    weights = counts[ids].astype(float)
    sums = np.column_stack(
        [
            np.bincount(bins, weights=rgb[:, channel], minlength=32768)[ids]
            for channel in range(3)
        ]
    )
    means = sums / np.maximum(weights[:, None], 1.0)
    colors_lab = rgb2lab(means.reshape(-1, 1, 3) / 255.0).reshape(-1, 3)

    k = int(min(max_colors, len(ids)))
    if k <= 1:
        centers = colors_lab.mean(axis=0, keepdims=True)
        labels = np.zeros(len(colors_lab), dtype=int)
    else:
        model = KMeans(
            n_clusters=k, random_state=42, n_init=5, max_iter=200
        ).fit(colors_lab, sample_weight=weights)
        centers = model.cluster_centers_
        labels = model.labels_

    fractions = np.array(
        [weights[labels == index].sum() for index in range(len(centers))], dtype=float
    )
    fractions /= fractions.sum()
    centers_rgb = (
        lab2rgb(centers.reshape(-1, 1, 3))[..., :3].reshape(-1, 3) * 255.0
    ).clip(0, 255)
    order = np.argsort(fractions)[::-1]
    dominant = [
        {
            "rgb": [int(round(v)) for v in centers_rgb[index]],
            "fraction": round(float(fractions[index]), 3),
        }
        for index in order
    ]
    if len(centers) < 2:
        return dominant, 0.0, 0.0
    separation = float(np.linalg.norm(centers[0] - centers[1]))
    minority = float(fractions.min() / fractions.sum()) if fractions.sum() else 0.0
    return dominant, separation, minority


def audit_labels(
    image: np.ndarray,
    labels: np.ndarray,
    *,
    config: AuditConfig | None = None,
) -> AuditReport:
    """Audit one label map for under-segmentation, text, and fragmentation."""
    config = config or AuditConfig()
    labels = np.asarray(labels, dtype=np.int32)
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    total = labels.size
    text_fraction = float(text_pixel_mask(image).mean())
    audits: list[LabelAudit] = []
    under: list[int] = []
    text: list[int] = []
    fragmented: list[int] = []
    notes: list[str] = []

    for label_id in sorted(int(v) for v in np.unique(labels) if v != 0):
        mask = labels == label_id
        area = int(mask.sum())
        if area == 0:
            continue
        pixels = image[mask][:, :3]
        white_fraction = float(np.mean(np.min(pixels, axis=1) >= 240))
        black_fraction = float(np.mean(np.max(pixels, axis=1) <= 30))
        text_pixels = (np.min(pixels, axis=1) >= 240) | (
            np.max(pixels, axis=1) <= 30
        )
        color_pixels = pixels[~text_pixels]
        if len(color_pixels) < max(20, int(0.1 * len(pixels))):
            color_pixels = pixels
        dominant, separation, minority = _dominant_colors(
            color_pixels, config.max_colors_per_label
        )
        component_map, n_components = measure_label(
            mask, connectivity=1, return_num=True
        )
        areas = np.bincount(component_map.reshape(-1))[1:]
        small_components = int((areas < config.small_component_pixels).sum())
        area_fraction = area / total

        flags: list[str] = []
        if (
            separation >= config.under_seg_delta_e
            and minority >= config.under_seg_minority_fraction
            and area_fraction >= config.min_label_area_fraction
        ):
            flags.append("under_segmented")
            under.append(label_id)
        if white_fraction >= config.text_white_fraction or black_fraction >= config.text_black_fraction:
            flags.append("text_like")
            text.append(label_id)
        if small_components >= config.fragmented_small_components:
            flags.append("fragmented")
            fragmented.append(label_id)

        audits.append(
            LabelAudit(
                label=label_id,
                area=area,
                area_fraction=area_fraction,
                n_components=int(n_components),
                small_components=small_components,
                dominant_colors=dominant,
                two_color_separation=separation,
                minority_fraction=minority,
                white_fraction=white_fraction,
                black_fraction=black_fraction,
                flags=flags,
            )
        )

    n_labels = len(audits)
    layer_count_issue = None
    if config.expected_min_layers is not None and n_labels < config.expected_min_layers:
        layer_count_issue = "too_few_layers"
    elif config.expected_max_layers is not None and n_labels > config.expected_max_layers:
        layer_count_issue = "too_many_layers"

    score = -(
        20.0 * len(under)
        + 12.0 * len(text)
        + 4.0 * len(fragmented)
        + sum(item.small_components for item in audits) * 0.2
    )
    if layer_count_issue == "too_few_layers":
        score -= 10.0
    if layer_count_issue == "too_many_layers":
        score -= 3.0

    if text:
        action: Action = "remove_text_labels"
        notes.append("white/black text or background is labelled as geology; remove it before judging layers")
    elif under:
        action = "increase_layers"
        notes.append("one or more labels contain multiple distinct dominant colours")
    elif layer_count_issue == "too_few_layers":
        action = "increase_layers"
        notes.append("fewer labels than the expected minimum")
    elif fragmented:
        action = "filter_small_components"
        notes.append("labels contain many tiny connected fragments")
    elif layer_count_issue == "too_many_layers":
        action = "switch_engine"
        notes.append("more labels than the expected maximum")
    else:
        action = "accept"

    accepted = action == "accept"
    return AuditReport(
        n_labels=n_labels,
        labels=audits,
        under_segmented_labels=sorted(set(under)),
        text_labels=sorted(set(text)),
        fragmented_labels=sorted(set(fragmented)),
        layer_count_issue=layer_count_issue,
        score=float(score),
        accepted=accepted,
        suggested_action=action,
        text_pixel_fraction=text_fraction,
        notes=notes,
    )


__all__ = [
    "Action",
    "AuditConfig",
    "AuditReport",
    "LabelAudit",
    "audit_labels",
]
