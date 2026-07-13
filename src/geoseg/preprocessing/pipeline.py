"""Artifact absorption pipeline for stacked tomography figures."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from geoseg.modules.cv_detect.colorbar_extractor import extract_colorbar
from geoseg.modules.segment_engines.v4_kmeans import (
    segment_colorbar_guided,
    segment as v4_segment,
)
from geoseg.modules.segment_engines.regional_fusion import generate_overlay_with_legend
from geoseg.core.image_ops import create_overlay
from geoseg.preprocessing.absorption import absorb_artifacts, visualize_mask_on_image
from geoseg.preprocessing.detectors import (
    detect_black_crosses,
    detect_red_lines,
    detect_text,
)
from geoseg.preprocessing.label_merge import merge_artifact_labels
from geoseg.preprocessing.panel_split import split_panels_colored_components


@dataclass
class ArtifactAbsorptionConfig:
    """Configuration for the artifact absorption pipeline."""

    image_path: Path
    output_dir: Path
    panel_bboxes: list[tuple[int, int, int, int]] | None = None
    colorbar_roi: tuple[int, int, int, int] | None = None
    n_layers: int = 5
    run_segmentation: bool = True
    detect_red: bool = True
    detect_crosses: bool = True
    red_params: dict[str, Any] | None = None
    cross_params: dict[str, Any] | None = None
    text_params: dict[str, Any] | None = None
    inpaint_radius: int = 7
    inpaint_dilate_iters: int = 2
    merge_min_area_frac: float = 0.001
    merge_max_brightness: int | None = 80
    per_panel: bool = False
    artifact_labels: dict[int, list[int]] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactAbsorptionConfig":
        """Build a config from a plain dictionary."""
        known = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def process_image(config: ArtifactAbsorptionConfig) -> dict[str, Any]:
    """Run per-panel artifact absorption and optional segmentation comparison."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    img_rgb = np.array(Image.open(config.image_path).convert("RGB"))
    h, w = img_rgb.shape[:2]
    cleaned = img_rgb.copy()
    full_mask = np.zeros((h, w), dtype=bool)

    if config.panel_bboxes:
        panel_bboxes = config.panel_bboxes
    else:
        panel_bboxes = split_panels_colored_components(img_rgb)

    per_panel_stats = []
    panel_masks: list[np.ndarray] = []
    for idx, (x, y, pw, ph) in enumerate(panel_bboxes):
        panel = img_rgb[y : y + ph, x : x + pw]
        text_mask = detect_text(panel, **(config.text_params or {}))

        red_mask = np.zeros_like(text_mask)
        if config.detect_red:
            red_mask = detect_red_lines(panel, **(config.red_params or {})) & ~text_mask

        cross_mask = np.zeros_like(text_mask)
        if config.detect_crosses:
            cross_mask = detect_black_crosses(panel, **(config.cross_params or {}))

        combined = red_mask | cross_mask
        cleaned_panel = absorb_artifacts(
            panel,
            combined,
            inpaint_radius=config.inpaint_radius,
            dilate_iters=config.inpaint_dilate_iters,
        )
        cleaned[y : y + ph, x : x + pw] = cleaned_panel
        full_mask[y : y + ph, x : x + pw] = combined
        panel_masks.append(combined)

        per_panel_stats.append(
            {
                "panel": idx,
                "bbox": [x, y, pw, ph],
                "red_pixels": int(red_mask.sum()),
                "cross_pixels": int(cross_mask.sum()),
            }
        )

    result: dict[str, Any] = {
        "image": str(config.image_path),
        "output_dir": str(config.output_dir),
        "panels": per_panel_stats,
        "red_pixels": sum(p["red_pixels"] for p in per_panel_stats),
        "cross_pixels": sum(p["cross_pixels"] for p in per_panel_stats),
        "total_artifact_pixels": int(full_mask.sum()),
    }

    _save_intermediates(config.output_dir, img_rgb, cleaned, full_mask)

    if config.run_segmentation:
        if config.per_panel:
            result.update(
                _run_per_panel_segmentation(
                    config, img_rgb, cleaned, panel_bboxes, panel_masks
                )
            )
        else:
            result.update(_run_segmentation(config, img_rgb, cleaned, full_mask))

    (config.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    return result


def _save_intermediates(
    output_dir: Path,
    img_rgb: np.ndarray,
    cleaned: np.ndarray,
    full_mask: np.ndarray,
) -> None:
    """Save original, mask overlay, and cleaned images."""
    Image.fromarray(img_rgb).save(output_dir / "01_original.jpg", quality=95)
    Image.fromarray(visualize_mask_on_image(img_rgb, full_mask)).save(
        output_dir / "02_combined_mask_full.jpg", quality=95
    )
    Image.fromarray(cleaned).save(output_dir / "03_cleaned.jpg", quality=95)


def _run_segmentation(
    config: ArtifactAbsorptionConfig,
    img_rgb: np.ndarray,
    cleaned: np.ndarray,
    artifact_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run baseline, cleaned, and colorbar-guided segmentation comparisons."""
    output_dir = config.output_dir
    n_layers = config.n_layers

    base_seg = v4_segment(img_rgb, n_layers=n_layers)
    clean_seg = v4_segment(cleaned, n_layers=n_layers)
    diff_mask = base_seg["labels"] != clean_seg["labels"]
    diff_vis = np.full_like(img_rgb, 128)
    diff_vis[diff_mask] = [255, 0, 0]

    Image.fromarray(base_seg["overlay"]).save(
        output_dir / "04_seg_original_overlay.jpg", quality=95
    )
    Image.fromarray(clean_seg["overlay"]).save(
        output_dir / "05_seg_cleaned_overlay.jpg", quality=95
    )
    Image.fromarray(diff_vis).save(output_dir / "06_seg_difference.jpg", quality=95)

    seg_result: dict[str, Any] = {
        "diff_fraction": float(diff_mask.sum() / diff_mask.size),
    }

    colorbar_rgb = extract_colorbar(img_rgb, colorbar_roi=config.colorbar_roi)
    if colorbar_rgb is not None and colorbar_rgb.size > 0:
        cb_clean_seg = segment_colorbar_guided(
            cleaned, colorbar_rgb, n_layers=n_layers
        )

        merged_labels = merge_artifact_labels(
            cb_clean_seg["labels"],
            image_rgb=cleaned,
            min_area_frac=config.merge_min_area_frac,
            max_mean_brightness=config.merge_max_brightness,
            artifact_mask=artifact_mask,
        )

        # Rebuild overlay with merged labels.
        palette = np.array(cb_clean_seg["seeds"], dtype=np.uint8)
        merged_overlay = create_overlay(
            cleaned, merged_labels, palette, skip_background=False
        )

        Image.fromarray(cb_clean_seg["overlay"]).save(
            output_dir / "07_seg_colorbar_guided_cleaned.jpg", quality=95
        )
        np.savez(
            output_dir / "07_seg_colorbar_guided_cleaned.npz",
            labels=cb_clean_seg["labels"],
        )
        Image.fromarray(merged_overlay).save(
            output_dir / "08_seg_colorbar_guided_merged.jpg", quality=95
        )
        np.savez(
            output_dir / "08_seg_colorbar_guided_merged.npz",
            labels=merged_labels,
        )
        seg_result["colorbar_guided_used"] = True
    else:
        seg_result["colorbar_guided_used"] = False

    return seg_result


def _run_per_panel_segmentation(
    config: ArtifactAbsorptionConfig,
    img_rgb: np.ndarray,
    cleaned: np.ndarray,
    panel_bboxes: list[tuple[int, int, int, int]],
    panel_masks: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    """Run v4 segmentation on each cleaned panel and merge explicit artifacts.

    The full image is assembled from per-panel results so the overlay background
    is the cleaned image, not the original artifact-laden source.
    """
    output_dir = config.output_dir
    n_layers = config.n_layers
    h, w = img_rgb.shape[:2]

    full_labels = np.full((h, w), -1, dtype=np.int32)
    full_overlay = cleaned.copy()
    per_panel_results = []
    next_label_offset = 0

    panel_dir = output_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    for panel_id, (x, y, pw, ph) in enumerate(panel_bboxes):
        panel_clean = cleaned[y : y + ph, x : x + pw]
        panel_mask = panel_masks[panel_id] if panel_masks else None
        seg = v4_segment(panel_clean, n_layers=n_layers)
        labels = seg["labels"].copy()

        artifact_labels_raw = config.artifact_labels or {}
        artifact_labels = artifact_labels_raw.get(
            panel_id, artifact_labels_raw.get(str(panel_id), [])
        )
        artifact_labels = [int(v) for v in artifact_labels]
        if artifact_labels or panel_mask is not None:
            labels = merge_artifact_labels(
                labels,
                image_rgb=panel_clean,
                max_mean_brightness=config.merge_max_brightness,
                artifact_labels=artifact_labels or None,
                artifact_mask=panel_mask,
            )

        # Offset labels per panel so the assembled full map is unambiguous.
        shifted = labels.copy()
        shifted[labels >= 0] += next_label_offset
        full_labels[y : y + ph, x : x + pw] = shifted
        next_label_offset = int(np.max(full_labels)) + 1

        audit_overlay = generate_overlay_with_legend(panel_clean, labels)
        palette = np.array(seg["seeds"], dtype=np.uint8)
        panel_overlay = create_overlay(
            panel_clean, labels, palette, skip_background=True, overlay_colors=palette
        )
        full_overlay[y : y + ph, x : x + pw] = panel_overlay

        panel_out = panel_dir / f"panel_{panel_id}"
        panel_out.mkdir(parents=True, exist_ok=True)
        np.savez(panel_out / "labels.npz", labels=labels)
        Image.fromarray(audit_overlay).save(
            panel_out / "overlay_legend.jpg", quality=95
        )

        per_panel_results.append(
            {
                "panel_id": panel_id,
                "bbox": [x, y, pw, ph],
                "labels": sorted({int(l) for l in np.unique(labels) if l >= 0}),
                "artifact_labels": artifact_labels,
            }
        )

    Image.fromarray(full_overlay).save(
        output_dir / "09_per_panel_overlay.jpg", quality=95
    )
    np.savez(output_dir / "09_per_panel_labels.npz", labels=full_labels)

    return {
        "per_panel": per_panel_results,
        "per_panel_overlay": str(output_dir / "09_per_panel_overlay.jpg"),
        "per_panel_labels": str(output_dir / "09_per_panel_labels.npz"),
    }
