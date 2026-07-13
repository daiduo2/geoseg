"""Regional fusion: multi-engine segmentation with label-level freeze.

Experimental feature. Triggered only when agent's whole-image audit indicates
poor quality. Agent identifies good regions by color/label ID in the overlay
legend; bad regions are re-segmented with a different engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from geoseg.modules.post_process.split import split_label_by_color_components
from geoseg.modules.segment_engines.internal.shared import (
    _create_overlay,
    _distinct_colors,
)
from geoseg.modules.segment_engines.v4_kmeans import _reorder_labels_by_median_y


@dataclass
class RegionalAudit:
    """Agent regional audit result."""

    frozen_labels: list[int] = field(default_factory=list)
    retry_labels: list[int] = field(default_factory=list)
    notes: str = ""
    iteration: int = 0
    repair_strategy: str = "regional_fusion"
    secondary_engine: str = ""
    local_fixes: list[dict] = field(default_factory=list)


@dataclass
class FusionConfig:
    """Regional fusion configuration."""

    primary_engine: str = "v4_kmeans"
    secondary_engines: list[str] = field(
        default_factory=lambda: ["edge_guided", "kmeans_full"]
    )
    max_iterations: int = 3
    seam_smooth_width: int = 3
    enable_legend: bool = True


def fuse_with_freeze(
    base_labels: np.ndarray,
    patch_labels: np.ndarray,
    freeze_mask: np.ndarray,
    seam_width: int = 3,
) -> np.ndarray:
    """Fuse two label arrays, keeping base where frozen and patch elsewhere.

    Seam smoothing resolves narrow label discontinuities at the freeze boundary
    by nearest-neighbor fill within a transition band.
    """
    if base_labels.shape != patch_labels.shape:
        raise ValueError(
            f"Shape mismatch: base {base_labels.shape} vs patch {patch_labels.shape}"
        )
    if base_labels.shape != freeze_mask.shape:
        raise ValueError(
            f"Shape mismatch: labels {base_labels.shape} vs mask {freeze_mask.shape}"
        )

    result = patch_labels.copy()
    result[freeze_mask] = base_labels[freeze_mask]

    if seam_width <= 0:
        return result

    struct = np.ones((3, 3), dtype=bool)
    dilated = ndimage.binary_dilation(freeze_mask, structure=struct)
    boundary = dilated & ~freeze_mask

    if not boundary.any():
        return result

    if seam_width > 1:
        wide_struct = np.ones((seam_width * 2 - 1, seam_width * 2 - 1), dtype=bool)
        transition = ndimage.binary_dilation(boundary, structure=wide_struct)
    else:
        transition = boundary

    if not transition.any():
        return result

    gap_mask = (result == 0) & transition
    if not gap_mask.any():
        return result

    nz_mask = result != 0
    _, indices = ndimage.distance_transform_edt(~nz_mask, return_indices=True)

    rr, cc = np.where(gap_mask)
    result[rr, cc] = result[indices[0][rr, cc], indices[1][rr, cc]]
    return result


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try common system fonts, fallback to default."""
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_legend(
    overlay_rgb: np.ndarray,
    labels: np.ndarray,
    label_colors: dict[int, np.ndarray] | None = None,
    box_size: int = 12,
    font_size: int = 10,
) -> np.ndarray:
    """Draw label legend on overlay bottom-right corner.

    Returns RGB array with semi-transparent legend overlay.
    """
    h, w = overlay_rgb.shape[:2]
    unique = sorted(set(labels.flatten()) - {0})
    n = len(unique)
    if n == 0:
        return overlay_rgb

    rgba = np.dstack([overlay_rgb, np.full((h, w), 255, dtype=np.uint8)])
    img = Image.fromarray(rgba, mode="RGBA")

    item_h = box_size + 4
    pad = 6
    leg_h = n * item_h + pad * 2
    leg_w = box_size + 28 + pad * 2

    lx = max(0, w - leg_w - 8)
    ly = max(0, h - leg_h - 8)

    bg = Image.new("RGBA", (leg_w, leg_h), (0, 0, 0, 180))
    img.paste(bg, (lx, ly), bg)

    draw = ImageDraw.Draw(img)
    base_colors = _distinct_colors(max(unique) + 1)
    font = _load_font(font_size)

    for i, lbl in enumerate(unique):
        y = ly + pad + i * item_h
        if label_colors is not None and lbl in label_colors:
            color = tuple(int(c) for c in label_colors[lbl]) + (255,)
        else:
            color = tuple(int(c) for c in base_colors[lbl]) + (255,)
        draw.rectangle(
            [lx + pad, y, lx + pad + box_size, y + box_size],
            fill=color,
        )
        draw.text(
            (lx + pad + box_size + 4, y - 1),
            str(lbl),
            fill=(255, 255, 255, 255),
            font=font,
        )

    return np.array(img.convert("RGB"))


def generate_overlay_with_legend(
    panel_rgb: np.ndarray,
    labels: np.ndarray,
    seeds_rgb: np.ndarray | None = None,
    alpha: float = 0.65,
) -> np.ndarray:
    """Create overlay with bottom-right label legend for agent audit."""
    overlay = _create_overlay(panel_rgb, labels, seeds_rgb, alpha=alpha)
    return _draw_legend(overlay, labels)


def _run_engine_by_name(
    engine: str,
    panel_rgb: np.ndarray,
    n_layers: int,
    reps: list[dict] | None = None,
    colorbar_rgb: np.ndarray | None = None,
) -> dict:
    """Run a segmentation engine by name."""
    from geoseg.modules.segment_engines.runner import run_engine

    return run_engine(engine, panel_rgb, reps, colorbar_rgb, n_layers)


def regional_segment(
    panel_rgb: np.ndarray,
    n_layers: int,
    *,
    reps: list[dict] | None = None,
    colorbar_rgb: np.ndarray | None = None,
    primary_result: dict | None = None,
    audit: RegionalAudit | None = None,
    config: FusionConfig | None = None,
) -> dict:
    """Region-level multi-engine segmentation with freeze fusion.

    Experimental. If primary_result and audit are provided and retry_labels
    is non-empty, performs freeze+patch fusion with a secondary engine.
    Otherwise returns primary engine result only.
    """
    config = config or FusionConfig()

    no_audit = audit is None or not audit.retry_labels
    if primary_result is None or no_audit:
        result = _run_engine_by_name(
            config.primary_engine, panel_rgb, n_layers, reps, colorbar_rgb
        )
        overlay = (
            generate_overlay_with_legend(panel_rgb, result["labels"])
            if config.enable_legend
            else result.get("overlay")
        )
        return {
            "labels": result["labels"],
            "overlay": overlay,
            "meta": {
                "engine": config.primary_engine,
                "path": "primary_only",
                "fusion_applied": False,
            },
        }

    labels_a = primary_result["labels"]
    # Align both engines to consistent top-to-bottom label ordering
    labels_a = _reorder_labels_by_median_y(labels_a)

    # Apply local fixes (e.g. label merges / colour-component splits) before any regional fusion.
    if audit is not None and audit.local_fixes:
        from geoseg.modules.post_process.merge import merge_labels_by_ids

        for fix in audit.local_fixes:
            action = fix.get("action")
            if action == "merge_labels":
                labels_a = merge_labels_by_ids(
                    labels_a,
                    fix["label_ids"],
                    target_id=fix.get("target_id", fix["label_ids"][0]),
                )
            elif action == "split_label_by_color_components":
                labels_a = split_label_by_color_components(
                    labels_a,
                    panel_rgb,
                    target_label=fix["label_id"],
                    color_space=fix.get("color_space", "LAB"),
                    k=fix.get("k", 3),
                    min_component_area=fix.get("min_component_area", 300),
                )

    freeze_mask = np.zeros(labels_a.shape, dtype=bool)
    for lbl in audit.frozen_labels:
        freeze_mask |= labels_a == lbl

    secondary = (
        audit.secondary_engine
        if audit and audit.secondary_engine
        else (config.secondary_engines[0] if config.secondary_engines else "v4_kmeans")
    )
    result_b = _run_engine_by_name(secondary, panel_rgb, n_layers, reps, colorbar_rgb)
    labels_b = _reorder_labels_by_median_y(result_b["labels"])

    fused = fuse_with_freeze(
        labels_a,
        labels_b,
        freeze_mask,
        seam_width=config.seam_smooth_width,
    )
    # No global reorder after fusion — freeze semantics are preserved

    overlay = generate_overlay_with_legend(panel_rgb, fused) if config.enable_legend else None

    return {
        "labels": fused,
        "overlay": overlay,
        "meta": {
            "engine": f"{config.primary_engine}+{secondary}",
            "path": "regional_fusion",
            "fusion_applied": True,
            "primary_engine": config.primary_engine,
            "secondary_engine": secondary,
            "repair_strategy": audit.repair_strategy if audit else "regional_fusion",
            "frozen_labels": audit.frozen_labels,
            "retry_labels": audit.retry_labels,
            "local_fixes": audit.local_fixes,
            "fusion_notes": audit.notes,
            "iteration": audit.iteration,
        },
    }
