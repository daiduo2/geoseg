"""Local regional refinement engine.

Re-segments a binary refine_mask region with a secondary engine while freezing
everything else. Useful when color-residual diagnostics propose a sub-region
that needs finer subdivision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from geoseg.modules.segment_engines.regional_fusion import (
    FusionConfig,
    fuse_with_freeze,
    generate_overlay_with_legend,
)
from geoseg.modules.segment_engines.router import _run_engine
from geoseg.modules.segment_engines.v4_kmeans import _reorder_labels_by_median_y


@dataclass
class RefinementConfig:
    """Configuration for residual-mask regional refinement."""

    secondary_engine: str = "edge_guided"
    seam_smooth_width: int = 3
    crop_margin: int = 10
    fusion_config: FusionConfig = field(default_factory=FusionConfig)


def _validate_shapes(
    base_labels: np.ndarray,
    panel_rgb: np.ndarray,
    refine_mask: np.ndarray,
) -> None:
    """Validate that all inputs share the same spatial shape."""
    if base_labels.shape != panel_rgb.shape[:2]:
        raise ValueError(
            f"Shape mismatch: base_labels {base_labels.shape} vs panel_rgb {panel_rgb.shape[:2]}"
        )
    if base_labels.shape != refine_mask.shape:
        raise ValueError(
            f"Shape mismatch: base_labels {base_labels.shape} vs refine_mask {refine_mask.shape}"
        )


def _crop_with_margin(
    mask: np.ndarray,
    image_rgb: np.ndarray,
    margin: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Crop image and mask to the mask's bounding box plus margin."""
    h, w = mask.shape
    ys, xs = np.where(mask)
    if ys.size == 0:
        raise ValueError("refine_mask is empty")

    y0 = max(0, ys.min() - margin)
    x0 = max(0, xs.min() - margin)
    y1 = min(h, ys.max() + 1 + margin)
    x1 = min(w, xs.max() + 1 + margin)

    crop_mask = mask[y0:y1, x0:x1].copy()
    crop_rgb = image_rgb[y0:y1, x0:x1].copy()
    return crop_mask, crop_rgb, (x0, y0, x1, y1)


def refine_by_residual_mask(
    base_labels: np.ndarray,
    panel_rgb: np.ndarray,
    refine_mask: np.ndarray,
    n_layers: int,
    *,
    reps: list[dict] | None = None,
    colorbar_rgb: np.ndarray | None = None,
    config: RefinementConfig | None = None,
) -> dict[str, Any]:
    """Re-segment refine_mask region locally and fuse back with frozen base.

    Args:
        base_labels: current global label map.
        panel_rgb: original RGB panel.
        refine_mask: boolean mask of pixels to re-segment.
        n_layers: layer count for the secondary engine inside the crop.
        reps: optional representative seeds for rep-dependent engines.
        colorbar_rgb: optional colorbar strip.
        config: RefinementConfig.

    Returns:
        dict with keys:
        - labels: refined global label map.
        - overlay: RGB overlay with legend.
        - meta: refinement metadata.
    """
    config = config or RefinementConfig()
    _validate_shapes(base_labels, panel_rgb, refine_mask)

    if not refine_mask.any():
        overlay = generate_overlay_with_legend(panel_rgb, base_labels)
        return {
            "labels": base_labels.copy(),
            "overlay": overlay,
            "meta": {
                "refined": False,
                "reason": "empty refine_mask",
                "secondary_engine": config.secondary_engine,
            },
        }

    crop_mask, crop_rgb, bbox = _crop_with_margin(
        refine_mask, panel_rgb, config.crop_margin
    )
    x0, y0, x1, y1 = bbox

    result_b = _run_engine(
        config.secondary_engine,
        crop_rgb,
        reps,
        colorbar_rgb,
        n_layers,
    )
    secondary_labels = result_b["labels"]
    secondary_labels = _reorder_labels_by_median_y(secondary_labels)

    # Offset secondary labels to avoid collision with base labels.
    max_base_label = int(base_labels.max())
    secondary_labels = np.where(
        secondary_labels > 0,
        secondary_labels + max_base_label,
        0,
    )

    patch_labels = base_labels.copy()
    patch_labels[y0:y1, x0:x1] = np.where(
        crop_mask,
        secondary_labels,
        patch_labels[y0:y1, x0:x1],
    )

    freeze_mask = ~refine_mask
    fused = fuse_with_freeze(
        base_labels,
        patch_labels,
        freeze_mask,
        seam_width=config.seam_smooth_width,
    )

    overlay = generate_overlay_with_legend(panel_rgb, fused)

    return {
        "labels": fused,
        "overlay": overlay,
        "meta": {
            "refined": True,
            "secondary_engine": config.secondary_engine,
            "n_layers": n_layers,
            "crop_bbox": [x0, y0, x1, y1],
            "refined_pixels": int(refine_mask.sum()),
            "frozen_pixels": int(freeze_mask.sum()),
            "label_offset": max_base_label,
        },
    }


def refine_by_candidate_regions(
    base_labels: np.ndarray,
    panel_rgb: np.ndarray,
    candidates: list[dict[str, Any]],
    n_layers: int,
    *,
    reps: list[dict] | None = None,
    colorbar_rgb: np.ndarray | None = None,
    config: RefinementConfig | None = None,
) -> dict[str, Any]:
    """Refine each candidate region independently and fuse sequentially.

    Processing candidates one-by-one avoids a single large crop that would
    re-segment unrelated areas. Each candidate is refined against the current
    label state, so earlier refinements are preserved while later ones run.

    Args:
        base_labels: current global label map.
        panel_rgb: original RGB panel.
        candidates: list of candidate dicts with a "bbox" key [x0, y0, x1, y1].
        n_layers: layer count for each secondary engine run.
        reps: optional representative seeds.
        colorbar_rgb: optional colorbar strip.
        config: RefinementConfig.

    Returns:
        dict with keys:
        - labels: final refined global label map.
        - overlay: RGB overlay with legend.
        - meta: refinement metadata including refined_region indices.
    """
    config = config or RefinementConfig()
    current_labels = base_labels.copy()
    refined_indices: list[int] = []
    region_meta: list[dict[str, Any]] = []

    for idx, cand in enumerate(candidates):
        bbox = cand.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = bbox
        h, w = base_labels.shape
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(w, x1)
        y1 = min(h, y1)
        if x1 <= x0 or y1 <= y0:
            continue

        refine_mask = np.zeros(base_labels.shape, dtype=bool)
        refine_mask[y0:y1, x0:x1] = True

        result = refine_by_residual_mask(
            current_labels,
            panel_rgb,
            refine_mask,
            n_layers,
            reps=reps,
            colorbar_rgb=colorbar_rgb,
            config=config,
        )

        if result["meta"].get("refined"):
            current_labels = result["labels"]
            refined_indices.append(idx)
            region_meta.append(
                {
                    "index": idx,
                    "bbox": [x0, y0, x1, y1],
                    "meta": result["meta"],
                }
            )

    overlay = generate_overlay_with_legend(panel_rgb, current_labels)
    return {
        "labels": current_labels,
        "overlay": overlay,
        "meta": {
            "refined": bool(refined_indices),
            "secondary_engine": config.secondary_engine,
            "n_layers": n_layers,
            "n_candidates": len(candidates),
            "refined_regions": refined_indices,
            "region_meta": region_meta,
        },
    }
