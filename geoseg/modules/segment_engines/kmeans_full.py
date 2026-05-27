"""K-means full panel clustering (e007).

Uses scipy.cluster.vq.kmeans2 in LAB space with VLM seeds + auto-k seeds as
initial centroids. K-means optimises globally, producing more coherent regions.
Same shape filter post-processing as other engines.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.vq import kmeans2
from skimage.color import rgb2lab

from geoseg.modules.segment_engines._shared import (
    _create_overlay,
    _shape_filter,
    _estimate_background_color,
    _cv_seeds,
    _refine_vlm_seeds,
    _auto_k,
    saturation_ratio,
)


def segment(
    panel_rgb: np.ndarray,
    reps: list[dict] | None = None,
    n_layers: int = 5,
    max_auto_k: int = 2,
) -> dict:
    """K-means segmentation for vivid jet-colormap panels.

    Args:
        panel_rgb: RGB uint8 array (H, W, 3).
        reps: Optional VLM representative points. If None, uses CV seeds only.
        n_layers: Target layer count when reps is None; kept for interface consistency.
        max_auto_k: Maximum extra seeds to auto-detect.

    Returns:
        dict with keys: labels, seeds, overlay, meta.
    """
    h, w, _ = panel_rgb.shape
    panel_lab = rgb2lab(panel_rgb)
    bg_rgb = _estimate_background_color(panel_rgb)
    min_auto_count = max(50, h * w // 2000)

    if reps:
        cv_seeds_rgb, cv_tags = _cv_seeds(panel_rgb, k=len(reps))
        used_cv_indices: set[int] = set()

        refined_seeds, refined_reps = _refine_vlm_seeds(
            panel_rgb, reps, bg_rgb, cv_seeds_rgb, cv_tags, used_cv_indices
        )
        color_names = [r.get("color_name", f"layer_{i + 1}") for i, r in enumerate(reps)]
    else:
        cv_seeds_rgb, cv_tags = _cv_seeds(panel_rgb, k=n_layers)
        used_cv_indices: set[int] = set()
        refined_seeds = []
        refined_reps = []
        color_names = [f"layer_{i + 1}" for i in range(n_layers)]

    refined_seeds, refined_reps = _auto_k(
        panel_rgb, panel_lab, bg_rgb,
        refined_seeds, refined_reps,
        cv_seeds_rgb, cv_tags, used_cv_indices,
        max_auto_k, min_auto_count,
    )
    if len(refined_reps) > len(color_names):
        color_names = color_names + [r["name"] for r in refined_reps[len(color_names):]]

    if not refined_seeds:
        # Fallback: use CV seeds directly
        refined_seeds = [cv_seeds_rgb[i] for i in range(min(n_layers, len(cv_seeds_rgb)))]

    refined_seeds_arr = np.array(refined_seeds, dtype=np.uint8)
    seeds_lab = rgb2lab(refined_seeds_arr[np.newaxis, ...])[0]

    flat_lab = panel_lab.reshape(-1, 3)
    centroids, labels_flat = kmeans2(flat_lab, seeds_lab, minit="matrix")
    labels = labels_flat.reshape(h, w).astype(np.int32)

    labels = _shape_filter(labels)

    overlay = _create_overlay(panel_rgb, labels, refined_seeds_arr)

    return {
        "labels": labels,
        "seeds": refined_seeds_arr.tolist(),
        "overlay": overlay,
        "meta": {
            "engine": "kmeans_full",
            "reps_refined": refined_reps,
            "cv_seeds": cv_seeds_rgb.tolist() if len(cv_seeds_rgb) else [],
            "bg_rgb": bg_rgb.tolist(),
            "auto_k_added": len(refined_reps) - (len(reps) if reps else 0),
            "centroids_lab": centroids.tolist(),
            "saturation_ratio": round(saturation_ratio(panel_rgb), 4),
        },
    }
