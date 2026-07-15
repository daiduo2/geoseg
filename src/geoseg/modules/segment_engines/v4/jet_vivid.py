"""Jet-vivid v4 K-Means segmentation path."""

from __future__ import annotations

import numpy as np
from skimage.color import rgb2lab

from geoseg.modules.segment_engines.internal.color import (
    _estimate_background_color,
    saturation_ratio,
)
from geoseg.modules.segment_engines.internal.overlay import _create_overlay
from geoseg.modules.segment_engines.internal.regions import _shape_filter
from geoseg.modules.segment_engines.internal.seeds import (
    _auto_k,
    _cv_seeds,
    _refine_vlm_seeds,
)
from geoseg.modules.segment_engines.v4.postprocess import _nearest_median


def segment_jet_vivid(
    panel_rgb: np.ndarray,
    reps: list[dict] | None = None,
    n_layers: int = 5,
    max_auto_k: int = 0,
) -> dict:
    """Nearest-median segmentation for vivid jet-colormap panels.

    Returns {"labels", "seeds", "overlay", "meta"}.
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
        refined_seeds = [cv_seeds_rgb[i] for i in range(min(n_layers, len(cv_seeds_rgb)))]

    refined_seeds_arr = np.array(refined_seeds, dtype=np.uint8)
    seeds_lab = rgb2lab(refined_seeds_arr[np.newaxis, ...])[0]

    labels = _nearest_median(panel_lab, seeds_lab, median_size=5)
    labels = _shape_filter(labels)

    overlay = _create_overlay(panel_rgb, labels, refined_seeds_arr)

    return {
        "labels": labels,
        "seeds": refined_seeds_arr.tolist(),
        "overlay": overlay,
        "meta": {
            "engine": "v4_kmeans",
            "path": "jet_vivid",
            "reps_refined": refined_reps,
            "cv_seeds": cv_seeds_rgb.tolist() if len(cv_seeds_rgb) else [],
            "bg_rgb": bg_rgb.tolist(),
            "auto_k_added": len(refined_reps) - (len(reps) if reps else 0),
            "saturation_ratio": round(saturation_ratio(panel_rgb), 4),
        },
    }

__all__ = ["segment_jet_vivid"]
