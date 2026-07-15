"""Pastel/faded fallback v4 K-Means segmentation path."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.cluster.vq import kmeans2
from skimage.color import lab2rgb, rgb2lab

from geoseg.modules.segment_engines.internal.color import saturation_ratio
from geoseg.modules.segment_engines.internal.overlay import _create_overlay
from geoseg.modules.segment_engines.internal.regions import _reorder_labels_by_median_y, _shape_filter
from geoseg.modules.segment_engines.internal.seeds import _cv_seeds
from geoseg.modules.segment_engines.v4.palette import _name_palette, _sample_colorbar_seeds


def segment_pastel_faded(
    panel_rgb: np.ndarray,
    colorbar_rgb: np.ndarray | None = None,
    n_layers: int = 5,
    n_color_zones: int = 0,
) -> dict:
    """K-means in LAB space, optionally seeded from the panel's colorbar.

    Returns {"labels", "seeds", "overlay", "meta"}.
    """
    h, w, _ = panel_rgb.shape
    panel_lab = rgb2lab(panel_rgb).reshape(-1, 3)

    if colorbar_rgb is not None and colorbar_rgb.size > 0:
        seeds_rgb, names = _sample_colorbar_seeds(colorbar_rgb, n_layers)
        seeds_lab = rgb2lab(seeds_rgb[np.newaxis, ...])[0]
        centroids, labels_flat = kmeans2(panel_lab, seeds_lab, minit="matrix")
        seed_origin = "colorbar"
    else:
        cv_seeds_rgb, cv_tags = _cv_seeds(panel_rgb, k=n_layers)
        if len(cv_seeds_rgb) >= n_layers:
            seeds_rgb = cv_seeds_rgb[:n_layers]
            seeds_lab = rgb2lab(seeds_rgb[np.newaxis, ...])[0]
            centroids, labels_flat = kmeans2(panel_lab, seeds_lab, minit="matrix")
            seed_origin = "cv_multi_source"
            names = [f"cv_{i}" for i in range(n_layers)]
        else:
            centroids, labels_flat = kmeans2(panel_lab, n_layers, minit="++", seed=42)
            approx = (lab2rgb(centroids[np.newaxis, ...])[0] * 255).clip(0, 255).astype(np.uint8)
            seeds_rgb = approx
            names = _name_palette(seeds_rgb, n_layers)
            seed_origin = "kmeans++_random"

    labels = labels_flat.reshape(h, w).astype(np.int32)
    labels = _shape_filter(labels)
    labels = ndimage.median_filter(labels, size=5)

    overlay = _create_overlay(panel_rgb, labels, seeds_rgb)

    return {
        "labels": labels,
        "seeds": seeds_rgb.tolist(),
        "overlay": overlay,
        "meta": {
            "engine": "v4_kmeans",
            "path": "pastel_faded",
            "seed_origin": seed_origin,
            "n_layers": n_layers,
            "saturation_ratio": round(saturation_ratio(panel_rgb), 4),
        },
    }

__all__ = ["segment_pastel_faded"]
