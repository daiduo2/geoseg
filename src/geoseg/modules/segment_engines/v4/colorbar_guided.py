"""Colorbar-guided v4 K-Means segmentation path."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.cluster.vq import kmeans2
from skimage.color import lab2rgb, rgb2lab

from geoseg.modules.segment_engines.internal.color import (
    _label_by_nearest,
    saturation_ratio,
)
from geoseg.modules.segment_engines.internal.overlay import _create_overlay
from geoseg.modules.segment_engines.v4.pastel import segment_pastel_faded
from geoseg.modules.segment_engines.v4.palette import _sample_colorbar_seeds
from geoseg.modules.segment_engines.v4.postprocess import (
    _enhance_close_boundaries,
    _fill_holes,
    _remove_small_components,
)


def segment_colorbar_guided(
    panel_rgb: np.ndarray,
    colorbar_rgb: np.ndarray,
    n_layers: int = 5,
    color_dist_threshold: float = 55.0,
    explicit_seeds: list[dict] | None = None,
    n_color_zones: int = 0,
) -> dict:
    """Colorbar-guided segmentation via nearest seed in LAB space.

    Each pixel is assigned to the closest colorbar seed.  This preserves the
    colorbar's color ordering and avoids k-means collapsing rare colors (e.g.
    red / blue) into the dominant green/cyan background.

    Returns {"labels", "seeds", "overlay", "meta"}.
    """
    h, w, _ = panel_rgb.shape
    panel_lab = rgb2lab(panel_rgb)

    effective_n_layers = n_layers
    effective_threshold = color_dist_threshold
    if n_color_zones >= 3:
        effective_n_layers = max(n_layers, n_color_zones + 1)
        effective_threshold = 35.0

    if explicit_seeds is not None and len(explicit_seeds) > 0:
        seeds_rgb = np.array([s["rgb"] for s in explicit_seeds], dtype=np.uint8)
        k = len(explicit_seeds)
    else:
        seeds_rgb, _ = _sample_colorbar_seeds(colorbar_rgb, effective_n_layers)
        k = effective_n_layers

    # Drop seeds that are nearly white/black (margin / text background), since
    # they do not represent colorbar colors.
    brightness = seeds_rgb.mean(axis=1)
    keep = (brightness > 45) & (brightness < 245)
    if keep.sum() >= 2:
        seeds_rgb = seeds_rgb[keep]
        k = len(seeds_rgb)
    elif keep.sum() == 1 and len(seeds_rgb) >= 2:
        # Keep at least one valid seed would collapse everything to a single
        # label. Fall back to the unfiltered palette so the caller still gets
        # a multi-label segmentation; visual review can remove margin labels.
        k = len(seeds_rgb)
    elif len(seeds_rgb) < 2:
        # Not enough seeds to produce a meaningful nearest-seed segmentation.
        # Fall back to unsupervised k-means.
        return segment_pastel_faded(
            panel_rgb, colorbar_rgb=None, n_layers=n_layers, n_color_zones=n_color_zones
        )

    seeds_lab = rgb2lab(seeds_rgb[np.newaxis, ...])[0]
    labels = _label_by_nearest(panel_lab, seeds_lab)
    labels = ndimage.median_filter(labels, size=5)
    labels = _fill_holes(labels)
    labels = _remove_small_components(labels, min_area_frac=0.001)

    # If the colorbar seeds collapse to a single effective label (e.g. the
    # supplied colorbar strip is wrong or the panel is nearly uniform), fall
    # back to unsupervised k-means rather than pretending there are multiple
    # layers.
    if len(set(labels.flatten()) - {0}) < 1 and len(seeds_rgb) >= 2:
        return segment_pastel_faded(
            panel_rgb, colorbar_rgb=None, n_layers=n_layers, n_color_zones=n_color_zones
        )

    final_palette = np.zeros((k, 3), dtype=np.uint8)
    for lbl in range(k):
        mask = labels == lbl
        if mask.any():
            final_palette[lbl] = panel_rgb[mask].mean(axis=0).astype(np.uint8)
        else:
            final_palette[lbl] = seeds_rgb[lbl]

    overlay = _create_overlay(panel_rgb, labels, final_palette, skip_background=False)

    return {
        "labels": labels,
        "seeds": final_palette.tolist(),
        "overlay": overlay,
        "meta": {
            "engine": "v4_kmeans",
            "path": "colorbar_guided",
            "seed_origin": "explicit_seeds" if explicit_seeds is not None else "colorbar",
            "n_layers": k,
            "saturation_ratio": round(saturation_ratio(panel_rgb), 4),
        },
    }

__all__ = ["segment_colorbar_guided"]
