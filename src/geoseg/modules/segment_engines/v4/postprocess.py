"""Post-processing helpers for v4 K-Means segmentation."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from geoseg.modules.segment_engines.internal.color import _label_by_nearest


def _nearest_median(
    panel_lab: np.ndarray,
    seeds_lab: np.ndarray,
    median_size: int = 5,
) -> np.ndarray:
    """Per-pixel nearest seed in LAB, followed by median-filter smoothing."""
    labels = _label_by_nearest(panel_lab, seeds_lab)
    if median_size > 1:
        labels = ndimage.median_filter(labels, size=median_size)
    return labels


def _fill_holes(labels: np.ndarray) -> np.ndarray:
    """Fill holes inside each labeled region."""
    out = labels.copy()
    for lbl in range(int(labels.max()) + 1):
        mask = labels == lbl
        if not mask.any():
            continue
        filled = ndimage.binary_fill_holes(mask)
        out[filled & (labels != lbl)] = lbl
    return out


def _remove_small_components(labels: np.ndarray, min_area_frac: float = 0.001) -> np.ndarray:
    """Merge tiny connected components (< min_area_frac of panel area) into neighbors."""
    h, w = labels.shape
    out = labels.copy()
    min_area = max(50, int(h * w * min_area_frac))

    for lbl in range(int(labels.max()) + 1):
        mask = out == lbl
        if not mask.any():
            continue
        labeled, num = ndimage.label(mask)
        if num <= 1:
            continue
        sizes = ndimage.sum(mask, labeled, range(1, num + 1))
        for comp_id in range(1, num + 1):
            if sizes[comp_id - 1] < min_area:
                comp_mask = labeled == comp_id
                dilated = ndimage.binary_dilation(comp_mask, structure=np.ones((3, 3), dtype=bool))
                neighbors = out[dilated & ~comp_mask & (out >= 0)]
                if len(neighbors) > 0:
                    new_lbl = int(np.bincount(neighbors).argmax())
                    out[comp_mask] = new_lbl
    return out


def _enhance_close_boundaries(
    panel_rgb: np.ndarray,
    labels: np.ndarray,
    palette_rgb: np.ndarray,
    color_dist_threshold: float = 55.0,
) -> np.ndarray:
    """Re-classify boundary pixels between adjacent layers with similar seed colors."""
    out = labels.copy()
    k = len(palette_rgb)
    if k < 2:
        return out

    for i in range(k - 1):
        d = float(np.linalg.norm(palette_rgb[i].astype(np.float32) - palette_rgb[i + 1].astype(np.float32)))
        if d >= color_dist_threshold:
            continue

        mask1 = out == i
        mask2 = out == (i + 1)
        if not mask1.any() or not mask2.any():
            continue

        dilated1 = ndimage.binary_dilation(mask1, structure=np.ones((3, 3), dtype=bool))
        dilated2 = ndimage.binary_dilation(mask2, structure=np.ones((3, 3), dtype=bool))
        boundary = dilated1 & dilated2
        if not boundary.any():
            continue

        coords = np.where(boundary)
        boundary_pixels = panel_rgb[coords].astype(np.float32)
        d1 = np.linalg.norm(boundary_pixels - palette_rgb[i].astype(np.float32), axis=1)
        d2 = np.linalg.norm(boundary_pixels - palette_rgb[i + 1].astype(np.float32), axis=1)
        reclass = d2 < d1
        out[coords[0][reclass], coords[1][reclass]] = i + 1
        out[coords[0][~reclass], coords[1][~reclass]] = i

    return out

__all__ = [
    "_enhance_close_boundaries",
    "_fill_holes",
    "_nearest_median",
    "_remove_small_components",
]
