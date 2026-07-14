"""Region cleanup helpers for segmentation engines."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.morphology import disk, erosion

from geoseg.core.image_ops import (
    detect_background_label,
    merge_small_regions,
)

SHAPE_RATIO_THRESHOLD = 35.0


def _detect_background_label(labels: np.ndarray) -> int | None:
    """Detect the label most likely to be background.

    Heuristic: the label that covers the largest fraction of the image edge
    AND occupies a substantial total area is treated as background.
    """
    return detect_background_label(labels)


def _erode_internal_point(mask: np.ndarray) -> tuple[int, int] | None:
    """Return (x, y) of a robustly-internal pixel of a binary mask via erosion."""
    m = mask.copy()
    for r in (5, 3, 1):
        eroded = erosion(m, footprint=disk(r))
        if eroded.any():
            m = eroded
            break
    if not m.any():
        return None
    ys, xs = np.where(m)
    cx, cy = int(xs.mean()), int(ys.mean())
    if not m[cy, cx]:
        cx, cy = int(np.median(xs)), int(np.median(ys))
    return cx, cy


def _shape_filter(labels: np.ndarray, ratio_threshold: float = SHAPE_RATIO_THRESHOLD) -> np.ndarray:
    """Post-process labels: merge thin 1-D components into adjacent 2-D zones."""
    h, w = labels.shape
    out = labels.copy()
    cc = label(labels > -1, connectivity=2)
    regions = regionprops(cc)
    if not regions:
        return out

    thin_mask = np.zeros((h, w), dtype=bool)
    thin_labels = set()
    for r in regions:
        area = max(r.area, 1e-9)
        perim = r.perimeter
        ratio = float("inf") if perim == 0 else (perim ** 2) / area
        if ratio > ratio_threshold:
            thin_mask[cc == r.label] = True
            thin_labels.add(r.label)

    if not thin_labels:
        return out

    for r in regions:
        if r.label not in thin_labels:
            continue
        comp_mask = cc == r.label
        neigh = ndimage.binary_dilation(comp_mask, structure=np.ones((3, 3), dtype=bool))
        neigh_pixels = out[neigh & ~thin_mask]
        if neigh_pixels.size == 0:
            continue
        vals, counts = np.unique(neigh_pixels, return_counts=True)
        best = vals[counts.argmax()]
        out[comp_mask] = best

    return out


def _reorder_labels_by_median_y(labels: np.ndarray) -> np.ndarray:
    """Reorder labels so top labels receive lower numeric IDs."""
    h, _w = labels.shape
    unique = np.unique(labels[labels >= 0])
    if len(unique) == 0:
        return labels.copy()

    median_y = {}
    for lbl in unique:
        ys = np.where(labels == lbl)[0]
        median_y[lbl] = np.median(ys) if len(ys) > 0 else h

    sorted_by_y = sorted(median_y.items(), key=lambda item: item[1])
    old_to_new = {old: new for new, (old, _) in enumerate(sorted_by_y)}

    out = np.full_like(labels, -1)
    for old, new in old_to_new.items():
        out[labels == old] = new
    return out


def _merge_small_regions(labels: np.ndarray, min_area_frac: float = 0.003) -> np.ndarray:
    """Merge tiny connected components (< min_area_frac of image) into largest neighbor."""
    return merge_small_regions(labels, min_area_frac=min_area_frac)


__all__ = [
    "SHAPE_RATIO_THRESHOLD",
    "_detect_background_label",
    "_erode_internal_point",
    "_reorder_labels_by_median_y",
    "_shape_filter",
    "_merge_small_regions",
]
