"""Grayscale / low-saturation panel segmentation.

For seismic reflection profiles, amplitude images, and conceptual diagrams
with minimal color information. Uses grayscale histogram peak detection +
morphological watershed.

Test scenario:
    >>> import numpy as np
    >>> img = np.full((100, 200, 3), 128, dtype=np.uint8)
    >>> img[30:70, :] = 200
    >>> result = segment(img, n_layers=2)
    >>> assert result["labels"].shape == (100, 200)
    >>> assert len(np.unique(result["labels"])) == 2
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.color import rgb2gray
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.segmentation import watershed as sk_watershed, find_boundaries

from geoseg.modules.segment_engines._shared import _create_overlay, saturation_ratio


def _gray_histogram_peaks(
    gray: np.ndarray,
    n_layers: int,
    smooth_sigma: float = 2.0,
    min_peak_prominence: float = 0.02,
) -> list[int]:
    """Find n most-prominent grayscale histogram peaks.

    Returns list of grayscale center values (0-255).
    """
    hist, bin_edges = np.histogram(gray.flatten(), bins=256, range=(0, 256))

    if smooth_sigma > 0:
        hist = ndimage.gaussian_filter1d(hist.astype(np.float64), sigma=smooth_sigma)

    # Find local maxima with minimum prominence
    total_pixels = gray.size
    min_height = min_peak_prominence * total_pixels

    # Use peak_local_max on smoothed histogram
    coords = peak_local_max(
        hist.reshape(1, -1),
        min_distance=max(5, 256 // (n_layers * 4)),
        threshold_abs=min_height,
        exclude_border=False,
    )
    peak_indices = sorted([int(c[1]) for c in coords], key=lambda i: hist[i], reverse=True)

    # Merge very close peaks
    merged: list[int] = []
    for p in peak_indices:
        if not merged or all(abs(p - m) > 10 for m in merged):
            merged.append(p)

    merged = merged[:n_layers]
    if not merged:
        # Fallback: evenly spaced
        merged = [int(np.linspace(10, 245, n_layers)[i]) for i in range(n_layers)]

    return merged


def _watershed_from_peaks(
    gray: np.ndarray,
    peak_vals: list[int],
    compactness: float = 0.5,
) -> np.ndarray:
    """Watershed segmentation using grayscale peaks as markers."""
    h, w = gray.shape

    # Create markers from peak regions
    markers = np.zeros((h, w), dtype=np.int32)
    for i, pv in enumerate(peak_vals):
        # Region within tolerance of peak value
        tol = max(15, 60 // len(peak_vals))
        mask = np.abs(gray.astype(np.int16) - pv) < tol
        if mask.any():
            # Label connected components for this peak
            labeled, num = label(mask, connectivity=2, return_num=True)
            if num > 0:
                # Keep the largest component as marker
                regions = regionprops(labeled)
                largest = max(regions, key=lambda r: r.area)
                markers[labeled == largest.label] = i + 1

    # If no markers found, fall back to grid markers
    if markers.max() == 0:
        for i in range(len(peak_vals)):
            y = int(h * (i + 0.5) / len(peak_vals))
            x = w // 2
            markers[y, x] = i + 1

    # Compute gradient magnitude as elevation map
    gy, gx = np.gradient(gray.astype(np.float64))
    elevation = np.sqrt(gx**2 + gy**2)
    elevation = ndimage.gaussian_filter(elevation, sigma=1.0)

    # Watershed
    labels = sk_watershed(elevation, markers=markers, compactness=compactness)

    # Background = 0, convert to 0-based labels
    out = labels.copy()
    present = np.unique(out[out > 0])
    if len(present) < len(peak_vals):
        # Some peaks didn't get regions - merge into nearest neighbor
        pass

    # Reorder by median y (top to bottom)
    out = _reorder_by_median_y(out)
    return out


def _reorder_by_median_y(labels: np.ndarray) -> np.ndarray:
    """Reorder labels so top=1, bottom=max (0 is reserved for background)."""
    unique = np.unique(labels[labels >= 0])
    if len(unique) == 0:
        return labels.copy()

    median_y = {}
    for lbl in unique:
        ys = np.where(labels == lbl)[0]
        median_y[lbl] = np.median(ys) if len(ys) > 0 else 0

    sorted_by_y = sorted(median_y.items(), key=lambda x: x[1])
    # Map to 1-based labels so 0 remains background
    old_to_new = {old: new + 1 for new, (old, _) in enumerate(sorted_by_y)}

    out = np.zeros_like(labels)
    for old, new in old_to_new.items():
        out[labels == old] = new
    return out


def _merge_tiny_regions(
    labels: np.ndarray,
    min_area_frac: float = 0.005,
) -> np.ndarray:
    """Merge regions smaller than min_area_frac into largest neighbor."""
    h, w = labels.shape
    total = h * w
    min_area = max(50, int(total * min_area_frac))

    out = labels.copy()
    unique = np.unique(out)

    for lbl in unique:
        if lbl < 0:
            continue
        mask = out == lbl
        area = mask.sum()
        if area >= min_area:
            continue

        # Find neighbor by dilating and taking majority
        dilated = ndimage.binary_dilation(mask, structure=np.ones((3, 3), dtype=bool))
        neigh = out[dilated & ~mask & (out >= 0)]
        if len(neigh) == 0:
            continue
        best = int(np.bincount(neigh).argmax())
        out[mask] = best

    # Renumber to be contiguous, starting from 1 (0 = background)
    present = sorted(np.unique(out[out > 0]))
    renum = {old: new + 1 for new, old in enumerate(present)}
    out_clean = np.zeros_like(out)
    for old, new in renum.items():
        out_clean[out == old] = new
    return out_clean


def segment(
    panel_rgb: np.ndarray,
    n_layers: int = 5,
    reps: list[dict] | None = None,
) -> dict:
    """Segment a grayscale or low-saturation panel.

    Args:
        panel_rgb: RGB uint8 array.
        n_layers: Number of layers to extract.
        reps: Optional VLM reps (used as hints for peak positions if provided).

    Returns:
        dict with keys: labels, seeds, overlay, meta.
    """
    h, w = panel_rgb.shape[:2]
    gray = (rgb2gray(panel_rgb) * 255).astype(np.uint8)

    # Determine target number of layers
    target_k = n_layers
    if reps and len(reps) > 0:
        target_k = max(2, min(len(reps), n_layers))

    # Find histogram peaks
    peak_vals = _gray_histogram_peaks(gray, target_k)

    # If reps provided, bias peaks toward rep y-positions
    if reps and len(reps) > 0:
        rep_y_vals = []
        for r in reps:
            y = int(r["representative_point"]["y"])
            y = max(0, min(h - 1, y))
            rep_y_vals.append(int(gray[y, w // 2]))
        # Blend: use rep values if they are distinct
        if len(rep_y_vals) >= target_k:
            peak_vals = rep_y_vals[:target_k]

    # Watershed segmentation
    labels = _watershed_from_peaks(gray, peak_vals)
    labels = _merge_tiny_regions(labels, min_area_frac=0.003)

    # Compute seed colors from median of each region
    n_actual = int(labels.max()) + 1
    seeds_rgb = np.zeros((n_actual, 3), dtype=np.uint8)
    for lbl in range(n_actual):
        mask = labels == lbl
        if mask.any():
            seeds_rgb[lbl] = np.median(panel_rgb[mask], axis=0).astype(np.uint8)
        else:
            seeds_rgb[lbl] = 128

    overlay = _create_overlay(panel_rgb, labels, seeds_rgb)

    return {
        "labels": labels,
        "seeds": seeds_rgb.tolist(),
        "overlay": overlay,
        "meta": {
            "engine": "grayscale",
            "path": "histogram_peaks_watershed",
            "n_layers_target": target_k,
            "n_layers_actual": n_actual,
            "peak_values": peak_vals,
            "saturation_ratio": round(saturation_ratio(panel_rgb), 4),
        },
    }
