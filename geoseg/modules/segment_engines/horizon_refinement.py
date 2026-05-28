"""Horizon refinement: smooth boundary fitting for fragmented segmentations.

Converts pixel-wise clustering results into geologically plausible layers
by fitting smooth curves to layer boundaries and re-partitioning the image.

Four-phase pipeline:
1. Coarse segmentation (blur + downsample + k-means)
2. Boundary point extraction (per-column max-gradient sampling)
3. Curve fitting (Savitzky-Golay / B-spline / LOESS)
4. Re-labeling (partition by fitted boundaries)
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from scipy import ndimage
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from skimage.transform import resize


def _coarse_segment(
    panel_rgb: np.ndarray,
    n_layers: int,
    blur_sigma: float = 2.0,
    downsample_factor: float = 0.25,
) -> np.ndarray:
    """Phase A: coarse layer segmentation at low resolution."""
    h, w = panel_rgb.shape[:2]

    # Gaussian blur to suppress decorative gradients and noise
    blurred = ndimage.gaussian_filter(panel_rgb, sigma=(blur_sigma, blur_sigma, 0))

    # Downsample for computational efficiency and further noise averaging
    small = resize(blurred, (int(h * downsample_factor), int(w * downsample_factor)),
                   order=1, preserve_range=True, anti_aliasing=True).astype(np.uint8)

    # K-means in RGB space at low resolution
    pixels = small.reshape(-1, 3).astype(np.float64)
    from scipy.cluster.vq import kmeans2
    centroids, labels_flat = kmeans2(pixels, n_layers, minit="++", seed=42)
    coarse_small = labels_flat.reshape(small.shape[:2]).astype(np.int32)

    # Upsample back to original size with nearest-neighbor (preserves sharp-ish edges)
    coarse = resize(coarse_small, (h, w), order=0, preserve_range=True,
                    anti_aliasing=False).astype(np.int32)

    return coarse


def _extract_boundary_points(
    panel_rgb: np.ndarray,
    coarse_labels: np.ndarray,
    layer_i: int,
    layer_j: int,
    band_margin: int = 15,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Phase B: sample boundary points between two adjacent layers.

    For each column x, find the y-coordinate of maximum RGB gradient
    within the transition band between layer_i and layer_j.
    """
    h, w = panel_rgb.shape[:2]

    # Build transition band mask
    mask_i = coarse_labels == layer_i
    mask_j = coarse_labels == layer_j
    transition = mask_i | mask_j

    # Dilate to capture transition neighborhood
    transition = ndimage.binary_dilation(transition, iterations=band_margin)

    xs = []
    ys = []

    for x in range(w):
        col_transition = np.where(transition[:, x])[0]
        if len(col_transition) < 3:
            continue

        y_min, y_max = col_transition[0], col_transition[-1]
        y_min = max(0, y_min - band_margin)
        y_max = min(h - 1, y_max + band_margin)

        # Compute vertical gradient magnitude in RGB
        col_rgb = panel_rgb[y_min:y_max, x, :].astype(np.float32)
        if len(col_rgb) < 2:
            continue

        grad = np.abs(np.diff(col_rgb, axis=0)).mean(axis=1)
        if grad.size == 0:
            continue

        # Find local maxima of gradient, preferring center of band
        local_max = (grad[1:-1] >= grad[:-2]) & (grad[1:-1] >= grad[2:])
        candidates = np.where(local_max)[0] + 1

        if len(candidates) == 0:
            # Fallback: global max
            y_rel = int(np.argmax(grad))
        else:
            # Pick candidate closest to center of band
            band_center = len(grad) // 2
            best = candidates[np.argmin(np.abs(candidates - band_center))]
            y_rel = int(best)

        y = y_min + y_rel
        y = max(0, min(h - 1, y))

        xs.append(x)
        ys.append(y)

    if len(xs) < 3:
        return None

    return np.array(xs, dtype=np.int32), np.array(ys, dtype=np.float32)


def _hampel_filter(y: np.ndarray, window: int = 21, n_sigma: float = 3.0) -> np.ndarray:
    """Remove outlier points using Hampel identifier."""
    if len(y) < window:
        return y.copy()

    out = y.copy()
    half = window // 2

    for i in range(len(y)):
        start = max(0, i - half)
        end = min(len(y), i + half + 1)
        window_vals = y[start:end]
        median = np.median(window_vals)
        mad = np.median(np.abs(window_vals - median))
        threshold = n_sigma * 1.4826 * mad
        if np.abs(y[i] - median) > threshold:
            out[i] = median

    return out


def _fit_savgol(x: np.ndarray, y: np.ndarray, smoothness: float) -> np.ndarray:
    """Savitzky-Golay filter for locally-adaptive smoothing."""
    n = len(y)
    window = max(5, int(smoothness * n))
    if window % 2 == 0:
        window += 1
    window = min(window, n - 1 if n % 2 == 0 else n - 2)
    if window < 5:
        return y.copy()

    polyorder = min(3, window - 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # mode='mirror' reduces edge extrapolation artifacts compared to default 'interp'
        return savgol_filter(y, window_length=window, polyorder=polyorder, mode="mirror")


def _fit_bspline(x: np.ndarray, y: np.ndarray, smoothness: float) -> np.ndarray:
    """B-spline fit for globally smooth curves."""
    # s parameter: smoothness * number_of_points * variance
    s = smoothness * len(y) * float(np.var(y)) * 0.01
    s = max(s, len(y) * 0.1)

    # Sort by x to ensure monotonicity for spline
    order = np.argsort(x)
    x_sorted = x[order].astype(np.float64)
    y_sorted = y[order].astype(np.float64)

    # Remove duplicate x values
    unique_mask = np.concatenate(([True], np.diff(x_sorted) > 0))
    x_unique = x_sorted[unique_mask]
    y_unique = y_sorted[unique_mask]

    if len(x_unique) < 4:
        return y.copy()

    try:
        spline = UnivariateSpline(x_unique, y_unique, s=s, k=3)
        return spline(x.astype(np.float64))
    except Exception:
        return y.copy()


def _fit_loess(x: np.ndarray, y: np.ndarray, smoothness: float) -> np.ndarray:
    """LOESS local regression."""
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except ImportError:
        # Fallback to Savitzky-Golay if statsmodels not available
        return _fit_savgol(x, y, smoothness)

    frac = max(0.02, min(0.5, smoothness * 0.1))
    result = lowess(y, x, frac=frac, return_sorted=False)
    return result[:, 1] if result.ndim > 1 else result


def _fit_curve(
    x: np.ndarray,
    y: np.ndarray,
    method: Literal["savgol", "bspline", "loess"],
    smoothness: float,
) -> np.ndarray:
    """Phase C: fit a smooth curve through boundary points."""
    if len(x) < 5:
        return y.copy()

    # Outlier rejection
    y_clean = _hampel_filter(y)

    if method == "savgol":
        return _fit_savgol(x, y_clean, smoothness)
    if method == "bspline":
        return _fit_bspline(x, y_clean, smoothness)
    return _fit_loess(x, y_clean, smoothness)


def _relabel_from_boundaries(
    h: int,
    w: int,
    boundaries: list[np.ndarray],
    unique_labels: list[int],
) -> np.ndarray:
    """Phase D: re-partition image by fitted boundary curves.

    Preserves layer IDs from coarse segmentation. Each column x is divided
    by the y-positions of boundaries at that x, and intervals are assigned
    the original coarse label values (not renumbered 0,1,2...).
    """
    labels = np.zeros((h, w), dtype=np.int32)
    n_layers = len(unique_labels)

    for x in range(w):
        y_bounds = [0]
        for b in boundaries:
            if len(b) > x:
                y = int(np.clip(round(b[x]), 0, h - 1))
                y_bounds.append(y)
        y_bounds.append(h)
        y_bounds = sorted(set(y_bounds))

        # Assign original coarse labels to intervals
        for i, (y_top, y_bottom) in enumerate(zip(y_bounds, y_bounds[1:])):
            if i < n_layers and y_top < y_bottom and y_top < h:
                labels[y_top:y_bottom, x] = unique_labels[i]

        # Fill any remaining bottom area with last layer
        if y_bounds and y_bounds[-1] < h and n_layers > 0:
            labels[y_bounds[-1]:h, x] = unique_labels[-1]

    return labels


def _blend_with_coarse(
    coarse_labels: np.ndarray,
    refined_labels: np.ndarray,
    boundaries: list[np.ndarray],
    blend_width: int = 5,
) -> np.ndarray:
    """Blend coarse and refined labels: keep coarse in interiors, use refined near boundaries.

    This is the core of "Strategy 2": smooth boundaries without re-partitioning
    the entire image. Only pixels within `blend_width` of a fitted boundary
    are allowed to change their label.
    """
    h, w = coarse_labels.shape
    result = coarse_labels.copy()

    # Mark boundary-adjacent pixels
    boundary_zone = np.zeros((h, w), dtype=bool)
    for b in boundaries:
        for x in range(w):
            if len(b) <= x:
                continue
            y = int(np.clip(round(b[x]), 0, h - 1))
            y_min = max(0, y - blend_width)
            y_max = min(h, y + blend_width + 1)
            boundary_zone[y_min:y_max, x] = True

    # Within boundary zone, accept refined label ONLY if it matches one of
    # the coarse labels present in that local neighborhood (prevents wild
    # label assignments from curve oscillations).
    for x in range(w):
        zone_y = np.where(boundary_zone[:, x])[0]
        if len(zone_y) == 0:
            continue

        # Collect coarse labels present in this column's boundary zone
        local_coarse = coarse_labels[zone_y, x]
        allowed = set(np.unique(local_coarse))

        for y in zone_y:
            refined_lbl = refined_labels[y, x]
            if refined_lbl in allowed:
                result[y, x] = refined_lbl

    return result


def _compute_fragmentation_score(labels: np.ndarray) -> float:
    """Compute a fragmentation score: fraction of pixels in tiny components."""
    from scipy import ndimage
    total_tiny = 0
    h, w = labels.shape
    min_area = max(50, int(h * w * 0.001))
    for lbl in np.unique(labels):
        mask = labels == lbl
        if not mask.any():
            continue
        labeled, num = ndimage.label(mask)
        if num <= 1:
            continue
        sizes = ndimage.sum(mask, labeled, range(1, num + 1))
        tiny = np.sum(sizes[sizes < min_area])
        total_tiny += int(tiny)
    return total_tiny / (h * w)


def refine_boundaries(
    panel_rgb: np.ndarray,
    coarse_labels: np.ndarray | None = None,
    n_layers: int | None = None,
    method: Literal["savgol", "bspline", "loess"] = "savgol",
    smoothness: float = 1.0,
    blur_sigma: float = 2.0,
    downsample_factor: float = 0.25,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Refine fragmented segmentation by fitting smooth horizons.

    Four-phase pipeline:
    1. Coarse segmentation (blur + downsample + k-means)
    2. Boundary point extraction (per-column max-gradient)
    3. Curve fitting (Savitzky-Golay / B-spline / LOESS)
    4. Re-labeling (partition by fitted boundaries)

    Args:
        panel_rgb: Original RGB image (H, W, 3) uint8.
        coarse_labels: Initial label map from any engine. If None, computed
            internally via Phase A using n_layers.
        n_layers: Target layer count. Required if coarse_labels is None.
        method: Curve fitting method.
        smoothness: Smoothness factor. Interpretation varies by method:
            - savgol: window_length = int(smoothness * W)
            - bspline: s = smoothness * 1e4 scale factor
            - loess: frac = smoothness * 0.1
        blur_sigma: Gaussian blur sigma for Phase A coarse segmentation.
        downsample_factor: Downsample ratio for Phase A.

    Returns:
        refined_labels: Re-labeled map with smooth boundaries (H, W).
        boundaries: List of y-coordinate arrays for each fitted horizon.
    """
    h, w = panel_rgb.shape[:2]

    if coarse_labels is None:
        if n_layers is None:
            raise ValueError("n_layers required when coarse_labels is None")
        coarse_labels = _coarse_segment(
            panel_rgb, n_layers, blur_sigma, downsample_factor
        )

    unique = sorted(u for u in np.unique(coarse_labels) if u >= 0)
    if len(unique) < 2:
        return coarse_labels.copy(), []

    # --- Robustness: spatially order labels (top-to-bottom by median y) ---
    # Background (0) is kept separate; non-background labels are sorted by
    # vertical position so boundaries are fitted between truly adjacent layers.
    bg_labels = [u for u in unique if u == 0]
    layer_labels = [u for u in unique if u != 0]

    if not layer_labels:
        return coarse_labels.copy(), []

    median_ys: dict[int, float] = {}
    for lbl in layer_labels:
        ys = np.where(coarse_labels == lbl)[0]
        median_ys[lbl] = float(np.median(ys)) if len(ys) > 0 else h

    # Sort by median y (top = smallest y)
    spatial_order = sorted(layer_labels, key=lambda lbl: median_ys[lbl])
    unique_spatial = bg_labels + spatial_order

    # --- Fit boundaries between spatially adjacent layers ---
    boundaries: list[np.ndarray] = []
    boundary_pairs: list[tuple[int, int]] = []

    for i in range(len(spatial_order) - 1):
        top_lbl = spatial_order[i]
        bot_lbl = spatial_order[i + 1]

        points = _extract_boundary_points(panel_rgb, coarse_labels, top_lbl, bot_lbl)
        if points is None:
            # Fallback: use horizontal line at median transition y
            mask_top = coarse_labels == top_lbl
            mask_bot = coarse_labels == bot_lbl
            transition = ndimage.binary_dilation(mask_top) & mask_bot
            if not transition.any():
                transition = mask_top | mask_bot
            median_y = int(np.median(np.where(transition)[0])) if transition.any() else h // 2
            boundary_y = np.full(w, median_y, dtype=np.float32)
        else:
            xs, ys = points
            boundary_y = _fit_curve(xs, ys, method, smoothness)
            # Fill gaps (columns where no point was sampled)
            full_y = np.full(w, np.nan, dtype=np.float32)
            full_y[xs] = boundary_y
            full_y = ndimage.generic_filter(
                full_y, lambda v: np.nanmedian(v) if np.any(~np.isnan(v)) else h // 2,
                size=11, mode="nearest"
            )
            # Interpolate remaining NaNs
            nan_mask = np.isnan(full_y)
            if nan_mask.any() and not nan_mask.all():
                full_y[nan_mask] = np.interp(
                    np.where(nan_mask)[0],
                    np.where(~nan_mask)[0],
                    full_y[~nan_mask],
                )
            boundary_y = full_y

        boundaries.append(boundary_y)
        boundary_pairs.append((top_lbl, bot_lbl))

    # --- Enforce monotonicity: sort boundaries by median y ---
    # This prevents crossing boundaries which create tiny slivers.
    if len(boundaries) > 1:
        medians = [float(np.median(b)) for b in boundaries]
        order = np.argsort(medians)
        boundaries = [boundaries[int(i)] for i in order]
        boundary_pairs = [boundary_pairs[int(i)] for i in order]

    # --- Sanity check: minimum boundary separation ---
    # If boundaries collapsed (gap < 3 px), the coarse labels are too mixed
    # to support meaningful boundary fitting.
    min_layer_height = max(3, h // 100)
    if len(boundaries) > 1:
        for i in range(len(boundaries) - 1):
            gap = np.median(boundaries[i + 1]) - np.median(boundaries[i])
            if gap < min_layer_height:
                return coarse_labels.copy(), boundaries

    # --- Re-label using spatially ordered boundaries ---
    refined_labels = _relabel_from_boundaries(h, w, boundaries, spatial_order)

    # --- Quality gate: layer count and fragmentation ---
    refined_unique = sorted(u for u in np.unique(refined_labels) if u >= 0)
    expected_layers = len(spatial_order)
    if len(refined_unique) < expected_layers - 1:
        # Lost more than 1 layer — refinement failed
        return coarse_labels.copy(), boundaries

    coarse_frag = _compute_fragmentation_score(coarse_labels)
    refined_frag = _compute_fragmentation_score(refined_labels)
    if refined_frag > coarse_frag * 1.5 + 0.01:
        # Refined result is significantly worse; skip refinement
        return coarse_labels.copy(), boundaries

    return refined_labels, boundaries
