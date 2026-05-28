"""Horizon refinement: smooth boundary fitting for fragmented segmentations.

Converts pixel-wise clustering results into geologically plausible layers
by fitting smooth curves to layer boundaries and adjusting only boundary
pixels, preserving the interior partition structure of the coarse result.

Four-phase pipeline:
1. Coarse segmentation (blur + downsample + k-means) — only if needed
2. Boundary point extraction (per-column max-gradient sampling)
3. Curve fitting (Savitzky-Golay / B-spline / LOESS)
4. Boundary adjustment (relabel only pixels near fitted boundaries)
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
    label_blur_sigma: float = 5.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Phase B: sample boundary points between two adjacent layers.

    Creates a signed map from the coarse labels (+1 for layer_i, -1 for
    layer_j), applies Gaussian blur to smooth out fragmentation, then finds
    zero-crossing points per column. This is more robust than gradient-based
    sampling on the original RGB image because it averages out small fragments
    and finds the visual center of the transition band.
    """
    h, w = panel_rgb.shape[:2]

    # Signed map: layer_i = +1, layer_j = -1, others = 0
    signed = np.zeros((h, w), dtype=np.float32)
    signed[coarse_labels == layer_i] = 1.0
    signed[coarse_labels == layer_j] = -1.0

    # Gaussian blur to smooth out fragments and noise
    blurred = ndimage.gaussian_filter(signed, sigma=label_blur_sigma)

    xs = []
    ys = []

    for x in range(w):
        col = blurred[:, x]

        # Find zero-crossing: where col transitions from positive to negative
        pos_mask = col > 0
        neg_mask = col < 0

        if not pos_mask.any() or not neg_mask.any():
            continue

        # Find the crossing from + to - (top to bottom)
        found = False
        for y in range(h - 1):
            if col[y] > 0 and col[y + 1] <= 0:
                denom = col[y] - col[y + 1]
                if denom > 1e-6:
                    y_interp = y + col[y] / denom
                else:
                    y_interp = y + 0.5
                xs.append(x)
                ys.append(float(y_interp))
                found = True
                break

        if not found:
            # Fallback: use bottom edge of positive region
            pos_ys = np.where(pos_mask)[0]
            if len(pos_ys) > 0:
                xs.append(x)
                ys.append(float(pos_ys[-1]))

    if len(xs) < 3:
        return None

    ys_arr = np.array(ys, dtype=np.float32)

    # If the zero-crossing position varies wildly (MAD > h/10), the boundary
    # is not spatially coherent — this usually means the layers are not truly
    # adjacent or the coarse segmentation is on a smooth gradient. Skip fitting.
    mad = float(np.median(np.abs(ys_arr - np.median(ys_arr))))
    if mad > h / 10:
        return None

    return np.array(xs, dtype=np.int32), ys_arr


def _extract_boundary_dense(
    coarse_labels: np.ndarray,
    top_lbl: int,
    bot_lbl: int,
) -> np.ndarray:
    """Extract boundary candidates for non-touching fragmented layers.

    When two layers are so fragmented that they don't share any touching
    pixels, the label-blur zero-crossing method fails. Instead, we sample
    per-column using percentile-based edge detection on the raw label map:
    - top layer's lower edge = 90th percentile of its pixels in the column
    - bottom layer's upper edge = 10th percentile of its pixels in the column
    - boundary candidate = midpoint between these edges

    This treats each layer's fragments as an "archipelago" and finds the
    transition band between archipelagos.
    """
    h, w = coarse_labels.shape
    ys = np.full(w, np.nan)

    for x in range(w):
        col = coarse_labels[:, x]
        top_mask = col == top_lbl
        bot_mask = col == bot_lbl

        if not top_mask.any() or not bot_mask.any():
            continue

        top_ys = np.where(top_mask)[0]
        bot_ys = np.where(bot_mask)[0]

        # Use percentiles to be robust to fragments
        top_lower = float(np.percentile(top_ys, 90))
        bot_upper = float(np.percentile(bot_ys, 10))

        if bot_upper > top_lower:
            # Normal separation: boundary is in the gap
            ys[x] = (top_lower + bot_upper) / 2
        else:
            # Interleaved: layers overlap in this column
            transition_start = min(top_ys.min(), bot_ys.min())
            transition_end = max(top_ys.max(), bot_ys.max())
            ys[x] = (transition_start + transition_end) / 2

    return ys


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


def _fit_quintic(y: np.ndarray, smoothness: float = 1.0) -> np.ndarray:
    """Quintic spline minimizing |y'''|^2 — curvature-variation prior.

    A quintic (k=5) spline naturally minimizes the integral of the squared
    third derivative, which corresponds to penalizing continuous large changes
    in curvature. This is the variational prior requested for extremely
    fragmented images like 16b0cf.

    Args:
        y: Boundary y-coordinates per column (may contain NaN gaps).
        smoothness: Controls trade-off between fidelity and smoothness.
            s = smoothness * 1e6 is passed to UnivariateSpline.
    """
    n = len(y)
    x = np.arange(n)
    valid = ~np.isnan(y)
    if np.sum(valid) < 10:
        return y.copy()

    # Fill gaps with linear interpolation
    ys_filled = y.copy()
    ys_filled[~valid] = np.interp(x[~valid], x[valid], y[valid])

    # Outlier rejection before spline fitting
    ys_clean = _hampel_filter(ys_filled)

    s = smoothness * 1e6
    try:
        spline = UnivariateSpline(x, ys_clean, k=5, s=s)
        return spline(x)
    except Exception:
        return ys_filled


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
    method: Literal["savgol", "bspline", "loess", "quintic"],
    smoothness: float,
) -> np.ndarray:
    """Phase C: fit a smooth curve through boundary points."""
    if method == "quintic":
        return _fit_quintic(y, smoothness)

    if len(x) < 5:
        return y.copy()

    # Outlier rejection
    y_clean = _hampel_filter(y)

    if method == "savgol":
        return _fit_savgol(x, y_clean, smoothness)
    if method == "bspline":
        return _fit_bspline(x, y_clean, smoothness)
    return _fit_loess(x, y_clean, smoothness)


def _adjust_boundaries(
    coarse_labels: np.ndarray,
    boundaries: list[np.ndarray],
    boundary_pairs: list[tuple[int, int]],
    blend_width: int = 5,
) -> np.ndarray:
    """Adjust only boundary-adjacent pixels, preserving coarse interior.

    For each fitted boundary between two layers, we identify the pixels in
    the coarse result that actually touch the adjacent layer (boundary pixels),
    dilate slightly to form an adjustment zone, and relabel pixels within
    that zone based on the smooth boundary position. Pixels far from the
    true boundary (interior of layers) are never touched.
    """
    h, w = coarse_labels.shape
    result = coarse_labels.copy()

    for boundary_y, (top_lbl, bot_lbl) in zip(boundaries, boundary_pairs):
        if len(boundary_y) != w:
            continue

        mask_top = coarse_labels == top_lbl
        mask_bot = coarse_labels == bot_lbl

        # Pixels of top_lbl that touch bot_lbl, and vice versa
        boundary_top = mask_top & ndimage.binary_dilation(mask_bot, iterations=1)
        boundary_bot = mask_bot & ndimage.binary_dilation(mask_top, iterations=1)

        # Dilate to create a narrow adjustment zone around the true boundary
        zone = ndimage.binary_dilation(boundary_top | boundary_bot, iterations=blend_width)

        for x in range(w):
            y_b = int(np.clip(round(boundary_y[x]), 0, h - 1))

            ys = np.where(zone[:, x])[0]
            if len(ys) == 0:
                continue

            for y in ys:
                if result[y, x] not in (top_lbl, bot_lbl):
                    continue
                result[y, x] = top_lbl if y <= y_b else bot_lbl

    return result


def _repartition_columns(
    coarse_labels: np.ndarray,
    spatial_order: list[int],
    boundaries: list[np.ndarray],
) -> np.ndarray:
    """Global column-wise repartitioning for severely fragmented images.

    When layer pairs are so fragmented they don't touch, local adjustment
    cannot reach the interior fragments. This function repartitions the
    ENTIRE image column-by-column using the fitted smooth boundaries:

    For each column x:
        y_0 = 0
        for each boundary i at position b_i[x]:
            assign pixels [y_{i-1}, b_i) to spatial_order[i]
        assign remaining pixels to spatial_order[-1]

    This treats each layer's fragments as an "archipelago" and redraws all
    maritime borders simultaneously. Foreign fragments (e.g. yellow pixels
    in the blue layer) are eliminated because every pixel is reassigned
    based on its vertical position relative to the smooth boundaries.

    Preserves: global layer ordering and identity.
    """
    h, w = coarse_labels.shape
    result = np.zeros_like(coarse_labels)

    for x in range(w):
        boundary_ys = [int(np.clip(round(b[x]), 0, h - 1)) for b in boundaries]
        boundary_ys = sorted(boundary_ys)

        prev_y = 0
        for i, y_b in enumerate(boundary_ys):
            lbl = spatial_order[i]
            y_b = min(y_b, h)
            result[prev_y:y_b, x] = lbl
            prev_y = y_b
        # Last layer
        if len(spatial_order) > len(boundary_ys):
            result[prev_y:h, x] = spatial_order[len(boundary_ys)]

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
    method: Literal["savgol", "bspline", "loess", "quintic"] = "savgol",
    smoothness: float = 1.0,
    blur_sigma: float = 2.0,
    downsample_factor: float = 0.25,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Refine fragmented segmentation by fitting smooth horizons.

    Four-phase pipeline:
    1. Coarse segmentation (blur + downsample + k-means) — only if needed
    2. Boundary point extraction (label-blur zero-crossing or density-based)
    3. Curve fitting (Savitzky-Golay / B-spline / LOESS / Quintic)
    4. Boundary adjustment (local for touching layers, column-repartition for fragmented)

    Key invariant: original coarse label IDs and global layer ordering are
    preserved. For touching layer pairs, only boundary-adjacent pixels are
    adjusted. For non-touching (severely fragmented) pairs, column-wise
    repartitioning redraws the border between archipelagos.

    Args:
        panel_rgb: Original RGB image (H, W, 3) uint8.
        coarse_labels: Initial label map from any engine. If None, computed
            internally via Phase A using n_layers.
        n_layers: Target layer count. Required if coarse_labels is None.
        method: Curve fitting method. Use "quintic" for curvature-variation
            prior (best for extremely fragmented images like 16b0cf).
        smoothness: Smoothness factor. Interpretation varies by method:
            - savgol: window_length = int(smoothness * W)
            - bspline: s = smoothness * 1e4 scale factor
            - loess: frac = smoothness * 0.1
            - quintic: s = smoothness * 1e6 (UnivariateSpline smoothing)
        blur_sigma: Gaussian blur sigma for Phase A coarse segmentation.
        downsample_factor: Downsample ratio for Phase A.

    Returns:
        refined_labels: Label map with smoothed boundaries (H, W).
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

    # --- Spatially order labels (top-to-bottom by median y) ---
    bg_labels = [u for u in unique if u == 0]
    layer_labels = [u for u in unique if u != 0]

    if not layer_labels:
        return coarse_labels.copy(), []

    median_ys: dict[int, float] = {}
    for lbl in layer_labels:
        ys = np.where(coarse_labels == lbl)[0]
        median_ys[lbl] = float(np.median(ys)) if len(ys) > 0 else h

    spatial_order = sorted(layer_labels, key=lambda lbl: median_ys[lbl])

    # --- Fit boundaries between spatially adjacent layers ---
    boundaries: list[np.ndarray] = []
    boundary_pairs: list[tuple[int, int]] = []
    broken_pairs: set[tuple[int, int]] = set()  # non-touching (fragmented) pairs

    for i in range(len(spatial_order) - 1):
        top_lbl = spatial_order[i]
        bot_lbl = spatial_order[i + 1]

        # Check if layers actually touch (share boundary pixels)
        mask_top = coarse_labels == top_lbl
        mask_bot = coarse_labels == bot_lbl
        touch = mask_top & ndimage.binary_dilation(mask_bot, iterations=1)
        is_broken = np.sum(touch) == 0

        if is_broken:
            # Non-touching: use density-based extraction + quintic spline
            ys_raw = _extract_boundary_dense(coarse_labels, top_lbl, bot_lbl)
            if ys_raw is None or np.sum(~np.isnan(ys_raw)) < 10:
                continue
            # Force quintic for broken boundaries (curvature-variation prior)
            boundary_y = _fit_quintic(ys_raw, smoothness)
            broken_pairs.add((top_lbl, bot_lbl))
        else:
            # Touching: use label-blur zero-crossing
            points = _extract_boundary_points(panel_rgb, coarse_labels, top_lbl, bot_lbl)
            if points is None:
                continue
            xs, ys = points
            boundary_y = _fit_curve(xs, ys, method, smoothness)

            # Fill gaps
            full_y = np.full(w, np.nan, dtype=np.float32)
            full_y[xs] = boundary_y
            full_y = ndimage.generic_filter(
                full_y, lambda v: np.nanmedian(v) if np.any(~np.isnan(v)) else h // 2,
                size=11, mode="nearest"
            )
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

    if not boundaries:
        return coarse_labels.copy(), []

    # --- Enforce monotonicity: sort boundaries by median y ---
    if len(boundaries) > 1:
        medians = [float(np.median(b)) for b in boundaries]
        order = np.argsort(medians)
        boundaries = [boundaries[int(i)] for i in order]
        boundary_pairs = [boundary_pairs[int(i)] for i in order]
        # Rebuild broken_pairs with sorted order
        broken_pairs_sorted = set()
        for i in order:
            pair = boundary_pairs[int(i)]
            if pair in broken_pairs:
                broken_pairs_sorted.add(pair)
        broken_pairs = broken_pairs_sorted

    # --- Sanity check: minimum boundary separation ---
    min_layer_height = max(3, h // 100)
    if len(boundaries) > 1:
        for i in range(len(boundaries) - 1):
            gap = np.median(boundaries[i + 1]) - np.median(boundaries[i])
            if gap < min_layer_height:
                return coarse_labels.copy(), boundaries

    # --- Apply adjustment strategy per boundary pair ---
    if broken_pairs:
        # Severely fragmented: global column-wise repartitioning
        refined_labels = _repartition_columns(coarse_labels, spatial_order, boundaries)
    else:
        # Normal fragmentation: local boundary adjustment only
        refined_labels = _adjust_boundaries(coarse_labels, boundaries, boundary_pairs, blend_width=5)

    # --- Quality gate: pixel change ratio ---
    pixel_change_ratio = np.sum(refined_labels != coarse_labels) / (h * w)
    # Broken boundaries may legitimately need high change ratios (up to ~40%)
    max_change = 0.50 if broken_pairs else 0.15
    if pixel_change_ratio > max_change:
        return coarse_labels.copy(), boundaries

    # --- Quality gate: layer count preservation ---
    refined_unique = sorted(u for u in np.unique(refined_labels) if u >= 0)
    expected_layers = len(layer_labels)
    if len(refined_unique) < expected_layers - 1:
        return coarse_labels.copy(), boundaries

    # --- Quality gate: fragmentation ---
    coarse_frag = _compute_fragmentation_score(coarse_labels)
    refined_frag = _compute_fragmentation_score(refined_labels)
    # Broken boundaries get a more lenient threshold since they target deep fragmentation
    frag_threshold = 1.5 if broken_pairs else 1.2
    if refined_frag > coarse_frag * frag_threshold:
        return coarse_labels.copy(), boundaries

    return refined_labels, boundaries
