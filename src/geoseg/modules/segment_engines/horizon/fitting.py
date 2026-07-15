"""Curve fitting helpers for horizon refinement."""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter


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


def _fit_multiscale_savgol(y: np.ndarray, base_smoothness: float = 0.15) -> np.ndarray:
    """Multi-scale Savgol fusion: preserve local features while smoothing flat regions.

    Fits two Savgol curves — one fine (small window, preserves detail) and one
    coarse (large window, smooth). Where they diverge significantly, the local
    structure is "real" (high curvature) so we trust the fine fit. Where they
    agree, we use the smooth fit. This is an edge-preserving smoothing strategy
    analogous to bilateral filtering.
    """
    n = len(y)
    if n < 5:
        return y.copy()

    # Fill NaN before fitting
    y_filled = y.copy()
    valid = ~np.isnan(y)
    if valid.any() and not valid.all():
        y_filled[~valid] = np.interp(
            np.where(~valid)[0], np.where(valid)[0], y[valid]
        )

    # Fine fit: small window preserves local peaks/valleys
    y_fine = _fit_savgol(np.arange(n), y_filled, base_smoothness * 0.5)

    # Coarse fit: large window for global smoothness
    y_coarse = _fit_savgol(np.arange(n), y_filled, base_smoothness * 1.5)

    # Divergence map: where do fine and coarse disagree?
    diff = np.abs(y_fine - y_coarse)
    max_diff = np.max(diff) + 1e-6
    weight_fine = np.clip(diff / max_diff, 0.0, 1.0)  # high where local structure exists
    weight_coarse = 1.0 - weight_fine

    # Fuse: edge-preserving blend
    result = weight_fine * y_fine + weight_coarse * y_coarse
    return result


def _detect_knots(
    y: np.ndarray,
    prominence: float = 5.0,
    min_distance: int = 30,
) -> np.ndarray:
    """Detect significant structural points (knots) on a boundary curve.

    Uses scipy.signal.find_peaks to locate local maxima and minima whose
    prominence exceeds a threshold. Small sawtooth wiggles (prominence <
    threshold) are treated as noise and ignored. Large geological features
    (prominence > threshold) are preserved as knots.

    Args:
        y: Boundary y-coordinates (NaN filled).
        prominence: Minimum peak prominence in pixels. A local extremum must
            rise/fall at least this much above its surrounding baseline to
            qualify as a knot.
        min_distance: Minimum horizontal distance between two knots in pixels.
            Prevents over-segmentation from dense small wiggles.

    Returns:
        Array of knot indices (includes start and end points).
    """
    from scipy.signal import find_peaks

    y_safe = np.nan_to_num(y, nan=float(np.nanmedian(y)))

    peaks_max, _ = find_peaks(y_safe, prominence=prominence, distance=min_distance)
    peaks_min, _ = find_peaks(-y_safe, prominence=prominence, distance=min_distance)

    knots = sorted(set(peaks_max) | set(peaks_min))

    # Always include boundaries
    if 0 not in knots:
        knots = [0] + knots
    if len(y) - 1 not in knots:
        knots = knots + [len(y) - 1]

    return np.array(sorted(set(knots)))


def _fit_knot_constrained(
    x: np.ndarray,
    y: np.ndarray,
    prominence: float = 5.0,
    min_distance: int = 30,
    base_smoothness: float = 0.15,
) -> np.ndarray:
    """Knot-constrained spline fit: preserve significant geological features.

    1. Detect structural knots (significant local extrema) on raw boundary.
    2. Build a weighted UnivariateSpline where knots have high weight
       (soft constraint) and non-knot regions have low weight (free to smooth).
    3. This yields a curve that is smooth overall but faithfully reproduces
       meaningful curvature changes (e.g. anticlines, synclines) while
       filtering pixel-level sawtooth noise.

    Args:
        x: Column indices.
        y: Raw boundary y-coordinates (NaN filled).
        prominence: Knot detection prominence threshold (pixels).
        min_distance: Minimum knot spacing (pixels).
        base_smoothness: Smoothing factor passed to UnivariateSpline.
    """
    if len(x) < 10:
        return y.copy()

    knots = _detect_knots(y, prominence=prominence, min_distance=min_distance)

    # Build weights: knots and their neighbours get high weight (soft anchor)
    weights = np.ones_like(x, dtype=np.float64) * 0.3
    for k in knots:
        neighbourhood = 15
        start = max(0, k - neighbourhood)
        end = min(len(x), k + neighbourhood + 1)
        weights[start:end] = np.maximum(weights[start:end], 3.0)
    weights[knots] = 8.0  # knots themselves get highest weight

    # Normalize weights so sum ≈ len(x) (standard spline expectation)
    weights = weights / np.mean(weights)

    # Hampel outlier rejection before fitting
    y_clean = _hampel_filter(y)

    try:
        # s parameter: smoothing trade-off. Higher = smoother.
        s = base_smoothness * len(x) * float(np.var(y_clean)) * 0.5
        spline = UnivariateSpline(x, y_clean, w=weights, s=s, k=3)
        return spline(x)
    except Exception:
        return _fit_savgol(x, y, base_smoothness)


def _fit_quintic(y: np.ndarray, smoothness: float = 0.5) -> np.ndarray:
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
    method: Literal["savgol", "bspline", "loess", "quintic", "knot_constrained"],
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
    if method == "knot_constrained":
        return _fit_knot_constrained(x, y_clean, prominence=5.0, min_distance=30, base_smoothness=smoothness)
    return _fit_loess(x, y_clean, smoothness)

__all__ = [
    "_detect_knots",
    "_fit_bspline",
    "_fit_curve",
    "_fit_knot_constrained",
    "_fit_loess",
    "_fit_multiscale_savgol",
    "_fit_quintic",
    "_fit_savgol",
    "_hampel_filter",
]
