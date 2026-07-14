"""Preprocessing filters and image-quality heuristics for segmentation engines."""

from __future__ import annotations

import numpy as np

from geoseg.core.image_ops import (
    adaptive_blur as _adaptive_blur,
    estimate_noise_level as _estimate_noise_level,
    row_median_filter as _row_median_filter,
)


def adaptive_blur(panel_rgb: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """Apply Gaussian blur to suppress high-frequency noise before segmentation.

    Blur is applied only to the spatial axes; the color channel axis is left
    untouched so that RGB relationships are preserved.

    Args:
        panel_rgb: uint8 array (H, W, 3).
        sigma: Gaussian sigma in pixels. If None, computed from image diagonal
            as max(1.0, diag / 1000.0) so that larger images get slightly more
            blur. Capped at 3.0 to avoid erasing legitimate fault boundaries.

    Returns:
        Blurred uint8 array of the same shape.
    """
    return _adaptive_blur(panel_rgb, sigma=sigma)


def row_median_filter(panel_rgb: np.ndarray, size: int = 5) -> np.ndarray:
    """Anisotropic median filter for text-robust preprocessing.

    Applies a 1-D median filter along image rows (horizontal axis), which
    suppresses horizontal text/noise strokes while preserving vertical
    geological layer boundaries.  Exploits the strong horizontal stratification
    prior in geophysical images.

    For text-heavy panels this is preferred over ``adaptive_blur``, because
    Gaussian blur turns text into smeared dirty traces that still interfere
    with clustering, whereas median filtering removes the text impulse entirely.

    Args:
        panel_rgb: uint8 array (H, W, 3).
        size: Median filter window size along the row axis.  Default 5.

    Returns:
        Filtered uint8 array of the same shape.
    """
    return _row_median_filter(panel_rgb, size=size)


def estimate_noise_level(panel_rgb: np.ndarray) -> float:
    """Estimate perceptual noise level [0.0, 1.0] from edge density.

    Noisy images (text, grid lines, annotation markers) have high edge density.
    Smooth velocity-model layers have lower edge density.  This is a fast proxy
    that correlates well with the number of noise warnings observed in batch
    tests.
    """
    return _estimate_noise_level(panel_rgb)


__all__ = ['adaptive_blur', 'row_median_filter', 'estimate_noise_level']
