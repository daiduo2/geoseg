"""Coarse segmentation and topology helpers for horizon refinement."""

from __future__ import annotations

import numpy as np
from skimage.transform import resize

from geoseg.modules.segment_engines.internal.preprocess import row_median_filter


def _coarse_segment(
    panel_rgb: np.ndarray,
    n_layers: int,
    blur_sigma: float = 2.0,
    downsample_factor: float = 0.25,
) -> np.ndarray:
    """Phase A: coarse layer segmentation at low resolution."""
    h, w = panel_rgb.shape[:2]

    # Anisotropic row-wise median filter to suppress text/noise,
    # preserving vertical layer boundaries (see _shared.row_median_filter).
    size = max(3, int(blur_sigma * 2) | 1)  # ensure odd
    blurred = row_median_filter(panel_rgb, size=size)

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


def _separator_mask(labels: np.ndarray, threshold_frac: float = 0.02) -> np.ndarray:
    """Detect separator labels (thin lines with small area fraction).

    In editor topology, label 0 = boundary line (separator) with tiny area.
    In pipeline engine output (e.g. v4_kmeans), label 0 = first layer with
    substantial area.  We distinguish by area-fraction heuristic.

    Args:
        labels: Label map (H, W).
        threshold_frac: Area fraction below which label 0 is treated as
            a separator rather than a layer.

    Returns:
        Boolean mask of separator pixels (empty if label 0 is a real layer).
    """
    mask = labels == 0
    if not mask.any():
        return np.zeros(labels.shape, dtype=bool)
    return mask if float(mask.sum() / labels.size) < threshold_frac else np.zeros(labels.shape, dtype=bool)

__all__ = ["_coarse_segment", "_separator_mask"]
