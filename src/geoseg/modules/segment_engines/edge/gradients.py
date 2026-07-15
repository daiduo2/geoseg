"""Gradient and edge-map helpers for edge-based engines."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.feature import canny
from skimage.filters import sobel
from skimage.morphology import closing, disk


def canny_edge_map(
    panel_lab: np.ndarray,
    canny_sigma: float = 1.0,
    canny_low: float = 0.05,
    canny_high: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect boundaries via Canny on the LAB L channel.

    Returns ``(gradient, edge_mask)`` to preserve the historical edge-guided
    engine contract.
    """
    l_norm = panel_lab[..., 0] / 100.0
    edge_mask = canny(
        l_norm,
        sigma=canny_sigma,
        low_threshold=canny_low,
        high_threshold=canny_high,
    )
    edge_mask = closing(edge_mask, footprint=disk(1))
    gradient = edge_mask.astype(np.float32)
    return gradient, edge_mask


def lab_sobel_edge_map(panel_lab: np.ndarray, sigma: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return LAB Sobel gradient magnitude and normalized edge map."""
    h, w = panel_lab.shape[:2]
    gradient = np.zeros((h, w), dtype=np.float32)
    for c in range(3):
        smoothed = ndimage.gaussian_filter(panel_lab[..., c], sigma=sigma)
        gradient += sobel(smoothed) ** 2
    gradient = np.sqrt(gradient)
    edge_map = gradient / (gradient.max() + 1e-9)
    return gradient, edge_map


__all__ = ["canny_edge_map", "lab_sobel_edge_map"]
