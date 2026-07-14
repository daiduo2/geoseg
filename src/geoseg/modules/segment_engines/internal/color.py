"""Color and background helpers for segmentation engines."""

from __future__ import annotations

import numpy as np
from skimage.color import rgb2lab

from geoseg.core.image_ops import (
    SATURATION_THRESHOLD,
    distinct_colors,
    estimate_background_color,
    saturation,
    saturation_ratio,
)


def _distinct_colors(n: int, saturation: float = 0.88, value: float = 0.95) -> np.ndarray:
    """Generate n perceptually distinct vivid colors using golden-ratio hue distribution."""
    return distinct_colors(n, saturation=saturation, value=value)


def _saturation(rgb: np.ndarray) -> np.ndarray:
    """Per-pixel max-min over RGB. Input (H,W,3) uint8 -> (H,W) int."""
    return saturation(rgb)


def _label_by_nearest(panel_lab: np.ndarray, palette_lab: np.ndarray) -> np.ndarray:
    """Label each pixel by index of nearest palette entry in LAB."""
    h, w, _ = panel_lab.shape
    flat = panel_lab.reshape(-1, 3)
    d2 = ((flat[:, None, :] - palette_lab[None, :, :]) ** 2).sum(axis=2)
    return d2.argmin(axis=1).reshape(h, w).astype(np.int32)


def _estimate_background_color(panel_rgb: np.ndarray) -> np.ndarray:
    """Estimate background colour from image corners and centre (median)."""
    return estimate_background_color(panel_rgb)


def _is_background_v2(
    rgb: np.ndarray, bg_rgb: np.ndarray, threshold: float = 60.0
) -> bool:
    """RGB Euclidean distance to the estimated background colour."""
    dist = float(np.linalg.norm(rgb.astype(np.float32) - bg_rgb.astype(np.float32)))
    return dist < threshold


__all__ = [
    "SATURATION_THRESHOLD",
    "_distinct_colors",
    "_saturation",
    "saturation_ratio",
    "_label_by_nearest",
    "_estimate_background_color",
    "_is_background_v2",
]
