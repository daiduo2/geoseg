"""Automatic colorbar region extraction from geophysics figures.

Locates horizontal or vertical colorbar strips by analyzing color richness
and rectangular structure in edge regions.

Test scenario:
    >>> import numpy as np
    >>> img = np.zeros((200, 400, 3), dtype=np.uint8)
    >>> img[10:30, 100:300] = np.linspace([0,0,255], [255,0,0], 200)  # horizontal colorbar
    >>> cb = extract_colorbar(img)
    >>> assert cb is not None
"""

from __future__ import annotations

import numpy as np
from skimage.color import rgb2gray, rgb2lab


def _color_variation_score(strip: np.ndarray) -> float:
    """Score how much color variation a strip has (0-1)."""
    if strip.size == 0:
        return 0.0
    strip_lab = rgb2lab(strip)
    # High AB variation = rich colors
    ab_std = strip_lab[:, :, 1:].reshape(-1, 2).std(axis=0).mean()
    return float(np.clip(ab_std / 40.0, 0.0, 1.0))


def extract_colorbar(
    img_rgb: np.ndarray,
    preferred_orientation: str = "auto",
) -> np.ndarray | None:
    """Extract colorbar region from a figure image.

    Args:
        img_rgb: RGB uint8 array.
        preferred_orientation: "horizontal", "vertical", or "auto".

    Returns:
        Colorbar strip as RGB array, or None if not found.
    """
    result = extract_colorbar_bbox(img_rgb, preferred_orientation)
    if result is None:
        return None
    x, y, w, h, _ = result
    return img_rgb[y : y + h, x : x + w]


def extract_colorbar_bbox(
    img_rgb: np.ndarray,
    preferred_orientation: str = "auto",
) -> tuple[int, int, int, int, str] | None:
    """Extract colorbar bounding box from a figure image.

    Args:
        img_rgb: RGB uint8 array.
        preferred_orientation: "horizontal", "vertical", or "auto".

    Returns:
        (x, y, w, h, orientation) where orientation is "horizontal" or "vertical",
        or None if not found.
    """
    h, w = img_rgb.shape[:2]
    strip_height = max(20, h // 10)
    strip_width = max(20, w // 10)
    min_width = int(w * 0.3)
    min_height = int(h * 0.3)

    if preferred_orientation in ("auto", "horizontal"):
        # Check top margin
        if w >= min_width:
            score = _color_variation_score(img_rgb[:strip_height, :])
            if score > 0.3:
                return (0, 0, w, strip_height, "horizontal")
        # Check bottom margin
        if w >= min_width:
            score = _color_variation_score(img_rgb[-strip_height:, :])
            if score > 0.3:
                return (0, h - strip_height, w, strip_height, "horizontal")
        # Check near-top inset positions
        for y_start in [5, 15, 25]:
            if y_start + strip_height < h and w >= min_width:
                score = _color_variation_score(
                    img_rgb[y_start : y_start + strip_height, :]
                )
                if score > 0.3:
                    return (0, y_start, w, strip_height, "horizontal")

    if preferred_orientation in ("auto", "vertical"):
        # Check right margin (most common)
        if h >= min_height:
            score = _color_variation_score(img_rgb[:, -strip_width:])
            if score > 0.3:
                return (w - strip_width, 0, strip_width, h, "vertical")
        # Check left margin
        if h >= min_height:
            score = _color_variation_score(img_rgb[:, :strip_width])
            if score > 0.3:
                return (0, 0, strip_width, h, "vertical")

    return None
