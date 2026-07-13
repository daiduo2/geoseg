"""Colorbar region extraction.

This module intentionally stays minimal.  Colorbar location is a visual
judgment; callers that already know the ROI (e.g. from agent visual review)
should pass it directly.  The auto-detection below is only a coarse fallback
and must not contain brittle scoring heuristics.
"""

from __future__ import annotations

import numpy as np
from skimage.color import rgb2lab


def _strip_color_variation(strip: np.ndarray) -> float:
    """Return the mean AB standard deviation of a strip.

    A very low value means the strip is essentially uniform and is unlikely
    to be a real colorbar.  This is a coarse safety check, not a visual-review
    scoring rule.
    """
    if strip.size == 0:
        return 0.0
    strip_lab = rgb2lab(strip)
    ab = strip_lab[:, :, 1:].reshape(-1, 2)
    return float(ab.std(axis=0).mean())


def extract_colorbar(
    img_rgb: np.ndarray,
    colorbar_roi: tuple[int, int, int, int] | None = None,
    preferred_orientation: str = "auto",
) -> np.ndarray | None:
    """Extract the colorbar region from a figure image.

    Args:
        img_rgb: RGB uint8 array.
        colorbar_roi: Optional (x, y, w, h) bbox supplied by visual review.
            When provided, it is used verbatim.
        preferred_orientation: Ignored when ``colorbar_roi`` is provided.

    Returns:
        Colorbar strip as an RGB array, or None if not found.
    """
    result = extract_colorbar_bbox(img_rgb, colorbar_roi, preferred_orientation)
    if result is None:
        return None
    x, y, w, h, _ = result
    return img_rgb[y : y + h, x : x + w]


def extract_colorbar_bbox(
    img_rgb: np.ndarray,
    colorbar_roi: tuple[int, int, int, int] | None = None,
    preferred_orientation: str = "auto",
) -> tuple[int, int, int, int, str] | None:
    """Extract the colorbar bounding box from a figure image.

    Args:
        img_rgb: RGB uint8 array.
        colorbar_roi: Optional (x, y, w, h) bbox supplied by visual review.
            When provided, it is used verbatim.
        preferred_orientation: "horizontal", "vertical", or "auto".  Only used
            by the coarse fallback when no ROI is supplied.

    Returns:
        (x, y, w, h, orientation), or None if not found.
    """
    if colorbar_roi is not None:
        x, y, w, h = colorbar_roi
        if w >= h:
            return (x, y, w, h, "horizontal")
        return (x, y, w, h, "vertical")

    return _coarse_fallback(img_rgb, preferred_orientation)


def _coarse_fallback(
    img_rgb: np.ndarray,
    preferred_orientation: str = "auto",
) -> tuple[int, int, int, int, str] | None:
    """Coarse fallback: assume a colorbar lives in a standard margin strip.

    This is intentionally dumb.  It returns the whole margin strip so that the
    caller can sample colors along it.  If the strip is wrong, the caller should
    supply a ``colorbar_roi`` from visual review instead of adding more rules
    here.
    """
    h, w = img_rgb.shape[:2]
    strip_height = max(20, h // 12)
    strip_width = max(20, w // 12)

    if preferred_orientation in ("auto", "horizontal"):
        strip = img_rgb[max(0, h - strip_height) : h, :]
        if _strip_color_variation(strip) > 3.0:
            return (0, max(0, h - strip_height), w, strip_height, "horizontal")
    if preferred_orientation in ("auto", "vertical"):
        strip = img_rgb[:, max(0, w - strip_width) : w]
        if _strip_color_variation(strip) > 3.0:
            return (max(0, w - strip_width), 0, strip_width, h, "vertical")
    return None
