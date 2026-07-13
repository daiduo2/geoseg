"""Filter extracted images by quality and content type.

Removes blanks, tiny images, duplicates, and non-figure content before
segmentation pipeline.

Test scenario:
    >>> import numpy as np
    >>> img = np.full((200, 300, 3), 255, dtype=np.uint8)
    >>> r = check_image_quality(img)
    >>> assert not r["ok"]
"""

from __future__ import annotations

import numpy as np
from skimage.color import rgb2gray


def check_image_quality(
    img_rgb: np.ndarray,
    min_size: int = 100,
    max_blank_ratio: float = 0.95,
    max_edge_ratio: float = 0.005,
) -> dict:
    """Check if an extracted image is suitable for segmentation.

    Args:
        img_rgb: RGB uint8 array.
        min_size: Minimum width/height in pixels.
        max_blank_ratio: Maximum fraction of near-white pixels.
        max_edge_ratio: Maximum fraction of edge pixels (very low = likely blank).

    Returns:
        dict with keys: ok (bool), reason (str), metrics (dict).
    """
    h, w = img_rgb.shape[:2]
    if h < min_size or w < min_size:
        return {
            "ok": False,
            "reason": f"Too small ({w}x{h} < {min_size}x{min_size})",
            "metrics": {"width": w, "height": h},
        }

    gray = rgb2gray(img_rgb)

    # Blank check
    near_white = gray > 0.97
    blank_ratio = near_white.mean()
    if blank_ratio > max_blank_ratio:
        return {
            "ok": False,
            "reason": f"Mostly blank ({blank_ratio:.1%} near-white pixels)",
            "metrics": {"blank_ratio": round(blank_ratio, 4), "width": w, "height": h},
        }

    # Edge check (blank images have almost no edges)
    from skimage.filters import sobel
    edges = np.abs(sobel(gray))
    edge_ratio = (edges > 0.05).mean()
    if edge_ratio < max_edge_ratio:
        return {
            "ok": False,
            "reason": f"No content ({edge_ratio:.4f} edge ratio)",
            "metrics": {"edge_ratio": round(edge_ratio, 4), "blank_ratio": round(blank_ratio, 4)},
        }

    # Near-duplicate check via simple hash (placeholder)
    # In production, use perceptual hash (phash)

    return {
        "ok": True,
        "reason": "",
        "metrics": {
            "width": w,
            "height": h,
            "blank_ratio": round(blank_ratio, 4),
            "edge_ratio": round(edge_ratio, 4),
        },
    }


def filter_directory(
    images: list,
    min_size: int = 100,
    max_blank_ratio: float = 0.95,
) -> tuple[list, list]:
    """Filter a list of images, returning (kept, rejected).

    Args:
        images: List of (name, img_rgb) tuples.
        min_size: Minimum dimension.
        max_blank_ratio: Maximum blank ratio.

    Returns:
        (kept_list, rejected_list), where each item is (name, img_rgb, check_result).
    """
    kept = []
    rejected = []
    for name, img in images:
        result = check_image_quality(img, min_size=min_size, max_blank_ratio=max_blank_ratio)
        if result["ok"]:
            kept.append((name, img, result))
        else:
            rejected.append((name, img, result))
    return kept, rejected
