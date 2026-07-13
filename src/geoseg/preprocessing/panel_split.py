"""Panel splitting for stacked tomography cross-sections.

Detects the five data panels as the largest coloured connected components,
excluding the topographic strip and colorbar.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage


def split_panels_colored_components(
    img_rgb: np.ndarray,
    margin: int = 5,
    min_area: int = 5000,
    n_panels: int = 5,
) -> list[tuple[int, int, int, int]]:
    """Return bounding boxes (x, y, w, h) for the largest coloured panels.

    Args:
        img_rgb: RGB uint8 array.
        margin: Pixels to expand around each detected component.
        min_area: Minimum component area to consider.
        n_panels: Number of panels to return.

    Returns:
        List of panel bboxes sorted top-to-bottom.
    """
    h, w = img_rgb.shape[:2]
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.int16)
    val = hsv[..., 2].astype(np.int16)

    grayish = (sat < 40) & (val > 60) & (val < 180)
    colored = (sat > 35) & (val > 50) & ~grayish

    labeled, n = ndimage.label(colored)
    stats: list[tuple[int, int, int, int, int]] = []
    for i in range(1, n + 1):
        comp = labeled == i
        ys, xs = np.where(comp)
        area = int(comp.sum())
        if area < min_area:
            continue
        stats.append((area, int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))

    stats.sort(reverse=True)
    top = stats[:n_panels]
    top.sort(key=lambda s: s[2])

    panels: list[tuple[int, int, int, int]] = []
    for _, x0, y0, x1, y1 in top:
        x0 = max(0, x0 - margin)
        y0 = max(0, y0 - margin)
        x1 = min(w - 1, x1 + margin)
        y1 = min(h - 1, y1 + margin)
        panels.append((x0, y0, x1 - x0 + 1, y1 - y0 + 1))

    return panels
