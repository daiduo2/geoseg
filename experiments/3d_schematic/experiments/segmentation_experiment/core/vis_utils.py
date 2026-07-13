"""Visualization utilities for segmentation experiments."""
from __future__ import annotations

import colorsys
import cv2
import numpy as np


def render_label_fill(labels: np.ndarray, overlay_label: int = -1) -> np.ndarray:
    """Render label fill with overlay in gray."""
    unique = sorted(np.unique(labels))
    h, w = labels.shape
    result = np.zeros((h, w, 3), dtype=np.uint8)
    colors = []
    for i, lbl in enumerate(unique):
        if lbl == overlay_label:
            colors.append([128, 128, 128])
        else:
            hue = (i * 0.618033988749895) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.75, 0.92)
            colors.append([int(c * 255) for c in rgb])
    for i, lbl in enumerate(unique):
        result[labels == lbl] = colors[i]
    return result


def draw_boundaries(image: np.ndarray, labels: np.ndarray,
                    color: tuple = (0, 0, 0), thickness: int = 2) -> np.ndarray:
    """Draw label boundaries on image."""
    result = image.copy()
    h, w = labels.shape
    for y in range(h - 1):
        for x in range(w):
            if labels[y, x] != labels[y + 1, x]:
                cv2.line(result, (x, y), (x, y + 1), color, thickness)
    for y in range(h):
        for x in range(w - 1):
            if labels[y, x] != labels[y, x + 1]:
                cv2.line(result, (x, y), (x + 1, y), color, thickness)
    return result


def create_comparison_grid(
    image: np.ndarray,
    results: list[tuple[str, np.ndarray]],
    crops: list[tuple[str, tuple[int, int, int, int]]] | None = None,
) -> np.ndarray:
    """Create a comparison grid: [original] + [result_1] + [result_2] + ...

    If crops provided, also create cropped comparison strips.
    Returns the full grid image.
    """
    strips = [image] + [render_label_fill(r[1]) for r in results]
    names = ["Original"] + [r[0] for r in results]

    # Full comparison
    grid = np.hstack(strips)
    h = grid.shape[0]
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, name in enumerate(names):
        x = i * image.shape[1] + 10
        cv2.putText(grid, name, (x, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(grid, name, (x, 30), font, 0.7, (0, 0, 0), 1)

    return grid


def create_crop_comparison(
    image: np.ndarray,
    results: list[tuple[str, np.ndarray]],
    crop_name: str,
    box: tuple[int, int, int, int],
) -> np.ndarray:
    """Create a single crop comparison strip."""
    y1, y2, x1, x2 = box
    strips = [image[y1:y2, x1:x2]] + [render_label_fill(r[1])[y1:y2, x1:x2] for r in results]
    comp = np.hstack(strips)
    # Label
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comp, crop_name, (10, 25), font, 0.6, (255, 255, 255), 2)
    cv2.putText(comp, crop_name, (10, 25), font, 0.6, (0, 0, 0), 1)
    return comp
