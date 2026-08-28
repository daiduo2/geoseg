"""Panel cropping with VLM bbox + white-gutter refinement.

Phase 0 finding: T1 mean IoU 0.89 (4 images). VLM bbox is reliable but
typically off by 5-10 px on each side because the model includes/excludes
the axis-tick zone inconsistently. We refine by:

1. Pad the VLM bbox by ±10 px each side
2. Within that padded ROI, scan rows and columns for white-gutter
   (mean RGB > 245) to trim back to the data area
3. Final box ∩ image bounds

Public API:
    crop_panel(image_path, bbox, padding=10) -> (panel_rgb_ndarray, refined_bbox)
    refine_bbox(image, bbox, padding=10) -> refined_bbox
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


WHITE_RGB_THRESHOLD = 245
GUTTER_MIN_RUN = 3  # consecutive nearly-white rows/cols to count as gutter


def _load_rgb(image_path: str | Path) -> np.ndarray:
    with Image.open(image_path) as im:
        return np.array(im.convert("RGB"))


def _row_is_white(row: np.ndarray) -> bool:
    return bool(row.mean(axis=0).min() >= WHITE_RGB_THRESHOLD)


def _col_is_white(col: np.ndarray) -> bool:
    return bool(col.mean(axis=0).min() >= WHITE_RGB_THRESHOLD)


def refine_bbox(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    padding: int = 10,
) -> tuple[int, int, int, int]:
    """Refine a VLM-supplied bbox by trimming white gutter inside.

    bbox: (x1, y1, x2, y2) inclusive-exclusive.
    Returns refined (x1, y1, x2, y2). Never expands beyond `bbox ± padding`.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    # Pad and clamp
    px1 = max(0, x1 - padding)
    py1 = max(0, y1 - padding)
    px2 = min(w, x2 + padding)
    py2 = min(h, y2 + padding)
    roi = image[py1:py2, px1:px2]

    # Trim top
    top = 0
    for r in range(roi.shape[0]):
        if not _row_is_white(roi[r]):
            top = max(0, r - 1)
            break
    # Trim bottom
    bottom = roi.shape[0]
    for r in range(roi.shape[0] - 1, -1, -1):
        if not _row_is_white(roi[r]):
            bottom = min(roi.shape[0], r + 1)
            break
    # Trim left
    left = 0
    for c in range(roi.shape[1]):
        if not _col_is_white(roi[:, c]):
            left = max(0, c - 1)
            break
    # Trim right
    right = roi.shape[1]
    for c in range(roi.shape[1] - 1, -1, -1):
        if not _col_is_white(roi[:, c]):
            right = min(roi.shape[1], c + 1)
            break

    rx1 = px1 + left
    ry1 = py1 + top
    rx2 = px1 + right
    ry2 = py1 + bottom

    # Sanity: never collapse below original bbox by more than padding
    rx1 = min(rx1, x1)
    ry1 = min(ry1, y1)
    rx2 = max(rx2, x2)
    ry2 = max(ry2, y2)
    return rx1, ry1, rx2, ry2


def crop_panel(
    image_path: str | Path,
    bbox: tuple[int, int, int, int],
    padding: int = 10,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Load image, refine bbox, return cropped panel + final coords."""
    img = _load_rgb(image_path)
    refined = refine_bbox(img, bbox, padding)
    x1, y1, x2, y2 = refined
    return img[y1:y2, x1:x2].copy(), refined


def crop_colorbar(
    image_path: str | Path,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """Load image and crop the colorbar strip (no gutter refinement — bar is precise)."""
    img = _load_rgb(image_path)
    x1, y1, x2, y2 = bbox
    return img[y1:y2, x1:x2].copy()


__all__ = ["crop_panel", "crop_colorbar", "refine_bbox"]
