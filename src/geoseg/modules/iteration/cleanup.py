"""Pixel-level text/background cleanup for the iteration workflow.

Whole-label removal handles labels that are entirely text, but white/black
strokes embedded inside a geological label still corrupt its colour audit.
This step removes those pixels: large border-connected white/black areas become
background 0, while small text/line fragments inside geology are filled from
the nearest remaining label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class TextPixelReport:
    text_fraction: float
    background_pixels: int
    filled_pixels: int


def text_pixel_mask(
    image: np.ndarray,
    *,
    white_threshold: int = 240,
    black_threshold: int = 30,
) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    rgb = array[..., :3]
    white = np.min(rgb, axis=2) >= white_threshold
    black = np.max(rgb, axis=2) <= black_threshold
    return white | black


def clean_text_pixels(
    image: np.ndarray,
    labels: np.ndarray,
    *,
    white_threshold: int = 240,
    black_threshold: int = 30,
    border_background_fraction: float = 0.02,
) -> tuple[np.ndarray, TextPixelReport]:
    """Remove text/background pixels and fill small internal fragments."""
    labels = np.asarray(labels, dtype=np.int32).copy()
    mask = text_pixel_mask(
        image, white_threshold=white_threshold, black_threshold=black_threshold
    )
    if not mask.any():
        return labels, TextPixelReport(0.0, 0, 0)

    components, n_components = ndimage.label(mask)
    h, w = labels.shape
    total = h * w
    border_ids = set()
    for component_id in range(1, n_components + 1):
        rows, cols = np.nonzero(components == component_id)
        touches_border = (
            rows.min() == 0
            or cols.min() == 0
            or rows.max() == h - 1
            or cols.max() == w - 1
        )
        if not touches_border:
            continue
        area = len(rows)
        if area >= border_background_fraction * total:
            border_ids.add(component_id)

    background_mask = np.isin(components, list(border_ids))
    internal_mask = mask & ~background_mask
    labels[background_mask] = 0

    valid = (~mask) & (labels > 0)
    filled = 0
    if internal_mask.any() and valid.any():
        _, indices = ndimage.distance_transform_edt(~valid, return_indices=True)
        rows, cols = np.nonzero(internal_mask)
        labels[rows, cols] = labels[indices[0][rows, cols], indices[1][rows, cols]]
        filled = int(internal_mask.sum())

    return labels, TextPixelReport(
        text_fraction=float(mask.mean()),
        background_pixels=int(background_mask.sum()),
        filled_pixels=filled,
    )


__all__ = ["TextPixelReport", "clean_text_pixels", "text_pixel_mask"]
