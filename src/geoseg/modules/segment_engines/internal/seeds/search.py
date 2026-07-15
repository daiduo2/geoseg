"""Seed pixel search helpers."""

from __future__ import annotations

import numpy as np
from skimage.measure import label, regionprops

from geoseg.modules.segment_engines.internal.color import (
    _estimate_background_color,
    _is_background_v2,
)


def _spiral_search(
    panel_rgb: np.ndarray,
    start_x: int,
    start_y: int,
    radius: int = 100,
    is_bg_func=None,
) -> tuple[int, int] | None:
    """Search outward in a square spiral for a non-background pixel."""
    h, w, _ = panel_rgb.shape
    _bg = is_bg_func if is_bg_func is not None else lambda c: _is_background_v2(c, _estimate_background_color(panel_rgb))
    if 0 <= start_x < w and 0 <= start_y < h:
        if not _bg(panel_rgb[start_y, start_x]):
            return start_x, start_y

    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    x, y = start_x, start_y
    step_len = 1
    dir_idx = 0

    while abs(x - start_x) <= radius and abs(y - start_y) <= radius:
        dx, dy = dirs[dir_idx]
        for _ in range(step_len):
            x += dx
            y += dy
            if 0 <= x < w and 0 <= y < h:
                if not _bg(panel_rgb[y, x]):
                    return x, y
            if abs(x - start_x) > radius or abs(y - start_y) > radius:
                return None
        dir_idx = (dir_idx + 1) % 4
        if dir_idx % 2 == 0:
            step_len += 1
    return None


def _find_pixel_for_color(
    panel_rgb: np.ndarray,
    target_rgb: np.ndarray,
    bg_rgb: np.ndarray,
    color_tol: float = 35.0,
    bg_tol: float = 40.0,
) -> tuple[int, int] | None:
    """Find the largest connected component of pixels matching target_rgb and not background."""
    diff = np.linalg.norm(
        panel_rgb.astype(np.float32) - target_rgb.astype(np.float32), axis=2
    )
    mask = diff <= color_tol
    bg_diff = np.linalg.norm(
        panel_rgb.astype(np.float32) - bg_rgb.astype(np.float32), axis=2
    )
    mask &= bg_diff > bg_tol

    if not mask.any():
        return None

    cc = label(mask, connectivity=2)
    regions = regionprops(cc)
    if not regions:
        return None
    largest = max(regions, key=lambda r: r.area)
    cy, cx = int(largest.centroid[0]), int(largest.centroid[1])
    return cx, cy

__all__ = ["_find_pixel_for_color", "_spiral_search"]
