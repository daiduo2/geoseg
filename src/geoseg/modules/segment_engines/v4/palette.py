"""Palette and colourbar helpers for v4 K-Means segmentation."""

from __future__ import annotations

import numpy as np


def _sample_colorbar_seeds(colorbar_rgb: np.ndarray, k: int) -> tuple[np.ndarray, list[str]]:
    """Sample k evenly-spaced RGBs along a colorbar strip."""
    h, w, _ = colorbar_rgb.shape
    if h >= w:
        ys = np.linspace(int(0.05 * h), int(0.95 * h) - 1, k).astype(int)
        cx = w // 2
        seeds = np.array([colorbar_rgb[y, cx] for y in ys])
    else:
        xs = np.linspace(int(0.05 * w), int(0.95 * w) - 1, k).astype(int)
        cy = h // 2
        seeds = np.array([colorbar_rgb[cy, x] for x in xs])
    names = _name_palette(seeds, k)
    return seeds.astype(np.uint8), names


def _name_palette(seeds_rgb: np.ndarray, k: int) -> list[str]:
    """Label k seed colors with conventional names."""
    standard = ["red", "orange", "yellow", "green", "blue", "purple"]
    if k > len(standard):
        standard = standard + [f"c{i}" for i in range(len(standard), k)]
    pool = standard[:k]
    rgb = seeds_rgb.astype(np.float32) / 255.0
    mx = rgb.max(axis=1)
    mn = rgb.min(axis=1)
    diff = mx - mn + 1e-9
    h = np.zeros(k)
    for i in range(k):
        r, g, b = rgb[i]
        if mx[i] == r:
            h[i] = (60 * ((g - b) / diff[i]) + 360) % 360
        elif mx[i] == g:
            h[i] = 60 * ((b - r) / diff[i]) + 120
        else:
            h[i] = 60 * ((r - g) / diff[i]) + 240
    order = np.argsort(h)
    names = [""] * k
    for rank, original_idx in enumerate(order):
        names[original_idx] = pool[rank]
    return names

__all__ = ["_name_palette", "_sample_colorbar_seeds"]
