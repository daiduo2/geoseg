"""Seeded region growing / watershed segmentation.

The caller provides one or more seed points (or seed masks). Each seed defines a
region that grows outward following low color-gradient paths. This is useful for
low-contrast panels such as panel_3, where a funnel-shaped plume has the same
hue as a gradient background but is separated by a subtle edge.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import ndimage
from skimage import filters
from skimage.color import rgb2lab
from skimage.segmentation import watershed

from geoseg.modules.segment_engines.internal.shared import _create_overlay


def _parse_seeds(
    seeds: Sequence[dict] | np.ndarray | None,
    h: int,
    w: int,
) -> np.ndarray:
    """Convert seeds into a (h, w) marker array.

    Supported formats:
    - list of dicts: [{"y": int, "x": int, "label": int}, ...]
    - binary mask array with seed pixels set to True (labelled sequentially)
    """
    if seeds is None:
        raise ValueError("seeded_region_grow requires seeds")

    if isinstance(seeds, np.ndarray):
        if seeds.dtype == bool:
            markers, _ = ndimage.label(seeds)
            return markers.astype(np.int32)
        return seeds.astype(np.int32)

    markers = np.zeros((h, w), dtype=np.int32)
    for s in seeds:
        y = int(s["y"])
        x = int(s["x"])
        lbl = int(s.get("label", 1))
        if 0 <= y < h and 0 <= x < w:
            markers[y, x] = lbl
    return markers


def _color_gradient(panel_rgb: np.ndarray, color_space: str, sigma: float = 2.0) -> np.ndarray:
    """Return a smoothed gradient magnitude of the color image."""
    if color_space.upper() == "LAB":
        features = rgb2lab(panel_rgb)
    else:
        features = panel_rgb.astype(np.float32)

    grad = np.zeros(panel_rgb.shape[:2], dtype=np.float32)
    for c in range(features.shape[2]):
        gy, gx = np.gradient(features[:, :, c])
        grad += gy ** 2 + gx ** 2
    grad = np.sqrt(grad)
    if sigma > 0:
        grad = filters.gaussian(grad, sigma=sigma)
    return grad


def segment(
    panel_rgb: np.ndarray,
    n_layers: int | None = None,
    reps: list[dict] | None = None,
    seeds: Sequence[dict] | np.ndarray | None = None,
    color_space: str = "LAB",
    gradient_sigma: float = 2.0,
    compactness: float = 0.0,
) -> dict:
    """Segment by marker-controlled watershed on a color-gradient cost map.

    Args:
        panel_rgb: RGB uint8 array.
        n_layers: Ignored; number of regions is determined by seeds.
        reps: Ignored; kept for protocol compatibility.
        seeds: Seed points/masks. Either a list of dicts with keys y, x, label
            or a binary/array mask.
        color_space: "LAB" or "RGB". LAB usually gives better lightness
            separation for low-contrast panels.
        gradient_sigma: Gaussian smoothing applied to the gradient magnitude
            before watershed. Larger values produce smoother boundaries.
        compactness: Watershed compactness (see skimage.segmentation.watershed).
            Values > 0 make regions more regularly shaped.

    Returns:
        dict with keys: labels, seeds, overlay, meta.
    """
    if panel_rgb.ndim != 3 or panel_rgb.shape[2] != 3:
        raise ValueError("panel_rgb must be an RGB image")

    h, w = panel_rgb.shape[:2]
    markers = _parse_seeds(seeds, h, w)

    if not markers.any():
        raise ValueError("no valid seeds provided")

    cost = _color_gradient(panel_rgb, color_space, sigma=gradient_sigma)
    labels = watershed(cost, markers=markers, compactness=compactness).astype(np.int32)

    # Compact label IDs starting from 1.
    present = sorted(set(labels.flatten()) - {0})
    renum = {old: new + 1 for new, old in enumerate(present)}
    clean = np.zeros_like(labels)
    for old, new in renum.items():
        clean[labels == old] = new
    labels = clean

    n_actual = int(labels.max())
    seeds_rgb = np.zeros((n_actual + 1, 3), dtype=np.uint8)
    for lbl in range(1, n_actual + 1):
        mask = labels == lbl
        if mask.any():
            seeds_rgb[lbl] = np.median(panel_rgb[mask], axis=0).astype(np.uint8)
        else:
            seeds_rgb[lbl] = 128

    overlay = _create_overlay(panel_rgb, labels, seeds_rgb)

    return {
        "labels": labels,
        "seeds": seeds_rgb.tolist(),
        "overlay": overlay,
        "meta": {
            "engine": "seeded_region_grow",
            "path": "seeded_watershed",
            "color_space": color_space.upper(),
            "n_seeds": len(present),
            "n_layers_actual": n_actual,
        },
    }


def segment_from_path(
    panel_path: str,
    seeds: Sequence[dict] | np.ndarray,
    color_space: str = "LAB",
) -> dict:
    """Convenience wrapper for file paths."""
    from PIL import Image

    img = np.array(Image.open(panel_path).convert("RGB"))
    return segment(img, seeds=seeds, color_space=color_space)
