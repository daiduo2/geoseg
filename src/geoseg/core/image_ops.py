"""Stable image utility functions shared across geoseg modules."""

from __future__ import annotations

import colorsys

import numpy as np
from scipy import ndimage
from skimage import segmentation
from skimage.measure import label, regionprops


SATURATION_THRESHOLD = 80


def distinct_colors(
    n: int, saturation: float = 0.88, value: float = 0.95
) -> np.ndarray:
    """Generate perceptually distinct vivid colors using golden-ratio hues."""
    colors = np.zeros((max(n, 1), 3), dtype=np.uint8)
    golden = 0.618033988749895
    h = 0.08
    for i in range(n):
        h = (h + golden) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, saturation, value)
        colors[i] = [int(r * 255), int(g * 255), int(b * 255)]
    return colors


def saturation(rgb: np.ndarray) -> np.ndarray:
    """Per-pixel max-min over RGB. Input (H,W,3) uint8 -> (H,W) int."""
    return rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)


def saturation_ratio(
    panel_rgb: np.ndarray, threshold: int = SATURATION_THRESHOLD
) -> float:
    """Return the fraction of pixels above the saturation threshold."""
    s = saturation(panel_rgb)
    return float((s > threshold).mean())


def estimate_background_color(panel_rgb: np.ndarray) -> np.ndarray:
    """Estimate background color from image corners and center."""
    h, w = panel_rgb.shape[:2]
    samples = np.array(
        [
            panel_rgb[0, 0],
            panel_rgb[0, w - 1],
            panel_rgb[h - 1, 0],
            panel_rgb[h - 1, w - 1],
            panel_rgb[h // 2, w // 2],
        ],
        dtype=np.float32,
    )
    return np.median(samples, axis=0).astype(np.uint8)


def adaptive_blur(panel_rgb: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """Apply spatial Gaussian blur while preserving RGB channel relationships."""
    h, w = panel_rgb.shape[:2]
    if sigma is None:
        diag = (h * h + w * w) ** 0.5
        sigma = min(2.0, max(0.5, diag / 2000.0))

    blurred = ndimage.gaussian_filter(panel_rgb, sigma=(sigma, sigma, 0))
    return np.clip(blurred, 0, 255).astype(np.uint8)


def row_median_filter(panel_rgb: np.ndarray, size: int = 5) -> np.ndarray:
    """Apply a row-wise median filter for text-robust preprocessing."""
    return ndimage.median_filter(panel_rgb, size=(1, size, 1))


def estimate_noise_level(panel_rgb: np.ndarray) -> float:
    """Estimate perceptual noise level [0.0, 1.0] from edge density."""
    from skimage.filters import sobel

    gray = panel_rgb.mean(axis=2).astype(np.float32)
    edges = sobel(gray)
    edge_density = float((np.abs(edges) > 0.05).mean())
    return round(float(np.clip(edge_density * 1.5, 0.0, 1.0)), 4)


def detect_background_label(labels: np.ndarray) -> int | None:
    """Detect the label most likely to be background from image-edge coverage."""
    h, w = labels.shape
    edge_margin = max(3, min(h, w) // 50)
    edge_mask = np.zeros((h, w), dtype=bool)
    edge_mask[:edge_margin, :] = True
    edge_mask[-edge_margin:, :] = True
    edge_mask[:, :edge_margin] = True
    edge_mask[:, -edge_margin:] = True

    best_label = None
    best_score = 0.0
    for lbl in np.unique(labels):
        mask = labels == lbl
        edge_count = int(mask[edge_mask].sum())
        total_count = int(mask.sum())
        if total_count == 0:
            continue
        edge_ratio = edge_count / edge_mask.sum()
        area_ratio = total_count / (h * w)
        score = edge_ratio * area_ratio
        if score > best_score and edge_ratio > 0.25 and area_ratio > 0.08:
            best_score = score
            best_label = int(lbl)
    return best_label


def merge_small_regions(labels: np.ndarray, min_area_frac: float = 0.003) -> np.ndarray:
    """Merge tiny connected components into the largest neighboring label."""
    h, w = labels.shape
    out = labels.copy()
    min_area = max(30, int(h * w * min_area_frac))

    cc = label(out >= 0, connectivity=2)
    regions = regionprops(cc)

    for region in regions:
        if region.area >= min_area:
            continue
        comp_mask = cc == region.label
        dilated = ndimage.binary_dilation(comp_mask, structure=np.ones((3, 3), dtype=bool))
        neighbors = out[dilated & ~comp_mask]
        if len(neighbors) == 0:
            continue
        vals, counts = np.unique(neighbors, return_counts=True)
        out[comp_mask] = vals[counts.argmax()]
    return out


def create_overlay(
    panel_rgb: np.ndarray,
    labels: np.ndarray,
    seeds_rgb: np.ndarray | None,
    alpha: float = 0.65,
    boundary_mode: str = "thin",
    skip_background: bool = True,
    min_area_frac: float = 0.002,
    fill_mode: str = "blend",
    overlay_colors: np.ndarray | None = None,
) -> np.ndarray:
    """Create a segmentation overlay with distinct region colors."""
    if seeds_rgb is None:
        seeds_rgb = distinct_colors(int(labels.max()) + 1)

    cleaned_labels = merge_small_regions(labels, min_area_frac=min_area_frac)

    bg_label = None
    if skip_background:
        bg_label = detect_background_label(cleaned_labels)

    effective_alpha = {"blend": alpha, "solid": 0.85, "mask": 1.0}.get(fill_mode, alpha)
    is_mask = fill_mode == "mask"

    unique_labels = sorted(np.unique(cleaned_labels))
    n_labels = len(unique_labels)

    if overlay_colors is not None:
        base_colors = overlay_colors.astype(np.uint8)
    else:
        base_colors = distinct_colors(n_labels)

    color_map: dict[int, np.ndarray] = {}
    for i, lbl in enumerate(unique_labels):
        if i < len(base_colors):
            color_map[int(lbl)] = base_colors[i]
        else:
            color_map[int(lbl)] = np.array([128, 128, 128], dtype=np.uint8)

    if is_mask:
        overlay = np.full_like(panel_rgb, 32)
    else:
        overlay = panel_rgb.copy()

    for lbl in unique_labels:
        if bg_label is not None and lbl == bg_label:
            continue
        mask = cleaned_labels == lbl
        if not mask.any():
            continue
        color = color_map.get(int(lbl), np.array([128, 128, 128], dtype=np.uint8))
        if is_mask:
            overlay[mask] = color
        else:
            overlay[mask] = (
                overlay[mask] * (1 - effective_alpha) + color * effective_alpha
            ).astype(np.uint8)

    if boundary_mode != "none":
        boundaries = segmentation.find_boundaries(cleaned_labels, mode=boundary_mode)
        if bg_label is not None:
            boundaries &= cleaned_labels != bg_label
        overlay[boundaries] = [255, 255, 255]
    return overlay


__all__ = [
    "SATURATION_THRESHOLD",
    "adaptive_blur",
    "create_overlay",
    "detect_background_label",
    "distinct_colors",
    "estimate_background_color",
    "estimate_noise_level",
    "merge_small_regions",
    "row_median_filter",
    "saturation",
    "saturation_ratio",
]
