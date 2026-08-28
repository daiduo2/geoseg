"""Artifact detectors for tomography panels.

Red fault traces, black earthquake hypocenter crosses, and text protection.
All thresholds are passed in via ``params``; no magic numbers are hard-coded.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from skimage.filters import frangi


def _is_reddish(
    rgb: np.ndarray,
    red_hue_range: tuple[int, int] = (165, 22),
    min_saturation: int = 50,
    min_value: int = 40,
    max_value: int = 245,
    min_redness: int = 18,
    red_ratio: float = 1.35,
) -> np.ndarray:
    """Return a boolean mask of red-dominant pixels."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype(np.int16)
    sat = hsv[..., 1].astype(np.int16)
    val = hsv[..., 2].astype(np.int16)

    if red_hue_range[0] <= red_hue_range[1]:
        red_hue = (hue >= red_hue_range[0]) & (hue <= red_hue_range[1])
    else:
        red_hue = (hue <= red_hue_range[1]) | (hue >= red_hue_range[0])

    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    redness = r - np.maximum(g, b)

    return (
        red_hue
        & (sat > min_saturation)
        & (val > min_value)
        & (val < max_value)
        & (redness > min_redness)
        & (r > red_ratio * g)
        & (r > red_ratio * b)
    )


def _local_red_spike(
    rgb: np.ndarray,
    median_ksize: int = 15,
    min_red_diff: int = 12,
    max_gb_diff: int = 12,
) -> np.ndarray:
    """Pixels where the red channel is locally elevated over the blurred bg."""
    bg = cv2.medianBlur(rgb.astype(np.uint8), median_ksize).astype(np.float32)
    diff = rgb.astype(np.float32) - bg
    return (
        (diff[..., 0] > min_red_diff)
        & (diff[..., 1] < max_gb_diff)
        & (diff[..., 2] < max_gb_diff)
    )


def detect_red_lines(
    panel_rgb: np.ndarray,
    frangi_sigmas: list[float] | None = None,
    frangi_threshold: float = 0.03,
    min_area: int = 50,
    max_width_frac: float = 0.55,
    min_elongation: float = 2.0,
    angle_ranges: list[tuple[int, int]] | None = None,
    dilation_kernel_size: int = 9,
    dilation_iters: int = 1,
) -> np.ndarray:
    """Detect thin red/orange fault lines with Frangi vesselness.

    Args:
        panel_rgb: RGB uint8 array.
        frangi_sigmas: Sigma values for Frangi filter.
        frangi_threshold: Response threshold.
        min_area: Minimum component area.
        max_width_frac: Drop components wider than this fraction of panel width.
        min_elongation: Minimum PCA elongation (s0 / s1).
        angle_ranges: Accepted ridge angle ranges in degrees.
        dilation_kernel_size: Size of the dilation kernel used to widen mask.
        dilation_iters: Dilation iterations.

    Returns:
        Boolean mask of detected red lines.
    """
    if frangi_sigmas is None:
        frangi_sigmas = [1, 2, 3, 4]
    if angle_ranges is None:
        angle_ranges = [(15, 75), (105, 165)]

    h, w = panel_rgb.shape[:2]
    r = panel_rgb[..., 0].astype(np.float32)
    g = panel_rgb[..., 1].astype(np.float32)
    b = panel_rgb[..., 2].astype(np.float32)
    redness = np.clip(r - np.maximum(g, b), 0, 255)
    spike = _local_red_spike(panel_rgb).astype(np.float32) * 255
    channel = redness * 0.5 + spike * 0.5

    response = frangi(channel, sigmas=frangi_sigmas, black_ridges=False)
    ridge = response > frangi_threshold

    labeled, n = ndimage.label(ridge)
    filtered = np.zeros_like(ridge)

    for i in range(1, n + 1):
        comp = labeled == i
        ys, xs = np.where(comp)
        area = int(comp.sum())
        if area < min_area:
            continue

        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        bb_h = y1 - y0 + 1
        bb_w = x1 - x0 + 1
        if bb_w > 0 and bb_w > max_width_frac * w:
            continue

        # Drop flat horizontal bands.
        if bb_h <= 6 and bb_w > 0.25 * w:
            continue
        if y0 <= 2 and bb_w > 0.20 * w:
            continue
        if y0 < 0.03 * h and bb_h < 0.15 * h and bb_w > 0.4 * w:
            continue

        coords = np.column_stack((xs, ys)).astype(np.float64)
        coords -= coords.mean(axis=0)
        if len(coords) < 2:
            continue
        _, s, v = np.linalg.svd(coords, full_matrices=False)
        if len(s) < 2:
            continue
        elongation = s[0] / max(s[1], 1e-6)
        if elongation < min_elongation:
            continue

        dx, dy = v[0]
        angle = np.degrees(np.arctan2(abs(dy), abs(dx)))
        if any(lo < angle < hi for lo, hi in angle_ranges):
            filtered |= comp

    if not filtered.any():
        return filtered

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size))
    mask_uint8 = filtered.astype(np.uint8) * 255
    return cv2.dilate(mask_uint8, kernel, iterations=dilation_iters) > 0


def detect_red_boundaries(
    panel_rgb: np.ndarray,
    *,
    min_area: int = 40,
    min_length_frac: float = 0.04,
    min_elongation: float = 1.8,
    closing_radius: int = 2,
) -> np.ndarray:
    """Detect red structural boundaries, including thick and solid traces.

    ``detect_red_lines`` is deliberately conservative because its original
    use is artifact absorption. Structural diagrams need the complementary
    behaviour: retain long red strokes while rejecting compact symbols and
    isolated red marks. This detector starts from the existing red-colour
    predicate and filters connected components by spatial extent and PCA
    elongation.
    """
    if panel_rgb.ndim != 3 or panel_rgb.shape[2] != 3:
        raise ValueError("panel_rgb must have shape (H, W, 3)")

    candidate = _is_reddish(panel_rgb)
    if closing_radius > 0:
        size = closing_radius * 2 + 1
        structure = np.ones((size, size), dtype=bool)
        candidate = ndimage.binary_closing(candidate, structure=structure)

    h, w = candidate.shape
    min_length = max(5.0, min(h, w) * min_length_frac)
    components, count = ndimage.label(candidate, structure=np.ones((3, 3)))
    result = np.zeros_like(candidate)

    for component_id in range(1, count + 1):
        ys, xs = np.where(components == component_id)
        if xs.size < min_area:
            continue

        bbox_length = float(max(np.ptp(xs) + 1, np.ptp(ys) + 1))
        if bbox_length < min_length:
            continue

        coords = np.column_stack((xs, ys)).astype(np.float64)
        coords -= coords.mean(axis=0)
        if len(coords) < 2:
            continue
        _, singular_values, _ = np.linalg.svd(coords, full_matrices=False)
        if len(singular_values) < 2:
            continue
        elongation = singular_values[0] / max(singular_values[1], 1e-6)
        if elongation < min_elongation:
            continue

        result[ys, xs] = True

    return result


def detect_text(
    panel_rgb: np.ndarray,
    median_ksize: int = 7,
    min_diff: int = 8,
    max_gray: int = 120,
    min_area: int = 200,
    min_width: int = 25,
    min_aspect: float = 4.0,
) -> np.ndarray:
    """Estimate a text mask to protect labels during inpainting."""
    gray = cv2.cvtColor(panel_rgb, cv2.COLOR_RGB2GRAY)
    bg = cv2.medianBlur(gray, median_ksize)
    diff = bg.astype(np.int16) - gray.astype(np.int16)
    dark = (diff > min_diff) & (gray < max_gray)

    labeled, n = ndimage.label(dark)
    text_mask = np.zeros_like(dark)
    for i in range(1, n + 1):
        comp = labeled == i
        ys, xs = np.where(comp)
        area = int(comp.sum())
        bb_h = int(ys.max() - ys.min() + 1)
        bb_w = int(xs.max() - xs.min() + 1)
        aspect = max(bb_h, bb_w) / max(min(bb_h, bb_w), 1)
        if area > min_area and (bb_w > min_width or aspect > min_aspect):
            text_mask |= comp
    return text_mask


def detect_black_crosses(
    panel_rgb: np.ndarray,
    median_ksize: int = 7,
    min_diff: int = 8,
    max_gray: int = 100,
    cross_area_range: tuple[int, int] = (4, 120),
    cross_aspect_range: tuple[float, float] = (0.3, 3.0),
    cross_max_size: int = 16,
    cluster_area_range: tuple[int, int] = (120, 1000),
    cluster_min_compactness: float = 0.35,
    cluster_max_aspect: float = 3.0,
    cluster_max_size: int = 50,
) -> np.ndarray:
    """Detect small black cross-shaped earthquake hypocenter markers."""
    h, w = panel_rgb.shape[:2]
    gray = cv2.cvtColor(panel_rgb, cv2.COLOR_RGB2GRAY)
    bg = cv2.medianBlur(gray, median_ksize)
    diff = bg.astype(np.int16) - gray.astype(np.int16)

    dark = (diff > min_diff) & (gray < max_gray)
    text_mask = detect_text(panel_rgb)

    candidate_dark = dark & ~text_mask
    labeled, n = ndimage.label(candidate_dark)
    filtered = np.zeros_like(dark)
    for i in range(1, n + 1):
        comp = labeled == i
        ys, xs = np.where(comp)
        area = int(comp.sum())
        bb_h = int(ys.max() - ys.min() + 1)
        bb_w = int(xs.max() - xs.min() + 1)
        aspect = max(bb_h, bb_w) / max(min(bb_h, bb_w), 1)
        compact = area / max(bb_h * bb_w, 1)

        is_cross = (
            cross_area_range[0] <= area <= cross_area_range[1]
            and cross_aspect_range[0] <= aspect <= cross_aspect_range[1]
            and bb_h <= cross_max_size
            and bb_w <= cross_max_size
        )
        is_cluster = (
            cluster_area_range[0] < area <= cluster_area_range[1]
            and compact > cluster_min_compactness
            and aspect <= cluster_max_aspect
            and bb_h <= cluster_max_size
            and bb_w <= cluster_max_size
        )
        if is_cross or is_cluster:
            filtered |= comp
    return filtered
