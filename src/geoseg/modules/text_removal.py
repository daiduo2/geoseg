"""Text removal for geophysics panel images.

Removes small annotations, axis labels, and velocity values that confuse
k-means / edge-guided segmentation. Uses a two-pass pipeline:

1. MSER + Laplacian detection followed by Telea inpainting.
2. Residual detection via region growing inside the initial mask, then
   large-kernel median replacement.

Designed to be called as a pre-processing step before segmentation engines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


DEFAULT_BRIGHTNESS_THRESH = 170
DEFAULT_MAX_STROKE_WIDTH = 0
DEFAULT_DILATE_ITER = 1
DEFAULT_INPAINT_RADIUS = 3
DEFAULT_MIN_AREA = 10
DEFAULT_MAX_AREA = 2000
DEFAULT_MAX_ASPECT = 20.0
DEFAULT_LAP_THRESHOLD = 15


def detect_text_mser(
    gray: np.ndarray,
    min_area: int = DEFAULT_MIN_AREA,
    max_area: int = DEFAULT_MAX_AREA,
    max_aspect: float = DEFAULT_MAX_ASPECT,
) -> np.ndarray:
    """Detect text-like regions with MSER on a grayscale image."""
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    mask = np.zeros(gray.shape, dtype=np.uint8)

    for region in regions:
        region = region.reshape(-1, 1, 2)
        area = cv2.contourArea(region)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(region)
        if w == 0 or h == 0:
            continue

        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > max_aspect:
            continue

        cv2.fillPoly(mask, [region], 255)

    return mask


def detect_text_laplacian(
    gray: np.ndarray,
    threshold: int = DEFAULT_LAP_THRESHOLD,
    max_area: int = DEFAULT_MAX_AREA,
) -> np.ndarray:
    """Detect text strokes via Laplacian edge response."""
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    _, mask = cv2.threshold(laplacian, threshold, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > max_area:
            mask[labels == i] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def _first_pass_mask(
    image_rgb: np.ndarray,
    brightness_thresh: int = DEFAULT_BRIGHTNESS_THRESH,
    max_stroke_width: int = DEFAULT_MAX_STROKE_WIDTH,
    dilate_iter: int = DEFAULT_DILATE_ITER,
    min_area: int = DEFAULT_MIN_AREA,
    max_area: int = DEFAULT_MAX_AREA,
    max_aspect: float = DEFAULT_MAX_ASPECT,
    lap_threshold: int = DEFAULT_LAP_THRESHOLD,
) -> np.ndarray:
    """Build the initial text mask."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    mask_orig = detect_text_mser(gray, min_area, max_area, max_aspect)
    mask_inv = detect_text_mser(255 - gray, min_area, max_area, max_aspect)
    mask_lap = detect_text_laplacian(gray, lap_threshold, max_area)

    combined = cv2.bitwise_or(mask_orig, mask_inv)
    combined = cv2.bitwise_or(combined, mask_lap)

    if brightness_thresh > 0:
        brightness_mask = (gray > brightness_thresh).astype(np.uint8) * 255
        combined = cv2.bitwise_and(combined, brightness_mask)

    if max_stroke_width > 0:
        dist = cv2.distanceTransform(combined, cv2.DIST_L2, 5)
        half = max(1, max_stroke_width // 2)
        stroke_mask = ((dist > 0) & (dist <= half)).astype(np.uint8) * 255
        combined = stroke_mask

    if dilate_iter > 0:
        kernel = np.ones((3, 3), np.uint8)
        combined = cv2.dilate(combined, kernel, iterations=dilate_iter)

    return combined


def _detect_residual_region_growing(
    first_result: np.ndarray,
    first_mask: np.ndarray,
    mser_min_area: int = 5,
    mser_max_area: int = 3000,
    mser_max_aspect: float = 30.0,
    lap_threshold: int = 10,
    lap_max_area: int = 3000,
    grow_threshold: int = 20,
    dilate_kernel_size: int = 3,
    dilate_iterations: int = 1,
) -> np.ndarray:
    """Detect text residuals after the first inpainting pass."""
    mask_bool = first_mask.astype(bool)
    gray_repaired = cv2.cvtColor(first_result, cv2.COLOR_RGB2GRAY)

    mser_mask = detect_text_mser(
        gray_repaired,
        min_area=mser_min_area,
        max_area=mser_max_area,
        max_aspect=mser_max_aspect,
    )
    lap_mask = detect_text_laplacian(
        gray_repaired, threshold=lap_threshold, max_area=lap_max_area
    )
    combined = cv2.bitwise_or(mser_mask, lap_mask)
    seeds = (combined > 0) & mask_bool

    if not np.any(seeds):
        return np.zeros_like(first_mask)

    residual_grown = seeds.copy()
    changed = True
    while changed:
        changed = False
        dilated = (
            cv2.dilate(
                residual_grown.astype(np.uint8) * 255,
                np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8),
                iterations=1,
            ).astype(bool)
        )
        candidates = dilated & mask_bool & (~residual_grown)
        if np.any(candidates):
            mean_bright = float(gray_repaired[residual_grown].mean())
            new_pixels = candidates & (
                np.abs(gray_repaired.astype(np.float32) - mean_bright) < grow_threshold
            )
            if np.any(new_pixels):
                residual_grown = residual_grown | new_pixels
                changed = True

    residual_mask = cv2.dilate(
        residual_grown.astype(np.uint8) * 255,
        np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8),
        iterations=dilate_iterations,
    )
    return residual_mask


def _repair_median_replace(
    image_rgb: np.ndarray, residual_mask: np.ndarray, ksize: int = 71
) -> np.ndarray:
    """Replace masked pixels with a large-kernel median blur."""
    mask_bool = residual_mask.astype(bool)
    result = image_rgb.copy().astype(np.float32)
    for ch in range(3):
        chan = result[:, :, ch]
        med = cv2.medianBlur(chan.astype(np.uint8), ksize).astype(np.float32)
        result[:, :, ch] = np.where(mask_bool, med, chan)
    return result.astype(np.uint8)


def remove_text(
    image_rgb: np.ndarray,
    *,
    brightness_thresh: int = DEFAULT_BRIGHTNESS_THRESH,
    max_stroke_width: int = DEFAULT_MAX_STROKE_WIDTH,
    dilate_iter: int = DEFAULT_DILATE_ITER,
    inpaint_radius: int = DEFAULT_INPAINT_RADIUS,
    min_area: int = DEFAULT_MIN_AREA,
    max_area: int = DEFAULT_MAX_AREA,
    max_aspect: float = DEFAULT_MAX_ASPECT,
    lap_threshold: int = DEFAULT_LAP_THRESHOLD,
    residual_mser_min_area: int = 5,
    residual_mser_max_area: int = 3000,
    residual_mser_max_aspect: float = 30.0,
    residual_lap_threshold: int = 10,
    residual_lap_max_area: int = 3000,
    residual_grow_threshold: int = 20,
    residual_dilate_kernel_size: int = 3,
    residual_dilate_iterations: int = 1,
    residual_median_ksize: int = 71,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove text annotations from a panel image.

    Args:
        image_rgb: RGB uint8 array.
        brightness_thresh: Keep MSER detections brighter than this value.
        max_stroke_width: Maximum stroke half-width to retain (distance transform).
        dilate_iter: Number of dilation iterations applied to the first-pass mask.
        inpaint_radius: Telea inpainting radius for the first pass.
        min_area: Minimum component area for MSER.
        max_area: Maximum component area for MSER / Laplacian.
        max_aspect: Maximum bounding-box aspect ratio for MSER.
        lap_threshold: Laplacian threshold for stroke detection.
        residual_*: Parameters for the second residual-repair pass.

    Returns:
        (cleaned_rgb, text_mask) where text_mask is uint8 with 255 = text.
    """
    first_mask = _first_pass_mask(
        image_rgb,
        brightness_thresh=brightness_thresh,
        max_stroke_width=max_stroke_width,
        dilate_iter=dilate_iter,
        min_area=min_area,
        max_area=max_area,
        max_aspect=max_aspect,
        lap_threshold=lap_threshold,
    )

    first_result = cv2.inpaint(
        image_rgb,
        first_mask,
        inpaintRadius=inpaint_radius,
        flags=cv2.INPAINT_TELEA,
    )

    residual_mask = _detect_residual_region_growing(
        first_result,
        first_mask,
        mser_min_area=residual_mser_min_area,
        mser_max_area=residual_mser_max_area,
        mser_max_aspect=residual_mser_max_aspect,
        lap_threshold=residual_lap_threshold,
        lap_max_area=residual_lap_max_area,
        grow_threshold=residual_grow_threshold,
        dilate_kernel_size=residual_dilate_kernel_size,
        dilate_iterations=residual_dilate_iterations,
    )

    if np.any(residual_mask):
        cleaned = _repair_median_replace(
            first_result, residual_mask, ksize=residual_median_ksize
        )
    else:
        cleaned = first_result

    text_mask = cv2.bitwise_or(first_mask, residual_mask)
    return cleaned, text_mask


def remove_text_from_path(
    input_path: str | Path,
    output_path: str | Path,
    mask_path: str | Path | None = None,
    **kwargs,
) -> np.ndarray:
    """Load an image, remove text, and save the cleaned result.

    Args:
        input_path: Path to the input RGB/RGBA image.
        output_path: Path to write the cleaned RGB image.
        mask_path: Optional path to write the uint8 text mask.
        **kwargs: Forwarded to remove_text().

    Returns:
        Cleaned RGB array.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    img = np.array(Image.open(input_path).convert("RGB"))
    cleaned, mask = remove_text(img, **kwargs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cleaned).save(output_path)

    if mask_path is not None:
        mask_path = Path(mask_path)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(mask_path)

    return cleaned
