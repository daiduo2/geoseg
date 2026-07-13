"""Two-pass text removal for geological schematic images.

Best configuration (validated on panels 1/2/3):
  Stage 1: MSER (brightness-filtered) + Laplacian edges (unfiltered)
           + dilate=1 + Gaussian mask expansion (sigma=7, thresh=0.3)
           + Telea inpaint(r=7)
  Stage 2: Re-detect residual text on repaired image + region growing
           + Telea inpaint(r=5) for cleanup

Key insight: Laplacian edge detection bypasses the brightness filter,
catching low-brightness anti-aliased text edges that MSER misses.
Gaussian expansion then smoothly grows the mask to cover edge pixels
without the destructive over-expansion caused by lowering the brightness
threshold globally. Large-radius Telea inpaint (r=7) provides sufficient
context for seamless texture restoration.
"""
from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Stage 1: Mask generation
# ---------------------------------------------------------------------------

def _detect_text_mser(gray: np.ndarray, min_area: int = 10,
                      max_area: int = 2000, max_aspect: float = 20) -> np.ndarray:
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
        if max(w, h) / min(w, h) > max_aspect:
            continue
        cv2.fillPoly(mask, [region], 255)
    return mask


def _detect_text_laplacian(gray: np.ndarray, threshold: int = 15,
                           max_area: int = 2000) -> np.ndarray:
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


def generate_text_mask(
    image_rgb: np.ndarray,
    brightness_thresh: int = 170,
    dilate_iter: int = 1,
    mser_min_area: int = 10,
    mser_max_area: int = 2000,
    mser_max_aspect: float = 20,
    lap_threshold: int = 15,
) -> np.ndarray:
    """Generate text mask using MSER + Laplacian + brightness filter."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    mask_orig = _detect_text_mser(gray, mser_min_area, mser_max_area, mser_max_aspect)
    mask_inv = _detect_text_mser(255 - gray, mser_min_area, mser_max_area, mser_max_aspect)
    mask_lap = _detect_text_laplacian(gray, lap_threshold, mser_max_area)

    # MSER results are filtered by brightness to avoid false positives on
    # dark geological textures that happen to have blob-like shapes.
    combined = cv2.bitwise_or(mask_orig, mask_inv)
    if brightness_thresh > 0:
        brightness_mask = (gray > brightness_thresh).astype(np.uint8) * 255
        combined = cv2.bitwise_and(combined, brightness_mask)

    # Laplacian edges bypass brightness filter — this is critical for catching
    # low-brightness anti-aliased text edges that MSER misses.
    combined = cv2.bitwise_or(combined, mask_lap)

    if dilate_iter > 0:
        kernel = np.ones((3, 3), np.uint8)
        combined = cv2.dilate(combined, kernel, iterations=dilate_iter)

    return combined


# ---------------------------------------------------------------------------
# Stage 1 repair: conservative inpaint
# ---------------------------------------------------------------------------

def expand_mask_gaussian(mask: np.ndarray, sigma: float = 7.0,
                         threshold: float = 0.3) -> np.ndarray:
    """Smoothly expand a binary mask via Gaussian blur + re-thresholding.

    Unlike morphological dilation, this produces a soft edge expansion that
    covers low-contrast text pixels while avoiding hard geometric artifacts.
    """
    ksize = int(sigma * 3) * 2 + 1
    mask_f = mask.astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(mask_f, (ksize, ksize), sigmaX=sigma)
    return (blurred > threshold).astype(np.uint8) * 255


def inpaint_masked(image_rgb: np.ndarray, mask: np.ndarray,
                   radius: int = 3) -> np.ndarray:
    """Inpaint masked regions using TELEA."""
    return cv2.inpaint(image_rgb, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)


# ---------------------------------------------------------------------------
# Stage 2: Residual detection + aggressive repair
# ---------------------------------------------------------------------------

def detect_residual_mask(
    first_result: np.ndarray,
    first_mask: np.ndarray,
    mser_min_area: int = 5,
    mser_max_area: int = 3000,
    mser_max_aspect: float = 30,
    lap_threshold: int = 10,
    grow_threshold: float = 20.0,
    dilate_kernel_size: int = 3,
    dilate_iterations: int = 1,
) -> np.ndarray:
    """Detect text residual after first-pass repair.

    Strategy: re-run text detection on repaired image, intersect with first
    mask to get seeds, then region-grow within masked areas.
    """
    mask_bool = first_mask.astype(bool)
    gray = cv2.cvtColor(first_result, cv2.COLOR_RGB2GRAY)

    mser = _detect_text_mser(gray, mser_min_area, mser_max_area, mser_max_aspect)
    lap = _detect_text_laplacian(gray, lap_threshold, mser_max_area)
    combined = cv2.bitwise_or(mser, lap)
    seeds = (combined > 0) & mask_bool

    if not np.any(seeds):
        return np.zeros_like(first_mask)

    grown = seeds.copy()
    changed = True
    while changed:
        changed = False
        dilated = cv2.dilate(
            grown.astype(np.uint8) * 255, np.ones((3, 3), np.uint8), iterations=1
        ).astype(bool)
        candidates = dilated & mask_bool & (~grown)
        if np.any(candidates):
            mean_bright = gray[grown].mean()
            new_pixels = candidates & (
                np.abs(gray.astype(float) - mean_bright) < grow_threshold
            )
            if np.any(new_pixels):
                grown = grown | new_pixels
                changed = True

    residual = cv2.dilate(
        grown.astype(np.uint8) * 255,
        np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8),
        iterations=dilate_iterations,
    )
    return residual


def median_replace(image_rgb: np.ndarray, mask: np.ndarray,
                   ksize: int = 71) -> np.ndarray:
    """Replace masked pixels with large-kernel median blur values."""
    mask_bool = mask.astype(bool)
    result = image_rgb.copy().astype(np.float32)
    for ch in range(3):
        chan = result[:, :, ch]
        med = cv2.medianBlur(chan.astype(np.uint8), ksize).astype(np.float32)
        result[:, :, ch] = np.where(mask_bool, med, chan)
    return result.astype(np.uint8)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def remove_text(
    image_rgb: np.ndarray,
    *,
    brightness_thresh: int = 160,
    dilate_iter: int = 1,
    expand_sigma: float = 7.0,
    expand_threshold: float = 0.3,
    inpaint_radius: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Single-pass text removal. Returns (result, mask)."""
    mask = generate_text_mask(image_rgb, brightness_thresh, dilate_iter)
    mask = expand_mask_gaussian(mask, expand_sigma, expand_threshold)
    result = inpaint_masked(image_rgb, mask, inpaint_radius)
    return result, mask


def remove_text_two_pass(
    image_rgb: np.ndarray,
    *,
    brightness_thresh: int = 160,
    dilate_iter: int = 1,
    expand_sigma: float = 7.0,
    expand_threshold: float = 0.3,
    inpaint_radius: int = 7,
    residual_grow_threshold: float = 20.0,
    repair_radius: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two-pass text removal. Returns (final, first_result, first_mask, residual_mask).

    Best configuration validated on 3-panel geological schematic:
      - Stage 1: brightness_thresh=160, dilate_iter=1, expand_sigma=7,
                 inpaint_radius=7
      - Stage 2: residual_grow_threshold=20, repair_radius=5 (Telea)
    """
    first_mask = generate_text_mask(image_rgb, brightness_thresh, dilate_iter)
    first_mask = expand_mask_gaussian(first_mask, expand_sigma, expand_threshold)
    first_result = inpaint_masked(image_rgb, first_mask, inpaint_radius)

    residual_mask = detect_residual_mask(
        first_result, first_mask, grow_threshold=residual_grow_threshold
    )

    if np.any(residual_mask):
        final_result = cv2.inpaint(
            first_result, residual_mask, repair_radius, cv2.INPAINT_TELEA
        )
    else:
        final_result = first_result

    return final_result, first_result, first_mask, residual_mask
