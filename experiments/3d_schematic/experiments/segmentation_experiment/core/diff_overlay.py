"""Diff-overlay segmentation pipeline for 3D schematic panels."""
from __future__ import annotations

import cv2
import numpy as np
from skimage.segmentation import felzenszwalb


def extract_detail_layer(
    image: np.ndarray,
    blur_ksize: int = 15,
    blur_sigma: float = 3.0,
) -> np.ndarray:
    """Step 1: high-pass detail extraction via Gaussian difference."""
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    blurred = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), sigmaX=blur_sigma)
    diff = np.abs(image.astype(np.float32) - blurred.astype(np.float32))
    return diff.max(axis=2)


def create_overlay_mask(
    detail: np.ndarray,
    diff_thresh: float = 20.0,
    expand_radius: int = 15,
) -> np.ndarray:
    """Step 2+3: threshold + smooth expansion."""
    binary = (detail > diff_thresh).astype(np.uint8) * 255
    if expand_radius > 0:
        ksize = expand_radius * 2 + 1
        blurred = cv2.GaussianBlur(binary, (ksize, ksize), sigmaX=expand_radius)
        return blurred > 64
    return binary > 0


def diff_overlay_pipeline(
    image: np.ndarray,
    blur_ksize: int = 15,
    blur_sigma: float = 3.0,
    diff_thresh: float = 20.0,
    expand_radius: int = 15,
    felz_scale: float = 300.0,
    felz_sigma: float = 0.5,
    overlay_label: int = -1,
) -> dict:
    """Full diff-overlay pipeline.

    Returns dict with keys:
        detail, overlay_mask, geo_labels, final_labels, overlay_only, inpainted
    """
    detail = extract_detail_layer(image, blur_ksize, blur_sigma)
    overlay_mask = create_overlay_mask(detail, diff_thresh, expand_radius)

    inpaint_mask = overlay_mask.astype(np.uint8) * 255
    inpainted = cv2.inpaint(image, inpaint_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

    geo_labels = felzenszwalb(inpainted, scale=felz_scale, sigma=felz_sigma, min_size=30)

    final_labels = geo_labels.copy()
    final_labels[overlay_mask] = overlay_label

    overlay_vis = image.copy()
    overlay_vis[overlay_mask] = [255, 0, 255]

    return {
        "detail": detail,
        "overlay_mask": overlay_mask,
        "geo_labels": geo_labels,
        "final_labels": final_labels,
        "overlay_only": overlay_vis,
        "inpainted": inpainted,
    }
