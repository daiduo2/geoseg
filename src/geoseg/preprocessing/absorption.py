"""Artifact absorption via inpainting."""
from __future__ import annotations

import cv2
import numpy as np


def absorb_artifacts(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    inpaint_radius: int = 3,
    dilate_iters: int = 0,
    dilate_kernel_size: int = 3,
    method: str = "NS",
) -> np.ndarray:
    """Inpaint the artifact mask so artifacts blend into the background.

    Args:
        img_rgb: RGB uint8 array.
        mask: Boolean artifact mask.
        inpaint_radius: Inpaint radius.
        dilate_iters: Optional mask dilation iterations.
        dilate_kernel_size: Dilation kernel size.
        method: "NS" or "TELEA".

    Returns:
        Inpainted RGB image.
    """
    if not mask.any():
        return img_rgb.copy()

    mask_uint8 = mask.astype(np.uint8) * 255
    if dilate_iters > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel_size, dilate_kernel_size))
        mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=dilate_iters)

    flag = cv2.INPAINT_NS if method == "NS" else cv2.INPAINT_TELEA
    return cv2.inpaint(img_rgb, mask_uint8, inpaint_radius, flag)


def visualize_mask_on_image(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 255),
    alpha: float = 0.3,
) -> np.ndarray:
    """Overlay the artifact mask in a bright color for debugging."""
    overlay = img_rgb.copy()
    overlay[mask] = color
    return cv2.addWeighted(img_rgb, 1 - alpha, overlay, alpha, 0)
