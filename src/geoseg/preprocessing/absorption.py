"""Artifact absorption via inpainting."""
from __future__ import annotations

import cv2
import numpy as np


def fill_mask_nearest_along_axis(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    axis: str = "horizontal",
) -> np.ndarray:
    """Fill masked pixels from the nearest unmasked pixel along one axis.

    Horizontal filling preserves stratigraphic bands through annotation boxes;
    vertical filling is useful for wide colorbars that occlude a lower layer.
    """
    if img_rgb.shape[:2] != mask.shape:
        raise ValueError("mask shape must match image spatial shape")
    if axis not in {"horizontal", "vertical"}:
        raise ValueError("axis must be 'horizontal' or 'vertical'")
    if axis == "vertical":
        transposed = fill_mask_nearest_along_axis(
            np.swapaxes(img_rgb, 0, 1), np.swapaxes(mask, 0, 1), axis="horizontal"
        )
        return np.swapaxes(transposed, 0, 1)

    result = img_rgb.copy()
    width = img_rgb.shape[1]
    all_x = np.arange(width)
    for row in range(img_rgb.shape[0]):
        valid_x = all_x[~mask[row]]
        masked_x = all_x[mask[row]]
        if not len(masked_x) or not len(valid_x):
            continue
        insertion = np.searchsorted(valid_x, masked_x)
        left_index = np.clip(insertion - 1, 0, len(valid_x) - 1)
        right_index = np.clip(insertion, 0, len(valid_x) - 1)
        left_x = valid_x[left_index]
        right_x = valid_x[right_index]
        nearest_x = np.where(masked_x - left_x <= right_x - masked_x, left_x, right_x)
        result[row, masked_x] = img_rgb[row, nearest_x]
    return result


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
