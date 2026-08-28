"""Geometric normalization helpers for perspective-rendered sections."""

from __future__ import annotations

import cv2
import numpy as np


def rectify_quadrilateral(
    image: np.ndarray,
    source_points: np.ndarray,
    output_size: tuple[int, int] | None = None,
    *,
    interpolation: int = cv2.INTER_CUBIC,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp four ordered corners (TL, TR, BR, BL) to a rectangle.

    Returns the rectified image and the 3x3 source-to-output homography.
    """
    points = np.asarray(source_points, dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError("source_points must have shape (4, 2)")

    if output_size is None:
        top = np.linalg.norm(points[1] - points[0])
        bottom = np.linalg.norm(points[2] - points[3])
        left = np.linalg.norm(points[3] - points[0])
        right = np.linalg.norm(points[2] - points[1])
        output_size = (
            max(1, int(round(max(top, bottom)))),
            max(1, int(round(max(left, right)))),
        )

    width, height = output_size
    if width <= 0 or height <= 0:
        raise ValueError("output_size values must be positive")

    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(points, destination)
    rectified = cv2.warpPerspective(
        image,
        transform,
        (width, height),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rectified, transform


__all__ = ["rectify_quadrilateral"]
