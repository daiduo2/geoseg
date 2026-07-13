"""Objective evaluation metrics for segmentation results."""
from __future__ import annotations

import numpy as np


def evaluate_segmentation(image: np.ndarray, labels: np.ndarray) -> dict:
    """Compute objective metrics for a segmentation result.

    Args:
        image: (H, W, 3) RGB array
        labels: (H, W) int array, may contain negative overlay labels

    Returns:
        dict with n_labels, fragment_ratio, color_purity, mean_label_size
    """
    h, w = image.shape[:2]
    total = h * w

    # Filter out overlay labels (< 0)
    valid_mask = labels >= 0
    if not np.any(valid_mask):
        return {"n_labels": 0, "fragment_ratio": 0.0, "color_purity": 0.0, "mean_label_size": 0.0}

    valid_labels = labels[valid_mask]
    unique, counts = np.unique(valid_labels, return_counts=True)

    n_labels = len(unique)

    # fragment_ratio: labels with area < 1% of image
    small = counts < total * 0.01
    fragment_ratio = small.sum() / len(unique) if len(unique) > 0 else 0.0

    # color_purity: mean std of colors within each label
    purities = []
    for lbl in unique:
        mask = labels == lbl
        colors = image[mask]
        if len(colors) > 1:
            purity = np.std(colors.astype(float), axis=0).mean()
            purities.append(purity)
    color_purity = np.mean(purities) if purities else 0.0

    mean_label_size = counts.mean() / total

    return {
        "n_labels": n_labels,
        "fragment_ratio": fragment_ratio,
        "color_purity": color_purity,
        "mean_label_size": mean_label_size,
    }
