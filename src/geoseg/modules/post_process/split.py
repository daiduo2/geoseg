"""Label splitting utilities for post-processing segmentation results.

Handles over-merged regions by separating them into colour-homogeneous
connected components.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import ndimage
from scipy.cluster.vq import kmeans2


def _run_kmeans(pixels: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Run k-means and return a cluster assignment for every pixel.

    Tries k-means++ first (best initialisation).  Falls back to ``points``
    for older SciPy versions, and retries with ``random`` if the first run
    produces empty clusters.
    """
    if k <= 1:
        return np.zeros(pixels.shape[0], dtype=np.int32)

    init_methods = [("++", True), ("points", True), ("random", False)]

    for minit, check_empty in init_methods:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, cluster_ids = kmeans2(pixels, k, seed=seed, minit=minit)
        except Exception:
            # Older SciPy may not support '++' initialisation, or the data may
            # be too low-rank for a covariance-based init.  Try the next one.
            continue

        if not check_empty:
            return cluster_ids

        unique_ids = set(np.unique(cluster_ids))
        if len(unique_ids) == k:
            return cluster_ids
        # Empty clusters: try the next init method.

    # All initialisations failed or produced empty clusters; fall back to a
    # single cluster so that spatial connectivity can still split the region.
    return np.zeros(pixels.shape[0], dtype=np.int32)


def split_label_by_color_components(
    labels: np.ndarray,
    img_rgb: np.ndarray,
    target_label: int,
    *,
    color_space: str = "LAB",
    k: int = 3,
    min_component_area: int = 300,
    seed: int = 42,
) -> np.ndarray:
    """Split a single over-merged label into color-based connected components.

    Steps:
    1. Mask target_label pixels.
    2. Cluster masked pixels in LAB or RGB space using k-means
       (``scipy.cluster.vq.kmeans2``).
    3. For each color cluster, run ``ndimage.label`` to get spatially
       connected components.
    4. Drop components smaller than ``min_component_area``.
    5. Assign unassigned small pixels to nearest surviving component.
    6. Renumber new labels compactly, preserving all other original labels.
    7. Return new label map.

    Args:
        labels: Integer label map.
        img_rgb: Original RGB image with the same spatial shape as ``labels``.
        target_label: The label ID to split.
        color_space: Either ``"LAB"`` or ``"RGB"``. LAB usually gives more
            perceptually uniform colour clusters.
        k: Number of colour clusters. The actual number of clusters is capped
            to the number of masked pixels.
        min_component_area: Minimum area (in pixels) for a component to be
            kept. Smaller components are discarded and their pixels reassigned.
        seed: Random seed passed to ``kmeans2`` for reproducibility.

    Returns:
        New label map with the target label split into compactly-numbered
        components and all other labels preserved.
    """
    if labels.shape[:2] != img_rgb.shape[:2]:
        raise ValueError("Shape mismatch between labels and image")

    if color_space not in {"LAB", "RGB"}:
        raise ValueError(
            f"Unsupported color_space: {color_space!r}. Use 'LAB' or 'RGB'."
        )

    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be an integer >= 1")

    if not isinstance(min_component_area, int) or min_component_area < 0:
        raise ValueError("min_component_area must be an integer >= 0")

    result = labels.copy()
    target_mask = labels == target_label
    if not target_mask.any():
        return result

    ys, xs = np.where(target_mask)
    pixels = img_rgb[ys, xs].astype(np.float32)

    if color_space == "LAB":
        import cv2

        lab = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2LAB)
        pixels = lab[ys, xs].astype(np.float32)

    n_pixels = pixels.shape[0]
    actual_k = min(k, n_pixels)

    if actual_k == 1:
        cluster_masks = [target_mask]
    else:
        cluster_ids = _run_kmeans(pixels, actual_k, seed=seed)
        cluster_masks = []
        for ci in range(actual_k):
            cluster_idx = np.where(cluster_ids == ci)[0]
            if cluster_idx.size == 0:
                continue
            cm = np.zeros_like(target_mask)
            cm[ys[cluster_idx], xs[cluster_idx]] = True
            cluster_masks.append(cm)

    # Extract spatially connected components from each colour cluster and
    # filter by area.
    component_ids: list[int] = []
    working = np.zeros_like(labels)
    next_id = 1
    for cm in cluster_masks:
        labeled, num = ndimage.label(cm)
        for i in range(1, num + 1):
            comp = labeled == i
            if comp.sum() >= min_component_area:
                working[comp] = next_id
                component_ids.append(next_id)
                next_id += 1

    survivor_mask = working > 0
    unassigned = target_mask & ~survivor_mask

    if unassigned.any():
        if not survivor_mask.any():
            # No surviving component: keep target label unchanged.
            return result
        _, indices = ndimage.distance_transform_edt(
            ~survivor_mask, return_indices=True
        )
        rr, cc = np.where(unassigned)
        nearest_rr = indices[0][rr, cc]
        nearest_cc = indices[1][rr, cc]
        working[rr, cc] = working[nearest_rr, nearest_cc]

    # Preserve all original labels except target_label, then assign new
    # compact labels starting above the highest preserved value.
    kept_labels = set(labels.flatten()) - {0, target_label}
    next_new_label = max(kept_labels, default=0) + 1

    final = labels.copy()
    for comp_id in component_ids:
        final[working == comp_id] = next_new_label
        next_new_label += 1

    # Any remaining target pixels that could not be mapped (e.g. all components
    # dropped) are left at their original value to avoid losing data silently.
    remaining = final == target_label
    if remaining.any() and not component_ids:
        return result

    return final
