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


def split_labels_by_red_boundaries(
    labels: np.ndarray,
    img_rgb: np.ndarray,
    *,
    boundary_mask: np.ndarray | None = None,
    erosion_radius: int | None = None,
    min_component_area_frac: float = 0.005,
) -> tuple[np.ndarray, dict[int, int], np.ndarray]:
    """Split colour labels into regions separated by incomplete red traces.

    The red trace may stop shortly before the enclosing colour boundary. To
    close that topological gap without shrinking the final regions, the
    colour mask minus the red trace is eroded into stable cores. Pixels in the
    erosion band are then assigned to the nearest core. New region labels map
    back to their original colour label through the returned ``parent_map``.

    Returns:
        ``(refined_labels, parent_map, boundary_mask)``.
    """
    if labels.shape != img_rgb.shape[:2]:
        raise ValueError("Shape mismatch between labels and image")
    if labels.ndim != 2:
        raise ValueError("labels must be a 2D array")
    if not 0.0 <= min_component_area_frac < 1.0:
        raise ValueError("min_component_area_frac must be in [0, 1)")

    if boundary_mask is None:
        from geoseg.preprocessing.detectors import detect_red_boundaries

        boundary_mask = detect_red_boundaries(img_rgb)
    else:
        boundary_mask = np.asarray(boundary_mask, dtype=bool)
        if boundary_mask.shape != labels.shape:
            raise ValueError("Shape mismatch between labels and boundary_mask")

    h, w = labels.shape
    if erosion_radius is None:
        erosion_radius = max(2, int(round(min(h, w) * 0.04)))
    if erosion_radius < 0:
        raise ValueError("erosion_radius must be >= 0")

    refined = np.zeros_like(labels, dtype=np.int32)
    parent_map: dict[int, int] = {}
    next_label = 1
    structure = np.ones((3, 3), dtype=bool)

    for parent_label in sorted(set(labels.flatten()) - {0}):
        region = labels == parent_label
        cut_region = region & ~boundary_mask
        min_area = max(20, int(region.sum() * min_component_area_frac))

        if erosion_radius > 0:
            core = ndimage.distance_transform_edt(cut_region) > erosion_radius
            baseline_core = ndimage.distance_transform_edt(region) > erosion_radius
        else:
            core = cut_region
            baseline_core = region

        core_labels, core_count = ndimage.label(core, structure=structure)
        baseline_labels, baseline_count = ndimage.label(
            baseline_core, structure=structure
        )

        core_ids = [
            component_id
            for component_id in range(1, core_count + 1)
            if int((core_labels == component_id).sum()) >= min_area
        ]
        baseline_ids = [
            component_id
            for component_id in range(1, baseline_count + 1)
            if int((baseline_labels == component_id).sum()) >= min_area
        ]

        # Only introduce a split when the structural boundary creates more
        # stable components than the colour region has on its own.
        if len(core_ids) <= max(1, len(baseline_ids)):
            refined[region] = next_label
            parent_map[next_label] = int(parent_label)
            next_label += 1
            continue

        stable_cores = np.zeros_like(labels, dtype=np.int32)
        for stable_id, component_id in enumerate(core_ids, start=1):
            stable_cores[core_labels == component_id] = stable_id

        _, indices = ndimage.distance_transform_edt(
            stable_cores == 0, return_indices=True
        )
        assignable = cut_region
        nearest = stable_cores[indices[0][assignable], indices[1][assignable]]

        for stable_id in range(1, len(core_ids) + 1):
            region_mask = np.zeros_like(region)
            region_mask[assignable] = nearest == stable_id
            refined[region_mask] = next_label
            parent_map[next_label] = int(parent_label)
            next_label += 1

    # Guarantee the downstream contract that every output label is one
    # connected region. Tiny islands created by symbols or antialiasing are
    # left as background instead of becoming spurious geological regions.
    connected = np.zeros_like(refined)
    connected_parent_map: dict[int, int] = {}
    connected_label = 1
    parent_areas = {
        int(parent_label): int((labels == parent_label).sum())
        for parent_label in set(parent_map.values())
    }
    for provisional_label, parent_label in parent_map.items():
        components, count = ndimage.label(
            refined == provisional_label, structure=structure
        )
        component_sizes = [
            (component_id, int((components == component_id).sum()))
            for component_id in range(1, count + 1)
        ]
        component_min_area = max(
            20, int(parent_areas[parent_label] * min_component_area_frac)
        )
        survivors = [
            (component_id, area)
            for component_id, area in component_sizes
            if area >= component_min_area
        ]
        if not survivors and component_sizes:
            survivors = [max(component_sizes, key=lambda item: item[1])]

        for component_id, _ in survivors:
            connected[components == component_id] = connected_label
            connected_parent_map[connected_label] = parent_label
            connected_label += 1

    return connected, connected_parent_map, boundary_mask
