"""Graph Cut segmentation for geophysics interpretation panels.

Multi-label segmentation using iterative binary graph cuts (alpha-expansion style)
with boundary refinement. Uses PyMaxflow for Boykov-Kolmogorov maxflow/mincut.

Public API:
    segment_jet_vivid_graphcut(panel_rgb, reps, max_auto_k=3) -> SegmentResult
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.color import rgb2lab

from lib.segment import (
    SegmentResult,
    saturation_ratio,
    segment_jet_vivid,
    _label_by_nearest,
    _shape_filter,
)


def _compute_seed_models(panel_lab: np.ndarray, seeds_xy: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Compute Gaussian color models (mean, std) for each seed from a small neighborhood.

    Returns:
        means: (k, 3) LAB means
        stds: (k,) per-seed average LAB std
    """
    h, w = panel_lab.shape[:2]
    k = len(seeds_xy)
    means = np.zeros((k, 3), dtype=np.float32)
    stds = np.zeros(k, dtype=np.float32)

    for i, (x, y) in enumerate(seeds_xy):
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        y0, y1 = max(0, y - 3), min(h, y + 4)
        x0, x1 = max(0, x - 3), min(w, x + 4)
        patch = panel_lab[y0:y1, x0:x1]
        means[i] = patch.mean(axis=(0, 1))
        stds[i] = max(patch.reshape(-1, 3).std(axis=0).mean(), 1.0)

    return means, stds


def _boundary_pixels(labels: np.ndarray, radius: int = 3) -> np.ndarray:
    """Return a boolean mask of pixels within ``radius`` of a different label boundary."""
    h, w = labels.shape
    # Find boundary pixels (4-connected neighbors differ)
    boundary = np.zeros((h, w), dtype=bool)
    boundary[1:, :] |= labels[1:, :] != labels[:-1, :]
    boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]

    if radius <= 1:
        return boundary

    # Dilate boundary mask
    from skimage.morphology import disk
    dilated = ndimage.binary_dilation(boundary, structure=disk(radius))
    return dilated


def _build_binary_graphcut(
    panel_lab: np.ndarray,
    labels: np.ndarray,
    alpha_label: int,
    beta_label: int,
    seed_means: np.ndarray,
    seed_stds: np.ndarray,
    sigma_color: float = 15.0,
) -> np.ndarray:
    """Refine labels for pixels currently labeled alpha or beta using binary graph cut.

    Returns updated labels array (only pixels that were alpha/beta may change).
    """
    import maxflow

    h, w = panel_lab.shape[:2]

    # Mask of pixels involved in this binary cut
    mask = (labels == alpha_label) | (labels == beta_label)
    if not mask.any():
        return labels.copy()

    # Map 2D coords to 1D node indices for pixels in mask
    idx_map = np.full((h, w), -1, dtype=np.int32)
    idx_map[mask] = np.arange(mask.sum())
    n_nodes = int(mask.sum())

    g = maxflow.Graph[float]()
    nodes = g.add_nodes(n_nodes)

    # --- Terminal edges (data term) ---
    # Cost to assign to alpha (source) or beta (sink)
    # Use Gaussian likelihood in LAB space
    mean_a = seed_means[alpha_label]
    mean_b = seed_means[beta_label]
    std_a = max(seed_stds[alpha_label], 3.0)
    std_b = max(seed_stds[beta_label], 3.0)

    flat_lab = panel_lab.reshape(-1, 3)
    flat_mask = mask.reshape(-1)
    pixels = flat_lab[flat_mask]  # (n_nodes, 3)

    # Negative log-likelihood (squared Mahalanobis-ish, using scalar std)
    d_a = np.linalg.norm(pixels - mean_a, axis=1) ** 2
    d_b = np.linalg.norm(pixels - mean_b, axis=1) ** 2

    # Normalize by variance to make costs comparable
    cost_a = d_a / (2 * std_a ** 2)
    cost_b = d_b / (2 * std_b ** 2)

    # Clamp costs to avoid extreme values
    cost_a = np.clip(cost_a, 0.0, 50.0)
    cost_b = np.clip(cost_b, 0.0, 50.0)

    for i in range(n_nodes):
        # PyMaxflow: add_tedge(i, cap_source, cap_sink)
        # If pixel is assigned to SOURCE (alpha), we cut the sink edge -> cost = cap_sink
        # If pixel is assigned to SINK (beta), we cut the source edge -> cost = cap_source
        # So: cap_sink = cost_to_be_alpha, cap_source = cost_to_be_beta
        g.add_tedge(nodes[i], float(cost_b[i]), float(cost_a[i]))

    # --- Smoothness term (4-neighbor edges) ---
    # Weight based on LAB color similarity: exp(-d^2 / (2*sigma^2))
    # Stronger edge = lower weight = less likely to cut
    sigma_sq = sigma_color ** 2

    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue
            u = int(idx_map[y, x])
            lab_u = panel_lab[y, x]

            # Right neighbor
            if x + 1 < w and mask[y, x + 1]:
                v = int(idx_map[y, x + 1])
                lab_v = panel_lab[y, x + 1]
                d2 = float(np.sum((lab_u - lab_v) ** 2))
                weight = np.exp(-d2 / (2 * sigma_sq))
                # Add bidirectional edge with capacity = weight
                g.add_edge(u, v, weight, weight)

            # Down neighbor
            if y + 1 < h and mask[y + 1, x]:
                v = int(idx_map[y + 1, x])
                lab_v = panel_lab[y + 1, x]
                d2 = float(np.sum((lab_u - lab_v) ** 2))
                weight = np.exp(-d2 / (2 * sigma_sq))
                g.add_edge(u, v, weight, weight)

    # --- Run maxflow ---
    g.maxflow()

    # --- Extract labels ---
    out = labels.copy()
    flat_out = out.reshape(-1)
    flat_mask_idx = np.where(flat_mask)[0]

    for i in range(n_nodes):
        seg = g.get_segment(nodes[i])
        pixel_idx = flat_mask_idx[i]
        if seg == 0:  # source side -> alpha
            flat_out[pixel_idx] = alpha_label
        else:  # sink side -> beta
            flat_out[pixel_idx] = beta_label

    return out.reshape(h, w)


def _alpha_expansion_simplified(
    panel_lab: np.ndarray,
    initial_labels: np.ndarray,
    seeds_xy: list[tuple[int, int]],
    seed_means: np.ndarray,
    seed_stds: np.ndarray,
    n_iter: int = 2,
    sigma_color: float = 15.0,
) -> np.ndarray:
    """Simplified alpha-expansion: iterate over label pairs and refine boundary regions.

    Instead of full alpha-expansion (which requires complex move construction),
    we identify boundary pixels and run binary graph cuts for adjacent label pairs.
    """
    labels = initial_labels.copy()
    k = len(seeds_xy)
    if k < 2:
        return labels

    # Find adjacent label pairs from initial segmentation
    adjacency = set()
    for y in range(labels.shape[0] - 1):
        for x in range(labels.shape[1] - 1):
            a = int(labels[y, x])
            b_right = int(labels[y, x + 1])
            b_down = int(labels[y + 1, x])
            if a != b_right:
                adjacency.add(tuple(sorted((a, b_right))))
            if a != b_down:
                adjacency.add(tuple(sorted((a, b_down))))

    adjacency = list(adjacency)
    if not adjacency:
        return labels

    for iteration in range(n_iter):
        changed = False
        for alpha, beta in adjacency:
            new_labels = _build_binary_graphcut(
                panel_lab, labels, alpha, beta,
                seed_means, seed_stds, sigma_color=sigma_color
            )
            if not np.array_equal(new_labels, labels):
                changed = True
                labels = new_labels
        if not changed:
            break

    return labels


def segment_jet_vivid_graphcut(
    panel_rgb: np.ndarray,
    reps: list[dict],
    max_auto_k: int = 3,
    sigma_color: float = 15.0,
    n_iter: int = 2,
    boundary_radius: int = 3,
) -> SegmentResult:
    """Graph Cut segmentation for vivid jet-colormap panels.

    Strategy:
        1. Reuse seed refinement logic from segment_jet_vivid (VLM seeds + auto-k).
        2. Compute regional Gaussian color models around each seed in LAB space.
        3. Start with nearest-neighbor labels.
        4. Identify boundary pixels (within ``boundary_radius`` of a label change).
        5. For adjacent label pairs, run binary graph cut on the boundary region
           using color-model data terms and LAB-similarity smoothness terms.
        6. Apply shape filter post-processing.

    Args:
        panel_rgb: (H, W, 3) uint8 image.
        reps: VLM representative points, same format as segment_jet_vivid.
        max_auto_k: Max auto-detected seeds to add.
        sigma_color: Smoothness sigma in LAB space (default 15.0).
        n_iter: Number of alpha-expansion iterations (default 2).
        boundary_radius: Pixel distance from boundary to include in graph cut (default 3).

    Returns:
        SegmentResult with graphcut-refined labels.
    """
    # --- Step 1: Get seeds using existing segment_jet_vivid logic ---
    # We call segment_jet_vivid but will discard its labels and recompute with graph cut.
    # This gives us refined seeds, auto-k, and all metadata.
    base_result = segment_jet_vivid(panel_rgb, reps, max_auto_k=max_auto_k)

    h, w, _ = panel_rgb.shape
    panel_lab = rgb2lab(panel_rgb)

    # Extract seed info from base_result notes
    refined_reps = base_result.notes["reps_refined"]
    seeds_xy = [(r["internal_x"], r["internal_y"]) for r in refined_reps]
    seeds_rgb = base_result.palette

    # --- Step 2: Compute seed color models ---
    seed_means, seed_stds = _compute_seed_models(panel_lab, seeds_xy)

    # --- Step 3: Initial nearest-neighbor labels ---
    seeds_lab = rgb2lab(seeds_rgb[np.newaxis, ...])[0]
    labels = _label_by_nearest(panel_lab, seeds_lab)

    # --- Step 4: Graph cut refinement on boundary regions ---
    labels = _alpha_expansion_simplified(
        panel_lab, labels, seeds_xy, seed_means, seed_stds,
        n_iter=n_iter, sigma_color=sigma_color
    )

    # --- Step 5: Shape filter ---
    labels = _shape_filter(labels)

    return SegmentResult(
        labels=labels,
        palette=seeds_rgb,
        color_names=base_result.color_names,
        path="jet_vivid_graphcut",
        saturation_ratio=saturation_ratio(panel_rgb),
        notes={
            **base_result.notes,
            "graphcut_params": {
                "sigma_color": sigma_color,
                "n_iter": n_iter,
                "boundary_radius": boundary_radius,
            },
            "seed_models": {
                "means_lab": seed_means.tolist(),
                "stds_lab": seed_stds.tolist(),
            },
        },
    )
