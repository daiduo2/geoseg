"""Parameterized version of process_final_v3 pipeline."""
from __future__ import annotations

import cv2
import numpy as np
from skimage.segmentation import felzenszwalb


def enhance_v(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def compute_color_gradient(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(sobel_x**2 + sobel_y**2)


def _build_adjacency_pairs(result: np.ndarray) -> np.ndarray:
    """Build all adjacent label pairs as (N, 2) array."""
    # Horizontal edges
    left = result[:, :-1].ravel()
    right = result[:, 1:].ravel()
    h_mask = left != right
    # Vertical edges
    up = result[:-1, :].ravel()
    down = result[1:, :].ravel()
    v_mask = up != down
    pairs = np.vstack([
        np.column_stack([left[h_mask], right[h_mask]]),
        np.column_stack([up[v_mask], down[v_mask]]),
    ])
    pairs = np.sort(pairs, axis=1)
    return pairs[pairs[:, 0] != pairs[:, 1]]


def post_merge(
    label_img: np.ndarray,
    image: np.ndarray,
    small_ratio: float = 0.015,
    max_score: float = 0.8,
    max_color: float = 45.0,
) -> np.ndarray:
    h, w = label_img.shape
    gradient = compute_color_gradient(image)
    result = label_img.copy()
    total = h * w

    # Phase 1: small fragments (max 3 iter)
    for _ in range(3):
        unique, counts = np.unique(result, return_counts=True)
        small = unique[counts < total * small_ratio]
        if len(small) == 0:
            break

        # Precompute mean colors for all labels
        mean_colors = {int(lbl): image[result == lbl].mean(axis=0) for lbl in unique}

        # Build adjacency pairs once per iteration
        pairs = _build_adjacency_pairs(result)

        merged_any = False
        for small_lbl in small:
            # Find neighbors
            mask_a = pairs[:, 0] == small_lbl
            mask_b = pairs[:, 1] == small_lbl
            if not np.any(mask_a) and not np.any(mask_b):
                continue
            neighbors = np.concatenate([pairs[mask_a, 1], pairs[mask_b, 0]])
            if len(neighbors) == 0:
                continue
            # Most frequent neighbor
            unique_nbrs, nbr_counts = np.unique(neighbors, return_counts=True)
            best_nbr = None
            best_count = -1
            best_dist = float("inf")
            for nbr, cnt in zip(unique_nbrs, nbr_counts):
                dist = np.linalg.norm(mean_colors[int(small_lbl)] - mean_colors[int(nbr)])
                # Tie-break by count, then by distance
                if cnt > best_count or (cnt == best_count and dist < best_dist):
                    best_count = cnt
                    best_dist = dist
                    best_nbr = nbr
            if best_nbr is not None:
                result[result == small_lbl] = best_nbr
                merged_any = True

        if not merged_any:
            break

    # Phase 2: conservative gradient-aware merge (max 3 iter)
    for _ in range(3):
        unique = np.unique(result)
        if len(unique) <= 4:
            break
        mean_colors = {int(lbl): image[result == lbl].mean(axis=0) for lbl in unique}

        # Build pair_data with gradients
        pairs = _build_adjacency_pairs(result)
        if len(pairs) == 0:
            break

        # Compute gradient at boundary pixels
        # Horizontal boundaries
        h_left = result[:, :-1]
        h_right = result[:, 1:]
        h_grad = gradient[:, :-1]
        h_mask = h_left != h_right
        # Vertical boundaries
        v_up = result[:-1, :]
        v_down = result[1:, :]
        v_grad = gradient[:-1, :]
        v_mask = v_up != v_down

        pair_data = {}
        for a, b, g in zip(h_left[h_mask], h_right[h_mask], h_grad[h_mask]):
            pair = (min(int(a), int(b)), max(int(a), int(b)))
            pair_data.setdefault(pair, []).append(g)
        for a, b, g in zip(v_up[v_mask], v_down[v_mask], v_grad[v_mask]):
            pair = (min(int(a), int(b)), max(int(a), int(b)))
            pair_data.setdefault(pair, []).append(g)

        if not pair_data:
            break
        scores = []
        for pair, grads in pair_data.items():
            i, j = pair
            cd = np.linalg.norm(mean_colors[i] - mean_colors[j])
            mg = np.mean(grads)
            score = cd / (mg + 1e-3)
            scores.append((score, pair, cd, mg))
        scores.sort(key=lambda x: x[0])
        best_score, best_pair, cd, mg = scores[0]
        if best_score >= max_score or cd >= max_color:
            break
        i, j = best_pair
        result[result == j] = i

    # Final renumber
    remap = {}
    next_lbl = 0
    new_result = np.zeros_like(result)
    for lbl in sorted(np.unique(result)):
        remap[lbl] = next_lbl
        next_lbl += 1
    for old, new in remap.items():
        new_result[result == old] = new
    return new_result


def v3_pipeline(
    image: np.ndarray,
    felz_scale: float = 300.0,
    felz_sigma: float = 0.5,
    felz_min_size: int = 30,
    small_ratio: float = 0.015,
    max_score: float = 0.8,
    max_color: float = 45.0,
) -> dict:
    """Run v3 pipeline with configurable parameters."""
    enhanced = enhance_v(image)
    labels = felzenszwalb(enhanced, scale=felz_scale, sigma=felz_sigma, min_size=felz_min_size)
    n_init = len(np.unique(labels))
    labels = post_merge(labels, enhanced, small_ratio, max_score, max_color)
    return {
        "labels": labels,
        "n_init": n_init,
        "n_final": len(np.unique(labels)),
    }
