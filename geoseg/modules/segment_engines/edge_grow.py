"""Edge-enhanced multi-source region growing (e016).

Uses LAB gradient magnitude as a barrier during Dijkstra expansion so that
region growing stops at actual geological boundaries rather than bleeding
across gradual colour transitions.
"""

from __future__ import annotations

import heapq

import numpy as np
from scipy import ndimage
from skimage.color import rgb2lab
from skimage.filters import sobel

from geoseg.modules.segment_engines._shared import (
    _create_overlay,
    _shape_filter,
    _merge_small_regions,
    _estimate_background_color,
    _cv_seeds,
    _refine_vlm_seeds,
    _auto_k,
    saturation_ratio,
)


def _region_grow_dijkstra_edge(
    panel_lab: np.ndarray,
    seeds_xy: list[tuple[int, int]],
    seeds_lab: np.ndarray,
    edge_map: np.ndarray,
    edge_penalty: float = 100.0,
) -> np.ndarray:
    """Multi-source Dijkstra in LAB space with edge barrier penalty."""
    h, w = panel_lab.shape[:2]
    k = len(seeds_xy)

    diff = panel_lab[:, :, None, :] - seeds_lab[None, None, :, :]
    dists = np.linalg.norm(diff, axis=3)

    best_cost = np.full((h, w), np.inf, dtype=np.float32)
    best_label = np.full((h, w), -1, dtype=np.int32)
    heap = []

    for i, (x, y) in enumerate(seeds_xy):
        d = float(dists[y, x, i])
        best_cost[y, x] = d
        best_label[y, x] = i
        heapq.heappush(heap, (d, i, x, y))

    while heap:
        cost, i, x, y = heapq.heappop(heap)
        if cost > best_cost[y, x] + 1e-6:
            continue
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                edge_cost = edge_penalty * float(edge_map[ny, nx])
                color_cost = float(dists[ny, nx, i])
                new_cost = cost + color_cost + edge_cost
                if new_cost < best_cost[ny, nx] - 1e-6:
                    best_cost[ny, nx] = new_cost
                    best_label[ny, nx] = i
                    heapq.heappush(heap, (new_cost, i, nx, ny))

    unassigned = best_label == -1
    if unassigned.any():
        nearest = dists.argmin(axis=2)
        best_label[unassigned] = nearest[unassigned]

    return best_label


def segment(
    panel_rgb: np.ndarray,
    reps: list[dict],
    n_layers: int = 5,
    max_auto_k: int = 2,
    edge_penalty: float = 150.0,
) -> dict:
    """Edge-enhanced multi-source region growing for vivid jet-colormap panels.

    Args:
        panel_rgb: RGB uint8 array (H, W, 3).
        reps: VLM representative points.
        n_layers: Not used directly (derived from reps), kept for interface consistency.
        max_auto_k: Maximum extra seeds to auto-detect.
        edge_penalty: Cost multiplier for crossing strong edges.

    Returns:
        dict with keys: labels, seeds, overlay, meta.
    """
    if not reps:
        raise ValueError("edge_grow path requires at least one rep")

    h, w, _ = panel_rgb.shape
    panel_lab = rgb2lab(panel_rgb)
    bg_rgb = _estimate_background_color(panel_rgb)
    min_auto_count = max(50, h * w // 2000)

    # Smooth LAB channels before gradient to reduce noise
    gradient = np.zeros((h, w), dtype=np.float32)
    for c in range(3):
        smoothed = ndimage.gaussian_filter(panel_lab[..., c], sigma=1.0)
        gradient += sobel(smoothed) ** 2
    gradient = np.sqrt(gradient)
    edge_map = gradient / (gradient.max() + 1e-9)

    cv_seeds_rgb, cv_tags = _cv_seeds(panel_rgb, k=len(reps))
    used_cv_indices: set[int] = set()

    refined_seeds, refined_reps = _refine_vlm_seeds(
        panel_rgb, reps, bg_rgb, cv_seeds_rgb, cv_tags, used_cv_indices
    )
    color_names = [r["color_name"] for r in reps]

    refined_seeds, refined_reps = _auto_k(
        panel_rgb, panel_lab, bg_rgb,
        refined_seeds, refined_reps,
        cv_seeds_rgb, cv_tags, used_cv_indices,
        max_auto_k, min_auto_count,
    )
    if len(refined_reps) > len(color_names):
        color_names = color_names + [r["name"] for r in refined_reps[len(color_names):]]

    refined_seeds_arr = np.array(refined_seeds, dtype=np.uint8)
    seeds_lab = rgb2lab(refined_seeds_arr[np.newaxis, ...])[0]
    seeds_xy = [(rep["internal_x"], rep["internal_y"]) for rep in refined_reps]

    labels = _region_grow_dijkstra_edge(
        panel_lab, seeds_xy, seeds_lab, edge_map, edge_penalty=edge_penalty
    )
    labels = _shape_filter(labels)
    labels = _merge_small_regions(labels, min_area_frac=0.003)

    overlay = _create_overlay(panel_rgb, labels, refined_seeds_arr)

    return {
        "labels": labels,
        "seeds": refined_seeds_arr.tolist(),
        "overlay": overlay,
        "meta": {
            "engine": "edge_grow",
            "reps_refined": refined_reps,
            "cv_seeds": cv_seeds_rgb.tolist() if len(cv_seeds_rgb) else [],
            "bg_rgb": bg_rgb.tolist(),
            "auto_k_added": len(refined_reps) - len(reps),
            "edge_penalty": edge_penalty,
            "edge_map_stats": {
                "min": float(edge_map.min()),
                "max": float(edge_map.max()),
                "mean": float(edge_map.mean()),
                "median": float(np.median(edge_map)),
            },
            "saturation_ratio": round(saturation_ratio(panel_rgb), 4),
        },
    }
