"""Edge-enhanced multi-source region growing (e016).

Uses LAB gradient magnitude as a barrier during Dijkstra expansion so that
region growing stops at actual geological boundaries rather than bleeding
across gradual colour transitions.
"""

from __future__ import annotations

import heapq

import numpy as np
from skimage.color import rgb2lab

from geoseg.modules.segment_engines.edge.gradients import lab_sobel_edge_map
from geoseg.modules.segment_engines.edge.postprocess import postprocess_edge_labels
from geoseg.modules.segment_engines.edge.seeds import prepare_edge_seeds, seeds_lab
from geoseg.modules.segment_engines.internal.color import saturation_ratio
from geoseg.modules.segment_engines.internal.overlay import _create_overlay


def _region_grow_dijkstra_edge(
    panel_lab: np.ndarray,
    seeds_xy: list[tuple[int, int]],
    seeds_lab: np.ndarray,
    edge_map: np.ndarray,
    edge_penalty: float = 100.0,
) -> np.ndarray:
    """Multi-source Dijkstra in LAB space with edge barrier penalty."""
    h, w = panel_lab.shape[:2]

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
    reps: list[dict] | None = None,
    n_layers: int = 5,
    max_auto_k: int = 0,
    edge_penalty: float = 150.0,
) -> dict:
    """Edge-enhanced multi-source region growing for vivid jet-colormap panels.

    Args:
        panel_rgb: RGB uint8 array (H, W, 3).
        reps: Optional VLM representative points. If None, uses CV seeds only.
        n_layers: Target layer count when reps is None; kept for interface consistency.
        max_auto_k: Maximum extra seeds to auto-detect.
        edge_penalty: Cost multiplier for crossing strong edges.

    Returns:
        dict with keys: labels, seeds, overlay, meta.
    """
    panel_lab = rgb2lab(panel_rgb)

    _, edge_map = lab_sobel_edge_map(panel_lab)
    prepared = prepare_edge_seeds(
        panel_rgb,
        panel_lab,
        reps,
        n_layers,
        max_auto_k,
    )

    refined_seeds_arr = prepared.refined_seeds_rgb
    seed_lab = seeds_lab(refined_seeds_arr)
    seeds_xy = [(rep["internal_x"], rep["internal_y"]) for rep in prepared.refined_reps]

    labels = _region_grow_dijkstra_edge(
        panel_lab,
        seeds_xy,
        seed_lab,
        edge_map,
        edge_penalty=edge_penalty,
    )
    labels = postprocess_edge_labels(labels)

    overlay = _create_overlay(panel_rgb, labels, refined_seeds_arr)

    return {
        "labels": labels,
        "seeds": refined_seeds_arr.tolist(),
        "overlay": overlay,
        "meta": {
            "engine": "edge_grow",
            "reps_refined": prepared.refined_reps,
            "cv_seeds": (
                prepared.cv_seeds_rgb.tolist() if len(prepared.cv_seeds_rgb) else []
            ),
            "bg_rgb": prepared.bg_rgb.tolist(),
            "auto_k_added": prepared.auto_k_added,
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
