"""Edge-guided K-means segmentation (e014).

K-means alone struggles because geological layer boundaries are gradual color
transitions. This module first detects edges via Canny in LAB space, then uses
the edge map as a spatial constraint during clustering.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy import ndimage
from skimage.color import rgb2lab
from skimage.feature import canny
from skimage.morphology import closing, disk
from skimage.measure import label, regionprops

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


def _compute_edge_map(
    panel_lab: np.ndarray,
    canny_sigma: float = 1.0,
    canny_low: float = 0.05,
    canny_high: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect geological layer boundaries via Canny on L channel.

    Returns (gradient, edge_mask).
    """
    l_norm = panel_lab[..., 0] / 100.0
    edge_mask = canny(l_norm, sigma=canny_sigma, low_threshold=canny_low, high_threshold=canny_high)
    edge_mask = closing(edge_mask, footprint=disk(1))
    gradient = edge_mask.astype(np.float32)
    return gradient, edge_mask


def _edge_guided_kmeans(
    panel_lab: np.ndarray,
    seeds_lab: np.ndarray,
    edge_mask: np.ndarray,
    edge_weight: float = 0.3,
    sigma: float = 4.0,
    max_iter: int = 30,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Edge-guided K-means: standard K-means + selective boundary-pixel snapping."""
    h, w = panel_lab.shape[:2]
    flat_lab = panel_lab.reshape(-1, 3)
    k = seeds_lab.shape[0]

    centroids, labels_flat = kmeans2(flat_lab, seeds_lab, minit="matrix", iter=max_iter, thresh=tol)
    labels = labels_flat.reshape(h, w).astype(np.int32)
    centroids = centroids.astype(np.float64)

    if edge_weight <= 0 or not edge_mask.any():
        return centroids, labels

    dist_to_edge = ndimage.distance_transform_edt(~edge_mask).astype(np.float32)
    snap_zone = dist_to_edge <= sigma

    regions = label(~edge_mask, connectivity=2)
    region_to_cluster: dict[int, int] = {}
    region_props = regionprops(regions)
    for rp in region_props:
        rid = rp.label
        mask = regions == rid
        vals, counts = np.unique(labels[mask], return_counts=True)
        region_to_cluster[rid] = int(vals[counts.argmax()])

    d_all = np.linalg.norm(flat_lab[:, None, :] - centroids[None, :, :], axis=2)
    d_sorted = np.partition(d_all, kth=1, axis=1)
    d_best = d_sorted[:, 0]
    d_second = d_sorted[:, 1]

    ambiguity = d_best / (d_second + 1e-9)
    ambiguous = ambiguity > (1.0 - edge_weight)
    ambiguous = ambiguous.reshape(h, w)

    labels_snapped = labels.copy()
    candidates = snap_zone & ambiguous
    snap_y, snap_x = np.where(candidates)
    for y, x in zip(snap_y, snap_x):
        rid = regions[y, x]
        if rid in region_to_cluster:
            labels_snapped[y, x] = region_to_cluster[rid]

    return centroids, labels_snapped


def segment(
    panel_rgb: np.ndarray,
    reps: list[dict],
    n_layers: int = 5,
    max_auto_k: int = 2,
    edge_weight: float = 0.5,
    sigma: float = 3.0,
) -> dict:
    """Edge-guided K-means segmentation for vivid jet-colormap panels.

    Args:
        panel_rgb: RGB uint8 array (H, W, 3).
        reps: VLM representative points.
        n_layers: Not used directly (derived from reps), kept for interface consistency.
        max_auto_k: Maximum extra seeds to auto-detect.
        edge_weight: Spatial penalty strength (0 = standard K-means).
        sigma: Gaussian fall-off width for edge penalty.

    Returns:
        dict with keys: labels, seeds, overlay, meta.
    """
    if not reps:
        raise ValueError("edge_guided path requires at least one rep")

    h, w, _ = panel_rgb.shape
    panel_lab = rgb2lab(panel_rgb)
    bg_rgb = _estimate_background_color(panel_rgb)
    min_auto_count = max(50, h * w // 2000)

    gradient, edge_mask = _compute_edge_map(panel_lab)

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

    centroids, labels = _edge_guided_kmeans(
        panel_lab, seeds_lab, edge_mask,
        edge_weight=edge_weight, sigma=sigma,
    )
    labels = _shape_filter(labels)
    labels = _merge_small_regions(labels, min_area_frac=0.003)

    overlay = _create_overlay(panel_rgb, labels, refined_seeds_arr)

    return {
        "labels": labels,
        "seeds": refined_seeds_arr.tolist(),
        "overlay": overlay,
        "meta": {
            "engine": "edge_guided",
            "reps_refined": refined_reps,
            "cv_seeds": cv_seeds_rgb.tolist() if len(cv_seeds_rgb) else [],
            "bg_rgb": bg_rgb.tolist(),
            "auto_k_added": len(refined_reps) - len(reps),
            "edge_weight": edge_weight,
            "sigma": sigma,
            "edge_pixels_pct": float(edge_mask.mean() * 100),
            "centroids_lab": centroids.tolist(),
            "saturation_ratio": round(saturation_ratio(panel_rgb), 4),
        },
    }
