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
from skimage.measure import label, regionprops

from geoseg.modules.segment_engines.edge.gradients import canny_edge_map
from geoseg.modules.segment_engines.edge.postprocess import postprocess_edge_labels
from geoseg.modules.segment_engines.edge.seeds import prepare_edge_seeds, seeds_lab
from geoseg.modules.segment_engines.internal.color import saturation_ratio
from geoseg.modules.segment_engines.internal.overlay import _create_overlay


def _compute_edge_map(
    panel_lab: np.ndarray,
    canny_sigma: float = 1.0,
    canny_low: float = 0.05,
    canny_high: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect geological layer boundaries via Canny on L channel.

    Returns (gradient, edge_mask).
    """
    return canny_edge_map(
        panel_lab,
        canny_sigma=canny_sigma,
        canny_low=canny_low,
        canny_high=canny_high,
    )


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

    centroids, labels_flat = kmeans2(
        flat_lab,
        seeds_lab,
        minit="matrix",
        iter=max_iter,
        thresh=tol,
    )
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
    reps: list[dict] | None = None,
    n_layers: int = 5,
    max_auto_k: int = 0,
    edge_weight: float = 0.5,
    sigma: float = 3.0,
) -> dict:
    """Edge-guided K-means segmentation for vivid jet-colormap panels.

    Args:
        panel_rgb: RGB uint8 array (H, W, 3).
        reps: Optional VLM representative points. If None, uses CV seeds only.
        n_layers: Target layer count when reps is None; kept for interface consistency.
        max_auto_k: Maximum extra seeds to auto-detect.
        edge_weight: Spatial penalty strength (0 = standard K-means).
        sigma: Gaussian fall-off width for edge penalty.

    Returns:
        dict with keys: labels, seeds, overlay, meta.
    """
    panel_lab = rgb2lab(panel_rgb)

    _, edge_mask = _compute_edge_map(panel_lab)
    prepared = prepare_edge_seeds(
        panel_rgb,
        panel_lab,
        reps,
        n_layers,
        max_auto_k,
    )

    refined_seeds_arr = prepared.refined_seeds_rgb
    seed_lab = seeds_lab(refined_seeds_arr)

    centroids, labels = _edge_guided_kmeans(
        panel_lab,
        seed_lab,
        edge_mask,
        edge_weight=edge_weight,
        sigma=sigma,
    )
    labels = postprocess_edge_labels(labels)

    overlay = _create_overlay(panel_rgb, labels, refined_seeds_arr)

    return {
        "labels": labels,
        "seeds": refined_seeds_arr.tolist(),
        "overlay": overlay,
        "meta": {
            "engine": "edge_guided",
            "reps_refined": prepared.refined_reps,
            "cv_seeds": (
                prepared.cv_seeds_rgb.tolist() if len(prepared.cv_seeds_rgb) else []
            ),
            "bg_rgb": prepared.bg_rgb.tolist(),
            "auto_k_added": prepared.auto_k_added,
            "edge_weight": edge_weight,
            "sigma": sigma,
            "edge_pixels_pct": float(edge_mask.mean() * 100),
            "centroids_lab": centroids.tolist(),
            "saturation_ratio": round(saturation_ratio(panel_rgb), 4),
        },
    }
