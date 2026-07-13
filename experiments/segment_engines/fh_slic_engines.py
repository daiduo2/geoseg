"""Segmentation engine wrappers for FH/SLIC experiment."""

from __future__ import annotations

import numpy as np
from skimage import segmentation
from skimage.color import rgb2lab, lab2rgb
from scipy.cluster.vq import kmeans2

from geoseg.modules.segment_engines import v4_kmeans
from geoseg.modules.segment_engines._shared import _create_overlay


def run_fh(
    panel_rgb: np.ndarray,
    scale: float = 100,
    sigma: float = 0.5,
    min_size: int = 50,
) -> dict:
    """Run Felzenszwalb-Huttenlocher segmentation."""
    labels = segmentation.felzenszwalb(
        panel_rgb,
        scale=scale,
        sigma=sigma,
        min_size=min_size,
    )
    unique = np.unique(labels)
    seeds = np.zeros((len(unique), 3), dtype=np.uint8)
    for i, lbl in enumerate(unique):
        mask = labels == lbl
        if mask.any():
            seeds[i] = panel_rgb[mask].mean(axis=0).astype(np.uint8)

    overlay = _create_overlay(panel_rgb, labels, seeds, alpha=0.5)
    return {
        "labels": labels,
        "overlay": overlay,
        "meta": {
            "engine": "felzenszwalb",
            "scale": scale,
            "sigma": sigma,
            "min_size": min_size,
            "n_segments": int(len(unique)),
        },
    }


def run_slic_clustering(
    panel_rgb: np.ndarray,
    n_segments: int = 500,
    compactness: float = 10,
    n_clusters: int = 5,
) -> dict:
    """Run SLIC superpixels + K-means on superpixel mean colors."""
    slic_labels = segmentation.slic(
        panel_rgb,
        n_segments=n_segments,
        compactness=compactness,
        sigma=1.0,
        start_label=0,
        channel_axis=-1,
    )

    n_sp = int(slic_labels.max()) + 1
    sp_means = np.zeros((n_sp, 3), dtype=np.float32)
    for sp_id in range(n_sp):
        mask = slic_labels == sp_id
        if mask.sum() > 0:
            sp_means[sp_id] = panel_rgb[mask].mean(axis=0)

    centroids, sp_cluster_labels = kmeans2(
        sp_means.astype(np.float64),
        n_clusters,
        minit="++",
        seed=42,
    )

    labels = sp_cluster_labels[slic_labels].astype(np.int32)

    # Reorder top-to-bottom by median y
    h, w = labels.shape
    unique = np.unique(labels)
    median_y = {}
    for lbl in unique:
        ys = np.where(labels == lbl)[0]
        median_y[lbl] = np.median(ys) if len(ys) > 0 else h
    sorted_by_y = sorted(median_y.items(), key=lambda x: x[1])
    old_to_new = {old: new for new, (old, _) in enumerate(sorted_by_y)}
    out = np.full_like(labels, -1)
    for old, new in old_to_new.items():
        out[labels == old] = new
    labels = out

    seeds = np.zeros((n_clusters, 3), dtype=np.uint8)
    for i in range(n_clusters):
        mask = labels == i
        if mask.any():
            seeds[i] = panel_rgb[mask].mean(axis=0).astype(np.uint8)
        else:
            seeds[i] = (centroids[i]).clip(0, 255).astype(np.uint8)

    overlay = _create_overlay(panel_rgb, labels, seeds, alpha=0.5)
    return {
        "labels": labels,
        "overlay": overlay,
        "meta": {
            "engine": "slic_kmeans",
            "n_segments": n_segments,
            "compactness": compactness,
            "n_clusters": n_clusters,
            "n_superpixels": n_sp,
        },
    }


def run_v4_baseline(panel_rgb: np.ndarray, n_layers: int = 5) -> dict:
    """Run v4_kmeans pastel_faded as baseline."""
    return v4_kmeans.segment_pastel_faded(panel_rgb, colorbar_rgb=None, n_layers=n_layers)
