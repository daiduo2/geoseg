"""E027: SLIC + Graph Cut segmentation engine.

Produces smoother boundaries than pure K-means by regularizing with
superpixel-based graph cut energy minimization.
"""

from __future__ import annotations

import numpy as np
from skimage import segmentation, morphology, color
from scipy import ndimage
from sklearn.cluster import KMeans

from geoseg.modules.segment_engines._shared import _distinct_colors


# Defaults tuned on ph01 conceptual model panels
_DEFAULT_COMPACTNESS = 5
_DEFAULT_SUPERPIXEL_AREA = 100
_DEFAULT_SIGMA = 25.0
_DEFAULT_MIN_COMPONENT_FRAC = 0.001
_DEFAULT_MAX_ITER = 50


def extract_seeds(panel_rgb: np.ndarray, n_layers: int) -> np.ndarray:
    """Run K-means on panel pixels, return seeds sorted by median y (top-to-bottom)."""
    h, w = panel_rgb.shape[:2]
    pixels = panel_rgb.reshape(-1, 3).astype(np.float32)

    kmeans = KMeans(n_clusters=n_layers, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_.astype(np.float32)

    yy, _ = np.mgrid[:h, :w]
    y_flat = yy.ravel()

    median_y = []
    for k in range(n_layers):
        mask_k = labels == k
        median_y.append(np.median(y_flat[mask_k]) if mask_k.sum() > 0 else h / 2)

    order = np.argsort(median_y)
    return centers[order].astype(np.uint8)


def _slic_segments(panel_rgb: np.ndarray, n_segments: int, compactness: int) -> np.ndarray:
    return segmentation.slic(
        panel_rgb,
        n_segments=n_segments,
        compactness=compactness,
        sigma=1.0,
        start_label=0,
        channel_axis=-1,
    )


def _sp_features(panel_rgb: np.ndarray, segments: np.ndarray) -> tuple:
    n_sp = int(segments.max()) + 1
    h, w = panel_rgb.shape[:2]
    yy, xx = np.mgrid[:h, :w]

    means = np.zeros((n_sp, 3), dtype=np.float32)
    for sp_id in range(n_sp):
        mask = segments == sp_id
        if mask.sum() > 0:
            means[sp_id] = panel_rgb[mask].mean(axis=0)
    return means


def _data_term(sp_means: np.ndarray, seeds_rgb: np.ndarray) -> np.ndarray:
    n_sp = sp_means.shape[0]
    n_labels = seeds_rgb.shape[0]
    sp_lab = color.rgb2lab(sp_means.reshape(n_sp, 1, 3) / 255.0).reshape(n_sp, 3)
    seeds_lab = color.rgb2lab(seeds_rgb.reshape(1, n_labels, 3) / 255.0).reshape(n_labels, 3)
    return np.linalg.norm(sp_lab[:, None, :] - seeds_lab[None, :, :], axis=2)


def _graph_cut_icm(segments: np.ndarray, sp_means: np.ndarray, seeds_rgb: np.ndarray, sigma: float, max_iter: int) -> np.ndarray:
    n_sp = sp_means.shape[0]
    n_labels = seeds_rgb.shape[0]
    data = _data_term(sp_means, seeds_rgb)

    adj = {}
    h, w = segments.shape
    for y in range(h):
        for x in range(w):
            sp = int(segments[y, x])
            adj.setdefault(sp, set())
            for dy, dx in ((0, 1), (1, 0)):
                ny, nx = y + dy, x + dx
                if ny < h and nx < w:
                    nsp = int(segments[ny, nx])
                    if nsp != sp:
                        adj[sp].add(nsp)
                        adj.setdefault(nsp, set()).add(sp)

    sp_lab = color.rgb2lab(sp_means.reshape(n_sp, 1, 3) / 255.0).reshape(n_sp, 3)

    def smoothness(u: int, v: int, lu: int, lv: int) -> float:
        if lu == lv:
            return 0.0
        dist = float(np.linalg.norm(sp_lab[u] - sp_lab[v]))
        return float(np.exp(-(dist ** 2) / (2 * sigma ** 2)) * 5.0)

    labels = np.argmin(data, axis=1)
    changed = True
    iteration = 0
    while changed and iteration < max_iter:
        changed = False
        iteration += 1
        for sp in range(n_sp):
            current = int(labels[sp])
            best_energy = float("inf")
            best_label = current
            for l in range(n_labels):
                e = float(data[sp, l])
                for neighbor in adj.get(sp, []):
                    e += smoothness(sp, int(neighbor), l, int(labels[neighbor]))
                if e < best_energy:
                    best_energy = e
                    best_label = l
            if best_label != current:
                labels[sp] = best_label
                changed = True

    return labels[segments]


def _postprocess(labels: np.ndarray, n_labels: int, min_component_frac: float) -> np.ndarray:
    h, w = labels.shape
    min_area = max(1, int(h * w * min_component_frac))
    cleaned = np.zeros_like(labels)
    for l in range(n_labels):
        mask = labels == l
        if not mask.any():
            continue
        mask_filled = ndimage.binary_fill_holes(mask)
        mask_clean = morphology.remove_small_objects(mask_filled, max_size=min_area)
        cleaned[mask_clean] = l
    return cleaned


def segment(
    panel_rgb: np.ndarray,
    n_layers: int = 7,
    compactness: int = _DEFAULT_COMPACTNESS,
    superpixel_area: int = _DEFAULT_SUPERPIXEL_AREA,
    sigma: float = _DEFAULT_SIGMA,
    min_component_frac: float = _DEFAULT_MIN_COMPONENT_FRAC,
    max_iter: int = _DEFAULT_MAX_ITER,
) -> dict:
    """Segment panel using SLIC + Graph Cut (ICM).

    Args:
        panel_rgb: RGB uint8 array (H, W, 3).
        n_layers: Number of layers to extract.
        compactness: SLIC compactness (lower = more color-driven).
        superpixel_area: Target pixels per superpixel.
        sigma: Graph cut smoothness sigma in Lab space.
        min_component_frac: Minimum component area fraction.
        max_iter: Max ICM iterations.

    Returns:
        dict with keys: labels, seeds, overlay, meta.
    """
    h, w = panel_rgb.shape[:2]
    n_segments = max(n_layers, int(h * w / superpixel_area))

    seeds = extract_seeds(panel_rgb, n_layers)
    segments = _slic_segments(panel_rgb, n_segments, compactness)
    sp_means = _sp_features(panel_rgb, segments)
    labels = _graph_cut_icm(segments, sp_means, seeds, sigma, max_iter)
    labels = _postprocess(labels, n_layers, min_component_frac)

    overlay = _create_overlay(panel_rgb, labels, seeds)

    return {
        "labels": labels,
        "seeds": seeds.tolist(),
        "overlay": overlay,
        "meta": {
            "engine": "e027_slic_graphcut",
            "compactness": compactness,
            "n_superpixels": int(segments.max()) + 1,
            "layers_found": int(len(np.unique(labels))),
        },
    }


def _create_overlay(panel_rgb: np.ndarray, labels: np.ndarray, seeds_rgb: np.ndarray) -> np.ndarray:
    overlay = panel_rgb.copy()
    colors = _distinct_colors(len(seeds_rgb))
    alpha = 0.65
    for l in range(len(colors)):
        mask = labels == l
        if mask.any():
            overlay[mask] = (overlay[mask] * (1 - alpha) + colors[l] * alpha).astype(np.uint8)
    boundaries = segmentation.find_boundaries(labels, mode="thin")
    overlay[boundaries] = [255, 255, 255]
    return overlay
