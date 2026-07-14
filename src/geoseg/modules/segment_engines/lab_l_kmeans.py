"""LAB L-channel K-Means segmentation.

Useful for panels where a feature (e.g. a mantle plume funnel) has the same
hue as a gradient background but differs in lightness. Clustering in the L*
channel suppresses hue/saturation distractions and separates regions by
brightness.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from sklearn.cluster import KMeans

from geoseg.modules.segment_engines.internal.overlay import _create_overlay


def segment(
    panel_rgb: np.ndarray,
    n_layers: int = 5,
    reps: list[dict] | None = None,
) -> dict:
    """Segment a panel using K-Means on the LAB L* channel.

    Args:
        panel_rgb: RGB uint8 array.
        n_layers: Number of layers to extract.
        reps: Ignored for this engine (kept for protocol compatibility).

    Returns:
        dict with keys: labels, seeds, overlay, meta.
    """
    if panel_rgb.ndim != 3 or panel_rgb.shape[2] != 3:
        raise ValueError("panel_rgb must be an RGB image")

    lab = rgb2lab(panel_rgb)
    L = lab[:, :, 0]
    h, w = L.shape

    target_k = max(2, int(n_layers))
    samples = L.reshape(-1, 1)

    kmeans = KMeans(
        n_clusters=target_k,
        random_state=0,
        n_init=10,
        max_iter=300,
    ).fit(samples)

    labels = kmeans.labels_.reshape(h, w).astype(np.int32)

    # Renumber labels compactly starting from 1 (0 reserved for background).
    present = sorted(np.unique(labels[labels > 0]))
    renum = {old: new + 1 for new, old in enumerate(present)}
    clean = np.zeros_like(labels)
    for old, new in renum.items():
        clean[labels == old] = new
    labels = clean

    # Compute seed colors as median RGB of each label.
    n_actual = int(labels.max())
    seeds_rgb = np.zeros((n_actual + 1, 3), dtype=np.uint8)
    for lbl in range(1, n_actual + 1):
        mask = labels == lbl
        if mask.any():
            seeds_rgb[lbl] = np.median(panel_rgb[mask], axis=0).astype(np.uint8)
        else:
            seeds_rgb[lbl] = 128

    overlay = _create_overlay(panel_rgb, labels, seeds_rgb)

    return {
        "labels": labels,
        "seeds": seeds_rgb.tolist(),
        "overlay": overlay,
        "meta": {
            "engine": "lab_l_kmeans",
            "path": "lab_l_kmeans",
            "n_layers_target": target_k,
            "n_layers_actual": n_actual,
        },
    }


def segment_from_path(panel_path: str, n_layers: int = 5) -> dict:
    """Convenience wrapper for file paths."""
    img = np.array(Image.open(panel_path).convert("RGB"))
    return segment(img, n_layers=n_layers)
