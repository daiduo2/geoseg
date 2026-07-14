"""SLIC superpixel + K-Means segmentation engine.

Text-robust alternative engine for panels with heavy annotation overlays.
SLIC groups pixels into perceptually-homogeneous superpixels; text regions
are typically smaller than a single superpixel and get absorbed naturally.
Superpixel-level K-Means then clusters superpixels into geological layers.

Trade-off: extremely low fragment count (near zero), but may under-segment
adjacent layers with similar colors.  Best used when fragment intolerance
outweighs exact layer count (e.g. agent sandbox retry).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.cluster.vq import kmeans2
from skimage.color import rgb2lab
from skimage.segmentation import slic

from geoseg.modules.segment_engines.internal.color import _distinct_colors
from geoseg.modules.segment_engines.internal.overlay import _create_overlay
from geoseg.modules.segment_engines.internal.regions import _reorder_labels_by_median_y


def segment(
    panel_rgb: np.ndarray,
    n_layers: int = 5,
    n_segments: int = 500,
    compactness: float = 10.0,
    median_post_size: int = 5,
    **kwargs: object,
) -> dict:
    """Segment panel using SLIC superpixels + superpixel-level K-Means.

    Args:
        panel_rgb: RGB uint8 array (H, W, 3).
        n_layers: Number of geological layers to extract.
        n_segments: Target number of SLIC superpixels.
        compactness: SLIC compactness (higher = more regular shapes).
        median_post_size: Median filter size on final labels (0 to disable).
        **kwargs: Absorbed for API compatibility.

    Returns:
        dict with keys: labels, overlay, meta.
    """
    h, w = panel_rgb.shape[:2]

    segments = slic(
        panel_rgb,
        n_segments=n_segments,
        compactness=compactness,
        channel_axis=2,
        start_label=0,
    )
    n_sp = int(segments.max()) + 1

    # Mean RGB per superpixel
    sp_means = np.zeros((n_sp, 3), dtype=np.float64)
    for sp_id in range(n_sp):
        mask = segments == sp_id
        if mask.any():
            sp_means[sp_id] = panel_rgb[mask].mean(axis=0)

    # K-Means on superpixel means
    centroids, sp_labels = kmeans2(sp_means, n_layers, minit="++", seed=42)

    # Map superpixel labels back to pixel labels
    labels = sp_labels[segments].astype(np.int32)

    # Reorder by median y (top to bottom)
    labels = _reorder_labels_by_median_y(labels)

    # Optional median postprocessing
    if median_post_size > 1:
        labels = ndimage.median_filter(labels, size=median_post_size)

    # Palette from mean color of each final label
    palette = np.zeros((n_layers, 3), dtype=np.uint8)
    for lbl in range(n_layers):
        mask = labels == lbl
        if mask.any():
            palette[lbl] = panel_rgb[mask].mean(axis=0).astype(np.uint8)
        else:
            palette[lbl] = _distinct_colors(n_layers)[lbl]

    overlay = _create_overlay(panel_rgb, labels, palette)

    return {
        "labels": labels,
        "overlay": overlay,
        "meta": {
            "engine": "slic_kmeans",
            "n_layers": n_layers,
            "n_superpixels": n_sp,
            "compactness": compactness,
            "median_post_size": median_post_size,
        },
    }
