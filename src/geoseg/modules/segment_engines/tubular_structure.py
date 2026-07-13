"""Tubular structure detection via Frangi vesselness filter.

Targets 3D rendered scientific schematics with thin tubular structures
embedded in smooth gradient backgrounds (e.g. geophysics plume diagrams).

The Frangi filter computes Hessian eigenvalues at each pixel and scores
"tubeness" based on the eigenvalue signature. Smooth gradients produce
low vesselness scores (ignored), while tube walls produce the characteristic
|λ1| ≈ 0, |λ2| << 0 signature (detected).
"""
from __future__ import annotations

import numpy as np
from skimage.color import rgb2lab, rgb2gray
from skimage.filters import frangi, threshold_otsu, threshold_yen
from skimage.morphology import remove_small_objects, remove_small_holes

from geoseg.modules.segment_engines.internal.shared import (
    _create_overlay,
    _estimate_background_color,
)


def _select_contrast_channel(
    panel_rgb: np.ndarray, bg_rgb: np.ndarray
) -> np.ndarray:
    """Select the grayscale channel that maximises tube-to-background contrast."""
    gray = rgb2gray(panel_rgb)
    lab = rgb2lab(panel_rgb)
    l_channel = lab[..., 0] / 100.0
    bg_dist = np.linalg.norm(
        panel_rgb.astype(np.float32) - bg_rgb.astype(np.float32), axis=2
    )
    bg_dist = bg_dist / (bg_dist.max() + 1e-9)

    candidates = {
        "gray": gray,
        "lab_l": l_channel,
        "bg_dist": bg_dist,
    }
    best = max(candidates, key=lambda k: candidates[k].std())
    return candidates[best]


def _frangi_segment(
    panel_rgb: np.ndarray,
    sigmas: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0),
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float | None = None,
    black_ridges: bool | None = None,
    min_tube_area_frac: float = 0.001,
) -> tuple[np.ndarray, np.ndarray]:
    """Segment tubular structure using multi-scale Frangi filter.

    Returns (labels, vesselness_map).  labels has tube=1, background=0.
    """
    h, w = panel_rgb.shape[:2]
    bg_rgb = _estimate_background_color(panel_rgb)

    img = _select_contrast_channel(panel_rgb, bg_rgb)

    if black_ridges is None:
        tube_center = img[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3].mean()
        bg_corner = np.array(
            [img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]]
        ).mean()
        black_ridges = tube_center < bg_corner

    vesselness = frangi(
        img,
        sigmas=sigmas,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        black_ridges=black_ridges,
    )

    # Yen works better for low-contrast; fallback to Otsu
    positive = vesselness[vesselness > 0]
    if len(positive) == 0:
        return np.zeros((h, w), dtype=np.int32), vesselness

    try:
        thresh = threshold_yen(positive)
    except Exception:
        thresh = threshold_otsu(positive)

    mask = vesselness > thresh

    min_area = max(30, int(h * w * min_tube_area_frac))
    mask = remove_small_objects(mask, max_size=min_area - 1)
    mask = remove_small_holes(mask, max_size=min_area * 2 - 1)

    labels = mask.astype(np.int32)
    return labels, vesselness


def segment(
    panel_rgb: np.ndarray,
    reps: list[dict] | None = None,
    n_layers: int = 2,
    **kwargs: object,
) -> dict:
    """Segmenter Protocol entry for tubular structure detection.

    Args:
        panel_rgb: RGB uint8 array (H, W, 3).
        reps: Accepted for Protocol compatibility but ignored; the tube is
            found structurally via Hessian analysis.
        n_layers: If 2, returns tube vs background.  If >2, the background
            region is further subdivided via v4_kmeans.

    Returns:
        dict with keys: labels, overlay, meta.
    """
    labels, vesselness = _frangi_segment(panel_rgb, **kwargs)

    if n_layers > 2 and labels.max() > 0:
        from geoseg.modules.segment_engines.v4_kmeans import segment as v4_segment

        tube_mask = labels > 0
        masked_rgb = panel_rgb.copy()
        masked_rgb[tube_mask] = [0, 0, 0]
        bg_labels = v4_segment(masked_rgb, n_layers=n_layers - 1)["labels"]
        # Offset background labels to avoid collision with tube=1
        bg_labels = bg_labels + 1
        bg_labels[tube_mask] = 1
        labels = bg_labels

    unique = sorted(np.unique(labels))
    palette = np.zeros((max(unique) + 1, 3), dtype=np.uint8)
    for lbl in unique:
        mask = labels == lbl
        if mask.any():
            palette[lbl] = panel_rgb[mask].mean(axis=0).astype(np.uint8)

    overlay = _create_overlay(
        panel_rgb, labels, seeds_rgb=palette, overlay_colors=palette
    )

    return {
        "labels": labels,
        "overlay": overlay,
        "meta": {
            "engine": "tubular_structure",
            "n_layers": len(unique),
            "frangi_params": {
                "sigmas": list(
                    kwargs.get("sigmas", (1.0, 2.0, 3.0, 4.0))
                ),
                "alpha": kwargs.get("alpha", 0.5),
                "beta": kwargs.get("beta", 0.5),
                "gamma": kwargs.get("gamma"),
                "black_ridges": kwargs.get("black_ridges"),
            },
            "vesselness_stats": {
                "min": float(vesselness.min()),
                "max": float(vesselness.max()),
                "mean": float(vesselness.mean()),
            },
        },
    }
