"""Mask-aware segmentation wrapper.

Runs segmentation engines on the original panel while excluding text pixels
from clustering. Text regions are filled with the nearest non-text color before
the engine runs, then assigned to the nearest non-text label afterwards. This
avoids the color/texture artifacts that inpainting introduces into clustering.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import ndimage


ENGINE_MODULES = {
    "v4_kmeans": "geoseg.modules.segment_engines.v4_kmeans",
    "kmeans_full": "geoseg.modules.segment_engines.kmeans_full",
    "edge_guided": "geoseg.modules.segment_engines.edge_guided",
    "edge_grow": "geoseg.modules.segment_engines.edge_grow",
    "ensemble": "geoseg.modules.segment_engines.ensemble",
    "slic_kmeans": "geoseg.modules.segment_engines.slic_kmeans",
    "grayscale": "geoseg.modules.segment_engines.grayscale",
}


def _fill_text_nearest(image_rgb: np.ndarray, text_mask: np.ndarray) -> np.ndarray:
    """Fill text pixels with the color of the nearest non-text pixel."""
    if not text_mask.any():
        return image_rgb.copy()

    filled = image_rgb.copy()
    non_text_mask = ~text_mask
    if not non_text_mask.any():
        return filled

    # Distance transform indices point to nearest non-text pixel for every pixel.
    _, indices = ndimage.distance_transform_edt(
        ~non_text_mask, return_indices=True
    )
    rr, cc = np.where(text_mask)
    filled[rr, cc] = image_rgb[indices[0][rr, cc], indices[1][rr, cc]]
    return filled


def _assign_text_to_nearest_label(
    labels: np.ndarray, text_mask: np.ndarray
) -> np.ndarray:
    """Assign text pixels to the nearest non-zero/non-text label."""
    if not text_mask.any():
        return labels.copy()

    cleaned = labels.copy()
    valid_mask = (~text_mask) & (labels != 0)
    if not valid_mask.any():
        return cleaned

    _, indices = ndimage.distance_transform_edt(~valid_mask, return_indices=True)
    rr, cc = np.where(text_mask)
    cleaned[rr, cc] = labels[indices[0][rr, cc], indices[1][rr, cc]]
    return cleaned


def segment_with_text_mask(
    engine_name: str,
    image_rgb: np.ndarray,
    text_mask: np.ndarray,
    n_layers: int,
    **engine_kwargs,
) -> dict:
    """Run a segmentation engine with text pixels excluded from clustering.

    Args:
        engine_name: One of the supported engine names (v4_kmeans, ensemble, etc.).
        image_rgb: Original RGB panel.
        text_mask: Boolean mask where True = text pixel to ignore.
        n_layers: Target layer count.
        **engine_kwargs: Extra arguments forwarded to the engine's segment().

    Returns:
        Engine result dict with `labels` updated so text pixels belong to the
        nearest non-text label. `overlay` is regenerated on `image_rgb`.
    """
    module_path = ENGINE_MODULES.get(engine_name)
    if module_path is None:
        raise ValueError(f"Unsupported engine for mask-aware segmentation: {engine_name}")

    import importlib

    mod = importlib.import_module(module_path)
    segment_fn: Callable = getattr(mod, "segment")

    filled = _fill_text_nearest(image_rgb, text_mask)
    result = segment_fn(filled, n_layers=n_layers, **engine_kwargs)

    labels = _assign_text_to_nearest_label(result["labels"], text_mask)
    result["labels"] = labels

    # Regenerate overlay on the original image so text artifacts don't leak in.
    from geoseg.modules.segment_engines.internal.overlay import _create_overlay

    seeds = np.array(result.get("seeds", []), dtype=np.uint8)
    result["overlay"] = _create_overlay(image_rgb, labels, seeds)
    return result


def overlay_with_text_suppressed(
    image_rgb: np.ndarray,
    labels: np.ndarray,
    text_mask: np.ndarray,
    alpha: float = 0.65,
) -> np.ndarray:
    """Create overlay on the original panel but suppress text regions.

    Text pixels are filled with the nearest non-text color from the original
    panel before the label overlay is applied. This hides annotation text
    without relying on imperfect inpainting.
    """
    from geoseg.modules.segment_engines.regional_fusion import generate_overlay_with_legend

    background = _fill_text_nearest(image_rgb, text_mask)
    overlay = generate_overlay_with_legend(background, labels, alpha=alpha)
    return overlay

def regional_segment_with_text_mask(
    image_rgb: np.ndarray,
    n_layers: int,
    text_mask: np.ndarray,
    *,
    reps: list[dict] | None = None,
    colorbar_rgb: np.ndarray | None = None,
    primary_result: dict | None = None,
    audit=None,
    config=None,
) -> dict:
    """Region-level fusion with text-mask awareness.

    Behaves like geoseg.modules.segment_engines.regional_fusion.regional_segment
    but fills text pixels with nearest non-text color before each engine run and
    assigns text pixels to the nearest non-text label afterwards.
    """
    from geoseg.modules.segment_engines.regional_fusion import (
        FusionConfig,
        RegionalAudit,
        regional_segment,
    )

    filled = _fill_text_nearest(image_rgb, text_mask)

    # Audit may contain local_fixes that reference the original labels.
    result = regional_segment(
        filled,
        n_layers,
        reps=reps,
        colorbar_rgb=colorbar_rgb,
        primary_result=primary_result,
        audit=audit,
        config=config,
    )

    result["labels"] = _assign_text_to_nearest_label(result["labels"], text_mask)
    from geoseg.modules.segment_engines.internal.overlay import _create_overlay

    seeds = np.array(result.get("seeds", []), dtype=np.uint8)
    result["overlay"] = _create_overlay(image_rgb, result["labels"], seeds)
    return result
