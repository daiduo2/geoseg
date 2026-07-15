"""Public horizon refinement orchestration."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import ndimage

from geoseg.core.models import SegmentationResult
from geoseg.modules.segment_engines.horizon.boundaries import (
    _adjust_boundaries,
    _extract_boundary_points,
)
from geoseg.modules.segment_engines.horizon.coarse import _coarse_segment, _separator_mask
from geoseg.modules.segment_engines.horizon.fitting import _fit_curve
from geoseg.modules.segment_engines.internal.overlay import _create_overlay


def _compute_fragmentation_score(labels: np.ndarray) -> float:
    """Compute a fragmentation score: fraction of pixels in tiny components."""
    from scipy import ndimage
    total_tiny = 0
    h, w = labels.shape
    min_area = max(50, int(h * w * 0.001))
    for lbl in np.unique(labels):
        mask = labels == lbl
        if not mask.any():
            continue
        labeled, num = ndimage.label(mask)
        if num <= 1:
            continue
        sizes = ndimage.sum(mask, labeled, range(1, num + 1))
        tiny = np.sum(sizes[sizes < min_area])
        total_tiny += int(tiny)
    return total_tiny / (h * w)


def refine_label_blur(coarse_labels: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    """Refine by spatial Gaussian smoothing in label space.

    For each unique label, create a binary mask, apply 2D Gaussian blur,
    then re-assign each pixel to the label with the highest blurred value.
    Small fragments get smoothed away because they have low spatial support.
    This produces visually smooth, geologically plausible layers without
    explicit boundary extraction or curve fitting.
    """
    separator_mask = _separator_mask(coarse_labels)
    unique = sorted(u for u in np.unique(coarse_labels) if u > 0 or not separator_mask.any())
    if len(unique) < 2:
        return coarse_labels.copy()

    prob_maps = []
    for lbl in unique:
        mask = (coarse_labels == lbl).astype(np.float32)
        prob = ndimage.gaussian_filter(mask, sigma=sigma)
        prob_maps.append(prob)

    prob_stack = np.stack(prob_maps, axis=0)
    result = np.array(unique)[np.argmax(prob_stack, axis=0)]
    result[separator_mask] = 0
    return result.astype(coarse_labels.dtype)


def refine_boundaries(
    panel_rgb: np.ndarray,
    coarse_labels: np.ndarray | None = None,
    n_layers: int | None = None,
    method: Literal["savgol", "bspline", "loess", "quintic", "knot_constrained"] = "savgol",
    smoothness: float = 1.0,
    blur_sigma: float = 2.0,
    downsample_factor: float = 0.25,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Refine fragmented segmentation by fitting smooth horizons.

    Two strategies:
    - Touching layer pairs: local boundary adjustment via label-blur
      zero-crossing + curve fitting + pixel relabeling near boundaries.
    - Non-touching (broken) layer pairs: label-space Gaussian blur.
      Each label mask is blurred in 2D and pixels are reassigned to the
      dominant label. This naturally eliminates small fragments and produces
      smooth, geologically plausible layers without explicit curve fitting.

    Key invariant: original coarse label IDs and global layer ordering are
    preserved. For touching layer pairs, only boundary-adjacent pixels are
    adjusted. For non-touching (severely fragmented) pairs, label-space blur
    redraws the border between archipelagos.

    Args:
        panel_rgb: Original RGB image (H, W, 3) uint8.
        coarse_labels: Initial label map from any engine. If None, computed
            internally via Phase A using n_layers.
        n_layers: Target layer count. Required if coarse_labels is None.
        method: Curve fitting method. "knot_constrained" detects significant
            local extrema (knots) and fits a weighted spline that preserves
            geological features while smoothing sawtooth noise. "quintic" uses
            curvature-variation prior.
        smoothness: Smoothness factor. Interpretation varies by method:
            - savgol: window_length = int(smoothness * W)
            - bspline: s = smoothness * 1e4 scale factor
            - loess: frac = smoothness * 0.1
            - quintic: s = smoothness * 1e6 (UnivariateSpline smoothing)
        blur_sigma: Gaussian blur sigma for Phase A coarse segmentation.
        downsample_factor: Downsample ratio for Phase A.

    Returns:
        refined_labels: Label map with smoothed boundaries (H, W).
        boundaries: List of y-coordinate arrays for each fitted horizon.
    """
    h, w = panel_rgb.shape[:2]

    if coarse_labels is None:
        if n_layers is None:
            raise ValueError("n_layers required when coarse_labels is None")
        coarse_labels = _coarse_segment(
            panel_rgb, n_layers, blur_sigma, downsample_factor
        )

    sep_mask = _separator_mask(coarse_labels)
    is_sep = sep_mask.any()

    unique = sorted(u for u in np.unique(coarse_labels) if u >= 0)
    if len(unique) < 2:
        return coarse_labels.copy(), []

    # --- Spatially order labels (top-to-bottom by median y) ---
    # Exclude label 0 only when it is a separator (editor topology).
    layer_labels = [u for u in unique if u != 0 or not is_sep]

    if not layer_labels:
        return coarse_labels.copy(), []

    median_ys: dict[int, float] = {}
    for lbl in layer_labels:
        ys = np.where(coarse_labels == lbl)[0]
        median_ys[lbl] = float(np.median(ys)) if len(ys) > 0 else h

    spatial_order = sorted(layer_labels, key=lambda lbl: median_ys[lbl])

    # --- Detect broken (non-touching) pairs ---
    broken_pairs: set[tuple[int, int]] = set()
    for i in range(len(spatial_order) - 1):
        top_lbl = spatial_order[i]
        bot_lbl = spatial_order[i + 1]
        touch = (coarse_labels == top_lbl) & ndimage.binary_dilation(
            coarse_labels == bot_lbl, iterations=1
        )
        if np.sum(touch) == 0:
            broken_pairs.add((top_lbl, bot_lbl))

    # --- Strategy dispatch ---
    if broken_pairs:
        # Severely fragmented: label-space Gaussian blur.
        # Directly smooths the label map in 2D, letting spatial competition
        # naturally eliminate small fragments. This produces visually smooth,
        # geologically plausible layers without explicit curve fitting.
        refined_labels = refine_label_blur(coarse_labels, sigma=15.0)
        boundaries: list[np.ndarray] = []

        # Quality gate: fragmentation must improve
        coarse_frag = _compute_fragmentation_score(coarse_labels)
        refined_frag = _compute_fragmentation_score(refined_labels)
        if refined_frag > coarse_frag * 1.5:
            return coarse_labels.copy(), []

        # Quality gate: layer count preservation (allow 1 loss due to merge)
        refined_unique = sorted(u for u in np.unique(refined_labels) if u >= 0)
        if len(refined_unique) < len(layer_labels) - 1:
            return coarse_labels.copy(), []

        return refined_labels, boundaries

    # --- Touching pairs: local boundary adjustment via curve fitting ---
    boundaries: list[np.ndarray] = []
    boundary_pairs: list[tuple[int, int]] = []

    for i in range(len(spatial_order) - 1):
        top_lbl = spatial_order[i]
        bot_lbl = spatial_order[i + 1]

        points = _extract_boundary_points(panel_rgb, coarse_labels, top_lbl, bot_lbl)
        if points is None:
            continue
        xs, ys = points
        boundary_y = _fit_curve(xs, ys, method, smoothness)

        # Fill gaps
        full_y = np.full(w, np.nan, dtype=np.float32)
        full_y[xs] = boundary_y
        full_y = ndimage.generic_filter(
            full_y, lambda v: np.nanmedian(v) if np.any(~np.isnan(v)) else h // 2,
            size=11, mode="nearest"
        )
        nan_mask = np.isnan(full_y)
        if nan_mask.any() and not nan_mask.all():
            full_y[nan_mask] = np.interp(
                np.where(nan_mask)[0],
                np.where(~nan_mask)[0],
                full_y[~nan_mask],
            )
        boundary_y = full_y

        boundaries.append(boundary_y)
        boundary_pairs.append((top_lbl, bot_lbl))

    if not boundaries:
        return coarse_labels.copy(), []

    # --- Enforce monotonicity: sort boundaries by median y ---
    if len(boundaries) > 1:
        medians = [float(np.median(b)) for b in boundaries]
        order = np.argsort(medians)
        boundaries = [boundaries[int(i)] for i in order]
        boundary_pairs = [boundary_pairs[int(i)] for i in order]

    # --- Sanity check: minimum boundary separation ---
    min_layer_height = max(3, h // 100)
    if len(boundaries) > 1:
        for i in range(len(boundaries) - 1):
            gap = np.median(boundaries[i + 1]) - np.median(boundaries[i])
            if gap < min_layer_height:
                return coarse_labels.copy(), boundaries

    refined_labels = _adjust_boundaries(coarse_labels, boundaries, boundary_pairs, blend_width=5)

    # --- Quality gates for touching pairs ---
    pixel_change_ratio = np.sum(refined_labels != coarse_labels) / (h * w)
    if pixel_change_ratio > 0.15:
        return coarse_labels.copy(), boundaries

    refined_unique = sorted(u for u in np.unique(refined_labels) if u >= 0)
    if len(refined_unique) < len(layer_labels) - 1:
        return coarse_labels.copy(), boundaries

    coarse_frag = _compute_fragmentation_score(coarse_labels)
    refined_frag = _compute_fragmentation_score(refined_labels)
    if refined_frag > coarse_frag * 1.2:
        return coarse_labels.copy(), boundaries

    return refined_labels, boundaries


def segment(
    panel_rgb: np.ndarray,
    *,
    n_layers: int = 5,
    coarse_labels: np.ndarray | None = None,
    **kwargs: object,
) -> SegmentationResult:
    """Protocol-compatible entry point for horizon refinement.

    Wraps ``refine_boundaries()`` into the standard ``SegmentationResult``
    dict expected by the pipeline: ``{"labels", "overlay", "meta"}``.

    Args:
        panel_rgb: RGB uint8 array (H, W, 3).
        n_layers: Target layer count. Required when ``coarse_labels`` is None.
        coarse_labels: Initial label map from any upstream engine.
            If None, a coarse segmentation is computed internally.
        **kwargs: Absorbed for API compatibility (e.g. ``reps``, ``colorbar_rgb``).

    Returns:
        dict with keys: labels, overlay, meta.
    """
    labels, boundaries = refine_boundaries(
        panel_rgb,
        coarse_labels=coarse_labels,
        n_layers=n_layers,
    )

    # Build overlay with auto-generated distinct colors
    unique = sorted(set(labels.flatten()) - {0})
    if len(unique) > 0:
        palette = np.zeros((len(unique), 3), dtype=np.uint8)
        for i, lbl in enumerate(unique):
            mask = labels == lbl
            if mask.any():
                palette[i] = panel_rgb[mask].mean(axis=0).astype(np.uint8)
        # Pad palette so _create_overlay can size from it
        full_palette = np.zeros((len(unique) + 1, 3), dtype=np.uint8)
        full_palette[1:] = palette
    else:
        full_palette = np.zeros((1, 3), dtype=np.uint8)

    overlay = _create_overlay(panel_rgb, labels, full_palette)

    return {
        "labels": labels,
        "overlay": overlay,
        "meta": {
            "engine": "horizon_refinement",
            "n_layers": len(unique),
            "boundaries": [b.tolist() for b in boundaries] if boundaries else [],
        },
    }

__all__ = [
    "_compute_fragmentation_score",
    "refine_boundaries",
    "refine_label_blur",
    "segment",
]
