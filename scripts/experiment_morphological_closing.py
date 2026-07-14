"""Experiment: morphological closing on 16b0cf fixture.

Per-layer closing with overlap resolution by nearest median y.
Tests radii r=2,5,10,15.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from imageio.v3 import imread
from scipy.cluster.vq import kmeans2
from scipy import ndimage
from skimage.morphology import disk, closing
from skimage.transform import resize

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geoseg.modules.segment_engines.horizon_refinement import (
    _extract_boundary_points,
    _fit_quintic,
    _repartition_columns,
    _compute_fragmentation_score,
)
from geoseg.core.image_ops import create_overlay


def kmeans_segment(img: np.ndarray, n_layers: int) -> np.ndarray:
    h, w = img.shape[:2]
    pixels = img.reshape(-1, 3).astype(np.float64)
    centroids, labels_flat = kmeans2(pixels, n_layers, minit="++", seed=42)
    return labels_flat.reshape(h, w).astype(np.int32)


def apply_morphological_closing(
    labels: np.ndarray, radius: int
) -> np.ndarray:
    """Apply closing per-layer, resolve overlaps by nearest median y."""
    h, w = labels.shape
    unique = sorted(u for u in np.unique(labels) if u >= 0)
    layer_labels = [u for u in unique if u != 0]

    # Compute median y for each layer
    median_ys: dict[int, float] = {}
    for lbl in layer_labels:
        ys = np.where(labels == lbl)[0]
        median_ys[lbl] = float(np.median(ys)) if len(ys) > 0 else h

    closed_masks: dict[int, np.ndarray] = {}
    for lbl in layer_labels:
        mask = (labels == lbl).astype(np.uint8)
        if radius > 0:
            closed = closing(mask, footprint=disk(radius))
        else:
            closed = mask.astype(bool)
        closed_masks[lbl] = closed

    # Resolve overlaps: for each pixel, choose label with closest median y
    result = np.zeros_like(labels)
    for y in range(h):
        for x in range(w):
            active = [(lbl, abs(y - median_ys[lbl])) for lbl in layer_labels if closed_masks[lbl][y, x]]
            if active:
                result[y, x] = min(active, key=lambda t: t[1])[0]
            else:
                # Fallback: keep original or background
                result[y, x] = labels[y, x]

    return result


def extract_boundaries_and_repartition(
    panel_rgb: np.ndarray,
    closed_labels: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Extract boundaries via label-blur zero-crossing, fit quintic, repartition."""
    h, w = panel_rgb.shape[:2]
    unique = sorted(u for u in np.unique(closed_labels) if u >= 0)
    layer_labels = [u for u in unique if u != 0]

    if not layer_labels:
        return closed_labels.copy(), []

    median_ys = {}
    for lbl in layer_labels:
        ys = np.where(closed_labels == lbl)[0]
        median_ys[lbl] = float(np.median(ys)) if len(ys) > 0 else h

    spatial_order = sorted(layer_labels, key=lambda lbl: median_ys[lbl])

    boundaries = []
    for i in range(len(spatial_order) - 1):
        top_lbl = spatial_order[i]
        bot_lbl = spatial_order[i + 1]
        pts = _extract_boundary_points(panel_rgb, closed_labels, top_lbl, bot_lbl)
        if pts is None:
            continue
        xs, ys = pts
        boundary_y = _fit_quintic(ys, smoothness=0.5)
        # Fill to full width
        full_y = np.full(w, np.nan, dtype=np.float32)
        full_y[xs] = boundary_y
        valid = ~np.isnan(full_y)
        if valid.any() and not valid.all():
            full_y[~valid] = np.interp(np.where(~valid)[0], np.where(valid)[0], full_y[valid])
        boundaries.append(full_y)

    if not boundaries:
        return closed_labels.copy(), []

    # Enforce monotonicity
    if len(boundaries) > 1:
        medians = [float(np.median(b)) for b in boundaries]
        order = np.argsort(medians)
        boundaries = [boundaries[int(i)] for i in order]

    repartitioned = _repartition_columns(closed_labels, spatial_order, boundaries)
    return repartitioned, boundaries


def run_experiment():
    img_path = Path("/Users/daiduo2/geoseg/runs/readme_examples_v2/gras2019_16b0cf/panel_cropped.png")
    out_dir = Path("/Users/daiduo2/geoseg/runs/horizon_refine/experiment_morpho")
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_rgb = imread(img_path)
    if panel_rgb.ndim == 2:
        panel_rgb = np.stack([panel_rgb] * 3, axis=-1)
    panel_rgb = panel_rgb[:, :, :3].astype(np.uint8)

    # Baseline k-means
    coarse = kmeans_segment(panel_rgb, n_layers=5)
    baseline_frag = _compute_fragmentation_score(coarse)
    print(f"Baseline (no closing) fragmentation: {baseline_frag:.4f}")

    # Save baseline overlay
    baseline_overlay = create_overlay(panel_rgb, coarse, np.empty((0, 3), dtype=np.uint8))
    from imageio.v3 import imwrite
    imwrite(out_dir / "r_0.png", baseline_overlay)

    radii = [2, 5, 10, 15]
    results = []

    for r in radii:
        closed = apply_morphological_closing(coarse, radius=r)
        closed_frag = _compute_fragmentation_score(closed)

        # Check if previously non-touching layers now touch
        unique = sorted(u for u in np.unique(coarse) if u >= 0)
        layer_labels = [u for u in unique if u != 0]
        median_ys = {lbl: float(np.median(np.where(coarse == lbl)[0])) for lbl in layer_labels}
        spatial_order = sorted(layer_labels, key=lambda lbl: median_ys[lbl])

        newly_touching = 0
        for i in range(len(spatial_order) - 1):
            top_lbl = spatial_order[i]
            bot_lbl = spatial_order[i + 1]
            # Baseline touch check
            mask_top_b = coarse == top_lbl
            mask_bot_b = coarse == bot_lbl
            touch_b = mask_top_b & ndimage.binary_dilation(mask_bot_b, iterations=1)
            was_touching = np.sum(touch_b) > 0

            mask_top_c = closed == top_lbl
            mask_bot_c = closed == bot_lbl
            touch_c = mask_top_c & ndimage.binary_dilation(mask_bot_c, iterations=1)
            now_touching = np.sum(touch_c) > 0

            if not was_touching and now_touching:
                newly_touching += 1

        repartitioned, boundaries = extract_boundaries_and_repartition(panel_rgb, closed)
        repartitioned_frag = _compute_fragmentation_score(repartitioned)

        # Save both closed-only and repartitioned overlays
        closed_overlay = create_overlay(panel_rgb, closed, np.empty((0, 3), dtype=np.uint8))
        imwrite(out_dir / f"r_{r}_closed.png", closed_overlay)
        overlay = create_overlay(panel_rgb, repartitioned, np.empty((0, 3), dtype=np.uint8))
        imwrite(out_dir / f"r_{r}.png", overlay)

        change_ratio = float(np.sum(repartitioned != closed) / closed.size)

        results.append({
            "radius": r,
            "closed_frag": closed_frag,
            "repartitioned_frag": repartitioned_frag,
            "newly_touching": newly_touching,
            "n_boundaries": len(boundaries),
            "change_ratio": change_ratio,
        })
        print(
            f"r={r:2d}: closed_frag={closed_frag:.4f}, repartitioned_frag={repartitioned_frag:.4f}, "
            f"newly_touching={newly_touching}, n_boundaries={len(boundaries)}, change={change_ratio:.4f}"
        )

    print("\nSummary:")
    print(f"{'Radius':>8} {'Closed Frag':>12} {'Repart Frag':>12} {'New Touch':>10} {'Boundaries':>10} {'Change':>10}")
    for res in results:
        print(
            f"{res['radius']:8d} {res['closed_frag']:12.4f} {res['repartitioned_frag']:12.4f} "
            f"{res['newly_touching']:10d} {res['n_boundaries']:10d} {res['change_ratio']:10.4f}"
        )


if __name__ == "__main__":
    run_experiment()
