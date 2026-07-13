"""Debug: check which refinement path is taken."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoseg.modules.segment_engines.kmeans_full import segment as seg_kmeans
from geoseg.modules.segment_engines.horizon_refinement import (
    _compute_fragmentation_score,
    _extract_boundary_points,
    _extract_boundary_dense,
    _fit_curve,
    _fit_savgol,
    _repartition_columns,
    _adjust_boundaries,
)


def spatial_relabel(labels: np.ndarray, threshold: float) -> np.ndarray:
    from scipy import ndimage
    h, w = labels.shape
    result = labels.copy()
    unique = sorted(u for u in np.unique(labels) if u != 0)
    peaks = {}
    for lbl in unique:
        ys = np.where(labels == lbl)[0]
        peaks[lbl] = float(np.median(ys)) if len(ys) > 0 else h / 2

    for lbl in unique:
        mask = labels == lbl
        labeled, num = ndimage.label(mask)
        if num <= 1:
            continue
        for frag_id in range(1, num + 1):
            frag_mask = labeled == frag_id
            frag_ys = np.where(frag_mask)[0]
            if len(frag_ys) == 0:
                continue
            frag_median = float(np.median(frag_ys))
            peak = peaks[lbl]
            if abs(frag_median - peak) > threshold:
                best_lbl = lbl
                best_dist = abs(frag_median - peak)
                for other_lbl in unique:
                    if other_lbl == lbl:
                        continue
                    dist = abs(frag_median - peaks[other_lbl])
                    if dist < best_dist:
                        best_dist = dist
                        best_lbl = other_lbl
                result[frag_mask] = best_lbl
    return result


def refine_debug(panel_rgb, coarse_labels, method="savgol", smoothness=1.0):
    """Copy of refine_boundaries with debug prints."""
    h, w = panel_rgb.shape[:2]
    unique = sorted(u for u in np.unique(coarse_labels) if u >= 0)
    layer_labels = [u for u in unique if u != 0]
    median_ys = {}
    for lbl in layer_labels:
        ys = np.where(coarse_labels == lbl)[0]
        median_ys[lbl] = float(np.median(ys)) if len(ys) > 0 else h
    spatial_order = sorted(layer_labels, key=lambda lbl: median_ys[lbl])
    print(f"  Spatial order: {spatial_order}")

    boundaries = []
    boundary_pairs = []
    broken_pairs = set()

    for i in range(len(spatial_order) - 1):
        top_lbl = spatial_order[i]
        bot_lbl = spatial_order[i + 1]
        mask_top = coarse_labels == top_lbl
        mask_bot = coarse_labels == bot_lbl
        touch = mask_top & ndimage.binary_dilation(mask_bot, iterations=1)
        is_broken = int(np.sum(touch)) == 0
        print(f"  Pair ({top_lbl},{bot_lbl}): touch={int(np.sum(touch))}, broken={is_broken}")

        if is_broken:
            ys_raw = _extract_boundary_dense(coarse_labels, top_lbl, bot_lbl)
            n_valid = int(np.sum(~np.isnan(ys_raw))) if ys_raw is not None else 0
            print(f"    -> dense path, valid_points={n_valid}")
            if ys_raw is None or n_valid < 10:
                print(f"    -> SKIPPED (too few points)")
                continue
            ys_filled = ys_raw.copy()
            valid = ~np.isnan(ys_raw)
            if valid.any() and not valid.all():
                ys_filled[~valid] = np.interp(np.where(~valid)[0], np.where(valid)[0], ys_raw[valid])
            boundary_y = _fit_savgol(np.arange(len(ys_filled)), ys_filled, smoothness=0.15)
            broken_pairs.add((top_lbl, bot_lbl))
        else:
            points = _extract_boundary_points(panel_rgb, coarse_labels, top_lbl, bot_lbl)
            print(f"    -> zero-crossing path, points={points is not None}")
            if points is None:
                print(f"    -> SKIPPED (extraction failed)")
                continue
            xs, ys = points
            boundary_y = _fit_curve(xs, ys, method, smoothness)
            full_y = np.full(w, np.nan, dtype=np.float32)
            full_y[xs] = boundary_y
            full_y = ndimage.generic_filter(
                full_y, lambda v: np.nanmedian(v) if np.any(~np.isnan(v)) else h // 2,
                size=11, mode="nearest"
            )
            nan_mask = np.isnan(full_y)
            if nan_mask.any() and not nan_mask.all():
                full_y[nan_mask] = np.interp(np.where(nan_mask)[0], np.where(~nan_mask)[0], full_y[~nan_mask])
            boundary_y = full_y

        boundaries.append(boundary_y)
        boundary_pairs.append((top_lbl, bot_lbl))

    print(f"  Boundaries fitted: {len(boundaries)}")
    print(f"  Broken pairs: {broken_pairs}")

    if not boundaries:
        return coarse_labels.copy(), []

    if len(boundaries) > 1:
        medians = [float(np.median(b)) for b in boundaries]
        order = np.argsort(medians)
        boundaries = [boundaries[int(i)] for i in order]
        boundary_pairs = [boundary_pairs[int(i)] for i in order]
        broken_pairs_sorted = set()
        for i in order:
            pair = boundary_pairs[int(i)]
            if pair in broken_pairs:
                broken_pairs_sorted.add(pair)
        broken_pairs = broken_pairs_sorted

    min_layer_height = max(3, h // 100)
    if len(boundaries) > 1:
        for i in range(len(boundaries) - 1):
            gap = np.median(boundaries[i + 1]) - np.median(boundaries[i])
            print(f"  Gap {i}: {gap:.1f} (min={min_layer_height})")
            if gap < min_layer_height:
                print(f"  -> FALLBACK: gap too small")
                return coarse_labels.copy(), boundaries

    if broken_pairs:
        print(f"  -> REPARTITION path")
        refined = _repartition_columns(coarse_labels, spatial_order, boundaries)
    else:
        print(f"  -> LOCAL ADJUST path")
        refined = _adjust_boundaries(coarse_labels, boundaries, boundary_pairs, blend_width=5)

    pixel_change = np.sum(refined != coarse_labels) / (h * w)
    max_change = 0.50 if broken_pairs else 0.15
    print(f"  Pixel change: {pixel_change:.4f} (max={max_change})")
    if pixel_change > max_change:
        print(f"  -> FALLBACK: pixel change too high")
        return coarse_labels.copy(), boundaries

    refined_unique = sorted(u for u in np.unique(refined) if u >= 0)
    if len(refined_unique) < len(layer_labels) - 1:
        print(f"  -> FALLBACK: layer count dropped")
        return coarse_labels.copy(), boundaries

    coarse_frag = _compute_fragmentation_score(coarse_labels)
    refined_frag = _compute_fragmentation_score(refined)
    frag_threshold = 1.5 if broken_pairs else 1.2
    print(f"  Frag: coarse={coarse_frag:.4f} refined={refined_frag:.4f} threshold={frag_threshold}")
    if refined_frag > coarse_frag * frag_threshold:
        print(f"  -> FALLBACK: fragmentation too high")
        return coarse_labels.copy(), boundaries

    return refined, boundaries


def main() -> None:
    fixture = (
        Path(__file__).parent.parent
        / "runs"
        / "readme_examples_v2"
        / "gras2019_16b0cf"
        / "panel_cropped.png"
    )
    img = np.array(Image.open(fixture).convert("RGB"))
    coarse_result = seg_kmeans(img, n_layers=5, max_auto_k=0)
    coarse_labels = coarse_result["labels"]

    print("=" * 60)
    print("BASELINE (no relabel)")
    print("=" * 60)
    refine_debug(img, coarse_labels)

    for thresh in [50, 80, 100]:
        print("\n" + "=" * 60)
        print(f"THRESHOLD = {thresh}")
        print("=" * 60)
        relabeled = spatial_relabel(coarse_labels, threshold=thresh)
        refine_debug(img, relabeled)


if __name__ == "__main__":
    main()
