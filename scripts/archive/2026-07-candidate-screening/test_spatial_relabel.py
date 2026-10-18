"""Spatial consistency relabeling experiment on 16b0cf fixture.

Route A: reassign outlier fragments to spatially nearest layer before
boundary fitting. Tests thresholds 50, 80, 100 px.

Algorithm-specific experiment: imports concrete engine internals intentionally.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from geoseg.modules.segment_engines.kmeans_full import segment as seg_kmeans
from geoseg.modules.segment_engines.horizon_refinement import (
    refine_boundaries,
    _compute_fragmentation_score,
)


def spatial_relabel(labels: np.ndarray, threshold: float) -> np.ndarray:
    """Reassign outlier fragments to their spatially nearest layer.

    For each label, compute its median y (density peak). For each connected
    component (fragment), if its median y deviates from the layer peak by
    more than threshold, reassign all its pixels to the spatially nearest
    layer (by absolute median-y difference).
    """
    from scipy import ndimage

    h, w = labels.shape
    result = labels.copy()
    unique = sorted(u for u in np.unique(labels) if u != 0)

    # Layer density peaks (median y per label)
    peaks: dict[int, float] = {}
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
                # Find nearest layer by peak distance
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


def create_overlay(img: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Simple colored overlay for visualization."""
    colors = np.array([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [255, 128, 0],
        [128, 0, 255],
    ], dtype=np.uint8)

    overlay = img.copy()
    unique = sorted(u for u in np.unique(labels) if u != 0)
    for i, lbl in enumerate(unique):
        mask = labels == lbl
        color = colors[i % len(colors)]
        overlay[mask] = (overlay[mask] * 0.4 + color * 0.6).astype(np.uint8)
    return overlay


def main() -> None:
    fixture = (
        Path(__file__).parent.parent
        / "runs"
        / "readme_examples_v2"
        / "gras2019_16b0cf"
        / "panel_cropped.png"
    )
    out_dir = Path(__file__).parent.parent / "runs" / "horizon_refine" / "experiment_spatial_relabel"
    out_dir.mkdir(parents=True, exist_ok=True)

    img = np.array(Image.open(fixture).convert("RGB"))

    # Step 2: k-means coarse
    coarse_result = seg_kmeans(img, n_layers=5, max_auto_k=0)
    coarse_labels = coarse_result["labels"]
    coarse_frag = _compute_fragmentation_score(coarse_labels)

    # Save coarse baseline
    Image.fromarray(create_overlay(img, coarse_labels)).save(out_dir / "00_coarse.png")

    thresholds = [50, 80, 100]
    results = []

    for thresh in thresholds:
        # Step 3: spatial relabel
        relabeled = spatial_relabel(coarse_labels, threshold=thresh)
        relabel_frag = _compute_fragmentation_score(relabeled)
        pixel_change = np.sum(relabeled != coarse_labels) / (img.shape[0] * img.shape[1])

        # Step 4: refine boundaries
        refined, boundaries = refine_boundaries(img, coarse_labels=relabeled, method="savgol")
        refined_frag = _compute_fragmentation_score(refined)

        # Debug: did refine fall back?
        fallback = np.array_equal(refined, relabeled)
        print(f"  Thresh {thresh}: fallback={fallback}, n_boundaries={len(boundaries)}")

        # Save overlay
        overlay = create_overlay(img, refined)
        Image.fromarray(overlay).save(out_dir / f"thresh_{thresh}.png")

        results.append({
            "threshold": thresh,
            "coarse_frag": coarse_frag,
            "relabel_frag": relabel_frag,
            "refined_frag": refined_frag,
            "pixel_change_ratio": pixel_change,
            "n_boundaries": len(boundaries),
        })

    # Also save standard refine without relabeling for comparison
    std_refined, _ = refine_boundaries(img, coarse_labels=coarse_labels, method="savgol")
    std_frag = _compute_fragmentation_score(std_refined)
    Image.fromarray(create_overlay(img, std_refined)).save(out_dir / "baseline_refined.png")

    # Print report
    print("=" * 60)
    print("Spatial Consistency Relabeling — 16b0cf Fixture")
    print("=" * 60)
    print(f"Baseline coarse fragmentation:     {coarse_frag:.4f}")
    print(f"Baseline refined (no relabel):     {std_frag:.4f}")
    print()
    print(f"{'Thresh':>6} {'RelabelFrag':>12} {'RefinedFrag':>12} {'PixelChange':>12} {'Boundaries':>10}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['threshold']:>6} "
            f"{r['relabel_frag']:>12.4f} "
            f"{r['refined_frag']:>12.4f} "
            f"{r['pixel_change_ratio']:>12.4f} "
            f"{r['n_boundaries']:>10}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
