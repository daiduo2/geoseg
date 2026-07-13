"""Debug spatial relabeling — why does refinement fail to improve?"""
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
    _fit_savgol,
    _repartition_columns,
    _adjust_boundaries,
)


def spatial_relabel(labels: np.ndarray, threshold: float) -> np.ndarray:
    from scipy import ndimage

    h, w = labels.shape
    result = labels.copy()
    unique = sorted(u for u in np.unique(labels) if u != 0)

    peaks: dict[int, float] = {}
    for lbl in unique:
        ys = np.where(labels == lbl)[0]
        peaks[lbl] = float(np.median(ys)) if len(ys) > 0 else h / 2

    reassigned = 0
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
                reassigned += int(np.sum(frag_mask))

    return result, reassigned


def diagnose(labels: np.ndarray, name: str) -> None:
    print(f"\n--- {name} ---")
    unique = sorted(u for u in np.unique(labels) if u != 0)
    print(f"Layers: {unique}")

    # Spatial order
    median_ys = {}
    for lbl in unique:
        ys = np.where(labels == lbl)[0]
        median_ys[lbl] = float(np.median(ys)) if len(ys) > 0 else 0
    spatial_order = sorted(unique, key=lambda lbl: median_ys[lbl])
    print(f"Spatial order: {spatial_order}")
    for lbl in spatial_order:
        print(f"  Layer {lbl}: median_y={median_ys[lbl]:.1f}")

    # Touching check
    for i in range(len(spatial_order) - 1):
        top = spatial_order[i]
        bot = spatial_order[i + 1]
        mask_top = labels == top
        mask_bot = labels == bot
        touch = mask_top & ndimage.binary_dilation(mask_bot, iterations=1)
        n_touch = int(np.sum(touch))
        print(f"  Pair ({top},{bot}): touching_pixels={n_touch}")

    frag = _compute_fragmentation_score(labels)
    print(f"  Fragmentation: {frag:.4f}")


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

    diagnose(coarse_labels, "COARSE (original)")

    for thresh in [50, 80, 100]:
        relabeled, reassigned = spatial_relabel(coarse_labels, threshold=thresh)
        diagnose(relabeled, f"RELABEL thresh={thresh} (reassigned={reassigned}px)")


if __name__ == "__main__":
    main()
