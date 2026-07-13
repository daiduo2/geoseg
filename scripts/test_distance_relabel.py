"""Test fragment spatial distance relabeling on 16b0cf fixture."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoseg.modules.segment_engines.kmeans_full import segment as seg_kmeans
from geoseg.modules.segment_engines.horizon_refinement import (
    refine_boundaries,
    _compute_fragmentation_score,
)


def _get_fragments(labels: np.ndarray, lbl: int) -> list[tuple[int, float, float, np.ndarray]]:
    """Return list of (area, cx, cy, mask) for each connected component of label."""
    mask = labels == lbl
    labeled, num = ndimage.label(mask)
    if num == 0:
        return []
    fragments = []
    for i in range(1, num + 1):
        ys, xs = np.where(labeled == i)
        if len(xs) == 0:
            continue
        frag_mask = labeled == i
        fragments.append((len(xs), float(xs.mean()), float(ys.mean()), frag_mask))
    return fragments


def _nearest_frag_distance(cx: float, cy: float, frags: list[tuple[int, float, float, np.ndarray]]) -> float:
    """Min Euclidean distance from point to any fragment centroid."""
    if not frags:
        return float("inf")
    return min(np.hypot(cx - f[1], cy - f[2]) for f in frags)


def spatial_distance_relabel(
    labels: np.ndarray,
    spatial_order: list[int],
    ratio_threshold: float = 2.0,
) -> tuple[np.ndarray, int]:
    """Relabel fragments based on spatial proximity to neighboring layers.

    Returns:
        (relabeled_labels, n_reassigned_fragments)
    """
    result = labels.copy()
    n_reassigned = 0

    # Build fragment index: list of (layer_idx, frag_idx, area, cx, cy, mask)
    all_frags = []
    for li, lbl in enumerate(spatial_order):
        frags = _get_fragments(result, lbl)
        for fi, (area, cx, cy, mask) in enumerate(frags):
            all_frags.append((li, fi, area, cx, cy, mask, lbl))

    # Sort by area ascending (small fragments first) to avoid main fragments
    # swallowing small ones before they can be reassigned
    all_frags_sorted = sorted(all_frags, key=lambda x: x[2])

    for li, fi, area, cx, cy, frag_mask, lbl in all_frags_sorted:
        # Skip if this fragment no longer exists (was already reassigned)
        if not np.any(result[frag_mask] == lbl):
            continue

        above = spatial_order[li - 1] if li > 0 else None
        below = spatial_order[li + 1] if li < len(spatial_order) - 1 else None

        # Recompute current layer fragments for d_own
        own_frags = _get_fragments(result, lbl)
        # Exclude self
        own_frags_others = [f for f in own_frags if not np.array_equal(f[3], frag_mask)]
        d_own = _nearest_frag_distance(cx, cy, own_frags_others)

        above_frags = _get_fragments(result, above) if above is not None else []
        below_frags = _get_fragments(result, below) if below is not None else []

        d_above = _nearest_frag_distance(cx, cy, above_frags)
        d_below = _nearest_frag_distance(cx, cy, below_frags)

        # Skip if fragment is well-placed within its own layer
        if d_own == float("inf"):
            continue

        reassigned = False
        if above and d_above < d_own and d_own / d_above > ratio_threshold:
            result[frag_mask] = above
            reassigned = True
        elif below and d_below < d_own and d_own / d_below > ratio_threshold:
            result[frag_mask] = below
            reassigned = True

        if reassigned:
            n_reassigned += 1

    return result, n_reassigned


def _make_overlay(img: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Simple colored overlay."""
    h, w = labels.shape
    unique = sorted(u for u in np.unique(labels) if u != 0)
    colors = np.array([
        [255, 0, 0], [0, 255, 0], [0, 0, 255],
        [255, 255, 0], [255, 0, 255], [0, 255, 255],
    ], dtype=np.uint8)
    overlay = img.copy()
    for i, lbl in enumerate(unique):
        mask = labels == lbl
        color = colors[i % len(colors)]
        overlay[mask] = (overlay[mask] * 0.4 + color * 0.6).astype(np.uint8)
    return overlay


def main() -> None:
    fixture = Path(__file__).parent.parent / "runs" / "readme_examples_v2" / "gras2019_16b0cf" / "panel_cropped.png"
    img = np.array(Image.open(fixture).convert("RGB"))

    # 1. Run k-means
    result = seg_kmeans(img, n_layers=5, max_auto_k=0)
    coarse = result["labels"]

    # 2. Spatial order by median y
    unique = sorted(u for u in np.unique(coarse) if u != 0)
    median_ys = {lbl: float(np.median(np.where(coarse == lbl)[0])) for lbl in unique}
    spatial_order = sorted(unique, key=lambda lbl: median_ys[lbl])
    print(f"Spatial order: {spatial_order}")

    out_dir = Path(__file__).parent.parent / "runs" / "horizon_refine" / "experiment_distance_relabel"
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_frag = _compute_fragmentation_score(coarse)
    print(f"Baseline frag: {baseline_frag:.4f}")

    # Save baseline
    overlay = _make_overlay(img, coarse)
    Image.fromarray(overlay).save(out_dir / "baseline.png")

    for ratio in [1.5, 2.0, 3.0]:
        relabeled, n_reassigned = spatial_distance_relabel(coarse, spatial_order, ratio_threshold=ratio)
        relabel_changed = np.sum(relabeled != coarse) / coarse.size

        # Save relabel-only (before refine)
        overlay_relabel = _make_overlay(img, relabeled)
        Image.fromarray(overlay_relabel).save(out_dir / f"relabel_only_ratio_{ratio}.png")

        # Then refine
        refined, boundaries = refine_boundaries(img, coarse_labels=relabeled)

        frag = _compute_fragmentation_score(refined)
        changed = np.sum(refined != coarse) / coarse.size

        overlay = _make_overlay(img, refined)
        Image.fromarray(overlay).save(out_dir / f"ratio_{ratio}.png")

        print(
            f"Ratio {ratio}: reassigned={n_reassigned}, "
            f"relabel_changed={relabel_changed:.4f}, frag={frag:.4f}, "
            f"changed={changed:.3f}, boundaries={len(boundaries)}"
        )

    # Also test refine on original coarse for comparison
    refined_orig, boundaries_orig = refine_boundaries(img, coarse_labels=coarse)
    frag_orig = _compute_fragmentation_score(refined_orig)
    changed_orig = np.sum(refined_orig != coarse) / coarse.size
    print(
        f"Original+refine: frag={frag_orig:.4f}, changed={changed_orig:.3f}, boundaries={len(boundaries_orig)}"
    )
    overlay_orig = _make_overlay(img, refined_orig)
    Image.fromarray(overlay_orig).save(out_dir / "original_refined.png")

    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
