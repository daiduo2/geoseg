"""Experiment: remove text annotation artifacts from fig6_profile_06 labels.

Text artifacts visible in the image:
- "BM" labels (two instances, left side)
- "LV-S" label (center-left, with star symbol)
- "PM" labels (two instances, center)
- "LV-N" label (right side)
- Black dots/symbols scattered throughout

These cause over-segmentation (small spurious labels) or distract in overlays.
We try several candidate approaches using existing utilities in merge.py and _shared.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from geoseg.modules.post_process.merge import (
    filter_small_components,
    merge_labels_by_ids,
    remove_labels_by_ids,
)
from geoseg.modules.segment_engines._shared import _create_overlay, _distinct_colors

# Paths
LABELS_PATH = Path("/Users/daiduo2/geoseg/runs/feng_fig6_final_v5/fig6_profile_06/labels.npz")
IMAGE_PATH = Path("/Users/daiduo2/geoseg/runs/feng_fig6_final_v4/crop_tests/fig6_profile_06_cropped.jpg")
OUT_DIR = Path("/Users/daiduo2/geoseg/runs/feng_fig6_comparisons_v6/fig6_profile_06")


def load_data():
    labels = np.load(LABELS_PATH, allow_pickle=True)["labels"]
    img = np.array(Image.open(IMAGE_PATH).convert("RGB"))
    return labels, img


def save_overlay(labels, img, path, title=""):
    """Generate and save overlay using existing _create_overlay."""
    overlay = _create_overlay(
        img,
        labels,
        seeds_rgb=_distinct_colors(int(labels.max()) + 1),
        alpha=0.65,
        boundary_mode="thin",
        skip_background=True,
        min_area_frac=0.001,
        fill_mode="blend",
    )
    Image.fromarray(overlay).save(path, quality=95)
    print(f"  Saved: {path} {title}")
    return overlay


def analyze_labels(labels, img):
    """Print detailed stats for each label to identify text artifacts."""
    h, w = labels.shape
    total = h * w
    unique = sorted(set(labels.flatten()) - {0})
    print(f"\nImage size: {h}x{w} = {total} pixels")
    print(f"Labels (excluding 0): {unique}")
    print("-" * 80)
    print(f"{'Label':>6} {'Pixels':>10} {'Area%':>8} {'BBox':>20} {'Centroid':>16} {'Perim^2/Area':>14}")
    print("-" * 80)

    for lbl in unique:
        mask = labels == lbl
        area = int(mask.sum())
        area_pct = area / total * 100
        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        bbox = f"({x0},{y0})-({x1},{y1})"
        cy, cx = ys.mean(), xs.mean()
        centroid = f"({cx:.1f},{cy:.1f})"

        # Shape ratio (perimeter^2 / area) - high = thin/fragmented
        cc = ndimage.label(mask)[0]
        from skimage.measure import regionprops
        regions = regionprops(cc)
        if regions:
            ratio = max((r.perimeter**2 / max(r.area, 1)) for r in regions)
        else:
            ratio = 0

        print(f"{lbl:>6} {area:>10} {area_pct:>7.3f}% {bbox:>20} {centroid:>16} {ratio:>13.1f}")

    print("-" * 80)


def approach_1_remove_small(labels, img):
    """Approach 1: Remove very small components (< 0.05% of image)."""
    print("\n--- Approach 1: filter_small_components (min_area_ratio=0.0005) ---")
    result = filter_small_components(labels, min_area_ratio=0.0005, fill="nearest")
    save_overlay(result, img, OUT_DIR / "approach1_small_removed.jpg", "(small removed)")
    return result


def approach_2_detect_dark_text(labels, img):
    """Approach 2: Detect dark text pixels and merge into nearest non-background label."""
    print("\n--- Approach 2: Dark text pixel detection + nearest fill ---")
    gray = img.mean(axis=2).astype(np.float32)

    # Dark pixels: low grayscale value (text is black/dark)
    # But also need to avoid removing actual dark geological features
    dark_mask = gray < 60

    # Also detect high-contrast edges (text has sharp edges)
    from skimage.filters import sobel
    edges = np.abs(sobel(gray))
    edge_mask = edges > 0.15

    # Text = dark AND edgy AND small isolated regions
    text_candidate = dark_mask & edge_mask

    # Dilate slightly to catch text halo
    text_dilated = ndimage.binary_dilation(text_candidate, iterations=2)

    # Only consider text candidates that are small fragments within larger labels
    result = labels.copy()
    for lbl in sorted(set(labels.flatten()) - {0}):
        mask = labels == lbl
        cc, num = ndimage.label(mask)
        if num <= 1:
            continue
        for i in range(1, num + 1):
            comp = cc == i
            # If component is small and overlaps with text candidate
            if comp.sum() < 200 and (comp & text_dilated).sum() > comp.sum() * 0.3:
                # Merge into nearest other label
                dilated = ndimage.binary_dilation(comp, structure=np.ones((3, 3), dtype=bool))
                neighbors = result[dilated & ~comp]
                neighbors = neighbors[neighbors != 0]
                if len(neighbors) > 0:
                    vals, counts = np.unique(neighbors, return_counts=True)
                    result[comp] = vals[counts.argmax()]

    save_overlay(result, img, OUT_DIR / "approach2_dark_text.jpg", "(dark text removed)")
    return result


def approach_3_merge_text_fragments(labels, img):
    """Approach 3: Identify text-like labels by shape (high perimeter^2/area) and merge them."""
    print("\n--- Approach 3: Shape-based text fragment detection + merge ---")
    from skimage.measure import regionprops, label as sklabel

    h, w = labels.shape
    result = labels.copy()
    text_labels = []

    for lbl in sorted(set(labels.flatten()) - {0}):
        mask = labels == lbl
        cc = sklabel(mask, connectivity=2)
        regions = regionprops(cc)

        for r in regions:
            area = r.area
            perim = r.perimeter
            ratio = (perim ** 2) / max(area, 1)
            bbox_w = r.bbox[3] - r.bbox[1]
            bbox_h = r.bbox[2] - r.bbox[0]
            aspect = max(bbox_w, bbox_h) / max(min(bbox_w, bbox_h), 1)

            # Text-like: high ratio, small area, elongated or compact
            is_text_like = (
                area < 500
                and ratio > 50
                and aspect > 1.5
            )

            if is_text_like:
                comp_mask = cc == r.label
                # Find which original label this component belongs to
                # Actually we need to operate on the component within the label
                # But the label might have multiple components
                # Let's mark this component for removal
                text_labels.append((comp_mask, lbl, area, ratio, aspect))

    print(f"  Found {len(text_labels)} text-like components")
    for comp_mask, lbl, area, ratio, aspect in text_labels:
        dilated = ndimage.binary_dilation(comp_mask, structure=np.ones((3, 3), dtype=bool))
        neighbors = result[dilated & ~comp_mask]
        neighbors = neighbors[neighbors != 0]
        if len(neighbors) > 0:
            vals, counts = np.unique(neighbors, return_counts=True)
            result[comp_mask] = vals[counts.argmax()]

    save_overlay(result, img, OUT_DIR / "approach3_shape_fragments.jpg", "(shape fragments removed)")
    return result


def approach_4_combined(labels, img):
    """Approach 4: Combined - remove small + shape-based text removal + nearest fill."""
    print("\n--- Approach 4: Combined (small + shape + dark text) ---")
    from skimage.measure import regionprops, label as sklabel

    h, w = labels.shape
    total = h * w
    result = labels.copy()

    gray = img.mean(axis=2).astype(np.float32)
    dark_mask = gray < 70
    from skimage.filters import sobel
    edges = np.abs(sobel(gray))
    edge_mask = edges > 0.12
    text_candidate = dark_mask & edge_mask
    text_dilated = ndimage.binary_dilation(text_candidate, iterations=2)

    # Step 1: Remove very small components
    min_area = max(30, int(total * 0.0003))
    for lbl in sorted(set(result.flatten()) - {0}):
        mask = result == lbl
        cc = sklabel(mask, connectivity=2)
        regions = regionprops(cc)
        for r in regions:
            if r.area < min_area:
                comp_mask = cc == r.label
                dilated = ndimage.binary_dilation(comp_mask, structure=np.ones((3, 3), dtype=bool))
                neighbors = result[dilated & ~comp_mask]
                neighbors = neighbors[neighbors != 0]
                if len(neighbors) > 0:
                    vals, counts = np.unique(neighbors, return_counts=True)
                    result[comp_mask] = vals[counts.argmax()]

    # Step 2: Remove text-like components (small + high ratio + overlaps with text mask)
    for lbl in sorted(set(result.flatten()) - {0}):
        mask = result == lbl
        cc = sklabel(mask, connectivity=2)
        regions = regionprops(cc)
        for r in regions:
            area = r.area
            ratio = (r.perimeter ** 2) / max(area, 1)
            bbox_w = r.bbox[3] - r.bbox[1]
            bbox_h = r.bbox[2] - r.bbox[0]
            aspect = max(bbox_w, bbox_h) / max(min(bbox_w, bbox_h), 1)

            comp_mask = cc == r.label
            text_overlap = (comp_mask & text_dilated).sum() / max(comp_mask.sum(), 1)

            is_text_like = (
                area < 800
                and ratio > 30
                and (aspect > 1.5 or text_overlap > 0.2)
                and text_overlap > 0.1
            )

            if is_text_like:
                dilated = ndimage.binary_dilation(comp_mask, structure=np.ones((3, 3), dtype=bool))
                neighbors = result[dilated & ~comp_mask]
                neighbors = neighbors[neighbors != 0]
                if len(neighbors) > 0:
                    vals, counts = np.unique(neighbors, return_counts=True)
                    result[comp_mask] = vals[counts.argmax()]

    save_overlay(result, img, OUT_DIR / "approach4_combined.jpg", "(combined)")
    return result


def approach_5_morphological_clean(labels, img):
    """Approach 5: Morphological opening on each label to remove thin text artifacts."""
    print("\n--- Approach 5: Morphological opening per label ---")
    from skimage.morphology import disk, opening

    result = labels.copy()
    for lbl in sorted(set(labels.flatten()) - {0}):
        mask = labels == lbl
        # Small opening to remove thin text strokes
        cleaned = opening(mask, footprint=disk(1))
        # Removed pixels get reassigned
        removed = mask & ~cleaned
        if removed.any():
            dilated = ndimage.binary_dilation(removed, structure=np.ones((3, 3), dtype=bool))
            neighbors = labels[dilated & ~removed]
            neighbors = neighbors[neighbors != 0]
            if len(neighbors) > 0:
                vals, counts = np.unique(neighbors, return_counts=True)
                result[removed] = vals[counts.argmax()]
            else:
                result[removed] = 0

    save_overlay(result, img, OUT_DIR / "approach5_morphological.jpg", "(morphological)")
    return result


def approach_6_region_grow_fill(labels, img):
    """Approach 6: For each label, remove isolated small holes and thin protrusions."""
    print("\n--- Approach 6: Region-based hole/thin-protrusion fill ---")
    from skimage.measure import regionprops, label as sklabel

    result = labels.copy()

    for lbl in sorted(set(labels.flatten()) - {0}):
        mask = labels == lbl
        cc = sklabel(mask, connectivity=2)
        regions = regionprops(cc)

        for r in regions:
            area = r.area
            ratio = (r.perimeter ** 2) / max(area, 1)
            comp_mask = cc == r.label

            # Remove very thin or very small components
            if area < 100 or ratio > 80:
                dilated = ndimage.binary_dilation(comp_mask, structure=np.ones((3, 3), dtype=bool))
                neighbors = result[dilated & ~comp_mask]
                neighbors = neighbors[neighbors != 0]
                if len(neighbors) > 0:
                    vals, counts = np.unique(neighbors, return_counts=True)
                    result[comp_mask] = vals[counts.argmax()]

    save_overlay(result, img, OUT_DIR / "approach6_region_grow.jpg", "(region grow fill)")
    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    labels, img = load_data()
    print(f"Labels shape: {labels.shape}, unique: {np.unique(labels)}")
    print(f"Image shape: {img.shape}")

    # Save original overlay for comparison
    save_overlay(labels, img, OUT_DIR / "original_overlay.jpg", "(ORIGINAL)")

    # Analyze labels
    analyze_labels(labels, img)

    # Try all approaches
    results = {}
    results["approach1"] = approach_1_remove_small(labels, img)
    results["approach2"] = approach_2_detect_dark_text(labels, img)
    results["approach3"] = approach_3_merge_text_fragments(labels, img)
    results["approach4"] = approach_4_combined(labels, img)
    results["approach5"] = approach_5_morphological_clean(labels, img)
    results["approach6"] = approach_6_region_grow_fill(labels, img)

    print(f"\n\nAll approaches saved to {OUT_DIR}")
    print("Compare overlays and pick the best one.")


if __name__ == "__main__":
    main()
