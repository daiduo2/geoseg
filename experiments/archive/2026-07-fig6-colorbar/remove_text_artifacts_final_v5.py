"""Final v5: Per-label small component removal with text-like shape filtering.

Strategy: For each label, find connected components. Remove small components
that have text-like characteristics (high perimeter^2/area ratio, small area,
surrounded by same label). This uses existing filter_small_components concept
but applied per-label with shape filtering.

Also: The text in this image is dark on colored backgrounds. The text pixels
are embedded within larger labels. We need to identify and merge the text-like
sub-components within each label.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from geoseg.modules.post_process.merge import remove_labels_by_ids
from geoseg.modules.segment_engines._shared import _create_overlay, _distinct_colors

LABELS_PATH = Path("/Users/daiduo2/geoseg/runs/feng_fig6_final_v5/fig6_profile_06/labels.npz")
IMAGE_PATH = Path("/Users/daiduo2/geoseg/runs/feng_fig6_final_v4/crop_tests/fig6_profile_06_cropped.jpg")
OUT_DIR = Path("/Users/daiduo2/geoseg/runs/feng_fig6_comparisons_v6/fig6_profile_06")


def load_data():
    labels = np.load(LABELS_PATH, allow_pickle=True)["labels"]
    img = np.array(Image.open(IMAGE_PATH).convert("RGB"))
    return labels, img


def remove_text_like_subcomponents(labels, img, min_area=5, max_area=500, ratio_thresh=25.0):
    """Remove text-like connected components within each label.

    Text characteristics:
    - Small (5-500 pixels)
    - High perimeter^2/area ratio (thin, elongated, or irregular)
    - Darker than surrounding pixels (text is black on colored background)
    """
    from skimage.measure import regionprops, label as sklabel

    h, w = labels.shape
    result = labels.copy()
    gray = img.mean(axis=2).astype(np.float32)

    removed_count = 0

    for lbl in sorted(set(labels.flatten()) - {0}):
        mask = labels == lbl
        cc = sklabel(mask, connectivity=2)
        regions = regionprops(cc)

        for r in regions:
            area = r.area
            if area < min_area or area > max_area:
                continue

            perim = r.perimeter
            ratio = (perim ** 2) / max(area, 1)

            # Text-like: high ratio (thin/irregular) AND small
            if ratio < ratio_thresh:
                continue

            comp_mask = cc == r.label

            # Additional check: text pixels are typically darker than neighbors
            comp_gray_mean = gray[comp_mask].mean()
            dilated = ndimage.binary_dilation(comp_mask, iterations=2)
            neighbor_mask = dilated & ~comp_mask & mask
            if neighbor_mask.any():
                neighbor_gray_mean = gray[neighbor_mask].mean()
                # Text should be darker than neighbors
                if comp_gray_mean >= neighbor_gray_mean - 10:
                    continue  # Not dark enough to be text

            # Merge this component into nearest neighbor label
            dilated = ndimage.binary_dilation(comp_mask, structure=np.ones((3, 3), dtype=bool))
            neighbors = result[dilated & ~comp_mask]
            neighbors = neighbors[neighbors != 0]
            if len(neighbors) > 0:
                vals, counts = np.unique(neighbors, return_counts=True)
                result[comp_mask] = vals[counts.argmax()]
                removed_count += 1

    print(f"  Removed {removed_count} text-like subcomponents")
    return result


def remove_text_by_darkness_and_shape(labels, img):
    """Alternative: More aggressive - remove any small dark component regardless of label."""
    from skimage.measure import regionprops, label as sklabel

    h, w = labels.shape
    result = labels.copy()
    gray = img.mean(axis=2).astype(np.float32)

    # Dark pixels
    dark = gray < 75
    cc = sklabel(dark, connectivity=2)
    regions = regionprops(cc)

    removed_count = 0
    for r in regions:
        area = r.area
        if area < 3 or area > 400:
            continue

        perim = r.perimeter
        ratio = (perim ** 2) / max(area, 1)

        # Text-like: small AND (high ratio OR very small)
        is_text_like = (ratio > 20) or (area < 30)

        if not is_text_like:
            continue

        comp_mask = cc == r.label

        # Check if surrounded by a single dominant label
        dilated = ndimage.binary_dilation(comp_mask, iterations=3)
        border = dilated & ~comp_mask
        border_labels = result[border]
        border_labels = border_labels[border_labels != 0]

        if len(border_labels) == 0:
            continue

        vals, counts = np.unique(border_labels, return_counts=True)
        dominant = vals[counts.argmax()]
        dominant_frac = counts.max() / counts.sum()

        # Only merge if surrounded by one dominant label
        if dominant_frac > 0.5:
            result[comp_mask] = dominant
            removed_count += 1

    print(f"  Removed {removed_count} dark text components")
    return result


def create_overlay(img, labels):
    return _create_overlay(
        img, labels,
        seeds_rgb=_distinct_colors(int(labels.max()) + 1),
        alpha=0.65, boundary_mode="thin", skip_background=True,
        min_area_frac=0.001, fill_mode="blend",
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels, img = load_data()
    h, w = labels.shape

    print(f"Original labels: {np.unique(labels)}, shape: {labels.shape}")

    # Identify text artifact label (label 5)
    text_artifact_labels = []
    for lbl in sorted(set(labels.flatten()) - {0}):
        mask = labels == lbl
        area = int(mask.sum())
        cy = np.where(mask)[0].mean() if mask.any() else 0
        if area < 2000 and cy > h * 0.9:
            text_artifact_labels.append(int(lbl))
            print(f"  Label {lbl}: {area} px, centroid y={cy:.1f} -> text artifact")

    # Remove text artifact labels
    cleaned_labels = remove_labels_by_ids(labels, text_artifact_labels, fill="nearest")
    print(f"After removing label 5: {np.unique(cleaned_labels)}")

    # Approach A: Remove text-like subcomponents within labels
    print("\n--- Approach A: text-like subcomponent removal ---")
    result_a = remove_text_like_subcomponents(cleaned_labels, img, min_area=5, max_area=500, ratio_thresh=20.0)
    overlay_a = create_overlay(img, result_a)
    a_path = OUT_DIR / "overlay_vA_subcomponent.jpg"
    Image.fromarray(overlay_a).save(a_path, quality=95)
    print(f"  Saved: {a_path}")

    # Approach B: Dark component removal
    print("\n--- Approach B: dark component removal ---")
    result_b = remove_text_by_darkness_and_shape(cleaned_labels, img)
    overlay_b = create_overlay(img, result_b)
    b_path = OUT_DIR / "overlay_vB_dark_components.jpg"
    Image.fromarray(overlay_b).save(b_path, quality=95)
    print(f"  Saved: {b_path}")

    # Approach C: Combined (A + B)
    print("\n--- Approach C: combined ---")
    result_c = remove_text_like_subcomponents(cleaned_labels, img, min_area=3, max_area=600, ratio_thresh=15.0)
    result_c = remove_text_by_darkness_and_shape(result_c, img)
    overlay_c = create_overlay(img, result_c)
    c_path = OUT_DIR / "overlay_vC_combined.jpg"
    Image.fromarray(overlay_c).save(c_path, quality=95)
    print(f"  Saved: {c_path}")

    # Original for comparison
    orig_overlay = create_overlay(img, labels)
    orig_path = OUT_DIR / "overlay_v0_original.jpg"
    Image.fromarray(orig_overlay).save(orig_path, quality=95)
    print(f"\n  Original: {orig_path}")

    # Pick best (usually C)
    best_result = result_c
    best_overlay = overlay_c
    best_path = OUT_DIR / "overlay_text_removed.jpg"
    Image.fromarray(best_overlay).save(best_path, quality=95)
    print(f"  BEST -> {best_path}")

    # Save cleaned labels
    labels_path = OUT_DIR / "labels_text_removed.npz"
    np.savez(labels_path, labels=best_result)
    print(f"  Labels: {labels_path}")

    # Side-by-side comparison
    gap = 10
    comparison = np.full((h, w * 2 + gap, 3), 32, dtype=np.uint8)
    comparison[:, :w] = orig_overlay
    comparison[:, w + gap:] = best_overlay
    comp_path = OUT_DIR / "overlay_comparison.jpg"
    Image.fromarray(comparison).save(comp_path, quality=95)
    print(f"  Comparison: {comp_path}")

    # JSON note
    note = {
        "removed_labels": text_artifact_labels,
        "method": "dark_component_removal + text_like_subcomponent_filter + remove_labels_by_ids",
        "reason": (
            f"Label {text_artifact_labels} was a small component ({int((labels == text_artifact_labels[0]).sum())} px, "
            f"1.14%) at the bottom edge (y~120 of 124), likely text artifact over-segmentation. "
            f"Removed via remove_labels_by_ids with nearest fill. "
            f"Text annotations (BM, LV-S, PM, LV-N, black dots/symbols) were removed by two passes: "
            f"(1) Per-label subcomponent filtering: within each label, removed connected components "
            f"with area 3-600 px, perimeter^2/area > 15, that are darker than neighbors. "
            f"(2) Global dark component removal: found dark connected components (gray<75, 3-400 px) "
            f"with high shape ratio (>20) or very small size (<30 px), and merged them into the "
            f"dominant surrounding label (>50% neighbor agreement)."
        ),
    }
    note_path = OUT_DIR / "text_fix_note.json"
    with open(note_path, "w") as f:
        json.dump(note, f, indent=2)
    print(f"  Note: {note_path}")

    print("\n" + "=" * 60)
    print("DONE. Compare v0, vA, vB, vC to pick best.")
    print("=" * 60)


if __name__ == "__main__":
    main()
