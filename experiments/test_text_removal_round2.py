#!/usr/bin/env python3
"""Refined text artifact removal for fig6_profile_07.

Key insight from first round: The "LV-N" text is NOT a separate label - it's
embedded within label 1 (pink region). The dark pixels are part of the same
connected component. Need to target dark pixels within labels and reassign them.

Also need to handle:
1. The small light-blue blob in upper-left (label 3 component, area=168)
2. The thin strip at bottom-left (label 3 component, area=1490)
3. The dark "LV-N" text and dots within label 1
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from geoseg.modules.post_process.merge import (
    filter_small_components,
    remove_labels_by_ids,
)
from geoseg.modules.segment_engines._shared import _create_overlay


def create_overlay(panel_rgb, labels, fill_mode="blend"):
    return _create_overlay(
        panel_rgb,
        labels,
        np.zeros((1, 3), dtype=np.uint8),
        alpha=0.65,
        boundary_mode="thin",
        skip_background=True,
        min_area_frac=0.0005,
        fill_mode=fill_mode,
    )


def approach_6_dark_pixel_inpaint(labels, img_rgb, dark_threshold=90, morph_open_size=3):
    """Approach 6: Inpaint dark pixels within each label by reassigning to nearest non-dark neighbor.

    This targets text like "LV-N" that is embedded within a label (same connected component).
    Uses morphological opening to identify text-like structures.
    """
    gray = img_rgb.mean(axis=2)
    dark_mask = gray < dark_threshold

    # Also detect very dark pixels (text is typically black/dark)
    very_dark = gray < 60

    result = labels.copy()
    h, w = labels.shape

    # For each label, find dark pixels and reassign them
    for lbl in sorted(set(labels.flatten()) - {0}):
        lbl_mask = labels == lbl
        lbl_dark = lbl_mask & dark_mask

        if not lbl_dark.any():
            continue

        # Use morphological opening to find small dark structures (text)
        from skimage.morphology import binary_opening, disk

        # Opening with small disk removes small text dots but keeps larger structures
        opened = binary_opening(lbl_dark, footprint=disk(2))

        # The difference between dark and opened = text pixels
        text_pixels = lbl_dark & (~opened)

        # Also include very dark pixels regardless
        text_pixels = text_pixels | (lbl_mask & very_dark)

        if not text_pixels.any():
            continue

        print(f"Label {lbl}: {text_pixels.sum()} text pixels to inpaint")

        # Dilate text pixels slightly to catch edges
        dilated_text = ndimage.binary_dilation(text_pixels, iterations=1)

        # Valid pixels to sample from: same label, not text
        valid_in_label = lbl_mask & (~dilated_text)

        if not valid_in_label.any():
            # Fall back to any non-text, non-background neighbor
            valid_neighbors = (~dilated_text) & (labels != 0) & (~lbl_mask)
            if not valid_neighbors.any():
                continue
            _, indices = ndimage.distance_transform_edt(~valid_neighbors, return_indices=True)
        else:
            # Prefer same-label neighbors, but allow other labels if too far
            _, indices = ndimage.distance_transform_edt(~valid_in_label, return_indices=True)

        rr, cc = np.where(text_pixels)
        result[rr, cc] = labels[indices[0][rr, cc], indices[1][rr, cc]]

    return result


def approach_7_component_split_and_merge(labels, img_rgb, dark_threshold=90):
    """Approach 7: Split labels by darkness, then merge small dark fragments into neighbors.

    1. Within each label, identify dark (text) pixels
    2. Temporarily separate them
    3. Small dark fragments get merged into nearest non-dark neighbor
    4. Large dark regions are preserved (could be real geological features)
    """
    gray = img_rgb.mean(axis=2)
    dark_mask = gray < dark_threshold

    result = labels.copy()

    for lbl in sorted(set(labels.flatten()) - {0}):
        lbl_mask = labels == lbl
        lbl_dark = lbl_mask & dark_mask

        if not lbl_dark.any():
            continue

        # Label the dark regions within this label
        dark_labeled, dark_num = ndimage.label(lbl_dark)
        print(f"Label {lbl}: {dark_num} dark sub-components")

        for i in range(1, dark_num + 1):
            comp = dark_labeled == i
            area = comp.sum()

            # Small dark regions = text artifacts
            if area < 200:  # Text is typically small
                # Find nearest non-dark pixel in the same or adjacent label
                valid = lbl_mask & (~comp)
                if not valid.any():
                    valid = (~comp) & (labels != 0)

                if valid.any():
                    _, indices = ndimage.distance_transform_edt(~valid, return_indices=True)
                    rr, cc = np.where(comp)
                    result[rr, cc] = labels[indices[0][rr, cc], indices[1][rr, cc]]
                    print(f"  Removed dark sub-comp {i}: area={area}")

    return result


def approach_8_morphological_text_clean(labels, img_rgb, dark_threshold=85):
    """Approach 8: Morphological text cleaning.

    1. Create a mask of dark pixels
    2. Use morphological opening to remove small text structures
    3. Reassign removed pixels to nearest non-dark neighbor
    """
    from skimage.morphology import binary_opening, disk, remove_small_objects

    gray = img_rgb.mean(axis=2)
    dark_mask = gray < dark_threshold

    # Remove small dark objects (text dots, small symbols)
    cleaned_dark = remove_small_objects(dark_mask, min_size=30)

    # Opening to remove thin text strokes
    opened = binary_opening(cleaned_dark, footprint=disk(2))

    # Text pixels = dark but not in opened
    text_mask = cleaned_dark & (~opened)

    # Also get very small dark objects entirely
    small_dark = dark_mask & (~remove_small_objects(dark_mask, min_size=100))

    combined_text = text_mask | small_dark

    print(f"Text pixels identified: {combined_text.sum()}")

    if not combined_text.any():
        return labels

    result = labels.copy()

    # For each label, reassign text pixels to nearest non-text neighbor in same label
    for lbl in sorted(set(labels.flatten()) - {0}):
        lbl_mask = labels == lbl
        lbl_text = lbl_mask & combined_text

        if not lbl_text.any():
            continue

        valid_in_label = lbl_mask & (~combined_text)
        if valid_in_label.sum() < 10:
            # Too few valid pixels in this label, use any non-text neighbor
            valid = (~combined_text) & (labels != 0)
        else:
            valid = valid_in_label

        if not valid.any():
            continue

        _, indices = ndimage.distance_transform_edt(~valid, return_indices=True)
        rr, cc = np.where(lbl_text)
        result[rr, cc] = labels[indices[0][rr, cc], indices[1][rr, cc]]

    return result


def approach_9_comprehensive(labels, img_rgb):
    """Approach 9: Comprehensive fix combining multiple techniques.

    Step 1: Remove small components from label 3 (the fragmented layer)
    Step 2: Inpaint dark text pixels within each label
    Step 3: Filter any remaining tiny components globally
    """
    result = labels.copy()
    h, w = labels.shape
    gray = img_rgb.mean(axis=2)
    dark_mask = gray < 85
    very_dark = gray < 55

    removed_info = []

    # Step 1: Remove small components from label 3
    lbl3_mask = labels == 3
    lbl3_labeled, lbl3_num = ndimage.label(lbl3_mask)
    for i in range(1, lbl3_num + 1):
        comp = lbl3_labeled == i
        area = comp.sum()
        if area < 1000:
            removed_info.append({
                "type": "small_component",
                "label": 3,
                "area": int(area),
                "reason": "fragment below 1000px threshold",
            })
            result[comp] = 0

    # Fill removed label 3 components
    removed_mask = result == 0
    valid_for_fill = (labels != 0) & (~removed_mask)
    if valid_for_fill.any():
        _, indices = ndimage.distance_transform_edt(~valid_for_fill, return_indices=True)
        rr, cc = np.where(removed_mask & (labels != 0))
        result[rr, cc] = labels[indices[0][rr, cc], indices[1][rr, cc]]

    # Step 2: Inpaint dark pixels within each label
    for lbl in sorted(set(result.flatten()) - {0}):
        lbl_mask = result == lbl
        lbl_dark = lbl_mask & (dark_mask | very_dark)

        if not lbl_dark.any():
            continue

        # Use morphological opening to identify text structures
        from skimage.morphology import binary_opening, disk
        opened = binary_opening(lbl_dark, footprint=disk(2))
        text_pixels = lbl_dark & (~opened)
        text_pixels = text_pixels | (lbl_mask & very_dark)

        if not text_pixels.any():
            continue

        # Count text pixels
        n_text = text_pixels.sum()

        # Dilate slightly
        dilated = ndimage.binary_dilation(text_pixels, iterations=1)

        # Valid pixels: same label, not text
        valid = lbl_mask & (~dilated)
        if not valid.any():
            valid = (~dilated) & (result != 0) & (~lbl_mask)

        if not valid.any():
            continue

        _, indices = ndimage.distance_transform_edt(~valid, return_indices=True)
        rr, cc = np.where(text_pixels)
        result[rr, cc] = result[indices[0][rr, cc], indices[1][rr, cc]]

        removed_info.append({
            "type": "dark_text_inpaint",
            "label": int(lbl),
            "pixels": int(n_text),
            "reason": "dark text pixels reassigned to neighbor",
        })

    # Step 3: Final small component filter
    result = filter_small_components(result, min_area_ratio=0.002, fill="nearest")

    return result, removed_info


def approach_10_label1_dark_inpaint_only(labels, img_rgb):
    """Approach 10: Target only label 1's dark pixels (where LV-N text lives).

    More conservative - only touches label 1 and small label 3 fragments.
    """
    result = labels.copy()
    gray = img_rgb.mean(axis=2)

    # Target label 1 dark pixels
    lbl1_mask = labels == 1
    lbl1_dark = lbl1_mask & (gray < 90)

    # Also catch the small label 3 fragments
    lbl3_mask = labels == 3
    lbl3_labeled, lbl3_num = ndimage.label(lbl3_mask)
    for i in range(1, lbl3_num + 1):
        comp = lbl3_labeled == i
        if comp.sum() < 1000:
            result[comp] = 0

    # Fill removed label 3 components
    removed_mask = result == 0
    valid_for_fill = (labels != 0) & (~removed_mask)
    if valid_for_fill.any():
        _, indices = ndimage.distance_transform_edt(~valid_for_fill, return_indices=True)
        rr, cc = np.where(removed_mask & (labels != 0))
        result[rr, cc] = labels[indices[0][rr, cc], indices[1][rr, cc]]

    # Now handle label 1 dark pixels
    if lbl1_dark.any():
        from skimage.morphology import binary_opening, disk
        opened = binary_opening(lbl1_dark, footprint=disk(2))
        text_pixels = lbl1_dark & (~opened)

        # Also include very dark pixels
        text_pixels = text_pixels | (lbl1_mask & (gray < 60))

        if text_pixels.any():
            dilated = ndimage.binary_dilation(text_pixels, iterations=1)
            valid = lbl1_mask & (~dilated)
            if not valid.any():
                valid = (~dilated) & (result != 0)

            if valid.any():
                _, indices = ndimage.distance_transform_edt(~valid, return_indices=True)
                rr, cc = np.where(text_pixels)
                result[rr, cc] = result[indices[0][rr, cc], indices[1][rr, cc]]

    return result


def main():
    base_dir = Path("/Users/daiduo2/geoseg")
    labels_path = base_dir / "runs/feng_fig6_final_v5/fig6_profile_07/labels.npz"
    img_path = base_dir / "runs/feng_fig6_final_v4/crop_tests/fig6_profile_07_cropped.jpg"
    out_dir = base_dir / "runs/feng_fig6_comparisons_v6/fig6_profile_07"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = np.load(labels_path)["labels"]
    img_rgb = np.array(Image.open(img_path).convert("RGB"))

    print(f"Labels shape: {labels.shape}, unique: {np.unique(labels)}")

    # Load original overlay for comparison
    orig_overlay = create_overlay(img_rgb, labels)

    approaches = []

    # Approach 6: Dark pixel inpaint
    print("\n--- Approach 6: dark_pixel_inpaint ---")
    a6_labels = approach_6_dark_pixel_inpaint(labels, img_rgb, dark_threshold=90)
    a6_overlay = create_overlay(img_rgb, a6_labels)
    Image.fromarray(a6_overlay).save(out_dir / "overlay_approach6_dark_inpaint.jpg")
    print(f"Unique labels after: {np.unique(a6_labels)}")
    approaches.append(("approach6_dark_inpaint", a6_labels, a6_overlay))

    # Approach 7: Component split and merge
    print("\n--- Approach 7: component_split_and_merge ---")
    a7_labels = approach_7_component_split_and_merge(labels, img_rgb, dark_threshold=90)
    a7_overlay = create_overlay(img_rgb, a7_labels)
    Image.fromarray(a7_overlay).save(out_dir / "overlay_approach7_split_merge.jpg")
    print(f"Unique labels after: {np.unique(a7_labels)}")
    approaches.append(("approach7_split_merge", a7_labels, a7_overlay))

    # Approach 8: Morphological text clean
    print("\n--- Approach 8: morphological_text_clean ---")
    a8_labels = approach_8_morphological_text_clean(labels, img_rgb, dark_threshold=85)
    a8_overlay = create_overlay(img_rgb, a8_labels)
    Image.fromarray(a8_overlay).save(out_dir / "overlay_approach8_morph_clean.jpg")
    print(f"Unique labels after: {np.unique(a8_labels)}")
    approaches.append(("approach8_morph_clean", a8_labels, a8_overlay))

    # Approach 9: Comprehensive
    print("\n--- Approach 9: comprehensive ---")
    a9_labels, a9_info = approach_9_comprehensive(labels, img_rgb)
    a9_overlay = create_overlay(img_rgb, a9_labels)
    Image.fromarray(a9_overlay).save(out_dir / "overlay_approach9_comprehensive.jpg")
    print(f"Unique labels after: {np.unique(a9_labels)}")
    print(f"Removed info: {a9_info}")
    approaches.append(("approach9_comprehensive", a9_labels, a9_overlay, a9_info))

    # Approach 10: Label 1 targeted only
    print("\n--- Approach 10: label1_dark_inpaint_only ---")
    a10_labels = approach_10_label1_dark_inpaint_only(labels, img_rgb)
    a10_overlay = create_overlay(img_rgb, a10_labels)
    Image.fromarray(a10_overlay).save(out_dir / "overlay_approach10_label1_only.jpg")
    print(f"Unique labels after: {np.unique(a10_labels)}")
    approaches.append(("approach10_label1_only", a10_labels, a10_overlay))

    # Create comparison grid
    print("\n--- Creating comparison grid (round 2) ---")
    n = len(approaches) + 1
    fig_rows = (n + 2) // 3
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(fig_rows, 3, figsize=(18, 6 * fig_rows))
    if fig_rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    axes[0].imshow(orig_overlay)
    axes[0].set_title("Original")
    axes[0].axis("off")

    for i, (name, lbls, ovly, *rest) in enumerate(approaches):
        axes[i + 1].imshow(ovly)
        axes[i + 1].set_title(name)
        axes[i + 1].axis("off")

    for j in range(i + 2, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(out_dir / "comparison_grid_round2.jpg", dpi=150, bbox_inches="tight")
    print(f"Saved comparison grid to {out_dir / 'comparison_grid_round2.jpg'}")

    # Save all candidate label arrays
    for name, lbls, ovly, *rest in approaches:
        np.savez(out_dir / f"labels_{name}.npz", labels=lbls)

    print(f"\nAll outputs saved to {out_dir}")
    return approaches, orig_overlay


if __name__ == "__main__":
    main()
