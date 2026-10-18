#!/usr/bin/env python3
"""Experiment with text artifact removal for fig6_profile_07.

Candidate approaches:
1. filter_small_components - remove tiny components (catches the small blobs)
2. remove_labels_by_ids - directly remove label 3's small components by identifying them
3. merge_labels_by_ids + filter_small_components - merge then filter
4. Dark-pixel mask based removal - identify text pixels by darkness and reassign
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.measure import label, regionprops

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from geoseg.modules.post_process.merge import (
    filter_small_components,
    merge_labels_by_ids,
    remove_labels_by_ids,
)
from geoseg.modules.segment_engines._shared import _create_overlay


def create_overlay(panel_rgb, labels, fill_mode="blend"):
    """Helper to create overlay using existing _create_overlay."""
    return _create_overlay(
        panel_rgb,
        labels,
        np.zeros((1, 3), dtype=np.uint8),  # dummy seeds
        alpha=0.65,
        boundary_mode="thin",
        skip_background=True,
        min_area_frac=0.0005,  # Very small to not merge things we want to see
        fill_mode=fill_mode,
    )


def approach_1_filter_small(labels, img_rgb, min_area_ratio=0.003):
    """Approach 1: Simply filter out very small components globally."""
    return filter_small_components(labels, min_area_ratio=min_area_ratio, fill="nearest")


def approach_2_remove_by_darkness(labels, img_rgb, dark_threshold=80):
    """Approach 2: Identify dark pixels (text/symbols) and reassign to nearest non-dark label.

    Uses a dark-pixel mask to find text annotations, then reassigns those pixels
    to the nearest non-text label neighbor.
    """
    gray = img_rgb.mean(axis=2)
    dark_mask = gray < dark_threshold

    # Also identify very small components within each label
    result = labels.copy()
    h, w = labels.shape
    total = h * w

    # Find small components per label and mark them for removal
    small_component_mask = np.zeros_like(labels, dtype=bool)
    for lbl in sorted(set(labels.flatten()) - {0}):
        lbl_mask = labels == lbl
        labeled, num = ndimage.label(lbl_mask)
        for i in range(1, num + 1):
            comp = labeled == i
            area = comp.sum()
            bbox = regionprops((comp).astype(int))[0].bbox
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            fill_ratio = area / max(bbox_area, 1)

            # Criteria for text artifact:
            # - Very small area (< 0.3% of image)
            # - OR has high fill ratio (compact blob) AND small area
            # - OR contains many dark pixels
            dark_in_comp = (dark_mask & comp).sum()
            dark_ratio = dark_in_comp / max(area, 1)

            is_artifact = (
                area < total * 0.003  # Very small
                or (area < total * 0.01 and fill_ratio > 0.6 and dark_ratio > 0.1)
                or dark_ratio > 0.15  # Contains significant dark pixels
            )

            if is_artifact:
                small_component_mask[comp] = True
                print(f"  Marked artifact: label={lbl}, comp area={area}, dark_ratio={dark_ratio:.3f}, fill={fill_ratio:.3f}")

    if not small_component_mask.any():
        return result

    # Reassign artifact pixels to nearest non-artifact, non-background label
    valid_mask = (~small_component_mask) & (labels != 0)
    if not valid_mask.any():
        result[small_component_mask] = 0
        return result

    _, indices = ndimage.distance_transform_edt(~valid_mask, return_indices=True)
    rr, cc = np.where(small_component_mask)
    result[rr, cc] = labels[indices[0][rr, cc], indices[1][rr, cc]]
    return result


def approach_3_component_wise_filter(labels, img_rgb, min_area=200, max_bbox_fill=0.4):
    """Approach 3: Component-wise analysis with smarter heuristics.

    For each connected component, decide if it's a text artifact based on:
    - Absolute area threshold
    - Aspect ratio (text is usually wide and thin or small and compact)
    - Fill ratio of bounding box
    - Presence of dark pixels
    """
    gray = img_rgb.mean(axis=2)
    dark_mask = gray < 80

    result = labels.copy()
    h, w = labels.shape

    removed_components = []

    for lbl in sorted(set(labels.flatten()) - {0}):
        lbl_mask = labels == lbl
        labeled, num = ndimage.label(lbl_mask)

        for i in range(1, num + 1):
            comp = labeled == i
            area = comp.sum()
            ys, xs = np.where(comp)
            y0, y1 = ys.min(), ys.max()
            x0, x1 = xs.min(), xs.max()
            bbox_h = y1 - y0 + 1
            bbox_w = x1 - x0 + 1
            bbox_area = bbox_h * bbox_w
            fill_ratio = area / max(bbox_area, 1)
            aspect = max(bbox_w, bbox_h) / max(min(bbox_w, bbox_h), 1)
            dark_in_comp = (dark_mask & comp).sum()
            dark_ratio = dark_in_comp / max(area, 1)

            # Text artifact criteria:
            # 1. Very small absolute area (< 200 pixels)
            # 2. Small area with high dark pixel ratio
            # 3. Compact with high fill ratio but small total area
            is_artifact = (
                area < min_area
                or (area < 500 and dark_ratio > 0.1 and fill_ratio > 0.5)
                or (area < 300 and aspect < 3 and fill_ratio > 0.5)
            )

            if is_artifact:
                removed_components.append({
                    "label": int(lbl),
                    "area": int(area),
                    "bbox": [int(x0), int(y0), int(x1), int(y1)],
                    "dark_ratio": float(dark_ratio),
                    "fill_ratio": float(fill_ratio),
                })
                result[comp] = 0  # Mark as background first

    # Now fill removed pixels from nearest non-background, non-removed label
    removed_mask = result == 0
    # But we need to exclude original background too
    original_bg = labels == 0
    valid_mask = (~removed_mask) | original_bg
    # Actually, we want to fill removed components from nearest VALID (non-zero, non-removed)
    valid_for_fill = (labels != 0) & (~removed_mask)

    if not valid_for_fill.any():
        return result

    _, indices = ndimage.distance_transform_edt(~valid_for_fill, return_indices=True)
    rr, cc = np.where(removed_mask & (~original_bg))
    result[rr, cc] = labels[indices[0][rr, cc], indices[1][rr, cc]]

    return result, removed_components


def approach_4_smart_merge_and_filter(labels, img_rgb):
    """Approach 4: Two-pass approach.
    Pass 1: Merge labels that are spatially adjacent and similar in color
    Pass 2: Filter remaining small components
    """
    # First, try to merge the fragmented label 3 components into their neighbors
    result = labels.copy()

    # Analyze label 3 components specifically
    lbl3_mask = labels == 3
    lbl3_labeled, lbl3_num = ndimage.label(lbl3_mask)

    gray = img_rgb.mean(axis=2)
    dark_mask = gray < 80

    removed_labels = []

    for i in range(1, lbl3_num + 1):
        comp = lbl3_labeled == i
        area = comp.sum()
        ys, xs = np.where(comp)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        bbox_area = (y1 - y0 + 1) * (x1 - x0 + 1)
        fill_ratio = area / max(bbox_area, 1)
        dark_ratio = (dark_mask & comp).sum() / max(area, 1)

        # Small or text-heavy components get removed
        if area < 500 or dark_ratio > 0.1 or (area < 1000 and fill_ratio > 0.6):
            removed_labels.append({
                "label": 3,
                "comp": i,
                "area": int(area),
                "dark_ratio": float(dark_ratio),
                "fill_ratio": float(fill_ratio),
            })
            result[comp] = 0

    # Fill removed pixels from nearest non-zero label
    removed_mask = result == 0
    valid_mask = (result != 0) | (labels == 0)
    # Actually, fill from original labels where they were non-zero and not removed
    valid_for_fill = (labels != 0) & (~removed_mask)

    if valid_for_fill.any():
        _, indices = ndimage.distance_transform_edt(~valid_for_fill, return_indices=True)
        rr, cc = np.where(removed_mask & (labels != 0))
        result[rr, cc] = labels[indices[0][rr, cc], indices[1][rr, cc]]

    # Now apply general small component filter
    result = filter_small_components(result, min_area_ratio=0.002, fill="nearest")

    return result, removed_labels


def approach_5_remove_label_3_fragments(labels, img_rgb):
    """Approach 5: Targeted removal of label 3's small fragments.

    Label 3 has 5 components. The main geological ones are large (1344, 1402 px).
    The artifacts are small (168, 173 px) or thin strips (1490 px at bottom).
    Remove components < 1000 px and reassign.
    """
    result = labels.copy()
    lbl3_mask = labels == 3
    lbl3_labeled, lbl3_num = ndimage.label(lbl3_mask)

    removed = []
    for i in range(1, lbl3_num + 1):
        comp = lbl3_labeled == i
        area = comp.sum()
        if area < 1000:
            removed.append({"comp": i, "area": int(area)})
            result[comp] = 0

    # Fill removed pixels from nearest non-zero, non-removed label
    removed_mask = result == 0
    valid_for_fill = (labels != 0) & (~removed_mask)

    if valid_for_fill.any():
        _, indices = ndimage.distance_transform_edt(~valid_for_fill, return_indices=True)
        rr, cc = np.where(removed_mask & (labels != 0))
        result[rr, cc] = labels[indices[0][rr, cc], indices[1][rr, cc]]

    return result, removed


def main():
    base_dir = Path("/Users/daiduo2/geoseg")
    labels_path = base_dir / "runs/feng_fig6_final_v5/fig6_profile_07/labels.npz"
    img_path = base_dir / "runs/feng_fig6_final_v4/crop_tests/fig6_profile_07_cropped.jpg"
    out_dir = base_dir / "runs/feng_fig6_comparisons_v6/fig6_profile_07"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = np.load(labels_path)["labels"]
    img_rgb = np.array(Image.open(img_path).convert("RGB"))

    print(f"Labels shape: {labels.shape}, unique: {np.unique(labels)}")
    print(f"Image shape: {img_rgb.shape}")

    # Create original overlay for comparison
    orig_overlay = create_overlay(img_rgb, labels)
    Image.fromarray(orig_overlay).save(out_dir / "overlay_original.jpg")
    print(f"Saved original overlay to {out_dir / 'overlay_original.jpg'}")

    approaches = []

    # Approach 1: Simple small component filter
    print("\n--- Approach 1: filter_small_components (min_area_ratio=0.003) ---")
    a1_labels = approach_1_filter_small(labels, img_rgb, min_area_ratio=0.003)
    a1_overlay = create_overlay(img_rgb, a1_labels)
    Image.fromarray(a1_overlay).save(out_dir / "overlay_approach1_filter_small.jpg")
    print(f"Unique labels after: {np.unique(a1_labels)}")
    approaches.append(("approach1_filter_small", a1_labels, a1_overlay))

    # Approach 2: Dark-pixel based removal
    print("\n--- Approach 2: remove_by_darkness ---")
    a2_labels = approach_2_remove_by_darkness(labels, img_rgb, dark_threshold=80)
    a2_overlay = create_overlay(img_rgb, a2_labels)
    Image.fromarray(a2_overlay).save(out_dir / "overlay_approach2_darkness.jpg")
    print(f"Unique labels after: {np.unique(a2_labels)}")
    approaches.append(("approach2_darkness", a2_labels, a2_overlay))

    # Approach 3: Component-wise smart filter
    print("\n--- Approach 3: component_wise_filter ---")
    a3_labels, a3_removed = approach_3_component_wise_filter(labels, img_rgb, min_area=200)
    a3_overlay = create_overlay(img_rgb, a3_labels)
    Image.fromarray(a3_overlay).save(out_dir / "overlay_approach3_component_wise.jpg")
    print(f"Unique labels after: {np.unique(a3_labels)}")
    print(f"Removed components: {len(a3_removed)}")
    for r in a3_removed:
        print(f"  {r}")
    approaches.append(("approach3_component_wise", a3_labels, a3_overlay, a3_removed))

    # Approach 4: Smart merge + filter
    print("\n--- Approach 4: smart_merge_and_filter ---")
    a4_labels, a4_removed = approach_4_smart_merge_and_filter(labels, img_rgb)
    a4_overlay = create_overlay(img_rgb, a4_labels)
    Image.fromarray(a4_overlay).save(out_dir / "overlay_approach4_smart_merge.jpg")
    print(f"Unique labels after: {np.unique(a4_labels)}")
    print(f"Removed from label 3: {len(a4_removed)}")
    for r in a4_removed:
        print(f"  {r}")
    approaches.append(("approach4_smart_merge", a4_labels, a4_overlay, a4_removed))

    # Approach 5: Targeted label 3 fragment removal
    print("\n--- Approach 5: remove_label_3_fragments (< 1000 px) ---")
    a5_labels, a5_removed = approach_5_remove_label_3_fragments(labels, img_rgb)
    a5_overlay = create_overlay(img_rgb, a5_labels)
    Image.fromarray(a5_overlay).save(out_dir / "overlay_approach5_label3_fragments.jpg")
    print(f"Unique labels after: {np.unique(a5_labels)}")
    print(f"Removed label 3 components: {len(a5_removed)}")
    for r in a5_removed:
        print(f"  {r}")
    approaches.append(("approach5_label3_fragments", a5_labels, a5_overlay, a5_removed))

    # Create comparison grid
    print("\n--- Creating comparison grid ---")
    n = len(approaches) + 1
    fig_rows = (n + 2) // 3
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(fig_rows, 3, figsize=(18, 6 * fig_rows))
    axes = axes.flatten() if n > 3 else [axes] if fig_rows == 1 else axes.flatten()

    # Original
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
    plt.savefig(out_dir / "comparison_grid.jpg", dpi=150, bbox_inches="tight")
    print(f"Saved comparison grid to {out_dir / 'comparison_grid.jpg'}")

    # Save all candidate label arrays for inspection
    for name, lbls, ovly, *rest in approaches:
        np.savez(out_dir / f"labels_{name}.npz", labels=lbls)

    print(f"\nAll outputs saved to {out_dir}")
    return approaches, orig_overlay


if __name__ == "__main__":
    main()
