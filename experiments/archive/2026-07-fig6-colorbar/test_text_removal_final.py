#!/usr/bin/env python3
"""Final text artifact removal for fig6_profile_07.

Best approach: Combine small component removal with targeted dark pixel inpainting.
The main artifacts are:
1. Label 3 component 1 (area=168): small cyan blob in upper-left (caused by text/dots)
2. Label 3 component 4 (area=173): thin strip at bottom-left
3. Dark text pixels within label 1 (the "LV-N" imprint)

Approach 9 (comprehensive) was best. This script produces the final deliverables.
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


def remove_text_artifacts(labels, img_rgb):
    """Remove text artifacts from segmentation labels.

    Steps:
    1. Remove small label 3 fragments (area < 1000 px)
    2. Inpaint dark pixels within each label (text imprint removal)
    3. Final small component cleanup
    """
    result = labels.copy()
    h, w = labels.shape
    gray = img_rgb.mean(axis=2)

    removed_info = []

    # Step 1: Remove small components from label 3
    lbl3_mask = labels == 3
    lbl3_labeled, lbl3_num = ndimage.label(lbl3_mask)
    for i in range(1, lbl3_num + 1):
        comp = lbl3_labeled == i
        area = comp.sum()
        if area < 1000:
            ys, xs = np.where(comp)
            removed_info.append({
                "type": "small_component",
                "label": 3,
                "component_id": i,
                "area": int(area),
                "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                "reason": "fragment below 1000px threshold",
            })
            result[comp] = 0

    # Fill removed label 3 components from nearest non-zero label
    removed_mask = result == 0
    valid_for_fill = (labels != 0) & (~removed_mask)
    if valid_for_fill.any():
        _, indices = ndimage.distance_transform_edt(~valid_for_fill, return_indices=True)
        rr, cc = np.where(removed_mask & (labels != 0))
        result[rr, cc] = labels[indices[0][rr, cc], indices[1][rr, cc]]

    # Step 2: Inpaint dark pixels within each label
    # Use a moderate threshold to catch text without over-removing
    dark_threshold = 85
    dark_mask = gray < dark_threshold
    very_dark = gray < 55

    for lbl in sorted(set(result.flatten()) - {0}):
        lbl_mask = result == lbl
        lbl_dark = lbl_mask & (dark_mask | very_dark)

        if not lbl_dark.any():
            continue

        # Use morphological opening to identify text structures
        from skimage.morphology import opening, disk
        opened = opening(lbl_dark, footprint=disk(2))
        text_pixels = lbl_dark & (~opened)
        text_pixels = text_pixels | (lbl_mask & very_dark)

        if not text_pixels.any():
            continue

        n_text = int(text_pixels.sum())

        # Dilate slightly to catch edges
        dilated = ndimage.binary_dilation(text_pixels, iterations=1)

        # Valid pixels: same label, not text
        valid = lbl_mask & (~dilated)
        if not valid.any():
            valid = (~dilated) & (result != 0)

        if not valid.any():
            continue

        _, indices = ndimage.distance_transform_edt(~valid, return_indices=True)
        rr, cc = np.where(text_pixels)
        result[rr, cc] = result[indices[0][rr, cc], indices[1][rr, cc]]

        removed_info.append({
            "type": "dark_text_inpaint",
            "label": int(lbl),
            "pixels": n_text,
            "reason": "dark text pixels reassigned to neighbor",
        })

    # Step 3: Final small component filter
    result = filter_small_components(result, min_area_ratio=0.002, fill="nearest")

    return result, removed_info


def main():
    base_dir = Path("/Users/daiduo2/geoseg")
    labels_path = base_dir / "runs/feng_fig6_final_v5/fig6_profile_07/labels.npz"
    img_path = base_dir / "runs/feng_fig6_final_v4/crop_tests/fig6_profile_07_cropped.jpg"
    out_dir = base_dir / "runs/feng_fig6_comparisons_v6/fig6_profile_07"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = np.load(labels_path)["labels"]
    img_rgb = np.array(Image.open(img_path).convert("RGB"))

    print(f"Input labels shape: {labels.shape}, unique: {np.unique(labels)}")
    print(f"Input image shape: {img_rgb.shape}")

    # Apply text artifact removal
    cleaned_labels, removed_info = remove_text_artifacts(labels, img_rgb)

    print(f"\nCleaned labels unique: {np.unique(cleaned_labels)}")
    print(f"Removed artifacts: {len(removed_info)}")
    for info in removed_info:
        print(f"  {info}")

    # Save cleaned labels
    labels_out = out_dir / "labels_text_removed.npz"
    np.savez(labels_out, labels=cleaned_labels)
    print(f"\nSaved cleaned labels to {labels_out}")

    # Create and save overlay
    overlay = create_overlay(img_rgb, cleaned_labels)
    overlay_out = out_dir / "overlay_text_removed.jpg"
    Image.fromarray(overlay).save(overlay_out)
    print(f"Saved overlay to {overlay_out}")

    # Also create a comparison image (original vs cleaned)
    from matplotlib import pyplot as plt
    orig_overlay = create_overlay(img_rgb, labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(orig_overlay)
    axes[0].set_title("Original (with artifacts)")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Cleaned (text artifacts removed)")
    axes[1].axis("off")

    plt.tight_layout()
    comparison_out = out_dir / "overlay_comparison.jpg"
    plt.savefig(comparison_out, dpi=150, bbox_inches="tight")
    print(f"Saved comparison to {comparison_out}")

    # Write JSON note
    removed_labels = [info for info in removed_info if info["type"] == "small_component"]
    note = {
        "removed_labels": [3],
        "removed_components": removed_info,
        "method": "remove_text_artifacts: small_component_removal + dark_pixel_inpaint + filter_small_components",
        "reason": "Label 3 had 5 connected components; 2 were small artifacts (168px and 173px) caused by text annotation 'LV-N' and black dots. These were removed and filled from nearest neighbor. Dark text pixels within labels were also inpainted to smooth the imprint.",
    }

    note_out = out_dir / "text_fix_note.json"
    with open(note_out, "w") as f:
        json.dump(note, f, indent=2)
    print(f"Saved note to {note_out}")

    print("\n--- Done ---")
    return cleaned_labels, removed_info


if __name__ == "__main__":
    main()
