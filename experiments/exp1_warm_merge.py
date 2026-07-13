"""Experiment 1: warm-color label merging on panel 3.

v4_kmeans split the plume funnel into two labels (4 and 5) due to color gradient.
This experiment uses HSV warm-color detection to identify and merge warm-dominant
labels into a single plume label.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from geoseg.modules.segment_engines._shared import _create_overlay
from geoseg.modules.segment_engines.v4_kmeans import _reorder_labels_by_median_y


def is_warm_dominant(label_mask: np.ndarray, hsv_image: np.ndarray) -> bool:
    """Check if a label is majority warm-colored.

    Warm = hue in [0, 60] (red to yellow) with saturation > 40.
    This catches vivid plume colors while excluding low-saturation
    gray/brown transitions and cool blue/green background.
    """
    label_hsv = hsv_image[label_mask]
    if len(label_hsv) == 0:
        return False

    hues = label_hsv[:, 0]
    sats = label_hsv[:, 1]

    # Warm hue range: 0-60 degrees in OpenCV HSV
    warm_hue = (hues >= 0) & (hues <= 60)
    # Red wrap-around at 180
    red_wrap = hues >= 170
    # Minimum saturation to avoid gray/brown
    saturated = sats > 40

    warm_and_saturated = ((warm_hue | red_wrap) & saturated)
    return warm_and_saturated.sum() / len(label_hsv) > 0.35


def main():
    # Paths
    enhanced_path = "runs/3d_schematic_correct_e2e/panel_3_front/00_enhanced.jpg"
    labels_path = "runs/3d_schematic_correct_e2e/panel_3_front/labels_primary.npz"
    out_overlay = "runs/tubular_panel3/exp1_warm_merge.jpg"
    out_labels = "runs/tubular_panel3/exp1_warm_merge.npz"

    # Load data
    enhanced = np.array(Image.open(enhanced_path).convert("RGB"))
    labels_orig = np.load(labels_path)["labels"]

    print(f"Enhanced image shape: {enhanced.shape}")
    print(f"Labels shape: {labels_orig.shape}")

    # Work on a reordered copy for analysis (top-to-bottom ordering)
    labels_reordered = _reorder_labels_by_median_y(labels_orig)

    # Build mapping from reordered -> original for reference
    unique_orig = sorted(np.unique(labels_orig))
    median_y_orig = {}
    for lbl in unique_orig:
        ys = np.where(labels_orig == lbl)[0]
        median_y_orig[lbl] = np.median(ys) if len(ys) > 0 else 0

    print(f"\nOriginal labels by median y (top to bottom):")
    for lbl, my in sorted(median_y_orig.items(), key=lambda x: x[1]):
        print(f"  Label {lbl}: median_y={my:.0f}")

    # Convert to HSV for warm-color detection
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)

    # Identify warm-dominant labels on ORIGINAL labels
    warm_labels = []
    label_pixel_counts = {}
    label_stats = {}

    for lbl in unique_orig:
        mask = labels_orig == lbl
        count = int(mask.sum())
        label_pixel_counts[lbl] = count

        label_hsv = hsv[mask]
        mean_hsv = label_hsv.mean(axis=0)
        median_h = np.median(label_hsv[:, 0])

        is_warm = is_warm_dominant(mask, hsv)
        if is_warm:
            warm_labels.append(lbl)

        label_stats[lbl] = {
            "mean_hsv": mean_hsv,
            "median_h": median_h,
            "is_warm": is_warm,
        }

    print(f"\nPer-label analysis (original labels):")
    for lbl in unique_orig:
        stats = label_stats[lbl]
        warm_marker = " *** WARM ***" if stats["is_warm"] else ""
        print(f"  Label {lbl}: count={label_pixel_counts[lbl]:,}, "
              f"median_h={stats['median_h']:.1f}, "
              f"mean_hsv=[{stats['mean_hsv'][0]:.1f}, {stats['mean_hsv'][1]:.1f}, {stats['mean_hsv'][2]:.1f}]{warm_marker}")

    print(f"\nWarm-dominant labels to merge: {warm_labels}")
    non_warm = sorted([l for l in unique_orig if l not in warm_labels])
    print(f"Non-warm labels: {non_warm}")

    # Merge warm labels into label 1 (plume)
    merged = labels_orig.copy()
    for lbl in warm_labels:
        merged[merged == lbl] = 1

    # Relabel non-plume labels starting from 2 (skip 0 to avoid confusion with background)
    next_label = 2
    for lbl in non_warm:
        if lbl == 1:
            # Original label 1 was not warm, shift it
            merged[merged == lbl] = next_label
            next_label += 1
        elif lbl != 1:
            merged[merged == lbl] = next_label
            next_label += 1

    final_unique = sorted(np.unique(merged))
    print(f"\nFinal labels after merge: {final_unique}")

    # Build palette from mean colors of each final label
    palette = np.zeros((len(final_unique), 3), dtype=np.uint8)
    for i, lbl in enumerate(final_unique):
        mask = merged == lbl
        if mask.any():
            palette[i] = enhanced[mask].mean(axis=0).astype(np.uint8)

    # Create overlay
    overlay = _create_overlay(enhanced, merged, palette, alpha=0.65)

    # Save outputs
    Image.fromarray(overlay).save(out_overlay, quality=95)
    np.savez_compressed(out_labels, labels=merged)

    print(f"\nSaved overlay to: {out_overlay}")
    print(f"Saved labels to: {out_labels}")

    # Summary
    print("\n" + "=" * 50)
    print("EXPERIMENT 1 SUMMARY")
    print("=" * 50)
    print(f"Merged labels: {warm_labels} -> label 1 (plume)")
    plume_count = int((merged == 1).sum())
    print(f"Plume pixel count: {plume_count:,}")
    print(f"Total labels before: {len(unique_orig)}")
    print(f"Total labels after:  {len(final_unique)}")
    for lbl in final_unique:
        count = int((merged == lbl).sum())
        print(f"  Label {lbl}: {count:,} pixels")
    print("\nVisual result:")
    print("  The warm-colored plume funnel regions are now unified under label 1.")
    print("  Non-warm layers (background, cool-colored zones) are relabeled 2, 3, ...")


if __name__ == "__main__":
    main()
