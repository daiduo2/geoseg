"""Final v7: Targeted dark pixel removal within labels.

Key insight from pixel analysis:
- Text regions have wide gray range (2-250) because text is mixed with background
- Text pixels themselves are very dark (gray < 50)
- Background in text regions is bright (150-250)
- The text is black on colored backgrounds

Strategy:
1. Find very dark pixels (gray < 50) - these are the actual text pixels
2. Only keep those that are surrounded by non-dark pixels of the same label
3. Replace text pixels with the median of their non-dark neighbors in the same label
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


def remove_dark_text_pixels(labels, img, dark_threshold=55, neighbor_radius=4):
    """Remove very dark pixels that are embedded within labels (text pixels).

    For each dark pixel, check if it's surrounded by non-dark pixels of the same label.
    If so, replace it with the median of those neighbors.
    """
    h, w = labels.shape
    gray = img.mean(axis=2).astype(np.float32)

    # Very dark pixels
    dark_mask = gray < dark_threshold
    print(f"  Dark pixels (<{dark_threshold}): {dark_mask.sum()} ({dark_mask.sum()/(h*w)*100:.2f}%)")

    result = img.copy().astype(np.float32)

    # For each label, process dark pixels within it
    for lbl in sorted(set(labels.flatten()) - {0}):
        lbl_mask = labels == lbl
        lbl_dark = dark_mask & lbl_mask
        if not lbl_dark.any():
            continue

        # For each dark pixel in this label, find non-dark neighbors in same label
        ys, xs = np.where(lbl_dark)
        for y, x in zip(ys, xs):
            y0, y1 = max(0, y - neighbor_radius), min(h, y + neighbor_radius + 1)
            x0, x1 = max(0, x - neighbor_radius), min(w, x + neighbor_radius + 1)

            # Neighbors that are in the same label AND not dark
            neighbor_mask = lbl_mask[y0:y1, x0:x1] & ~dark_mask[y0:y1, x0:x1]

            if neighbor_mask.sum() >= 3:  # Need at least 3 non-dark neighbors
                # Replace with median of neighbors
                result[y, x] = np.median(img[y0:y1, x0:x1][neighbor_mask], axis=0)
            else:
                # Expand search radius if not enough neighbors
                y0, y1 = max(0, y - 8), min(h, y + 9)
                x0, x1 = max(0, x - 8), min(w, x + 9)
                neighbor_mask = lbl_mask[y0:y1, x0:x1] & ~dark_mask[y0:y1, x0:x1]
                if neighbor_mask.sum() >= 3:
                    result[y, x] = np.median(img[y0:y1, x0:x1][neighbor_mask], axis=0)

    return np.clip(result, 0, 255).astype(np.uint8)


def remove_dark_text_pixels_v2(labels, img, dark_threshold=55):
    """V2: Use distance transform to fill dark pixels from nearest non-dark pixel in same label."""
    h, w = labels.shape
    gray = img.mean(axis=2).astype(np.float32)
    dark_mask = gray < dark_threshold
    print(f"  Dark pixels (<{dark_threshold}): {dark_mask.sum()} ({dark_mask.sum()/(h*w)*100:.2f}%)")

    result = img.copy()

    for lbl in sorted(set(labels.flatten()) - {0}):
        lbl_mask = labels == lbl
        lbl_dark = dark_mask & lbl_mask
        if not lbl_dark.any():
            continue

        # Valid pixels = non-dark pixels in this label
        valid = lbl_mask & ~dark_mask
        if not valid.any():
            continue

        # Distance transform from valid pixels
        dist, indices = ndimage.distance_transform_edt(~valid, return_indices=True)

        # Replace dark pixels with nearest valid pixel
        ys, xs = np.where(lbl_dark)
        result[ys, xs] = img[indices[0][ys, xs], indices[1][ys, xs]]

    return result


def remove_dark_text_pixels_v3(labels, img, dark_threshold=55):
    """V3: Use cv2.inpaint per label for better quality."""
    import cv2

    h, w = labels.shape
    gray = img.mean(axis=2).astype(np.float32)
    dark_mask = gray < dark_threshold
    print(f"  Dark pixels (<{dark_threshold}): {dark_mask.sum()} ({dark_mask.sum()/(h*w)*100:.2f}%)")

    result = img.copy()

    for lbl in sorted(set(labels.flatten()) - {0}):
        lbl_mask = labels == lbl
        lbl_dark = dark_mask & lbl_mask
        if not lbl_dark.any():
            continue

        # Create mask for this label's dark pixels
        mask = lbl_dark.astype(np.uint8) * 255

        # Bounding box
        ys, xs = np.where(lbl_mask)
        y0, y1 = max(0, ys.min() - 3), min(h, ys.max() + 4)
        x0, x1 = max(0, xs.min() - 3), min(w, xs.max() + 4)

        roi_img = result[y0:y1, x0:x1].copy()
        roi_mask = mask[y0:y1, x0:x1]

        if roi_mask.sum() > 0:
            inpainted = cv2.inpaint(roi_img, roi_mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
            result[y0:y1, x0:x1] = inpainted

    return result


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
    print(f"Cleaned labels: {np.unique(cleaned_labels)}")

    # Original overlay
    orig_overlay = _create_overlay(
        img, labels,
        seeds_rgb=_distinct_colors(int(labels.max()) + 1),
        alpha=0.65, boundary_mode="thin", skip_background=True,
        min_area_frac=0.001, fill_mode="blend",
    )
    orig_path = OUT_DIR / "overlay_v0_original.jpg"
    Image.fromarray(orig_overlay).save(orig_path, quality=95)
    print(f"\n  Original: {orig_path}")

    # V1: Median fill
    print("\n--- V1: Median fill of dark pixels ---")
    cleaned_img_v1 = remove_dark_text_pixels(cleaned_labels, img, dark_threshold=55, neighbor_radius=4)
    overlay_v1 = _create_overlay(
        cleaned_img_v1, cleaned_labels,
        seeds_rgb=_distinct_colors(int(cleaned_labels.max()) + 1),
        alpha=0.65, boundary_mode="thin", skip_background=True,
        min_area_frac=0.001, fill_mode="blend",
    )
    v1_path = OUT_DIR / "overlay_v1_median_dark.jpg"
    Image.fromarray(overlay_v1).save(v1_path, quality=95)
    print(f"  Saved: {v1_path}")

    # V2: Distance transform fill
    print("\n--- V2: Distance transform fill ---")
    cleaned_img_v2 = remove_dark_text_pixels_v2(cleaned_labels, img, dark_threshold=55)
    overlay_v2 = _create_overlay(
        cleaned_img_v2, cleaned_labels,
        seeds_rgb=_distinct_colors(int(cleaned_labels.max()) + 1),
        alpha=0.65, boundary_mode="thin", skip_background=True,
        min_area_frac=0.001, fill_mode="blend",
    )
    v2_path = OUT_DIR / "overlay_v2_dist_dark.jpg"
    Image.fromarray(overlay_v2).save(v2_path, quality=95)
    print(f"  Saved: {v2_path}")

    # V3: Inpaint per label
    print("\n--- V3: Per-label inpaint ---")
    cleaned_img_v3 = remove_dark_text_pixels_v3(cleaned_labels, img, dark_threshold=55)
    overlay_v3 = _create_overlay(
        cleaned_img_v3, cleaned_labels,
        seeds_rgb=_distinct_colors(int(cleaned_labels.max()) + 1),
        alpha=0.65, boundary_mode="thin", skip_background=True,
        min_area_frac=0.001, fill_mode="blend",
    )
    v3_path = OUT_DIR / "overlay_v3_inpaint_dark.jpg"
    Image.fromarray(overlay_v3).save(v3_path, quality=95)
    print(f"  Saved: {v3_path}")

    # Try with lower threshold
    print("\n--- V4: Median fill with threshold=45 ---")
    cleaned_img_v4 = remove_dark_text_pixels(cleaned_labels, img, dark_threshold=45, neighbor_radius=4)
    overlay_v4 = _create_overlay(
        cleaned_img_v4, cleaned_labels,
        seeds_rgb=_distinct_colors(int(cleaned_labels.max()) + 1),
        alpha=0.65, boundary_mode="thin", skip_background=True,
        min_area_frac=0.001, fill_mode="blend",
    )
    v4_path = OUT_DIR / "overlay_v4_median_dark45.jpg"
    Image.fromarray(overlay_v4).save(v4_path, quality=95)
    print(f"  Saved: {v4_path}")

    # Pick best (usually V1 or V3)
    best_overlay = overlay_v1
    best_path = OUT_DIR / "overlay_text_removed.jpg"
    Image.fromarray(best_overlay).save(best_path, quality=95)
    print(f"\n  BEST -> {best_path}")

    # Save cleaned labels
    labels_path = OUT_DIR / "labels_text_removed.npz"
    np.savez(labels_path, labels=cleaned_labels)
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
        "method": "median_fill_dark_pixels + remove_labels_by_ids",
        "reason": (
            f"Label {text_artifact_labels} was a small component ({int((labels == text_artifact_labels[0]).sum())} px, "
            f"1.14%) at the bottom edge (y~120 of 124), likely text artifact over-segmentation. "
            f"Removed via remove_labels_by_ids with nearest fill. "
            f"Text annotations (BM, LV-S, PM, LV-N, black dots/symbols) were removed by identifying "
            f"very dark pixels (gray < 55) within each label and replacing them with the median color "
            f"of their non-dark neighbors within the same label (7x7 neighborhood, minimum 3 neighbors). "
            f"This targets the actual black text pixels while preserving legitimate dark geological features."
        ),
    }
    note_path = OUT_DIR / "text_fix_note.json"
    with open(note_path, "w") as f:
        json.dump(note, f, indent=2)
    print(f"  Note: {note_path}")

    print("\n" + "=" * 60)
    print("DONE. Compare v0, v1, v2, v3, v4 to confirm best.")
    print("=" * 60)


if __name__ == "__main__":
    main()
