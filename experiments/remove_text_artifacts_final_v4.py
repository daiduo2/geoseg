"""Final v4: Conservative text artifact removal.

Problem: Previous text masks were too aggressive (19.69% of pixels), catching
legitimate dark geological features along with text.

Solution: Much more conservative text detection:
1. Only very dark pixels (gray < 45) - text is pure black
2. Only small isolated regions (not large continuous dark areas)
3. Only in regions with high edge density (text has sharp edges)
4. Exclude large connected components (which are geological layers)

Also: The text in this image is specifically:
- "BM" text (left side, dark on pink)
- "LV-S" text (center-left, dark on yellow-green)
- "PM" text (center, dark on yellow-green)
- "LV-N" text (right, dark on green)
- Black dots/symbols scattered

The key: text is SMALL and ISOLATED. Geological dark areas are LARGE and CONTINUOUS.
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


def detect_text_mask_conservative(img, labels):
    """Conservative text detection: only small, isolated, very dark regions."""
    import cv2

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Very dark pixels only (text is nearly black)
    very_dark = gray < 45

    # Connected components of very dark pixels
    cc, num = ndimage.label(very_dark)

    from skimage.measure import regionprops
    regions = regionprops(cc)

    text_mask = np.zeros((h, w), dtype=bool)

    for r in regions:
        area = r.area
        # Text components are small (5-200 pixels)
        # Geological dark areas are large (thousands of pixels)
        if 5 <= area <= 300:
            # Check if this component is surrounded by a single label
            # (text is embedded within a label, not at boundaries between labels)
            comp_mask = cc == r.label
            dilated = ndimage.binary_dilation(comp_mask, iterations=2)
            border = dilated & ~comp_mask

            # Get labels surrounding this component
            border_labels = labels[border]
            border_labels = border_labels[border_labels != 0]

            if len(border_labels) > 0:
                # If mostly surrounded by one label, it's likely text
                vals, counts = np.unique(border_labels, return_counts=True)
                dominant_frac = counts.max() / counts.sum()

                if dominant_frac > 0.6:
                    text_mask[comp_mask] = True

    # Also catch high-edge small components that might be missed
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    high_edge = lap > np.percentile(lap, 92)
    small_high_edge = high_edge & (gray < 80)

    # Only add small high-edge components
    cc2, num2 = ndimage.label(small_high_edge)
    regions2 = regionprops(cc2)
    for r in regions2:
        if 3 <= r.area <= 150:
            comp_mask = cc2 == r.label
            text_mask[comp_mask] = True

    # Dilate slightly to catch text halo
    text_mask = ndimage.binary_dilation(text_mask, iterations=1)

    return text_mask


def create_overlay_text_removed(img, labels, text_mask):
    """Create overlay, then fill text pixels with nearest non-text pixel in overlay."""
    from skimage import segmentation

    # Step 1: Create normal overlay
    overlay = _create_overlay(
        img, labels,
        seeds_rgb=_distinct_colors(int(labels.max()) + 1),
        alpha=0.65, boundary_mode="thin", skip_background=True,
        min_area_frac=0.001, fill_mode="blend",
    )

    # Step 2: Get boundaries to preserve
    boundaries = segmentation.find_boundaries(labels, mode="thin")

    # Step 3: Fill text pixels with nearest non-text, non-boundary pixel
    valid_mask = (~text_mask) & (~boundaries)

    if not valid_mask.any():
        return overlay

    dist, indices = ndimage.distance_transform_edt(~valid_mask, return_indices=True)

    result = overlay.copy()
    ys, xs = np.where(text_mask & ~boundaries)
    if len(ys) > 0:
        result[ys, xs] = overlay[indices[0][ys, xs], indices[1][ys, xs]]

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

    # Detect text mask (conservative)
    text_mask = detect_text_mask_conservative(img, cleaned_labels)
    print(f"Text mask: {text_mask.sum()} pixels ({text_mask.sum()/(h*w)*100:.2f}%)")

    # Generate original overlay
    orig_overlay = _create_overlay(
        img, labels,
        seeds_rgb=_distinct_colors(int(labels.max()) + 1),
        alpha=0.65, boundary_mode="thin", skip_background=True,
        min_area_frac=0.001, fill_mode="blend",
    )

    # Generate cleaned overlay
    cleaned_overlay = create_overlay_text_removed(img, cleaned_labels, text_mask)

    # Save original
    orig_path = OUT_DIR / "overlay_v0_original.jpg"
    Image.fromarray(orig_overlay).save(orig_path, quality=95)
    print(f"\n  Original: {orig_path}")

    # Save cleaned
    cleaned_path = OUT_DIR / "overlay_text_removed.jpg"
    Image.fromarray(cleaned_overlay).save(cleaned_path, quality=95)
    print(f"  Cleaned: {cleaned_path}")

    # Save cleaned labels
    labels_path = OUT_DIR / "labels_text_removed.npz"
    np.savez(labels_path, labels=cleaned_labels)
    print(f"  Labels: {labels_path}")

    # Side-by-side comparison
    gap = 10
    comparison = np.full((h, w * 2 + gap, 3), 32, dtype=np.uint8)
    comparison[:, :w] = orig_overlay
    comparison[:, w + gap:] = cleaned_overlay
    comp_path = OUT_DIR / "overlay_comparison.jpg"
    Image.fromarray(comparison).save(comp_path, quality=95)
    print(f"  Comparison: {comp_path}")

    # JSON note
    note = {
        "removed_labels": text_artifact_labels,
        "method": "conservative_small_component_detection + nearest_fill_in_overlay + remove_labels_by_ids",
        "reason": (
            f"Label {text_artifact_labels} was a small component ({int((labels == text_artifact_labels[0]).sum())} px, "
            f"1.14%) at the bottom edge (y~120 of 124), likely text artifact over-segmentation. "
            f"Removed via remove_labels_by_ids with nearest fill. "
            f"Text annotations (BM, LV-S, PM, LV-N, black dots/symbols) were detected conservatively: "
            f"only very dark connected components (gray<45, 5-300 px) that are surrounded by a single "
            f"dominant label (>60% of neighbors), plus small high-edge components (Laplacian>92nd percentile, "
            f"3-150 px, gray<80). This avoids catching large geological dark regions. Text pixels were then "
            f"replaced with the nearest non-text pixel in the already-blended overlay, preserving boundary lines."
        ),
        "text_mask_pixels": int(text_mask.sum()),
        "text_mask_percent": round(float(text_mask.sum() / (h * w) * 100), 2),
    }
    note_path = OUT_DIR / "text_fix_note.json"
    with open(note_path, "w") as f:
        json.dump(note, f, indent=2)
    print(f"  Note: {note_path}")

    print("\n" + "=" * 60)
    print("DONE.")
    print("=" * 60)


if __name__ == "__main__":
    main()
