"""Final: Best text artifact removal for fig6_profile_06.

Best approach: v3a (per-label cv2.inpaint) - removes text visually while
preserving label boundaries. Combined with removing label 5 (small bottom
artifact, likely text-related over-segmentation).
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


def detect_text_mask(img):
    """Detect text pixels using darkness + edge cues."""
    import cv2

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Dark text
    dark = gray < 55

    # High Laplacian (sharp edges)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    lap_mask = lap > np.percentile(lap, 80)

    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=15, C=3,
    )
    adaptive_mask = adaptive > 0

    text_mask = dark | (lap_mask & adaptive_mask)
    text_mask = ndimage.binary_dilation(text_mask, iterations=1)
    return text_mask


def inpaint_within_labels(img, labels, text_mask):
    """Inpaint text pixels per-label, preserving boundaries."""
    import cv2

    h, w = img.shape[:2]
    result_img = img.copy()

    for lbl in sorted(set(labels.flatten()) - {0}):
        lbl_mask = labels == lbl
        lbl_text = text_mask & lbl_mask
        if not lbl_text.any():
            continue

        lbl_text_uint8 = lbl_text.astype(np.uint8) * 255

        ys, xs = np.where(lbl_mask)
        y0, y1 = max(0, ys.min() - 5), min(h, ys.max() + 6)
        x0, x1 = max(0, xs.min() - 5), min(w, xs.max() + 6)

        roi_img = result_img[y0:y1, x0:x1].copy()
        roi_mask = lbl_text_uint8[y0:y1, x0:x1]

        if roi_mask.sum() > 0:
            inpainted_roi = cv2.inpaint(roi_img, roi_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            result_img[y0:y1, x0:x1] = inpainted_roi

    return result_img


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

    print(f"Original labels: {np.unique(labels)}, shape: {labels.shape}")
    print(f"Image shape: {img.shape}")

    # Analyze labels for text artifacts
    h, w = labels.shape
    total = h * w
    unique = sorted(set(labels.flatten()) - {0})

    print("\nLabel analysis:")
    text_artifact_labels = []
    for lbl in unique:
        mask = labels == lbl
        area = int(mask.sum())
        area_pct = area / total * 100
        ys, xs = np.where(mask)
        cy = ys.mean()
        print(f"  Label {lbl}: {area} pixels ({area_pct:.3f}%), centroid y={cy:.1f}")

        # Label 5 is very small (1.14%) and at the bottom (y=120 of 124)
        # This is likely a text artifact over-segmentation
        if area < 2000 and cy > h * 0.9:
            text_artifact_labels.append(int(lbl))
            print(f"    -> Likely text artifact (small, near bottom edge)")

    print(f"\nText artifact labels to remove: {text_artifact_labels}")

    # Step 1: Remove text artifact labels (label 5)
    cleaned_labels = remove_labels_by_ids(labels, text_artifact_labels, fill="nearest")
    print(f"Cleaned labels: {np.unique(cleaned_labels)}")

    # Step 2: Detect and inpaint text pixels
    text_mask = detect_text_mask(img)
    print(f"Text mask: {text_mask.sum()} pixels ({text_mask.sum()/(h*w)*100:.2f}%)")

    cleaned_img = inpaint_within_labels(img, cleaned_labels, text_mask)

    # Step 3: Generate overlay with cleaned image + cleaned labels
    overlay = create_overlay(cleaned_img, cleaned_labels)

    # Save deliverables
    # 1. Cleaned labels
    labels_path = OUT_DIR / "labels_text_removed.npz"
    np.savez(labels_path, labels=cleaned_labels)
    print(f"\nSaved cleaned labels: {labels_path}")

    # 2. Overlay image
    overlay_path = OUT_DIR / "overlay_text_removed.jpg"
    Image.fromarray(overlay).save(overlay_path, quality=95)
    print(f"Saved overlay: {overlay_path}")

    # 3. Also save original overlay for comparison
    orig_overlay = create_overlay(img, labels)
    orig_path = OUT_DIR / "overlay_original.jpg"
    Image.fromarray(orig_overlay).save(orig_path, quality=95)
    print(f"Saved original overlay: {orig_path}")

    # 4. Side-by-side comparison
    gap = 10
    h, w = overlay.shape[:2]
    comparison = np.full((h, w * 2 + gap, 3), 32, dtype=np.uint8)
    comparison[:, :w] = orig_overlay
    comparison[:, w + gap:] = overlay
    comp_path = OUT_DIR / "overlay_comparison.jpg"
    Image.fromarray(comparison).save(comp_path, quality=95)
    print(f"Saved comparison: {comp_path}")

    # 5. JSON note
    note = {
        "removed_labels": text_artifact_labels,
        "method": "per_label_cv2_inpaint + remove_labels_by_ids",
        "reason": (
            f"Label {text_artifact_labels} was a small component ("
            f"{int((labels == text_artifact_labels[0]).sum())} px, 1.14%) at the bottom edge "
            f"(y~120 of 124), likely text artifact over-segmentation. "
            f"Removed via remove_labels_by_ids with nearest fill. "
            f"Text annotations (BM, LV-S, PM, LV-N, black dots) were visually removed by "
            f"per-label cv2.inpaint: detected text pixels via darkness (<55) + Laplacian edges + "
            f"adaptive threshold, then inpainted within each label boundary to preserve "
            f"geological layer boundaries."
        ),
        "text_mask_pixels": int(text_mask.sum()),
        "text_mask_percent": round(float(text_mask.sum() / (h * w) * 100), 2),
    }
    note_path = OUT_DIR / "text_fix_note.json"
    with open(note_path, "w") as f:
        json.dump(note, f, indent=2)
    print(f"Saved note: {note_path}")

    print("\n" + "=" * 60)
    print("DELIVERABLES SUMMARY")
    print("=" * 60)
    print(f"1. Cleaned labels:     {labels_path}")
    print(f"2. Overlay:            {overlay_path}")
    print(f"3. Comparison:         {comp_path}")
    print(f"4. JSON note:          {note_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
