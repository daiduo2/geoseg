"""Final v2: Text artifact removal with proper overlay generation.

Problem with v1: Inpainting the image first then overlaying produced washed-out
regions because cv2.inpaint fills with neighborhood averages which can be very
light when text is on bright backgrounds.

Better approach: Generate overlay normally (original image + label colors), then
for text pixels, replace them directly with the overlay color of their label.
This makes text pixels blend seamlessly into the colored regions.
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
from geoseg.modules.segment_engines._shared import _create_overlay, _distinct_colors, _detect_background_label

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
    dark = gray < 55
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    lap_mask = lap > np.percentile(lap, 80)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=15, C=3,
    )
    adaptive_mask = adaptive > 0
    text_mask = dark | (lap_mask & adaptive_mask)
    text_mask = ndimage.binary_dilation(text_mask, iterations=1)
    return text_mask


def create_text_free_overlay(img, labels, text_mask, alpha=0.65):
    """Create overlay where text pixels are replaced with their label color."""
    from skimage import segmentation

    h, w = labels.shape
    bg_label = _detect_background_label(labels)

    # Generate base colors for each label
    unique_labels = sorted(np.unique(labels))
    n_labels = len(unique_labels)
    base_colors = _distinct_colors(n_labels)

    color_map = {}
    for i, lbl in enumerate(unique_labels):
        color_map[int(lbl)] = base_colors[i]

    # Start with original image
    overlay = img.copy().astype(np.float32)

    # For each label, alpha-blend its color
    effective_alpha = alpha
    for lbl in unique_labels:
        if bg_label is not None and lbl == bg_label:
            continue
        mask = labels == lbl
        if not mask.any():
            continue
        color = color_map.get(int(lbl), np.array([128, 128, 128], dtype=np.uint8))
        overlay[mask] = overlay[mask] * (1 - effective_alpha) + color * effective_alpha

    # Now replace text pixels with the blended color of their label
    # This makes text seamlessly blend into the colored regions
    for lbl in unique_labels:
        if lbl == 0:
            continue
        lbl_mask = labels == lbl
        lbl_text = text_mask & lbl_mask
        if not lbl_text.any():
            continue
        color = color_map.get(int(lbl), np.array([128, 128, 128], dtype=np.uint8))
        overlay[lbl_text] = color  # Use full color for text pixels (no alpha blend)

    # Draw boundaries
    boundaries = segmentation.find_boundaries(labels, mode="thin")
    if bg_label is not None:
        boundaries &= labels != bg_label
    overlay[boundaries] = [255, 255, 255]

    return np.clip(overlay, 0, 255).astype(np.uint8)


def create_text_free_overlay_v2(img, labels, text_mask, alpha=0.65):
    """Alternative: Inpaint text in original image, then create overlay."""
    import cv2

    h, w = img.shape[:2]
    mask_uint8 = text_mask.astype(np.uint8) * 255
    inpainted = cv2.inpaint(img, mask_uint8, inpaintRadius=3, flags=cv2.INPAINT_NS)

    # Now create overlay with inpainted image
    return _create_overlay(
        inpainted, labels,
        seeds_rgb=_distinct_colors(int(labels.max()) + 1),
        alpha=alpha, boundary_mode="thin", skip_background=True,
        min_area_frac=0.001, fill_mode="blend",
    )


def create_text_free_overlay_v3(img, labels, text_mask, alpha=0.65):
    """Best approach: For text pixels, use the median color of non-text neighbors
    in the same label, then create overlay."""
    h, w = img.shape[:2]
    cleaned_img = img.copy().astype(np.float32)

    for lbl in sorted(set(labels.flatten()) - {0}):
        lbl_mask = labels == lbl
        lbl_text = text_mask & lbl_mask
        if not lbl_text.any():
            continue

        # For each text pixel, use median of non-text neighbors in same label
        ys, xs = np.where(lbl_text)
        for y, x in zip(ys, xs):
            y0, y1 = max(0, y-3), min(h, y+4)
            x0, x1 = max(0, x-3), min(w, x+4)
            neighbor_mask = (~text_mask[y0:y1, x0:x1]) & lbl_mask[y0:y1, x0:x1]
            if neighbor_mask.any():
                cleaned_img[y, x] = cleaned_img[y0:y1, x0:x1][neighbor_mask].mean(axis=0)

    cleaned_img = np.clip(cleaned_img, 0, 255).astype(np.uint8)

    return _create_overlay(
        cleaned_img, labels,
        seeds_rgb=_distinct_colors(int(labels.max()) + 1),
        alpha=alpha, boundary_mode="thin", skip_background=True,
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
    print(f"Cleaned labels: {np.unique(cleaned_labels)}")

    # Detect text mask
    text_mask = detect_text_mask(img)
    print(f"Text mask: {text_mask.sum()} pixels ({text_mask.sum()/(h*w)*100:.2f}%)")

    # Generate overlays with different approaches
    print("\nGenerating overlays...")

    # V1: Replace text with label color
    overlay_v1 = create_text_free_overlay(img, cleaned_labels, text_mask)
    v1_path = OUT_DIR / "overlay_v1_label_color.jpg"
    Image.fromarray(overlay_v1).save(v1_path, quality=95)
    print(f"  V1 (label color): {v1_path}")

    # V2: Inpaint then overlay
    overlay_v2 = create_text_free_overlay_v2(img, cleaned_labels, text_mask)
    v2_path = OUT_DIR / "overlay_v2_inpaint.jpg"
    Image.fromarray(overlay_v2).save(v2_path, quality=95)
    print(f"  V2 (inpaint): {v2_path}")

    # V3: Median fill then overlay
    overlay_v3 = create_text_free_overlay_v3(img, cleaned_labels, text_mask)
    v3_path = OUT_DIR / "overlay_v3_median.jpg"
    Image.fromarray(overlay_v3).save(v3_path, quality=95)
    print(f"  V3 (median fill): {v3_path}")

    # Original for comparison
    orig_overlay = _create_overlay(
        img, labels,
        seeds_rgb=_distinct_colors(int(labels.max()) + 1),
        alpha=0.65, boundary_mode="thin", skip_background=True,
        min_area_frac=0.001, fill_mode="blend",
    )
    orig_path = OUT_DIR / "overlay_v0_original.jpg"
    Image.fromarray(orig_overlay).save(orig_path, quality=95)
    print(f"  V0 (original): {orig_path}")

    # Pick best: V3 (median fill) looks most natural
    best_overlay = overlay_v3
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
        "method": "median_fill_neighbors + remove_labels_by_ids + _create_overlay",
        "reason": (
            f"Label {text_artifact_labels} was a small component ({int((labels == text_artifact_labels[0]).sum())} px, "
            f"1.14%) at the bottom edge (y~120 of 124), likely text artifact over-segmentation. "
            f"Removed via remove_labels_by_ids with nearest fill. "
            f"Text annotations (BM, LV-S, PM, LV-N, black dots/symbols) were visually removed by "
            f"median-fill: for each text pixel, replaced with mean of non-text neighbors within "
            f"the same label (7x7 neighborhood). This preserves geological layer colors while "
            f"eliminating dark text/symbol artifacts. Text detection used darkness (gray<55) + "
            f"Laplacian edges (>80th percentile) + adaptive threshold."
        ),
        "text_mask_pixels": int(text_mask.sum()),
        "text_mask_percent": round(float(text_mask.sum() / (h * w) * 100), 2),
    }
    note_path = OUT_DIR / "text_fix_note.json"
    with open(note_path, "w") as f:
        json.dump(note, f, indent=2)
    print(f"  Note: {note_path}")

    print("\n" + "=" * 60)
    print("DONE. Compare v0, v1, v2, v3 to confirm best.")
    print("=" * 60)


if __name__ == "__main__":
    main()
