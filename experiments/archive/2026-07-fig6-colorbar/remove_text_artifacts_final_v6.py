"""Final v6: Direct text pixel replacement in overlay with label colors.

Key insight: Text pixels are embedded within larger label regions (not separate
components). The only way to remove them is pixel-level replacement.

Approach:
1. Create normal overlay (original image + label colors alpha-blended)
2. Detect text pixels (dark + high edge response)
3. For each text pixel, replace it with the SOLID label color (not blended)
   This makes text pixels match the overlay color of their label perfectly.

This is a VISUAL fix for the overlay only. The labels themselves are unchanged.
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


def detect_text_mask(img, labels):
    """Detect text pixels: dark, high edge, small isolated regions."""
    import cv2

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Dark pixels (text is black/dark on colored backgrounds)
    dark = gray < 80

    # High edge response (text has sharp edges)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    high_edge = lap > np.percentile(lap, 75)

    # Text pixels are dark AND have high edge response
    text_candidates = dark & high_edge

    # But we need to exclude large continuous dark regions (geological layers)
    # Keep only small connected components
    cc, num = ndimage.label(text_candidates)

    text_mask = np.zeros((h, w), dtype=bool)
    from skimage.measure import regionprops
    for r in regionprops(cc):
        # Text components: small (3-500 px), not huge geological regions
        if 3 <= r.area <= 500:
            comp_mask = cc == r.label
            text_mask[comp_mask] = True

    # Dilate slightly to catch text halo
    text_mask = ndimage.binary_dilation(text_mask, iterations=1)

    return text_mask


def create_clean_overlay(img, labels, text_mask, alpha=0.65):
    """Create overlay where text pixels are replaced with solid label colors."""
    from skimage import segmentation

    h, w = labels.shape

    # Detect background label
    bg_label = None
    edge_margin = max(3, min(h, w) // 50)
    edge_mask = np.zeros((h, w), dtype=bool)
    edge_mask[:edge_margin, :] = True
    edge_mask[-edge_margin:, :] = True
    edge_mask[:, :edge_margin] = True
    edge_mask[:, -edge_margin:] = True
    for lbl in np.unique(labels):
        mask = labels == lbl
        edge_count = int(mask[edge_mask].sum())
        total_count = int(mask.sum())
        if total_count > 0:
            edge_ratio = edge_count / edge_mask.sum()
            area_ratio = total_count / (h * w)
            if edge_ratio > 0.25 and area_ratio > 0.08:
                bg_label = int(lbl)
                break

    # Generate colors
    unique_labels = sorted(np.unique(labels))
    n_labels = len(unique_labels)
    base_colors = _distinct_colors(n_labels)

    color_map = {}
    for i, lbl in enumerate(unique_labels):
        color_map[int(lbl)] = base_colors[i]

    # Start with original image
    overlay = img.copy().astype(np.float32)

    # Alpha blend each label's color
    for lbl in unique_labels:
        if bg_label is not None and lbl == bg_label:
            continue
        mask = labels == lbl
        if not mask.any():
            continue
        color = color_map.get(int(lbl), np.array([128, 128, 128], dtype=np.uint8))
        overlay[mask] = overlay[mask] * (1 - alpha) + color * alpha

    # For text pixels, use FULL label color (no alpha blend with original)
    # This completely replaces the dark text with the label color
    for lbl in unique_labels:
        if lbl == 0:
            continue
        lbl_mask = labels == lbl
        lbl_text = text_mask & lbl_mask
        if not lbl_text.any():
            continue
        color = color_map.get(int(lbl), np.array([128, 128, 128], dtype=np.uint8))
        overlay[lbl_text] = color  # Full color, no alpha

    # Draw white boundaries
    boundaries = segmentation.find_boundaries(labels, mode="thin")
    if bg_label is not None:
        boundaries &= labels != bg_label
    overlay[boundaries] = [255, 255, 255]

    return np.clip(overlay, 0, 255).astype(np.uint8)


def create_clean_overlay_v2(img, labels, text_mask, alpha=0.65):
    """V2: For text pixels, use the median of non-text neighbors in the same label
    from the ALREADY BLENDED overlay (not the original image)."""
    from skimage import segmentation

    h, w = labels.shape

    # First create the normal blended overlay
    bg_label = None
    edge_margin = max(3, min(h, w) // 50)
    edge_mask = np.zeros((h, w), dtype=bool)
    edge_mask[:edge_margin, :] = True
    edge_mask[-edge_margin:, :] = True
    edge_mask[:, :edge_margin] = True
    edge_mask[:, -edge_margin:] = True
    for lbl in np.unique(labels):
        mask = labels == lbl
        edge_count = int(mask[edge_mask].sum())
        total_count = int(mask.sum())
        if total_count > 0:
            edge_ratio = edge_count / edge_mask.sum()
            area_ratio = total_count / (h * w)
            if edge_ratio > 0.25 and area_ratio > 0.08:
                bg_label = int(lbl)
                break

    unique_labels = sorted(np.unique(labels))
    n_labels = len(unique_labels)
    base_colors = _distinct_colors(n_labels)

    color_map = {}
    for i, lbl in enumerate(unique_labels):
        color_map[int(lbl)] = base_colors[i]

    # Create blended overlay
    overlay = img.copy().astype(np.float32)
    for lbl in unique_labels:
        if bg_label is not None and lbl == bg_label:
            continue
        mask = labels == lbl
        if not mask.any():
            continue
        color = color_map.get(int(lbl), np.array([128, 128, 128], dtype=np.uint8))
        overlay[mask] = overlay[mask] * (1 - alpha) + color * alpha

    overlay = np.clip(overlay, 0, 255)

    # Now for text pixels, fill with median of non-text neighbors in same label
    result = overlay.copy()
    for lbl in unique_labels:
        if lbl == 0:
            continue
        lbl_mask = labels == lbl
        lbl_text = text_mask & lbl_mask
        if not lbl_text.any():
            continue

        ys, xs = np.where(lbl_text)
        for y, x in zip(ys, xs):
            y0, y1 = max(0, y-3), min(h, y+4)
            x0, x1 = max(0, x-3), min(w, x+4)
            neighbor_mask = (~text_mask[y0:y1, x0:x1]) & lbl_mask[y0:y1, x0:x1]
            if neighbor_mask.any():
                result[y, x] = overlay[y0:y1, x0:x1][neighbor_mask].mean(axis=0)

    # Draw boundaries
    boundaries = segmentation.find_boundaries(labels, mode="thin")
    if bg_label is not None:
        boundaries &= labels != bg_label
    result[boundaries] = [255, 255, 255]

    return np.clip(result, 0, 255).astype(np.uint8)


def create_clean_overlay_v3(img, labels, text_mask, alpha=0.65):
    """V3: Two-pass. First replace text with solid label color, then blur slightly
    to make transitions natural."""
    import cv2

    # Start with V1 approach
    overlay = create_clean_overlay(img, labels, text_mask, alpha)

    # Slight Gaussian blur to smooth text replacement edges
    blurred = cv2.GaussianBlur(overlay, (3, 3), 0.5)

    # But keep boundaries sharp
    from skimage import segmentation
    boundaries = segmentation.find_boundaries(labels, mode="thin")
    bg_label = None
    h, w = labels.shape
    edge_margin = max(3, min(h, w) // 50)
    edge_mask = np.zeros((h, w), dtype=bool)
    edge_mask[:edge_margin, :] = True
    edge_mask[-edge_margin:, :] = True
    edge_mask[:, :edge_margin] = True
    edge_mask[:, -edge_margin:] = True
    for lbl in np.unique(labels):
        mask = labels == lbl
        edge_count = int(mask[edge_mask].sum())
        total_count = int(mask.sum())
        if total_count > 0:
            edge_ratio = edge_count / edge_mask.sum()
            area_ratio = total_count / (h * w)
            if edge_ratio > 0.25 and area_ratio > 0.08:
                bg_label = int(lbl)
                break
    if bg_label is not None:
        boundaries &= labels != bg_label

    # Only blur text regions, keep rest as-is
    result = overlay.copy()
    text_dilated = ndimage.binary_dilation(text_mask, iterations=2)
    result[text_dilated] = blurred[text_dilated]
    result[boundaries] = [255, 255, 255]

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

    # Detect text mask
    text_mask = detect_text_mask(img, cleaned_labels)
    print(f"Text mask: {text_mask.sum()} pixels ({text_mask.sum()/(h*w)*100:.2f}%)")

    # Generate original overlay
    orig_overlay = _create_overlay(
        img, labels,
        seeds_rgb=_distinct_colors(int(labels.max()) + 1),
        alpha=0.65, boundary_mode="thin", skip_background=True,
        min_area_frac=0.001, fill_mode="blend",
    )
    orig_path = OUT_DIR / "overlay_v0_original.jpg"
    Image.fromarray(orig_overlay).save(orig_path, quality=95)
    print(f"\n  Original: {orig_path}")

    # V1: Solid color replacement
    overlay_v1 = create_clean_overlay(img, cleaned_labels, text_mask)
    v1_path = OUT_DIR / "overlay_v1_solid_color.jpg"
    Image.fromarray(overlay_v1).save(v1_path, quality=95)
    print(f"  V1 (solid color): {v1_path}")

    # V2: Median of neighbors in overlay
    overlay_v2 = create_clean_overlay_v2(img, cleaned_labels, text_mask)
    v2_path = OUT_DIR / "overlay_v2_median_overlay.jpg"
    Image.fromarray(overlay_v2).save(v2_path, quality=95)
    print(f"  V2 (median overlay): {v2_path}")

    # V3: Solid + blur
    overlay_v3 = create_clean_overlay_v3(img, cleaned_labels, text_mask)
    v3_path = OUT_DIR / "overlay_v3_blur.jpg"
    Image.fromarray(overlay_v3).save(v3_path, quality=95)
    print(f"  V3 (blur): {v3_path}")

    # Pick best (V2 usually looks most natural)
    best_overlay = overlay_v2
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
        "method": "median_fill_in_blended_overlay + remove_labels_by_ids",
        "reason": (
            f"Label {text_artifact_labels} was a small component ({int((labels == text_artifact_labels[0]).sum())} px, "
            f"1.14%) at the bottom edge (y~120 of 124), likely text artifact over-segmentation. "
            f"Removed via remove_labels_by_ids with nearest fill. "
            f"Text annotations (BM, LV-S, PM, LV-N, black dots/symbols) were detected as small "
            f"connected components (3-500 px) that are both dark (gray<80) and have high Laplacian "
            f"edge response (>75th percentile). Text pixels were then replaced with the mean color "
            f"of non-text neighbors within the same label from the already-alpha-blended overlay. "
            f"This preserves the natural blended colors while eliminating dark text artifacts."
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
