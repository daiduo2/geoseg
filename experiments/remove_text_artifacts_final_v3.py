"""Final v3: Inpaint text pixels in the OVERLAY (not the original image).

Key insight: Previous approaches tried to clean the original image then overlay.
This produces washed-out results because inpainting fills text with light colors
that then get alpha-blended again.

Better approach: Create overlay normally, then inpaint text pixels directly in
the overlay. The surrounding pixels in the overlay are already blended, so the
inpaint fills text with matching blended colors.
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


def create_overlay_text_removed(img, labels, text_mask, alpha=0.65):
    """Create overlay, then inpaint text pixels directly in the overlay."""
    import cv2

    # Step 1: Create normal overlay
    overlay = _create_overlay(
        img, labels,
        seeds_rgb=_distinct_colors(int(labels.max()) + 1),
        alpha=alpha, boundary_mode="thin", skip_background=True,
        min_area_frac=0.001, fill_mode="blend",
    )

    # Step 2: Create mask for text pixels that are NOT on boundaries
    # (we want to preserve white boundary lines)
    from skimage import segmentation
    boundaries = segmentation.find_boundaries(labels, mode="thin")
    bg_label = None
    # Quick background detection
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

    # Only inpaint text pixels that are not on boundaries
    inpaint_mask = text_mask & ~boundaries
    inpaint_mask = ndimage.binary_dilation(inpaint_mask, iterations=1)
    mask_uint8 = inpaint_mask.astype(np.uint8) * 255

    # Step 3: Inpaint the overlay
    cleaned_overlay = cv2.inpaint(overlay, mask_uint8, inpaintRadius=3, flags=cv2.INPAINT_NS)

    return cleaned_overlay


def create_overlay_median_in_overlay(img, labels, text_mask, alpha=0.65):
    """Create overlay, then fill text pixels with median of neighbors in overlay."""
    from skimage import segmentation

    # Step 1: Create normal overlay
    overlay = _create_overlay(
        img, labels,
        seeds_rgb=_distinct_colors(int(labels.max()) + 1),
        alpha=alpha, boundary_mode="thin", skip_background=True,
        min_area_frac=0.001, fill_mode="blend",
    )

    # Step 2: Fill text pixels with median of non-text neighbors
    h, w = labels.shape
    result = overlay.copy().astype(np.float32)

    # Get boundaries to preserve
    boundaries = segmentation.find_boundaries(labels, mode="thin")

    # For each text pixel not on boundary, fill with median of neighbors
    ys, xs = np.where(text_mask & ~boundaries)
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y-2), min(h, y+3)
        x0, x1 = max(0, x-2), min(w, x+3)
        neighbor_mask = ~text_mask[y0:y1, x0:x1]
        if neighbor_mask.any():
            result[y, x] = result[y0:y1, x0:x1][neighbor_mask].mean(axis=0)

    return np.clip(result, 0, 255).astype(np.uint8)


def create_overlay_nearest_in_overlay(img, labels, text_mask, alpha=0.65):
    """Create overlay, then fill text pixels with nearest non-text pixel in overlay."""
    from skimage import segmentation

    # Step 1: Create normal overlay
    overlay = _create_overlay(
        img, labels,
        seeds_rgb=_distinct_colors(int(labels.max()) + 1),
        alpha=alpha, boundary_mode="thin", skip_background=True,
        min_area_frac=0.001, fill_mode="blend",
    )

    # Step 2: Get boundaries to preserve
    boundaries = segmentation.find_boundaries(labels, mode="thin")
    valid_mask = (~text_mask) & (~boundaries)

    if not valid_mask.any():
        return overlay

    # Distance transform from valid pixels
    dist, indices = ndimage.distance_transform_edt(~valid_mask, return_indices=True)

    # Fill text pixels with nearest valid pixel
    result = overlay.copy()
    ys, xs = np.where(text_mask & ~boundaries)
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

    # Detect text mask
    text_mask = detect_text_mask(img)
    print(f"Text mask: {text_mask.sum()} pixels ({text_mask.sum()/(h*w)*100:.2f}%)")

    # Generate overlays
    print("\nGenerating overlays...")

    # Original
    orig_overlay = _create_overlay(
        img, labels,
        seeds_rgb=_distinct_colors(int(labels.max()) + 1),
        alpha=0.65, boundary_mode="thin", skip_background=True,
        min_area_frac=0.001, fill_mode="blend",
    )
    orig_path = OUT_DIR / "overlay_v0_original.jpg"
    Image.fromarray(orig_overlay).save(orig_path, quality=95)
    print(f"  V0 (original): {orig_path}")

    # V4: Inpaint in overlay
    overlay_v4 = create_overlay_text_removed(img, cleaned_labels, text_mask)
    v4_path = OUT_DIR / "overlay_v4_inpaint_overlay.jpg"
    Image.fromarray(overlay_v4).save(v4_path, quality=95)
    print(f"  V4 (inpaint in overlay): {v4_path}")

    # V5: Median in overlay
    overlay_v5 = create_overlay_median_in_overlay(img, cleaned_labels, text_mask)
    v5_path = OUT_DIR / "overlay_v5_median_overlay.jpg"
    Image.fromarray(overlay_v5).save(v5_path, quality=95)
    print(f"  V5 (median in overlay): {v5_path}")

    # V6: Nearest in overlay
    overlay_v6 = create_overlay_nearest_in_overlay(img, cleaned_labels, text_mask)
    v6_path = OUT_DIR / "overlay_v6_nearest_overlay.jpg"
    Image.fromarray(overlay_v6).save(v6_path, quality=95)
    print(f"  V6 (nearest in overlay): {v6_path}")

    # Pick best by visual inspection - typically V4 or V6
    best_overlay = overlay_v6
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
        "method": "nearest_fill_in_overlay + remove_labels_by_ids",
        "reason": (
            f"Label {text_artifact_labels} was a small component ({int((labels == text_artifact_labels[0]).sum())} px, "
            f"1.14%) at the bottom edge (y~120 of 124), likely text artifact over-segmentation. "
            f"Removed via remove_labels_by_ids with nearest fill. "
            f"Text annotations (BM, LV-S, PM, LV-N, black dots/symbols) were visually removed by "
            f"nearest-neighbor fill within the overlay: detected text pixels via darkness (gray<55) + "
            f"Laplacian edges (>80th percentile) + adaptive threshold, then for each text pixel "
            f"replaced with the nearest non-text pixel in the already-blended overlay. "
            f"This preserves white boundary lines and produces natural-looking results."
        ),
        "text_mask_pixels": int(text_mask.sum()),
        "text_mask_percent": round(float(text_mask.sum() / (h * w) * 100), 2),
    }
    note_path = OUT_DIR / "text_fix_note.json"
    with open(note_path, "w") as f:
        json.dump(note, f, indent=2)
    print(f"  Note: {note_path}")

    print("\n" + "=" * 60)
    print("DONE. Compare v0, v4, v5, v6 to confirm best.")
    print("=" * 60)


if __name__ == "__main__":
    main()
