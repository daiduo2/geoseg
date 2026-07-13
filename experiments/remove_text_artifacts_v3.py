"""Experiment v3: Visual text removal - inpaint within labels, preserve boundaries.

Key insight: Previous approaches changed label assignments, which distorted
geological boundaries. The correct approach is to keep labels intact but
visually remove text by filling text pixels with colors from their surrounding
pixels within the SAME label. This is a visual cleanup, not a re-segmentation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from geoseg.modules.segment_engines._shared import _create_overlay, _distinct_colors

LABELS_PATH = Path("/Users/daiduo2/geoseg/runs/feng_fig6_final_v5/fig6_profile_06/labels.npz")
IMAGE_PATH = Path("/Users/daiduo2/geoseg/runs/feng_fig6_final_v4/crop_tests/fig6_profile_06_cropped.jpg")
OUT_DIR = Path("/Users/daiduo2/geoseg/runs/feng_fig6_comparisons_v6/fig6_profile_06")


def load_data():
    labels = np.load(LABELS_PATH, allow_pickle=True)["labels"]
    img = np.array(Image.open(IMAGE_PATH).convert("RGB"))
    return labels, img


def save_overlay_on_image(img, path, title=""):
    """Save image directly (for inpainted results)."""
    Image.fromarray(img).save(path, quality=95)
    print(f"  Saved: {path} {title}")


def create_overlay_custom(panel_rgb, labels, alpha=0.65):
    """Create overlay using existing utility."""
    return _create_overlay(
        panel_rgb, labels,
        seeds_rgb=_distinct_colors(int(labels.max()) + 1),
        alpha=alpha, boundary_mode="thin", skip_background=True,
        min_area_frac=0.001, fill_mode="blend",
    )


def detect_text_mask(img, labels):
    """Detect text pixels using multiple cues."""
    import cv2

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Cue 1: Very dark pixels (black text/symbols)
    dark = gray < 55

    # Cue 2: High Laplacian (sharp edges of text)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    lap_mask = lap > np.percentile(lap, 80)

    # Cue 3: Adaptive threshold (local contrast)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=15, C=3,
    )
    adaptive_mask = adaptive > 0

    # Combine: dark OR (edgy AND adaptive)
    text_mask = dark | (lap_mask & adaptive_mask)

    # Dilate slightly to catch text halo
    text_mask = ndimage.binary_dilation(text_mask, iterations=1)

    return text_mask


def approach_v3a_inpaint_within_labels(labels, img):
    """Inpaint text pixels within each label separately, preserving boundaries."""
    print("\n--- Approach V3a: Inpaint within labels (per-label cv2.inpaint) ---")
    import cv2

    h, w = img.shape[:2]
    text_mask = detect_text_mask(img, labels)
    print(f"  Text mask: {text_mask.sum()} pixels ({text_mask.sum()/(h*w)*100:.2f}%)")

    result_img = img.copy()

    for lbl in sorted(set(labels.flatten()) - {0}):
        lbl_mask = labels == lbl
        # Text pixels within this label
        lbl_text = text_mask & lbl_mask
        if not lbl_text.any():
            continue

        # Create a mask just for this label's text
        lbl_text_uint8 = lbl_text.astype(np.uint8) * 255

        # Extract the region bounding box for efficiency
        ys, xs = np.where(lbl_mask)
        y0, y1 = max(0, ys.min() - 5), min(h, ys.max() + 6)
        x0, x1 = max(0, xs.min() - 5), min(w, xs.max() + 6)

        roi_img = result_img[y0:y1, x0:x1].copy()
        roi_mask = lbl_text_uint8[y0:y1, x0:x1]

        if roi_mask.sum() > 0:
            inpainted_roi = cv2.inpaint(roi_img, roi_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            result_img[y0:y1, x0:x1] = inpainted_roi

    overlay = create_overlay_custom(result_img, labels)
    save_overlay_on_image(overlay, OUT_DIR / "v3a_inpaint_within_labels.jpg", "(inpaint within labels)")
    return result_img, labels  # Return cleaned image, original labels


def approach_v3b_median_fill_within_labels(labels, img):
    """Fill text pixels with median of non-text neighbors in same label."""
    print("\n--- Approach V3b: Median fill within labels ---")

    h, w = img.shape[:2]
    text_mask = detect_text_mask(img, labels)
    print(f"  Text mask: {text_mask.sum()} pixels ({text_mask.sum()/(h*w)*100:.2f}%)")

    result_img = img.copy().astype(np.float32)

    for lbl in sorted(set(labels.flatten()) - {0}):
        lbl_mask = labels == lbl
        lbl_text = text_mask & lbl_mask
        if not lbl_text.any():
            continue

        # For each text pixel, find non-text neighbors in same label
        ys, xs = np.where(lbl_text)
        for y, x in zip(ys, xs):
            # Look at 5x5 neighborhood
            y0, y1 = max(0, y-2), min(h, y+3)
            x0, x1 = max(0, x-2), min(w, x+3)
            neighbors = result_img[y0:y1, x0:x1]
            neighbor_mask = (~text_mask[y0:y1, x0:x1]) & lbl_mask[y0:y1, x0:x1]
            if neighbor_mask.any():
                result_img[y, x] = neighbors[neighbor_mask].mean(axis=0)

    result_img = np.clip(result_img, 0, 255).astype(np.uint8)
    overlay = create_overlay_custom(result_img, labels)
    save_overlay_on_image(overlay, OUT_DIR / "v3b_median_fill.jpg", "(median fill within labels)")
    return result_img, labels


def approach_v3c_distance_fill_within_labels(labels, img):
    """Fill text pixels using distance-transform nearest non-text pixel in same label."""
    print("\n--- Approach V3c: Distance-transform fill within labels ---")

    h, w = img.shape[:2]
    text_mask = detect_text_mask(img, labels)
    print(f"  Text mask: {text_mask.sum()} pixels ({text_mask.sum()/(h*w)*100:.2f}%)")

    result_img = img.copy()

    for lbl in sorted(set(labels.flatten()) - {0}):
        lbl_mask = labels == lbl
        lbl_text = text_mask & lbl_mask
        if not lbl_text.any():
            continue

        # Valid pixels = non-text within this label
        valid = lbl_mask & ~text_mask
        if not valid.any():
            continue

        # Distance transform from valid pixels
        dist, indices = ndimage.distance_transform_edt(~valid, return_indices=True)

        # For text pixels in this label, copy from nearest valid pixel
        ys, xs = np.where(lbl_text)
        result_img[ys, xs] = img[indices[0][ys, xs], indices[1][ys, xs]]

    overlay = create_overlay_custom(result_img, labels)
    save_overlay_on_image(overlay, OUT_DIR / "v3c_distance_fill.jpg", "(distance fill within labels)")
    return result_img, labels


def approach_v3d_inpaint_full(labels, img):
    """Full image inpaint then overlay original labels."""
    print("\n--- Approach V3d: Full image inpaint + original labels overlay ---")
    import cv2

    h, w = img.shape[:2]
    text_mask = detect_text_mask(img, labels)
    print(f"  Text mask: {text_mask.sum()} pixels ({text_mask.sum()/(h*w)*100:.2f}%)")

    mask_uint8 = text_mask.astype(np.uint8) * 255
    inpainted = cv2.inpaint(img, mask_uint8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    overlay = create_overlay_custom(inpainted, labels)
    save_overlay_on_image(overlay, OUT_DIR / "v3d_full_inpaint.jpg", "(full inpaint)")
    return inpainted, labels


def approach_v3e_conservative_inpaint(labels, img):
    """More conservative: only very dark pixels, smaller inpaint radius."""
    print("\n--- Approach V3e: Conservative inpaint (only very dark) ---")
    import cv2

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Only very dark pixels
    text_mask = gray < 50
    text_mask = ndimage.binary_dilation(text_mask, iterations=1)

    print(f"  Text mask: {text_mask.sum()} pixels ({text_mask.sum()/(h*w)*100:.2f}%)")

    mask_uint8 = text_mask.astype(np.uint8) * 255
    inpainted = cv2.inpaint(img, mask_uint8, inpaintRadius=2, flags=cv2.INPAINT_NS)

    overlay = create_overlay_custom(inpainted, labels)
    save_overlay_on_image(overlay, OUT_DIR / "v3e_conservative_inpaint.jpg", "(conservative inpaint)")
    return inpainted, labels


def approach_v3f_two_pass_inpaint(labels, img):
    """Two-pass: first remove dark text, then remove light text/artifacts."""
    print("\n--- Approach V3f: Two-pass inpaint (dark then light) ---")
    import cv2

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Pass 1: Dark text
    dark_mask = gray < 55

    # Pass 2: Light text/artifacts (some annotations are light)
    light_mask = gray > 200

    # Pass 3: High edge response areas that are small
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    edge_mask = lap > np.percentile(lap, 85)

    # Combine
    text_mask = dark_mask | light_mask | edge_mask
    text_mask = ndimage.binary_dilation(text_mask, iterations=1)

    print(f"  Text mask: {text_mask.sum()} pixels ({text_mask.sum()/(h*w)*100:.2f}%)")

    mask_uint8 = text_mask.astype(np.uint8) * 255
    inpainted = cv2.inpaint(img, mask_uint8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    overlay = create_overlay_custom(inpainted, labels)
    save_overlay_on_image(overlay, OUT_DIR / "v3f_two_pass.jpg", "(two-pass inpaint)")
    return inpainted, labels


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels, img = load_data()

    print(f"Labels: {np.unique(labels)}, shape: {labels.shape}")
    print(f"Image shape: {img.shape}")

    # Save original overlay for comparison
    orig_overlay = create_overlay_custom(img, labels)
    save_overlay_on_image(orig_overlay, OUT_DIR / "v3_original.jpg", "(ORIGINAL)")

    # Run all v3 approaches
    results = {}
    results["v3a"] = approach_v3a_inpaint_within_labels(labels, img)
    results["v3b"] = approach_v3b_median_fill_within_labels(labels, img)
    results["v3c"] = approach_v3c_distance_fill_within_labels(labels, img)
    results["v3d"] = approach_v3d_inpaint_full(labels, img)
    results["v3e"] = approach_v3e_conservative_inpaint(labels, img)
    results["v3f"] = approach_v3f_two_pass_inpaint(labels, img)

    print(f"\n\nAll v3 approaches saved to {OUT_DIR}")
    print("Compare and pick the best one.")


if __name__ == "__main__":
    main()
