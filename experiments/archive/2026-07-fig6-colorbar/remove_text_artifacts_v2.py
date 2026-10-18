"""Experiment v2: Target text pixel removal within labels.

The text artifacts (BM, LV-S, PM, LV-N, black dots) are dark pixels embedded
within larger color labels. Previous approaches that remove small components
don't work because text is part of large labels.

Strategy: Detect text pixels by darkness + edge characteristics, then reassign
them to the nearest non-text label using distance transform.
"""
from __future__ import annotations

import json
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


def save_overlay(labels, img, path, title=""):
    overlay = _create_overlay(
        img, labels, seeds_rgb=_distinct_colors(int(labels.max()) + 1),
        alpha=0.65, boundary_mode="thin", skip_background=True,
        min_area_frac=0.001, fill_mode="blend",
    )
    Image.fromarray(overlay).save(path, quality=95)
    print(f"  Saved: {path} {title}")
    return overlay


def save_pure_mask(labels, path):
    """Save a pure label mask without text overlay for clean comparison."""
    h, w = labels.shape
    n = int(labels.max()) + 1
    colors = _distinct_colors(n)
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    for lbl in np.unique(labels):
        if lbl == 0:
            continue
        mask[labels == lbl] = colors[int(lbl) % len(colors)]
    Image.fromarray(mask).save(path, quality=95)
    print(f"  Saved pure mask: {path}")
    return mask


def approach_v2a_text_mask_reassign(labels, img):
    """Detect text pixels and reassign to nearest non-text label."""
    print("\n--- Approach V2a: Text mask + nearest label reassignment ---")
    h, w = img.shape[:2]
    gray = img.mean(axis=2).astype(np.float32)

    # Dark text: very dark pixels
    dark = gray < 55

    # High local variance (text edges)
    from skimage.filters import sobel
    edges = np.abs(sobel(gray))
    edgy = edges > np.percentile(edges, 85)

    # Text candidates: dark AND edgy
    text_mask = dark | edgy
    text_mask = ndimage.binary_dilation(text_mask, iterations=1)

    print(f"  Text mask pixels: {text_mask.sum()} ({text_mask.sum() / (h*w) * 100:.2f}%)")

    # Only reassign text pixels that are within non-background labels
    valid_mask = (~text_mask) & (labels != 0)
    if not valid_mask.any():
        return labels.copy()

    _, indices = ndimage.distance_transform_edt(~valid_mask, return_indices=True)
    result = labels.copy()
    rr, cc = np.where(text_mask & (labels != 0))
    result[rr, cc] = labels[indices[0][rr, cc], indices[1][rr, cc]]

    save_overlay(result, img, OUT_DIR / "v2a_text_mask.jpg", "(text mask reassigned)")
    return result


def approach_v2b_adaptive_text_threshold(labels, img):
    """Use adaptive threshold to detect text, then reassign."""
    print("\n--- Approach V2b: Adaptive threshold text detection ---")
    import cv2

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Adaptive threshold: text is darker than local neighborhood
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=15, C=3,
    )

    # Also Laplacian for sharp edges
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    lap_mask = lap > np.percentile(lap, 80)

    text_mask = (adaptive > 0) | lap_mask
    text_mask = ndimage.binary_dilation(text_mask, iterations=1)

    print(f"  Text mask pixels: {text_mask.sum()} ({text_mask.sum() / (h*w) * 100:.2f}%)")

    valid_mask = (~text_mask) & (labels != 0)
    if not valid_mask.any():
        return labels.copy()

    _, indices = ndimage.distance_transform_edt(~valid_mask, return_indices=True)
    result = labels.copy()
    rr, cc = np.where(text_mask & (labels != 0))
    result[rr, cc] = labels[indices[0][rr, cc], indices[1][rr, cc]]

    save_overlay(result, img, OUT_DIR / "v2b_adaptive_text.jpg", "(adaptive threshold)")
    return result


def approach_v2c_dark_blob_removal(labels, img):
    """Remove dark blobs (text + symbols) by color similarity to black."""
    print("\n--- Approach V2c: Dark blob removal by RGB distance to black ---")
    h, w = img.shape[:2]

    # Distance to black in RGB
    dist_to_black = np.linalg.norm(img.astype(np.float32) - np.array([0, 0, 0]), axis=2)
    dark_mask = dist_to_black < 80

    # But also near-white text (some annotations might be white)
    dist_to_white = np.linalg.norm(img.astype(np.float32) - np.array([255, 255, 255]), axis=2)
    bright_mask = dist_to_white < 60

    text_mask = dark_mask | bright_mask
    text_mask = ndimage.binary_dilation(text_mask, iterations=1)

    print(f"  Text mask pixels: {text_mask.sum()} ({text_mask.sum() / (h*w) * 100:.2f}%)")

    valid_mask = (~text_mask) & (labels != 0)
    if not valid_mask.any():
        return labels.copy()

    _, indices = ndimage.distance_transform_edt(~valid_mask, return_indices=True)
    result = labels.copy()
    rr, cc = np.where(text_mask & (labels != 0))
    result[rr, cc] = labels[indices[0][rr, cc], indices[1][rr, cc]]

    save_overlay(result, img, OUT_DIR / "v2c_dark_blob.jpg", "(dark blob removal)")
    return result


def approach_v2d_conservative_text(labels, img):
    """Conservative: only very dark pixels with high edge response."""
    print("\n--- Approach V2d: Conservative dark+edge text removal ---")
    from skimage.filters import sobel

    h, w = img.shape[:2]
    gray = img.mean(axis=2).astype(np.float32)

    # Very dark
    dark = gray < 50

    # Strong edges
    edges = np.abs(sobel(gray))
    edgy = edges > np.percentile(edges, 90)

    # Must be both dark AND edgy (conservative)
    text_mask = dark & edgy
    text_mask = ndimage.binary_dilation(text_mask, iterations=1)

    print(f"  Text mask pixels: {text_mask.sum()} ({text_mask.sum() / (h*w) * 100:.2f}%)")

    valid_mask = (~text_mask) & (labels != 0)
    if not valid_mask.any():
        return labels.copy()

    _, indices = ndimage.distance_transform_edt(~valid_mask, return_indices=True)
    result = labels.copy()
    rr, cc = np.where(text_mask & (labels != 0))
    result[rr, cc] = labels[indices[0][rr, cc], indices[1][rr, cc]]

    save_overlay(result, img, OUT_DIR / "v2d_conservative.jpg", "(conservative)")
    return result


def approach_v2e_morphological_text(labels, img):
    """Morphological approach: opening to remove text, then fill holes."""
    print("\n--- Approach V2e: Morphological opening per label + hole fill ---")
    from skimage.morphology import disk, opening, closing

    result = labels.copy()

    for lbl in sorted(set(labels.flatten()) - {0}):
        mask = labels == lbl
        # Opening removes small bright features (text on dark background)
        # But text is dark on colored background, so we need different approach
        # Use closing to fill dark text holes within the label
        cleaned = closing(mask, footprint=disk(2))
        # Then opening to remove thin protrusions
        cleaned = opening(cleaned, footprint=disk(1))

        # Pixels that were removed (text) get reassigned
        removed = mask & ~cleaned
        added = cleaned & ~mask

        if removed.any():
            dilated = ndimage.binary_dilation(removed, structure=np.ones((3, 3), dtype=bool))
            neighbors = labels[dilated & ~removed]
            neighbors = neighbors[neighbors != 0]
            if len(neighbors) > 0:
                vals, counts = np.unique(neighbors, return_counts=True)
                result[removed] = vals[counts.argmax()]

        if added.any():
            # Don't expand labels too much
            pass

    save_overlay(result, img, OUT_DIR / "v2e_morphological.jpg", "(morphological)")
    return result


def approach_v2f_inpaint_style(labels, img):
    """Inpaint-style: detect text, then fill from neighborhood median."""
    print("\n--- Approach V2f: Inpaint-style median fill ---")
    import cv2

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Detect text: dark OR high Laplacian
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=11, C=2,
    )
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    lap_mask = lap > np.percentile(lap, 75)

    text_mask = ((adaptive > 0) | lap_mask).astype(np.uint8) * 255

    # Inpaint the image
    inpainted = cv2.inpaint(img, text_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # Now re-segment or just use inpainted for overlay display
    # For labels: reassign text pixels to nearest non-text label
    text_bool = text_mask > 0
    valid_mask = (~text_bool) & (labels != 0)
    if not valid_mask.any():
        return labels.copy()

    _, indices = ndimage.distance_transform_edt(~valid_mask, return_indices=True)
    result = labels.copy()
    rr, cc = np.where(text_bool & (labels != 0))
    result[rr, cc] = labels[indices[0][rr, cc], indices[1][rr, cc]]

    # Save inpainted overlay for visual comparison
    save_overlay(result, inpainted, OUT_DIR / "v2f_inpaint_overlay.jpg", "(inpaint overlay)")
    save_overlay(result, img, OUT_DIR / "v2f_inpaint_labels.jpg", "(inpaint labels on original)")
    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels, img = load_data()

    print(f"Labels: {np.unique(labels)}, shape: {labels.shape}")
    print(f"Image shape: {img.shape}")

    # Run all v2 approaches
    results = {}
    results["v2a"] = approach_v2a_text_mask_reassign(labels, img)
    results["v2b"] = approach_v2b_adaptive_text_threshold(labels, img)
    results["v2c"] = approach_v2c_dark_blob_removal(labels, img)
    results["v2d"] = approach_v2d_conservative_text(labels, img)
    results["v2e"] = approach_v2e_morphological_text(labels, img)
    results["v2f"] = approach_v2f_inpaint_style(labels, img)

    print(f"\n\nAll v2 approaches saved to {OUT_DIR}")
    print("Compare and pick the best one.")


if __name__ == "__main__":
    main()
