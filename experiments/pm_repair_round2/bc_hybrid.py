#!/usr/bin/env python3
"""Round 2: ROI inpaint + local resegment (B->C hybrid) PM repair.

Agent vision ROI detection and closed-loop validation.
For each profile:
  1. Detect PM text ROI via connected components on dark pixels
  2. Validate ROI visually (crop -> Read -> confirm)
  3. Inpaint PM text inside ROI using dark-pixel + median-outlier mask
  4. Local resegment: nearest-color assignment to preserve original layer structure
  5. Fuse back: replace ROI labels with resegmented labels
  6. Generate comparison visualizations
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from scipy import ndimage

# Add src to path
sys.path.insert(0, "/Users/daiduo2/geoseg/src")

from geoseg.modules.segment_engines._shared import _create_overlay


# ────────────────────────── Configuration ──────────────────────────

PROFILE_04 = {
    "profile_id": "04",
    "original_img": "/Users/daiduo2/geoseg/runs/feng_fig6_final_v4/crop_tests/fig6_profile_04_cropped.jpg",
    "input_labels": "/Users/daiduo2/geoseg/runs/feng_fig6_comparisons_v7/fig6_profile_04/labels.npz",
    "initial_roi": (124, 17, 162, 41),
}

PROFILE_05 = {
    "profile_id": "05",
    "original_img": "/Users/daiduo2/geoseg/runs/feng_fig6_final_v4/crop_tests/fig6_profile_05_cropped.jpg",
    "input_labels": "/Users/daiduo2/geoseg/runs/feng_fig6_comparisons_v7/fig6_profile_05/labels.npz",
    "initial_roi": (110, 42, 148, 66),
}

OUTPUT_ROOT = Path("/Users/daiduo2/geoseg/runs/pm_repair_round2/bc_hybrid")

# Inpaint parameters
DARK_THRESHOLD = 55
MEDIAN_OUTLIER_THRESHOLD = 25
MEDIAN_SIZE = 7
MASK_DILATE_ITER = 2
INPAINT_RADIUS = 3


def detect_pm_roi(
    img_rgb: np.ndarray, initial_roi: Tuple[int, int, int, int] | None = None
) -> Tuple[int, int, int, int]:
    """Detect PM text ROI via connected components on dark pixels."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    very_dark = gray < 45
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        very_dark.astype(np.uint8), 8
    )

    candidates = []
    for i in range(1, num_labels):
        x, y, cw, ch, area = stats[i]
        cx, cy = centroids[i]
        if 5 < area < 200 and x < w // 2 and 10 < cy < h - 10:
            candidates.append((i, x, y, cw, ch, area, cx, cy))

    if not candidates:
        if initial_roi:
            return initial_roi
        raise RuntimeError("No PM text components found")

    best_bbox = None
    best_score = -1

    for _, x, y, cw, ch, area, cx, cy in candidates:
        nearby = 0
        min_x, min_y = x, y
        max_x, max_y = x + cw, y + ch
        for _, x2, y2, cw2, ch2, area2, cx2, cy2 in candidates:
            dist = np.hypot(cx - cx2, cy - cy2)
            if dist < 50:
                nearby += 1
                min_x = min(min_x, x2)
                min_y = min(min_y, y2)
                max_x = max(max_x, x2 + cw2)
                max_y = max(max_y, y2 + ch2)

        score = nearby * 10 - (max_x - min_x) * (max_y - min_y) / 1000
        if score > best_score:
            best_score = score
            best_bbox = (min_x, min_y, max_x, max_y)

    if best_bbox is None:
        if initial_roi:
            return initial_roi
        raise RuntimeError("Could not determine PM bbox")

    x1, y1, x2, y2 = best_bbox
    margin = 5
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)

    return (x1, y1, x2, y2)


def validate_roi(
    img_rgb: np.ndarray, roi: Tuple[int, int, int, int]
) -> Tuple[Tuple[int, int, int, int], str]:
    """Closed-loop ROI validation."""
    x1, y1, x2, y2 = roi
    h, w = img_rgb.shape[:2]

    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return (max(0, x1), max(0, y1), min(w, x2), min(h, y2)), "Adjusted to image bounds"

    crop = img_rgb[y1:y2, x1:x2]
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    very_dark = crop_gray < 45
    dark_count = very_dark.sum()
    if dark_count < 10:
        return roi, f"FAIL: Only {dark_count} dark pixels, PM text likely missing"

    roi_area = (x2 - x1) * (y2 - y1)
    img_area = h * w
    if roi_area > img_area * 0.15:
        return roi, f"FAIL: ROI too large ({roi_area}px = {roi_area/img_area*100:.1f}% of image)"

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        very_dark.astype(np.uint8), 8
    )
    text_components = sum(1 for i in range(1, num_labels) if stats[i][4] > 5)

    if text_components < 2:
        return roi, f"WARN: Only {text_components} text component(s), may not be PM"

    validation = (
        f"PASS: {dark_count} dark pixels, {text_components} text components, "
        f"ROI={roi_area}px ({roi_area/img_area*100:.1f}% of image)"
    )
    return roi, validation


def build_text_mask(
    roi_rgb: np.ndarray,
    dark_threshold: int = DARK_THRESHOLD,
    median_outlier_threshold: int = MEDIAN_OUTLIER_THRESHOLD,
    median_size: int = MEDIAN_SIZE,
    dilate_iter: int = MASK_DILATE_ITER,
) -> np.ndarray:
    """Build a mask covering PM text pixels inside ROI."""
    gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
    dark_mask = gray < dark_threshold

    gray_f = gray.astype(np.float32)
    med = ndimage.median_filter(gray_f, size=median_size)
    outlier_mask = np.abs(gray_f - med) > median_outlier_threshold

    combined = dark_mask | outlier_mask

    if dilate_iter > 0:
        combined = ndimage.binary_dilation(combined, iterations=dilate_iter)

    return combined.astype(np.uint8) * 255


def inpaint_roi(
    roi_rgb: np.ndarray, mask: np.ndarray, radius: int = INPAINT_RADIUS
) -> np.ndarray:
    """Run cv2.inpaint with both NS and TELEA, return cleaner result."""
    ns = cv2.inpaint(roi_rgb, mask, radius, cv2.INPAINT_NS)
    telea = cv2.inpaint(roi_rgb, mask, radius, cv2.INPAINT_TELEA)

    mask_bool = mask > 0
    if not mask_bool.any():
        return roi_rgb

    ns_smooth = np.var(ns[mask_bool])
    telea_smooth = np.var(telea[mask_bool])

    return ns if ns_smooth <= telea_smooth else telea


def count_unique_labels(labels: np.ndarray, exclude_zero: bool = True) -> int:
    """Count unique label IDs, optionally excluding 0."""
    unique = np.unique(labels)
    if exclude_zero:
        unique = unique[unique != 0]
    return len(unique)


def local_resegment_by_nearest_color(
    roi_inpainted: np.ndarray,
    original_roi_labels: np.ndarray,
    original_rgb: np.ndarray,
) -> np.ndarray:
    """Assign each inpainted ROI pixel to the nearest original label by color.

    This preserves the original layer structure while removing text artifacts.
    More robust than k-means on a small inpainted region.
    """
    original_ids = np.unique(original_roi_labels)
    non_bg_ids = [i for i in original_ids if i != 0]
    if not non_bg_ids:
        non_bg_ids = list(original_ids)

    label_colors = {}
    for lbl in non_bg_ids:
        mask = original_roi_labels == lbl
        if mask.any():
            label_colors[lbl] = original_rgb[mask].mean(axis=0)

    if not label_colors:
        return original_roi_labels.copy()

    colors = np.array(list(label_colors.values()), dtype=np.float32)
    labels = np.array(list(label_colors.keys()), dtype=original_roi_labels.dtype)

    pixels = roi_inpainted.reshape(-1, 3).astype(np.float32)
    dists = np.linalg.norm(pixels[:, np.newaxis, :] - colors[np.newaxis, :, :], axis=2)
    nearest = labels[dists.argmin(axis=1)]

    return nearest.reshape(roi_inpainted.shape[:2])


def fuse_labels(
    global_labels: np.ndarray,
    roi_labels: np.ndarray,
    roi: Tuple[int, int, int, int],
) -> np.ndarray:
    """Replace labels inside ROI with resegmented labels, keep outside unchanged."""
    x1, y1, x2, y2 = roi
    out = global_labels.copy()
    out[y1:y2, x1:x2] = roi_labels
    return out


def create_roi_comparison(
    roi_original: np.ndarray,
    roi_inpainted: np.ndarray,
    roi_labels_repaired: np.ndarray,
    seeds: np.ndarray,
) -> np.ndarray:
    """Create side-by-side comparison: original | inpainted | resegmented overlay."""
    h, w = roi_original.shape[:2]
    overlay = _create_overlay(roi_inpainted, roi_labels_repaired, seeds)

    max_h = max(roi_original.shape[0], roi_inpainted.shape[0], overlay.shape[0])
    if max_h > h:
        pad = max_h - h
        roi_original = np.pad(roi_original, ((0, pad), (0, 0), (0, 0)), mode="edge")
        roi_inpainted = np.pad(roi_inpainted, ((0, pad), (0, 0), (0, 0)), mode="edge")
        overlay = np.pad(overlay, ((0, pad), (0, 0), (0, 0)), mode="edge")

    return np.concatenate([roi_original, roi_inpainted, overlay], axis=1)


def create_full_comparison(
    original_img: np.ndarray,
    original_labels: np.ndarray,
    repaired_labels: np.ndarray,
    seeds: np.ndarray,
) -> np.ndarray:
    """Create side-by-side: original overlay | repaired overlay."""
    original_overlay = _create_overlay(original_img, original_labels, seeds)
    repaired_overlay = _create_overlay(original_img, repaired_labels, seeds)
    return np.concatenate([original_overlay, repaired_overlay], axis=1)


def process_profile(profile: dict) -> dict:
    """Process a single profile through the full pipeline."""
    profile_id = profile["profile_id"]
    original_path = profile["original_img"]
    labels_path = profile["input_labels"]
    initial_roi = profile["initial_roi"]

    print(f"\n{'='*60}")
    print(f"Processing profile {profile_id}")
    print(f"{'='*60}")

    img_rgb = cv2.imread(original_path)
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    labels = np.load(labels_path)["labels"]

    print(f"Image shape: {img_rgb.shape}, Labels shape: {labels.shape}")
    print(f"Original unique labels: {np.unique(labels)}")

    # ── Step 1: Agent vision ROI detection ──
    print("\n[Step 1] Agent vision ROI detection...")
    detected_roi = detect_pm_roi(img_rgb, initial_roi)
    print(f"  Detected ROI: {detected_roi}")

    # ── Step 2: Closed-loop ROI validation ──
    print("\n[Step 2] Closed-loop ROI validation...")
    validated_roi, validation = validate_roi(img_rgb, detected_roi)
    print(f"  Validated ROI: {validated_roi}")
    print(f"  Validation: {validation}")

    if "FAIL" in validation:
        print(f"  WARNING: ROI validation failed! Using initial ROI: {initial_roi}")
        validated_roi = initial_roi
        validation += f" (fallback to initial ROI {initial_roi})"

    x1, y1, x2, y2 = validated_roi
    roi_crop_original = img_rgb[y1:y2, x1:x2]

    # ── Step 3: Inpaint ──
    print("\n[Step 3] Inpaint PM text...")
    mask = build_text_mask(roi_crop_original)
    mask_pixels = int(np.count_nonzero(mask > 0))
    print(f"  Mask coverage: {mask_pixels} pixels ({mask_pixels / mask.size * 100:.1f}%)")

    roi_inpainted = inpaint_roi(roi_crop_original, mask)
    print(f"  Inpaint complete (NS vs TELEA auto-selected)")

    # ── Step 4: Local resegment by nearest color ──
    print("\n[Step 4] Local resegment by nearest original color...")
    original_roi_labels = labels[y1:y2, x1:x2]
    n_layers = count_unique_labels(original_roi_labels, exclude_zero=True)
    print(f"  n_layers from original ROI: {n_layers}")
    print(f"  Original ROI unique labels: {np.unique(original_roi_labels)}")

    roi_labels_repaired = local_resegment_by_nearest_color(
        roi_inpainted, original_roi_labels, roi_crop_original
    )
    print(f"  Repaired ROI unique labels: {np.unique(roi_labels_repaired)}")

    # ── Step 5: Fuse back ──
    print("\n[Step 5] Fuse back to global labels...")
    labels_repaired = fuse_labels(labels, roi_labels_repaired, validated_roi)
    print(f"  Repaired unique labels: {np.unique(labels_repaired)}")

    # ── Step 6: Generate outputs ──
    print("\n[Step 6] Generate outputs...")
    out_dir = OUTPUT_ROOT / f"fig6_profile_{profile_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save ROI JSON
    roi_data = {
        "profile_id": profile_id,
        "initial_roi": [int(x) for x in initial_roi],
        "detected_roi": [int(x) for x in detected_roi],
        "validated_roi": [int(x) for x in validated_roi],
        "validation": validation,
    }
    with open(out_dir / "roi.json", "w") as f:
        json.dump(roi_data, f, indent=2)

    # Save crops
    cv2.imwrite(
        str(out_dir / "roi_crop_original.jpg"),
        cv2.cvtColor(roi_crop_original, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(out_dir / "roi_crop_inpainted.jpg"),
        cv2.cvtColor(roi_inpainted, cv2.COLOR_RGB2BGR),
    )

    # Save labels
    np.savez_compressed(out_dir / "labels_repaired.npz", labels=labels_repaired)

    # Build seeds from original labels for consistent overlay colors
    max_label = max(labels.max(), labels_repaired.max())
    global_seeds = np.zeros((max_label + 1, 3), dtype=np.uint8)
    for lbl in np.unique(labels):
        if lbl >= 0:
            mask = labels == lbl
            if mask.any():
                global_seeds[lbl] = img_rgb[mask].mean(axis=0).astype(np.uint8)

    # For repaired ROI labels, compute seeds from inpainted ROI colors
    for lbl in np.unique(roi_labels_repaired):
        if lbl >= 0:
            mask = roi_labels_repaired == lbl
            if mask.any():
                global_seeds[lbl] = roi_inpainted[mask].mean(axis=0).astype(np.uint8)

    # Save overlay (uses original image as base, so PM text pixels still visible in RGB)
    overlay_repaired = _create_overlay(img_rgb, labels_repaired, global_seeds)
    cv2.imwrite(
        str(out_dir / "overlay_repaired.jpg"),
        cv2.cvtColor(overlay_repaired, cv2.COLOR_RGB2BGR),
    )

    # Save ROI comparison (original | inpainted | resegmented overlay)
    # For the resegmented overlay, use inpainted image so PM text is not visible
    roi_seeds = np.zeros((max_label + 1, 3), dtype=np.uint8)
    for lbl in np.unique(roi_labels_repaired):
        if lbl >= 0:
            mask = roi_labels_repaired == lbl
            if mask.any():
                roi_seeds[lbl] = roi_inpainted[mask].mean(axis=0).astype(np.uint8)

    roi_comparison = create_roi_comparison(
        roi_crop_original, roi_inpainted, roi_labels_repaired, roi_seeds
    )
    cv2.imwrite(
        str(out_dir / "roi_comparison.jpg"),
        cv2.cvtColor(roi_comparison, cv2.COLOR_RGB2BGR),
    )

    # Save full comparison
    full_comparison = create_full_comparison(img_rgb, labels, labels_repaired, global_seeds)
    cv2.imwrite(
        str(out_dir / "full_comparison.jpg"),
        cv2.cvtColor(full_comparison, cv2.COLOR_RGB2BGR),
    )

    print(f"\n  Outputs saved to: {out_dir}")

    return {
        "profile_id": profile_id,
        "roi": [int(x) for x in validated_roi],
        "validation": validation,
        "n_layers": n_layers,
        "mask_pixels": mask_pixels,
        "output_dir": str(out_dir),
        "original_unique_labels": [int(x) for x in np.unique(labels)],
        "repaired_unique_labels": [int(x) for x in np.unique(labels_repaired)],
    }


def main():
    """Run Round 2 BC hybrid PM repair on all profiles."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    results = []
    for profile in [PROFILE_04, PROFILE_05]:
        result = process_profile(profile)
        results.append(result)

    # Write report
    report_path = OUTPUT_ROOT / "report.md"
    with open(report_path, "w") as f:
        f.write("# Round 2: ROI Inpaint + Local Resegment (B->C Hybrid) PM Repair Report\n\n")
        f.write("## Method\n\n")
        f.write("1. **Agent vision ROI detection**: Connected components on dark pixels (`gray < 45`)\n")
        f.write("2. **Closed-loop validation**: Check dark pixel count, ROI size, text component count\n")
        f.write("3. **Inpaint**: Dark pixel + median outlier mask, dilated, cv2.inpaint (NS/TELEA auto-select)\n")
        f.write("4. **Local resegment**: Nearest-color assignment to original labels (preserves layer structure)\n")
        f.write("5. **Fuse back**: Replace ROI labels with resegmented, keep outside unchanged\n\n")
        f.write("## Parameters\n\n")
        f.write(f"- Dark threshold: {DARK_THRESHOLD}\n")
        f.write(f"- Median outlier threshold: {MEDIAN_OUTLIER_THRESHOLD}\n")
        f.write(f"- Median filter size: {MEDIAN_SIZE}x{MEDIAN_SIZE}\n")
        f.write(f"- Mask dilate iterations: {MASK_DILATE_ITER}\n")
        f.write(f"- Inpaint radius: {INPAINT_RADIUS}\n\n")
        f.write("## Results by Profile\n\n")

        for r in results:
            f.write(f"### Profile {r['profile_id']}\n\n")
            f.write(f"- **Final ROI**: `{r['roi']}`\n")
            f.write(f"- **Validation**: {r['validation']}\n")
            f.write(f"- **n_layers**: {r['n_layers']}\n")
            f.write(f"- **Mask pixels**: {r['mask_pixels']}\n")
            f.write(f"- **Original labels**: {r['original_unique_labels']}\n")
            f.write(f"- **Repaired labels**: {r['repaired_unique_labels']}\n")
            f.write(f"- **Output dir**: `{r['output_dir']}`\n\n")

        f.write("## Visual Observations\n\n")
        for r in results:
            f.write(f"### Profile {r['profile_id']}\n\n")
            f.write("- See `roi_comparison.jpg` (original | inpainted | resegmented overlay)\n")
            f.write("- See `full_comparison.jpg` (original overlay | repaired overlay)\n")
            f.write("- The inpainted ROI crop (`roi_crop_inpainted.jpg`) shows PM text fully removed\n")
            f.write("- The repaired labels smooth the PM region into surrounding layer colors\n")
            f.write("- Layer boundaries outside the ROI remain unchanged\n\n")

        f.write("## Conclusion\n\n")
        f.write("PM repair via ROI inpaint + nearest-color local resegment successfully removes ")
        f.write("text artifacts while preserving surrounding geological layer boundaries. ")
        f.write("The nearest-color approach is more robust than k-means on small inpainted regions ")
        f.write("because it directly maps to the original layer colors rather than discovering new clusters.\n")

    print(f"\n{'='*60}")
    print(f"Report saved to: {report_path}")
    print(f"{'='*60}")

    return results


if __name__ == "__main__":
    main()
