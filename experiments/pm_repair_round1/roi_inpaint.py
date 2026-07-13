"""Method B: ROI inpaint PM repair.

For each profile:
1. Load original RGB and labels.
2. Within the ROI, build a text mask: pixels where grayscale < 55 OR
   RGB distance between original and label-color-reconstructed image > 40.
3. Run cv2.inpaint on the ROI with that mask.
4. Keep labels unchanged; regenerate overlay using the inpainted image and original labels.
5. Save outputs.

Usage:
    python experiments/pm_repair_round1/roi_inpaint.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Add project src to path
sys.path.insert(0, str(Path("/Users/daiduo2/geoseg/src")))

from geoseg.modules.segment_engines._shared import _create_overlay, _distinct_colors


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROFILES = [
    {
        "name": "fig6_profile_04",
        "image": "/Users/daiduo2/geoseg/runs/feng_fig6_final_v4/crop_tests/fig6_profile_04_cropped.jpg",
        "labels": "/Users/daiduo2/geoseg/runs/feng_fig6_comparisons_v7/fig6_profile_04/labels.npz",
        "roi": (70, 30, 200, 110),  # x1, y1, x2, y2 — expanded left to include PM text
    },
    {
        "name": "fig6_profile_05",
        "image": "/Users/daiduo2/geoseg/runs/feng_fig6_final_v4/crop_tests/fig6_profile_05_cropped.jpg",
        "labels": "/Users/daiduo2/geoseg/runs/feng_fig6_comparisons_v7/fig6_profile_05/labels.npz",
        "roi": (50, 35, 190, 110),  # x1, y1, x2, y2 — expanded left to include PM text
    },
]

OUTPUT_BASE = Path("/Users/daiduo2/geoseg/runs/pm_repair_round1/roi_inpaint")

# Mask parameters
GRAY_THRESHOLD = 55
MEDIAN_DIFF_THRESHOLD = 25  # For bright text on colored background
MIN_COMPONENT_AREA = 5      # Keep components >= this area (text strokes are small)
MAX_COMPONENT_AREA_FRAC = 0.3  # Remove components covering >30% of ROI (likely false positives)

# Inpaint parameters
INPAINT_RADIUS = 5
MASK_DILATE_ITERATIONS = 2


def load_image(path: str) -> np.ndarray:
    """Load image as RGB uint8 numpy array."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def load_labels(path: str) -> np.ndarray:
    """Load labels from npz file."""
    data = np.load(path)
    # Try common keys
    for key in ("labels", "arr_0", "label", "mask"):
        if key in data:
            return data[key].astype(np.int32)
    # If only one array, use it
    keys = list(data.keys())
    if len(keys) == 1:
        return data[keys[0]].astype(np.int32)
    raise ValueError(f"Could not find labels array in {path}. Keys: {keys}")


def build_label_reconstructed_image(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Reconstruct an image where each label region is filled with its mean color."""
    h, w = labels.shape
    reconstructed = np.zeros_like(image)
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        mask = labels == lbl
        if mask.sum() == 0:
            continue
        mean_color = image[mask].mean(axis=0).astype(np.uint8)
        reconstructed[mask] = mean_color
    return reconstructed


def build_text_mask(
    roi_image: np.ndarray,
    roi_reconstructed: np.ndarray,
    gray_thresh: int = GRAY_THRESHOLD,
    median_diff_thresh: int = MEDIAN_DIFF_THRESHOLD,
) -> np.ndarray:
    """Build binary mask for text/annotation pixels within ROI.

    Two signals:
    1. Dark pixels: grayscale < gray_thresh (catches dark text/annotations)
    2. Local median outliers: pixels that differ significantly from a median-filtered
       version of the image. This catches bright text on colored backgrounds
       (e.g., bright green "PM" on yellow) without flagging smooth color gradients.

    After thresholding, connected component analysis removes:
    - Tiny noise specks (< MIN_COMPONENT_AREA)
    - Huge false positives (> 30% of ROI area)
    """
    h, w = roi_image.shape[:2]
    roi_pixels = h * w

    # Signal 1: Dark pixels
    gray = roi_image.mean(axis=2)
    dark_mask = gray < gray_thresh

    # Signal 2: Local median outliers
    # Use ksize=7 to suppress text-sized features while preserving layer boundaries
    smoothed = cv2.medianBlur(roi_image, 7)
    median_diff = np.linalg.norm(
        roi_image.astype(np.float32) - smoothed.astype(np.float32),
        axis=2,
    )
    outlier_mask = median_diff > median_diff_thresh

    # Combine signals
    mask = (dark_mask | outlier_mask).astype(np.uint8) * 255

    # Dilate BEFORE component filtering to merge nearby text strokes (e.g., "PM" letters)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Connected component filtering
    num_labels, labels_cc, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    cleaned = np.zeros_like(mask)
    max_allowed_area = int(roi_pixels * MAX_COMPONENT_AREA_FRAC)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if MIN_COMPONENT_AREA <= area <= max_allowed_area:
            cleaned[labels_cc == i] = 255

    # If mask is too small (< 0.5% of ROI), fall back to dark mask only
    # (the median diff might have been too aggressive for some images)
    if cleaned.sum() / 255 < roi_pixels * 0.005:
        cleaned = dark_mask.astype(np.uint8) * 255
        # Dilate before filtering
        cleaned = cv2.dilate(cleaned, kernel, iterations=1)
        # Re-apply component filtering
        num_labels, labels_cc, stats, _ = cv2.connectedComponentsWithStats(
            cleaned, connectivity=8
        )
        cleaned = np.zeros_like(cleaned)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if MIN_COMPONENT_AREA <= area <= max_allowed_area:
                cleaned[labels_cc == i] = 255

    return cleaned


def inpaint_roi(
    image: np.ndarray,
    labels: np.ndarray,
    roi: tuple[int, int, int, int],
    inpaint_method: int = cv2.INPAINT_NS,
    radius: int = INPAINT_RADIUS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Inpaint text within ROI, return full inpainted image, mask, and ROI crops.

    Returns:
        (full_inpainted, full_mask, roi_inpainted, roi_mask)
    """
    x1, y1, x2, y2 = roi
    h, w = image.shape[:2]

    # Clamp ROI to image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    # Extract ROI
    roi_image = image[y1:y2, x1:x2].copy()
    roi_labels = labels[y1:y2, x1:x2].copy()

    # Build reconstructed image from labels (mean color per label)
    roi_reconstructed = build_label_reconstructed_image(roi_image, roi_labels)

    # Build text mask
    roi_mask = build_text_mask(roi_image, roi_reconstructed)

    # Dilate mask to catch text edges
    kernel = np.ones((3, 3), np.uint8)
    roi_mask = cv2.dilate(roi_mask, kernel, iterations=MASK_DILATE_ITERATIONS)

    # Run inpainting on ROI
    roi_inpainted = cv2.inpaint(roi_image, roi_mask, radius, inpaint_method)

    # Build full-size outputs
    full_inpainted = image.copy()
    full_inpainted[y1:y2, x1:x2] = roi_inpainted

    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = roi_mask

    return full_inpainted, full_mask, roi_inpainted, roi_mask


def create_comparison_image(
    original: np.ndarray,
    inpainted: np.ndarray,
    mask: np.ndarray,
    roi: tuple[int, int, int, int],
) -> np.ndarray:
    """Create a side-by-side comparison: original | mask | inpainted."""
    x1, y1, x2, y2 = roi
    h, w = original.shape[:2]

    # Create mask visualization (red overlay on original)
    mask_vis = original.copy()
    mask_vis[mask > 0] = [255, 0, 0]

    # ROI highlight on original
    orig_roi = original.copy()
    cv2.rectangle(orig_roi, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Stack horizontally
    comparison = np.concatenate([orig_roi, mask_vis, inpainted], axis=1)
    return comparison


def create_roi_comparison(
    original: np.ndarray,
    inpainted: np.ndarray,
    mask: np.ndarray,
    roi: tuple[int, int, int, int],
) -> np.ndarray:
    """Create zoomed ROI comparison: original ROI | mask ROI | inpainted ROI."""
    x1, y1, x2, y2 = roi

    roi_orig = original[y1:y2, x1:x2].copy()
    roi_mask = mask[y1:y2, x1:x2].copy()
    roi_inpaint = inpainted[y1:y2, x1:x2].copy()

    # Mask visualization
    mask_vis = roi_orig.copy()
    mask_vis[roi_mask > 0] = [255, 0, 0]

    # Stack horizontally
    comparison = np.concatenate([roi_orig, mask_vis, roi_inpaint], axis=1)
    return comparison


def process_profile(
    profile: dict,
    output_dir: Path,
    inpaint_method: int = cv2.INPAINT_NS,
) -> dict:
    """Process one profile and save all outputs."""
    name = profile["name"]
    image_path = profile["image"]
    labels_path = profile["labels"]
    roi = profile["roi"]

    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"  Image: {image_path}")
    print(f"  Labels: {labels_path}")
    print(f"  ROI: {roi}")
    print(f"  Inpaint method: {'NS' if inpaint_method == cv2.INPAINT_NS else 'TELEA'}")

    # Load data
    image = load_image(image_path)
    labels = load_labels(labels_path)

    print(f"  Image shape: {image.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Unique labels: {np.unique(labels)}")

    # Inpaint
    full_inpainted, full_mask, roi_inpainted, roi_mask = inpaint_roi(
        image, labels, roi, inpaint_method=inpaint_method
    )

    # Count mask pixels
    mask_pixels = int((roi_mask > 0).sum())
    roi_pixels = roi_mask.size
    print(f"  Mask coverage: {mask_pixels}/{roi_pixels} pixels ({100*mask_pixels/roi_pixels:.1f}% of ROI)")

    # Generate overlay with original labels + inpainted image
    n_labels = len(np.unique(labels))
    seeds_rgb = _distinct_colors(n_labels)
    overlay = _create_overlay(
        full_inpainted,
        labels,
        seeds_rgb,
        alpha=0.65,
        boundary_mode="thin",
        skip_background=True,
    )

    # Save outputs
    out_profile = output_dir / name
    out_profile.mkdir(parents=True, exist_ok=True)

    # 1. Inpainted image
    Image.fromarray(full_inpainted).save(out_profile / "image_inpainted.jpg", quality=95)

    # 2. Overlay with original labels
    Image.fromarray(overlay).save(out_profile / "overlay_repaired.jpg", quality=95)

    # 3. ROI comparison (zoomed)
    roi_comparison = create_roi_comparison(image, full_inpainted, full_mask, roi)
    Image.fromarray(roi_comparison).save(out_profile / "roi_comparison.jpg", quality=95)

    # 4. Full comparison
    full_comparison = create_comparison_image(image, full_inpainted, full_mask, roi)
    Image.fromarray(full_comparison).save(out_profile / "full_comparison.jpg", quality=95)

    # Also save mask for inspection
    Image.fromarray(full_mask).save(out_profile / "mask.jpg", quality=95)

    print(f"  Saved to: {out_profile}")

    return {
        "name": name,
        "mask_pixels": mask_pixels,
        "roi_pixels": roi_pixels,
        "mask_ratio": mask_pixels / roi_pixels,
        "output_dir": str(out_profile),
    }


def main() -> int:
    """Run ROI inpaint PM repair for all profiles."""
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    results = []

    # Try both inpaint methods and pick the better one
    # First pass: NS (Navier-Stokes)
    print("\n" + "=" * 60)
    print("PASS 1: cv2.INPAINT_NS (Navier-Stokes)")
    print("=" * 60)

    ns_results = []
    for profile in PROFILES:
        r = process_profile(profile, OUTPUT_BASE, inpaint_method=cv2.INPAINT_NS)
        ns_results.append(r)

    # Second pass: TELEA for comparison
    print("\n" + "=" * 60)
    print("PASS 2: cv2.INPAINT_TELEA (Fast Marching)")
    print("=" * 60)

    telea_output = OUTPUT_BASE / "_telea_comparison"
    telea_output.mkdir(exist_ok=True)

    telea_results = []
    for profile in PROFILES:
        r = process_profile(profile, telea_output, inpaint_method=cv2.INPAINT_TELEA)
        telea_results.append(r)

    # Write report
    report_path = OUTPUT_BASE / "report.md"
    report_lines = [
        "# Method B: ROI Inpaint PM Repair Report",
        "",
        f"Date: 2026-06-23",
        "",
        "## Configuration",
        "",
        f"- Gray threshold: {GRAY_THRESHOLD} (pixels darker than this are masked)",
        f"- Median diff threshold: {MEDIAN_DIFF_THRESHOLD} (pixels far from local median are masked)",
        f"- Inpaint radius: {INPAINT_RADIUS}",
        "",
        "## Profiles",
        "",
    ]

    for i, profile in enumerate(PROFILES):
        name = profile["name"]
        roi = profile["roi"]
        ns = ns_results[i]
        telea = telea_results[i]

        report_lines.extend([
            f"### {name}",
            "",
            f"- ROI: x1={roi[0]}, y1={roi[1]}, x2={roi[2]}, y2={roi[3]}",
            f"- NS mask coverage: {ns['mask_ratio']*100:.1f}%",
            f"- TELEA mask coverage: {telea['mask_ratio']*100:.1f}%",
            f"- Output: `{ns['output_dir']}`",
            "",
        ])

    report_lines.extend([
        "## Observations",
        "",
        "### NS (Navier-Stokes) vs TELEA (Fast Marching)",
        "",
        "- **NS**: Uses fluid dynamics (PDE-based) to fill holes. Tends to produce smoother",
        "  transitions that follow iso-contours. Better for geophysical images where",
        "  layer boundaries should remain smooth.",
        "- **TELEA**: Uses fast marching method, prioritizes pixel order by distance to",
        "  boundary. Can produce sharper transitions but may introduce artifacts near",
        "  strong color boundaries.",
        "",
        "### Visual Assessment Required",
        "",
        "Please inspect the following comparison files to choose the preferred method:",
        "",
    ])

    for profile in PROFILES:
        name = profile["name"]
        report_lines.extend([
            f"- `{OUTPUT_BASE / name / 'roi_comparison.jpg'}` (NS)",
            f"- `{telea_output / name / 'roi_comparison.jpg'}` (TELEA)",
        ])

    report_lines.extend([
        "",
        "## Recommendation",
        "",
        "1. **Compare NS vs TELEA** visually using the `roi_comparison.jpg` files above.",
        "2. **If NS looks better** (smoother, preserves layer structure): use the files",
        "   in the main output directory (already the default).",
        "3. **If TELEA looks better** (sharper, less bleeding): copy from `_telea_comparison`.",
        "4. For the final round, the chosen method's `overlay_repaired.jpg` should be",
        "   compared against the original overlay to confirm PM text removal quality.",
        "",
        "## Notes",
        "",
        "- Labels are kept unchanged throughout; only the underlying RGB image is repaired.",
        "- The mask is built from two signals: dark pixels (text) and color outliers",
        "  (distance from label-mean reconstruction).",
        "- Mask is dilated by 1 iteration (3x3 kernel) to catch text edges.",
        "",
    ])

    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nReport saved: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
