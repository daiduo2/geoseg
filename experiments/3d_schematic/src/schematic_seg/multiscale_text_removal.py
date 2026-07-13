#!/usr/bin/env python3
"""
Multiscale edge + local contrast based text detection and removal.

Strategy:
1. Load each image as RGB numpy array, convert to grayscale
2. Compute MULTIPLE text-detection channels and combine them:
   a) Laplacian edges (catches thin strokes)
   b) Adaptive threshold with large window (catches low-contrast text like Panel 3)
   c) Local contrast: abs(image - GaussianBlur(image, (31,31), 0)) — high local variation = text
   d) Low saturation mask: text is desaturated
3. Combine all masks with bitwise_or
4. Filter connected components: remove very large blobs (>500 px, those are geological structures)
5. Dilate mask and inpaint with cv2.inpaint(INPAINT_TELEA, radius=5)
6. Generate 3x2 comparison figure
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def load_image(path: str) -> np.ndarray:
    """Load image as RGB numpy array."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def compute_text_mask(image_rgb: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Compute multiscale text detection mask.
    Returns (combined_mask, channel_info).
    """
    h, w = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)

    # --- Channel a: Laplacian edges (catches thin strokes) ---
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    laplacian_abs = np.abs(laplacian).astype(np.uint8)
    _, mask_laplacian = cv2.threshold(laplacian_abs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- Channel b: Adaptive threshold with large window (catches low-contrast text) ---
    # Use a large blockSize to catch large text blocks; subtract small C to be sensitive
    mask_adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=51,  # Large window for low-contrast large text
        C=5
    )

    # --- Channel c: Local contrast (high local variation = text) ---
    blurred = cv2.GaussianBlur(gray, (31, 31), 0)
    local_contrast = cv2.absdiff(gray, blurred)
    _, mask_contrast = cv2.threshold(local_contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- Channel d: Low saturation mask (text is desaturated) ---
    # Text tends to have low saturation compared to colored geological fills
    sat_blurred = cv2.GaussianBlur(saturation, (15, 15), 0)
    _, mask_low_sat = cv2.threshold(sat_blurred.astype(np.uint8), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # --- Channel e: Difference of Gaussians (DoG) for blob detection ---
    # Catches text of various scales without global thresholding artifacts
    blur_small = cv2.GaussianBlur(gray, (3, 3), 0.5)
    blur_large = cv2.GaussianBlur(gray, (11, 11), 2.0)
    dog = cv2.absdiff(blur_small, blur_large)
    _, mask_dog = cv2.threshold(dog, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- Channel f: Canny edges (complementary to Laplacian) ---
    canny_edges = cv2.Canny(gray, 50, 150)

    # --- Channel g: Local brightness anomaly (catches low-contrast text like Panel 3) ---
    # Compute local mean with large kernel, flag pixels that deviate significantly
    local_mean = cv2.boxFilter(gray.astype(np.float32), -1, (21, 21), normalize=True)
    brightness_deviation = np.abs(gray.astype(np.float32) - local_mean).astype(np.uint8)
    _, mask_local_brightness = cv2.threshold(brightness_deviation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- Combine all masks ---
    combined = np.zeros_like(gray)
    combined = cv2.bitwise_or(combined, mask_laplacian)
    combined = cv2.bitwise_or(combined, mask_adaptive)
    combined = cv2.bitwise_or(combined, mask_contrast)
    combined = cv2.bitwise_or(combined, mask_low_sat)
    combined = cv2.bitwise_or(combined, mask_dog)
    combined = cv2.bitwise_or(combined, canny_edges)
    combined = cv2.bitwise_or(combined, mask_local_brightness)

    # --- Morphological opening to separate merged components before filtering ---
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # --- Filter connected components: remove very large blobs (>8000 px = geological structures) ---
    # Text blocks (even multi-line paragraphs) are typically < 3000 px in these images
    # Geological structures (the plume, crust layers) are much larger
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
    filtered_mask = np.zeros_like(combined)
    removed_large = 0
    kept_components = 0
    for i in range(1, num_labels):  # Skip background
        area = stats[i, cv2.CC_STAT_AREA]
        x, y, bw, bh = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]

        # Remove very large blobs — these are geological structures, not text
        if area > 8000:
            removed_large += 1
            continue

        # Remove very small noise
        if area < 15:
            continue

        # Aspect ratio filter: text tends to be elongated (words, lines)
        # But also keep compact blobs that could be individual characters
        aspect_ratio = max(bw, bh) / max(min(bw, bh), 1)
        # Keep if: elongated (text line/word) OR small compact (individual char)
        is_text_like = (aspect_ratio > 1.5 and area < 4000) or (aspect_ratio <= 1.5 and area < 500)
        if not is_text_like:
            removed_large += 1
            continue

        filtered_mask[labels == i] = 255
        kept_components += 1

    # --- Dilate mask to cover text strokes fully ---
    # Use a slightly larger kernel to ensure low-contrast text (Panel 3) is fully covered
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_mask = cv2.dilate(filtered_mask, kernel, iterations=2)

    channel_info = {
        'laplacian': mask_laplacian,
        'adaptive': mask_adaptive,
        'contrast': mask_contrast,
        'low_sat': mask_low_sat,
        'dog': mask_dog,
        'canny': canny_edges,
        'local_brightness': mask_local_brightness,
        'combined_raw': combined,
        'filtered': filtered_mask,
        'dilated': dilated_mask,
        'removed_large': removed_large,
        'kept_components': kept_components,
    }
    return dilated_mask, channel_info


def inpaint_image(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Inpaint image using TELEA algorithm with larger radius for better text removal."""
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    # Use larger radius (7) to better fill low-contrast text regions
    result_bgr = cv2.inpaint(image_bgr, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)


def process_panel(image_path: str) -> dict:
    """Process a single panel and return results."""
    image = load_image(image_path)
    mask, info = compute_text_mask(image)
    inpainted = inpaint_image(image, mask)
    return {
        'original': image,
        'mask': mask,
        'inpainted': inpainted,
        'info': info,
        'name': Path(image_path).stem,
    }


def create_comparison_figure(results: list[dict], output_path: str) -> None:
    """Create 3x2 comparison figure (original + inpainted for each panel)."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 18))
    fig.suptitle('Multiscale Text Removal: Original vs Inpainted', fontsize=14, fontweight='bold')

    for i, result in enumerate(results):
        # Original
        axes[i, 0].imshow(result['original'])
        axes[i, 0].set_title(f"{result['name']} - Original")
        axes[i, 0].axis('off')

        # Inpainted
        axes[i, 1].imshow(result['inpainted'])
        info = result['info']
        axes[i, 1].set_title(
            f"{result['name']} - Inpainted\n"
            f"(removed {info['removed_large']} large blobs, kept {info['kept_components']} components)"
        )
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison figure to: {output_path}")


def create_channel_breakdown_figure(results: list[dict], output_path: str) -> None:
    """Create detailed channel breakdown for each panel."""
    fig, axes = plt.subplots(3, 8, figsize=(24, 15))
    fig.suptitle('Channel Breakdown: Laplacian | Adaptive | Contrast | LowSat | DoG | Canny | LocalBright | Combined', fontsize=12)

    channel_names = ['laplacian', 'adaptive', 'contrast', 'low_sat', 'dog', 'canny', 'local_brightness', 'combined_raw']
    display_names = ['Laplacian', 'Adaptive', 'Contrast', 'LowSat', 'DoG', 'Canny', 'LocalBright', 'Combined']

    for i, result in enumerate(results):
        for j, (ch_name, disp_name) in enumerate(zip(channel_names, display_names)):
            axes[i, j].imshow(result['info'][ch_name], cmap='gray')
            if i == 0:
                axes[i, j].set_title(disp_name, fontsize=10)
            axes[i, j].axis('off')
        # Add panel name on left
        axes[i, 0].set_ylabel(result['name'], rotation=0, ha='right', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved channel breakdown to: {output_path}")


def analyze_contributions(results: list[dict]) -> None:
    """Analyze which channels contributed most to each panel."""
    print("\n" + "=" * 60)
    print("CHANNEL CONTRIBUTION ANALYSIS")
    print("=" * 60)

    for result in results:
        name = result['name']
        info = result['info']
        h, w = result['original'].shape[:2]
        total_pixels = h * w

        print(f"\n{name}:")
        print(f"  Image size: {w}x{h} = {total_pixels} px")
        print(f"  Final mask coverage: {np.count_nonzero(info['dilated']) / total_pixels * 100:.2f}%")
        print(f"  Components: removed {info['removed_large']} large blobs, kept {info['kept_components']}")

        # Measure individual channel contributions (before filtering)
        print("  Channel coverages (raw, before filtering):")
        for ch_name in ['laplacian', 'adaptive', 'contrast', 'low_sat', 'dog', 'canny', 'local_brightness']:
            coverage = np.count_nonzero(info[ch_name]) / total_pixels * 100
            print(f"    {ch_name:20s}: {coverage:6.2f}%")

        # Check which channels overlap with final mask
        print("  Channel overlap with final mask:")
        final_mask_bool = info['dilated'] > 0
        for ch_name in ['laplacian', 'adaptive', 'contrast', 'low_sat', 'dog', 'canny', 'local_brightness']:
            ch_bool = info[ch_name] > 0
            overlap = np.logical_and(ch_bool, final_mask_bool)
            overlap_pct = np.count_nonzero(overlap) / max(np.count_nonzero(final_mask_bool), 1) * 100
            print(f"    {ch_name:20s}: {overlap_pct:6.1f}% of final mask")


def main():
    base_dir = Path('/Users/daiduo2/geoseg/src/3d_schematic')
    output_dir = base_dir / 'agent_results'
    output_dir.mkdir(exist_ok=True)

    panel_paths = [
        base_dir / 'panel_1_front.png',
        base_dir / 'panel_2_front.png',
        base_dir / 'panel_3_front.png',
    ]

    print("Processing panels with multiscale text detection...")
    results = []
    for path in panel_paths:
        print(f"  Processing {path.name}...")
        result = process_panel(str(path))
        results.append(result)

    # Create comparison figure
    comparison_path = output_dir / 'multiscale_comparison.png'
    create_comparison_figure(results, str(comparison_path))

    # Create channel breakdown
    breakdown_path = output_dir / 'multiscale_channels.png'
    create_channel_breakdown_figure(results, str(breakdown_path))

    # Analyze contributions
    analyze_contributions(results)

    # Save individual inpainted results
    for result in results:
        out_path = output_dir / f"{result['name']}_inpainted.png"
        cv2.imwrite(
            str(out_path),
            cv2.cvtColor(result['inpainted'], cv2.COLOR_RGB2BGR)
        )
        print(f"Saved {out_path.name}")

    print("\nDone!")


if __name__ == '__main__':
    main()
