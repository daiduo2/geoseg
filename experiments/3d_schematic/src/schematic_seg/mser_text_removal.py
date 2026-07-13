"""
MSER-based text detection and removal for geological schematic images.

Strategy:
1. MSER on grayscale (catches stable bright/dark regions)
2. MSER on inverted grayscale (catches light text on dark bg)
3. Laplacian edge detection as backup for low-contrast text
4. Morphological closing to fill text regions from edges
5. Combine masks, dilate, inpaint
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def detect_text_mser(gray: np.ndarray, min_area: int = 10, max_area: int = 2000) -> np.ndarray:
    """Detect text-like regions using MSER on a grayscale image."""
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)

    mask = np.zeros(gray.shape, dtype=np.uint8)

    for region in regions:
        region = region.reshape(-1, 1, 2)
        area = cv2.contourArea(region)

        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(region)
        if w == 0 or h == 0:
            continue

        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio > 20:
            continue

        cv2.fillPoly(mask, [region], 255)

    return mask


def detect_text_laplacian(gray: np.ndarray, threshold: int = 15) -> np.ndarray:
    """Detect high-frequency regions (text edges) using Laplacian, then fill them."""
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))

    _, mask = cv2.threshold(laplacian, threshold, 255, cv2.THRESH_BINARY)

    # Remove very large connected components (non-text)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 2000:
            mask[labels == i] = 0

    # Close to fill text characters from edge outlines
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


def detect_text_regions(image: np.ndarray) -> np.ndarray:
    """
    Detect text-like regions using combined MSER + Laplacian approach.

    Args:
        image: RGB image as numpy array (H, W, 3)

    Returns:
        Binary mask of detected text regions
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # MSER on original and inverted
    mask_mser_orig = detect_text_mser(gray)
    mask_mser_inv = detect_text_mser(255 - gray)

    # Laplacian backup for low-contrast text
    mask_laplacian = detect_text_laplacian(gray)

    # Combine all masks
    combined = cv2.bitwise_or(mask_mser_orig, mask_mser_inv)
    combined = cv2.bitwise_or(combined, mask_laplacian)

    return combined


def remove_text(image: np.ndarray, mask: np.ndarray, dilate_iter: int = 4) -> np.ndarray:
    """
    Inpaint detected text regions.

    Args:
        image: RGB image
        mask: Binary mask of text regions
        dilate_iter: Number of dilation iterations

    Returns:
        Inpainted image
    """
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=dilate_iter)

    inpainted = cv2.inpaint(
        image,
        dilated_mask,
        inpaintRadius=7,
        flags=cv2.INPAINT_TELEA
    )

    return inpainted


def process_panel(image_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process a single panel: detect text and remove it.

    Returns:
        (original_rgb, mask, inpainted_rgb)
    """
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not load image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask = detect_text_regions(image_rgb)
    inpainted = remove_text(image_rgb, mask)

    return image_rgb, mask, inpainted


def main():
    base_dir = Path(__file__).parent.parent.parent
    output_dir = base_dir / "agent_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    panels = [
        base_dir / "panel_1_front.png",
        base_dir / "panel_2_front.png",
        base_dir / "panel_3_front.png",
    ]

    results = []
    for panel_path in panels:
        print(f"Processing {panel_path.name}...")
        original, mask, inpainted = process_panel(panel_path)
        results.append((panel_path.name, original, mask, inpainted))
        print(f"  - Mask coverage: {np.count_nonzero(mask) / mask.size * 100:.2f}%")

    fig, axes = plt.subplots(3, 2, figsize=(12, 18))

    for i, (name, original, mask, inpainted) in enumerate(results):
        axes[i, 0].imshow(original)
        axes[i, 0].set_title(f"{name} - Original")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(inpainted)
        axes[i, 1].set_title(f"{name} - Text Removed")
        axes[i, 1].axis("off")

    plt.tight_layout()
    output_path = output_dir / "mser_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved comparison to {output_path}")

    return results


if __name__ == "__main__":
    main()
