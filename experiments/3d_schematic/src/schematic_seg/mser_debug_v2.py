"""
Debug script for MSER + Laplacian text detection.
Shows mask overlay and individual mask components.
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def detect_text_mser(gray: np.ndarray, min_area: int = 10, max_area: int = 2000) -> np.ndarray:
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
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))

    _, mask = cv2.threshold(laplacian, threshold, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 2000:
            mask[labels == i] = 0

    return mask


def main():
    base_dir = Path(__file__).parent.parent.parent
    output_dir = base_dir / "agent_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    panels = [
        base_dir / "panel_1_front.png",
        base_dir / "panel_2_front.png",
        base_dir / "panel_3_front.png",
    ]

    for panel_path in panels:
        print(f"Processing {panel_path.name}...")
        image_bgr = cv2.imread(str(panel_path))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        mask_mser_orig = detect_text_mser(gray)
        mask_mser_inv = detect_text_mser(255 - gray)
        mask_laplacian = detect_text_laplacian(gray)
        combined = cv2.bitwise_or(mask_mser_orig, mask_mser_inv)
        combined = cv2.bitwise_or(combined, mask_laplacian)

        coverage = np.count_nonzero(combined) / combined.size * 100
        print(f"  Coverage: {coverage:.2f}%")

        # Create overlay
        overlay = image_rgb.copy()
        overlay[combined > 0] = [255, 0, 0]

        # Save debug figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title("Original")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(mask_mser_orig, cmap="gray")
        axes[0, 1].set_title("MSER (orig)")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(mask_mser_inv, cmap="gray")
        axes[0, 2].set_title("MSER (inv)")
        axes[0, 2].axis("off")

        axes[1, 0].imshow(mask_laplacian, cmap="gray")
        axes[1, 0].set_title("Laplacian")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(combined, cmap="gray")
        axes[1, 1].set_title(f"Combined ({coverage:.1f}%)")
        axes[1, 1].axis("off")

        axes[1, 2].imshow(overlay)
        axes[1, 2].set_title("Overlay")
        axes[1, 2].axis("off")

        plt.tight_layout()
        debug_path = output_dir / f"{panel_path.stem}_mser_v2_debug.png"
        plt.savefig(debug_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved debug to {debug_path}\n")


if __name__ == "__main__":
    main()
