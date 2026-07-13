"""
Debug script for MSER text detection.
Saves mask overlay and standalone mask for inspection.
"""

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def detect_text_regions(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)

    mask = np.zeros(gray.shape, dtype=np.uint8)
    kept = 0
    rejected_area = 0
    rejected_ratio = 0

    for region in regions:
        region = region.reshape(-1, 1, 2)
        area = cv2.contourArea(region)

        if area < 20 or area > 800:
            rejected_area += 1
            continue

        x, y, w, h = cv2.boundingRect(region)
        if w == 0 or h == 0:
            continue

        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio > 15:
            rejected_ratio += 1
            continue

        cv2.fillPoly(mask, [region], 255)
        kept += 1

    print(f"  Kept: {kept}, Rejected (area): {rejected_area}, Rejected (ratio): {rejected_ratio}")
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

        mask = detect_text_regions(image_rgb)
        coverage = np.count_nonzero(mask) / mask.size * 100
        print(f"  Mask coverage: {coverage:.2f}%")

        # Create overlay: red mask on original
        overlay = image_rgb.copy()
        overlay[mask > 0] = [255, 0, 0]

        # Save standalone mask
        mask_path = output_dir / f"{panel_path.stem}_mser_mask.png"
        cv2.imwrite(str(mask_path), mask)
        print(f"  Saved mask to {mask_path}")

        # Save overlay
        overlay_path = output_dir / f"{panel_path.stem}_mser_overlay.png"
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"  Saved overlay to {overlay_path}")

        # Save 3-panel debug figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image_rgb)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title(f"MSER Mask ({coverage:.1f}%)")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        debug_path = output_dir / f"{panel_path.stem}_mser_debug.png"
        plt.savefig(debug_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved debug figure to {debug_path}")
        print()


if __name__ == "__main__":
    main()
