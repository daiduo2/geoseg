"""
Median blur kernel size sweep: 5x5, 7x7, 9x9, 11x11
Compare on panel_3 with the improved v2 mask.
"""
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from mser_v2_framework import detect_text_mser, detect_text_laplacian


def build_refined_mask_v2(image_rgb, brightness_thresh=180, max_stroke_width=6, dilate_iter=1):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    mask_orig = detect_text_mser(gray, min_area=10, max_area=2000, max_aspect=20)
    mask_inv = detect_text_mser(255 - gray, min_area=10, max_area=2000, max_aspect=20)
    mask_lap = detect_text_laplacian(gray, threshold=15, max_area=2000)
    combined = cv2.bitwise_or(mask_orig, mask_inv)
    combined = cv2.bitwise_or(combined, mask_lap)
    brightness_mask = (gray < brightness_thresh).astype(np.uint8) * 255
    combined = cv2.bitwise_and(combined, brightness_mask)
    dist = cv2.distanceTransform(combined, cv2.DIST_L2, 5)
    half = max(1, max_stroke_width // 2)
    stroke_mask = ((dist > 0) & (dist <= half)).astype(np.uint8) * 255
    combined = stroke_mask
    if dilate_iter > 0:
        kernel = np.ones((3, 3), np.uint8)
        combined = cv2.dilate(combined, kernel, iterations=dilate_iter)
    return combined


def fill_median_blur(image_rgb, mask, ksize=7):
    mask_bool = mask.astype(bool)
    result = image_rgb.copy().astype(np.float32)
    for ch in range(3):
        chan = result[:, :, ch]
        med = cv2.medianBlur(chan.astype(np.uint8), ksize).astype(np.float32)
        result[:, :, ch] = np.where(mask_bool, med, chan)
    return result.astype(np.uint8)


def main():
    BASE = Path("/Users/daiduo2/geoseg/src/3d_schematic")
    PANELS_DIR = BASE / "figures" / "panels"
    OUT_DIR = BASE / "experiments" / "text_removal_v2" / "fill_strategy"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for panel_name in ["panel_3", "panel_1", "panel_2"]:
        panel_path = PANELS_DIR / f"{panel_name}.png"
        img = cv2.imread(str(panel_path), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = build_refined_mask_v2(img_rgb, brightness_thresh=180, max_stroke_width=6, dilate_iter=1)

        for ksize in [5, 7, 9, 11]:
            result = fill_median_blur(img_rgb, mask, ksize=ksize)
            out_path = OUT_DIR / f"{panel_name}_median_blur_{ksize}_v2.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            print(f"  {panel_name} median_blur_{ksize} saved")

    print(f"\nMedian blur sweep saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
