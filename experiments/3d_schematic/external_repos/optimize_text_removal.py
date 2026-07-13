"""Test text_removal_v2 variants to find optimal repair strategy.

Comparing:
  A. baseline:   two_pass (Telea r=3 + median blur 71x71)
  B. no_stage2:  Stage 1 only (Telea r=3)
  C. telea_s2:   Stage 1 Telea r=3 + Stage 2 Telea r=3 on residual
  D. larger_r5:  Stage 1 Telea r=5, no Stage 2
  E. larger_r7:  Stage 1 Telea r=7, no Stage 2
  F. small_med:  Stage 1 Telea r=3 + Stage 2 median blur 21x21
"""
import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path("/Users/daiduo2/geoseg/src/3d_schematic/src/schematic_seg")))
from text_removal import (
    generate_text_mask, inpaint_masked, detect_residual_mask, median_replace,
    remove_text_two_pass
)

BASE = Path("/Users/daiduo2/geoseg/src/3d_schematic")
OUT = BASE / "results" / "experiment_plan_repair" / "text_removal_optimized"
OUT.mkdir(parents=True, exist_ok=True)


def variant_no_stage2(image, mask):
    return inpaint_masked(image, mask, radius=3)


def variant_telea_stage2(image, mask):
    first = inpaint_masked(image, mask, radius=3)
    residual = detect_residual_mask(first, mask)
    if np.any(residual):
        return cv2.inpaint(first, residual, 3, cv2.INPAINT_TELEA)
    return first


def variant_larger_r5(image, mask):
    return inpaint_masked(image, mask, radius=5)


def variant_larger_r7(image, mask):
    return inpaint_masked(image, mask, radius=7)


def variant_small_median(image, mask):
    first = inpaint_masked(image, mask, radius=3)
    residual = detect_residual_mask(first, mask)
    if np.any(residual):
        return median_replace(first, residual, ksize=21)
    return first


def run_all(panel_idx: int):
    stem = f"panel_{panel_idx}"
    image = cv2.imread(str(BASE / f"figures/panels/{stem}.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = generate_text_mask(image)

    results = {
        "A_baseline": remove_text_two_pass(image)[0],
        "B_no_stage2": variant_no_stage2(image, mask),
        "C_telea_s2": variant_telea_stage2(image, mask),
        "D_larger_r5": variant_larger_r5(image, mask),
        "E_larger_r7": variant_larger_r7(image, mask),
        "F_small_med": variant_small_median(image, mask),
    }

    # Save full images
    for name, result in results.items():
        cv2.imwrite(str(OUT / f"{stem}_{name}.png"), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

    # Crop comparison at heavy text region
    if panel_idx == 1:
        box = (1800, 2100, 150, 450)
    elif panel_idx == 2:
        box = (2400, 2700, 150, 450)
    else:
        box = (1800, 2100, 150, 450)
    y1, y2, x1, x2 = box

    # Build comparison strip: orig + each variant
    strips = [image[y1:y2, x1:x2]]
    for name in ["A_baseline", "B_no_stage2", "C_telea_s2", "D_larger_r5", "E_larger_r7", "F_small_med"]:
        strips.append(results[name][y1:y2, x1:x2])

    comp = np.hstack(strips)
    cv2.imwrite(str(OUT / f"{stem}_crop_compare.png"), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
    print(f"[{stem}] Saved crop comparison: {OUT / f'{stem}_crop_compare.png'}")

    # Compute simple metrics: residual detection on each variant
    for name, result in results.items():
        # Re-detect residual on result
        gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        residual_mask = np.zeros(gray.shape, dtype=np.uint8)
        for region in regions:
            region = region.reshape(-1, 1, 2)
            area = cv2.contourArea(region)
            if 5 <= area <= 3000:
                cv2.fillPoly(residual_mask, [region], 255)
        # Count residual pixels within original mask area
        in_mask = np.sum((residual_mask > 0) & (mask > 127))
        total_mask = np.sum(mask > 127)
        print(f"  {name}: residual_in_mask={in_mask}/{total_mask} ({in_mask/max(1,total_mask)*100:.1f}%)")


if __name__ == "__main__":
    for idx in [1, 2, 3]:
        run_all(idx)
    print("\nAll variants complete.")
