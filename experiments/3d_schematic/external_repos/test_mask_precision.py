"""Test if improving mask precision reduces residual without median blur."""
import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path("/Users/daiduo2/geoseg/src/3d_schematic/src/schematic_seg")))
from text_removal import generate_text_mask, inpaint_masked, detect_residual_mask, median_replace

BASE = Path("/Users/daiduo2/geoseg/src/3d_schematic")
OUT = BASE / "results" / "experiment_plan_repair" / "text_removal_optimized"


def test_variant(image, mask, label):
    first = inpaint_masked(image, mask, radius=3)
    residual = detect_residual_mask(first, mask)
    has_res = np.any(residual)

    # Metrics
    total_mask = np.sum(mask > 127)

    # Re-detect residual on result (same heuristic as optimize_text_removal.py)
    gray = cv2.cvtColor(first, cv2.COLOR_RGB2GRAY)
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    res_mask = np.zeros(gray.shape, dtype=np.uint8)
    for region in regions:
        region = region.reshape(-1, 1, 2)
        area = cv2.contourArea(region)
        if 5 <= area <= 3000:
            cv2.fillPoly(res_mask, [region], 255)
    in_mask = np.sum((res_mask > 0) & (mask > 127))

    # Also compute median-blur repaired version
    if has_res:
        med = median_replace(first, residual, ksize=71)
    else:
        med = first

    return first, med, in_mask, total_mask, has_res


for idx in [1, 2, 3]:
    stem = f"panel_{idx}"
    image = cv2.imread(str(BASE / f"figures/panels/{stem}.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(f"\n=== {stem} ===")

    # Test different brightness thresholds
    for bt in [170, 150, 130, 100, 0]:
        mask = generate_text_mask(image, brightness_thresh=bt)
        first, med, res, total, has_res = test_variant(image, mask, f"bt={bt}")
        pct = res / max(1, total) * 100
        print(f"  brightness_thresh={bt:3d}: mask_pixels={total:6d} residual={res:5d} ({pct:5.1f}%) has_res={has_res}")

        # Save crop for bt=150 (most promising)
        if bt == 150:
            if idx == 1:
                box = (1800, 2100, 150, 450)
            elif idx == 2:
                box = (2400, 2700, 150, 450)
            else:
                box = (1800, 2100, 150, 450)
            y1, y2, x1, x2 = box
            comp = np.hstack([
                image[y1:y2, x1:x2],
                first[y1:y2, x1:x2],
                med[y1:y2, x1:x2],
            ])
            cv2.imwrite(
                str(OUT / f"{stem}_bt150_compare.png"),
                cv2.cvtColor(comp, cv2.COLOR_RGB2BGR)
            )
