"""Compare v2 (final integrated) vs v3 (gauss expand) vs v4 (lower bt + larger expand)."""
import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path("/Users/daiduo2/geoseg/src/3d_schematic/src/schematic_seg")))
from text_removal import generate_text_mask, expand_mask_gaussian, inpaint_masked, detect_residual_mask, remove_text_two_pass

BASE = Path("/Users/daiduo2/geoseg/src/3d_schematic")
OUT = BASE / "results" / "experiment_plan_repair" / "v3_v4_audit"
OUT.mkdir(parents=True, exist_ok=True)

# Define crop regions focused on text-edge artifacts
CROPS = {
    1: [
        ("partial_melting", (1850, 2100, 100, 450)),
        ("mantle", (3100, 3350, 1200, 1600)),
        ("continental_crust", (100, 350, 150, 450)),
        ("upper_right", (100, 350, 1300, 1700)),
    ],
    2: [
        ("upper_left", (100, 350, 100, 450)),
        ("upper_right", (100, 350, 1300, 1700)),
        ("middle_text", (2400, 2700, 150, 450)),
    ],
    3: [
        ("upper_left", (100, 350, 100, 450)),
        ("upper_right", (100, 350, 1300, 1700)),
        ("middle_text", (1800, 2100, 150, 450)),
    ],
}

for idx in [1, 2, 3]:
    stem = f"panel_{idx}"
    image = cv2.imread(str(BASE / f"figures/panels/{stem}.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # v2: final integrated (from code history: bt=160, dilate=1, r=3 + stage2 telea r=5)
    v2_final, v2_first, v2_mask, v2_residual = remove_text_two_pass(
        image,
        brightness_thresh=160,
        dilate_iter=1,
        expand_sigma=0.0,  # No gaussian expansion in v2
        inpaint_radius=3,
        repair_radius=5,
    )

    # v3: current best (bt=160, gauss expand sigma=7, r=7 + stage2)
    v3_final, v3_first, v3_mask, v3_residual = remove_text_two_pass(
        image,
        brightness_thresh=160,
        dilate_iter=1,
        expand_sigma=7.0,
        expand_threshold=0.3,
        inpaint_radius=7,
        repair_radius=5,
    )

    # v4: lower bt + larger expand
    v4_mask = generate_text_mask(image, brightness_thresh=130, dilate_iter=1)
    v4_mask = expand_mask_gaussian(v4_mask, sigma=12.0, threshold=0.2)
    v4_first = inpaint_masked(image, v4_mask, radius=7)
    v4_residual = detect_residual_mask(v4_first, v4_mask, grow_threshold=20.0)
    v4_final = cv2.inpaint(v4_first, v4_residual, 5, cv2.INPAINT_TELEA) if np.any(v4_residual) else v4_first

    # v5: even lower bt + no brightness filter on edges? Just try bt=100 + sigma=15
    v5_mask = generate_text_mask(image, brightness_thresh=100, dilate_iter=1)
    v5_mask = expand_mask_gaussian(v5_mask, sigma=15.0, threshold=0.15)
    v5_first = inpaint_masked(image, v5_mask, radius=7)
    v5_residual = detect_residual_mask(v5_first, v5_mask, grow_threshold=20.0)
    v5_final = cv2.inpaint(v5_first, v5_residual, 5, cv2.INPAINT_TELEA) if np.any(v5_residual) else v5_first

    # Report mask coverage
    for name, mask in [("v2", v2_mask), ("v3", v3_mask), ("v4", v4_mask), ("v5", v5_mask)]:
        pct = np.sum(mask > 127) / mask.size * 100
        print(f"  {stem} {name}: mask={pct:.2f}%")

    # Save full images
    for name, img in [("v2", v2_final), ("v3", v3_final), ("v4", v4_final), ("v5", v5_final)]:
        cv2.imwrite(str(OUT / f"{stem}_{name}_full.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # Save crop comparisons
    for crop_name, box in CROPS[idx]:
        y1, y2, x1, x2 = box
        strips = [
            image[y1:y2, x1:x2],
            v2_final[y1:y2, x1:x2],
            v3_final[y1:y2, x1:x2],
            v4_final[y1:y2, x1:x2],
            v5_final[y1:y2, x1:x2],
        ]
        comp = np.hstack(strips)
        fname = OUT / f"{stem}_{crop_name}.png"
        cv2.imwrite(str(fname), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
        print(f"  Saved {fname}")

print("\nDone.")
