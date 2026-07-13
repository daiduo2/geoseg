"""Compare residual repair methods: median blur vs Telea re-inpaint."""
import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path("/Users/daiduo2/geoseg/src/3d_schematic/src/schematic_seg")))
from text_removal import remove_text_two_pass

BASE = Path("/Users/daiduo2/geoseg/src/3d_schematic")
OUT = BASE / "results" / "experiment_plan_repair" / "pypatchmatch_test"

for idx in [1, 2, 3]:
    stem = f"panel_{idx}"
    image = cv2.imread(str(BASE / f"figures/panels/{stem}.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    final, first, first_mask, residual = remove_text_two_pass(image)

    # Method 1: original median blur
    median = final

    # Method 2: Telea re-inpaint on residual
    telea_r3 = cv2.inpaint(first, residual, 3, cv2.INPAINT_TELEA)
    telea_r5 = cv2.inpaint(first, residual, 5, cv2.INPAINT_TELEA)
    telea_r7 = cv2.inpaint(first, residual, 7, cv2.INPAINT_TELEA)

    # Method 3: larger initial inpaint radius (skip stage 2)
    larger_r5 = cv2.inpaint(image, first_mask, 5, cv2.INPAINT_TELEA)
    larger_r7 = cv2.inpaint(image, first_mask, 7, cv2.INPAINT_TELEA)

    # Crop
    if idx == 1:
        box = (1800, 2100, 150, 450)
    elif idx == 2:
        box = (2400, 2700, 150, 450)
    else:
        box = (1800, 2100, 150, 450)
    y1, y2, x1, x2 = box

    c_orig = image[y1:y2, x1:x2]
    c_first = first[y1:y2, x1:x2]
    c_median = median[y1:y2, x1:x2]
    c_t3 = telea_r3[y1:y2, x1:x2]
    c_t5 = telea_r5[y1:y2, x1:x2]
    c_t7 = telea_r7[y1:y2, x1:x2]
    c_lr5 = larger_r5[y1:y2, x1:x2]
    c_lr7 = larger_r7[y1:y2, x1:x2]

    # Compare: orig | first | median | telea_r5 | larger_r5
    comp = np.hstack([c_orig, c_first, c_median, c_t5, c_lr5])
    fname = OUT / f"{stem}_repair_compare.png"
    cv2.imwrite(str(fname), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
    print(f"Saved {fname}")
