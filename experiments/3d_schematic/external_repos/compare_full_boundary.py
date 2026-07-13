"""Compare full-image boundary preservation across variants."""
import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path("/Users/daiduo2/geoseg/src/3d_schematic/src/schematic_seg")))
from text_removal import generate_text_mask, inpaint_masked, detect_residual_mask, median_replace, remove_text_two_pass

BASE = Path("/Users/daiduo2/geoseg/src/3d_schematic")
OUT = BASE / "results" / "experiment_plan_repair" / "text_removal_optimized"

for idx in [1, 2, 3]:
    stem = f"panel_{idx}"
    image = cv2.imread(str(BASE / f"figures/panels/{stem}.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Baseline
    baseline = remove_text_two_pass(image)[0]

    # bt=150 no_stage2
    mask150 = generate_text_mask(image, brightness_thresh=150)
    no_stage2_150 = inpaint_masked(image, mask150, radius=3)

    # bt=150 + telea stage2 r=5
    first = inpaint_masked(image, mask150, radius=3)
    residual = detect_residual_mask(first, mask150)
    telea_s2_150 = cv2.inpaint(first, residual, 5, cv2.INPAINT_TELEA) if np.any(residual) else first

    # Pick boundary regions for each panel
    if idx == 1:
        boxes = [(1800, 2100, 150, 450), (3100, 3400, 1300, 1700)]
    elif idx == 2:
        boxes = [(2400, 2700, 150, 450), (1400, 1700, 1300, 1700)]
    else:
        boxes = [(1800, 2100, 150, 450), (2800, 3100, 200, 500)]

    for bi, box in enumerate(boxes):
        y1, y2, x1, x2 = box
        strips = [
            image[y1:y2, x1:x2],
            baseline[y1:y2, x1:x2],
            no_stage2_150[y1:y2, x1:x2],
            telea_s2_150[y1:y2, x1:x2],
        ]
        comp = np.hstack(strips)
        fname = OUT / f"{stem}_boundary_{bi}.png"
        cv2.imwrite(str(fname), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
        print(f"Saved {fname}")
