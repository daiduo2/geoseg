"""Fine-tune brightness threshold: 150 vs 160 vs 170 baseline."""
import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path("/Users/daiduo2/geoseg/src/3d_schematic/src/schematic_seg")))
from text_removal import generate_text_mask, inpaint_masked, detect_residual_mask, remove_text_two_pass

BASE = Path("/Users/daiduo2/geoseg/src/3d_schematic")
OUT = BASE / "results" / "experiment_plan_repair" / "text_removal_optimized"

for idx in [1, 2, 3]:
    stem = f"panel_{idx}"
    image = cv2.imread(str(BASE / f"figures/panels/{stem}.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    baseline = remove_text_two_pass(image)[0]

    def run(bt):
        mask = generate_text_mask(image, brightness_thresh=bt)
        first = inpaint_masked(image, mask, radius=3)
        residual = detect_residual_mask(first, mask)
        if np.any(residual):
            return cv2.inpaint(first, residual, 5, cv2.INPAINT_TELEA)
        return first

    bt150 = run(150)
    bt160 = run(160)

    if idx == 1:
        boxes = [(1800, 2100, 150, 450), (100, 400, 150, 450)]
    elif idx == 2:
        boxes = [(2400, 2700, 150, 450), (1400, 1700, 1300, 1700)]
    else:
        boxes = [(1800, 2100, 150, 450), (2800, 3100, 200, 500)]

    for bi, box in enumerate(boxes):
        y1, y2, x1, x2 = box
        comp = np.hstack([image[y1:y2, x1:x2], baseline[y1:y2, x1:x2], bt150[y1:y2, x1:x2], bt160[y1:y2, x1:x2]])
        fname = OUT / f"{stem}_tune_{bi}.png"
        cv2.imwrite(str(fname), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
        print(f"Saved {fname}")

print("\nDone.")
