"""Analyze text_removal_v2 pipeline stages."""
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

    # Same crop as patchmatch tests
    if idx == 1:
        box = (1800, 2100, 150, 450)
    elif idx == 2:
        box = (2400, 2700, 150, 450)
    else:
        box = (1800, 2100, 150, 450)

    y1, y2, x1, x2 = box
    c_orig = image[y1:y2, x1:x2]
    c_first = first[y1:y2, x1:x2]
    c_final = final[y1:y2, x1:x2]
    c_mask = first_mask[y1:y2, x1:x2]
    c_res = residual[y1:y2, x1:x2]

    # 5 columns: orig, mask, first_result, residual_mask, final
    m3 = np.stack([c_mask] * 3, axis=-1)
    r3 = np.stack([c_res] * 3, axis=-1)
    comp = np.hstack([c_orig, m3.astype(np.uint8), c_first, r3.astype(np.uint8), c_final])

    fname = OUT / f"{stem}_pipeline_stages.png"
    cv2.imwrite(str(fname), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
    print(f"Saved {fname}")
    print(f"  mask pixels: {np.sum(c_mask > 127)}, residual pixels: {np.sum(c_res > 127)}")
