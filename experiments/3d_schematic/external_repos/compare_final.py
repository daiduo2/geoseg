import cv2
import numpy as np
from pathlib import Path

BASE = Path("/Users/daiduo2/geoseg/src/3d_schematic")

for idx in [1, 2, 3]:
    stem = f"panel_{idx}"
    orig = cv2.imread(str(BASE / f"figures/panels/{stem}.png"))
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    final = cv2.imread(str(BASE / f"experiments/text_removal_v2/final_pipeline/{stem}_final.png"))
    final = cv2.cvtColor(final, cv2.COLOR_BGR2RGB)

    # Same crop as patchmatch tests
    if idx == 1:
        box = (1800, 2100, 150, 450)
    elif idx == 2:
        box = (2400, 2700, 150, 450)
    else:
        box = (1800, 2100, 150, 450)

    y1, y2, x1, x2 = box
    c_orig = orig[y1:y2, x1:x2]
    c_final = final[y1:y2, x1:x2]

    comp = np.hstack([c_orig, c_final])
    out = BASE / "results" / "experiment_plan_repair" / "pypatchmatch_test" / f"{stem}_final_compare.png"
    cv2.imwrite(str(out), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
    print(f"Saved {out}")
