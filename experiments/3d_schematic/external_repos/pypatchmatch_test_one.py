"""Quick single-crop test for memory/time estimation."""
import numpy as np
import cv2
from pathlib import Path
import time
from pypatchmatch_safe import patchmatch_safe

BASE = Path("/Users/daiduo2/geoseg/src/3d_schematic")
OUT = BASE / "results" / "experiment_plan_repair" / "pypatchmatch_test"
OUT.mkdir(parents=True, exist_ok=True)

panel_idx = 1
stem = f"panel_{panel_idx}"
image = cv2.imread(str(BASE / f"figures/panels/{stem}.png"))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask = cv2.imread(str(BASE / f"experiments/text_removal_v2/final_pipeline/{stem}_mask.png"), cv2.IMREAD_GRAYSCALE)

box = (50, 350, 50, 500)
y1, y2, x1, x2 = box
crop_img = image[y1:y2, x1:x2]
crop_mask = mask[y1:y2, x1:x2]

print(f"crop={crop_img.shape} hole%={np.sum(crop_mask>127)/crop_mask.size*100:.2f}")

for ps, stride in [(7, 5), (11, 6), (15, 8)]:
    t0 = time.time()
    pm = patchmatch_safe(crop_img, crop_mask, patch_size=ps, source_stride=stride, hole_batch=64)
    print(f"ps={ps} stride={stride}: {time.time()-t0:.2f}s")
    m3 = np.stack([crop_mask] * 3, axis=-1)
    telea = cv2.inpaint(crop_img, crop_mask, 3, cv2.INPAINT_TELEA)
    ns = cv2.inpaint(crop_img, crop_mask, 3, cv2.INPAINT_NS)
    comp = np.hstack([crop_img, m3.astype(np.uint8), telea, ns, pm])
    cv2.imwrite(str(OUT / f"panel_1_top_left_ps{ps}_st{stride}.png"), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))

print("Quick test done.")
