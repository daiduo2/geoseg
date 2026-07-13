"""Real-world PatchMatch test on high-density text regions."""
import numpy as np
import cv2
from pathlib import Path
import time
from pypatchmatch_fast_test import patchmatch_fast

BASE = Path("/Users/daiduo2/geoseg/src/3d_schematic")
OUT = BASE / "results" / "experiment_plan_repair" / "pypatchmatch_test"
OUT.mkdir(parents=True, exist_ok=True)


def run(panel_idx: int, name: str, box: tuple, ps: int, stride: int):
    stem = f"panel_{panel_idx}"
    image = cv2.imread(str(BASE / f"figures/panels/{stem}.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(BASE / f"experiments/text_removal_v2/final_pipeline/{stem}_mask.png"), cv2.IMREAD_GRAYSCALE)

    y1, y2, x1, x2 = box
    crop_img = image[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]

    hole_pct = np.sum(crop_mask > 127) / crop_mask.size * 100
    print(f"[{stem} {name}] {crop_img.shape} hole={hole_pct:.2f}% ps={ps} stride={stride}")

    t0 = time.time()
    telea = cv2.inpaint(crop_img, crop_mask, 3, cv2.INPAINT_TELEA)
    t_telea = time.time() - t0

    t0 = time.time()
    ns = cv2.inpaint(crop_img, crop_mask, 3, cv2.INPAINT_NS)
    t_ns = time.time() - t0

    t0 = time.time()
    pm = patchmatch_fast(crop_img, crop_mask, patch_size=ps, source_stride=stride,
                         hole_batch=128, src_batch=512)
    t_pm = time.time() - t0

    print(f"  Telea {t_telea:.2f}s | NS {t_ns:.2f}s | PM {t_pm:.2f}s")

    m3 = np.stack([crop_mask] * 3, axis=-1)
    comp = np.hstack([crop_img, m3.astype(np.uint8), telea, ns, pm])
    fname = OUT / f"{stem}_{name}_ps{ps}_st{stride}.png"
    cv2.imwrite(str(fname), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
    print(f"  Saved: {fname}")


if __name__ == "__main__":
    # High-density text regions from find_text_regions.py
    TESTS = [
        # Panel 1: big text block near layer boundary
        (1, "heavy_text", (1800, 2100, 150, 450), 9, 12),
        (1, "heavy_text", (1800, 2100, 150, 450), 15, 15),
        # Panel 2: big text block
        (2, "heavy_text", (2400, 2700, 150, 450), 9, 12),
        (2, "heavy_text", (2400, 2700, 150, 450), 15, 15),
        # Panel 3: big text block (texture background)
        (3, "heavy_text", (1800, 2100, 150, 450), 9, 12),
        (3, "heavy_text", (1800, 2100, 150, 450), 15, 15),
    ]

    for panel_idx, name, box, ps, stride in TESTS:
        run(panel_idx, name, box, ps, stride)

    print("\nDone.")
