"""Final comparison: original vs old (Lap filtered) vs new (Lap unfiltered)."""
import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path("/Users/daiduo2/geoseg/src/3d_schematic/src/schematic_seg")))
from text_removal import generate_text_mask, expand_mask_gaussian, inpaint_masked, detect_residual_mask

BASE = Path("/Users/daiduo2/geoseg/src/3d_schematic")
OUT = BASE / "results" / "experiment_plan_repair" / "final_lap_audit"
OUT.mkdir(parents=True, exist_ok=True)


def run_old_strategy(image):
    """Old strategy: Laplacian filtered by brightness."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    from text_removal import _detect_text_mser, _detect_text_laplacian
    mask_orig = _detect_text_mser(gray, 10, 2000, 20)
    mask_inv = _detect_text_mser(255 - gray, 10, 2000, 20)
    mask_lap = _detect_text_laplacian(gray, 15, 2000)
    combined = cv2.bitwise_or(mask_orig, mask_inv)
    combined = cv2.bitwise_or(combined, mask_lap)
    brightness_mask = (gray > 160).astype(np.uint8) * 255
    combined = cv2.bitwise_and(combined, brightness_mask)
    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.dilate(combined, kernel, iterations=1)
    mask = expand_mask_gaussian(combined, 7.0, 0.3)
    first = inpaint_masked(image, mask, radius=7)
    residual = detect_residual_mask(first, mask)
    final = cv2.inpaint(first, residual, 5, cv2.INPAINT_TELEA) if np.any(residual) else first
    return final, mask


def run_new_strategy(image):
    """New strategy: Laplacian bypasses brightness filter."""
    mask = generate_text_mask(image, brightness_thresh=160, dilate_iter=1)
    mask = expand_mask_gaussian(mask, 7.0, 0.3)
    first = inpaint_masked(image, mask, radius=7)
    residual = detect_residual_mask(first, mask)
    final = cv2.inpaint(first, residual, 5, cv2.INPAINT_TELEA) if np.any(residual) else first
    return final, mask


for idx in [1, 2, 3]:
    stem = f"panel_{idx}"
    image = cv2.imread(str(BASE / f"figures/panels/{stem}.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    old_final, old_mask = run_old_strategy(image)
    new_final, new_mask = run_new_strategy(image)

    old_pct = np.sum(old_mask > 127) / old_mask.size * 100
    new_pct = np.sum(new_mask > 127) / new_mask.size * 100
    print(f"{stem}: old_mask={old_pct:.2f}% new_mask={new_pct:.2f}%")

    # Save full comparison
    comp_full = np.hstack([image, old_final, new_final])
    cv2.imwrite(str(OUT / f"{stem}_full_compare.png"), cv2.cvtColor(comp_full, cv2.COLOR_RGB2BGR))

    # Save mask comparison
    mask_comp = np.hstack([
        np.stack([old_mask]*3, axis=-1),
        np.stack([new_mask]*3, axis=-1),
    ])
    cv2.imwrite(str(OUT / f"{stem}_mask_compare.png"), mask_comp)

    # Crops focused on text regions
    if idx == 1:
        crops = [("partial_melting", (1850, 2100, 100, 450)), ("mantle", (3100, 3350, 1200, 1600)), ("crust", (100, 350, 150, 450))]
    elif idx == 2:
        crops = [("upper", (100, 350, 100, 450)), ("middle", (2400, 2700, 150, 450))]
    else:
        crops = [("upper", (100, 350, 100, 450)), ("middle", (1800, 2100, 150, 450))]

    for cname, box in crops:
        y1, y2, x1, x2 = box
        comp = np.hstack([image[y1:y2, x1:x2], old_final[y1:y2, x1:x2], new_final[y1:y2, x1:x2]])
        fname = OUT / f"{stem}_{cname}_compare.png"
        cv2.imwrite(str(fname), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
        print(f"  Saved {fname}")

print("\nDone.")
