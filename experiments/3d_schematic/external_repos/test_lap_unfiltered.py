"""Test strategy: Laplacian bypasses brightness filter + larger expand."""
import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path("/Users/daiduo2/geoseg/src/3d_schematic/src/schematic_seg")))
from text_removal import _detect_text_mser, _detect_text_laplacian, expand_mask_gaussian, inpaint_masked, detect_residual_mask

BASE = Path("/Users/daiduo2/geoseg/src/3d_schematic")
OUT = BASE / "results" / "experiment_plan_repair" / "lap_unfiltered_test"
OUT.mkdir(parents=True, exist_ok=True)

def generate_text_mask_v2(
    image_rgb: np.ndarray,
    brightness_thresh: int = 160,
    dilate_iter: int = 1,
    lap_threshold: int = 10,
) -> np.ndarray:
    """Generate text mask: MSER filtered by brightness, Laplacian unfiltered."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    mask_orig = _detect_text_mser(gray, 10, 2000, 20)
    mask_inv = _detect_text_mser(255 - gray, 10, 2000, 20)
    mask_lap = _detect_text_laplacian(gray, lap_threshold, 2000)

    combined = cv2.bitwise_or(mask_orig, mask_inv)
    if brightness_thresh > 0:
        brightness_mask = (gray > brightness_thresh).astype(np.uint8) * 255
        combined = cv2.bitwise_and(combined, brightness_mask)

    # Laplacian bypasses brightness filter
    combined = cv2.bitwise_or(combined, mask_lap)

    if dilate_iter > 0:
        kernel = np.ones((3, 3), np.uint8)
        combined = cv2.dilate(combined, kernel, iterations=dilate_iter)

    return combined


def run_variant(image, name, bt, expand_sigma, expand_thresh, lap_thresh):
    mask = generate_text_mask_v2(image, brightness_thresh=bt, lap_threshold=lap_thresh)
    mask = expand_mask_gaussian(mask, expand_sigma, expand_thresh)
    first = inpaint_masked(image, mask, radius=7)
    residual = detect_residual_mask(first, mask, grow_threshold=20.0)
    final = cv2.inpaint(first, residual, 5, cv2.INPAINT_TELEA) if np.any(residual) else first
    pct = np.sum(mask > 127) / mask.size * 100
    print(f"  {name}: mask={pct:.2f}%")
    return final, mask


for idx in [1, 2, 3]:
    stem = f"panel_{idx}"
    image = cv2.imread(str(BASE / f"figures/panels/{stem}.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Original v3
    mask_v3 = generate_text_mask_v2(image, brightness_thresh=160, lap_threshold=15)
    mask_v3 = expand_mask_gaussian(mask_v3, 7.0, 0.3)
    first_v3 = inpaint_masked(image, mask_v3, radius=7)
    residual_v3 = detect_residual_mask(first_v3, mask_v3)
    v3 = cv2.inpaint(first_v3, residual_v3, 5, cv2.INPAINT_TELEA) if np.any(residual_v3) else first_v3
    pct_v3 = np.sum(mask_v3 > 127) / mask_v3.size * 100
    print(f"{stem} v3: mask={pct_v3:.2f}%")

    # Variant A: Laplacian unfiltered + larger expand
    vA, mA = run_variant(image, "vA", bt=160, expand_sigma=10, expand_thresh=0.2, lap_thresh=10)

    # Variant B: Lower bt + Laplacian unfiltered
    vB, mB = run_variant(image, "vB", bt=130, expand_sigma=7, expand_thresh=0.3, lap_thresh=10)

    # Variant C: Lower bt + larger expand
    vC, mC = run_variant(image, "vC", bt=130, expand_sigma=10, expand_thresh=0.2, lap_thresh=10)

    # Full image outputs
    for name, img in [("v3", v3), ("vA", vA), ("vB", vB), ("vC", vC)]:
        cv2.imwrite(str(OUT / f"{stem}_{name}_full.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # Crop comparisons
    if idx == 1:
        crops = [("partial", (1850, 2100, 100, 450)), ("mantle", (3100, 3350, 1200, 1600))]
    elif idx == 2:
        crops = [("upper", (100, 350, 100, 450)), ("middle", (2400, 2700, 150, 450))]
    else:
        crops = [("upper", (100, 350, 100, 450)), ("middle", (1800, 2100, 150, 450))]

    for cname, box in crops:
        y1, y2, x1, x2 = box
        comp = np.hstack([image[y1:y2, x1:x2], v3[y1:y2, x1:x2], vA[y1:y2, x1:x2], vB[y1:y2, x1:x2], vC[y1:y2, x1:x2]])
        cv2.imwrite(str(OUT / f"{stem}_{cname}.png"), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
        print(f"  Saved {stem}_{cname}.png")

print("\nDone.")
