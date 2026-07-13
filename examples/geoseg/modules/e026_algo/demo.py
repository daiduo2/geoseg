"""M0 standalone demo: verify e026 algorithm移植 correctness.

Input: ph01_page8_300dpi.png (from e026 fixture, temporary until M0.5 pdf_extractor)
Output: runs/M0/pattern1_overlay.png + segmentation_result.json
Verification: compare overlay MSE against e026 original (≤ 1%).
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .core import auto_extract_seeds, segment_fixed_nn, create_vivid_overlay
from .components import build_segmentation_result


# Pattern1 coordinates from e026 original (hardcoded for demo baseline verification)
PATTERN1 = {"x1": 494, "x2": 752, "y1": 1197, "y2": 1560, "cx1": 28, "cx2": 256}
N_LAYERS = 7


def main():
    base = Path(__file__).resolve().parents[3]
    fixture = base / "tests" / "fixtures" / "ph01"
    out_dir = base / "runs" / "M0"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load page image (temporary: will switch to PDF extracted image after M0.5)
    page_path = fixture / "ph01_page8_300dpi.png"
    page = np.array(Image.open(page_path).convert("RGB"))
    print(f"Loaded page: {page.shape[1]}x{page.shape[0]}")

    # Extract pattern1 panel region (same as e026 original)
    p = PATTERN1
    full_crop = page[p["y1"]:p["y2"], p["x1"]:p["x2"]]
    content_crop = full_crop[:, p["cx1"]:p["cx2"]]
    print(f"Content crop: {content_crop.shape[1]}x{content_crop.shape[0]}")

    # Run e026 algorithm
    print(f"Extracting {N_LAYERS} seeds...")
    seeds = auto_extract_seeds(content_crop, n_layers=N_LAYERS)
    print(f"Seeds: {seeds}")

    print("Segmenting...")
    labels = segment_fixed_nn(content_crop, seeds)

    print("Creating overlay...")
    overlay, vivid_colors = create_vivid_overlay(content_crop, labels, alpha=0.5)

    # Extract components
    print("Extracting components...")
    result = build_segmentation_result(labels, content_crop, min_area=200)
    result["overlay_path"] = str(out_dir / "pattern1_overlay.png")

    # Save outputs
    Image.fromarray(overlay).save(out_dir / "pattern1_overlay.png")
    (out_dir / "segmentation_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    print(f"Saved overlay: {out_dir / 'pattern1_overlay.png'}")
    print(f"Saved result:  {out_dir / 'segmentation_result.json'}")
    print(f"Components: {len(result['components'])}")
    for c in result["components"]:
        print(f"  Layer {c['layer_id']}: id={c['id']}, area={c['area']}, bbox={c['bbox']}")

    # Verification: compare against e026 original overlay
    original_overlay_path = Path(
        "/Users/daiduo2/Documents/knowlege/Projects/精密院-地震逆散射/"
        "photo/experiments/e026_ph01_conceptual/outputs/rowa_v5_vivid_saturation/"
        "pattern1_overlay.png"
    )
    if original_overlay_path.exists():
        orig = np.array(Image.open(original_overlay_path).convert("RGB"))
        # Resize to match if sizes differ
        if orig.shape != overlay.shape:
            orig = np.array(Image.fromarray(orig).resize((overlay.shape[1], overlay.shape[0])))
        mse = float(np.mean((overlay.astype(float) - orig.astype(float)) ** 2))
        max_val = 255.0
        mse_pct = (mse / (max_val ** 2)) * 100
        print(f"\nMSE vs original: {mse:.2f} ({mse_pct:.3f}%)")
        if mse_pct <= 1.0:
            print("PASS: MSE ≤ 1%")
        else:
            print(f"FAIL: MSE > 1% (threshold: 1%)")
    else:
        print("\nOriginal overlay not found, skipping MSE verification")


if __name__ == "__main__":
    main()
