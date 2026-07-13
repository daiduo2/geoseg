"""M3 standalone demo: verify pipeline on ph01 panel.

Input: ph01.jpg (concept model image)
Output: runs/M3/pattern1_overlay.png + pattern1_segmentation_result.json
Verification: segmentation_result passes schema validation, components >= 5.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .pipeline import run_panel, save_result


def main():
    base = Path(__file__).resolve().parents[3]
    img_path = Path(
        "/Users/daiduo2/Documents/knowlege/Projects/精密院-地震逆散射/"
        "photo/experiments/e026_ph01_conceptual/inputs/ph01.jpg"
    )
    out_dir = base / "runs" / "M3"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not img_path.exists():
        print(f"ERROR: {img_path} not found.")
        return

    img = np.array(Image.open(img_path).convert("RGB"))
    print(f"Loaded image: {img.shape[1]}x{img.shape[0]}")

    # Use the first detected panel from M1b (candidate id=1: x=255, w=157)
    # Hardcoded for reproducibility; in practice this comes from cv_detect
    panel_bbox = [255, 38, 157, 633]
    x, y, w, h = panel_bbox
    content_crop = img[y:y+h, x:x+w]
    print(f"Panel content: {content_crop.shape[1]}x{content_crop.shape[0]}")

    # Run pipeline
    print("Running segmentation pipeline...")
    result = run_panel(content_crop, n_layers=7, min_area=200, alpha=0.5)

    # Save
    result = save_result(result, out_dir, "pattern1")

    # Report
    n_components = len(result["components"])
    n_layers = len(result["layers"])
    print(f"\nComponents: {n_components}")
    print(f"Layers: {n_layers}")
    print(f"Overlay: {result['overlay_path']}")

    for c in result["components"][:5]:
        print(f"  Layer {c['layer_id']}: id={c['id']}, area={c['area']}, bbox={c['bbox']}")

    # Verification
    checks = []
    checks.append(("components >= 5", n_components >= 5))
    checks.append(("layers == 7", n_layers == 7))
    checks.append(("overlay_exists", Path(result["overlay_path"]).exists()))

    # Schema validation
    required_keys = {"components", "layers", "overlay_path"}
    result_keys = set(json.loads(Path(result["overlay_path"]).with_suffix("").with_name("pattern1_segmentation_result.json").read_text()).keys())
    checks.append(("schema_has_required_keys", required_keys.issubset(result_keys)))

    print("\n=== Verification ===")
    all_pass = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nM3 PASS: All checks passed")
    else:
        print("\nM3 FAIL: Some checks failed")


if __name__ == "__main__":
    main()
