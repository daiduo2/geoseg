"""Group C: v3 pipeline parameter sweep.
Varies one parameter at a time to study sensitivity.
Uses fast mode: felzenszwalb only (no slow post_merge) for speed.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path("/Users/daiduo2/geoseg")))
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from skimage.segmentation import felzenszwalb

from core.evaluator import evaluate_segmentation
from core.vis_utils import render_label_fill

BASE = Path("/Users/daiduo2/geoseg/src/3d_schematic")
OUT = BASE / "results" / "segmentation_experiment" / "group_c"
OUT.mkdir(parents=True, exist_ok=True)


def load_text_removed(idx: int) -> np.ndarray:
    image = cv2.imread(str(BASE / f"figures/panels/panel_{idx}.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sys.path.insert(0, str(BASE / "src/schematic_seg"))
    from text_removal import remove_text_two_pass
    result, _, _, _ = remove_text_two_pass(image)
    return result


def enhance_v(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def v3_fast(image: np.ndarray, felz_scale: float, felz_sigma: float, felz_min_size: int):
    """Fast v3: felzenszwalb only, no post_merge."""
    enhanced = enhance_v(image)
    labels = felzenszwalb(enhanced, scale=felz_scale, sigma=felz_sigma, min_size=felz_min_size)
    return labels


def main():
    configs = []
    for v in [300.0, 600.0]:
        configs.append(("felz_scale", v, {"felz_scale": v, "felz_sigma": 0.5, "felz_min_size": 30}))
    for v in [0.3, 0.5, 0.8]:
        configs.append(("felz_sigma", v, {"felz_scale": 300.0, "felz_sigma": v, "felz_min_size": 30}))
    for v in [10, 30, 50]:
        configs.append(("felz_min_size", v, {"felz_scale": 300.0, "felz_sigma": 0.5, "felz_min_size": v}))

    summary = []
    for idx in [1, 2, 3]:
        stem = f"panel_{idx}"
        print(f"\n[{stem}] Loading text-removed image...")
        image = load_text_removed(idx)

        for param_name, param_val, cfg in configs:
            key = f"{stem}_{param_name}_{param_val}"
            print(f"  Running {key}...")
            try:
                labels = v3_fast(image, **cfg)
                metrics = evaluate_segmentation(image, labels)
                n_labels = len(np.unique(labels))

                fill = render_label_fill(labels)
                cv2.imwrite(str(OUT / f"{key}_fill.png"), cv2.cvtColor(fill, cv2.COLOR_RGB2BGR))

                print(f"    n_labels={n_labels}, frag={metrics['fragment_ratio']:.2f}, purity={metrics['color_purity']:.1f}")

                summary.append({
                    "panel": idx,
                    "param_name": param_name,
                    "param_val": param_val,
                    **cfg,
                    "n_init": n_labels,
                    "n_final": n_labels,
                    **metrics,
                })
            except Exception as e:
                print(f"    FAILED: {e}")
                summary.append({
                    "panel": idx,
                    "param_name": param_name,
                    "param_val": param_val,
                    **cfg,
                    "error": str(e),
                })

    import csv
    with open(OUT / "summary.csv", "w", newline="") as f:
        if summary:
            writer = csv.DictWriter(f, fieldnames=summary[0].keys())
            writer.writeheader()
            writer.writerows(summary)

    print(f"\nGroup C complete. Results in {OUT}")


if __name__ == "__main__":
    main()
