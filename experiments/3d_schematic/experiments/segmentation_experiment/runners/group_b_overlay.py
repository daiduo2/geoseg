"""Group B: diff-overlay parameter sweep.
Varies one parameter at a time to study sensitivity.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path("/Users/daiduo2/geoseg")))
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from core.diff_overlay import diff_overlay_pipeline
from core.evaluator import evaluate_segmentation
from core.vis_utils import render_label_fill

BASE = Path("/Users/daiduo2/geoseg/src/3d_schematic")
OUT = BASE / "results" / "segmentation_experiment" / "group_b"
OUT.mkdir(parents=True, exist_ok=True)


def load_text_removed(idx: int) -> np.ndarray:
    image = cv2.imread(str(BASE / f"figures/panels/panel_{idx}.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sys.path.insert(0, str(BASE / "src/schematic_seg"))
    from text_removal import remove_text_two_pass
    result, _, _, _ = remove_text_two_pass(image)
    return result


def main():
    # Baseline: all middle values
    baseline = dict(blur_ksize=15, blur_sigma=3.0, diff_thresh=20.0, expand_radius=15)

    # Vary one parameter at a time
    configs = []
    for v in [7, 15, 21]:
        c = baseline.copy(); c["blur_ksize"] = v; configs.append(("blur_ksize", v, c))
    for v in [1.5, 3.0, 5.0]:
        c = baseline.copy(); c["blur_sigma"] = v; configs.append(("blur_sigma", v, c))
    for v in [10.0, 20.0, 30.0]:
        c = baseline.copy(); c["diff_thresh"] = v; configs.append(("diff_thresh", v, c))
    for v in [8, 15, 25]:
        c = baseline.copy(); c["expand_radius"] = v; configs.append(("expand_radius", v, c))

    summary = []
    for idx in [1, 2, 3]:
        stem = f"panel_{idx}"
        print(f"\n[{stem}] Loading text-removed image...")
        image = load_text_removed(idx)

        for param_name, param_val, cfg in configs:
            key = f"{stem}_{param_name}_{param_val}"
            print(f"  Running {key}...")
            try:
                result = diff_overlay_pipeline(image, **cfg)
                labels = result["final_labels"]
                overlay_mask = result["overlay_mask"]
                metrics = evaluate_segmentation(image, labels)
                overlay_ratio = overlay_mask.sum() / overlay_mask.size

                # Save key visuals
                fill = render_label_fill(labels)
                cv2.imwrite(str(OUT / f"{key}_fill.png"), cv2.cvtColor(fill, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(OUT / f"{key}_overlay.png"), cv2.cvtColor(result["overlay_only"], cv2.COLOR_RGB2BGR))

                # Save detail map as normalized grayscale
                detail = result["detail"]
                detail_norm = (detail / detail.max() * 255).astype(np.uint8) if detail.max() > 0 else detail.astype(np.uint8)
                cv2.imwrite(str(OUT / f"{key}_detail.png"), detail_norm)

                print(f"    n_labels={metrics['n_labels']}, frag={metrics['fragment_ratio']:.2f}, overlay_ratio={overlay_ratio:.3f}")

                summary.append({
                    "panel": idx,
                    "param_name": param_name,
                    "param_val": param_val,
                    **cfg,
                    "overlay_ratio": round(overlay_ratio, 4),
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

    print(f"\nGroup B complete. Results in {OUT}")


if __name__ == "__main__":
    main()
