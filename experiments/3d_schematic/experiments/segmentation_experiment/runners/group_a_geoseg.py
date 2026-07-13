"""Group A: geoseg engine family baseline experiment.
Runs v4_kmeans, slic_kmeans, edge_guided, grayscale, ensemble on text-removed panels.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path("/Users/daiduo2/geoseg")))
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from PIL import Image

from core.evaluator import evaluate_segmentation
from core.vis_utils import render_label_fill, draw_boundaries

BASE = Path("/Users/daiduo2/geoseg/src/3d_schematic")
OUT = BASE / "results" / "segmentation_experiment" / "group_a"
OUT.mkdir(parents=True, exist_ok=True)


def load_text_removed(idx: int) -> np.ndarray:
    stem = f"panel_{idx}"
    image = cv2.imread(str(BASE / f"figures/panels/{stem}.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sys.path.insert(0, str(BASE / "src/schematic_seg"))
    from text_removal import remove_text_two_pass
    result, _, _, _ = remove_text_two_pass(image)
    return result


def run_v4_kmeans(image: np.ndarray, n_layers: int):
    from geoseg.modules.segment_engines.v4_kmeans import segment
    result = segment(image, n_layers=n_layers)
    return result["labels"]


def run_slic_kmeans(image: np.ndarray, n_layers: int):
    from geoseg.modules.segment_engines.slic_kmeans import segment
    result = segment(image, n_layers=n_layers)
    return result["labels"]


def run_edge_guided(image: np.ndarray, n_layers: int):
    from geoseg.modules.segment_engines.edge_guided import segment
    result = segment(image, n_layers=n_layers)
    return result["labels"]


def run_grayscale(image: np.ndarray, n_layers: int):
    from geoseg.modules.segment_engines.grayscale import segment
    result = segment(image, n_layers=n_layers)
    return result["labels"]


def run_ensemble(image: np.ndarray, n_layers: int):
    from geoseg.modules.segment_engines.ensemble import segment
    result = segment(image, n_layers=n_layers)
    return result["labels"]


ENGINES = {
    "v4_kmeans": run_v4_kmeans,
    "slic_kmeans": run_slic_kmeans,
    "edge_guided": run_edge_guided,
    "grayscale": run_grayscale,
    "ensemble": run_ensemble,
}


def main():
    summary = []
    for idx in [1, 2, 3]:
        stem = f"panel_{idx}"
        print(f"\n[{stem}] Loading text-removed image...")
        image = load_text_removed(idx)

        for name, engine_fn in ENGINES.items():
            for n_layers in [4, 6, 8]:
                key = f"{stem}_{name}_nl{n_layers}"
                print(f"  Running {key}...")
                try:
                    labels = engine_fn(image, n_layers)
                    metrics = evaluate_segmentation(image, labels)
                    print(f"    n_labels={metrics['n_labels']}, frag={metrics['fragment_ratio']:.2f}, purity={metrics['color_purity']:.1f}")

                    # Save visualization
                    fill = render_label_fill(labels)
                    boundaries = draw_boundaries(fill, labels)
                    cv2.imwrite(str(OUT / f"{key}_fill.png"), cv2.cvtColor(fill, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(OUT / f"{key}_boundaries.png"), cv2.cvtColor(boundaries, cv2.COLOR_RGB2BGR))

                    summary.append({
                        "panel": idx,
                        "engine": name,
                        "n_layers": n_layers,
                        **metrics,
                    })
                except Exception as e:
                    print(f"    FAILED: {e}")
                    summary.append({
                        "panel": idx,
                        "engine": name,
                        "n_layers": n_layers,
                        "error": str(e),
                    })

    # Save summary CSV
    import csv
    with open(OUT / "summary.csv", "w", newline="") as f:
        if summary:
            writer = csv.DictWriter(f, fieldnames=summary[0].keys())
            writer.writeheader()
            writer.writerows(summary)

    print(f"\nGroup A complete. Results in {OUT}")


if __name__ == "__main__":
    main()
