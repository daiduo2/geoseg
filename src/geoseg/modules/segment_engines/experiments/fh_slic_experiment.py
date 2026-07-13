"""Experiment 1a/1b: Felzenszwalb-Huttenlocher and SLIC segmentation on
text-annotated geophysics panels.

Evaluates whether graph-based / superpixel methods are more robust than
pixel-level K-means when text overlays are present.

Usage:
    python -m geoseg.modules.segment_engines.experiments.fh_slic_experiment

Output:
    runs/experiments/fh_slic/   -- overlays, labels, and report.md
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geoseg.modules.segment_engines import v4_kmeans
from geoseg.modules.segment_engines import metrics as seg_metrics
from geoseg.modules.segment_engines._shared import _create_overlay

from geoseg.modules.segment_engines.experiments.fh_slic_engines import (
    run_fh,
    run_slic_clustering,
    run_v4_baseline,
)
from geoseg.modules.segment_engines.experiments.fh_slic_images import (
    _load_real_image,
    _synthetic_panel_with_text,
)
from geoseg.modules.segment_engines.experiments.fh_slic_report import (
    generate_report,
    _fmt_table,
)


def evaluate(result: dict, panel_rgb: np.ndarray, expected_n_layers: int = 5) -> dict:
    """Compute metrics for a segmentation result."""
    labels = result["labels"]
    t0 = time.perf_counter()
    all_metrics = seg_metrics.compute_all(labels, panel_rgb)
    elapsed = time.perf_counter() - t0

    n = all_metrics["n_layers"]
    return {
        "n_layers": n,
        "n_layers_diff": abs(n - expected_n_layers),
        "boundary_alignment": all_metrics["boundary_alignment"],
        "n_fragments": len(all_metrics["tiny_fragments"]),
        "total_fragment_area": all_metrics["total_fragment_area_fraction"],
        "has_noise_warnings": all_metrics["has_noise_warnings"],
        "noise_suspect_count": all_metrics["noise_warnings"]["suspect_count"],
        "metrics_time_ms": round(elapsed * 1000, 2),
    }


def sweep_fh(panel_rgb: np.ndarray, output_dir: Path, expected_n_layers: int = 5) -> list[dict]:
    """Systematic FH parameter sweep."""
    scales = [1, 10, 100, 500]
    sigmas = [0.5, 1.0]
    min_sizes = [10, 50, 100, 500]

    results = []
    for scale in scales:
        for sigma in sigmas:
            for min_size in min_sizes:
                t0 = time.perf_counter()
                seg_result = run_fh(panel_rgb, scale=scale, sigma=sigma, min_size=min_size)
                elapsed = time.perf_counter() - t0

                ev = evaluate(seg_result, panel_rgb, expected_n_layers)
                ev.update({
                    "engine": "felzenszwalb",
                    "scale": scale,
                    "sigma": sigma,
                    "min_size": min_size,
                    "runtime_ms": round(elapsed * 1000, 2),
                })
                results.append(ev)

                name = f"fh_s{scale}_sig{sigma}_ms{min_size}"
                Image.fromarray(seg_result["overlay"]).save(output_dir / f"{name}.png")
    return results


def sweep_slic(panel_rgb: np.ndarray, output_dir: Path, expected_n_layers: int = 5) -> list[dict]:
    """Systematic SLIC + K-means parameter sweep."""
    n_segments_list = [100, 500, 1000]
    compactness_list = [0.01, 0.1, 1, 10]

    results = []
    for n_segments in n_segments_list:
        for compactness in compactness_list:
            t0 = time.perf_counter()
            seg_result = run_slic_clustering(
                panel_rgb,
                n_segments=n_segments,
                compactness=compactness,
                n_clusters=expected_n_layers,
            )
            elapsed = time.perf_counter() - t0

            ev = evaluate(seg_result, panel_rgb, expected_n_layers)
            ev.update({
                "engine": "slic_kmeans",
                "n_segments": n_segments,
                "compactness": compactness,
                "runtime_ms": round(elapsed * 1000, 2),
            })
            results.append(ev)

            name = f"slic_n{n_segments}_c{compactness}"
            Image.fromarray(seg_result["overlay"]).save(output_dir / f"{name}.png")
    return results


def run_baseline(panel_rgb: np.ndarray, output_dir: Path, expected_n_layers: int = 5) -> dict:
    """Run and evaluate v4_kmeans baseline."""
    t0 = time.perf_counter()
    seg_result = run_v4_baseline(panel_rgb, n_layers=expected_n_layers)
    elapsed = time.perf_counter() - t0

    ev = evaluate(seg_result, panel_rgb, expected_n_layers)
    ev.update({"engine": "v4_kmeans", "runtime_ms": round(elapsed * 1000, 2)})

    Image.fromarray(seg_result["overlay"]).save(output_dir / "baseline_v4_kmeans.png")
    return ev


def main() -> int:
    output_dir = PROJECT_ROOT / "runs" / "experiments" / "fh_slic"
    output_dir.mkdir(parents=True, exist_ok=True)

    real_img = _load_real_image(PROJECT_ROOT)
    if real_img is not None:
        h, w = real_img.shape[:2]
        if max(h, w) > 800:
            scale = 800 / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            real_img = np.array(Image.fromarray(real_img).resize((new_w, new_h), Image.LANCZOS))
        panel_rgb = real_img
        image_desc = f"Real image: ph01_page8_300dpi.png resized to {panel_rgb.shape[1]}x{panel_rgb.shape[0]}"
    else:
        panel_rgb = _synthetic_panel_with_text(n_layers=5, text_density="medium")
        image_desc = f"Synthetic panel: {panel_rgb.shape[1]}x{panel_rgb.shape[0]}, 5 layers, medium text density"

    Image.fromarray(panel_rgb).save(output_dir / "input.png")

    print("Running baseline v4_kmeans...")
    baseline = run_baseline(panel_rgb, output_dir, expected_n_layers=5)
    print(f"  Baseline: {baseline}")

    print("Running FH sweep (32 combos)...")
    fh_results = sweep_fh(panel_rgb, output_dir, expected_n_layers=5)
    print(f"  FH best BA: {max(fh_results, key=lambda r: r['boundary_alignment'])}")

    print("Running SLIC sweep (12 combos)...")
    slic_results = sweep_slic(panel_rgb, output_dir, expected_n_layers=5)
    print(f"  SLIC best BA: {max(slic_results, key=lambda r: r['boundary_alignment'])}")

    report = generate_report(fh_results, slic_results, baseline, output_dir, image_desc)
    report_path = output_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")

    json_path = output_dir / "results.json"
    json_path.write_text(
        json.dumps(
            {
                "image_desc": image_desc,
                "baseline": baseline,
                "fh_results": fh_results,
                "slic_results": slic_results,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(f"\nResults saved to: {output_dir}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
