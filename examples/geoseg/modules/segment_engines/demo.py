"""Demo script for all segmentation engines.

Usage:
    python -m geoseg.modules.segment_engines.demo <image_path> [--n_layers N] [--output_dir DIR]

If no image is provided, generates a synthetic test panel.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from geoseg.modules.segment_engines import e027_slic_graphcut
from geoseg.modules.segment_engines import v4_kmeans
from geoseg.modules.segment_engines import kmeans_full
from geoseg.modules.segment_engines import edge_guided
from geoseg.modules.segment_engines import edge_grow
from geoseg.modules.segment_engines import ensemble
from geoseg.modules.segment_engines import grayscale
from geoseg.modules.segment_engines.router import route_and_segment
from geoseg.modules.segment_engines.full_pipeline import process_figure


def _synthetic_panel(h: int = 400, w: int = 600, n_layers: int = 5) -> np.ndarray:
    """Generate a synthetic jet-colormap panel for testing."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    band_h = h // n_layers
    colors = [
        [0, 0, 255],     # blue
        [0, 255, 255],   # cyan
        [0, 255, 0],     # green
        [255, 255, 0],   # yellow
        [255, 0, 0],     # red
        [128, 0, 0],     # dark red
        [255, 0, 255],   # magenta
    ]
    for i in range(n_layers):
        y0 = i * band_h
        y1 = (i + 1) * band_h if i < n_layers - 1 else h
        img[y0:y1] = colors[i % len(colors)]
    # Add slight noise
    noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def _make_dummy_reps(panel_rgb: np.ndarray, n_layers: int) -> list[dict]:
    """Create dummy VLM reps evenly spaced vertically."""
    h, w = panel_rgb.shape[:2]
    reps = []
    for i in range(n_layers):
        y = int(h * (i + 0.5) / n_layers)
        x = w // 2
        reps.append({
            "color_name": f"layer_{i+1}",
            "representative_point": {"x": x, "y": y},
        })
    return reps


def _save_result(result: dict, name: str, output_dir: Path) -> None:
    """Save overlay and labels from a segmentation result."""
    out_dir = output_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)

    overlay = Image.fromarray(result["overlay"])
    overlay.save(out_dir / "overlay.png")

    labels = result["labels"]
    # Save labels as grayscale normalized for viewing
    unique = np.unique(labels)
    if len(unique) > 1:
        labels_norm = (labels.astype(np.float32) / labels.max() * 255).astype(np.uint8)
    else:
        labels_norm = labels.astype(np.uint8)
    Image.fromarray(labels_norm).save(out_dir / "labels.png")

    meta = result.get("meta", {})
    print(f"  {name}: layers={meta.get('layers_found', len(unique))}, "
          f"engine={meta.get('engine', 'unknown')}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Segmentation engines demo")
    parser.add_argument("image", nargs="?", help="Path to test image (PNG/JPG)")
    parser.add_argument("--n_layers", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="runs/segment_engines_demo")
    parser.add_argument("--reps", action="store_true", help="Use dummy VLM reps for rep-based engines")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.image:
        img = np.array(Image.open(args.image).convert("RGB"))
        print(f"Loaded: {args.image} ({img.shape[1]}x{img.shape[0]})")
    else:
        img = _synthetic_panel(n_layers=args.n_layers)
        print(f"Generated synthetic panel ({img.shape[1]}x{img.shape[0]})")

    n_layers = args.n_layers
    reps = _make_dummy_reps(img, n_layers) if args.reps else None

    # --- Engines that don't need reps ---
    print("\nRunning no-rep engines...")

    result = e027_slic_graphcut.segment(img, n_layers=n_layers)
    _save_result(result, "e027_slic_graphcut", output_dir)

    result = v4_kmeans.segment_pastel_faded(img, colorbar_rgb=None, n_layers=n_layers)
    _save_result(result, "v4_kmeans_pastel", output_dir)

    result = grayscale.segment(img, n_layers=n_layers)
    _save_result(result, "grayscale", output_dir)

    if reps:
        print("\nRunning rep-based engines...")

        result = v4_kmeans.segment_jet_vivid(img, reps, max_auto_k=2)
        _save_result(result, "v4_kmeans_jet_vivid", output_dir)

        result = kmeans_full.segment(img, reps, n_layers=n_layers)
        _save_result(result, "kmeans_full", output_dir)

        result = edge_guided.segment(img, reps, n_layers=n_layers)
        _save_result(result, "edge_guided", output_dir)

        result = edge_grow.segment(img, reps, n_layers=n_layers)
        _save_result(result, "edge_grow", output_dir)

        result = ensemble.segment(img, reps, n_layers=n_layers)
        _save_result(result, "ensemble", output_dir)

    # --- Router ---
    print("\nRunning router...")
    result = route_and_segment(
        img,
        reps=reps,
        n_layers=n_layers,
        quality_preference="balanced",
    )
    _save_result(result, "router_balanced", output_dir)

    # --- Full pipeline ---
    print("\nRunning full pipeline (classify + detect panels + segment)...")
    pipeline_result = process_figure(img, n_layers=n_layers, skip_non_velocity_model=False)
    print(f"  classification: {pipeline_result['classification']['figure_type']}")
    print(f"  panels: {pipeline_result['summary']['n_panels']}")
    for p in pipeline_result['panels']:
        seg = p['segmentation']
        if seg:
            print(f"    panel {p['panel_id']}: engine={seg['meta']['engine']}, "
                  f"layers={len(np.unique(seg['labels']))}")
    pipeline_dir = output_dir / "full_pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    for p in pipeline_result['panels']:
        seg = p['segmentation']
        if seg:
            Image.fromarray(seg['overlay']).save(
                pipeline_dir / f"panel_{p['panel_id']}_overlay.png"
            )

    print(f"\nResults saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
