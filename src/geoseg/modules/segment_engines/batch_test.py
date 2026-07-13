"""Batch test segmentation pipeline on literature figures.

Uses full_pipeline (classify + panel_detect + colorbar_extract + segment)
instead of calling individual engines directly.

Usage:
    python -m geoseg.modules.segment_engines.batch_test \
        --images_dir runs/literature_test/gras2019/mineru/extracted/images \
        --output_dir runs/literature_test/gras2019/segment_results \
        --n_layers 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from geoseg.modules.segment_engines.full_pipeline import process_figure
from geoseg.modules.segment_engines._shared import saturation_ratio


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch test segmentation pipeline on literature figures")
    parser.add_argument("--images_dir", required=True, help="Directory containing figure images")
    parser.add_argument("--output_dir", required=True, help="Directory to save results")
    parser.add_argument("--n_layers", type=int, default=5)
    parser.add_argument("--max_size", type=int, default=1200, help="Max dimension for downsampling")
    parser.add_argument("--skip_non_velocity", action="store_true", default=True,
                        help="Skip images classified as observational_data or other")
    parser.add_argument("--no_vlm", action="store_true", default=False,
                        help="Skip VLM calls, use colorbar fallback only")
    parser.add_argument("--quality", type=str, default="balanced",
                        choices=["fast", "balanced", "best"],
                        help="Segmentation quality preference")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    print(f"Found {len(image_files)} images in {images_dir}")

    summary = {}
    skipped_count = 0
    processed_count = 0

    for img_path in image_files:
        print(f"\nProcessing {img_path.name} ...")
        img = Image.open(img_path).convert("RGB")

        # Downsample if too large
        if max(img.size) > args.max_size:
            ratio = args.max_size / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
            print(f"  Resized to {img.size}")

        arr = np.array(img)
        sat = saturation_ratio(arr)
        print(f"  Size: {arr.shape[1]}x{arr.shape[0]}, Saturation: {sat:.3f}")

        # Run full pipeline
        t0 = time.perf_counter()
        result = process_figure(
            arr,
            n_layers=args.n_layers,
            quality_preference=args.quality,
            skip_non_velocity_model=args.skip_non_velocity,
            use_vlm=not args.no_vlm,
        )
        t1 = time.perf_counter()
        elapsed = t1 - t0

        cls = result["classification"]
        figure_type = cls["figure_type"]
        print(f"  Classification: {figure_type} (conf={cls['confidence']:.2f})")

        if result["summary"]["status"] == "skipped":
            skipped_count += 1
            print(f"  SKIPPED: {result['summary']['reason']}")
            summary[img_path.name] = {
                "size": f"{arr.shape[1]}x{arr.shape[0]}",
                "saturation": round(sat, 4),
                "figure_type": figure_type,
                "status": "skipped",
                "reason": result["summary"]["reason"],
            }
            continue

        processed_count += 1
        n_panels = result["summary"]["n_panels"]
        print(f"  Panels detected: {n_panels}, Time: {elapsed:.2f}s")

        # Collect per-panel results
        panel_results = []
        for p in result["panels"]:
            seg = p["segmentation"]
            if seg:
                layers = int(len(np.unique(seg["labels"])))
                engine = seg["meta"]["engine"]
                path = seg["meta"].get("path", "unknown")
                panel_results.append({
                    "panel_id": p["panel_id"],
                    "bbox": p["bbox"],
                    "engine": engine,
                    "path": path,
                    "layers": layers,
                })
                print(f"    Panel {p['panel_id']}: {engine}/{path}, {layers} layers")

                # Save segmentation outputs for comparison grids
                base_name = f"{img_path.stem}_panel{p['panel_id']}"
                overlay_path = output_dir / f"{base_name}_overlay.jpg"
                labels_path = output_dir / f"{base_name}_labels.npz"
                Image.fromarray(seg["overlay"]).save(overlay_path, quality=90)
                np.savez_compressed(labels_path, labels=seg["labels"])
            else:
                panel_results.append({
                    "panel_id": p["panel_id"],
                    "bbox": p["bbox"],
                    "skipped": True,
                })

        summary[img_path.name] = {
            "size": f"{arr.shape[1]}x{arr.shape[0]}",
            "saturation": round(sat, 4),
            "figure_type": figure_type,
            "status": "ok",
            "time_s": round(elapsed, 3),
            "n_panels": n_panels,
            "panels": panel_results,
        }

    # Save summary
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n{'='*50}")
    print(f"Total: {len(image_files)}, Processed: {processed_count}, Skipped: {skipped_count}")
    print(f"Summary saved to: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
