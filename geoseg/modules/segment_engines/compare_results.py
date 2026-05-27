"""Generate visual comparison grids for segmentation pipeline results.

Uses full_pipeline (classify + panel_detect + segment) instead of individual engines.

Usage:
    python -m geoseg.modules.segment_engines.compare_results \
        --summary runs/literature_test/gras2019/segment_results/summary.json \
        --images_dir runs/literature_test/gras2019/mineru/extracted/images \
        --output_dir runs/literature_test/gras2019/comparisons \
        --top_n 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from geoseg.modules.segment_engines.full_pipeline import process_figure
from geoseg.modules.segment_engines._shared import saturation_ratio


def _make_grid(original: np.ndarray, pipeline_result: dict, title: str) -> Image.Image:
    """Create a comparison grid: original + pipeline overlays + metadata."""
    panels = pipeline_result.get("panels", [])
    cls = pipeline_result.get("classification", {})

    images = [Image.fromarray(original)]
    labels = ["original"]

    # Add each panel's overlay
    for p in panels:
        seg = p["segmentation"]
        if seg is None:
            # Create a placeholder for skipped panels
            bbox = p["bbox"]
            placeholder = Image.new("RGB", (bbox[2], bbox[3]), (220, 220, 220))
            images.append(placeholder)
            labels.append(f"panel_{p['panel_id']}: skipped")
        else:
            images.append(Image.fromarray(seg["overlay"]))
            engine = seg["meta"]["engine"]
            path = seg["meta"].get("path", "")
            layers = len(np.unique(seg["labels"]))
            labels.append(f"p{p['panel_id']}: {engine}\n{path}, {layers}L")

    n = len(images)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    # Resize all to same size
    target_w, target_h = images[0].size
    resized = []
    for img in images:
        if img.size != (target_w, target_h):
            img = img.resize((target_w, target_h), Image.LANCZOS)
        resized.append(img)

    cell_w, cell_h = target_w, target_h + 50  # extra for label + metadata
    grid_w = cols * cell_w
    grid_h = rows * cell_h + 60  # top metadata bar

    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except Exception:
        font = ImageFont.load_default()
        font_small = font

    # Draw metadata bar at top
    figure_type = cls.get("figure_type", "unknown")
    confidence = cls.get("confidence", 0.0)
    meta_text = f"{title[:30]} | type={figure_type} | conf={confidence:.2f} | panels={len(panels)}"
    draw.text((5, 5), meta_text, fill=(0, 0, 0), font=font)

    for i, (img, label) in enumerate(zip(resized, labels)):
        col = i % cols
        row = i // cols
        x = col * cell_w
        y = row * cell_h + 60
        grid.paste(img, (x, y + 25))
        # Draw label with word wrap
        draw.text((x + 5, y), label, fill=(0, 0, 0), font=font_small)

    return grid


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare segmentation pipeline results")
    parser.add_argument("--summary", required=True, help="Path to summary.json")
    parser.add_argument("--images_dir", required=True, help="Directory with source images")
    parser.add_argument("--output_dir", required=True, help="Directory to save comparison grids")
    parser.add_argument("--top_n", type=int, default=5, help="Number of top saturation images to compare")
    parser.add_argument("--n_layers", type=int, default=5)
    parser.add_argument("--max_size", type=int, default=1000)
    parser.add_argument("--no_vlm", action="store_true", default=False,
                        help="Skip VLM calls, use colorbar fallback only")
    args = parser.parse_args()

    summary = json.loads(Path(args.summary).read_text())
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort by saturation descending
    top = sorted(summary.items(), key=lambda x: x[1].get("saturation", 0), reverse=True)[:args.top_n]

    for name, data in top:
        if data.get("status") == "skipped":
            print(f"\nSkipping {name} (sat={data.get('saturation', 0):.3f}) — {data.get('reason', '')}")
            continue

        print(f"\nProcessing {name} (sat={data.get('saturation', 0):.3f}) ...")
        img_path = images_dir / name
        if not img_path.exists():
            print(f"  Image not found: {img_path}")
            continue

        img = Image.open(img_path).convert("RGB")
        if max(img.size) > args.max_size:
            ratio = args.max_size / max(img.size)
            img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
        arr = np.array(img)

        # Run full pipeline
        result = process_figure(
            arr,
            n_layers=args.n_layers,
            skip_non_velocity_model=False,
            use_vlm=not args.no_vlm,
        )
        grid = _make_grid(arr, result, name)

        out_path = output_dir / f"{name[:20]}_pipeline.jpg"
        grid.save(out_path, quality=90)
        print(f"  Saved: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
