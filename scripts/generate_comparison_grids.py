"""Generate before/after comparison grids from batch test results.

Reads saved overlays from batch_test output directories.

Usage:
    python3 scripts/generate_comparison_grids.py \
        --dataset gras2019 \
        --output_dir runs/literature_test/gras2019/comparison_grids
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _make_grid(
    img_path: Path,
    panels: list[dict],
    results_dir: Path,
    max_cols: int = 3,
    thumb_size: tuple[int, int] = (400, 400),
) -> Image.Image | None:
    """Create a comparison grid for a single figure."""
    if not panels:
        return None

    overlays: list[Image.Image] = []
    labels_list: list[str] = []

    for p in panels:
        if p.get("skipped"):
            continue
        panel_id = p["panel_id"]
        engine = p.get("engine", "unknown")
        layers = p.get("layers", "?")

        # Load saved overlay
        overlay_path = results_dir / f"{img_path.stem}_panel{panel_id}_overlay.jpg"
        if not overlay_path.exists():
            continue

        overlay = Image.open(overlay_path)
        overlay.thumbnail(thumb_size)
        overlays.append(overlay)
        labels_list.append(f"P{panel_id}: {engine}\n{layers} layers")

    if not overlays:
        return None

    n = len(overlays)
    cols = min(n, max_cols)
    rows = math.ceil(n / cols)

    cell_w, cell_h = thumb_size
    text_h = 40
    grid_w = cols * cell_w
    grid_h = rows * (cell_h + text_h) + 30  # extra for title

    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except Exception:
        font = ImageFont.load_default()
        font_title = font

    # Title
    draw.text((5, 5), img_path.name[:40], fill=(0, 0, 0), font=font_title)

    for i, (img, label) in enumerate(zip(overlays, labels_list)):
        row, col = divmod(i, cols)
        x = col * cell_w
        y = row * (cell_h + text_h) + 30

        grid.paste(img, (x, y))
        draw.text((x + 5, y + cell_h + 5), label, fill=(0, 0, 0), font=font)

    return grid


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate comparison grids from batch test results")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., gras2019)")
    parser.add_argument("--output_dir", required=True, help="Directory to save grids")
    parser.add_argument("--max_cols", type=int, default=3)
    parser.add_argument("--thumb_size", type=int, default=400)
    args = parser.parse_args()

    base_dir = Path(f"/Users/daiduo2/geoseg/runs/literature_test/{args.dataset}")
    summary_path = base_dir / "segment_results_vlm" / "summary.json"
    images_dir = base_dir / "mineru" / "extracted" / "images"
    results_dir = base_dir / "segment_results_vlm"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not summary_path.exists():
        print(f"Summary not found: {summary_path}")
        return 1

    summary = json.loads(summary_path.read_text())

    # Sort by saturation descending
    items = sorted(summary.items(), key=lambda x: x[1].get("saturation", 0), reverse=True)

    generated = 0
    for img_name, data in items:
        if data.get("status") != "ok":
            continue

        panels = data.get("panels", [])
        if not panels:
            continue

        img_path = images_dir / img_name
        if not img_path.exists():
            continue

        grid = _make_grid(
            img_path,
            panels,
            results_dir,
            max_cols=args.max_cols,
            thumb_size=(args.thumb_size, args.thumb_size),
        )
        if grid is None:
            continue

        out_path = output_dir / f"{img_name[:30]}_grid.jpg"
        grid.save(out_path, quality=85)
        generated += 1
        print(f"Saved: {out_path}")

    print(f"\nGenerated {generated} comparison grids in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
