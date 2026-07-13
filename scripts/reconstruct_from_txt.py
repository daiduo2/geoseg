#!/usr/bin/env python3
"""Reconstruct segmentation masks from exported txt label + palette files.

Reads:
  - {profile}_labels.txt : rows of "x y label_id"
  - {profile}_palette.txt: rows of "label_id r g b"

Writes:
  - Individual reconstructed masks as {profile}_reconstructed.jpg
  - A combined grid of all profiles as combined_reconstructed.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int):
    for p in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Arial.ttf",
    ]:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def load_labels_and_shape(labels_path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    """Load x,y,label_id data and infer image shape from max x,y."""
    data = np.loadtxt(labels_path, skiprows=1, dtype=np.int32)
    xs, ys, labels = data[:, 0], data[:, 1], data[:, 2]
    height = int(ys.max()) + 1
    width = int(xs.max()) + 1
    label_map = np.zeros((height, width), dtype=np.int32)
    label_map[ys, xs] = labels
    return label_map, (height, width)


def load_palette(palette_path: Path) -> dict[int, tuple[int, int, int]]:
    """Load label_id -> (r,g,b) mapping."""
    data = np.loadtxt(palette_path, skiprows=1, dtype=np.int32)
    palette: dict[int, tuple[int, int, int]] = {}
    for row in data:
        label_id, r, g, b = row
        palette[int(label_id)] = (int(r), int(g), int(b))
    return palette


def reconstruct_mask(label_map: np.ndarray, palette: dict[int, tuple[int, int, int]]) -> np.ndarray:
    """Render RGB mask from label map and palette."""
    height, width = label_map.shape
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    for label_id, (r, g, b) in palette.items():
        mask[label_map == label_id] = [r, g, b]
    return mask


def assemble_grid(
    images: list[np.ndarray], titles: list[str], orientation: str = "vertical"
) -> Image.Image:
    """Assemble images into a single grid with titles.

    Args:
        images: RGB masks to assemble.
        titles: Title string for each image.
        orientation: "horizontal" or "vertical".
    """
    if not images:
        raise ValueError("No images to assemble")

    n = len(images)
    height, width = images[0].shape[:2]
    header_h = 40
    gap = 6
    font = _load_font(16)

    if orientation == "horizontal":
        total_width = n * width + (n - 1) * gap
        total_height = height + header_h
    else:
        total_width = width
        total_height = n * (height + header_h) + (n - 1) * gap

    canvas = Image.new("RGB", (total_width, total_height), (32, 32, 32))
    draw = ImageDraw.Draw(canvas)

    for i, (img, title) in enumerate(zip(images, titles)):
        if orientation == "horizontal":
            x = i * (width + gap)
            y = header_h
            title_x = x + (width // 2)
            title_y = header_h // 2
        else:
            x = 0
            y = i * (height + header_h + gap)
            title_x = width // 2
            title_y = y + header_h // 2

        canvas.paste(Image.fromarray(img), (x, y))
        draw.text(
            (title_x, title_y),
            title,
            fill=(220, 220, 220),
            font=font,
            anchor="mm",
        )

    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconstruct masks from txt label + palette files.")
    parser.add_argument("input_dir", type=Path, help="Directory containing {profile}_labels.txt and {profile}_palette.txt files")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: input_dir)")
    parser.add_argument("--orientation", choices=["horizontal", "vertical"], default="vertical", help="Combined grid orientation (default: vertical)")
    parser.add_argument("--profiles", nargs="+", default=None, help="Profiles to process (default: fig6_profile_03..07)")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.profiles:
        profiles = args.profiles
    else:
        profiles = [f"fig6_profile_{i:02d}" for i in range(3, 8)]

    images: list[np.ndarray] = []
    titles: list[str] = []

    for profile in profiles:
        labels_path = input_dir / f"{profile}_labels.txt"
        palette_path = input_dir / f"{profile}_palette.txt"

        if not labels_path.exists():
            print(f"Skipping {profile}: {labels_path} not found", file=sys.stderr)
            continue
        if not palette_path.exists():
            print(f"Skipping {profile}: {palette_path} not found", file=sys.stderr)
            continue

        label_map, _ = load_labels_and_shape(labels_path)
        palette = load_palette(palette_path)
        mask = reconstruct_mask(label_map, palette)

        out_path = output_dir / f"{profile}_reconstructed.jpg"
        Image.fromarray(mask).save(out_path, quality=95)
        print(f"Saved: {out_path}")

        images.append(mask)
        titles.append(profile)

    if len(images) > 1:
        grid = assemble_grid(images, titles, orientation=args.orientation)
        grid_path = output_dir / "combined_reconstructed.jpg"
        grid.save(grid_path, quality=95)
        print(f"Saved combined grid: {grid_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
