#!/usr/bin/env python3
"""Export segmentation results to txt label + palette files and reconstructed images.

Writes the same format consumed by scripts/reconstruct_from_txt.py:
  - {profile}_labels.txt  : rows of "x y label_id"
  - {profile}_palette.txt : rows of "label_id r g b"

Also creates:
  - {profile}_reconstructed.jpg : palette-colored reconstruction
  - {profile}_comparison.jpg    : original | reconstructed side-by-side
  - combined_reconstructed.jpg
  - combined_comparison.jpg
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
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


def _zip_output_dir(output_dir: Path) -> Path:
    """Create a zip archive containing the entire output directory."""
    zip_path = output_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(output_dir.rglob("*")):
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.relative_to(output_dir))
    return zip_path


def _build_palette(labels: np.ndarray, img_rgb: np.ndarray) -> dict[int, tuple[int, int, int]]:
    """Map every label (including 0) to its median RGB in the original image."""
    palette: dict[int, tuple[int, int, int]] = {}
    for lbl in sorted(set(labels.flatten())):
        mask = labels == lbl
        if mask.any():
            color = tuple(int(c) for c in np.median(img_rgb[mask], axis=0))
        else:
            color = (0, 0, 0)
        palette[int(lbl)] = color
    return palette


def _reconstruct_image(labels: np.ndarray, palette: dict[int, tuple[int, int, int]]) -> np.ndarray:
    rec = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for lbl, (r, g, b) in palette.items():
        rec[labels == lbl] = [r, g, b]
    return rec


def _write_labels_txt(labels: np.ndarray, out_path: Path) -> None:
    """Write x y label_id table. Vectorized for speed."""
    ys, xs = np.indices(labels.shape, dtype=np.int32)
    flat = np.column_stack([xs.ravel(), ys.ravel(), labels.ravel()])
    out_path.write_text("x y label_id\n" + "\n".join(f"{x} {y} {l}" for x, y, l in flat), encoding="utf-8")


def _write_palette_txt(palette: dict[int, tuple[int, int, int]], out_path: Path) -> None:
    lines = ["label_id r g b"] + [f"{lbl} {r} {g} {b}" for lbl, (r, g, b) in sorted(palette.items())]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _save_comparison(original: np.ndarray, reconstructed: np.ndarray, out_path: Path) -> None:
    h, w = original.shape[:2]
    canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)
    canvas[:, :w] = original
    canvas[:, w:] = reconstructed
    Image.fromarray(canvas).save(out_path, quality=95)


def _assemble_grid(images: Sequence[np.ndarray], titles: Sequence[str]) -> Image.Image:
    if not images:
        raise ValueError("No images to assemble")
    n = len(images)
    header_h = 40
    gap = 6
    font = _load_font(16)
    max_width = max(img.shape[1] for img in images)
    total_height = sum(img.shape[0] for img in images) + n * header_h + (n - 1) * gap

    canvas = Image.new("RGB", (max_width, total_height), (32, 32, 32))
    draw = ImageDraw.Draw(canvas)

    y = 0
    for img, title in zip(images, titles):
        h, w = img.shape[:2]
        x = (max_width - w) // 2
        canvas.paste(Image.fromarray(img), (x, y + header_h))
        draw.text(
            (max_width // 2, y + header_h // 2),
            title,
            fill=(220, 220, 220),
            font=font,
            anchor="mm",
        )
        y += header_h + h + gap
    return canvas


def _resolve_labels_path(panel_dir: Path, label_version: str) -> Path | None:
    candidates = {
        "best_v3": panel_dir / "visual_audit" / "labels_best_split_v3.npz",
        "best_v2": panel_dir / "visual_audit" / "labels_best_split_v2.npz",
        "best": panel_dir / "visual_audit" / "labels_best_split.npz",
        "raw": panel_dir / "labels.npz",
    }
    if label_version in candidates:
        return candidates[label_version]
    # Auto-detect: prefer best_v3, then best_v2, then best, then raw.
    for key in ("best_v3", "best_v2", "best", "raw"):
        p = candidates[key]
        if p.exists():
            return p
    return None


def _load_original(run_dir: Path, panel_id: int, summary: dict | None) -> np.ndarray:
    """Return raw original crop for a panel. Falls back to cleaned panel.png."""
    if summary is not None:
        for entry in summary.get("per_panel", []):
            if entry["panel_id"] == panel_id:
                x, y, w, h = entry["bbox"]
                origin_path = run_dir / "01_original.jpg"
                if origin_path.exists():
                    img = Image.open(origin_path).convert("RGB")
                    return np.array(img.crop((x, y, x + w, y + h)))
                break
    panel_png = run_dir / "panels" / f"panel_{panel_id}" / "visual_audit" / "panel.png"
    if panel_png.exists():
        return np.array(Image.open(panel_png).convert("RGB"))
    raise FileNotFoundError(f"Cannot find original image for panel {panel_id}")


def export_run(
    run_dir: Path,
    output_dir: Path,
    profiles: Sequence[str] | None,
    label_version: str,
) -> int:
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None

    output_dir.mkdir(parents=True, exist_ok=True)

    panels_dir = run_dir / "panels"
    panel_dirs = sorted(panels_dir.glob("panel_*"))
    if not panel_dirs:
        print(f"No panel directories found in {panels_dir}", file=sys.stderr)
        return 1

    if profiles is None:
        profiles = [f"{run_dir.name}_panel_{i:02d}" for i in range(len(panel_dirs))]
    if len(profiles) != len(panel_dirs):
        print("--profiles count must match panel count", file=sys.stderr)
        return 1

    reconstructed_images: list[np.ndarray] = []
    comparison_images: list[np.ndarray] = []
    titles: list[str] = []

    for panel_dir, profile in zip(panel_dirs, profiles):
        panel_id = int(panel_dir.name.split("_")[-1])
        labels_path = _resolve_labels_path(panel_dir, label_version)
        if labels_path is None:
            print(f"Skipping panel {panel_id}: no labels file found", file=sys.stderr)
            continue

        labels = np.load(labels_path)["labels"]
        original = _load_original(run_dir, panel_id, summary)
        if original.shape[:2] != labels.shape[:2]:
            print(
                f"Shape mismatch in panel {panel_id}: image {original.shape[:2]} vs labels {labels.shape[:2]}",
                file=sys.stderr,
            )
            continue

        palette = _build_palette(labels, original)
        reconstructed = _reconstruct_image(labels, palette)
        comparison = np.concatenate([original, reconstructed], axis=1)

        _write_labels_txt(labels, output_dir / f"{profile}_labels.txt")
        _write_palette_txt(palette, output_dir / f"{profile}_palette.txt")
        Image.fromarray(reconstructed).save(output_dir / f"{profile}_reconstructed.jpg", quality=95)
        Image.fromarray(comparison).save(output_dir / f"{profile}_comparison.jpg", quality=95)
        print(f"Exported {profile}")

        reconstructed_images.append(reconstructed)
        comparison_images.append(comparison)
        titles.append(profile)

    if len(reconstructed_images) > 1:
        _assemble_grid(reconstructed_images, titles).save(
            output_dir / "combined_reconstructed.jpg", quality=95
        )
        _assemble_grid(comparison_images, titles).save(
            output_dir / "combined_comparison.jpg", quality=95
        )
        print(f"Saved combined grids in {output_dir}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Export segmentation to txt + reconstructed images.")
    parser.add_argument("run_dir", type=Path, help="Run directory (e.g. runs/preprocess_newimage_merged)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output dir (default: run_dir/txt_export)")
    parser.add_argument("--profiles", nargs="+", default=None, help="Profile names, one per panel")
    parser.add_argument(
        "--label-version",
        choices=["auto", "best_v3", "best_v2", "best", "raw"],
        default="auto",
        help="Which labels to export (default: auto-detect best_v3 > best_v2 > best > raw)",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Also create a zip archive of the output directory",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (args.run_dir / "txt_export")
    rc = export_run(args.run_dir, output_dir, args.profiles, args.label_version)
    if rc != 0:
        return rc
    if args.zip:
        zip_path = _zip_output_dir(output_dir)
        print(f"Created zip archive: {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
