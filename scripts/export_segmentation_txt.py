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
    """Write x y label_id table in bounded-memory row chunks."""
    height, width = labels.shape
    xs = np.arange(width, dtype=np.int32)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("x y label_id\n")
        for y0 in range(0, height, 64):
            y1 = min(y0 + 64, height)
            ys = np.repeat(np.arange(y0, y1, dtype=np.int32), width)
            chunk_xs = np.tile(xs, y1 - y0)
            flat = np.column_stack(
                [chunk_xs, ys, labels[y0:y1].reshape(-1)]
            )
            np.savetxt(handle, flat, fmt="%d %d %d")


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


def _normalize_direct_colorbar_labels(
    labels: np.ndarray,
    palette_rgb: np.ndarray,
) -> tuple[np.ndarray, dict[int, tuple[int, int, int]], dict[int, int]]:
    """Map -1 background to 0 and direct color classes to positive IDs."""
    if labels.ndim != 2:
        raise ValueError("labels must be a 2D array")
    if palette_rgb.ndim != 2 or palette_rgb.shape[1] != 3:
        raise ValueError("palette_rgb must have shape (N, 3)")
    foreground = labels >= 0
    if foreground.any() and int(labels[foreground].max()) >= len(palette_rgb):
        raise ValueError("labels reference colors outside palette_rgb")

    normalized = np.zeros(labels.shape, dtype=np.int32)
    normalized[foreground] = labels[foreground].astype(np.int32) + 1
    palette: dict[int, tuple[int, int, int]] = {0: (255, 255, 255)}
    palette.update(
        {
            index + 1: tuple(int(channel) for channel in rgb)
            for index, rgb in enumerate(palette_rgb)
        }
    )
    source_to_export = {-1: 0}
    source_to_export.update({index: index + 1 for index in range(len(palette_rgb))})
    return normalized, palette, source_to_export


def _export_direct_colorbar_run(
    run_dir: Path,
    output_dir: Path,
    profiles: Sequence[str] | None,
) -> int:
    """Export named figures produced by the direct-colorbar workflow."""
    figure_dirs = sorted(
        path
        for path in run_dir.iterdir()
        if path.is_dir()
        and (path / "annotation_cleanup" / "labels.npz").exists()
        and (path / "01_panel.png").exists()
    )
    if not figure_dirs:
        print(f"No exportable figures found in {run_dir}", file=sys.stderr)
        return 1
    if profiles is None:
        profiles = [path.name for path in figure_dirs]
    if len(profiles) != len(figure_dirs):
        print("--profiles count must match figure count", file=sys.stderr)
        return 1

    reconstructed_images: list[np.ndarray] = []
    comparison_images: list[np.ndarray] = []
    titles: list[str] = []
    manifest_profiles: list[dict[str, object]] = []

    for figure_dir, profile in zip(figure_dirs, profiles):
        labels_path = figure_dir / "annotation_cleanup" / "labels.npz"
        original_path = figure_dir / "01_panel.png"
        with np.load(labels_path) as data:
            if "palette_rgb" not in data.files:
                print(
                    f"Skipping {figure_dir.name}: palette_rgb is missing",
                    file=sys.stderr,
                )
                continue
            source_labels = data["labels"]
            palette_rgb = data["palette_rgb"]
        labels, palette, source_to_export = _normalize_direct_colorbar_labels(
            source_labels, palette_rgb
        )
        original = np.asarray(Image.open(original_path).convert("RGB"))
        if original.shape[:2] != labels.shape:
            print(
                f"Shape mismatch in {figure_dir.name}: image "
                f"{original.shape[:2]} vs labels {labels.shape}",
                file=sys.stderr,
            )
            continue

        reconstructed = _reconstruct_image(labels, palette)
        comparison = np.concatenate([original, reconstructed], axis=1)
        _write_labels_txt(labels, output_dir / f"{profile}_labels.txt")
        _write_palette_txt(palette, output_dir / f"{profile}_palette.txt")
        Image.fromarray(reconstructed).save(
            output_dir / f"{profile}_reconstructed.png"
        )
        Image.fromarray(comparison).save(output_dir / f"{profile}_comparison.png")
        Image.fromarray(reconstructed).save(
            output_dir / f"{profile}_reconstructed.jpg", quality=95
        )
        Image.fromarray(comparison).save(
            output_dir / f"{profile}_comparison.jpg", quality=95
        )
        print(f"Exported {profile}")

        reconstructed_images.append(reconstructed)
        comparison_images.append(comparison)
        titles.append(profile)
        manifest_profiles.append(
            {
                "profile": profile,
                "source_figure": figure_dir.name,
                "source_labels": str(labels_path.relative_to(run_dir)),
                "source_original": str(original_path.relative_to(run_dir)),
                "shape_yx": list(labels.shape),
                "background_export_label": 0,
                "source_to_export_label": {
                    str(source): exported
                    for source, exported in source_to_export.items()
                },
                "palette_source": "labels.npz:palette_rgb_exact_reviewed_colors",
            }
        )

    if not reconstructed_images:
        return 1
    if len(reconstructed_images) > 1:
        _assemble_grid(reconstructed_images, titles).save(
            output_dir / "combined_reconstructed.jpg", quality=95
        )
        _assemble_grid(comparison_images, titles).save(
            output_dir / "combined_comparison.jpg", quality=95
        )
    manifest = {
        "format": "geoseg_txt_export_v1",
        "coordinate_order": "x y label_id",
        "coordinate_origin": "top_left",
        "x_direction": "right",
        "y_direction": "down",
        "profiles": manifest_profiles,
    }
    (output_dir / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return 0


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
        return _export_direct_colorbar_run(run_dir, output_dir, profiles)

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
