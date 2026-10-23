"""Three-group boundary comparison: source | mask | boundary.

The source is the original panel image with the detected data-domain bbox.
The mask is the recoloured label partition cropped to that bbox.  The boundary
group on the right contains pure boundary masks: raw label interfaces,
foreground border closure, and full-rectangle closure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from geoseg import api
from experiments.layer2.common import recolour_labels

LABEL_WIDTH = 190
CELL = (420, 280)

# The curated paper23_fig7a label is the reviewed 15-region result, which is
# over-segmented for a geometry demo.  Use the 9-region autonomous result from
# the same direct-result chain; keep the reviewed result available elsewhere.
PANEL_OVERRIDES: dict[str, dict] = {}



def _font(size: int = 14):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()


def _fit(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray(image).convert("RGB")
    scale = min(size[0] / pil.width, size[1] / pil.height)
    resized = pil.resize(
        (max(1, int(pil.width * scale)), max(1, int(pil.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", size, (18, 18, 18))
    canvas.paste(resized, ((size[0] - resized.width) // 2, (size[1] - resized.height) // 2))
    return np.asarray(canvas)


def _fit_mask(mask: np.ndarray, size: tuple[int, int], shape: tuple[int, int]) -> np.ndarray:
    scale = min(size[0] / shape[1], size[1] / shape[0])
    width = max(1, int(shape[1] * scale))
    height = max(1, int(shape[0] * scale))
    resized = np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (width, height), Image.Resampling.BOX
        )
    ) > 0
    canvas = np.zeros((size[1], size[0]), dtype=bool)
    left = (size[0] - width) // 2
    top = (size[1] - height) // 2
    canvas[top : top + height, left : left + width] = resized
    return canvas


def _display_boundary(mask: np.ndarray, size: tuple[int, int], shape: tuple[int, int], thickness: int = 3) -> np.ndarray:
    small = _fit_mask(mask, size, shape)
    small = ndimage.binary_dilation(small, iterations=thickness - 1)
    out = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    out[small] = (255, 255, 255)
    return out


def _bbox_overlay(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    out = image.copy()
    x0, y0, x1, y1 = bbox
    width = max(2, int(round(min(out.shape[:2]) * 0.004)))
    out[y0 : y0 + width, x0:x1] = (255, 220, 60)
    out[max(0, y1 - width) : y1, x0:x1] = (255, 220, 60)
    out[y0:y1, x0 : x0 + width] = (255, 220, 60)
    out[y0:y1, max(0, x1 - width) : x1] = (255, 220, 60)
    return out


def _group_headers(draw, y: int, cell: tuple[int, int], font) -> None:
    x0 = LABEL_WIDTH
    draw.text((x0 + 8, y), "source", fill=(240, 220, 80), font=font)
    draw.text((x0 + cell[0] + 8, y), "mask", fill=(240, 220, 80), font=font)
    draw.text((x0 + 2 * cell[0] + 8, y), "boundary", fill=(240, 220, 80), font=font)
    for index, title in enumerate(["raw", "+ foreground border", "+ full rectangle"]):
        draw.text(
            (x0 + (2 + index) * cell[0] + 8, y + 22),
            title,
            fill=(190, 190, 190),
            font=_font(12),
        )


def _row_cells(
    source: np.ndarray,
    labels: np.ndarray,
    raw: np.ndarray,
    closed_foreground: np.ndarray,
    closed_full: np.ndarray,
    cell: tuple[int, int],
) -> list[np.ndarray]:
    return [
        _fit(source, cell),
        _fit(recolour_labels(labels), cell),
        _display_boundary(raw, cell, labels.shape),
        _display_boundary(closed_foreground, cell, labels.shape),
        _display_boundary(closed_full, cell, labels.shape),
    ]


def _readme_preview(
    panel_id: str,
    source: np.ndarray,
    labels: np.ndarray,
    closed_full: np.ndarray,
    out_dir: Path,
) -> None:
    """Three-column README image: source, mask, and final closed boundary only."""
    cell = (500, 330)
    headers = ["source", "mask", "closed boundary (full rectangle)"]
    canvas = Image.new("RGB", (3 * cell[0], cell[1] + 34), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    for index, title in enumerate(headers):
        draw.text((index * cell[0] + 8, 9), title, fill=(240, 220, 80), font=_font(14))
    cells = [
        _fit(source, cell),
        _fit(recolour_labels(labels), cell),
        _display_boundary(closed_full, cell, labels.shape),
    ]
    for index, cell_image in enumerate(cells):
        canvas.paste(Image.fromarray(cell_image), (index * cell[0], 34))
    out_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(out_dir / f"{panel_id}.png")


def _panel_comparison(
    panel_id: str,
    source: np.ndarray,
    labels: np.ndarray,
    raw: np.ndarray,
    closed_foreground: np.ndarray,
    closed_full: np.ndarray,
    out_dir: Path,
) -> None:
    cell = (460, 320)
    width = LABEL_WIDTH + 5 * cell[0]
    height = 76 + cell[1]
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    _group_headers(draw, 10, cell, _font(14))
    draw.text((6, 10), panel_id, fill=(235, 235, 235), font=_font(14))
    for index, cell_image in enumerate(
        _row_cells(source, labels, raw, closed_foreground, closed_full, cell)
    ):
        canvas.paste(Image.fromarray(cell_image), (LABEL_WIDTH + index * cell[0], 76))
    out_dir.mkdir(parents=True, exist_ok=True)
    canvas.save(out_dir / "comparison.png")


def _contact(rows: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]], out_path: Path) -> None:
    width = LABEL_WIDTH + 5 * CELL[0]
    header_height = 78
    height = header_height + len(rows) * CELL[1]
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    _group_headers(draw, 9, CELL, _font(14))
    for row_index, (panel_id, source, labels, raw, closed_foreground, closed_full, metrics) in enumerate(rows):
        y = header_height + row_index * CELL[1]
        draw.text((6, y + 8), panel_id, fill=(235, 235, 235), font=_font(14))
        draw.text((6, y + 30), f"bbox={metrics['bbox']}", fill=(170, 170, 170), font=_font(12))
        draw.text(
            (6, y + 48),
            f"raw={metrics['raw_pixels']}  full={metrics['full_pixels']}",
            fill=(170, 170, 170),
            font=_font(12),
        )
        for index, cell_image in enumerate(
            _row_cells(source, labels, raw, closed_foreground, closed_full, CELL)
        ):
            canvas.paste(Image.fromarray(cell_image), (LABEL_WIDTH + index * CELL[0], y))
    footer = "source = original panel with data bbox; mask = cropped labels; boundary = pure 1 px masks (3 px display)"
    draw.text((6, height - 20), footer, fill=(140, 140, 140), font=_font(12))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("validation/curated/manifest.jsonl"))
    parser.add_argument("--root", type=Path, default=Path("validation/curated"))
    parser.add_argument("--out", type=Path, default=Path("runs/boundary_comparison"))
    parser.add_argument("--readme-dir", type=Path, default=None)
    parser.add_argument(
        "--panels",
        nargs="+",
        default=["newimage_p0", "paper22_fig5a", "ph01_page8_panel3", "paper22_fig6", "schematic_panel2"],
    )
    args = parser.parse_args(argv)

    records = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    wanted = set(args.panels)
    records = [record for record in records if record["panel_id"] in wanted]
    rows = []
    report = []
    for record in records:
        panel_id = record["panel_id"]
        override = PANEL_OVERRIDES.get(panel_id)
        if override is not None:
            labels_path = override["labels"]
            image_path = override["image"]
        else:
            labels_path = args.root / record["files"]["labels"]
            image_path = args.root / record["files"]["image"]
        labels_full = np.load(labels_path)["labels"]
        source_full = np.asarray(Image.open(image_path).convert("RGB"))
        bbox = api.detect_panel_bbox(source_full)
        x0, y0, x1, y1 = bbox
        labels = labels_full[y0:y1, x0:x1].copy()
        source_cropped = source_full[y0:y1, x0:x1].copy()
        source_display = _bbox_overlay(source_full, bbox)
        raw = api.boundary_union(labels)
        closed_foreground = api.closed_boundary_union(labels, include_background=False)
        closed_full = api.closed_boundary_union(labels)
        metrics = {
            "panel_id": panel_id,
            "source_kind": override["kind"] if override else "curated",
            "image": str(image_path),
            "labels": str(labels_path),
            "bbox": list(bbox),
            "shape": list(labels.shape),
            "raw_pixels": int(raw.sum()),
            "foreground_pixels": int(closed_foreground.sum()),
            "full_pixels": int(closed_full.sum()),
        }
        report.append(metrics)
        panel_dir = args.out / panel_id
        panel_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(source_display).save(panel_dir / "source.png")
        Image.fromarray(source_cropped).save(panel_dir / "source_cropped.png")
        Image.fromarray(recolour_labels(labels)).save(panel_dir / "mask.png")
        Image.fromarray(np.where(raw[..., None], 255, 0).astype(np.uint8) * np.ones(3, dtype=np.uint8)).save(panel_dir / "boundary_raw.png")
        Image.fromarray(np.where(closed_foreground[..., None], 255, 0).astype(np.uint8) * np.ones(3, dtype=np.uint8)).save(panel_dir / "boundary_foreground.png")
        Image.fromarray(np.where(closed_full[..., None], 255, 0).astype(np.uint8) * np.ones(3, dtype=np.uint8)).save(panel_dir / "boundary_full.png")
        _panel_comparison(panel_id, source_display, labels, raw, closed_foreground, closed_full, panel_dir)
        if args.readme_dir is not None:
            _readme_preview(panel_id, source_display, labels, closed_full, args.readme_dir)
        rows.append((panel_id, source_display, labels, raw, closed_foreground, closed_full, metrics))
        print(
            f"{panel_id:24s} bbox={bbox} raw={metrics['raw_pixels']:6d} "
            f"fg={metrics['foreground_pixels']:6d} full={metrics['full_pixels']:6d}"
        )

    _contact(rows, args.out / "contact.png")
    (args.out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out / 'contact.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
