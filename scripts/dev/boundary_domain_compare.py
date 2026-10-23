"""Compare panel-domain cropping and outer-rectangle closure.

For each panel:
- detect the coloured panel data bbox from the source image;
- crop the labels to that rectangle;
- show raw boundary, cropped raw boundary, foreground border closure, and
  full-rectangle closure.

The point is to separate two different objects:
1. interfaces between labels inside the data rectangle;
2. the domain edge (the rectangle itself), which must be added explicitly.
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


def _font(size: int = 14):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()


def _pure(mask: np.ndarray, thickness: int = 1) -> np.ndarray:
    if thickness > 1:
        mask = ndimage.binary_dilation(mask, iterations=thickness - 1)
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask] = (255, 255, 255)
    return out


def _fit(image: np.ndarray, size: tuple[int, int], resample=Image.Resampling.LANCZOS) -> np.ndarray:
    pil = Image.fromarray(image).convert("RGB")
    scale = min(size[0] / pil.width, size[1] / pil.height)
    resized = pil.resize(
        (max(1, int(pil.width * scale)), max(1, int(pil.height * scale))),
        resample,
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


def _display_labels(labels: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return _fit(recolour_labels(labels), size)


def _bbox_overlay(labels: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    out = recolour_labels(labels).copy()
    x0, y0, x1, y1 = bbox
    out[y0 : y0 + 2, x0:x1] = (255, 230, 80)
    out[max(0, y1 - 2) : y1, x0:x1] = (255, 230, 80)
    out[y0:y1, x0 : x0 + 2] = (255, 230, 80)
    out[y0:y1, max(0, x1 - 2) : x1] = (255, 230, 80)
    return out


def _panel_grid(
    panel_id: str,
    original_labels: np.ndarray,
    cropped_labels: np.ndarray,
    original_raw: np.ndarray,
    cropped_raw: np.ndarray,
    closed_foreground: np.ndarray,
    closed_all: np.ndarray,
    bbox: tuple[int, int, int, int],
    out_dir: Path,
) -> None:
    panels = [
        ("original labels + bbox", _bbox_overlay(original_labels, bbox)),
        ("cropped labels", recolour_labels(cropped_labels)),
        ("original raw union", _pure(original_raw)),
        ("cropped raw union", _pure(cropped_raw)),
        ("cropped + foreground border", _pure(closed_foreground)),
        ("cropped + full rectangle", _pure(closed_all)),
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = Image.new("RGB", (2 * 500, 3 * 330), (18, 18, 18))
    for index, (title, array) in enumerate(panels):
        image = Image.fromarray(_fit(array, (500, 300))).convert("RGB")
        x = (index % 2) * 500
        y = (index // 2) * 330
        grid.paste(image, (x, y))
        draw = ImageDraw.Draw(grid)
        draw.rectangle([x, y, x + 260, y + 20], fill=(0, 0, 0))
        draw.text((x + 6, y + 3), title, fill=(240, 220, 80), font=_font(14))
    grid.save(out_dir / "comparison.png")


def _contact(rows: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, dict]], out_path: Path) -> None:
    headers = ["cropped labels", "cropped raw", "+ foreground border", "+ full rectangle"]
    height = 60 + len(rows) * CELL[1]
    canvas = Image.new("RGB", (LABEL_WIDTH + 4 * CELL[0], height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    for index, title in enumerate(headers):
        draw.text((LABEL_WIDTH + index * CELL[0] + 8, 9), title, fill=(240, 220, 80), font=_font(14))
    for row_index, (panel_id, labels, raw, closed, metrics) in enumerate(rows):
        y = 36 + row_index * CELL[1]
        draw.text((6, y + 8), panel_id, fill=(235, 235, 235), font=_font(14))
        draw.text((6, y + 30), f"bbox={metrics['bbox']}", fill=(170, 170, 170), font=_font(12))
        draw.text(
            (6, y + 48),
            f"raw={metrics['cropped_raw_pixels']} closed={metrics['closed_foreground_pixels']}",
            fill=(170, 170, 170),
            font=_font(12),
        )
        cells = [
            _display_labels(labels, CELL),
            _display_boundary(raw, CELL, labels.shape),
            _display_boundary(closed, CELL, labels.shape),
            _display_boundary(raw, CELL, labels.shape),
        ]
        # last cell: full rectangle closure
        cells[3] = _display_boundary(metrics["closed_all_mask"], CELL, labels.shape)
        for index, cell in enumerate(cells):
            canvas.paste(Image.fromarray(cell), (LABEL_WIDTH + index * CELL[0], y))
    footer = "contact boundaries are thickened to 3 px for display; per-panel pure PNGs are exact 1 px masks"
    draw.text((6, height - 20), footer, fill=(140, 140, 140), font=_font(12))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("validation/curated/manifest.jsonl"))
    parser.add_argument("--root", type=Path, default=Path("validation/curated"))
    parser.add_argument("--out", type=Path, default=Path("runs/boundary_domain_compare"))
    parser.add_argument(
        "--panels",
        nargs="+",
        default=["newimage_p0", "ph01_page8_panel3", "paper22_fig5a", "paper23_fig7a", "schematic_panel2"],
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
        labels = np.load(args.root / record["files"]["labels"])["labels"]
        image = np.asarray(Image.open(args.root / record["files"]["image"]).convert("RGB"))
        bbox = api.detect_panel_bbox(image)
        x0, y0, x1, y1 = bbox
        cropped = labels[y0:y1, x0:x1].copy()
        original_raw = api.boundary_union(labels)
        cropped_raw = api.boundary_union(cropped)
        closed_foreground = api.closed_boundary_union(cropped)
        closed_all = api.closed_boundary_union(cropped, include_background=True)
        metrics = {
            "panel_id": panel_id,
            "bbox": list(bbox),
            "shape_before": list(labels.shape),
            "shape_after": list(cropped.shape),
            "original_raw_pixels": int(original_raw.sum()),
            "cropped_raw_pixels": int(cropped_raw.sum()),
            "closed_foreground_pixels": int(closed_foreground.sum()),
            "closed_all_pixels": int(closed_all.sum()),
            "closed_all_mask": closed_all,
        }
        report.append({key: value for key, value in metrics.items() if key != "closed_all_mask"})
        panel_dir = args.out / panel_id
        panel_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(recolour_labels(cropped)).save(panel_dir / "cropped_labels.png")
        Image.fromarray(_pure(cropped_raw)).save(panel_dir / "cropped_raw_pure.png")
        Image.fromarray(_pure(closed_foreground)).save(panel_dir / "closed_foreground_pure.png")
        Image.fromarray(_pure(closed_all)).save(panel_dir / "closed_full_pure.png")
        _panel_grid(panel_id, labels, cropped, original_raw, cropped_raw, closed_foreground, closed_all, bbox, panel_dir)
        rows.append((panel_id, cropped, cropped_raw, closed_foreground, metrics))
        print(
            f"{panel_id:24s} bbox={bbox} raw={metrics['cropped_raw_pixels']:6d} "
            f"fg_close={metrics['closed_foreground_pixels']:6d} full={metrics['closed_all_pixels']:6d}"
        )

    _contact(rows, args.out / "contact.png")
    (args.out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out / 'contact.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
