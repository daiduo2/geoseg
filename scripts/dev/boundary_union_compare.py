"""Compare raw partition boundaries and pure boundary masks.

Outputs:
- per-panel exact PNGs at native resolution;
- a contact sheet where exact masks are resampled and thickened only for display;
- a JSON report with raw union counts and deterministic cleanup merge counts.

No smoothing, no skeleton, no keep/remove decisions.
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

BACKGROUND = (0, 0, 0)
WHITE = (255, 255, 255)
LABEL_WIDTH = 170
CELL = (430, 300)


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
    out[mask] = WHITE
    return out


def _overlay(labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = recolour_labels(labels)
    out[mask] = WHITE
    return out


def _fit(image: np.ndarray, size: tuple[int, int]) -> Image.Image:
    pil = Image.fromarray(image).convert("RGB")
    scale = min(size[0] / pil.width, size[1] / pil.height)
    resized = pil.resize(
        (max(1, int(pil.width * scale)), max(1, int(pil.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", size, (18, 18, 18))
    canvas.paste(resized, ((size[0] - resized.width) // 2, (size[1] - resized.height) // 2))
    return canvas


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
    out[small] = WHITE
    return out


def _display_overlay(labels: np.ndarray, mask: np.ndarray, size: tuple[int, int], thickness: int = 3) -> np.ndarray:
    base = np.asarray(_fit(recolour_labels(labels), size))
    small = _fit_mask(mask, size, labels.shape)
    small = ndimage.binary_dilation(small, iterations=thickness - 1)
    base[small] = WHITE
    return base


def _panel_grid(panel_id: str, labels: np.ndarray, raw_both: np.ndarray, raw_lower: np.ndarray, out_dir: Path) -> Image.Image:
    panels = {
        "labels": _overlay(labels, np.zeros(labels.shape, dtype=bool)),
        "raw union (both sides)": _pure(raw_both),
        "raw union (canonical lower side)": _pure(raw_lower),
        "raw union on labels": _overlay(labels, raw_both),
    }
    cell = (max(1, labels.shape[1]), max(1, labels.shape[0]))
    grid = Image.new("RGB", (cell[0] * 2, cell[1] * 2), (18, 18, 18))
    for index, (title, array) in enumerate(panels.items()):
        image = Image.fromarray(array).convert("RGB")
        image.thumbnail(cell, Image.Resampling.LANCZOS)
        x = (index % 2) * cell[0]
        y = (index // 2) * cell[1]
        grid.paste(image, (x, y))
        draw = ImageDraw.Draw(grid)
        draw.rectangle([x, y, x + min(260, image.width), y + 20], fill=(0, 0, 0))
        draw.text((x + 5, y + 4), title, fill=(255, 230, 80), font=_font(13))
    out_dir.mkdir(parents=True, exist_ok=True)
    grid.save(out_dir / "comparison.png")
    return grid


def _contact(rows: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, dict]], out_path: Path) -> None:
    height = 60 + len(rows) * CELL[1]
    canvas = Image.new("RGB", (LABEL_WIDTH + 3 * CELL[0], height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    font = _font(14)
    small = _font(12)
    for index, title in enumerate(["labels", "raw union (both)", "raw union (lower)"]):
        draw.text((LABEL_WIDTH + index * CELL[0] + 8, 9), title, fill=(240, 220, 80), font=font)
    for row_index, (panel_id, labels, raw_both, raw_lower, metrics) in enumerate(rows):
        y = 36 + row_index * CELL[1]
        draw.text((6, y + 8), panel_id, fill=(235, 235, 235), font=font)
        draw.text(
            (6, y + 30),
            f"both={metrics['both_pixels']}  lower={metrics['lower_pixels']}",
            fill=(170, 170, 170),
            font=small,
        )
        draw.text(
            (6, y + 48),
            f"cleanup merges={metrics['merges']}",
            fill=(170, 170, 170),
            font=small,
        )
        labels_cell = _fit(recolour_labels(labels), CELL)
        both_cell = Image.fromarray(_display_boundary(raw_both, CELL, labels.shape))
        lower_cell = Image.fromarray(_display_boundary(raw_lower, CELL, labels.shape))
        canvas.paste(labels_cell, (LABEL_WIDTH, y))
        canvas.paste(both_cell, (LABEL_WIDTH + CELL[0], y))
        canvas.paste(lower_cell, (LABEL_WIDTH + 2 * CELL[0], y))
    footer = "contact uses 3 px display thickening; per-panel raw_*_pure.png files are exact 1 px masks"
    draw.text((6, height - 18), footer, fill=(140, 140, 140), font=small)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("validation/curated/manifest.jsonl")
    )
    parser.add_argument("--root", type=Path, default=Path("validation/curated"))
    parser.add_argument("--out", type=Path, default=Path("runs/boundary_union_compare"))
    parser.add_argument(
        "--panels",
        nargs="+",
        default=[
            "newimage_p0",
            "paper22_fig5a",
            "paper23_fig7a",
            "schematic_panel2",
            "ph01_page8_panel3",
        ],
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
        raw_both = api.boundary_union(labels)
        raw_lower = api.boundary_union(labels, side="lower")
        cleaned, merges = api.prune_small_components(labels)
        metrics = {
            "panel_id": panel_id,
            "shape": list(labels.shape),
            "n_labels": int(len(np.unique(labels))),
            "both_pixels": int(raw_both.sum()),
            "lower_pixels": int(raw_lower.sum()),
            "merges": len(merges),
            "changed_pixels": int((cleaned != labels).sum()),
        }
        report.append(metrics)
        panel_dir = args.out / panel_id
        panel_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(_overlay(labels, np.zeros(labels.shape, dtype=bool))).save(panel_dir / "labels.png")
        Image.fromarray(_pure(raw_both)).save(panel_dir / "raw_both_pure.png")
        Image.fromarray(_pure(raw_lower)).save(panel_dir / "raw_lower_pure.png")
        Image.fromarray(_overlay(labels, raw_both)).save(panel_dir / "raw_both_overlay.png")
        _panel_grid(panel_id, labels, raw_both, raw_lower, panel_dir)
        rows.append((panel_id, labels, raw_both, raw_lower, metrics))
        print(
            f"{panel_id:24s} both={metrics['both_pixels']:6d} "
            f"lower={metrics['lower_pixels']:6d} merges={metrics['merges']}"
        )

    _contact(rows, args.out / "contact.png")
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {args.out / 'contact.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
