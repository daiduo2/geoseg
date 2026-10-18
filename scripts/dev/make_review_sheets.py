#!/usr/bin/env python3
"""Render numbered contact sheets for visual review of the validation corpus.

    python scripts/dev/make_review_sheets.py [--split dev] [--all]

Writes validation/review/sheet_XX.jpg plus index.csv/index.md. Each tile shows
the panel with its label overlay and a numbered header; refer to tiles by that
number when selecting good examples.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "validation"


def load_rows(root: Path, split: str | None, include_negatives: bool) -> list[dict]:
    rows = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]
    if split:
        rows = [r for r in rows if r["split"] == split]
    if not include_negatives:
        rows = [r for r in rows if r["labels"]["n_labels"] >= 2]
    return rows


def make_overlay(image: np.ndarray, labels: np.ndarray, convention: str) -> np.ndarray:
    from geoseg.core.image_ops import create_overlay

    if convention == "negative_is_background":
        labels = labels.copy()
        labels[labels < 0] = 0
        skip_bg = True
    elif convention == "zero_is_region":
        skip_bg = False
    else:
        skip_bg = True
    return create_overlay(
        image,
        labels.astype(np.int32),
        None,
        alpha=0.62,
        boundary_mode="thin",
        skip_background=skip_bg,
    )


def _font(size: int):
    for path in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="make validation review sheets")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--split", default=None, choices=["dev", "holdout", "rare", "negative"])
    parser.add_argument("--all", action="store_true", help="include negative panels")
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--tile-width", type=int, default=440)
    args = parser.parse_args(argv)

    rows = load_rows(args.root, args.split, args.all)
    out = args.root / "review"
    out.mkdir(parents=True, exist_ok=True)

    small = _font(13)
    per_sheet = args.cols * args.rows
    index = []

    for sheet_no in range(math.ceil(len(rows) / per_sheet)):
        chunk = rows[sheet_no * per_sheet : (sheet_no + 1) * per_sheet]
        tiles = []
        for offset, row in enumerate(chunk):
            number = sheet_no * per_sheet + offset + 1
            image = np.array(
                Image.open(args.root / row["files"]["image"]).convert("RGB")
            )
            labels = np.load(args.root / row["files"]["labels"])["labels"]
            overlay = make_overlay(image, labels, row["labels"]["convention"])

            tile_w = args.tile_width
            tile_h = int(overlay.shape[0] * tile_w / overlay.shape[1])
            tile_h = max(160, min(tile_h, 340))
            tile = Image.fromarray(overlay).resize((tile_w, tile_h), Image.LANCZOS)
            canvas = Image.new("RGB", (tile_w, tile_h + 44), (20, 20, 20))
            canvas.paste(tile, (0, 44))
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 4), f"#{number}  {row['panel_id']}", font=small, fill=(255, 255, 255))
            draw.text(
                (8, 23),
                f"n={row['labels']['n_labels']}  {row['split']}  {','.join(row['flags']) or '-'}",
                font=small,
                fill=(180, 220, 255),
            )
            canvas.save(out / f"tile_{number:03d}.jpg", quality=92)
            tiles.append(canvas)
            index.append({
                "number": number,
                "panel_id": row["panel_id"],
                "split": row["split"],
                "tier": row["tier"],
                "source": row["source"]["root"],
                "n_labels": row["labels"]["n_labels"],
                "shape": row["labels"]["shape"],
                "flags": row["flags"],
                "overlay": f"review/tile_{number:03d}.jpg",
            })

        tile_w = args.tile_width
        tile_h = max(t.size[1] for t in tiles)
        sheet = Image.new(
            "RGB",
            (args.cols * tile_w, args.rows * tile_h),
            (12, 12, 12),
        )
        for i, tile in enumerate(tiles):
            c, r = i % args.cols, i // args.cols
            sheet.paste(tile, (c * tile_w, r * tile_h))
        sheet.save(out / f"sheet_{sheet_no + 1:02d}.jpg", quality=90)
        print(f"wrote {out / f'sheet_{sheet_no + 1:02d}.jpg'}  ({len(tiles)} tiles)")

    (out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Panel review index", "", "| # | panel_id | split | tier | source | n | flags |", "|---:|---|---|---|---|---:|---|"]
    for item in index:
        lines.append(
            f"| {item['number']} | `{item['panel_id']}` | {item['split']} | {item['tier']} | "
            f"{item['source']} | {item['n_labels']} | {','.join(item['flags']) or '-'} |"
        )
    (out / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"panels: {len(index)}  sheets: {math.ceil(len(index) / per_sheet)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
