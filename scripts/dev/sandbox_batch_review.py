"""Batch sandbox attempts + visual-review contact sheets.

This is a recorder, not an automatic acceptance policy.  It crops each panel,
runs the requested engines/layer counts, audits every attempt, and writes
per-panel and master contact sheets for human/agent visual review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from geoseg import api


def _font(size: int = 14):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()


def _fit(image: np.ndarray | Image.Image, size: tuple[int, int]) -> Image.Image:
    picture = image if isinstance(image, Image.Image) else Image.fromarray(image)
    picture = picture.convert("RGB")
    scale = min(size[0] / picture.width, size[1] / picture.height)
    resized = picture.resize(
        (max(1, int(picture.width * scale)), max(1, int(picture.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", size, (18, 18, 18))
    canvas.paste(resized, ((size[0] - resized.width) // 2, (size[1] - resized.height) // 2))
    return canvas


def _panel_sheet(
    cropped: np.ndarray,
    attempts,
    out: Path,
    *,
    cell: tuple[int, int] = (300, 190),
) -> None:
    font = _font(14)
    small = _font(12)
    headers = ["cropped panel"] + [f"n={a.n_layers}" for a in attempts]
    width = len(headers) * cell[0]
    height = 30 + len(attempts) * cell[1] if False else 30 + cell[1]
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    for index, header in enumerate(headers):
        draw.text((index * cell[0] + 6, 7), header, fill=(240, 220, 80), font=font)
    canvas.paste(_fit(cropped, cell), (0, 30))
    for index, attempt in enumerate(attempts, start=1):
        overlay = Image.open(attempt.artifacts["overlay"])
        canvas.paste(_fit(overlay, cell), (index * cell[0], 30))
        flags = []
        if attempt.audit.under_segmented_labels:
            flags.append(f"under={attempt.audit.under_segmented_labels}")
        if attempt.audit.text_labels:
            flags.append(f"text={attempt.audit.text_labels}")
        if attempt.audit.fragmented_labels:
            flags.append(f"frag={len(attempt.audit.fragmented_labels)}")
        text = f"labels={attempt.audit.n_labels} " + " ".join(flags)
        draw.text(
            (index * cell[0] + 6, 30 + cell[1] - 20),
            text[:80],
            fill=(255, 220, 90),
            font=small,
        )
    canvas.save(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("validation/curated/manifest.jsonl")
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("runs/sandbox/batch"))
    parser.add_argument("--engine", default="v4_kmeans")
    parser.add_argument("--layers", type=int, nargs="+", default=[6, 8, 10, 12])
    parser.add_argument("--panels", nargs="+", default=None)
    args = parser.parse_args(argv)

    root = args.root or args.manifest.parent
    records = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.panels:
        wanted = set(args.panels)
        records = [record for record in records if record["panel_id"] in wanted]
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    report = []
    for record in records:
        panel_id = record["panel_id"]
        image = np.asarray(Image.open(root / record["files"]["image"]).convert("RGB"))
        session = api.IterationSession(
            image,
            output_dir=args.out / panel_id,
            audit_config=api.AuditConfig(),
        )
        attempts = []
        for n_layers in args.layers:
            attempt = session.run_attempt(
                args.engine, n_layers, note=f"batch {args.engine} n={n_layers}"
            )
            attempts.append(attempt)
            print(
                f"{panel_id:24s} n={n_layers:2d} labels={attempt.audit.n_labels:2d} "
                f"under={attempt.audit.under_segmented_labels} "
                f"text={attempt.audit.text_labels} "
                f"frag={len(attempt.audit.fragmented_labels)}"
            )
        _panel_sheet(session.crop.image, attempts, args.out / panel_id / "contact.png")
        rows.append((panel_id, session.crop.image, attempts))
        report.append(
            {
                "panel_id": panel_id,
                "crop": {
                    "bbox": list(session.crop.bbox),
                    "offset": list(session.crop.offset),
                    "method": session.crop.method,
                    "shape": list(session.crop.image.shape[:2]),
                },
                "attempts": [
                    {
                        "n_layers": attempt.n_layers,
                        "n_labels": attempt.audit.n_labels,
                        "score": round(attempt.audit.score, 3),
                        "under_segmented": attempt.audit.under_segmented_labels,
                        "text_labels": attempt.audit.text_labels,
                        "fragmented": attempt.audit.fragmented_labels,
                        "text_pixel_fraction": round(attempt.audit.text_pixel_fraction, 4),
                    }
                    for attempt in attempts
                ],
            }
        )

    # Master sheet: rows = panels, columns = cropped + attempts.
    cell = (300, 190)
    max_attempts = max(len(row[2]) for row in rows) if rows else 0
    width = (1 + max_attempts) * cell[0]
    height = 30 + len(rows) * cell[1]
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    font = _font(14)
    for index, header in enumerate(["cropped", *[f"n={n}" for n in args.layers]]):
        draw.text((index * cell[0] + 6, 7), header, fill=(240, 220, 80), font=font)
    for row_index, (panel_id, cropped, attempts) in enumerate(rows):
        y = 30 + row_index * cell[1]
        draw.text((6, y + 7), panel_id, fill=(230, 230, 230), font=font)
        canvas.paste(_fit(cropped, (cell[0] - 150, cell[1])), (0, y))
        for index, attempt in enumerate(attempts, start=1):
            overlay = Image.open(attempt.artifacts["overlay"])
            canvas.paste(_fit(overlay, cell), (index * cell[0], y))
    canvas.save(args.out / "batch_contact.jpg")

    (args.out / "batch_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {args.out / 'batch_contact.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
