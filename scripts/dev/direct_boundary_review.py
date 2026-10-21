"""Run the existing label-boundary extraction on recovered direct labels.

This does not segment from pixels.  It takes the recovered direct label map,
extracts boundary curves, and renders a review sheet + metrics.
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


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    scale = min(size[0] / image.width, size[1] / image.height)
    resized = image.resize(
        (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", size, (18, 18, 18))
    canvas.paste(resized, ((size[0] - resized.width) // 2, (size[1] - resized.height) // 2))
    return canvas


def _draw_curves(image: np.ndarray, curves) -> Image.Image:
    out = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(out)
    for curve in curves:
        points = [(float(x), float(y)) for x, y in curve.points]
        if len(points) < 2:
            continue
        if curve.closed:
            points.append(points[0])
        draw.line(points, fill=(255, 30, 40), width=2)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("validation/curated/manifest.jsonl")
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("runs/direct_boundary_review"))
    parser.add_argument("--panels", nargs="+", default=None)
    args = parser.parse_args(argv)

    root = args.root or args.manifest.parent
    records = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    wanted = set(args.panels or [])
    if not wanted:
        wanted = {record["panel_id"] for record in records} - {"ph01_page8_panel3"}
    records = [record for record in records if record["panel_id"] in wanted]
    args.out.mkdir(parents=True, exist_ok=True)
    report = []
    rows = []

    for record in records:
        panel_id = record["panel_id"]
        labels = np.load(root / record["files"]["labels"])["labels"]
        image = np.asarray(Image.open(root / record["files"]["image"]).convert("RGB"))
        curves = api.curves_from_labels(labels)
        refined = api.refine_curves(curves, labels)
        model = api.curve_model_from_labels(labels)
        rasterized = model.rasterize()
        accuracy = float((rasterized == labels).mean())
        lengths = []
        for curve in curves:
            points = np.asarray(curve.points, dtype=float)
            lengths.append(
                float(np.hypot(*np.diff(points, axis=0).T).sum())
                if len(points) > 1
                else 0.0
            )
        boundary_curves = _draw_curves(image, curves)
        refined_curves = _draw_curves(image, refined)
        panel_dir = args.out / panel_id
        panel_dir.mkdir(parents=True, exist_ok=True)
        boundary_curves.save(panel_dir / "boundary_curves_raw.png")
        refined_curves.save(panel_dir / "boundary_curves_refined.png")
        (panel_dir / "refined_curves.json").write_text(
            json.dumps(
                [
                    {
                        "points": [[round(float(x), 3), round(float(y), 3)] for x, y in curve.points],
                        "closed": curve.closed,
                        "labels": list(curve.labels) if curve.labels else None,
                        "length": round(curve.length, 3),
                        "max_curvature": round(curve.max_curvature, 5),
                        "smoothing": round(curve.smoothing, 3),
                    }
                    for curve in refined
                ],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        np.savez_compressed(panel_dir / "labels_rasterized.npz", labels=rasterized)

        metrics = {
            "panel_id": panel_id,
            "shape": list(labels.shape),
            "n_curves_raw": len(curves),
            "n_curves_refined": len(refined),
            "n_dropped": len(curves) - len(refined),
            "n_closed_refined": sum(1 for curve in refined if curve.closed),
            "n_short_lt20_raw": sum(1 for length in lengths if length < 20),
            "total_curve_length_raw": round(float(sum(lengths)), 2),
            "total_curve_length_refined": round(float(sum(curve.length for curve in refined)), 2),
            "max_curvature_refined": round(
                max((curve.max_curvature for curve in refined), default=0.0), 4
            ),
            "roundtrip_accuracy": round(accuracy, 4),
            "boundary_pixels": int(
                ((labels[:, :-1] != labels[:, 1:]).sum() + (labels[:-1, :] != labels[1:, :]).sum())
            ),
        }
        report.append(metrics)
        overlay = Image.open(root / record["files"]["overlay"]).convert("RGB")
        rows.append((panel_id, image, overlay, boundary_curves, refined_curves, metrics))
        print(
            f"{panel_id:24s} raw={metrics['n_curves_raw']:3d} "
            f"refined={metrics['n_curves_refined']:3d} "
            f"dropped={metrics['n_dropped']:2d} "
            f"maxk={metrics['max_curvature_refined']:.3f} "
            f"acc={accuracy:.3f}"
        )

    # Contact sheet: source | direct labels overlay | raw | refined boundaries.
    cell = (300, 190)
    row_label = 150
    height = 30 + len(rows) * cell[1]
    canvas = Image.new("RGB", (row_label + 4 * cell[0], height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    font = _font(14)
    for index, header in enumerate(
        ["source", "direct labels", "raw boundaries", "refined boundaries"]
    ):
        draw.text((row_label + index * cell[0] + 6, 7), header, fill=(240, 220, 80), font=font)
    for row_index, (panel_id, image, overlay, boundary, refined, metrics) in enumerate(rows):
        y = 30 + row_index * cell[1]
        draw.text((6, y + 8), panel_id, fill=(235, 235, 235), font=font)
        draw.text(
            (6, y + 28),
            f"{metrics['n_curves_raw']}->{metrics['n_curves_refined']} curves, "
            f"acc={metrics['roundtrip_accuracy']:.3f}",
            fill=(170, 170, 170),
            font=font,
        )
        canvas.paste(_fit(Image.fromarray(image), cell), (row_label, y))
        canvas.paste(_fit(overlay, cell), (row_label + cell[0], y))
        canvas.paste(_fit(boundary, cell), (row_label + 2 * cell[0], y))
        canvas.paste(_fit(refined, cell), (row_label + 3 * cell[0], y))
    canvas.save(args.out / "contact.png")
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {args.out / 'contact.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
