"""Visual audit of exact partition closures on recoloured label masks.

The baseline is the partition itself, recoloured into a clean categorical mask.
No photograph (raw or denoised) is used as the audit baseline.  Exact closure
pixels are drawn thick with a dark halo so the boundary is the dominant visual
object.  This is Layer 1 only: no spline and no smoothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from geoseg import api

_BACKGROUND = np.array([28, 28, 32], dtype=np.uint8)
_HALO = np.array([8, 8, 10], dtype=np.uint8)
# Deliberately avoids green/yellow, which are reserved for keep/review.
_LABEL_COLOURS = [
    (225, 100, 100),
    (70, 130, 210),
    (150, 100, 205),
    (235, 140, 60),
    (80, 190, 205),
    (200, 90, 170),
    (110, 170, 230),
    (170, 110, 75),
    (220, 130, 160),
    (100, 200, 170),
    (180, 180, 220),
    (150, 150, 150),
    (240, 160, 120),
    (60, 100, 180),
    (190, 140, 230),
    (120, 80, 150),
    (245, 180, 140),
    (140, 80, 50),
    (60, 160, 190),
    (210, 170, 200),
    (130, 140, 240),
    (185, 185, 160),
]
_ACTION_COLOURS = {
    "keep": np.array([0, 255, 90], dtype=np.uint8),
    "review": np.array([255, 210, 0], dtype=np.uint8),
    "remove": np.array([255, 40, 40], dtype=np.uint8),
}
_EXACT_COLOUR = np.array([255, 255, 255], dtype=np.uint8)


def _font(size: int = 14):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()


def recolour_labels(labels: np.ndarray) -> np.ndarray:
    """Categorical mask: each nonzero label gets a flat colour, background dark."""
    labels = np.asarray(labels)
    out = np.empty((*labels.shape, 3), dtype=np.uint8)
    out[:] = _BACKGROUND
    for label in np.unique(labels):
        value = int(label)
        if value == 0:
            continue
        out[labels == value] = _LABEL_COLOURS[(value - 1) % len(_LABEL_COLOURS)]
    return out


def _mask_for(pixels: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for item in pixels:
        rows = np.rint(item[:, 1]).astype(int)
        cols = np.rint(item[:, 0]).astype(int)
        valid = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
        mask[rows[valid], cols[valid]] = True
    return mask


def _thicken(mask: np.ndarray, thickness: int) -> np.ndarray:
    thickness = max(1, int(thickness))
    if thickness == 1:
        return mask
    return ndimage.binary_dilation(mask, iterations=thickness - 1)


def _paint(
    out: np.ndarray,
    layers: list[tuple[np.ndarray, np.ndarray]],
    thickness: int,
) -> np.ndarray:
    """Paint masks with a dark halo, then bright cores on top."""
    if not layers:
        return out
    halo = np.zeros(out.shape[:2], dtype=bool)
    for mask, _ in layers:
        halo |= _thicken(mask, thickness + 3)
    out[halo] = _HALO
    for mask, colour in layers:
        out[_thicken(mask, thickness)] = colour
    return out


def _render(
    base: np.ndarray,
    layers: list[tuple[np.ndarray, np.ndarray]],
    thickness: int,
) -> np.ndarray:
    out = base.copy()
    return _paint(out, layers, thickness)


def _thickness(shape: tuple[int, int]) -> int:
    return max(4, int(round(min(shape[:2]) * 0.008)))


def _fit(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray(image).convert("RGB")
    scale = min(size[0] / pil.width, size[1] / pil.height)
    resized = pil.resize(
        (max(1, int(pil.width * scale)), max(1, int(pil.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", size, (18, 18, 18))
    canvas.paste(resized, ((size[0] - resized.width) // 2, (size[1] - resized.height) // 2))
    return np.asarray(canvas).copy()


def _fit_mask(mask: np.ndarray, size: tuple[int, int], shape: tuple[int, int]) -> np.ndarray:
    scale = min(size[0] / shape[1], size[1] / shape[0])
    width = max(1, int(shape[1] * scale))
    height = max(1, int(shape[0] * scale))
    pil = Image.fromarray(mask.astype(np.uint8) * 255)
    resized = pil.resize((width, height), Image.Resampling.BOX)
    small = np.asarray(resized) > 0
    canvas = np.zeros((size[1], size[0]), dtype=bool)
    left = (size[0] - width) // 2
    top = (size[1] - height) // 2
    canvas[top : top + height, left : left + width] = small
    return canvas


def _contact_cell(
    baseline: np.ndarray,
    masks: list[tuple[np.ndarray, np.ndarray]],
    size: tuple[int, int],
) -> np.ndarray:
    cell = _fit(baseline, size)
    layers = [
        (_fit_mask(mask, size, baseline.shape[:2]), colour) for mask, colour in masks
    ]
    return _paint(cell, layers, thickness=3)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("validation/curated/manifest.jsonl")
    )
    parser.add_argument("--root", type=Path, default=Path("validation/curated"))
    parser.add_argument("--out", type=Path, default=Path("runs/mask_boundary_audit"))
    parser.add_argument("--panels", nargs="+", default=None)
    args = parser.parse_args(argv)

    records = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    wanted = set(args.panels or [])
    if wanted:
        records = [record for record in records if record["panel_id"] in wanted]

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    report = []
    for record in records:
        panel_id = record["panel_id"]
        labels = np.load(args.root / record["files"]["labels"])["labels"]
        shape = tuple(int(value) for value in labels.shape)
        baseline = recolour_labels(labels)
        boundaries = api.extract_exact_boundaries(labels)
        decisions = api.decide_exact_boundaries(labels, boundaries)
        keep = [item.boundary.pixels for item in decisions if item.action == "keep"]
        review = [item.boundary.pixels for item in decisions if item.action == "review"]
        remove = [item.boundary.pixels for item in decisions if item.action == "remove"]
        masks = [
            (_mask_for(keep, shape), _ACTION_COLOURS["keep"]),
            (_mask_for(review, shape), _ACTION_COLOURS["review"]),
            (_mask_for(remove, shape), _ACTION_COLOURS["remove"]),
        ]
        exact_mask = _mask_for([item.pixels for item in boundaries], shape)
        thickness = _thickness(shape)
        exact_img = _render(baseline, [(exact_mask, _EXACT_COLOUR)], thickness)
        decision_img = _render(baseline, masks, thickness)

        panel_dir = args.out / panel_id
        panel_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(baseline).save(panel_dir / "labels_recoloured.png")
        Image.fromarray(exact_img).save(panel_dir / "exact_boundaries.png")
        Image.fromarray(decision_img).save(panel_dir / "boundary_decisions.png")
        (panel_dir / "decisions.json").write_text(
            json.dumps(
                [
                    {
                        "id": item.boundary.id,
                        "labels": list(item.boundary.labels_pair),
                        "background_role": item.background_role,
                        "action": item.action,
                        "reasons": item.reasons,
                        "length_pixels": item.boundary.length,
                    }
                    for item in decisions
                ],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        metrics = {
            "panel_id": panel_id,
            "status": "audited_on_recoloured_labels",
            "baseline": "recoloured_labels",
            "n_labels": int(len(np.unique(labels))),
            "n_exact_objects": len(boundaries),
            "n_keep": len(keep),
            "n_review": len(review),
            "n_remove": len(remove),
            "boundary_thickness_px": thickness,
        }
        if panel_id == "ph01_page8_panel3":
            metrics["caveat"] = "source labels are under-segmented; missing layers cannot be audited"
        report.append(metrics)
        rows.append((panel_id, baseline, exact_mask, masks, metrics))
        print(
            f"{panel_id:24s} keep={metrics['n_keep']:2d} review={metrics['n_review']:2d} "
            f"remove={metrics['n_remove']:2d} labels={metrics['n_labels']:2d}"
        )

    cell = (440, 280)
    row_label = 210
    height = 48 + len(rows) * cell[1]
    canvas = Image.new("RGB", (row_label + 3 * cell[0], height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    font = _font(15)
    small = _font(13)
    draw.text(
        (6, 8),
        "baseline = labels recoloured\nwhite = exact closure  |  green = keep  yellow = review  red = remove",
        fill=(240, 220, 80),
        font=small,
    )
    for index, header in enumerate(["labels recoloured", "exact closure", "keep / review"]):
        draw.text((row_label + index * cell[0] + 8, 8), header, fill=(240, 220, 80), font=font)
    for row_index, (panel_id, baseline, exact_mask, masks, metrics) in enumerate(rows):
        y = 48 + row_index * cell[1]
        draw.text((6, y + 8), panel_id, fill=(235, 235, 235), font=font)
        draw.text(
            (6, y + 30),
            f"K{metrics['n_keep']} R{metrics['n_review']} X{metrics['n_remove']}",
            fill=(170, 170, 170),
            font=small,
        )
        draw.text(
            (6, y + 48),
            f"{metrics['n_labels']} labels",
            fill=(140, 140, 140),
            font=small,
        )
        cell_a = _contact_cell(baseline, [(exact_mask, _EXACT_COLOUR)], cell)
        cell_b = _contact_cell(baseline, masks, cell)
        canvas.paste(Image.fromarray(_fit(baseline, cell)), (row_label, y))
        canvas.paste(Image.fromarray(cell_a), (row_label + cell[0], y))
        canvas.paste(Image.fromarray(cell_b), (row_label + 2 * cell[0], y))
    canvas.save(args.out / "contact.png")
    (args.out / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {args.out / 'contact.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
