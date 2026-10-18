#!/usr/bin/env python3
"""Build validation/best from the most-iterated, visually stable panels.

The default set is the five ``newimage`` panels: 11-12 campaign families and
15-37 runs each between 2026-06-06 and 2026-07-06, with a stable 5-region
merge result. Optional complex cases can be added with ``--include-optional``.

    python scripts/dev/build_best_examples.py
    python scripts/dev/build_best_examples.py --include-optional
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from geoseg.core.image_ops import create_overlay

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS = Path.home() / "geoseg-artifacts" / "2026-10" / "runs"
OUT = REPO_ROOT / "validation" / "best"

CORE = [
    {
        "panel_id": "newimage_p0",
        "labels": "merge_experiment/panel_0/labels.npz",
        "image": "visual_audit_merge_experiment/panel_0/original_panel.jpg",
        "overlay": "visual_audit_merge_experiment/panel_0/overlay.jpg",
        "date": "2026-07-06",
        "campaigns": 11,
        "runs": 15,
        "note": "5 regions; stable merge result after 11 campaigns",
    },
    {
        "panel_id": "newimage_p1",
        "labels": "merge_experiment/panel_1/labels.npz",
        "image": "visual_audit_merge_experiment/panel_1/original_panel.jpg",
        "overlay": "visual_audit_merge_experiment/panel_1/overlay.jpg",
        "date": "2026-07-06",
        "campaigns": 12,
        "runs": 27,
        "note": "5 regions; stable merge result after 12 campaigns",
    },
    {
        "panel_id": "newimage_p2",
        "labels": "merge_experiment/panel_2/labels.npz",
        "image": "visual_audit_merge_experiment/panel_2/original_panel.jpg",
        "overlay": "visual_audit_merge_experiment/panel_2/overlay.jpg",
        "date": "2026-07-06",
        "campaigns": 12,
        "runs": 37,
        "note": "5 regions; most-iterated newimage panel",
    },
    {
        "panel_id": "newimage_p3",
        "labels": "merge_experiment/panel_3/labels.npz",
        "image": "visual_audit_merge_experiment/panel_3/original_panel.jpg",
        "overlay": "visual_audit_merge_experiment/panel_3/overlay.jpg",
        "date": "2026-07-06",
        "campaigns": 12,
        "runs": 17,
        "note": "5 regions; stable merge result",
    },
    {
        "panel_id": "newimage_p4",
        "labels": "merge_experiment/panel_4/labels.npz",
        "image": "visual_audit_merge_experiment/panel_4/original_panel.jpg",
        "overlay": "visual_audit_merge_experiment/panel_4/overlay.jpg",
        "date": "2026-07-06",
        "campaigns": 11,
        "runs": 15,
        "note": "5 regions; stable merge result",
    },
]

OPTIONAL = [
    {
        "panel_id": "c11b8db_edge_guided",
        "labels": "readme_examples_v2/gras2019_c11b8db/labels_edge_guided.npz",
        "image": "readme_examples_v2/gras2019_c11b8db/panel_cropped.png",
        "overlay": "readme_examples_v2/gras2019_c11b8db/overlay_edge_guided_blend.png",
        "date": "2026-05-27",
        "campaigns": 4,
        "runs": 4,
        "note": "clean 5-layer section; older but stable; white gap band",
    },
    {
        "panel_id": "jtecto_fig7",
        "labels": "j_tecto_2019_06_024/fig7/final/labels.npz",
        "image": "j_tecto_2019_06_024/fig7/01_panel.png",
        "overlay": "j_tecto_2019_06_024/fig7/final/labels.npz",
        "date": "2026-07-23",
        "campaigns": 5,
        "runs": 5,
        "note": "latest project; 13 zones, complex real section",
    },
    {
        "panel_id": "jtecto_fig6b",
        "labels": "j_tecto_2019_06_024/fig6b/final/labels.npz",
        "image": "j_tecto_2019_06_024/fig6b/01_panel.png",
        "overlay": None,
        "date": "2026-07-21",
        "campaigns": 3,
        "runs": 3,
        "note": "latest project; 4 zones, thin section",
    },
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def overlay_image(image: np.ndarray, labels: np.ndarray, convention: str) -> np.ndarray:
    if convention == "negative_is_background":
        labels = labels.copy()
        labels[labels < 0] = 0
        skip = True
    else:
        skip = convention != "zero_is_region"
    return create_overlay(image, labels, None, alpha=0.62, boundary_mode="thin", skip_background=skip)


def build(items: list[dict], out: Path) -> list[dict]:
    for sub in ("images", "labels", "overlays"):
        shutil.rmtree(out / sub, ignore_errors=True)
        (out / sub).mkdir(parents=True, exist_ok=True)

    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    manifest, tiles = [], []
    for item in items:
        labels_path = RUNS / item["labels"]
        image_path = RUNS / item["image"]
        labels = np.load(labels_path)["labels"].astype(np.int32)
        image = np.array(Image.open(image_path).convert("RGB"))
        if labels.shape != image.shape[:2]:
            raise ValueError(f"{item['panel_id']}: {labels.shape} != {image.shape[:2]}")
        convention = "negative_is_background" if item["panel_id"].startswith("jtecto") else "zero_is_background"
        out_labels = out / "labels" / f"{item['panel_id']}.npz"
        out_image = out / "images" / f"{item['panel_id']}.jpg"
        np.savez_compressed(out_labels, labels=labels)
        Image.fromarray(image).save(out_image, quality=92)

        ov = overlay_image(image, labels, convention)
        ov_path = out / "overlays" / f"{item['panel_id']}.jpg"
        Image.fromarray(ov).save(ov_path, quality=90)

        manifest.append({
            **item,
            "convention": convention,
            "n_labels": len(set(np.unique(labels).tolist())),
            "shape": list(labels.shape),
            "files": {
                "image": f"images/{item['panel_id']}.jpg",
                "labels": f"labels/{item['panel_id']}.npz",
                "overlay": f"overlays/{item['panel_id']}.jpg",
                "image_sha256": sha256(out_image),
                "labels_sha256": sha256(out_labels),
            },
        })

        tile = Image.fromarray(ov)
        w, h = tile.size
        scale = min(560 / w, 300 / h)
        tile = tile.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        tiles.append((item, tile))

    cols, tw, th = 3, 580, 300
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * (th + 42)), (12, 12, 12))
    for i, (item, tile) in enumerate(tiles):
        c, r = i % cols, i // cols
        ox, oy = c * tw, r * (th + 42)
        sheet.paste(tile, (ox, oy + 42))
        d = ImageDraw.Draw(sheet)
        d.text((ox + 8, oy + 5), f"{item['panel_id']}  {item['date']}", font=font, fill=(255, 255, 255))
        d.text(
            (ox + 8, oy + 23),
            f"{item['campaigns']} campaigns / {item['runs']} runs  |  {item['note']}",
            font=small,
            fill=(150, 220, 255),
        )
    sheet.save(out / "contact_sheet.jpg", quality=92)

    (out / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in manifest) + "\n",
        encoding="utf-8",
    )
    lines = ["# Best examples", ""]
    for r in manifest:
        lines.append(
            f"- `{r['panel_id']}` — {r['date']}, {r['campaigns']} campaigns / "
            f"{r['runs']} runs, n={r['n_labels']}, {r['note']}"
        )
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="build validation/best")
    parser.add_argument("--include-optional", action="store_true")
    args = parser.parse_args(argv)

    items = CORE + (OPTIONAL if args.include_optional else [])
    manifest = build(items, OUT)
    print(f"wrote {OUT} ({len(manifest)} panels)")
    for r in manifest:
        print(f"  {r['panel_id']:26s} n={r['n_labels']:2d} {r['shape']} {r['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
