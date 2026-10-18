#!/usr/bin/env python3
"""Build validation/curated from human-curated session results.

These four panels were iterated with explicit user corrections in the Codex
sessions (2026-07-16/17) and then fixed as final results:

- paper22 fig5a: 8 density zones with real rho values
- paper22 fig6:  5 boundary-driven crustal units on the rectified panel
- paper23 fig7a:  fault-guided section, 11 intervention history entries
- paper23 fault_guided: fault-bounded section, legend absorbed, 14 regions

    python scripts/dev/build_curated_examples.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from geoseg.core.image_ops import create_overlay
from geoseg.modules.post_process.properties import generate_properties_for_layers
from geoseg.pipeline.export import run_post_process_and_export

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS = Path.home() / "geoseg-artifacts" / "2026-10" / "runs"
OUT = REPO_ROOT / "validation" / "curated"
EXPORT_ROOT = Path.home() / "geoseg-artifacts" / "2026-10" / "curated_exports"

ITEMS = [
    {
        "panel_id": "paper22_fig5a",
        "labels": "paper22/final/fig5a/labels.npz",
        "image": "paper22/inputs/fig5a_density_model.png",
        "overlay": "paper22/final/fig5a/overlay.png",
        "group": "paper22",
        "structure": "layered",
        "session_evidence": (
            "Final version after OCR ROI masking, residual classification, "
            "label_blur sigma=2.5 and targeted 2.91/3.3 boundary tracing; "
            "user asked for the boundary correction and accepted the final."
        ),
        "density": {"1": 2.65, "2": 2.81, "3": 2.83, "4": 2.85, "5": 2.91, "6": 3.01, "7": 3.3, "8": 3.5},
    },
    {
        "panel_id": "paper22_fig6",
        "labels": "paper22/final/fig6/labels.npz",
        "image": "paper22/inputs/fig6_rectified_front.png",
        "overlay": "paper22/final/fig6/overlay_rectified.png",
        "group": "paper22",
        "structure": "boundary_driven",
        "session_evidence": (
            "User: fig6 boundaries come from short black/red barriers, not color; "
            "the boundary-driven watershed result is the primary output. "
            "Boundary alignment 0.9306, fragment_count 0."
        ),
        "semantics": {
            "1": "western upper crust",
            "2": "western lower crust and central underthrust region",
            "3": "eastern upper crust",
            "4": "eastern lower crust",
            "5": "eclogitized root",
        },
    },
    {
        "panel_id": "paper23_fig7a",
        "labels": "sandbox/paper23_fig7a_clear/reviewed/secondary_labels.npz",
        "image": "sandbox/paper23_fig7a_clear/fig7a_crop.png",
        "overlay": "sandbox/paper23_fig7a_clear/reviewed/secondary_overlay.png",
        "group": "paper23",
        "structure": "faulted",
        "session_evidence": (
            "11-step intervention history: legend absorption, F3 right extension, "
            "valley-vs-interior correction, final reviewed label set."
        ),
    },
    {
        "panel_id": "schematic_panel1",
        "labels": "3d_schematic_retune/best/panel_1_best_labels.npz",
        "image": "3d_schematic_retune/panel_1/text_removed_best.png",
        "overlay": "3d_schematic_retune/best/panel_1_best_overlay.jpg",
        "group": "3d_schematic",
        "structure": "schematic_3d",
        "export": False,
        "overlay_existing": True,
        "session_evidence": (
            "2026-06-19 user accepted panel2 and asked to rerun panel1 with the "
            "same workflow; this is the retuned panel1 result."
        ),
    },
    {
        "panel_id": "schematic_panel2",
        "labels": "3d_schematic_retune/best/panel_2_best_labels.npy",
        "image": "3d_schematic_retune/panel_2/text_removed_best.png",
        "overlay": "3d_schematic_retune/best/panel_2_best_overlay.jpg",
        "group": "3d_schematic",
        "structure": "schematic_3d",
        "export": False,
        "overlay_existing": True,
        "session_evidence": (
            "User: '2026-06-19 我接受,你先把当前的 panel2 归档到此前的最佳 overlay 目录下'."
        ),
    },
    {
        "panel_id": "schematic_panel3_front",
        "labels": "3d_schematic_correct_e2e/panel_3_front/labels_fused.npz",
        "image": "3d_schematic_correct_e2e/panel_3_front/00_cleaned.jpg",
        "overlay": "3d_schematic_correct_e2e/panel_3_front/03_fused.jpg",
        "group": "3d_schematic",
        "structure": "schematic_3d_front",
        "export": False,
        "overlay_existing": True,
        "session_evidence": (
            "2026-06-19 user: panel3 '当前的效果远远优于此前任何一次'; this is the "
            "front-face fused result from the corrected e2e run."
        ),
    },
    {
        "panel_id": "paper23_fault_guided",
        "labels": "sandbox/paper23_fault_guided/secondary_labels.npz",
        "image": "sandbox/paper23_connected/cropped_profile.png",
        "overlay": "sandbox/paper23_fault_guided/secondary_overlay.png",
        "group": "paper23",
        "structure": "faulted",
        "session_evidence": (
            "User corrected legend absorption and three missing boundaries; "
            "final uses color labels + long red barriers + tangent completion."
        ),
    },
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def overlay(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return create_overlay(
        image, labels, None, alpha=0.62, boundary_mode="thin", skip_background=True
    )


def export_panel(labels: np.ndarray, item: dict) -> dict:
    color_names = [f"layer_{i}" for i in sorted(set(int(v) for v in np.unique(labels)) - {0})]
    props = generate_properties_for_layers(color_names)
    if "density" in item:
        for i, name in enumerate(color_names, start=1):
            if str(i) in item["density"]:
                props[name]["rho"] = item["density"][str(i)] * 1000.0
    export_dir = EXPORT_ROOT / item["panel_id"]
    export_dir.mkdir(parents=True, exist_ok=True)
    result = run_post_process_and_export(
        labels,
        color_names=color_names,
        properties_map=props,
        output_dir=export_dir,
        save_intermediates=True,
    )
    small = OUT / "exports" / item["panel_id"]
    small.mkdir(parents=True, exist_ok=True)
    copied = {}
    for name in ("parfile_snippet.txt", "properties.json", "polygons.geojson"):
        src = export_dir / name
        if src.exists():
            shutil.copy2(src, small / name)
            copied[name] = f"exports/{item['panel_id']}/{name}"
    tomo = export_dir / "tomo.xyz"
    return {
        "status": result["status"],
        "n_components": result.get("n_components", 0),
        "n_polygons": result.get("n_polygons", 0),
        "properties_source": (
            "rho from paper density values; Vp/Vs placeholder"
            if "density" in item
            else "synthetic_gradient_placeholder"
        ),
        "tomo_xyz": str(tomo),
        "tomo_size_mb": round(tomo.stat().st_size / 1e6, 2) if tomo.exists() else 0.0,
        "files": copied,
    }


def build() -> list[dict]:
    for sub in ("images", "labels", "overlays", "exports"):
        shutil.rmtree(OUT / sub, ignore_errors=True)
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    manifest = []
    tiles = []
    for item in ITEMS:
        lp = RUNS / item["labels"]
        ip = RUNS / item["image"]
        labels = (np.load(lp)["labels"] if lp.suffix == ".npz" else np.load(lp)).astype(np.int32)
        image = np.array(Image.open(ip).convert("RGB"))
        if labels.shape != image.shape[:2]:
            raise ValueError(f"{item['panel_id']}: {labels.shape} != {image.shape[:2]}")

        out_labels = OUT / "labels" / f"{item['panel_id']}.npz"
        out_image = OUT / "images" / f"{item['panel_id']}.jpg"
        out_overlay = OUT / "overlays" / f"{item['panel_id']}.jpg"
        np.savez_compressed(out_labels, labels=labels)
        Image.fromarray(image).save(out_image, quality=92)
        if item.get("overlay_existing"):
            shutil.copy2(RUNS / item["overlay"], out_overlay)
            ov = np.array(Image.open(out_overlay).convert("RGB"))
        else:
            ov = overlay(image, labels)
            Image.fromarray(ov).save(out_overlay, quality=92)

        if item.get("export", True):
            export = export_panel(labels, item)
        else:
            export = {"status": "skipped", "reason": "schematic_3d", "n_components": 0, "n_polygons": 0, "properties_source": None, "tomo_xyz": None, "tomo_size_mb": 0.0, "files": {}}
        manifest.append({
            **item,
            "tier": "human",
            "convention": "zero_is_background",
            "n_labels": len(set(np.unique(labels).tolist())),
            "shape": list(labels.shape),
            "exported": export["status"] == "ok",
            "export": export,
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
        scale = min(700 / w, 360 / h)
        tiles.append((item, tile.resize((int(w * scale), int(h * scale)), Image.LANCZOS)))

    cols, tw, th = 2, 720, 360
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * (th + 46)), (12, 12, 12))
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    for i, (item, tile) in enumerate(tiles):
        c, r = i % cols, i // cols
        ox, oy = c * tw, r * (th + 46)
        sheet.paste(tile, (ox, oy + 46))
        d = ImageDraw.Draw(sheet)
        d.text((ox + 8, oy + 5), f"{item['panel_id']}  |  {item['group']} / {item['structure']}", font=font, fill=(255, 255, 255))
        d.text((ox + 8, oy + 25), item["session_evidence"][:110], font=small, fill=(150, 220, 255))
    sheet.save(OUT / "contact_sheet.jpg", quality=92)

    (OUT / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in manifest) + "\n",
        encoding="utf-8",
    )
    lines = ["# Curated session results", ""]
    for r in manifest:
        lines.append(
            f"- `{r['panel_id']}` — {r['structure']}, n={r['n_labels']}, "
            f"exported={r['exported']} ({r['export']['n_polygons']} polygons). {r['session_evidence']}"
        )
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    manifest = build()
    print(f"wrote {OUT} ({len(manifest)} panels)")
    for r in manifest:
        print(f"  {r['panel_id']:24s} {r['structure']:16s} n={r['n_labels']:2d} {r['shape']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
