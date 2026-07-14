#!/usr/bin/env python3
"""Generate vivid overlays for target panels from velocity_model figures.

Uses the segmentation stage so that figure classification,
target-panel filtering, and colorbar cropping are all applied.
Only the VLM-identified target panel (or the largest panel as fallback)
is processed per figure, enforcing one figure = one model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoseg.pipeline.segment import run_segmentation_stage

RESULTS_FILE = Path("runs/new_papers_vlm/pipeline_results.json")
OUT_DIR = Path("runs/new_papers_vlm/all_overlays")


def vivid_color(rgb: np.ndarray, sat_boost: float = 0.45, val_boost: float = 0.15) -> np.ndarray:
    from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
    rgb_norm = rgb.astype(float) / 255.0
    hsv = rgb_to_hsv(rgb_norm.reshape(1, 1, 3)).reshape(3)
    hsv[1] = min(1.0, hsv[1] + sat_boost)
    hsv[2] = min(1.0, hsv[2] + val_boost)
    vivid_rgb = hsv_to_rgb(hsv.reshape(1, 1, 3)).reshape(3)
    return (vivid_rgb * 255).astype(np.uint8)


def create_vivid_overlay(original: np.ndarray, labels: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    from scipy import ndimage
    h, w = labels.shape
    n_layers = int(labels.max())
    vivid_colors = []
    for lbl in range(1, n_layers + 1):
        mask = labels == lbl
        if mask.any():
            mean_color = original[mask].mean(axis=0)
            vivid = vivid_color(mean_color, sat_boost=0.45, val_boost=0.15)
            vivid_colors.append(vivid)
        else:
            vivid_colors.append(np.array([200, 200, 200], dtype=np.uint8))
    colored = np.zeros_like(original)
    for lbl in range(1, n_layers + 1):
        mask = labels == lbl
        if mask.any():
            colored[mask] = vivid_colors[lbl - 1]
    blended = (original.astype(float) * (1 - alpha) + colored.astype(float) * alpha).astype(np.uint8)
    boundaries = np.zeros((h, w), dtype=bool)
    for lbl in range(1, n_layers + 1):
        mask = labels == lbl
        if mask.any():
            eroded = ndimage.binary_erosion(mask)
            boundaries |= (mask & ~eroded)
    boundaries = ndimage.binary_dilation(boundaries, iterations=1)
    blended[boundaries] = [255, 255, 255]
    return blended


def compose_side_by_side(left: np.ndarray, right: np.ndarray, gap: int = 20, bg_color=(40, 40, 40)) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont
    h1, w1 = left.shape[:2]
    h2, w2 = right.shape[:2]
    h = max(h1, h2)
    w = w1 + gap + w2
    canvas = np.full((h, w, 3), bg_color, dtype=np.uint8)
    y1 = (h - h1) // 2
    y2 = (h - h2) // 2
    canvas[y1:y1+h1, :w1] = left
    canvas[y2:y2+h2, w1+gap:] = right
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", max(14, h // 40))
    except Exception:
        font = ImageFont.load_default()
    draw.text((10, 10), "ORIGINAL", fill=(255, 255, 255), font=font)
    draw.text((w1 + gap + 10, 10), "SEGMENTATION", fill=(255, 255, 255), font=font)
    return np.array(pil)


def main() -> int:
    pipeline = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    vm_figures = [
        r for r in pipeline
        if r.get("vlm_type") == "velocity_model" and r.get("status") == "ok"
    ]

    print(f"Processing {len(vm_figures)} velocity_model figures")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for fig_idx, record in enumerate(vm_figures, 1):
        fig_key = record["fig_key"]
        img_path = Path(record.get("img_path", ""))
        if not img_path.exists():
            print(f"  [{fig_idx}/{len(vm_figures)}] {fig_key}: image not found, skip")
            continue

        print(f"\n[{fig_idx}/{len(vm_figures)}] {fig_key}")
        img_rgb = np.array(Image.open(img_path).convert("RGB"))

        # Run through full pipeline (skip VLM to save cost; rely on JSON filter).
        # skip_non_velocity_model=False because we already filtered by vlm_type.
        vlm_target_id = record.get("vlm_target_panel_id", -1)
        n_layers = max(2, record.get("total_layers", 5))
        seg_result = run_segmentation_stage(
            img_rgb,
            caption="",
            text_blocks=[],
            n_layers=n_layers,
            quality_preference="balanced",
            skip_non_velocity_model=False,
            use_vlm=False,
            target_panel_id=vlm_target_id,
        )

        if seg_result["summary"]["status"] == "skipped":
            print(f"  skipped: {seg_result['summary'].get('reason', '')}")
            continue

        # Pick the panel that was actually segmented.
        # If the VLM target_id doesn't match any CV-detected panel (stale data),
        # fall back to the panel with the most layers.
        target_panel = None
        panels_with_seg = [
            p for p in seg_result["panels"]
            if p.get("segmentation") is not None
        ]
        if panels_with_seg:
            target_panel = max(
                panels_with_seg,
                key=lambda p: p["review"].get("n_layers_found", 0),
            )
        else:
            # target_panel_id mismatched: no panel was segmented.
            # Re-run without target filtering and pick the best panel.
            seg_result = run_segmentation_stage(
                img_rgb,
                caption="",
                text_blocks=[],
                n_layers=n_layers,
                quality_preference="balanced",
                skip_non_velocity_model=False,
                use_vlm=False,
                target_panel_id=-1,
            )
            panels_with_seg = [
                p for p in seg_result["panels"]
                if p.get("segmentation") is not None
            ]
            if panels_with_seg:
                target_panel = max(
                    panels_with_seg,
                    key=lambda p: p["review"].get("n_layers_found", 0),
                )
                print(f"  target_id mismatch fallback")

        if target_panel is None:
            print(f"  no panel with segmentation")
            continue

        seg = target_panel["segmentation"]
        labels = seg["labels"]
        n_found = len(set(labels.flatten()) - {0})
        if n_found < 2:
            print(f"  panel {target_panel['panel_id']}: only {n_found} layers, skip")
            continue

        # Use the cropped panel image from the segmentation stage if available,
        # otherwise fall back to original bbox crop.
        x, y, pw, ph = target_panel["bbox"]
        panel_img = img_rgb[y:y+ph, x:x+pw]

        overlay = create_vivid_overlay(panel_img, labels)
        composed = compose_side_by_side(panel_img, overlay)

        out_name = f"{fig_key.replace('/', '_')}_panel{target_panel['panel_id']}_{n_found}layers.png"
        out_path = OUT_DIR / out_name
        Image.fromarray(composed).save(out_path)

        print(f"  Panel {target_panel['panel_id']}: {n_found} layers ({seg['meta']['engine']}) -> {out_name}")
        results.append({
            "fig_key": fig_key,
            "panel_id": target_panel["panel_id"],
            "file": out_name,
            "n_found": n_found,
            "engine": seg["meta"]["engine"],
            "path": str(out_path),
        })

    report_path = OUT_DIR / "report.json"
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nDone. {len(results)} overlays saved to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
