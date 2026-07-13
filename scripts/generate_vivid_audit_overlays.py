#!/usr/bin/env python3
"""Generate vivid overlay + side-by-side audit images using e026_algo style.

For each target with >=2 layers:
1. Extract each panel's original image
2. Create vivid overlay (boost sat + val in HSV, white boundaries)
3. Compose original + overlay side-by-side into one large image
4. Save to runs/new_papers_vlm/vivid_audit/

This produces images suitable for VLM segmentation quality review.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoseg.modules.segment_engines.full_pipeline import process_figure

RESULTS_FILE = Path("runs/new_papers_vlm/pipeline_results.json")
RETRY_FILE = Path("runs/new_papers_vlm/retry_results.json")
OUT_DIR = Path("runs/new_papers_vlm/vivid_audit")


def vivid_color(rgb: np.ndarray, sat_boost: float = 0.45, val_boost: float = 0.15) -> np.ndarray:
    """Boost saturation and value in HSV space."""
    rgb_norm = rgb.astype(float) / 255.0
    hsv = rgb_to_hsv(rgb_norm.reshape(1, 1, 3)).reshape(3)
    hsv[1] = min(1.0, hsv[1] + sat_boost)
    hsv[2] = min(1.0, hsv[2] + val_boost)
    vivid_rgb = hsv_to_rgb(hsv.reshape(1, 1, 3)).reshape(3)
    return (vivid_rgb * 255).astype(np.uint8)


def create_vivid_overlay(original: np.ndarray, labels: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Create vivid overlay: boost each region's color + white boundaries."""
    h, w = labels.shape
    n_layers = int(labels.max())

    # Compute vivid color for each layer from original image
    vivid_colors = []
    for lbl in range(1, n_layers + 1):
        mask = labels == lbl
        if mask.any():
            mean_color = original[mask].mean(axis=0)
            vivid = vivid_color(mean_color, sat_boost=0.45, val_boost=0.15)
            vivid_colors.append(vivid)
        else:
            vivid_colors.append(np.array([200, 200, 200], dtype=np.uint8))

    # Build colored mask
    colored = np.zeros_like(original)
    for lbl in range(1, n_layers + 1):
        mask = labels == lbl
        if mask.any():
            colored[mask] = vivid_colors[lbl - 1]

    # Blend with original
    blended = (original.astype(float) * (1 - alpha) + colored.astype(float) * alpha).astype(np.uint8)

    # White boundaries
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
    """Compose two images side by side with gap and background."""
    h1, w1 = left.shape[:2]
    h2, w2 = right.shape[:2]
    h = max(h1, h2)
    w = w1 + gap + w2

    canvas = np.full((h, w, 3), bg_color, dtype=np.uint8)

    # Center vertically
    y1 = (h - h1) // 2
    y2 = (h - h2) // 2
    canvas[y1:y1+h1, :w1] = left
    canvas[y2:y2+h2, w1+gap:] = right

    # Add labels
    from PIL import Image, ImageDraw, ImageFont
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", max(14, h // 40))
    except:
        font = ImageFont.load_default()

    draw.text((10, 10), "ORIGINAL", fill=(255, 255, 255), font=font)
    draw.text((w1 + gap + 10, 10), "SEGMENTATION", fill=(255, 255, 255), font=font)

    return np.array(pil)


def main() -> None:
    pipeline = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    retry = json.loads(RETRY_FILE.read_text(encoding="utf-8")) if RETRY_FILE.exists() else []

    # Use best result per fig_key
    best = {r["fig_key"]: r for r in pipeline}
    for r in retry:
        orig = r.get("original_fig_key", r["fig_key"])
        if orig not in best or r.get("total_layers", 0) > best[orig].get("total_layers", 0):
            best[orig] = r

    targets = [r for r in best.values() if r.get("total_layers", 0) >= 2]
    print(f"Generating vivid audit images for {len(targets)} targets\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for i, target in enumerate(targets, 1):
        fig_key = target["fig_key"]
        paper = target.get("paper", "unknown")

        # Find image path
        img_path = target.get("img_path", "")
        if not img_path or img_path == "N/A":
            p = [x for x in pipeline if x["fig_key"] == fig_key]
            if p:
                img_path = p[0].get("img_path", "")
        if not img_path:
            print(f"[{i}/{len(targets)}] SKIP {fig_key}: no image path")
            continue

        img_path = Path(img_path)
        if not img_path.exists():
            print(f"[{i}/{len(targets)}] SKIP {fig_key}: image not found")
            continue

        print(f"[{i}/{len(targets)}] {fig_key}")

        # Run pipeline
        img_rgb = np.array(Image.open(img_path).convert("RGB"))
        result = process_figure(
            img_rgb,
            caption="",
            n_layers=target.get("total_layers", 5),
            quality_preference="balanced",
            skip_non_velocity_model=True,
            use_vlm=True,
        )

        # For each panel with segmentation, generate side-by-side
        for panel in result.get("panels", []):
            if panel["segmentation"] is None:
                continue

            seg = panel["segmentation"]
            labels = seg["labels"]
            panel_id = panel["panel_id"]
            x, y, pw, ph = panel["bbox"]
            n_layers_found = panel["review"].get("n_layers_found", 0)

            if n_layers_found < 2:
                continue

            panel_img = img_rgb[y:y+ph, x:x+pw]

            # Resize labels if needed
            if labels.shape[:2] != panel_img.shape[:2]:
                labels_pil = Image.fromarray(labels.astype(np.uint8))
                labels_pil = labels_pil.resize((pw, ph), Image.NEAREST)
                labels = np.array(labels_pil)

            # Create vivid overlay
            overlay = create_vivid_overlay(panel_img, labels)

            # Compose side-by-side
            composed = compose_side_by_side(panel_img, overlay)

            # Save
            out_name = f"{fig_key.replace('/', '_')}_panel{panel_id}_{n_layers_found}layers.png"
            out_path = OUT_DIR / out_name
            Image.fromarray(composed).save(out_path)
            print(f"  -> {out_path} ({n_layers_found} layers)")

    print(f"\nDone. Audit images in: {OUT_DIR}")


if __name__ == "__main__":
    main()
