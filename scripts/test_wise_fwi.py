#!/usr/bin/env python3
"""Quick test of wise_fwi_2024 page5_img2 segmentation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoseg.modules.segment_engines.vlm_reps import vertical_scan_reps
from geoseg.modules.segment_engines import route_and_segment

IMG_PATH = Path("papers_new/to_process/wise_fwi_2024/wise_fwi_2024_page5_img2.png")
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


def main() -> None:
    print(f"Loading {IMG_PATH}")
    img_rgb = np.array(Image.open(IMG_PATH).convert("RGB"))
    print(f"Image shape: {img_rgb.shape}")

    # Resize if too large (vertical_scan on 10k x 6k is slow)
    h, w = img_rgb.shape[:2]
    max_dim = 2000
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_rgb = np.array(Image.fromarray(img_rgb).resize((new_w, new_h), Image.LANCZOS))
        print(f"Resized to {img_rgb.shape}")

    reps = vertical_scan_reps(img_rgb, n_layers_hint=5)
    print(f"Reps: {len(reps)}")

    if len(reps) < 2:
        print("SKIP: insufficient reps")
        return

    seg = route_and_segment(
        img_rgb,
        reps=reps,
        n_layers=len(reps),
        quality_preference="balanced",
        is_velocity_model=True,
        retry_on_underseg=True,
    )
    labels = seg["labels"]
    n_found = len(set(labels.flatten()) - {0})
    print(f"Segmented: {n_found} layers, engine={seg['meta']['engine']}")

    overlay = create_vivid_overlay(img_rgb, labels)
    composed = compose_side_by_side(img_rgb, overlay)

    out_name = "wise_fwi_2024_wise_fwi_2024_page5_img2_panel0_resized.png"
    out_path = OUT_DIR / out_name
    Image.fromarray(composed).save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
