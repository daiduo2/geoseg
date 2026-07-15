#!/usr/bin/env python3
"""Test seismic_inv_2024 page9_img1 — it IS a color figure (low saturation but visible color zones)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoseg.experiments import vertical_scan_reps
from geoseg.experiments import route_and_segment
from geoseg.experiments import detect_panels
from geoseg.experiments import review_segmentation_quality

IMG_PATH = Path("papers_new/to_process/seismic_inv_2024/seismic_inv_2024_page9_img1.png")
OUT_DIR = Path("runs/new_papers_vlm/all_overlays")
AUDIT_DIR = Path("runs/new_papers_vlm/quality_review_audit_seismic")


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

    # Check saturation manually
    from matplotlib.colors import rgb_to_hsv
    hsv = rgb_to_hsv(img_rgb.astype(float) / 255.0)
    print(f"HSV saturation: min={hsv[:,:,1].min():.4f} max={hsv[:,:,1].max():.4f} mean={hsv[:,:,1].mean():.4f} median={np.median(hsv[:,:,1]):.4f}")

    panel_bboxes = detect_panels(img_rgb)
    print(f"Detected {len(panel_bboxes)} panels")

    if not panel_bboxes:
        print("No panels detected, using whole image")
        h, w = img_rgb.shape[:2]
        panel_bboxes = [{"id": 0, "bbox": [0, 0, w, h]}]

    panel_bboxes.sort(key=lambda pb: (pb["bbox"][1], pb["bbox"][0]))

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for pb in panel_bboxes:
        x, y, pw, ph = pb["bbox"]
        panel_id = pb["id"]
        panel_img = img_rgb[y:y+ph, x:x+pw]

        if pw < 50 or ph < 50:
            continue

        print(f"\n[Panel {panel_id}] shape={panel_img.shape}")

        reps = vertical_scan_reps(panel_img, n_layers_hint=5)
        print(f"  vertical_scan reps: {len(reps)}")

        if len(reps) < 2:
            print(f"  SKIP: insufficient reps")
            continue

        seg = route_and_segment(
            panel_img,
            reps=reps,
            n_layers=len(reps),
            quality_preference="balanced",
            is_velocity_model=True,
            retry_on_underseg=True,
        )
        labels = seg["labels"]
        n_found = len(set(labels.flatten()) - {0})
        print(f"  Segmented: {n_found} layers, engine={seg['meta']['engine']}")

        overlay = create_vivid_overlay(panel_img, labels)
        composed = compose_side_by_side(panel_img, overlay)

        out_name = f"seismic_inv_page9_img1_panel{panel_id}_{n_found}layers.png"
        out_path = OUT_DIR / out_name
        Image.fromarray(composed).save(out_path)
        print(f"  Saved: {out_path}")

        print(f"  Running VLM quality review...")
        try:
            review = review_segmentation_quality(
                composed,
                audit_dir=AUDIT_DIR,
                mode="auto",
                min_confidence=0.5,
            )
            print(f"  -> score={review.overall_score:.2f} rec={review.recommendation}")
            results.append({
                "panel_id": panel_id,
                "score": review.overall_score,
                "recommendation": review.recommendation,
                "n_expected": review.n_layers_expected,
                "n_found": review.n_layers_found,
                "file": out_name,
            })
        except Exception as exc:
            print(f"  -> ERROR: {exc}")
            results.append({"panel_id": panel_id, "error": str(exc), "file": out_name})

    report_path = AUDIT_DIR / "report.json"
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
