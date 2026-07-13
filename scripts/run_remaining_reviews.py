#!/usr/bin/env python3
"""Re-run quality review for timeout cases + optimize ml_velocity segmentation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoseg.modules.vlm_client.client import review_segmentation_quality
from geoseg.modules.segment_engines.vlm_reps import vertical_scan_reps
from geoseg.modules.segment_engines.router import route_and_segment

OUT_DIR = Path("runs/new_papers_vlm/all_overlays")
AUDIT_DIR = Path("runs/new_papers_vlm/quality_review_audit_v2")


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


def review_image(img_path: Path, desc: str) -> dict:
    img_rgb = np.array(Image.open(img_path).convert("RGB"))
    try:
        review = review_segmentation_quality(
            img_rgb,
            audit_dir=AUDIT_DIR,
            mode="auto",
            min_confidence=0.5,
        )
        return {
            "desc": desc,
            "score": review.overall_score,
            "recommendation": review.recommendation,
            "n_expected": review.n_layers_expected,
            "n_found": review.n_layers_found,
            "over_seg": review.over_segmentation,
            "under_seg": review.under_segmentation,
        }
    except Exception as exc:
        return {"desc": desc, "error": str(exc)}


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    # 1. Re-run timeout cases
    timeout_cases = [
        ("tomography_review_2024_tomography_review_2024_page5_img2_panel2_9layers.png", "tomography_review page5_img2 panel2"),
        ("wise_fwi_2024_wise_fwi_2024_page5_img2_panel0_resized.png", "wise_fwi page5_img2 panel0"),
    ]

    for filename, desc in timeout_cases:
        img_path = OUT_DIR / filename
        if not img_path.exists():
            print(f"SKIP: {img_path}")
            continue
        print(f"\nReviewing {desc} ...")
        result = review_image(img_path, desc)
        print(f"  -> {result}")
        results.append(result)

    # 2. Re-segment ml_velocity with lower n_layers_hint to reduce over-seg
    print("\n\nRe-segmenting ml_velocity page22_img2 with n_layers_hint=4...")
    img_path = Path("papers_new/to_process/ml_velocity_2024/ml_velocity_2024_page22_img2.png")
    img_rgb = np.array(Image.open(img_path).convert("RGB"))
    reps = vertical_scan_reps(img_rgb, n_layers_hint=4)
    print(f"  Reps: {len(reps)}")

    if len(reps) >= 2:
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
        print(f"  Segmented: {n_found} layers, engine={seg['meta']['engine']}")

        overlay = create_vivid_overlay(img_rgb, labels)
        composed = compose_side_by_side(img_rgb, overlay)

        out_name = "ml_velocity_2024_ml_velocity_2024_page22_img2_panel0_v2_4hint.png"
        out_path = OUT_DIR / out_name
        Image.fromarray(composed).save(out_path)
        print(f"  Saved: {out_path}")

        print(f"\nReviewing ml_velocity page22_img2 v2 ...")
        result = review_image(out_path, "ml_velocity page22_img2 v2 (4hint)")
        print(f"  -> {result}")
        results.append(result)

    # Save report
    report_path = AUDIT_DIR / "v2_report.json"
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
