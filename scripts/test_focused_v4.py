#!/usr/bin/env python3
"""Focused test using v4_kmeans directly and similar candidates to successes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoseg.modules.segment_engines.router import _run_engine
from geoseg.modules.vlm_client.client import review_segmentation_quality
from geoseg.modules.cv_detect.panel_detector import detect_panels

OUT_DIR = Path("runs/new_papers_vlm/all_overlays")
AUDIT_DIR = Path("runs/new_papers_vlm/quality_review_audit_v4focus")


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


def process_v4(img_path: Path, desc: str, n_layers: int) -> dict | None:
    print(f"\n{'='*60}")
    print(f"Processing {desc} with v4_kmeans n_layers={n_layers}")
    print(f"{'='*60}")

    img_rgb = np.array(Image.open(img_path).convert("RGB"))
    print(f"Image shape: {img_rgb.shape}")

    panel_bboxes = detect_panels(img_rgb)
    if not panel_bboxes:
        h, w = img_rgb.shape[:2]
        panel_bboxes = [{"id": 0, "bbox": [0, 0, w, h]}]

    panel_bboxes.sort(key=lambda pb: (pb["bbox"][1], pb["bbox"][0]))
    pb = panel_bboxes[0]
    x, y, pw, ph = pb["bbox"]
    panel_img = img_rgb[y:y+ph, x:x+pw]

    if pw < 50 or ph < 50:
        return None

    print(f"  [Panel {pb['id']}] shape={panel_img.shape}")

    # Use v4_kmeans without reps (falls through to pastel_faded or colorbar_guided)
    seg = _run_engine("v4_kmeans", panel_img, reps=None, colorbar_rgb=None, n_layers=n_layers)
    labels = seg["labels"]
    n_found = len(set(labels.flatten()) - {0})
    print(f"  Segmented: {n_found} layers, engine={seg['meta']['engine']} path={seg['meta'].get('path', 'unknown')}")

    overlay = create_vivid_overlay(panel_img, labels)
    composed = compose_side_by_side(panel_img, overlay)

    out_name = f"{desc.replace(' ', '_').replace('/', '_')}_v4_{n_found}l.png"
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
        return {
            "desc": desc,
            "n_layers": n_layers,
            "score": review.overall_score,
            "recommendation": review.recommendation,
            "n_expected": review.n_layers_expected,
            "n_found": review.n_layers_found,
            "file": out_name,
        }
    except Exception as exc:
        print(f"  -> ERROR: {exc}")
        return {
            "desc": desc,
            "n_layers": n_layers,
            "error": str(exc),
            "file": out_name,
        }


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    # fwi_deeponet page1_img3 - very similar to 0.88 accepts
    result = process_v4(Path("papers_new/to_process/fwi_deeponet_2024/fwi_deeponet_2024_page1_img3.png"), "fwi_deeponet p1i3", 4)
    if result:
        all_results.append(result)

    # tomography_review page5_img2 - try v4_kmeans with 4-5 layers instead of edge_guided 8
    for n in [4, 5]:
        result = process_v4(Path("papers_new/to_process/tomography_review_2024/tomography_review_2024_page5_img2.png"), f"tomography p5i2", n)
        if result:
            all_results.append(result)

    # uncertainty_fwi page5_img4 - larger panel, try v4_kmeans
    for n in [4, 5]:
        result = process_v4(Path("papers_new/to_process/uncertainty_fwi_2024/uncertainty_fwi_2024_page5_img4.png"), f"uncertainty p5i4", n)
        if result:
            all_results.append(result)

    # ml_velocity page12_img0 - try v4_kmeans with 5-7 layers
    for n in [5, 6, 7]:
        result = process_v4(Path("papers_new/to_process/ml_velocity_2024/ml_velocity_2024_page12_img0.png"), f"ml_velocity p12i0", n)
        if result:
            all_results.append(result)

    report_path = AUDIT_DIR / "v4focus_report.json"
    report_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    print(f"\n{'='*60}")
    print(f"Report: {report_path}")

    ok = [r for r in all_results if r.get("recommendation") == "accept"]
    manual = [r for r in all_results if r.get("recommendation") == "manual_fix"]
    errors = [r for r in all_results if "error" in r]
    print(f"accept={len(ok)} manual_fix={len(manual)} errors={len(errors)}")
    for r in all_results:
        if "score" in r:
            print(f"  {r['desc']} n={r['n_layers']}: score={r['score']:.2f} rec={r['recommendation']}")
        else:
            print(f"  {r['desc']} n={r['n_layers']}: ERROR")


if __name__ == "__main__":
    main()
