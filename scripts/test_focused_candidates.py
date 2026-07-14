#!/usr/bin/env python3
"""Test focused candidate images for the 5-paper set."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoseg.modules.segment_engines.vlm_reps import vertical_scan_reps
from geoseg.modules.segment_engines import route_and_segment
from geoseg.modules.vlm_client.client import review_segmentation_quality
from geoseg.modules.cv_detect.panel_detector import detect_panels

OUT_DIR = Path("runs/new_papers_vlm/all_overlays")
AUDIT_DIR = Path("runs/new_papers_vlm/quality_review_audit_v4")


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


def process_and_review(img_path: Path, desc: str, n_hint: int = 5, target_panel_id: int | None = None) -> dict:
    print(f"\n{'='*60}")
    print(f"Processing {desc}")
    print(f"{'='*60}")

    img_rgb = np.array(Image.open(img_path).convert("RGB"))
    print(f"Image shape: {img_rgb.shape}")

    panel_bboxes = detect_panels(img_rgb)
    if not panel_bboxes:
        h, w = img_rgb.shape[:2]
        panel_bboxes = [{"id": 0, "bbox": [0, 0, w, h]}]

    panel_bboxes.sort(key=lambda pb: (pb["bbox"][1], pb["bbox"][0]))

    if target_panel_id is not None:
        target_panels = [pb for pb in panel_bboxes if pb["id"] == target_panel_id]
        if not target_panels:
            print(f"  Target panel {target_panel_id} not found, using first panel")
            target_panels = [panel_bboxes[0]]
    else:
        target_panels = [panel_bboxes[0]]

    results = []
    for pb in target_panels:
        x, y, pw, ph = pb["bbox"]
        panel_id = pb["id"]
        panel_img = img_rgb[y:y+ph, x:x+pw]

        if pw < 50 or ph < 50:
            continue

        print(f"\n  [Panel {panel_id}] shape={panel_img.shape}")

        reps = vertical_scan_reps(panel_img, n_layers_hint=n_hint)
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

        out_name = f"{desc.replace(' ', '_').replace('/', '_')}_panel{panel_id}_{n_found}layers_v4.png"
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
                "desc": desc,
                "panel_id": panel_id,
                "score": review.overall_score,
                "recommendation": review.recommendation,
                "n_expected": review.n_layers_expected,
                "n_found": review.n_layers_found,
                "over_seg": review.over_segmentation,
                "under_seg": review.under_segmentation,
                "file": out_name,
            })
        except Exception as exc:
            print(f"  -> ERROR: {exc}")
            results.append({
                "desc": desc,
                "panel_id": panel_id,
                "error": str(exc),
                "file": out_name,
            })

    return results


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    test_cases = [
        # Paper 1: seismic_inv (known 0.85)
        ("papers_new/to_process/seismic_inv_2024/seismic_inv_2024_page9_img1.png", "seismic_inv page9", 5, 2),

        # Paper 2: fwi_deeponet (known 0.83)
        ("papers_new/to_process/fwi_deeponet_2024/fwi_deeponet_2024_page8_img0.png", "fwi_deeponet page8", 7, 0),

        # Paper 3: ml_velocity - simpler candidate
        ("papers_new/to_process/ml_velocity_2024/ml_velocity_2024_page21_img0.png", "ml_velocity page21", 4, None),

        # Paper 4: tomography_review - best candidate
        ("papers_new/to_process/tomography_review_2024/tomography_review_2024_page5_img2.png", "tomography_review page5", 4, 2),

        # Paper 5: wise_fwi - smaller candidate with 3 reps
        ("papers_new/to_process/wise_fwi_2024/wise_fwi_2024_page7_img1.png", "wise_fwi page7", 4, None),
    ]

    for img_path_str, desc, n_hint, panel_id in test_cases:
        img_path = Path(img_path_str)
        if not img_path.exists():
            print(f"SKIP: {img_path} not found")
            continue
        results = process_and_review(img_path, desc, n_hint=n_hint, target_panel_id=panel_id)
        all_results.extend(results)

    report_path = AUDIT_DIR / "v4_focused_report.json"
    report_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    print(f"\n{'='*60}")
    print(f"Report: {report_path}")

    ok = [r for r in all_results if r.get("recommendation") == "accept"]
    manual = [r for r in all_results if r.get("recommendation") == "manual_fix"]
    reject = [r for r in all_results if r.get("recommendation") == "reject"]
    errors = [r for r in all_results if "error" in r]

    print(f"accept={len(ok)} manual_fix={len(manual)} reject={len(reject)} errors={len(errors)}")
    for r in all_results:
        if "score" in r:
            print(f"  {r['desc']} panel={r['panel_id']}: score={r['score']:.2f} rec={r['recommendation']}")
        else:
            print(f"  {r['desc']} panel={r['panel_id']}: ERROR")


if __name__ == "__main__":
    main()
