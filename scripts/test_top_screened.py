#!/usr/bin/env python3
"""Test top-screened candidates for ACCEPT."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoseg.modules.segment_engines.vlm_reps import vertical_scan_reps
from geoseg.modules.segment_engines.router import route_and_segment, _run_engine
from geoseg.modules.vlm_client.client import review_segmentation_quality
from geoseg.modules.cv_detect.panel_detector import detect_panels

OUT_DIR = Path("runs/new_papers_vlm/all_overlays")
AUDIT_DIR = Path("runs/new_papers_vlm/quality_review_audit_screened")


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


def process(img_path: Path, desc: str, n_hint: int, engine: str | None = None, panel_id: int | None = None) -> dict | None:
    img_rgb = np.array(Image.open(img_path).convert("RGB"))
    panel_bboxes = detect_panels(img_rgb)
    if not panel_bboxes:
        h, w = img_rgb.shape[:2]
        panel_bboxes = [{"id": 0, "bbox": [0, 0, w, h]}]
    panel_bboxes.sort(key=lambda pb: (pb["bbox"][1], pb["bbox"][0]))

    if panel_id is not None:
        target_panels = [pb for pb in panel_bboxes if pb["id"] == panel_id]
        if not target_panels:
            target_panels = [panel_bboxes[0]]
    else:
        target_panels = [panel_bboxes[0]]

    pb = target_panels[0]
    x, y, pw, ph = pb["bbox"]
    panel_img = img_rgb[y:y+ph, x:x+pw]
    if pw < 50 or ph < 50:
        return None

    reps = vertical_scan_reps(panel_img, n_layers_hint=n_hint)
    if len(reps) < 2:
        return None

    n_layers = len(reps)
    if engine:
        seg = _run_engine(engine, panel_img, reps, None, n_layers)
    else:
        seg = route_and_segment(
            panel_img, reps=reps, n_layers=n_layers,
            quality_preference="balanced", is_velocity_model=True, retry_on_underseg=True,
        )
    labels = seg["labels"]
    n_found = len(set(labels.flatten()) - {0})

    overlay = create_vivid_overlay(panel_img, labels)
    composed = compose_side_by_side(panel_img, overlay)

    engine_str = seg["meta"]["engine"]
    out_name = f"{desc.replace(' ', '_').replace('/', '_')}_n{n_hint}_{engine_str}_{n_found}l_screened.png"
    out_path = OUT_DIR / out_name
    Image.fromarray(composed).save(out_path)

    try:
        review = review_segmentation_quality(composed, audit_dir=AUDIT_DIR, mode="auto", min_confidence=0.5)
        return {
            "desc": desc, "n_hint": n_hint, "engine": engine_str, "n_found": n_found,
            "score": review.overall_score, "recommendation": review.recommendation,
        }
    except Exception as exc:
        return {"desc": desc, "n_hint": n_hint, "engine": engine_str, "n_found": n_found, "error": str(exc)}


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    # Top candidates from screening
    test_cases = [
        # uncertainty_fwi - very high saturation images
        ("papers_new/to_process/uncertainty_fwi_2024/uncertainty_fwi_2024_page12_img3.png", "uncertainty p12i3", [3, 4, 5]),
        ("papers_new/to_process/uncertainty_fwi_2024/uncertainty_fwi_2024_page12_img5.png", "uncertainty p12i5", [4, 5, 6]),
        ("papers_new/to_process/uncertainty_fwi_2024/uncertainty_fwi_2024_page12_img7.png", "uncertainty p12i7", [3, 4]),
        ("papers_new/to_process/uncertainty_fwi_2024/uncertainty_fwi_2024_page7_img7.png", "uncertainty p7i7", [3, 4, 5]),
        ("papers_new/to_process/uncertainty_fwi_2024/uncertainty_fwi_2024_page7_img9.png", "uncertainty p7i9", [3, 4, 5]),
        # ml_velocity - high saturation + high max_diff
        ("papers_new/to_process/ml_velocity_2024/ml_velocity_2024_page10_img1.png", "ml_velocity p10i1", [5, 6, 7]),
        ("papers_new/to_process/ml_velocity_2024/ml_velocity_2024_page12_img0.png", "ml_velocity p12i0", [3, 4, 5]),
        ("papers_new/to_process/ml_velocity_2024/ml_velocity_2024_page26_img2.png", "ml_velocity p26i2", [5, 6, 7]),
        # wise_fwi - high saturation
        ("papers_new/to_process/wise_fwi_2024/wise_fwi_2024_page5_img0.png", "wise_fwi p5i0", [4, 5, 6]),
        ("papers_new/to_process/wise_fwi_2024/wise_fwi_2024_page7_img0.png", "wise_fwi p7i0", [4, 5, 6]),
    ]

    for img_path_str, desc, n_hints in test_cases:
        img_path = Path(img_path_str)
        if not img_path.exists():
            print(f"SKIP: {img_path} not found")
            continue
        for n in n_hints:
            for eng in [None, "edge_guided", "edge_grow"]:
                r = process(img_path, desc, n, eng)
                if r:
                    all_results.append(r)
                    print(f"{desc} n={n} eng={eng}: score={r.get('score', 'ERR')} rec={r.get('recommendation', 'ERR')}")

    report_path = AUDIT_DIR / "screened_report.json"
    report_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    print(f"\nReport: {report_path}")

    ok = [r for r in all_results if r.get("recommendation") == "accept"]
    manual = [r for r in all_results if r.get("recommendation") == "manual_fix"]
    errors = [r for r in all_results if "error" in r]
    print(f"accept={len(ok)} manual_fix={len(manual)} errors={len(errors)} total={len(all_results)}")

    if ok:
        print("\nACCEPT results:")
        for r in ok:
            print(f"  {r['desc']} n={r['n_hint']} eng={r['engine']}: {r['score']:.2f}")


if __name__ == "__main__":
    main()
