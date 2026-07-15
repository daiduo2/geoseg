#!/usr/bin/env python3
"""Test deep_tomo_2024 figures for velocity model extraction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoseg.experiments import vertical_scan_reps
from geoseg.experiments import run_engine
from geoseg.experiments import review_segmentation_quality
from geoseg.experiments import detect_panels

OUT_DIR = Path("runs/new_papers_vlm/all_overlays")
AUDIT_DIR = Path("runs/new_papers_vlm/quality_review_audit_deep_tomo")


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


def smooth_labels(labels: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    from scipy import ndimage
    result = np.zeros_like(labels)
    n_layers = int(labels.max())
    for lbl in range(1, n_layers + 1):
        mask = labels == lbl
        if not mask.any():
            continue
        dist = ndimage.distance_transform_edt(mask)
        dist_out = ndimage.distance_transform_edt(~mask)
        signed = dist - dist_out
        smoothed = ndimage.gaussian_filter(signed, sigma=sigma)
        result[smoothed > 0] = lbl
    return result


def process(img_path: Path, desc: str, n_hint: int, engine: str | None = None, smooth: bool = False) -> dict | None:
    img_rgb = np.array(Image.open(img_path).convert("RGB"))
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

    reps = vertical_scan_reps(panel_img, n_layers_hint=n_hint)
    if len(reps) < 2:
        return None

    n_layers = len(reps)
    if engine:
        seg = run_engine(engine, panel_img, reps, None, n_layers)
    else:
        from geoseg.experiments import route_and_segment
        seg = route_and_segment(
            panel_img, reps=reps, n_layers=n_layers,
            quality_preference="balanced", is_velocity_model=True, retry_on_underseg=True,
        )
    labels = seg["labels"]
    n_found = len(set(labels.flatten()) - {0})

    if smooth:
        labels = smooth_labels(labels, sigma=0.8)
        n_found = len(set(labels.flatten()) - {0})

    overlay = create_vivid_overlay(panel_img, labels)
    composed = compose_side_by_side(panel_img, overlay)

    engine_str = seg["meta"]["engine"]
    smooth_str = "_smooth" if smooth else ""
    out_name = f"{desc.replace(' ', '_').replace('/', '_')}_n{n_hint}_{engine_str}_{n_found}l{smooth_str}_deep.png"
    out_path = OUT_DIR / out_name
    Image.fromarray(composed).save(out_path)

    try:
        review = review_segmentation_quality(composed, audit_dir=AUDIT_DIR, mode="auto", min_confidence=0.5)
        return {
            "desc": desc, "n_hint": n_hint, "engine": engine_str, "n_found": n_found,
            "score": review.overall_score, "recommendation": review.recommendation,
            "smooth": smooth,
        }
    except Exception as exc:
        return {"desc": desc, "n_hint": n_hint, "engine": engine_str, "n_found": n_found, "error": str(exc), "smooth": smooth}


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    candidates = [
        ("papers_new/to_process/deep_tomo_2024/deep_tomo_2024_page4_img0.png", "deep_tomo p4i0", [4, 5, 6]),
        ("papers_new/to_process/deep_tomo_2024/deep_tomo_2024_page7_img0.png", "deep_tomo p7i0", [4, 5, 6]),
        ("papers_new/to_process/deep_tomo_2024/deep_tomo_2024_page7_img1.png", "deep_tomo p7i1", [4, 5, 6]),
        ("papers_new/to_process/deep_tomo_2024/deep_tomo_2024_page7_img2.png", "deep_tomo p7i2", [4, 5, 6]),
        ("papers_new/to_process/deep_tomo_2024/deep_tomo_2024_page8_img0.png", "deep_tomo p8i0", [4, 5, 6]),
        ("papers_new/to_process/deep_tomo_2024/deep_tomo_2024_page9_img0.png", "deep_tomo p9i0", [4, 5, 6]),
        ("papers_new/to_process/deep_tomo_2024/deep_tomo_2024_page9_img1.png", "deep_tomo p9i1", [4, 5, 6]),
        ("papers_new/to_process/deep_tomo_2024/deep_tomo_2024_page11_img0.png", "deep_tomo p11i0", [4, 5, 6]),
        ("papers_new/to_process/deep_tomo_2024/deep_tomo_2024_page11_img1.png", "deep_tomo p11i1", [4, 5, 6]),
        ("papers_new/to_process/deep_tomo_2024/deep_tomo_2024_page20_img0.png", "deep_tomo p20i0", [4, 5, 6]),
        ("papers_new/to_process/deep_tomo_2024/deep_tomo_2024_page21_img0.png", "deep_tomo p21i0", [4, 5, 6]),
        ("papers_new/to_process/deep_tomo_2024/deep_tomo_2024_page22_img0.png", "deep_tomo p22i0", [4, 5, 6]),
        ("papers_new/to_process/deep_tomo_2024/deep_tomo_2024_page22_img1.png", "deep_tomo p22i1", [4, 5, 6]),
        ("papers_new/to_process/deep_tomo_2024/deep_tomo_2024_page23_img0.png", "deep_tomo p23i0", [4, 5, 6]),
        ("papers_new/to_process/deep_tomo_2024/deep_tomo_2024_page23_img1.png", "deep_tomo p23i1", [4, 5, 6]),
        ("papers_new/to_process/deep_tomo_2024/deep_tomo_2024_page24_img0.png", "deep_tomo p24i0", [4, 5, 6]),
        ("papers_new/to_process/deep_tomo_2024/deep_tomo_2024_page24_img1.png", "deep_tomo p24i1", [4, 5, 6]),
    ]

    for fname, desc, n_hints in candidates:
        img_path = Path(fname)
        if not img_path.exists():
            print(f"SKIP: {img_path}")
            continue
        for n in n_hints:
            for eng in [None, "kmeans_full", "edge_grow"]:
                eng_str = eng or "auto"
                print(f"\nTesting {desc} n={n} eng={eng_str}...")
                r = process(img_path, desc, n, eng)
                if r:
                    all_results.append(r)
                    print(f"  -> score={r.get('score', 'ERR')} rec={r.get('recommendation', 'ERR')}")
                    if r.get("recommendation") == "accept":
                        print("  *** ACCEPT! ***")
                else:
                    print(f"  -> SKIP")

                if r and r.get("recommendation") in ("manual_fix", "retry") and r.get("score", 0) >= 0.55:
                    print(f"  Trying smooth...")
                    r_smooth = process(img_path, desc, n, eng, smooth=True)
                    if r_smooth:
                        all_results.append(r_smooth)
                        print(f"  -> smooth score={r_smooth.get('score', 'ERR')} rec={r_smooth.get('recommendation', 'ERR')}")
                        if r_smooth.get("recommendation") == "accept":
                            print("  *** ACCEPT with smoothing! ***")

    report_path = AUDIT_DIR / "deep_tomo_report.json"
    report_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    print(f"\nReport: {report_path}")

    ok = [r for r in all_results if r.get("recommendation") == "accept"]
    manual = [r for r in all_results if r.get("recommendation") == "manual_fix"]
    errors = [r for r in all_results if "error" in r]
    print(f"accept={len(ok)} manual_fix={len(manual)} errors={len(errors)} total={len(all_results)}")

    if ok:
        print("\nACCEPT results:")
        for r in ok:
            print(f"  {r['desc']} n={r['n_hint']} eng={r['engine']} smooth={r.get('smooth', False)}: {r['score']:.2f}")


if __name__ == "__main__":
    main()
