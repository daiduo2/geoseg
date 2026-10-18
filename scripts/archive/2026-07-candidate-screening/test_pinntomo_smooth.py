#!/usr/bin/env python3
"""Test pinntomo with boundary smoothing to push 0.76 -> accept."""

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
AUDIT_DIR = Path("runs/new_papers_vlm/quality_review_audit_pinntomo_smooth")


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
    """Smooth boundaries using distance transform + gaussian filtering."""
    from scipy import ndimage
    result = np.zeros_like(labels)
    n_layers = int(labels.max())

    # For each layer, compute signed distance and smooth
    for lbl in range(1, n_layers + 1):
        mask = labels == lbl
        if not mask.any():
            continue
        # Distance transform
        dist = ndimage.distance_transform_edt(mask)
        dist_out = ndimage.distance_transform_edt(~mask)
        signed = dist - dist_out
        # Smooth the signed distance
        smoothed = ndimage.gaussian_filter(signed, sigma=sigma)
        result[smoothed > 0] = lbl

    return result


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    img_rgb = np.array(Image.open("papers_new/to_process/pinntomo_2021/pinntomo_2021_page4_render.png").convert("RGB"))
    panel_bboxes = detect_panels(img_rgb)
    if not panel_bboxes:
        h, w = img_rgb.shape[:2]
        panel_bboxes = [{"id": 0, "bbox": [0, 0, w, h]}]
    panel_bboxes.sort(key=lambda pb: (pb["bbox"][1], pb["bbox"][0]))

    pb = [p for p in panel_bboxes if p["id"] == 0][0]
    x, y, pw, ph = pb["bbox"]
    panel_img = img_rgb[y:y+ph, x:x+pw]

    for n_hint in [6, 7]:
        for engine in ["edge_grow", "kmeans_full"]:
            reps = vertical_scan_reps(panel_img, n_layers_hint=n_hint)
            if len(reps) < 2:
                continue
            n_layers = len(reps)
            seg = run_engine(engine, panel_img, reps, None, n_layers)
            labels = seg["labels"]
            n_found = len(set(labels.flatten()) - {0})

            # Try different smoothing strategies
            for smooth_sigma in [0, 0.8, 1.2]:
                if smooth_sigma > 0:
                    proc_labels = smooth_labels(labels, sigma=smooth_sigma)
                    n_found_proc = len(set(proc_labels.flatten()) - {0})
                    suffix = f"smooth{smooth_sigma}"
                else:
                    proc_labels = labels
                    n_found_proc = n_found
                    suffix = "nosmooth"

                overlay = create_vivid_overlay(panel_img, proc_labels)
                composed = compose_side_by_side(panel_img, overlay)

                out_name = f"pinntomo_p4_p0_n{n_hint}_{engine}_{suffix}_{n_found_proc}l.png"
                out_path = OUT_DIR / out_name
                Image.fromarray(composed).save(out_path)

                try:
                    review = review_segmentation_quality(composed, audit_dir=AUDIT_DIR, mode="auto", min_confidence=0.5)
                    print(f"pinntomo n={n_hint} eng={engine} {suffix}: score={review.overall_score:.2f} rec={review.recommendation}")
                    all_results.append({
                        "n_hint": n_hint, "engine": engine, "smooth": smooth_sigma,
                        "n_found": n_found_proc, "score": review.overall_score,
                        "recommendation": review.recommendation,
                    })
                    if review.recommendation == "accept":
                        print("*** ACCEPT! ***")
                except Exception as exc:
                    print(f"pinntomo n={n_hint} eng={engine} {suffix}: ERROR {exc}")
                    all_results.append({
                        "n_hint": n_hint, "engine": engine, "smooth": smooth_sigma,
                        "error": str(exc),
                    })

    report_path = AUDIT_DIR / "pinntomo_smooth_report.json"
    report_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    print(f"\nReport: {report_path}")

    ok = [r for r in all_results if r.get("recommendation") == "accept"]
    print(f"accept={len(ok)} total={len(all_results)}")


if __name__ == "__main__":
    main()
