#!/usr/bin/env python3
"""Screen all unprocessed images for saturation and max_diff to find promising candidates."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoseg.experiments import vertical_scan_reps
from geoseg.experiments import detect_panels


def analyze_panel(panel_img: np.ndarray) -> dict:
    h, w = panel_img.shape[:2]
    rgb = panel_img.astype(float) / 255.0
    max_val = rgb.max(axis=2)
    min_val = rgb.min(axis=2)
    saturation = np.where(max_val > 0, (max_val - min_val) / max_val, 0)
    sat_score = float(np.median(saturation))

    import cv2
    lab = cv2.cvtColor(panel_img, cv2.COLOR_RGB2LAB).astype(float)
    l_diff = np.abs(np.diff(lab[:, w // 2, 0])).astype(float)
    median_diff = float(np.median(l_diff)) if len(l_diff) > 0 else 0
    max_diff = float(np.max(l_diff)) if len(l_diff) > 0 else 0

    reps = vertical_scan_reps(panel_img, n_layers_hint=5)

    return {
        "shape": f"{w}x{h}",
        "saturation": round(sat_score, 3),
        "median_diff": round(median_diff, 2),
        "max_diff": round(max_diff, 2),
        "n_reps": len(reps),
    }


def main() -> None:
    papers = [
        ("papers_new/to_process/uncertainty_fwi_2024", "uncertainty_fwi"),
        ("papers_new/to_process/tomography_review_2024", "tomography_review"),
        ("papers_new/to_process/ml_velocity_2024", "ml_velocity"),
        ("papers_new/to_process/wise_fwi_2024", "wise_fwi"),
        ("papers_new/to_process/deep_tomo_2024", "deep_tomo"),
        ("papers_new/to_process/pinntomo_2021", "pinntomo"),
    ]

    all_results = []
    for img_dir, paper_name in papers:
        img_dir = Path(img_dir)
        if not img_dir.exists():
            continue
        img_paths = sorted(img_dir.glob("*.png"))
        print(f"\n{'='*60}")
        print(f"{paper_name}: {len(img_paths)} images")
        print(f"{'='*60}")

        for img_path in img_paths:
            try:
                img_rgb = np.array(Image.open(img_path).convert("RGB"))
                panel_bboxes = detect_panels(img_rgb)
                if not panel_bboxes:
                    h, w = img_rgb.shape[:2]
                    panel_bboxes = [{"id": 0, "bbox": [0, 0, w, h]}]

                panel_bboxes.sort(key=lambda pb: (pb["bbox"][1], pb["bbox"][0]))

                for pb in panel_bboxes:
                    x, y, pw, ph = pb["bbox"]
                    if pw < 200 or ph < 200:
                        continue
                    panel_img = img_rgb[y:y+ph, x:x+pw]
                    info = analyze_panel(panel_img)
                    info["paper"] = paper_name
                    info["img"] = img_path.name
                    info["panel_id"] = pb["id"]
                    all_results.append(info)

                    # Print promising candidates
                    if info["saturation"] > 0.5 and info["max_diff"] > 10 and info["n_reps"] >= 2:
                        print(f"  PROMISING: {img_path.name} panel={pb['id']} "
                              f"sat={info['saturation']:.2f} max_diff={info['max_diff']:.1f} "
                              f"reps={info['n_reps']} {info['shape']}")
            except Exception as exc:
                print(f"  ERROR: {img_path.name}: {exc}")

    # Sort by composite score
    for r in all_results:
        r["score"] = r["saturation"] * min(r["max_diff"] / 20, 1.0) * min(r["n_reps"] / 3, 1.0)

    all_results.sort(key=lambda r: r["score"], reverse=True)

    print(f"\n{'='*60}")
    print("TOP 30 CANDIDATES (sorted by composite score)")
    print(f"{'='*60}")
    for r in all_results[:30]:
        print(f"{r['paper']:20s} {r['img']:40s} p{r['panel_id']} "
              f"sat={r['saturation']:.2f} md={r['median_diff']:.1f} "
              f"mx={r['max_diff']:.1f} reps={r['n_reps']} {r['shape']} score={r['score']:.2f}")


if __name__ == "__main__":
    main()
