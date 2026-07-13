#!/usr/bin/env python3
"""Quick test of vertical_scan_reps + segmentation on problematic panels."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoseg.modules.segment_engines.vlm_reps import vertical_scan_reps
from geoseg.modules.segment_engines.router import route_and_segment
from geoseg.modules.vlm_client.client import review_segmentation_quality

OUT_DIR = Path("runs/new_papers_vlm/vertical_scan_test")

# Test panels: (fig_key, panel_id, expected_layers_approx)
TEST_PANELS = [
    ("fwi_deeponet_2024/fwi_deeponet_2024_page8_img0", 1, 7),
    ("fwi_deeponet_2024/fwi_deeponet_2024_page8_img0", 0, 5),
    ("tomography_review_2024/tomography_review_2024_page8_img3", 2, 5),
]


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
    except:
        font = ImageFont.load_default()
    draw.text((10, 10), "ORIGINAL", fill=(255, 255, 255), font=font)
    draw.text((w1 + gap + 10, 10), "SEGMENTATION", fill=(255, 255, 255), font=font)
    return np.array(pil)


def main() -> None:
    pipeline = json.loads(Path("runs/new_papers_vlm/pipeline_results.json").read_text(encoding="utf-8"))
    best = {r["fig_key"]: r for r in pipeline}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for fig_key, panel_id_hint, expected_layers in TEST_PANELS:
        target = best.get(fig_key)
        if not target:
            print(f"SKIP {fig_key}")
            continue

        img_path = target.get("img_path", "")
        if not img_path or not Path(img_path).exists():
            print(f"SKIP {fig_key}: no image")
            continue

        print(f"\n=== {fig_key} panel ~{panel_id_hint} ===")
        img_rgb = np.array(Image.open(img_path).convert("RGB"))

        # Load original panel bbox from audit overlay
        overlay_dir = Path(f"runs/new_papers_vlm/audit_overlays/{fig_key.replace('/', '_')}")
        panel_meta_path = overlay_dir / f"panel_{panel_id_hint}_meta.json"
        if not panel_meta_path.exists():
            print(f"  No panel meta found, using whole image")
            panel_img = img_rgb
        else:
            meta = json.loads(panel_meta_path.read_text())
            x, y, pw, ph = meta["bbox"]
            panel_img = img_rgb[y:y+ph, x:x+pw]
            print(f"  Panel bbox: {meta['bbox']}, shape={panel_img.shape}")

        # Generate reps with vertical scan
        reps = vertical_scan_reps(panel_img, n_layers_hint=expected_layers)
        print(f"  Generated {len(reps)} reps via vertical scan")
        for r in reps:
            print(f"    {r['color_name']}: ({r['representative_point']['x']}, {r['representative_point']['y']})")

        if len(reps) < 2:
            print(f"  SKIP: insufficient reps")
            continue

        # Segment
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

        # Generate vivid overlay
        overlay = create_vivid_overlay(panel_img, labels)
        composed = compose_side_by_side(panel_img, overlay)

        out_name = f"{fig_key.replace('/', '_')}_panel{panel_id_hint}_{n_found}layers_vscan.png"
        out_path = OUT_DIR / out_name
        Image.fromarray(composed).save(out_path)
        print(f"  Saved: {out_path}")

        # VLM review
        print(f"  Running VLM quality review...")
        try:
            review = review_segmentation_quality(
                composed,
                audit_dir=OUT_DIR / "audit",
                mode="auto",
                min_confidence=0.5,
            )
            print(f"  -> score={review.overall_score:.2f} rec={review.recommendation}")
            results.append({
                "file": out_name,
                "score": review.overall_score,
                "recommendation": review.recommendation,
                "n_expected": review.n_layers_expected,
                "n_found": review.n_layers_found,
                "over_seg": review.over_segmentation,
                "under_seg": review.under_segmentation,
            })
        except Exception as exc:
            print(f"  -> ERROR: {exc}")
            results.append({"file": out_name, "error": str(exc)})

    report_path = OUT_DIR / "test_report.json"
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nTest report: {report_path}")

    ok = [r for r in results if r.get("recommendation") == "accept"]
    manual = [r for r in results if r.get("recommendation") == "manual_fix"]
    reject = [r for r in results if r.get("recommendation") == "reject"]
    print(f"accept={len(ok)} manual_fix={len(manual)} reject={len(reject)} errors={len([r for r in results if 'error' in r])}")


if __name__ == "__main__":
    main()
