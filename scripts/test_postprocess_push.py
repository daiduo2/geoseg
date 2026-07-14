#!/usr/bin/env python3
"""Post-process segmentation to smooth boundaries and merge thin layers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoseg.modules.segment_engines.vlm_reps import vertical_scan_reps
from geoseg.modules.segment_engines import run_engine
from geoseg.modules.vlm_client.client import review_segmentation_quality
from geoseg.modules.cv_detect.panel_detector import detect_panels

OUT_DIR = Path("runs/new_papers_vlm/all_overlays")
AUDIT_DIR = Path("runs/new_papers_vlm/quality_review_audit_postprocess")


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


def merge_thin_layers(labels: np.ndarray, min_area_ratio: float = 0.05) -> np.ndarray:
    """Merge layers that are smaller than min_area_ratio with adjacent layers."""
    from scipy import ndimage
    result = labels.copy()
    n_layers = int(result.max())
    total_area = result.shape[0] * result.shape[1]

    for lbl in range(1, n_layers + 1):
        mask = result == lbl
        area = mask.sum()
        if area < total_area * min_area_ratio:
            # Find adjacent layers
            dilated = ndimage.binary_dilation(mask)
            neighbors = set(result[dilated]) - {0, lbl}
            if neighbors:
                # Merge with the largest neighbor
                best_neighbor = max(neighbors, key=lambda n: (result == n).sum())
                result[mask] = best_neighbor

    # Renumber labels to be contiguous
    unique = sorted(set(result.flatten()) - {0})
    relabeled = np.zeros_like(result)
    for new_id, old_id in enumerate(unique, 1):
        relabeled[result == old_id] = new_id

    return relabeled


def smooth_boundaries(labels: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Apply morphological smoothing to each layer."""
    from scipy import ndimage
    result = labels.copy()
    n_layers = int(result.max())

    for _ in range(iterations):
        for lbl in range(1, n_layers + 1):
            mask = result == lbl
            if mask.any():
                # Opening then closing to smooth boundaries
                smoothed = ndimage.binary_opening(mask, iterations=1)
                smoothed = ndimage.binary_closing(smoothed, iterations=1)
                # Only update if we don't lose the layer entirely
                if smoothed.sum() > 0:
                    result[mask] = 0
                    result[smoothed] = lbl

    # Renumber
    unique = sorted(set(result.flatten()) - {0})
    relabeled = np.zeros_like(result)
    for new_id, old_id in enumerate(unique, 1):
        relabeled[result == old_id] = new_id

    return relabeled


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    img_rgb = np.array(Image.open("papers_new/to_process/ml_velocity_2024/ml_velocity_2024_page10_img1.png").convert("RGB"))
    panel_bboxes = detect_panels(img_rgb)
    if not panel_bboxes:
        h, w = img_rgb.shape[:2]
        panel_bboxes = [{"id": 0, "bbox": [0, 0, w, h]}]
    panel_bboxes.sort(key=lambda pb: (pb["bbox"][1], pb["bbox"][0]))
    pb = panel_bboxes[0]
    x, y, pw, ph = pb["bbox"]
    panel_img = img_rgb[y:y+ph, x:x+pw]

    reps = vertical_scan_reps(panel_img, n_layers_hint=6)
    n_layers = len(reps)
    seg = run_engine("kmeans_full", panel_img, reps, None, n_layers)
    labels = seg["labels"]
    n_found = len(set(labels.flatten()) - {0})
    print(f"Original: {n_found} layers")

    all_results = []

    # Try different post-processing strategies
    strategies = [
        ("merge_thin_3pct", lambda l: merge_thin_layers(l, 0.03)),
        ("merge_thin_5pct", lambda l: merge_thin_layers(l, 0.05)),
        ("merge_thin_8pct", lambda l: merge_thin_layers(l, 0.08)),
        ("smooth_1iter", lambda l: smooth_boundaries(l, 1)),
        ("smooth_2iter", lambda l: smooth_boundaries(l, 2)),
        ("merge_5pct_smooth", lambda l: smooth_boundaries(merge_thin_layers(l, 0.05), 1)),
        ("merge_8pct_smooth", lambda l: smooth_boundaries(merge_thin_layers(l, 0.08), 1)),
    ]

    for name, processor in strategies:
        processed = processor(labels)
        n_new = len(set(processed.flatten()) - {0})
        print(f"\nStrategy {name}: {n_new} layers")

        overlay = create_vivid_overlay(panel_img, processed)
        composed = compose_side_by_side(panel_img, overlay)

        out_name = f"ml_velocity_p10i1_{name}_{n_new}l_postprocess.png"
        out_path = OUT_DIR / out_name
        Image.fromarray(composed).save(out_path)

        try:
            review = review_segmentation_quality(composed, audit_dir=AUDIT_DIR, mode="auto", min_confidence=0.5)
            print(f"  -> score={review.overall_score:.2f} rec={review.recommendation}")
            all_results.append({
                "strategy": name, "n_layers": n_new,
                "score": review.overall_score, "recommendation": review.recommendation,
            })
        except Exception as exc:
            print(f"  -> ERROR: {exc}")
            all_results.append({"strategy": name, "n_layers": n_new, "error": str(exc)})

    report_path = AUDIT_DIR / "postprocess_report.json"
    report_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    print(f"\nReport: {report_path}")

    ok = [r for r in all_results if r.get("recommendation") == "accept"]
    print(f"accept={len(ok)} total={len(all_results)}")
    if ok:
        for r in ok:
            print(f"  {r['strategy']}: {r['score']:.2f}")


if __name__ == "__main__":
    main()
