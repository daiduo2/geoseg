#!/usr/bin/env python3
"""Apply boundary smoothing to best near-ACCEPT candidates."""

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
AUDIT_DIR = Path("runs/new_papers_vlm/quality_review_audit_smoothing_push")


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


def merge_thin_layers(labels: np.ndarray, min_pixels: int) -> np.ndarray:
    from scipy import ndimage
    result = labels.copy()
    n_layers = int(labels.max())
    for lbl in range(1, n_layers + 1):
        mask = result == lbl
        if mask.sum() < min_pixels:
            # Merge with largest neighbor
            dilated = ndimage.binary_dilation(mask)
            neighbors = dilated & ~mask & (result > 0)
            if neighbors.any():
                neighbor_labels = result[neighbors]
                unique, counts = np.unique(neighbor_labels, return_counts=True)
                target = unique[counts.argmax()]
                result[mask] = target
    # Renumber
    unique = sorted(set(result.flatten()) - {0})
    renumbered = np.zeros_like(result)
    for new_id, old_id in enumerate(unique, 1):
        renumbered[result == old_id] = new_id
    return renumbered


def test_candidate(img_path: Path, desc: str, n_hint: int, engine: str, smooth_sigma: float | None = None, merge_min: int | None = None) -> dict | None:
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
    seg = run_engine(engine, panel_img, reps, None, n_layers)
    labels = seg["labels"]

    if merge_min:
        labels = merge_thin_layers(labels, min_pixels=merge_min)
    if smooth_sigma:
        labels = smooth_labels(labels, sigma=smooth_sigma)

    n_found = len(set(labels.flatten()) - {0})
    overlay = create_vivid_overlay(panel_img, labels)
    composed = compose_side_by_side(panel_img, overlay)

    mods = []
    if merge_min:
        mods.append(f"merge{merge_min}")
    if smooth_sigma:
        mods.append(f"smooth{smooth_sigma}")
    mod_str = "_".join(mods) if mods else "raw"

    out_name = f"{desc.replace(' ', '_')}_n{n_hint}_{engine}_{n_found}l_{mod_str}_smoothpush.png"
    out_path = OUT_DIR / out_name
    Image.fromarray(composed).save(out_path)

    try:
        review = review_segmentation_quality(composed, audit_dir=AUDIT_DIR, mode="auto", min_confidence=0.5)
        return {
            "desc": desc, "n_hint": n_hint, "engine": engine, "n_found": n_found,
            "score": review.overall_score, "recommendation": review.recommendation,
            "mods": mod_str,
        }
    except Exception as exc:
        return {"desc": desc, "n_hint": n_hint, "engine": engine, "n_found": n_found, "error": str(exc), "mods": mod_str}


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    # Best near-ACCEPT candidates
    candidates = [
        # ml_velocity p10i1 - best was 0.75 manual_fix with n=6 kmeans_full
        ("papers_new/to_process/ml_velocity_2024/ml_velocity_2024_page10_img1.png", "ml_velocity p10i1", 6, "kmeans_full"),
        # tomography p9i3 - best was 0.68 manual_fix with n=4 edge_grow
        ("papers_new/to_process/tomography_review_2024/tomography_review_2024_page9_img3.png", "tomography p9i3", 4, "edge_grow"),
        ("papers_new/to_process/tomography_review_2024/tomography_review_2024_page9_img3.png", "tomography p9i3", 5, "kmeans_full"),
    ]

    for fname, desc, n, eng in candidates:
        img_path = Path(fname)
        if not img_path.exists():
            print(f"SKIP: {img_path}")
            continue

        # Test raw
        print(f"\n{desc} n={n} eng={eng} raw...")
        r = test_candidate(img_path, desc, n, eng)
        if r:
            all_results.append(r)
            print(f"  -> score={r.get('score', 'ERR')} rec={r.get('recommendation', 'ERR')}")

        # Test with smoothing
        for sigma in [0.5, 0.8, 1.0]:
            print(f"{desc} n={n} eng={eng} smooth{sigma}...")
            r = test_candidate(img_path, desc, n, eng, smooth_sigma=sigma)
            if r:
                all_results.append(r)
                print(f"  -> score={r.get('score', 'ERR')} rec={r.get('recommendation', 'ERR')}")
                if r.get("recommendation") == "accept":
                    print("  *** ACCEPT! ***")

        # Test with merge + smoothing
        for merge_min in [100, 500]:
            for sigma in [0.8]:
                print(f"{desc} n={n} eng={eng} merge{merge_min}+smooth{sigma}...")
                r = test_candidate(img_path, desc, n, eng, smooth_sigma=sigma, merge_min=merge_min)
                if r:
                    all_results.append(r)
                    print(f"  -> score={r.get('score', 'ERR')} rec={r.get('recommendation', 'ERR')}")
                    if r.get("recommendation") == "accept":
                        print("  *** ACCEPT! ***")

    report_path = AUDIT_DIR / "smoothing_push_report.json"
    report_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    print(f"\nReport: {report_path}")

    ok = [r for r in all_results if r.get("recommendation") == "accept"]
    manual = [r for r in all_results if r.get("recommendation") == "manual_fix"]
    errors = [r for r in all_results if "error" in r]
    print(f"accept={len(ok)} manual_fix={len(manual)} errors={len(errors)} total={len(all_results)}")

    if ok:
        print("\nACCEPT results:")
        for r in ok:
            print(f"  {r['desc']} n={r['n_hint']} eng={r['engine']} mods={r['mods']}: {r['score']:.2f}")


if __name__ == "__main__":
    main()
