#!/usr/bin/env python3
"""Batch-process ALL panels from velocity-model figures with vertical-scan reps.

For each velocity_model figure in pipeline_results.json:
  1. Detect all panels (no target-panel filtering)
  2. Generate reps via vertical_scan_reps per panel
  3. Segment with route_and_segment (improved params)
  4. Create vivid overlay + side-by-side composite
  5. Run VLM quality review (budget-limited)

Outputs:
  - Composite images: runs/new_papers_vlm/vscan_audit/<fig_key>_panel<id>_<n>layers.png
  - JSON report: runs/new_papers_vlm/vscan_audit/report.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoseg.modules.segment_engines.vlm_reps import vertical_scan_reps
from geoseg.modules.segment_engines import route_and_segment
from geoseg.modules.cv_detect.panel_detector import detect_panels
from geoseg.modules.vlm_client.client import review_segmentation_quality

RESULTS_FILE = Path("runs/new_papers_vlm/pipeline_results.json")
OUT_DIR = Path("runs/new_papers_vlm/vscan_audit")


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


def main() -> None:
    if not RESULTS_FILE.exists():
        print(f"Results file not found: {RESULTS_FILE}")
        sys.exit(1)

    pipeline = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    vm_figures = [
        r for r in pipeline
        if r.get("vlm_type") == "velocity_model" and r.get("status") == "ok"
    ]

    print(f"Found {len(vm_figures)} velocity_model figures to process")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    review_budget_total = 12  # Limit total VLM reviews to avoid excessive API calls/time
    review_count = 0

    for fig_idx, record in enumerate(vm_figures, 1):
        fig_key = record["fig_key"]
        img_path = Path(record.get("img_path", ""))
        if not img_path.exists():
            print(f"\n[{fig_idx}/{len(vm_figures)}] SKIP {fig_key}: image not found")
            continue

        print(f"\n{'='*60}")
        print(f"[{fig_idx}/{len(vm_figures)}] {fig_key}")
        print(f"{'='*60}")

        img_rgb = np.array(Image.open(img_path).convert("RGB"))
        print(f"  Image shape: {img_rgb.shape}")

        panel_bboxes = detect_panels(img_rgb)
        print(f"  Detected {len(panel_bboxes)} panels")

        if not panel_bboxes:
            # Whole image as one panel
            h, w = img_rgb.shape[:2]
            panel_bboxes = [{"id": 0, "bbox": [0, 0, w, h]}]
            print(f"  Fallback: whole image as single panel")

        # Sort panels: top-to-bottom, left-to-right
        panel_bboxes.sort(key=lambda pb: (pb["bbox"][1], pb["bbox"][0]))

        # Review budget per figure: at most 2 panels
        review_budget_fig = 2
        fig_review_count = 0

        for pb in panel_bboxes:
            x, y, pw, ph = pb["bbox"]
            panel_id = pb["id"]
            panel_img = img_rgb[y:y+ph, x:x+pw]

            print(f"\n  [Panel {panel_id}] shape={panel_img.shape}")

            # Skip tiny panels
            if pw < 50 or ph < 50:
                print(f"    SKIP: too small ({pw}x{ph})")
                continue

            # Generate reps via vertical scan
            n_hint = record.get("total_layers", 5)
            if n_hint < 2:
                n_hint = 5
            reps = vertical_scan_reps(panel_img, n_layers_hint=n_hint)
            print(f"    vertical_scan reps: {len(reps)}")

            if len(reps) < 2:
                print(f"    SKIP: insufficient reps (< 2)")
                results.append({
                    "fig_key": fig_key,
                    "panel_id": panel_id,
                    "status": "skip",
                    "reason": "insufficient_reps",
                })
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
            print(f"    Segmented: {n_found} layers, engine={seg['meta']['engine']}")

            # Generate vivid overlay and composite
            overlay = create_vivid_overlay(panel_img, labels)
            composed = compose_side_by_side(panel_img, overlay)

            out_name = f"{fig_key.replace('/', '_')}_panel{panel_id}_{n_found}layers.png"
            out_path = OUT_DIR / out_name
            Image.fromarray(composed).save(out_path)
            print(f"    Saved: {out_path}")

            # VLM quality review (budget-limited)
            do_review = (
                review_count < review_budget_total
                and fig_review_count < review_budget_fig
                and n_found >= 2
            )

            if do_review:
                print(f"    Running VLM quality review...")
                try:
                    review = review_segmentation_quality(
                        composed,
                        audit_dir=OUT_DIR / "audit",
                        mode="auto",
                        min_confidence=0.5,
                    )
                    print(f"    -> score={review.overall_score:.2f} rec={review.recommendation}")
                    results.append({
                        "fig_key": fig_key,
                        "panel_id": panel_id,
                        "file": out_name,
                        "score": review.overall_score,
                        "recommendation": review.recommendation,
                        "n_expected": review.n_layers_expected,
                        "n_found": review.n_layers_found,
                        "over_seg": review.over_segmentation,
                        "under_seg": review.under_segmentation,
                        "engine": seg["meta"]["engine"],
                    })
                    review_count += 1
                    fig_review_count += 1
                except Exception as exc:
                    print(f"    -> ERROR: {exc}")
                    results.append({
                        "fig_key": fig_key,
                        "panel_id": panel_id,
                        "file": out_name,
                        "error": str(exc),
                        "engine": seg["meta"]["engine"],
                    })
            else:
                results.append({
                    "fig_key": fig_key,
                    "panel_id": panel_id,
                    "file": out_name,
                    "status": "segmented_no_review",
                    "n_found": n_found,
                    "engine": seg["meta"]["engine"],
                })

    # Save report
    report_path = OUT_DIR / "report.json"
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n{'='*60}")
    print(f"Batch report: {report_path}")

    ok = [r for r in results if r.get("recommendation") == "accept"]
    manual = [r for r in results if r.get("recommendation") == "manual_fix"]
    reject = [r for r in results if r.get("recommendation") == "reject"]
    errors = [r for r in results if "error" in r]
    no_review = [r for r in results if r.get("status") == "segmented_no_review"]

    print(f"accept={len(ok)} manual_fix={len(manual)} reject={len(reject)} errors={len(errors)} no_review={len(no_review)}")

    if ok:
        print("\nAccepted:")
        for r in ok:
            print(f"  {r['fig_key']} panel={r['panel_id']} score={r['score']:.2f}")
    if manual:
        print("\nManual fix:")
        for r in manual:
            print(f"  {r['fig_key']} panel={r['panel_id']} score={r['score']:.2f}")
    if reject:
        print("\nRejected:")
        for r in reject:
            print(f"  {r['fig_key']} panel={r['panel_id']} score={r['score']:.2f}")


if __name__ == "__main__":
    main()
