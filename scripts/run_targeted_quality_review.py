#!/usr/bin/env python3
"""Run VLM quality review on selected key overlays."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoseg.modules.vlm_client.client import review_segmentation_quality

OUT_DIR = Path("runs/new_papers_vlm/all_overlays")
AUDIT_DIR = Path("runs/new_papers_vlm/quality_review_audit")

# Key panels to review: (overlay_filename, description)
TARGETS = [
    ("fwi_deeponet_2024_fwi_deeponet_2024_page8_img0_panel0_8layers.png", "fwi_deeponet page8 panel0"),
    ("ml_velocity_2024_ml_velocity_2024_page22_img2_panel0_6layers.png", "ml_velocity page22_img2 panel0"),
    ("tomography_review_2024_tomography_review_2024_page5_img2_panel2_9layers.png", "tomography_review page5_img2 panel2"),
    ("wise_fwi_2024_wise_fwi_2024_page5_img2_panel0_resized.png", "wise_fwi page5_img2 panel0"),
]


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for i, (filename, desc) in enumerate(TARGETS, 1):
        img_path = OUT_DIR / filename
        if not img_path.exists():
            print(f"[{i}/{len(TARGETS)}] SKIP: {img_path} not found")
            continue

        print(f"\n[{i}/{len(TARGETS)}] Reviewing {desc} ...")
        img_rgb = np.array(Image.open(img_path).convert("RGB"))

        try:
            review = review_segmentation_quality(
                img_rgb,
                audit_dir=AUDIT_DIR,
                mode="auto",
                min_confidence=0.5,
            )
            print(f"  -> score={review.overall_score:.2f} rec={review.recommendation}")
            results.append({
                "file": filename,
                "desc": desc,
                "score": review.overall_score,
                "recommendation": review.recommendation,
                "n_expected": review.n_layers_expected,
                "n_found": review.n_layers_found,
                "over_seg": review.over_segmentation,
                "under_seg": review.under_segmentation,
                "fix_hints": review.fix_hints,
            })
        except Exception as exc:
            print(f"  -> ERROR: {exc}")
            results.append({
                "file": filename,
                "desc": desc,
                "error": str(exc),
            })

    report_path = AUDIT_DIR / "targeted_review_report.json"
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nReport saved: {report_path}")

    ok = [r for r in results if r.get("recommendation") == "accept"]
    manual = [r for r in results if r.get("recommendation") == "manual_fix"]
    reject = [r for r in results if r.get("recommendation") == "reject"]
    errors = [r for r in results if "error" in r]

    print(f"\nSummary: accept={len(ok)} manual_fix={len(manual)} reject={len(reject)} errors={len(errors)}")
    for r in results:
        if "score" in r:
            print(f"  {r['desc']}: score={r['score']:.2f} rec={r['recommendation']}")
        else:
            print(f"  {r['desc']}: ERROR")


if __name__ == "__main__":
    main()
