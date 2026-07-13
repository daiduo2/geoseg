#!/usr/bin/env python3
"""Run VLM review multiple times on ml_velocity p10i1 overlay."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoseg.modules.vlm_client.client import review_segmentation_quality

OUT_DIR = Path("runs/new_papers_vlm/all_overlays")
AUDIT_DIR = Path("runs/new_papers_vlm/quality_review_audit_vlm_luck_mlvel")

# Best ml_velocity overlay
BEST_OVERLAY = OUT_DIR / "ml_velocity_p10i1_n6_kmeans_full_8l_focused.png"


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    if not BEST_OVERLAY.exists():
        print(f"Overlay not found: {BEST_OVERLAY}")
        return

    composed = np.array(Image.open(BEST_OVERLAY).convert("RGB"))

    for run in range(5):
        print(f"\nRun {run + 1}/5")
        try:
            review = review_segmentation_quality(composed, audit_dir=AUDIT_DIR, mode="auto", min_confidence=0.5)
            print(f"  -> score={review.overall_score:.2f} rec={review.recommendation}")
            all_results.append({
                "run": run, "score": review.overall_score,
                "recommendation": review.recommendation,
                "assessment": review.overall_assessment,
            })
            if review.recommendation == "accept":
                print("  *** ACCEPT! ***")
                break
        except Exception as exc:
            print(f"  -> ERROR: {exc}")
            all_results.append({"run": run, "error": str(exc)})

    report_path = AUDIT_DIR / "vlm_luck_mlvel_report.json"
    report_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    print(f"\nReport: {report_path}")

    ok = [r for r in all_results if r.get("recommendation") == "accept"]
    print(f"accept={len(ok)} total={len(all_results)}")
    for r in all_results:
        if "score" in r:
            print(f"  run={r['run']}: score={r['score']:.2f} rec={r['recommendation']}")


if __name__ == "__main__":
    main()
