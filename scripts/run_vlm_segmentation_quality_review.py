#!/usr/bin/env python3
"""Run VLM segmentation quality review on all vivid audit images.

Iterates over runs/new_papers_vlm/vivid_audit/ and calls
review_segmentation_quality() for each image, collecting structured
results into a JSON report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoseg.modules.vlm_client.client import review_segmentation_quality

VIVID_DIR = Path("runs/new_papers_vlm/vivid_audit")
OUT_DIR = Path("runs/new_papers_vlm/quality_review")


def main() -> None:
    if not VIVID_DIR.exists():
        print(f"No vivid audit dir: {VIVID_DIR}")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(VIVID_DIR.glob("*.png"))
    print(f"Found {len(images)} vivid audit images to review\n")

    results = []
    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] Reviewing {img_path.name} ...")
        try:
            img = np.array(Image.open(img_path).convert("RGB"))
            result = review_segmentation_quality(
                img,
                audit_dir=OUT_DIR / "audit",
                mode="auto",
                min_confidence=0.5,
            )
            record = {
                "file": img_path.name,
                "overall_score": result.overall_score,
                "recommendation": result.recommendation,
                "n_layers_expected": result.n_layers_expected,
                "n_layers_found": result.n_layers_found,
                "over_segmentation": result.over_segmentation,
                "under_segmentation": result.under_segmentation,
                "noise_regions": result.noise_regions,
                "missing_boundaries": result.missing_boundaries,
                "fix_hints": result.fix_hints,
                "layer_qualities": [
                    {
                        "layer_id": lq.layer_id,
                        "boundary_alignment": lq.boundary_alignment,
                        "color_consistency": lq.color_consistency,
                        "is_continuous": lq.is_continuous,
                        "fragmentation_issues": lq.fragmentation_issues,
                    }
                    for lq in result.layer_qualities
                ],
                "overall_assessment": result.overall_assessment,
                "error": None,
            }
            print(
                f"  -> score={result.overall_score:.2f} "
                f"rec={result.recommendation} "
                f"layers={result.n_layers_found}/{result.n_layers_expected}"
            )
        except Exception as exc:
            record = {
                "file": img_path.name,
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"  -> ERROR: {exc}")

        results.append(record)

    # Save report
    report_path = OUT_DIR / "vlm_quality_review.json"
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nReport saved: {report_path}")

    # Summary
    ok = [r for r in results if r.get("recommendation") == "accept"]
    manual = [r for r in results if r.get("recommendation") == "manual_fix"]
    reject = [r for r in results if r.get("recommendation") == "reject"]
    errors = [r for r in results if r.get("error")]

    print(f"\nSummary:")
    print(f"  accept:      {len(ok)}")
    print(f"  manual_fix:  {len(manual)}")
    print(f"  reject:      {len(reject)}")
    print(f"  errors:      {len(errors)}")


if __name__ == "__main__":
    main()
