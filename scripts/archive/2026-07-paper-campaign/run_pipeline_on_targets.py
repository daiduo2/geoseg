#!/usr/bin/env python3
"""Run full agent pipeline with VLM review on confirmed targets.

Reads targets from vlm_confirmed_targets.json, runs controller.run_pipeline on each,
saves results and generates a report.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoseg.controller import run_pipeline

OUTPUT_ROOT = Path("runs/5_papers_vlm_test/pipeline_results")
REPORT_PATH = Path("runs/5_papers_vlm_test/pipeline_report.json")


def main() -> None:
    targets_path = Path("runs/5_papers_vlm_test/vlm_confirmed_targets.json")
    if not targets_path.exists():
        print(f"No targets file: {targets_path}")
        sys.exit(1)

    targets = json.loads(targets_path.read_text(encoding="utf-8"))
    print(f"Processing {len(targets)} confirmed targets...")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    report: list[dict] = []

    for i, target in enumerate(targets, 1):
        paper = target["paper"]
        fig_name = target["fig_name"]
        img_path = Path(target["path"])

        print(f"\n[{i}/{len(targets)}] {paper}/{fig_name}")
        if not img_path.exists():
            print(f"  [SKIP] Image not found: {img_path}")
            report.append({
                "paper": paper,
                "fig_name": fig_name,
                "status": "skipped",
                "reason": "image_not_found",
            })
            continue

        try:
            img_rgb = np.array(Image.open(img_path).convert("RGB"))
            result = run_pipeline(
                img_rgb,
                caption="",
                text_blocks=[],
                n_layers=5,
                quality_preference="balanced",
                skip_non_velocity_model=True,
                use_vlm=True,
                output_dir=OUTPUT_ROOT / f"{paper}_{fig_name}",
                save_intermediates=True,
            )

            status = result["status"]
            panels = result.get("panels", [])
            summary = result.get("summary", {})
            review_warnings = summary.get("review_warnings", [])
            n_processed = sum(1 for p in panels if p.get("status") == "ok")

            print(f"  status={status}, panels={len(panels)}, processed={n_processed}")
            if review_warnings:
                for w in review_warnings:
                    print(f"    WARN: {w}")

            report.append({
                "paper": paper,
                "fig_name": fig_name,
                "status": status,
                "classification": result.get("classification", {}),
                "n_panels": len(panels),
                "n_processed": n_processed,
                "review_warnings": review_warnings,
                "engines_used": summary.get("engines_used", []),
                "saturation_ratio": summary.get("saturation_ratio"),
                "vlm_has_colorbar": summary.get("vlm_has_colorbar"),
                "vlm_target_panel_id": summary.get("vlm_target_panel_id"),
            })

        except Exception as exc:
            print(f"  [ERROR] {exc}")
            traceback.print_exc()
            report.append({
                "paper": paper,
                "fig_name": fig_name,
                "status": "error",
                "reason": str(exc),
            })

        # Save report after each target
        REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # Summary
    n_ok = sum(1 for r in report if r["status"] == "ok")
    n_skipped = sum(1 for r in report if r["status"] == "skipped")
    n_error = sum(1 for r in report if r["status"] == "error")

    print(f"\n{'='*60}")
    print(f"Pipeline Complete")
    print(f"Total targets: {len(report)}")
    print(f"OK: {n_ok}, Skipped: {n_skipped}, Error: {n_error}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
