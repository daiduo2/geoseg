#!/usr/bin/env python3
"""Run full pipeline on VLM-classified targets from new papers.

Reads runs/new_papers_vlm/vlm_selective_results.json
Runs process_figure() with VLM review on each target
Saves results to runs/new_papers_vlm/pipeline_results.json

Usage:
    python3 scripts/run_pipeline_on_targets_new.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoseg.modules.segment_engines.full_pipeline import process_figure


def _convert_for_json(obj):
    """Recursively convert numpy types and other non-JSON-serializable objects."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_for_json(v) for v in obj]
    return obj

VLM_RESULTS_FILE = Path("runs/new_papers_vlm/vlm_selective_results.json")
PIPELINE_RESULTS_FILE = Path("runs/new_papers_vlm/pipeline_results.json")
TARGET_TYPES = {"velocity_model", "geological_cross_section"}


def main() -> None:
    if not VLM_RESULTS_FILE.exists():
        print(f"ERROR: {VLM_RESULTS_FILE} not found. Run classify_new_papers_selective.py first.")
        sys.exit(1)

    vlm_results = json.loads(VLM_RESULTS_FILE.read_text(encoding="utf-8"))
    targets = [r for r in vlm_results if r.get("is_target") and r.get("vlm_type") in TARGET_TYPES]

    print(f"Found {len(targets)} VLM-classified targets")
    for t in targets:
        print(f"  {t['fig_key']}: {t['vlm_type']} (conf={t['vlm_confidence']:.2f})")

    pipeline_results: list[dict] = []
    if PIPELINE_RESULTS_FILE.exists():
        pipeline_results = json.loads(PIPELINE_RESULTS_FILE.read_text(encoding="utf-8"))
    existing_pipeline = {r["fig_key"] for r in pipeline_results}

    print(f"\n{'='*60}")
    print(f"Running full pipeline on {len(targets)} targets")
    print(f"Already processed: {len(existing_pipeline)}")
    print(f"{'='*60}")

    for i, target in enumerate(targets, 1):
        fig_key = target["fig_key"]
        if fig_key in existing_pipeline:
            print(f"[{i}/{len(targets)}] [resume] {fig_key}")
            continue

        img_path = Path(target["img_path"])
        print(f"[{i}/{len(targets)}] Pipeline {fig_key} ...", end=" ", flush=True)

        try:
            img_rgb = np.array(Image.open(img_path).convert("RGB"))
            result = process_figure(
                img_rgb,
                caption="",
                n_layers=5,
                quality_preference="balanced",
                skip_non_velocity_model=True,
                use_vlm=True,
            )

            record = {
                "fig_key": fig_key,
                "paper": target["paper"],
                "fig_name": target["fig_name"],
                "img_path": str(img_path),
                "vlm_type": target["vlm_type"],
                "vlm_confidence": target["vlm_confidence"],
                "status": result["summary"]["status"],
                "reason": result["summary"].get("reason", ""),
                "n_panels": result["summary"].get("n_panels", 0),
                "total_layers": result["summary"].get("total_layers", 0),
                "engines_used": result["summary"].get("engines_used", []),
                "saturation_ratio": result["summary"].get("saturation_ratio", 0),
                "review_warnings": result["summary"].get("review_warnings", []),
                "vlm_has_colorbar": result["summary"].get("vlm_has_colorbar", False),
                "vlm_target_panel_id": result["summary"].get("vlm_target_panel_id", -1),
                "classification": result["classification"],
            }
            pipeline_results.append(record)
            existing_pipeline.add(fig_key)

            status = record["status"]
            n_layers = record.get("total_layers", 0)
            warnings = len(record.get("review_warnings", []))
            print(f"{status} (layers={n_layers}, warnings={warnings})")

            # Save incrementally
            PIPELINE_RESULTS_FILE.write_text(
                json.dumps(_convert_for_json(pipeline_results), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        except Exception as exc:
            print(f"ERROR: {exc}")
            record = {
                "fig_key": fig_key,
                "paper": target["paper"],
                "fig_name": target["fig_name"],
                "img_path": str(img_path),
                "vlm_type": target["vlm_type"],
                "vlm_confidence": target["vlm_confidence"],
                "status": "error",
                "reason": str(exc),
            }
            pipeline_results.append(record)
            existing_pipeline.add(fig_key)
            PIPELINE_RESULTS_FILE.write_text(
                json.dumps(_convert_for_json(pipeline_results), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    print(f"\n{'='*60}")
    print(f"Pipeline Complete")
    print(f"Results: {PIPELINE_RESULTS_FILE}")
    print(f"{'='*60}")

    ok_results = [r for r in pipeline_results if r["status"] == "ok"]
    skipped_results = [r for r in pipeline_results if r["status"] == "skipped"]
    error_results = [r for r in pipeline_results if r["status"] == "error"]

    print(f"\nSummary:")
    print(f"  Total targets: {len(targets)}")
    print(f"  OK: {len(ok_results)}")
    print(f"  Skipped: {len(skipped_results)}")
    print(f"  Errors: {len(error_results)}")

    print(f"\nPer-paper breakdown:")
    for paper in sorted(set(t["paper"] for t in targets)):
        paper_results = [r for r in pipeline_results if r["paper"] == paper]
        paper_ok = [r for r in paper_results if r["status"] == "ok"]
        paper_skipped = [r for r in paper_results if r["status"] == "skipped"]
        print(f"  {paper}: {len(paper_ok)} OK, {len(paper_skipped)} skipped")

    print(f"\nDetailed results:")
    for r in pipeline_results:
        warnings = r.get("review_warnings", [])
        warn_str = f", warnings={len(warnings)}" if warnings else ""
        print(f"  {r['fig_key']}: {r['status']} (reason={r.get('reason', 'N/A')}{warn_str})")


if __name__ == "__main__":
    main()
