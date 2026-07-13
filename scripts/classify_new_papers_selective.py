#!/usr/bin/env python3
"""Selective VLM classification: only largest images per paper.

Picks top N largest images from each paper (most likely to be figures),
runs VLM with full reasoning, then runs full pipeline on targets.

Usage:
    python3 scripts/classify_new_papers_selective.py [--top_n 5]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoseg.modules.vlm_client.client import classify_figure
from geoseg.modules.segment_engines.full_pipeline import process_figure

PAPERS_DIR = Path("papers_new/to_process")
OUTPUT_ROOT = Path("runs/new_papers_vlm")
RESULTS_FILE = OUTPUT_ROOT / "vlm_selective_results.json"
PIPELINE_RESULTS_FILE = OUTPUT_ROOT / "pipeline_results.json"
TARGET_TYPES = {"velocity_model", "geological_cross_section"}


def get_top_images(paper_dir: Path, top_n: int = 5) -> list[Path]:
    """Get top N largest images from a paper directory."""
    images = list(paper_dir.glob("*.png")) + list(paper_dir.glob("*.jpg"))
    sized = []
    for img_path in images:
        try:
            with Image.open(img_path) as img:
                sized.append((img_path, img.size[0] * img.size[1]))
        except Exception:
            continue
    sized.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in sized[:top_n]]


def run_vlm_classification(img_path: Path, paper: str) -> dict:
    """Run VLM classification and return full record."""
    img_rgb = np.array(Image.open(img_path).convert("RGB"))
    result = classify_figure(img_rgb, mode="auto", min_confidence=0.0)

    return {
        "fig_key": f"{paper}/{img_path.stem}",
        "paper": paper,
        "fig_name": img_path.stem,
        "img_path": str(img_path),
        "vlm_type": result.figure_type,
        "vlm_confidence": result.confidence,
        "vlm_reason": result.reason,
        "vlm_segmentation_recommendation": getattr(
            result, "segmentation_recommendation", None
        ),
        "vlm_visual_features": getattr(result, "visual_features", None),
        "vlm_primary_evidence": getattr(result, "primary_evidence", None),
        "vlm_conflicting_evidence": getattr(result, "conflicting_evidence", None),
        "vlm_category_checklist": [
            {
                "category": c.category,
                "applicable": c.applicable,
                "evidence": c.evidence,
            }
            for c in getattr(result, "category_checklist", [])
        ],
        "is_target": result.figure_type in TARGET_TYPES,
    }


def run_pipeline(img_path: Path, paper: str, fig_name: str) -> dict:
    """Run full segmentation pipeline on an image."""
    img_rgb = np.array(Image.open(img_path).convert("RGB"))
    result = process_figure(
        img_rgb,
        caption="",
        n_layers=5,
        quality_preference="balanced",
        skip_non_velocity_model=True,
        use_vlm=True,
    )
    return {
        "fig_key": f"{paper}/{fig_name}",
        "paper": paper,
        "fig_name": fig_name,
        "img_path": str(img_path),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_n", type=int, default=5, help="Top N largest images per paper")
    parser.add_argument("--vlm_only", action="store_true", help="Only run VLM classification")
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Phase 1: VLM classification
    vlm_results: list[dict] = []
    if RESULTS_FILE.exists():
        vlm_results = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    existing_vlm = {r["fig_key"] for r in vlm_results}

    all_targets: list[dict] = []

    for paper_dir in sorted(PAPERS_DIR.iterdir()):
        if not paper_dir.is_dir():
            continue
        paper = paper_dir.name
        top_images = get_top_images(paper_dir, args.top_n)
        print(f"\n{'='*60}")
        print(f"Paper: {paper} ({len(top_images)} images)")
        print(f"{'='*60}")

        paper_targets = []
        for img_path in top_images:
            fig_key = f"{paper}/{img_path.stem}"
            if fig_key in existing_vlm:
                print(f"  [resume] {img_path.name}")
                continue

            print(f"  VLM {img_path.name} ...", end=" ", flush=True)
            try:
                record = run_vlm_classification(img_path, paper)
                vlm_results.append(record)
                existing_vlm.add(fig_key)

                marker = "TARGET" if record["is_target"] else "skip"
                print(f"{record['vlm_type']} (conf={record['vlm_confidence']:.2f}) [{marker}]")

                if record["is_target"]:
                    paper_targets.append(record)
                    all_targets.append(record)

                # Save after each classification
                RESULTS_FILE.write_text(
                    json.dumps(vlm_results, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception as exc:
                print(f"ERROR: {exc}")

        print(f"  -> {len(paper_targets)} targets found")

    print(f"\n{'='*60}")
    print(f"VLM Classification Complete")
    print(f"Total targets: {len(all_targets)}")
    print(f"Results: {RESULTS_FILE}")
    print(f"{'='*60}")

    for t in all_targets:
        print(f"  {t['fig_key']}: {t['vlm_type']} (conf={t['vlm_confidence']:.2f})")

    if args.vlm_only:
        return

    # Phase 2: Full pipeline on targets
    pipeline_results: list[dict] = []
    if PIPELINE_RESULTS_FILE.exists():
        pipeline_results = json.loads(PIPELINE_RESULTS_FILE.read_text(encoding="utf-8"))
    existing_pipeline = {r["fig_key"] for r in pipeline_results}

    print(f"\n{'='*60}")
    print(f"Running full pipeline on {len(all_targets)} targets")
    print(f"{'='*60}")

    for target in all_targets:
        fig_key = target["fig_key"]
        if fig_key in existing_pipeline:
            print(f"  [resume] {fig_key}")
            continue

        print(f"  Pipeline {fig_key} ...", end=" ", flush=True)
        try:
            result = run_pipeline(
                Path(target["img_path"]),
                target["paper"],
                target["fig_name"],
            )
            pipeline_results.append(result)
            existing_pipeline.add(fig_key)

            status = result["status"]
            n_layers = result.get("total_layers", 0)
            warnings = len(result.get("review_warnings", []))
            print(f"{status} (layers={n_layers}, warnings={warnings})")

            PIPELINE_RESULTS_FILE.write_text(
                json.dumps(pipeline_results, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            print(f"ERROR: {exc}")

    print(f"\n{'='*60}")
    print(f"Pipeline Complete")
    print(f"Results: {PIPELINE_RESULTS_FILE}")
    print(f"{'='*60}")

    ok_results = [r for r in pipeline_results if r["status"] == "ok"]
    skipped_results = [r for r in pipeline_results if r["status"] == "skipped"]
    print(f"OK: {len(ok_results)}, Skipped: {len(skipped_results)}")

    for r in pipeline_results:
        warnings = r.get("review_warnings", [])
        warn_str = f", warnings={len(warnings)}" if warnings else ""
        print(f"  {r['fig_key']}: {r['status']} (reason={r.get('reason', 'N/A')}{warn_str})")


if __name__ == "__main__":
    main()
