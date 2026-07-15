#!/usr/bin/env python3
"""Batch VLM classification on new papers with full reasoning capture.

Saves all new schema fields (visual_features, category_checklist, etc.)
for post-hoc analysis of VLM decision patterns.

Usage:
    python3 scripts/classify_new_papers.py

Outputs:
    runs/new_papers_vlm/vlm_classification_results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoseg.experiments import classify_figure

PAPERS_DIR = Path("papers_new/to_process")
OUTPUT_ROOT = Path("runs/new_papers_vlm")
RESULTS_FILE = OUTPUT_ROOT / "vlm_classification_results.json"
TARGET_TYPES = {"velocity_model", "geological_cross_section"}


def load_existing_results() -> dict[str, dict]:
    if RESULTS_FILE.exists():
        data = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
        return {r["fig_key"]: r for r in data}
    return {}


def save_results(results: list[dict]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(RESULTS_FILE)


def main() -> None:
    existing = load_existing_results()
    results: list[dict] = list(existing.values())

    all_figures: list[tuple[str, Path]] = []
    for paper_dir in sorted(PAPERS_DIR.iterdir()):
        if not paper_dir.is_dir():
            continue
        images = sorted(paper_dir.glob("*.png")) + sorted(paper_dir.glob("*.jpg"))
        for img_file in images:
            all_figures.append((paper_dir.name, img_file))

    total = len(all_figures)
    print(f"Total figures to classify: {total}")
    print(f"Already processed (resume): {len(existing)}")
    print(f"Remaining: {total - len(existing)}")
    print()

    processed = 0
    errors = 0

    for i, (paper, img_file) in enumerate(all_figures, 1):
        fig_name = img_file.stem
        fig_key = f"{paper}/{fig_name}"

        if fig_key in existing:
            continue

        print(f"[{i}/{total}] VLM {fig_key} ...", end=" ", flush=True)

        try:
            img_rgb = np.array(Image.open(img_file).convert("RGB"))
            result = classify_figure(img_rgb, mode="auto", min_confidence=0.0)

            record = {
                "fig_key": fig_key,
                "paper": paper,
                "fig_name": fig_name,
                "img_path": str(img_file),
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
            results.append(record)
            existing[fig_key] = record
            processed += 1

            marker = "TARGET" if record["is_target"] else "skip"
            print(f"{result.figure_type} (conf={result.confidence:.2f}, rec={record['vlm_segmentation_recommendation']}) [{marker}]")

            if processed % 3 == 0:
                save_results(results)

        except Exception as exc:
            errors += 1
            record = {
                "fig_key": fig_key,
                "paper": paper,
                "fig_name": fig_name,
                "img_path": str(img_file),
                "vlm_type": f"ERROR: {exc}",
                "vlm_confidence": 0.0,
                "vlm_reason": str(exc),
                "is_target": False,
            }
            results.append(record)
            existing[fig_key] = record
            print(f"ERROR: {exc}")

    save_results(results)

    targets = [r for r in results if r.get("is_target")]
    errors_list = [r for r in results if r["vlm_type"].startswith("ERROR")]

    print(f"\n{'='*60}")
    print(f"Total: {total}")
    print(f"Processed: {processed}")
    print(f"Errors: {len(errors_list)}")
    print(f"Targets: {len(targets)}")
    print(f"Results: {RESULTS_FILE}")
    print()

    for paper_dir in sorted(PAPERS_DIR.iterdir()):
        if not paper_dir.is_dir():
            continue
        paper = paper_dir.name
        paper_targets = [r for r in targets if r["paper"] == paper]
        paper_all = [r for r in results if r["paper"] == paper and not r["vlm_type"].startswith("ERROR")]
        print(f"{paper}: {len(paper_targets)}/{len(paper_all)} targets")
        for r in paper_targets:
            print(f"  {r['fig_name']}: {r['vlm_type']} (conf={r['vlm_confidence']:.2f})")


if __name__ == "__main__":
    main()
