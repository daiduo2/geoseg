#!/usr/bin/env python3
"""Batch VLM figure classification: run on ALL literature figures.

Usage:
    cd /Users/daiduo2/geoseg
    .venv/bin/python scripts/batch_vlm_classify.py

Outputs:
    runs/agent_review/vlm_classification_results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoseg.modules.vlm_client.client import classify_figure

PAPERS: dict[str, str] = {
    "ph01": "runs/M0.5/images",
    "gras2019": "runs/literature_test/gras2019/mineru/extracted/images",
    "ma_2022": "runs/literature_test/ma_2022/mineru/extracted/images",
    "zailac2023": "runs/literature_test/zailac2023/mineru/extracted/images",
    "silixa2021": "runs/literature_test/silixa2021/mineru/extracted/images",
}

OUTPUT_ROOT = Path("runs/agent_review")
RESULTS_FILE = OUTPUT_ROOT / "vlm_classification_results.json"
TARGET_TYPES = {"velocity_model", "geological_cross_section"}


def load_existing_results() -> dict[str, dict]:
    """Load existing results for resume support."""
    if RESULTS_FILE.exists():
        data = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
        return {r["fig_key"]: r for r in data}
    return {}


def save_results(results: list[dict]) -> None:
    """Atomically save results to JSON."""
    tmp = RESULTS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(RESULTS_FILE)


def main() -> None:
    existing = load_existing_results()
    results: list[dict] = list(existing.values())

    total = 0
    skipped = 0
    processed = 0
    errors = 0

    all_figures: list[tuple[str, Path]] = []
    for paper, img_dir in PAPERS.items():
        img_path = Path(img_dir)
        if not img_path.exists():
            print(f"[SKIP] {paper}: directory not found: {img_dir}")
            continue
        images = sorted(img_path.glob("*.png")) + sorted(img_path.glob("*.jpg"))
        for img_file in images:
            all_figures.append((paper, img_file))

    total = len(all_figures)
    print(f"Total figures to classify: {total}")
    print(f"Already processed (resume): {len(existing)}")
    print(f"Remaining: {total - len(existing)}")
    print()

    for i, (paper, img_file) in enumerate(all_figures, 1):
        fig_name = img_file.stem
        fig_key = f"{paper}/{fig_name}"

        if fig_key in existing:
            skipped += 1
            continue

        print(f"[{i}/{total}] VLM {fig_key} ...", end=" ", flush=True)

        try:
            img_rgb = np.array(Image.open(img_file).convert("RGB"))
            result = classify_figure(img_rgb, mode="auto")

            record = {
                "fig_key": fig_key,
                "paper": paper,
                "fig_name": fig_name,
                "vlm_type": result.figure_type,
                "vlm_confidence": result.confidence,
                "vlm_reason": result.reason,
                "is_target": result.figure_type in TARGET_TYPES,
            }
            results.append(record)
            existing[fig_key] = record
            processed += 1

            marker = "TARGET" if record["is_target"] else "skip"
            print(f"{result.figure_type} (conf={result.confidence:.2f}) [{marker}]")

            # Save every 5 processed figures for crash recovery
            if processed % 5 == 0:
                save_results(results)

        except Exception as exc:
            errors += 1
            record = {
                "fig_key": fig_key,
                "paper": paper,
                "fig_name": fig_name,
                "vlm_type": f"ERROR: {exc}",
                "vlm_confidence": 0.0,
                "vlm_reason": str(exc),
                "is_target": False,
            }
            results.append(record)
            existing[fig_key] = record
            print(f"ERROR: {exc}")

    # Final save
    save_results(results)

    # Summary
    targets = [r for r in results if r.get("is_target")]
    errors_list = [r for r in results if r["vlm_type"].startswith("ERROR")]

    print(f"\n{'='*60}")
    print(f"Total: {total}")
    print(f"Processed: {processed}")
    print(f"Skipped (resume): {skipped}")
    print(f"Errors: {len(errors_list)}")
    print(f"Targets (velocity_model + geological_cross_section): {len(targets)}")
    print(f"Results: {RESULTS_FILE}")
    print()

    for paper in PAPERS:
        paper_targets = [r for r in targets if r["paper"] == paper]
        if paper_targets:
            print(f"\n--- {paper}: {len(paper_targets)} targets ---")
            for r in paper_targets:
                print(f"  {r['fig_name']}: {r['vlm_type']} (conf={r['vlm_confidence']:.2f})")


if __name__ == "__main__":
    main()
