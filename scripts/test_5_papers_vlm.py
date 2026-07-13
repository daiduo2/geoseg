#!/usr/bin/env python3
"""Test 5 papers with VLM classification + full pipeline review.

Strategy:
1. For each paper, sort figures by size (larger = more likely main figure)
2. Run VLM classify on top N figures per paper
3. Run full pipeline with VLM review on confirmed targets
4. Collect findings and generate report
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoseg.controller import run_pipeline
from geoseg.modules.vlm_client.client import classify_figure

PAPERS: dict[str, str] = {
    "khosroanjom2024": "runs/literature_test/khosroanjom2024/mineru/extracted/images",
    "wudalianchi2025": "runs/literature_test/wudalianchi2025/mineru/extracted/images",
    "franz2023": "runs/literature_test/franz2023/mineru/extracted/images",
    "paffrath_2021": "runs/literature_test/paffrath_2021/mineru/extracted/images",
    "zailac2023": "runs/literature_test/zailac2023/mineru/extracted/images",
}

OUTPUT_ROOT = Path("runs/5_papers_vlm_test")
VLM_CACHE = Path("runs/5_papers_vlm_test/vlm_classification_results.json")
MAX_CLASSIFY_PER_PAPER = 12  # Top N largest figures to classify


def load_cache() -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if VLM_CACHE.exists():
        data = json.loads(VLM_CACHE.read_text(encoding="utf-8"))
        for record in data:
            key = f"{record['paper']}/{record['fig_name']}"
            cache[key] = record
    return cache


def save_cache(cache_data: list[dict]) -> None:
    VLM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    VLM_CACHE.write_text(json.dumps(cache_data, indent=2, default=str), encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cache = load_cache()
    cache_data: list[dict] = list(cache.values())

    all_targets: list[dict] = []

    for paper, img_dir in PAPERS.items():
        img_path = Path(img_dir)
        if not img_path.exists():
            print(f"\n[SKIP] {paper}: directory not found")
            continue

        images = sorted(img_path.glob("*.png")) + sorted(img_path.glob("*.jpg"))
        print(f"\n[{paper}] {len(images)} total figures")

        # Sort by file size (larger = more likely main figure)
        images_with_size = [(p, p.stat().st_size) for p in images]
        images_with_size.sort(key=lambda x: x[1], reverse=True)
        to_classify = images_with_size[:MAX_CLASSIFY_PER_PAPER]

        paper_targets: list[dict] = []

        for img_file, size in to_classify:
            fig_name = img_file.stem
            cache_key = f"{paper}/{fig_name}"
            cached = cache.get(cache_key)

            if cached:
                cls_result = cached
                print(f"  {fig_name}: CACHED {cls_result['vlm_type']} (conf={cls_result['vlm_confidence']:.2f})")
            else:
                try:
                    img_rgb = np.array(Image.open(img_file).convert("RGB"))
                    vlm_result = classify_figure(img_rgb, mode="auto")
                    cls_result = {
                        "paper": paper,
                        "fig_name": fig_name,
                        "vlm_type": vlm_result.figure_type,
                        "vlm_confidence": vlm_result.confidence,
                        "vlm_reason": vlm_result.reason,
                    }
                    cache_data.append(cls_result)
                    cache[cache_key] = cls_result
                    print(f"  {fig_name}: {cls_result['vlm_type']} (conf={cls_result['vlm_confidence']:.2f})")
                except Exception as exc:
                    print(f"  {fig_name}: VLM ERROR {exc}")
                    continue

            # Save cache after each figure
            save_cache(cache_data)

            if cls_result["vlm_type"] in ("velocity_model", "geological_cross_section"):
                paper_targets.append({
                    "paper": paper,
                    "fig_name": fig_name,
                    "path": str(img_file),
                    "vlm_confidence": cls_result["vlm_confidence"],
                    "vlm_reason": cls_result["vlm_reason"],
                })

        print(f"  -> {len(paper_targets)} VLM-confirmed targets")
        all_targets.extend(paper_targets)

    # Save targets list
    targets_path = OUTPUT_ROOT / "vlm_confirmed_targets.json"
    targets_path.write_text(json.dumps(all_targets, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"VLM Classification Complete")
    print(f"Total figures classified: {len(cache_data)}")
    print(f"VLM-confirmed targets: {len(all_targets)}")
    print(f"Cache: {VLM_CACHE}")
    print(f"Targets list: {targets_path}")


if __name__ == "__main__":
    main()
