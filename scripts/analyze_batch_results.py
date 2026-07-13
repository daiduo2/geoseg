"""Analyze batch test results across all datasets.

Usage:
    python3 scripts/analyze_batch_results.py
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    base_dir = Path("/Users/daiduo2/geoseg/runs/literature_test")
    datasets = ["gras2019", "zailac2023", "ma_2022"]

    all_engines: dict[str, int] = {}
    all_figure_types: dict[str, int] = {}
    vivid_total = 0
    vivid_non_v4 = 0
    total_images = 0
    total_processed = 0
    total_skipped = 0
    total_panels = 0

    for dataset in datasets:
        summary_path = base_dir / dataset / "segment_results_vlm" / "summary.json"
        if not summary_path.exists():
            print(f"[!] Missing: {summary_path}")
            continue

        summary = json.loads(summary_path.read_text())
        total_images += len(summary)

        for fname, data in summary.items():
            all_figure_types[data.get("figure_type", "unknown")] = (
                all_figure_types.get(data.get("figure_type", "unknown"), 0) + 1
            )

            if data.get("status") == "skipped":
                total_skipped += 1
                continue

            total_processed += 1
            sat = data.get("saturation", 0)
            panels = data.get("panels", [])
            total_panels += len(panels)

            for p in panels:
                if p.get("skipped"):
                    continue
                engine = p.get("engine", "unknown")
                all_engines[engine] = all_engines.get(engine, 0) + 1

                if sat >= 0.5 and engine != "v4_kmeans":
                    vivid_non_v4 += 1
                if sat >= 0.5:
                    vivid_total += 1

    print("=" * 60)
    print("BATCH TEST ANALYSIS (VLM enabled)")
    print("=" * 60)
    print(f"\nDatasets: {', '.join(datasets)}")
    print(f"Total images: {total_images}")
    print(f"Processed: {total_processed}")
    print(f"Skipped: {total_skipped}")
    print(f"Total panels: {total_panels}")

    print(f"\n--- Figure Types ---")
    for ft, count in sorted(all_figure_types.items(), key=lambda x: -x[1]):
        print(f"  {ft}: {count}")

    print(f"\n--- Engine Distribution ---")
    for engine, count in sorted(all_engines.items(), key=lambda x: -x[1]):
        pct = count / total_panels * 100 if total_panels > 0 else 0
        print(f"  {engine}: {count} ({pct:.1f}%)")

    print(f"\n--- Vivid Panel Routing ---")
    print(f"  Vivid panels (sat >= 0.5): {vivid_total}")
    print(f"  Vivid -> non-v4 engines: {vivid_non_v4}")
    if vivid_total > 0:
        print(f"  Non-v4 ratio: {vivid_non_v4 / vivid_total * 100:.1f}%")
    else:
        print(f"  Non-v4 ratio: N/A")

    # Check if all engines were exercised
    expected_engines = {"grayscale", "v4_kmeans", "edge_guided", "edge_grow", "kmeans_full", "ensemble"}
    exercised = set(all_engines.keys())

    # Also check best-quality runs for ensemble
    for dataset in datasets:
        best_path = base_dir / dataset / "segment_results_ensemble" / "summary.json"
        if best_path.exists():
            best_summary = json.loads(best_path.read_text())
            for data in best_summary.values():
                if data.get("status") == "ok":
                    for p in data.get("panels", []):
                        if not p.get("skipped"):
                            engine = p.get("engine", "unknown")
                            exercised.add(engine)
                            all_engines[engine] = all_engines.get(engine, 0) + 1

    missing = expected_engines - exercised
    if missing:
        print(f"\n[!] MISSING ENGINES: {', '.join(missing)}")
    else:
        print(f"\n[OK] All expected engines exercised")
        print(f"  Final engine counts including best-quality runs:")
        for engine, count in sorted(all_engines.items(), key=lambda x: -x[1]):
            print(f"    {engine}: {count}")

    print("=" * 60)


if __name__ == "__main__":
    main()
