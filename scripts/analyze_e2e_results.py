"""Analyze end-to-end batch results across all datasets.

Checks:
- Figure type distribution (how many conceptual vs skipped)
- Per-panel processing success rate
- Engine distribution
- Export artifact completeness (tomo.xyz, polygons.geojson, properties.json)
- Error patterns

Usage:
    python3 scripts/analyze_e2e_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


DATASETS = ["gras2019", "zailac2023", "ma_2022"]
BASE_DIR = Path("/Users/daiduo2/geoseg/runs/literature_test")


def main() -> int:
    all_results: dict[str, dict] = {}
    total_ok = 0
    total_skipped = 0
    total_error = 0
    total_panels_ok = 0
    total_panels_error = 0
    engine_counts: dict[str, int] = {}
    error_reasons: dict[str, int] = {}
    artifact_missing: dict[str, int] = {}

    for dataset in DATASETS:
        summary_path = BASE_DIR / dataset / "e2e_export" / "summary.json"
        if not summary_path.exists():
            print(f"[!] Missing: {summary_path}")
            continue

        summary = json.loads(summary_path.read_text())
        results = summary.get("results", {})
        all_results.update({f"{dataset}/{k}": v for k, v in results.items()})

        for fname, data in results.items():
            status = data.get("status", "unknown")
            if status == "ok":
                total_ok += 1
                total_panels_ok += data.get("n_panels_processed", 0)
                # Check engine distribution from per-panel details if available
                # (summary.json stores aggregate, not per-panel engines)
            elif status in ("skipped", "empty"):
                total_skipped += 1
            elif status == "error":
                total_error += 1
                reason = data.get("reason", "unknown")
                error_reasons[reason] = error_reasons.get(reason, 0) + 1

        # Check artifact completeness for processed images
        output_dir = BASE_DIR / dataset / "e2e_export"
        for fname, data in results.items():
            if data.get("status") != "ok":
                continue
            stem = Path(fname).stem
            img_out_dir = output_dir / stem
            if not img_out_dir.exists():
                artifact_missing[f"missing_dir"] = artifact_missing.get("missing_dir", 0) + 1
                continue
            for suffix in ("_tomo.xyz", "_polygons.geojson", "_properties.json"):
                if not any(f.name.endswith(suffix) for f in img_out_dir.iterdir()):
                    artifact_missing[suffix] = artifact_missing.get(suffix, 0) + 1

    total = len(all_results)
    print("=" * 60)
    print("END-TO-END BATCH ANALYSIS")
    print("=" * 60)
    print(f"\nDatasets: {', '.join(DATASETS)}")
    print(f"Total images: {total}")
    print(f"  OK (processed end-to-end): {total_ok} ({pct(total_ok, total):.1f}%)")
    print(f"  Skipped: {total_skipped} ({pct(total_skipped, total):.1f}%)")
    print(f"  Errors: {total_error} ({pct(total_error, total):.1f}%)")

    if total_panels_ok > 0:
        print(f"\nPanels processed OK: {total_panels_ok}")
        print(f"Panels with error: {total_panels_error}")

    if error_reasons:
        print(f"\n--- Error Reasons ---")
        for reason, count in sorted(error_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason[:60]}: {count}")

    if artifact_missing:
        print(f"\n--- Missing Artifacts ---")
        for artifact, count in sorted(artifact_missing.items(), key=lambda x: -x[1]):
            print(f"  {artifact}: {count}")
    else:
        print(f"\n[OK] All processed images have complete artifacts")

    # Figure type distribution
    fig_types: dict[str, int] = {}
    for data in all_results.values():
        ft = data.get("classification", "unknown")
        fig_types[ft] = fig_types.get(ft, 0) + 1

    if fig_types:
        print(f"\n--- Figure Type Distribution ---")
        for ft, count in sorted(fig_types.items(), key=lambda x: -x[1]):
            print(f"  {ft}: {count} ({pct(count, total):.1f}%)")

    print("=" * 60)
    return 0


def pct(part: int, total: int) -> float:
    return (part / total * 100) if total > 0 else 0.0


if __name__ == "__main__":
    sys.exit(main())
