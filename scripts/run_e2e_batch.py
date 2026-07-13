"""Run full end-to-end pipeline on all literature test datasets.

Processes: classify -> segment -> post_process -> export SPECFEM
Outputs: per-image artifacts + summary.json in runs/literature_test/*/e2e_export/

Usage:
    python3 scripts/run_e2e_batch.py [--quality fast|balanced|best] [--no_vlm]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from geoseg.batch_processor import process_directory


DATASETS = ["gras2019", "zailac2023", "ma_2022"]
BASE_DIR = Path("/Users/daiduo2/geoseg/runs/literature_test")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run end-to-end pipeline on literature datasets")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, help="Datasets to process")
    parser.add_argument("--quality", type=str, default="balanced", choices=["fast", "balanced", "best"])
    parser.add_argument("--no_vlm", action="store_true", help="Skip VLM calls")
    parser.add_argument("--no_resume", action="store_true", help="Re-process all images")
    parser.add_argument("--properties_json", type=str, default=None, help="Custom property table")
    args = parser.parse_args()

    properties_map = None
    if args.properties_json:
        from geoseg.modules.post_process.properties import load_properties_json
        properties_map = load_properties_json(args.properties_json)

    for dataset in args.datasets:
        images_dir = BASE_DIR / dataset / "mineru" / "extracted" / "images"
        if not images_dir.exists():
            print(f"[!] Images dir not found: {images_dir}")
            continue

        output_dir = BASE_DIR / dataset / "e2e_export"
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"Images:  {images_dir}")
        print(f"Output:  {output_dir}")
        print(f"{'='*60}")

        process_directory(
            images_dir=images_dir,
            output_dir=output_dir,
            n_layers=5,
            quality_preference=args.quality,
            use_vlm=not args.no_vlm,
            properties_map=properties_map,
            resume=not args.no_resume,
            skip_non_velocity_model=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
