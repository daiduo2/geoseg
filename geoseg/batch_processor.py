"""Batch process a directory of figure images through the full pipeline.

Features:
- Resume support (skip already-processed images)
- Per-image error isolation (one failure does not stop the batch)
- Structured JSON summary with decision audit trail

Usage:
    python -m geoseg.batch_processor \
        --images_dir runs/literature_test/gras2019/mineru/extracted/images \
        --output_dir runs/literature_test/gras2019/geoseg_export \
        --n_layers 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from geoseg.controller import run_pipeline


def _load_summary(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "summary.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def process_directory(
    images_dir: str | Path,
    output_dir: str | Path,
    n_layers: int = 5,
    quality_preference: str = "balanced",
    use_vlm: bool = True,
    properties_map: dict[str, dict] | None = None,
    resume: bool = True,
    skip_non_velocity_model: bool = True,
) -> dict[str, Any]:
    """Process all images in a directory through the full geoseg pipeline.

    Args:
        images_dir: Directory containing figure images (.jpg / .png).
        output_dir: Directory to save per-image artifacts and summary.
        n_layers: Number of layers to extract per panel.
        quality_preference: "fast", "balanced", or "best".
        use_vlm: Whether to use VLM for rep generation.
        properties_map: Optional custom property table.
        resume: If True, skip images already present in output_dir summary.
        skip_non_velocity_model: If True, skip observational_data and other.

    Returns:
        Summary dict with aggregate stats and per-image results.
    """
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if not image_files:
        return {"status": "no_images", "count": 0, "results": {}}

    # Load existing summary for resume
    existing = _load_summary(output_dir) if resume else {}
    existing_results = existing.get("results", {})

    results: dict[str, Any] = dict(existing_results)
    processed_count = 0
    skipped_count = 0
    error_count = 0

    print(f"Found {len(image_files)} images in {images_dir}")
    if resume and existing_results:
        print(f"Resuming: {len(existing_results)} already processed")

    for img_path in image_files:
        img_name = img_path.name
        if resume and img_name in existing_results:
            print(f"  [skip] {img_name} (already processed)")
            continue

        print(f"\nProcessing {img_name} ...")
        t0 = time.perf_counter()

        try:
            img = Image.open(img_path).convert("RGB")
            arr = np.array(img)

            img_out_dir = output_dir / img_path.stem
            result = run_pipeline(
                arr,
                n_layers=n_layers,
                quality_preference=quality_preference,
                skip_non_velocity_model=skip_non_velocity_model,
                use_vlm=use_vlm,
                properties_map=properties_map,
                output_dir=img_out_dir,
                save_intermediates=True,
            )
            elapsed = time.perf_counter() - t0

            if result["status"] == "ok":
                processed_count += 1
                ok_panels = sum(1 for p in result["panels"] if p["status"] == "ok")
                print(f"  OK: {ok_panels}/{len(result['panels'])} panels, {elapsed:.2f}s")
            elif result["status"] == "empty":
                skipped_count += 1
                print(f"  EMPTY: {result.get('reason', '')}")
            else:
                skipped_count += 1
                print(f"  SKIPPED: {result.get('reason', '')}")

            results[img_name] = {
                "status": result["status"],
                "reason": result.get("reason", ""),
                "classification": result["classification"]["figure_type"],
                "n_panels": result["summary"].get("n_panels", 0),
                "n_panels_processed": result["summary"].get("n_panels_processed", 0),
                "time_s": round(elapsed, 3),
            }

        except Exception as exc:
            error_count += 1
            elapsed = time.perf_counter() - t0
            print(f"  ERROR: {exc}")
            results[img_name] = {
                "status": "error",
                "reason": str(exc),
                "traceback": traceback.format_exc(),
                "time_s": round(elapsed, 3),
            }

        # Save incremental summary after each image
        _save_summary(output_dir, {
            "images_dir": str(images_dir),
            "output_dir": str(output_dir),
            "total": len(image_files),
            "processed": processed_count,
            "skipped": skipped_count,
            "errors": error_count,
            "config": {
                "n_layers": n_layers,
                "quality_preference": quality_preference,
                "use_vlm": use_vlm,
                "skip_non_velocity_model": skip_non_velocity_model,
            },
            "results": results,
        })

    summary = {
        "images_dir": str(images_dir),
        "output_dir": str(output_dir),
        "total": len(image_files),
        "processed": processed_count,
        "skipped": skipped_count,
        "errors": error_count,
        "config": {
            "n_layers": n_layers,
            "quality_preference": quality_preference,
            "use_vlm": use_vlm,
            "skip_non_velocity_model": skip_non_velocity_model,
        },
        "results": results,
    }

    _save_summary(output_dir, summary)
    print(f"\n{'='*50}")
    print(f"Total: {len(image_files)}, Processed: {processed_count}, Skipped: {skipped_count}, Errors: {error_count}")
    print(f"Summary: {output_dir / 'summary.json'}")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch process figures through geoseg pipeline")
    parser.add_argument("--images_dir", required=True, help="Directory containing figure images")
    parser.add_argument("--output_dir", required=True, help="Directory to save results")
    parser.add_argument("--n_layers", type=int, default=5)
    parser.add_argument("--quality", type=str, default="balanced", choices=["fast", "balanced", "best"])
    parser.add_argument("--no_vlm", action="store_true", help="Skip VLM calls")
    parser.add_argument("--no_resume", action="store_true", help="Re-process all images")
    parser.add_argument("--properties_json", type=str, default=None, help="Custom property table JSON")
    parser.add_argument("--skip_non_velocity", action="store_true", default=True,
                        help="Skip observational_data and other figure types")
    parser.add_argument("--no_skip_non_velocity", action="store_true", default=False,
                        help="Process all figure types")
    args = parser.parse_args()

    properties_map = None
    if args.properties_json:
        from geoseg.modules.post_process.properties import load_properties_json
        properties_map = load_properties_json(args.properties_json)

    process_directory(
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        n_layers=args.n_layers,
        quality_preference=args.quality,
        use_vlm=not args.no_vlm,
        properties_map=properties_map,
        resume=not args.no_resume,
        skip_non_velocity_model=not args.no_skip_non_velocity,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
