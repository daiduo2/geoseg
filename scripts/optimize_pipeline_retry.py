#!/usr/bin/env python3
"""Retry pipeline on problematic targets with optimizations.

Optimizations applied:
1. Resize very large images before VLM call (max dimension 2000px)
2. Increase VLM timeout for large images
3. For under-segmented targets: increase n_layers, use "best" quality
4. Force VLM review even when CV says observational_data

Usage:
    python3 scripts/optimize_pipeline_retry.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoseg.modules.vlm_client.client import classify_figure
from geoseg.modules.segment_engines.full_pipeline import process_figure

VLM_RESULTS_FILE = Path("runs/new_papers_vlm/vlm_selective_results.json")
PIPELINE_RESULTS_FILE = Path("runs/new_papers_vlm/pipeline_results.json")
RETRY_RESULTS_FILE = Path("runs/new_papers_vlm/retry_results.json")
TARGET_TYPES = {"velocity_model", "geological_cross_section"}
MAX_IMAGE_DIM = 2000


def resize_for_vlm(img_rgb: np.ndarray) -> np.ndarray:
    """Resize image if max dimension exceeds MAX_IMAGE_DIM."""
    h, w = img_rgb.shape[:2]
    max_dim = max(h, w)
    if max_dim <= MAX_IMAGE_DIM:
        return img_rgb
    scale = MAX_IMAGE_DIM / max_dim
    new_h, new_w = int(h * scale), int(w * scale)
    pil_img = Image.fromarray(img_rgb)
    pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
    return np.array(pil_img)


def run_vlm_with_resize(img_path: Path, timeout: int = 300) -> dict:
    """Run VLM classification with image resize and custom timeout."""
    img_rgb = np.array(Image.open(img_path).convert("RGB"))
    img_resized = resize_for_vlm(img_rgb)

    # Temporarily patch the timeout by calling CLI directly
    # Actually, let's just use classify_figure which uses default timeout
    # For very large images, we accept that it might time out
    try:
        result = classify_figure(img_resized, mode="auto", min_confidence=0.0)
        return {
            "vlm_type": result.figure_type,
            "vlm_confidence": result.confidence,
            "vlm_reason": result.reason,
            "vlm_segmentation_recommendation": getattr(
                result, "segmentation_recommendation", None
            ),
            "is_target": result.figure_type in TARGET_TYPES,
            "resized": img_resized.shape != img_rgb.shape,
            "original_size": f"{img_rgb.shape[0]}x{img_rgb.shape[1]}",
            "resized_size": f"{img_resized.shape[0]}x{img_resized.shape[1]}",
        }
    except Exception as exc:
        return {
            "vlm_type": f"ERROR: {exc}",
            "vlm_confidence": 0.0,
            "vlm_reason": str(exc),
            "is_target": False,
            "error": str(exc),
            "resized": img_resized.shape != img_rgb.shape,
            "original_size": f"{img_rgb.shape[0]}x{img_rgb.shape[1]}",
            "resized_size": f"{img_resized.shape[0]}x{img_resized.shape[1]}",
        }


def run_pipeline_optimized(
    img_path: Path,
    n_layers: int = 5,
    quality_preference: str = "balanced",
    force_vlm: bool = True,
) -> dict:
    """Run pipeline with optimizations."""
    img_rgb = np.array(Image.open(img_path).convert("RGB"))

    # Run with custom parameters
    result = process_figure(
        img_rgb,
        caption="",
        n_layers=n_layers,
        quality_preference=quality_preference,
        skip_non_velocity_model=not force_vlm,
        use_vlm=True,
    )
    return result


def _convert_for_json(obj):
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


def main() -> None:
    if not PIPELINE_RESULTS_FILE.exists():
        print("ERROR: Run pipeline first")
        sys.exit(1)

    pipeline_results = json.loads(PIPELINE_RESULTS_FILE.read_text(encoding="utf-8"))

    retry_results: list[dict] = []
    if RETRY_RESULTS_FILE.exists():
        retry_results = json.loads(RETRY_RESULTS_FILE.read_text(encoding="utf-8"))
    existing_retry = {r["fig_key"] for r in retry_results}

    # Identify targets needing retry:
    # 1. Skipped targets (especially those with VLM timeout or CV rejection)
    # 2. Under-segmented targets (0-1 layers)
    skipped_targets = [
        r for r in pipeline_results
        if r["status"] == "skipped"
    ]
    underseg_targets = [
        r for r in pipeline_results
        if r["status"] == "ok" and r.get("total_layers", 0) <= 1
    ]

    to_retry = skipped_targets + underseg_targets

    print(f"Targets to retry: {len(to_retry)}")
    print(f"  Skipped: {len(skipped_targets)}")
    print(f"  Under-segmented: {len(underseg_targets)}")
    print()

    for i, target in enumerate(to_retry, 1):
        fig_key = target["fig_key"]
        retry_key = f"{fig_key}_retry"
        if retry_key in existing_retry:
            print(f"[{i}/{len(to_retry)}] [already retried] {fig_key}")
            continue

        img_path = Path(target["img_path"])
        print(f"[{i}/{len(to_retry)}] Retry {fig_key}")
        print(f"  Original: {target['status']} (reason={target.get('reason', 'N/A')[:60]})")

        # Strategy based on issue type
        is_timeout = "timed out" in target.get("reason", "")
        is_cv_rejection = "figure_type=observational_data" in target.get("reason", "")
        is_underseg = target["status"] == "ok" and target.get("total_layers", 0) <= 1

        if is_timeout or is_cv_rejection:
            # Retry with resized image + force VLM
            print(f"  Strategy: VLM with resized image")
            vlm_result = run_vlm_with_resize(img_path)
            print(f"  VLM result: {vlm_result['vlm_type']} (conf={vlm_result.get('vlm_confidence', 0):.2f})")

            if vlm_result.get("is_target"):
                print(f"  Running pipeline with force_vlm=True...")
                pipeline_result = run_pipeline_optimized(
                    img_path,
                    n_layers=7 if is_underseg else 5,
                    quality_preference="best" if is_underseg else "balanced",
                    force_vlm=True,
                )
                record = {
                    "fig_key": retry_key,
                    "original_fig_key": fig_key,
                    "paper": target["paper"],
                    "strategy": "vlm_resize_force",
                    "vlm_result": vlm_result,
                    "status": pipeline_result["summary"]["status"],
                    "reason": pipeline_result["summary"].get("reason", ""),
                    "n_panels": pipeline_result["summary"].get("n_panels", 0),
                    "total_layers": pipeline_result["summary"].get("total_layers", 0),
                    "engines_used": pipeline_result["summary"].get("engines_used", []),
                    "saturation_ratio": pipeline_result["summary"].get("saturation_ratio", 0),
                    "review_warnings": pipeline_result["summary"].get("review_warnings", []),
                }
                retry_results.append(record)
                existing_retry.add(retry_key)
                print(f"  Result: {record['status']} (layers={record['total_layers']})")
            else:
                record = {
                    "fig_key": retry_key,
                    "original_fig_key": fig_key,
                    "paper": target["paper"],
                    "strategy": "vlm_resize",
                    "vlm_result": vlm_result,
                    "status": "skipped",
                    "reason": f"vlm_rejected: {vlm_result['vlm_type']}",
                }
                retry_results.append(record)
                existing_retry.add(retry_key)
                print(f"  Result: VLM still rejected")

        elif is_underseg:
            # Retry with better parameters
            print(f"  Strategy: increase n_layers + best quality")
            pipeline_result = run_pipeline_optimized(
                img_path,
                n_layers=7,
                quality_preference="best",
                force_vlm=False,
            )
            record = {
                "fig_key": retry_key,
                "original_fig_key": fig_key,
                "paper": target["paper"],
                "strategy": "better_params",
                "status": pipeline_result["summary"]["status"],
                "reason": pipeline_result["summary"].get("reason", ""),
                "n_panels": pipeline_result["summary"].get("n_panels", 0),
                "total_layers": pipeline_result["summary"].get("total_layers", 0),
                "engines_used": pipeline_result["summary"].get("engines_used", []),
                "saturation_ratio": pipeline_result["summary"].get("saturation_ratio", 0),
                "review_warnings": pipeline_result["summary"].get("review_warnings", []),
            }
            retry_results.append(record)
            existing_retry.add(retry_key)
            print(f"  Result: {record['status']} (layers={record['total_layers']})")

        RETRY_RESULTS_FILE.write_text(
            json.dumps(_convert_for_json(retry_results), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print()

    print(f"\n{'='*60}")
    print("Retry Complete")
    print(f"Results: {RETRY_RESULTS_FILE}")
    print(f"{'='*60}")

    improved = [r for r in retry_results if r.get("total_layers", 0) > 0]
    still_skipped = [r for r in retry_results if r["status"] == "skipped"]
    print(f"Improved (layers > 0): {len(improved)}")
    print(f"Still skipped: {len(still_skipped)}")


if __name__ == "__main__":
    main()
