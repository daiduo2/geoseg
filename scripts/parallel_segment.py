"""Single-panel sandbox-segment agent script.

Usage:
    python3 parallel_segment.py --image <path> --panel-id <id>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from geoseg.modules.segment_engines.metrics import compute_all
from geoseg.modules.segment_engines.runner import run_engine
from geoseg.modules.segment_engines.strategy_memory import record_attempt
from geoseg.core.image_ops import (
    saturation_ratio,
    adaptive_blur,
    estimate_noise_level,
    create_overlay,
)
from geoseg.modules.segment_engines.vlm_reps import color_zones_to_reps
from geoseg.modules.cv_detect.colorbar_extractor import extract_colorbar


def _compute_score(metrics: dict) -> float:
    """Heuristic score for comparing segmentation results."""
    n_found = metrics["n_layers"]
    ba = metrics["boundary_alignment"]
    frag = metrics["total_fragment_area_fraction"]
    frag_penalty = 0.0 if frag < 0.01 else (frag - 0.01) * 2.0
    return min(n_found / 5.0, 1.0) * 0.4 + ba * 0.4 - frag_penalty * 0.2


def _run_engine(
    eng_name: str,
    img: np.ndarray,
    reps: list,
) -> dict | None:
    """Run a single registered engine."""
    try:
        return run_engine(eng_name, img, reps or None, None, 5)
    except Exception as exc:
        print(f"  [{eng_name}] FAILED: {exc}")
    return None


def process_panel(image_path: str, panel_id: str) -> dict:
    """Run sandbox-segment workflow on a single panel."""
    img_orig = np.array(Image.open(image_path).convert("RGB"))
    h, w = img_orig.shape[:2]

    sat = saturation_ratio(img_orig)
    noise_score = estimate_noise_level(img_orig)
    use_blur = noise_score > 0.3

    print(f"[{panel_id}] Image: {w}x{h}, sat={sat:.4f}, noise={noise_score:.4f}, blur={use_blur}")

    # Determine which engines to try
    engines_to_try = []
    if sat >= 0.5:
        engines_to_try = ["v4_kmeans", "ensemble"]
    elif sat < 0.1:
        engines_to_try = ["grayscale_agglomerative", "v4_kmeans"]
    else:
        engines_to_try = ["v4_kmeans", "ensemble"]

    # Generate reps for engines that need them
    reps = []
    cb = extract_colorbar(img_orig)
    if cb is not None:
        reps = color_zones_to_reps(img_orig, color_zones=[], colorbar_rgb=cb, n_layers=5)

    # Optionally blur image for denoising
    img_blurred = adaptive_blur(img_orig) if use_blur else None

    results = []

    for eng_name in engines_to_try:
        # Always try original image first
        raw = _run_engine(eng_name, img_orig, reps)
        if raw is not None:
            labels = raw["labels"]
            metrics = compute_all(labels, img_orig)
            score = _compute_score(metrics)
            # Re-generate overlay from ORIGINAL image so it is never blurred
            seeds_arr = np.array(raw.get("seeds", []), dtype=np.uint8)
            if seeds_arr.size == 0:
                seeds_arr = np.zeros((0, 3), dtype=np.uint8)
            overlay = create_overlay(img_orig, labels, seeds_arr)
            results.append({
                "engine": eng_name,
                "preprocess": "none",
                "labels": labels,
                "overlay": overlay,
                "meta": {**raw["meta"], "preprocess": "none"},
                "metrics": metrics,
                "score": score,
            })
            print(
                f"  [{eng_name}|orig] layers={metrics['n_layers']}, "
                f"ba={metrics['boundary_alignment']:.4f}, "
                f"frag={metrics['total_fragment_area_fraction']:.5f}, score={score:.4f}"
            )

        # If noise is high, also try blurred version
        if use_blur and img_blurred is not None:
            raw_b = _run_engine(eng_name, img_blurred, reps)
            if raw_b is not None:
                labels_b = raw_b["labels"]
                metrics_b = compute_all(labels_b, img_orig)
                score_b = _compute_score(metrics_b)
                seeds_b = np.array(raw_b.get("seeds", []), dtype=np.uint8)
                if seeds_b.size == 0:
                    seeds_b = np.zeros((0, 3), dtype=np.uint8)
                overlay_b = create_overlay(img_orig, labels_b, seeds_b)
                results.append({
                    "engine": eng_name,
                    "preprocess": "blur",
                    "labels": labels_b,
                    "overlay": overlay_b,
                    "meta": {**raw_b["meta"], "preprocess": "blur"},
                    "metrics": metrics_b,
                    "score": score_b,
                })
                print(
                    f"  [{eng_name}|blur] layers={metrics_b['n_layers']}, "
                    f"ba={metrics_b['boundary_alignment']:.4f}, "
                    f"frag={metrics_b['total_fragment_area_fraction']:.5f}, score={score_b:.4f}"
                )

    if not results:
        print(f"[{panel_id}] All engines failed!")
        return {"status": "failed", "panel_id": panel_id}

    # Select best result by score
    best = max(results, key=lambda r: r["score"])
    print(
        f"[{panel_id}] Best: {best['engine']} ({best['preprocess']}) "
        f"score={best['score']:.4f}"
    )

    # Save outputs
    out_dir = Path(f"runs/sandbox/{panel_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "labels.npy", best["labels"])
    Image.fromarray(best["overlay"]).save(out_dir / "overlay.png")
    (out_dir / "meta.json").write_text(
        json.dumps(best["meta"], ensure_ascii=False, indent=2)
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(best["metrics"], ensure_ascii=False, indent=2)
    )

    # Record to strategy memory
    outcome = "success" if best["metrics"]["n_layers"] >= 2 else "retry"
    record_attempt(
        panel_rgb=img_orig,
        engine=best["engine"],
        params={"n_layers": 5, "reps": len(reps), "preprocess": best["preprocess"]},
        scores=best["metrics"],
        outcome=outcome,
        notes=(
            f"Auto-selected from {len(results)} variants. "
            f"Best: {best['engine']} ({best['preprocess']}) score={best['score']:.4f}. "
            f"noise_score={noise_score:.4f}"
        ),
    )

    return {
        "status": "ok",
        "panel_id": panel_id,
        "best_engine": best["engine"],
        "best_preprocess": best["preprocess"],
        "score": best["score"],
        "metrics": best["metrics"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--panel-id", required=True)
    args = parser.parse_args()

    result = process_panel(args.image, args.panel_id)
    print(json.dumps(result, indent=2))
