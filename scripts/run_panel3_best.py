"""Re-run Panel 3 segmentation across multiple engines and pick the best by audit.

Usage:
    PYTHONPATH=/Users/daiduo2/geoseg python scripts/run_panel3_best.py

Outputs:
    runs/panel3_best_search/{config_name}/labels.npz
    runs/panel3_best_search/{config_name}/overlay.jpg
    runs/panel3_best_search/{config_name}/visual_audit/...
    runs/panel3_best_search/ranked_results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from geoseg.modules.segment_engines import (
    v4_kmeans,
    kmeans_full,
    edge_guided,
    slic_kmeans,
    grayscale,
    ensemble,
)
from geoseg.modules.segment_engines._shared import row_median_filter
from geoseg.modules.segment_engines.horizon_refinement import refine_boundaries
from geoseg.modules.visual_audit import create_audit_report


PANEL_PATH = Path("runs/3d_schematic_correct_e2e/panel_3_front/00_enhanced.jpg")
GT_MASK_PATH = Path("runs/engine_compare_panel3/visuals/manual_gt_mask.jpg")
OUTPUT_ROOT = Path("runs/panel3_best_search")
N_LAYERS = 5


def _load_gt_mask() -> np.ndarray | None:
    if not GT_MASK_PATH.exists():
        return None
    img = np.array(Image.open(GT_MASK_PATH).convert("L"))
    return img > 0


def _audit(labels: np.ndarray, panel_rgb: np.ndarray, audit_dir: Path, labels_path: str | None = None, gt_mask_path: str | None = None) -> dict:
    audit_dir.mkdir(parents=True, exist_ok=True)
    return create_audit_report(
        labels=labels,
        panel_rgb=panel_rgb,
        output_dir=str(audit_dir),
        panel3_mode=True,
        labels_path=labels_path,
        gt_mask_path=gt_mask_path,
    )


def _score(report: dict) -> tuple:
    """Lower is better. Primary: not rejected, then semantic fidelity product."""
    metrics = report["hard_reject_metrics"]
    semantic = report.get("semantic_metrics", {})
    plume = semantic.get("plume_fidelity", {})
    plume_iou = plume.get("iou", 0.0) if plume else 0.0
    avg_ba = semantic.get("avg_boundary_alignment", 0.0)
    semantic_score = round(plume_iou * avg_ba, 4)

    return (
        int(report["rejected"]),
        -semantic_score,       # higher semantic fidelity -> lower score
        -round(plume_iou, 4),  # tie-breaker: higher plume IoU
        -round(avg_ba, 4),     # tie-breaker: higher boundary alignment
        metrics.get("tiny_island_count", 9999),
        int(metrics.get("central_plume_disconnected", True)),
        round(metrics.get("fragment_ratio", 1.0), 4),
        round(metrics.get("text_mask_label_overlap", 1.0), 4),
        metrics.get("n_labels", 9999),
    )


def _save_result(name: str, labels: np.ndarray, panel_rgb: np.ndarray) -> Path:
    out_dir = OUTPUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "labels.npz", labels=labels)

    # Simple overlay
    from geoseg.modules.segment_engines._shared import _create_overlay
    overlay = _create_overlay(panel_rgb, labels, panel_rgb)
    Image.fromarray(overlay).save(out_dir / "overlay.jpg", quality=90)
    return out_dir


def _run_config(config: dict, panel_rgb: np.ndarray, gt_mask: np.ndarray | None) -> dict:
    name = config["name"]
    engine = config["engine"]
    preprocess = config.get("preprocess")
    postprocess = config.get("postprocess")
    kwargs = config.get("kwargs", {})

    img = panel_rgb
    if preprocess == "row_median_5":
        img = row_median_filter(panel_rgb, size=5)
    elif preprocess == "row_median_7":
        img = row_median_filter(panel_rgb, size=7)
    elif preprocess == "gaussian_1":
        img = cv2.GaussianBlur(panel_rgb, (0, 0), sigmaX=1.0)
    elif preprocess == "gaussian_2":
        img = cv2.GaussianBlur(panel_rgb, (0, 0), sigmaX=2.0)

    result = engine.segment(img, n_layers=N_LAYERS, **kwargs)
    labels = result["labels"]

    if postprocess == "horizon_refinement":
        labels, _ = refine_boundaries(panel_rgb, coarse_labels=labels, method="savgol")

    out_dir = _save_result(name, labels, panel_rgb)
    labels_path = str(out_dir / "labels.npz")
    report = _audit(
        labels,
        panel_rgb,
        out_dir / "visual_audit",
        labels_path=labels_path,
        gt_mask_path=str(GT_MASK_PATH) if GT_MASK_PATH.exists() else None,
    )

    semantic = report.get("semantic_metrics", {})
    plume = semantic.get("plume_fidelity", {})

    return {
        "name": name,
        "engine": engine.__name__.split(".")[-1],
        "preprocess": preprocess,
        "postprocess": postprocess,
        "kwargs": kwargs,
        "rejected": report["rejected"],
        "reasons": report["reasons"],
        "metrics": report["hard_reject_metrics"],
        "semantic": semantic,
        "plume_iou": plume.get("iou", 0.0) if plume else 0.0,
        "avg_boundary_alignment": semantic.get("avg_boundary_alignment", 0.0),
        "out_dir": str(out_dir),
        "score": _score(report),
    }


CONFIGS = [
    {"name": "v4_kmeans", "engine": v4_kmeans},
    {"name": "v4_kmeans_row5", "engine": v4_kmeans, "preprocess": "row_median_5"},
    {"name": "v4_kmeans_row7", "engine": v4_kmeans, "preprocess": "row_median_7"},
    {"name": "v4_kmeans_blur1", "engine": v4_kmeans, "preprocess": "gaussian_1"},
    {"name": "v4_kmeans_blur2", "engine": v4_kmeans, "preprocess": "gaussian_2"},
    {"name": "kmeans_full", "engine": kmeans_full},
    {"name": "kmeans_full_row5", "engine": kmeans_full, "preprocess": "row_median_5"},
    {"name": "kmeans_full_blur1", "engine": kmeans_full, "preprocess": "gaussian_1"},
    {"name": "edge_guided", "engine": edge_guided},
    {"name": "edge_guided_lowedge", "engine": edge_guided, "kwargs": {"edge_weight": 0.2}},
    {"name": "edge_guided_highsigma", "engine": edge_guided, "kwargs": {"sigma": 4.0}},
    {"name": "edge_guided_row5", "engine": edge_guided, "preprocess": "row_median_5"},
    {"name": "slic_kmeans", "engine": slic_kmeans},
    {"name": "slic_kmeans_compact5", "engine": slic_kmeans, "kwargs": {"compactness": 5.0}},
    {"name": "slic_kmeans_seg1000", "engine": slic_kmeans, "kwargs": {"n_segments": 1000}},
    {"name": "grayscale", "engine": grayscale},
    {"name": "ensemble", "engine": ensemble},
    {"name": "ensemble_row5", "engine": ensemble, "preprocess": "row_median_5"},
]


def main() -> int:
    if not PANEL_PATH.exists():
        print(f"Panel image not found: {PANEL_PATH}")
        return 1

    panel_rgb = np.array(Image.open(PANEL_PATH).convert("RGB"))
    print(f"Panel shape: {panel_rgb.shape}")

    gt_mask = _load_gt_mask()
    if gt_mask is None:
        print(f"[warn] No manual GT mask at {GT_MASK_PATH}; semantic scoring will be degraded.")
    else:
        print(f"Manual GT mask shape: {gt_mask.shape}")

    results: list[dict] = []
    for config in CONFIGS:
        print(f"\nRunning {config['name']} ...")
        try:
            result = _run_config(config, panel_rgb, gt_mask)
            results.append(result)
            status = "REJECTED" if result["rejected"] else "PASSED"
            print(f"  {status}: {result['reasons']}")
            print(
                f"  plume_iou={result.get('plume_iou', 0.0):.3f}, "
                f"boundary_alignment={result.get('avg_boundary_alignment', 0.0):.3f}"
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            results.append({
                "name": config["name"],
                "status": "error",
                "reason": str(exc),
            })

    # Rank
    ranked = sorted(
        [r for r in results if r.get("score") is not None],
        key=lambda r: r["score"],
    )

    summary = {
        "panel_path": str(PANEL_PATH),
        "gt_mask_path": str(GT_MASK_PATH) if GT_MASK_PATH.exists() else None,
        "n_configs": len(CONFIGS),
        "passed": sum(1 for r in results if r.get("rejected") is False),
        "rejected": sum(1 for r in results if r.get("rejected") is True),
        "errors": sum(1 for r in results if r.get("status") == "error"),
        "ranked": ranked,
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_ROOT / "ranked_results.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("TOP RESULTS:")
    for i, r in enumerate(ranked[:5], 1):
        status = "PASS" if not r["rejected"] else "FAIL"
        print(
            f"{i}. [{status}] {r['name']}: "
            f"plume_iou={r.get('plume_iou', 0.0):.3f}, "
            f"boundary_alignment={r.get('avg_boundary_alignment', 0.0):.3f}, "
            f"fragment_ratio={r['metrics'].get('fragment_ratio')}, "
            f"text_overlap={r['metrics'].get('text_mask_label_overlap')}"
        )
        print(f"   reasons: {r['reasons']}")
        print(f"   out: {r['out_dir']}")

    print(f"\nFull summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
