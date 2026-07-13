"""Re-audit historical Panel 3 segmentation results with visual-audit gates.

Usage:
    python scripts/reaudit_panel3.py \
        --runs runs/3d_schematic_correct_e2e \
              runs/3d_schematic_e2e \
              runs/3d_schematic_edge_guided \
              runs/engine_compare_panel3 \
              runs/merge_panel3_audit \
              runs/tubular_panel3 \
        --panel runs/3d_schematic_correct_e2e/panel_3_front/00_enhanced.jpg \
        --panel3

If --panel is omitted, the script tries to find a matching panel image next to
each labels.npz (00_enhanced.jpg, 00_original.jpg, panel.jpg, etc.).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from geoseg.modules.visual_audit import create_audit_report


DEFAULT_RUN_DIRS = [
    "runs/3d_schematic_correct_e2e",
    "runs/3d_schematic_e2e",
    "runs/3d_schematic_edge_guided",
    "runs/engine_compare_panel3",
    "runs/merge_panel3_audit",
    "runs/tubular_panel3",
]

PANEL_IMAGE_CANDIDATES = [
    "00_enhanced.jpg",
    "00_original.jpg",
    "panel.jpg",
    "panel.png",
    "00_cleaned.jpg",
]


def _load_labels(npz_path: Path) -> np.ndarray | None:
    """Load labels array from an npz file, accepting either 'labels' or first array."""
    try:
        data = np.load(npz_path)
        if "labels" in data:
            return data["labels"]
        keys = [k for k in data.files if data[k].ndim == 2]
        if keys:
            return data[keys[0]]
    except Exception as exc:
        print(f"  [warn] cannot load {npz_path}: {exc}")
    return None


def _find_panel_image(label_path: Path, labels: np.ndarray, fallback: Path | None) -> Path | None:
    """Find a panel image whose shape matches the labels array."""
    h, w = labels.shape[:2]
    expected_size = (w, h)

    # 1. Same-name image next to the npz
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = label_path.with_suffix(ext)
        if candidate.exists() and Image.open(candidate).size == expected_size:
            return candidate

    # 2. Named candidates in the same directory
    for name in PANEL_IMAGE_CANDIDATES:
        candidate = label_path.parent / name
        if candidate.exists() and Image.open(candidate).size == expected_size:
            return candidate

    # 3. Any matching-shape image in the same directory or direct subdirectories
    for candidate in sorted(label_path.parent.rglob("*")):
        if candidate.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if candidate.parent.name == "visual_audit":
            continue
        try:
            if Image.open(candidate).size == expected_size:
                return candidate
        except Exception:
            continue

    # 4. Fallback panel image if provided
    if fallback is not None and fallback.exists():
        try:
            if Image.open(fallback).size == expected_size:
                return fallback
        except Exception:
            pass

    return None


def _audit_one(label_path: Path, panel_path: Path, panel3_mode: bool, gt_mask_path: Path | None = None) -> dict:
    """Run visual audit on a single labels file."""
    labels = _load_labels(label_path)
    if labels is None:
        return {"status": "error", "reason": "failed_to_load_labels"}

    panel_rgb = np.array(Image.open(panel_path).convert("RGB"))
    if panel_rgb.shape[:2] != labels.shape:
        return {
            "status": "error",
            "reason": f"shape_mismatch: labels={labels.shape}, panel={panel_rgb.shape[:2]}",
        }

    audit_dir = label_path.parent / "visual_audit" / label_path.stem
    audit_dir.mkdir(parents=True, exist_ok=True)

    report = create_audit_report(
        labels=labels,
        panel_rgb=panel_rgb,
        output_dir=str(audit_dir),
        panel3_mode=panel3_mode,
        labels_path=str(label_path),
        gt_mask_path=str(gt_mask_path) if gt_mask_path else None,
    )
    return {
        "status": "ok",
        "rejected": report["rejected"],
        "reasons": report["reasons"],
        "metrics": report["hard_reject_metrics"],
        "semantic": report.get("semantic_metrics", {}),
        "summary_image": report.get("summary_image_path"),
        "audit_dir": str(audit_dir),
    }


def _discover_label_files(run_dir: Path) -> list[Path]:
    """Find all .npz label files under a run directory, excluding visual_audit dirs."""
    files: list[Path] = []
    for npz in sorted(run_dir.rglob("*.npz")):
        if "visual_audit" in npz.parts:
            continue
        files.append(npz)
    return files


def _is_panel3_path(path: Path) -> bool:
    """Heuristic: path contains panel3 or panel_3."""
    lower = str(path).lower()
    return "panel3" in lower or "panel_3" in lower


def reaudit_runs(
    run_dirs: list[Path],
    fallback_panel: Path | None,
    panel3_mode: bool | None,
    gt_mask_path: Path | None = None,
) -> dict:
    """Run visual audit across all discovered label files."""
    results: list[dict] = []
    total = 0
    rejected = 0
    passed = 0
    errors = 0

    for run_dir in run_dirs:
        if not run_dir.exists():
            print(f"[skip] run dir not found: {run_dir}")
            continue
        print(f"\n[scan] {run_dir}")
        for label_path in _discover_label_files(run_dir):
            total += 1
            print(f"  [{total}] {label_path.relative_to(run_dir)}")
            labels = _load_labels(label_path)
            if labels is None:
                errors += 1
                results.append({
                    "run": str(run_dir),
                    "label_path": str(label_path),
                    "status": "error",
                    "reason": "load_failed",
                })
                continue

            panel_path = _find_panel_image(label_path, labels, fallback_panel)
            if panel_path is None:
                errors += 1
                results.append({
                    "run": str(run_dir),
                    "label_path": str(label_path),
                    "status": "error",
                    "reason": "panel_image_not_found",
                })
                continue

            use_panel3 = panel3_mode if panel3_mode is not None else _is_panel3_path(label_path)
            result = _audit_one(label_path, panel_path, use_panel3, gt_mask_path)
            result.update({
                "run": str(run_dir),
                "label_path": str(label_path),
                "panel_path": str(panel_path),
            })
            results.append(result)

            if result["status"] == "ok":
                if result["rejected"]:
                    rejected += 1
                    print(f"      REJECTED: {'; '.join(result['reasons'])}")
                else:
                    passed += 1
                    print(f"      PASSED")
            else:
                errors += 1
                print(f"      ERROR: {result.get('reason', '')}")

    return {
        "total": total,
        "passed": passed,
        "rejected": rejected,
        "errors": errors,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-audit historical Panel 3 results")
    parser.add_argument(
        "--runs",
        nargs="+",
        default=DEFAULT_RUN_DIRS,
        help="Run directories to scan for .npz label files",
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=None,
        help="Fallback panel image path (used when no matching image is found next to labels)",
    )
    parser.add_argument(
        "--panel3",
        action="store_true",
        default=None,
        dest="panel3",
        help="Force Panel-3-specific hard-reject rules on all files",
    )
    parser.add_argument(
        "--no-panel3",
        action="store_false",
        default=None,
        dest="panel3",
        help="Disable Panel-3-specific hard-reject rules on all files",
    )
    parser.add_argument(
        "--gt-mask",
        type=Path,
        default=None,
        help="Manual GT plume mask (panel3 mode). Enables plume IoU hard gate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/reaudit_panel3_summary.json"),
        help="Path to write JSON summary",
    )
    args = parser.parse_args(argv)

    run_dirs = [Path(p) for p in args.runs]
    summary = reaudit_runs(run_dirs, args.panel, args.panel3, args.gt_mask)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"Total audited: {summary['total']}")
    print(f"Passed:        {summary['passed']}")
    print(f"Rejected:      {summary['rejected']}")
    print(f"Errors:        {summary['errors']}")
    print(f"Summary saved: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
