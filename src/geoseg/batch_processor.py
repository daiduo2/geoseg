"""Batch process a directory of figure images through the full pipeline.

Features:
- SessionState-based persistence (replaces ad-hoc summary.json)
- Resume support (skip already-processed images)
- Per-image error isolation (one failure does not stop the batch)
- Structured session export for downstream HITL review

Usage:
    # Stage 1-3: automated classify + segment (creates/updates session)
    python -m geoseg.batch_processor \
        --images_dir runs/literature_test/gras2019/mineru/extracted/images \
        --output_dir runs/literature_test/gras2019/geoseg_export \
        --n_layers 5

    # Stage 5: export all figures marked REVIEWED (after HITL)
    python -m geoseg.batch_processor \
        --session runs/literature_test/gras2019/geoseg_export/session.json \
        --export_only
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
from geoseg.session_state import (
    FigureStatus,
    SessionState,
    create_session,
    get_summary,
    list_ready_for_export,
    list_ready_for_review,
    load_session,
    save_session,
    update_figure,
)
from geoseg.modules.visual_audit import create_audit_report


def _run_visual_audit(
    labels: np.ndarray,
    panel_rgb: np.ndarray,
    audit_dir: Path,
    panel3_mode: bool = False,
    labels_path: str | None = None,
    gt_mask_path: str | None = None,
) -> dict:
    """Run visual audit on a segmentation result.

    Saves audit artifacts to audit_dir and returns the report dict.
    """
    audit_dir.mkdir(parents=True, exist_ok=True)
    return create_audit_report(
        labels=labels,
        panel_rgb=panel_rgb,
        output_dir=str(audit_dir),
        panel3_mode=panel3_mode,
        labels_path=labels_path,
        gt_mask_path=gt_mask_path,
    )


def _init_session(images_dir: Path, output_dir: Path) -> SessionState:
    """Create or load existing session for the batch."""
    session_path = output_dir / "session.json"
    if session_path.exists():
        try:
            return load_session(session_path)
        except Exception:
            pass
    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    return create_session([str(p) for p in image_files])


def _process_entry(
    entry,
    output_dir: Path,
    n_layers: int,
    quality_preference: str,
    skip_non_velocity_model: bool,
    use_vlm: bool,
    properties_map: dict[str, dict] | None,
) -> dict[str, Any]:
    """Run pipeline for a single figure entry. Returns result dict."""
    img_path = Path(entry.source_path)
    t0 = time.perf_counter()

    try:
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img)

        img_out_dir = output_dir / entry.figure_id
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

        # Build SegmentationRecord from result
        seg_record = None
        if result["status"] in ("ok", "empty") and result.get("panels"):
            panel = result["panels"][0] if result["panels"] else {}
            if panel.get("status") == "ok":
                from geoseg.session_state import SegmentationRecord, SegmentationAttempt
                panel_id = panel.get("panel_id", 0)
                panel_dir = img_out_dir / f"panel{panel_id}"
                seg_record = SegmentationRecord(
                    result_dir=str(panel_dir),
                    engine=panel.get("engines_used", "unknown"),
                    n_layers=panel.get("n_layers", n_layers),
                    quality_score=0.0,
                    overlay_path=str(panel_dir / "overlay.jpg"),
                    labels_path=str(panel_dir / "labels.npz"),
                    attempts=[
                        SegmentationAttempt(
                            engine=panel.get("engines_used", "unknown"),
                            n_layers=panel.get("n_layers", n_layers),
                            quality_score=0.0,
                        )
                    ],
                )

                # Run visual-audit report generation on the cropped panel.
                # v2 visual audit does not apply hard-reject gates; the agent
                # inspects the overlay and decides whether repair is needed.
                try:
                    labels = np.load(seg_record.labels_path)["labels"]
                    bbox = panel.get("bbox", [0, 0, arr.shape[1], arr.shape[0]])
                    x, y, w, h = bbox
                    panel_rgb = arr[y : y + h, x : x + w]
                    panel3_mode = (
                        "panel3" in entry.figure_id.lower()
                        or "panel_3" in entry.figure_id.lower()
                    )
                    audit_dir = panel_dir / "visual_audit"
                    report = _run_visual_audit(
                        labels,
                        panel_rgb,
                        audit_dir,
                        panel3_mode=panel3_mode,
                        labels_path=seg_record.labels_path,
                    )
                    seg_record.audit = report
                    if report.get("rejected"):
                        return {
                            "status": "audit_failed",
                            "reason": "; ".join(report.get("reasons", [])),
                            "classification": result["classification"],
                            "elapsed": time.perf_counter() - t0,
                            "seg_record": seg_record,
                        }
                except Exception as audit_exc:
                    return {
                        "status": "error",
                        "reason": f"visual_audit_failed: {audit_exc}",
                        "classification": result["classification"],
                        "elapsed": time.perf_counter() - t0,
                        "seg_record": seg_record,
                    }

        return {
            "status": result["status"],
            "reason": result.get("reason", ""),
            "classification": result["classification"],
            "elapsed": elapsed,
            "seg_record": seg_record,
        }

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {
            "status": "error",
            "reason": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed": elapsed,
            "seg_record": None,
        }


def process_directory(
    images_dir: str | Path,
    output_dir: str | Path,
    n_layers: int = 5,
    quality_preference: str = "balanced",
    use_vlm: bool = True,
    properties_map: dict[str, dict] | None = None,
    resume: bool = True,
    skip_non_velocity_model: bool = True,
) -> SessionState:
    """Process all images in a directory through the full geoseg pipeline.

    Returns:
        Updated SessionState (persisted to output_dir/session.json).
    """
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state = _init_session(images_dir, output_dir)

    print(f"Session: {state.session_id}")
    print(f"Workset: {len(state.workset)} figures")

    for entry in state.workset:
        if resume and entry.status in (
            FigureStatus.SEGMENTED,
            FigureStatus.NEEDS_EDIT,
            FigureStatus.AUDIT_FAILED,
            FigureStatus.REVIEWED,
            FigureStatus.EXPORTED,
        ):
            print(f"  [skip] {entry.figure_id} ({entry.status.value})")
            continue

        if resume and entry.status == FigureStatus.SKIPPED:
            print(f"  [skip] {entry.figure_id} (skipped)")
            continue

        print(f"\nProcessing {entry.figure_id} ...")
        result = _process_entry(
            entry,
            output_dir,
            n_layers,
            quality_preference,
            skip_non_velocity_model,
            use_vlm,
            properties_map,
        )

        new_status = FigureStatus.SEGMENTED
        if result["status"] == "skipped":
            new_status = FigureStatus.SKIPPED
        elif result["status"] == "error":
            new_status = FigureStatus.ERROR
        elif result["status"] == "audit_failed":
            new_status = FigureStatus.AUDIT_FAILED

        kwargs: dict[str, Any] = {
            "status": new_status,
            "error_message": result.get("reason") if result["status"] == "error" else None,
        }
        if result["status"] == "skipped":
            kwargs["skip_reason"] = result.get("reason", "")
        if result.get("classification"):
            from geoseg.session_state import ClassificationRecord
            kwargs["classification"] = ClassificationRecord(
                figure_type=result["classification"].get("figure_type", "unknown"),
                confidence=result["classification"].get("confidence", 0.0),
                reason=result["classification"].get("reason", ""),
            )
        if result.get("seg_record"):
            kwargs["segmentation"] = result["seg_record"]

        state = update_figure(state, entry.figure_id, **kwargs)
        save_session(state, output_dir / "session.json")

        if result["status"] == "ok":
            print(f"  OK  ({result['elapsed']:.2f}s)")
        elif result["status"] == "empty":
            print(f"  EMPTY: {result.get('reason', '')} ({result['elapsed']:.2f}s)")
        elif result["status"] == "skipped":
            print(f"  SKIP: {result.get('reason', '')} ({result['elapsed']:.2f}s)")
        else:
            print(f"  ERROR: {result.get('reason', '')} ({result['elapsed']:.2f}s)")

    summary = get_summary(state)
    print(f"\n{'='*50}")
    print(
        f"Total: {summary['total_figures']}, "
        f"Segmented: {summary.get('segmented', 0)}, "
        f"Needs Edit: {summary.get('needs_edit', 0)}, "
        f"Audit Failed: {summary.get('audit_failed', 0)}, "
        f"Reviewed: {summary.get('reviewed', 0)}, "
        f"Exported: {summary.get('exported', 0)}, "
        f"Skipped: {summary.get('skipped', 0)}, "
        f"Errors: {summary.get('errors', 0)}"
    )
    print(f"Session: {output_dir / 'session.json'}")

    return state


def export_reviewed(session_path: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Export all figures marked REVIEWED in the session.

    Uses labels_edited.npz when available, otherwise falls back to original labels.
    """
    from geoseg.controller import run_post_process_and_export

    state = load_session(session_path)
    ready = list_ready_for_export(state)
    if not ready:
        print("No figures ready for export.")
        return {"exported": 0, "skipped": 0}

    base_out = Path(output_dir) if output_dir else Path(session_path).parent
    exported = 0
    skipped = 0

    for entry in ready:
        seg = entry.segmentation
        if seg is None:
            skipped += 1
            continue

        labels_path = seg.edited_labels_path or seg.labels_path
        if not labels_path or not Path(labels_path).exists():
            skipped += 1
            continue

        labels = np.load(labels_path)["labels"]
        panel_out = base_out / entry.figure_id

        try:
            result = run_post_process_and_export(
                labels,
                output_dir=panel_out,
                save_intermediates=True,
            )
            if result["status"] == "ok":
                exported += 1
                print(f"  EXPORTED: {entry.figure_id}")
            else:
                skipped += 1
                print(f"  SKIP: {entry.figure_id} ({result.get('reason')})")
        except Exception as exc:
            skipped += 1
            print(f"  ERROR: {entry.figure_id} ({exc})")

    print(f"\nExported: {exported}, Skipped: {skipped}")
    return {"exported": exported, "skipped": skipped}


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch process figures through geoseg pipeline")
    parser.add_argument("--images_dir", help="Directory containing figure images")
    parser.add_argument("--output_dir", help="Directory to save results")
    parser.add_argument("--session", help="Path to existing session JSON (for resume or export)")
    parser.add_argument("--export_only", action="store_true", help="Export all REVIEWED figures in session")
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

    if args.export_only:
        if not args.session:
            parser.error("--session is required with --export_only")
        export_reviewed(args.session, args.output_dir)
        return 0

    if not args.images_dir or not args.output_dir:
        parser.error("--images_dir and --output_dir are required")

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
