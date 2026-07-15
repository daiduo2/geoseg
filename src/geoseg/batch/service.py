"""Batch processing service orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from geoseg.batch.audit import run_visual_audit
from geoseg.batch.entry import process_entry
from geoseg.batch.export import export_reviewed
from geoseg.batch.session import init_session
from geoseg.session_state import (
    FigureStatus,
    SessionState,
    get_summary,
    save_session,
    update_figure,
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
) -> SessionState:
    """Process all images in a directory through the full geoseg pipeline."""
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state = init_session(images_dir, output_dir)

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
        result = process_entry(
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
            "error_message": result.get("reason")
            if result["status"] == "error"
            else None,
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


__all__ = [
    "export_reviewed",
    "init_session",
    "process_directory",
    "process_entry",
    "run_visual_audit",
]
