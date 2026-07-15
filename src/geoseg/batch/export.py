"""Batch export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from geoseg.session_state import list_ready_for_export, load_session


def export_reviewed(
    session_path: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Export all figures marked REVIEWED in the session."""
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


__all__ = ["export_reviewed"]
