"""Batch visual audit helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from geoseg.modules.visual_audit import create_audit_report


def run_visual_audit(
    labels: np.ndarray,
    panel_rgb: np.ndarray,
    audit_dir: Path,
    panel3_mode: bool = False,
    labels_path: str | None = None,
    gt_mask_path: str | None = None,
) -> dict:
    """Run visual audit on a segmentation result."""
    audit_dir.mkdir(parents=True, exist_ok=True)
    return create_audit_report(
        labels=labels,
        panel_rgb=panel_rgb,
        output_dir=str(audit_dir),
        panel3_mode=panel3_mode,
        labels_path=labels_path,
        gt_mask_path=gt_mask_path,
    )


__all__ = ["run_visual_audit"]
