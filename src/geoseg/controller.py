"""Backward-compatible controller entry points.

New stage-level orchestration should live under ``geoseg.pipeline``. This
module remains the public compatibility facade for existing users.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from geoseg.pipeline.export import export_segmented_panels, run_post_process_and_export
from geoseg.pipeline.segment import run_segmentation_stage


def run_pipeline(
    img_rgb: np.ndarray,
    caption: str = "",
    text_blocks: list[dict] | None = None,
    n_layers: int = 5,
    quality_preference: str = "balanced",
    skip_non_velocity_model: bool = True,
    use_vlm: bool = True,
    target_panel_id: int = -1,
    properties_map: dict[str, dict] | None = None,
    output_dir: str | Path | None = None,
    save_intermediates: bool = True,
) -> dict[str, Any]:
    """Run the full geoseg pipeline on a single figure image."""
    seg_result = run_segmentation_stage(
        img_rgb,
        caption=caption,
        text_blocks=text_blocks,
        n_layers=n_layers,
        quality_preference=quality_preference,
        skip_non_velocity_model=skip_non_velocity_model,
        use_vlm=use_vlm,
        target_panel_id=target_panel_id,
    )

    if seg_result["summary"]["status"] == "skipped":
        return {
            "status": "skipped",
            "reason": seg_result["summary"]["reason"],
            "classification": seg_result["classification"],
            "panels": [],
            "summary": seg_result["summary"],
        }

    panel_outputs, summary = export_segmented_panels(
        seg_result,
        properties_map=properties_map,
        output_dir=output_dir,
        save_intermediates=save_intermediates,
    )

    n_processed = sum(1 for panel in panel_outputs if panel["status"] == "ok")
    if n_processed == 0 and panel_outputs:
        return {
            "status": "empty",
            "reason": "all_panels_skipped_or_no_segmentation",
            "classification": seg_result["classification"],
            "panels": panel_outputs,
            "summary": summary,
        }

    return {
        "status": "ok",
        "classification": seg_result["classification"],
        "panels": panel_outputs,
        "summary": summary,
    }


__all__ = ["run_pipeline", "run_post_process_and_export"]
