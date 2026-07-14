"""Segmentation stage orchestration.

This module owns the controller-facing segmentation stage and composes the
pipeline from focused stage helpers.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from geoseg.pipeline.stages import (
    classify_figure_stage,
    detect_panels_stage,
    maybe_skip_tiny_image,
    resolve_target_panel_stage,
    review_figure_stage,
    segment_panel_stage,
    summarize_pipeline_result,
)


def run_segmentation_stage(
    img_rgb: np.ndarray,
    *,
    caption: str = "",
    text_blocks: list[dict] | None = None,
    n_layers: int = 5,
    quality_preference: str = "balanced",
    skip_non_velocity_model: bool = True,
    use_vlm: bool = True,
    target_panel_id: int = -1,
) -> dict[str, Any]:
    """Run classification, panel detection, and segmentation for one figure."""
    review_warnings: list[str] = []

    tiny_result = maybe_skip_tiny_image(img_rgb, review_warnings)
    if tiny_result is not None:
        return tiny_result

    cls, skipped_result = classify_figure_stage(
        img_rgb,
        skip_non_velocity_model=skip_non_velocity_model,
        use_vlm=use_vlm,
        review_warnings=review_warnings,
    )
    if skipped_result is not None:
        return skipped_result

    panel_bboxes = detect_panels_stage(img_rgb)
    panel_bboxes, color_zones, has_colorbar_hint, target_panel_id = review_figure_stage(
        img_rgb,
        caption=caption,
        text_blocks=text_blocks,
        panel_bboxes=panel_bboxes,
        target_panel_id=target_panel_id,
        use_vlm=use_vlm,
        review_warnings=review_warnings,
    )

    target_panel_id = resolve_target_panel_stage(
        img_rgb,
        panel_bboxes,
        target_panel_id,
        review_warnings,
    )

    panel_results: list[dict[str, Any]] = []
    total_layers = 0
    engines_used: set[str] = set()
    for panel in panel_bboxes:
        result, n_layers_found, engine = segment_panel_stage(
            img_rgb,
            panel,
            panel_bboxes=panel_bboxes,
            target_panel_id=target_panel_id,
            color_zones=color_zones,
            n_layers=n_layers,
            quality_preference=quality_preference,
            review_warnings=review_warnings,
        )
        if result is None:
            continue
        panel_results.append(result)
        total_layers += n_layers_found
        if engine:
            engines_used.add(engine)

    return summarize_pipeline_result(
        img_rgb,
        classification=cls,
        panel_results=panel_results,
        total_layers=total_layers,
        engines_used=engines_used,
        review_warnings=review_warnings,
        has_colorbar_hint=has_colorbar_hint,
        target_panel_id=target_panel_id,
    )


__all__ = ["run_segmentation_stage"]
