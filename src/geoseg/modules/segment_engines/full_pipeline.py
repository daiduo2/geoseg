"""Legacy full segmentation pipeline facade.

The stage implementations live in ``pipeline_stages.py`` so classification,
panel detection/review, panel segmentation, and summary construction can evolve
independently. ``process_figure`` remains available for backward-compatible
script usage.
"""

from __future__ import annotations

import numpy as np

from geoseg.modules.segment_engines.pipeline_stages import (
    classify_figure_stage,
    detect_panels_stage,
    maybe_skip_tiny_image,
    panel_complexity_score as _panel_complexity_score,
    resolve_target_panel_stage,
    review_figure_stage,
    segment_panel_stage,
    summarize_pipeline_result,
)


def process_figure(
    img_rgb: np.ndarray,
    caption: str = "",
    text_blocks: list[dict] | None = None,
    n_layers: int = 5,
    quality_preference: str = "balanced",
    skip_non_velocity_model: bool = True,
    use_vlm: bool = True,
    target_panel_id: int = -1,
) -> dict:
    """Process a raw extracted figure image through the legacy full pipeline."""
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
    panel_bboxes, color_zones, has_colorbar_hint, resolved_target_panel_id = (
        review_figure_stage(
            img_rgb,
            caption=caption,
            text_blocks=text_blocks,
            panel_bboxes=panel_bboxes,
            target_panel_id=target_panel_id,
            use_vlm=use_vlm,
            review_warnings=review_warnings,
        )
    )

    resolved_target_panel_id = resolve_target_panel_stage(
        img_rgb,
        panel_bboxes,
        resolved_target_panel_id,
        review_warnings,
    )

    panel_results = []
    total_layers = 0
    engines_used: set[str] = set()
    for panel in panel_bboxes:
        result, n_layers_found, engine = segment_panel_stage(
            img_rgb,
            panel,
            panel_bboxes=panel_bboxes,
            target_panel_id=resolved_target_panel_id,
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
        target_panel_id=resolved_target_panel_id,
    )


__all__ = ["_panel_complexity_score", "process_figure"]
