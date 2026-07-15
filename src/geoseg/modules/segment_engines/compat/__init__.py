"""Compatibility shims for legacy segment engine import paths."""

from geoseg.modules.segment_engines.compat.full_pipeline import process_figure
from geoseg.modules.segment_engines.compat.pipeline_stages import (
    MIN_AUTO_SEGMENT_HEIGHT,
    MIN_AUTO_SEGMENT_WIDTH,
    classify_figure_stage,
    crop_panel_for_segmentation,
    detect_panels_stage,
    maybe_skip_tiny_image,
    panel_complexity_score,
    resolve_target_panel_stage,
    review_figure_stage,
    segment_panel_stage,
    summarize_pipeline_result,
)

__all__ = [
    "MIN_AUTO_SEGMENT_HEIGHT",
    "MIN_AUTO_SEGMENT_WIDTH",
    "classify_figure_stage",
    "crop_panel_for_segmentation",
    "detect_panels_stage",
    "maybe_skip_tiny_image",
    "panel_complexity_score",
    "process_figure",
    "resolve_target_panel_stage",
    "review_figure_stage",
    "segment_panel_stage",
    "summarize_pipeline_result",
]
