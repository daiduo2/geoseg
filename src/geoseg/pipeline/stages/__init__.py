"""Stable imports for figure-level segmentation stage helpers."""

from geoseg.pipeline.stages.classify import (
    MIN_AUTO_SEGMENT_HEIGHT,
    MIN_AUTO_SEGMENT_WIDTH,
    classify_figure_stage,
    maybe_skip_tiny_image,
)
from geoseg.pipeline.stages.detect import detect_panels_stage, panel_complexity_score
from geoseg.pipeline.stages.panel import (
    crop_panel_for_segmentation,
    segment_panel_stage,
)
from geoseg.pipeline.stages.review import (
    resolve_target_panel_stage,
    review_figure_stage,
)
from geoseg.pipeline.stages.summary import summarize_pipeline_result

__all__ = [
    "MIN_AUTO_SEGMENT_HEIGHT",
    "MIN_AUTO_SEGMENT_WIDTH",
    "classify_figure_stage",
    "crop_panel_for_segmentation",
    "detect_panels_stage",
    "maybe_skip_tiny_image",
    "panel_complexity_score",
    "resolve_target_panel_stage",
    "review_figure_stage",
    "segment_panel_stage",
    "summarize_pipeline_result",
]
