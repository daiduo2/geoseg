"""Backward-compatible imports for pipeline classification stages."""

from geoseg.pipeline.stages import (
    MIN_AUTO_SEGMENT_HEIGHT,
    MIN_AUTO_SEGMENT_WIDTH,
    classify_figure_stage,
    maybe_skip_tiny_image,
)

__all__ = [
    "MIN_AUTO_SEGMENT_HEIGHT",
    "MIN_AUTO_SEGMENT_WIDTH",
    "classify_figure_stage",
    "maybe_skip_tiny_image",
]
