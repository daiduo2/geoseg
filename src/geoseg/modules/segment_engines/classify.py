"""Legacy import path; implementation lives in ``segment_engines.compat``."""

from geoseg.modules.segment_engines.compat.classify import (
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
