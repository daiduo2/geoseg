"""Legacy import path; implementation lives in ``segment_engines.compat``."""

from geoseg.modules.segment_engines.compat.panel_segment import (
    crop_panel_for_segmentation,
    segment_panel_stage,
)

__all__ = ["crop_panel_for_segmentation", "segment_panel_stage"]
