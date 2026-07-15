"""Legacy import path; implementation lives in ``segment_engines.compat``."""

from geoseg.modules.segment_engines.compat.detect import (
    detect_panels_stage,
    panel_complexity_score,
)

__all__ = ["detect_panels_stage", "panel_complexity_score"]
