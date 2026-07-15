"""Legacy import path; implementation lives in ``segment_engines.compat``."""

from geoseg.modules.segment_engines.compat.full_pipeline import (
    _panel_complexity_score,
    process_figure,
)

__all__ = ["_panel_complexity_score", "process_figure"]
