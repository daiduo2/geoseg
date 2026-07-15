"""Regional repair helpers for segmentation engines."""

from geoseg.modules.segment_engines.regional.fusion import regional_segment
from geoseg.modules.segment_engines.regional.models import FusionConfig, RegionalAudit
from geoseg.modules.segment_engines.regional.overlay import (
    _draw_legend,
    draw_legend,
    generate_overlay_with_legend,
)
from geoseg.modules.segment_engines.regional.split_merge import fuse_with_freeze

__all__ = [
    "FusionConfig",
    "RegionalAudit",
    "_draw_legend",
    "draw_legend",
    "fuse_with_freeze",
    "generate_overlay_with_legend",
    "regional_segment",
]
