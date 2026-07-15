"""Compatibility facade for regional fusion helpers."""

from geoseg.modules.segment_engines.regional import (
    FusionConfig,
    RegionalAudit,
    _draw_legend,
    draw_legend,
    fuse_with_freeze,
    generate_overlay_with_legend,
    regional_segment,
)

__all__ = [
    "FusionConfig",
    "RegionalAudit",
    "_draw_legend",
    "draw_legend",
    "fuse_with_freeze",
    "generate_overlay_with_legend",
    "regional_segment",
]
