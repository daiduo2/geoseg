"""Backward-compatible facade for the v4 K-Means segmentation engine."""

from __future__ import annotations

from geoseg.modules.segment_engines.v4 import (
    JET_VIVID_RATIO,
    segment,
    segment_colorbar_guided,
    segment_jet_vivid,
    segment_pastel_faded,
)
from geoseg.modules.segment_engines.v4.palette import _name_palette, _sample_colorbar_seeds
from geoseg.modules.segment_engines.v4.postprocess import (
    _enhance_close_boundaries,
    _fill_holes,
    _nearest_median,
    _remove_small_components,
)

__all__ = [
    "JET_VIVID_RATIO",
    "_enhance_close_boundaries",
    "_fill_holes",
    "_name_palette",
    "_nearest_median",
    "_remove_small_components",
    "_sample_colorbar_seeds",
    "segment",
    "segment_colorbar_guided",
    "segment_jet_vivid",
    "segment_pastel_faded",
]
