"""Horizon refinement public facade."""

from __future__ import annotations

from geoseg.modules.segment_engines.horizon.boundaries import (
    _adjust_boundaries,
    _extract_boundary_dense,
    _extract_boundary_points,
    _repartition_columns,
)
from geoseg.modules.segment_engines.horizon.coarse import _coarse_segment, _separator_mask
from geoseg.modules.segment_engines.horizon.fitting import (
    _detect_knots,
    _fit_bspline,
    _fit_curve,
    _fit_knot_constrained,
    _fit_loess,
    _fit_multiscale_savgol,
    _fit_quintic,
    _fit_savgol,
    _hampel_filter,
)
from geoseg.modules.segment_engines.horizon.refine import (
    _compute_fragmentation_score,
    refine_boundaries,
    refine_label_blur,
    segment,
)

__all__ = [
    "_adjust_boundaries",
    "_coarse_segment",
    "_compute_fragmentation_score",
    "_detect_knots",
    "_extract_boundary_dense",
    "_extract_boundary_points",
    "_fit_bspline",
    "_fit_curve",
    "_fit_knot_constrained",
    "_fit_loess",
    "_fit_multiscale_savgol",
    "_fit_quintic",
    "_fit_savgol",
    "_hampel_filter",
    "_repartition_columns",
    "_separator_mask",
    "refine_boundaries",
    "refine_label_blur",
    "segment",
]
