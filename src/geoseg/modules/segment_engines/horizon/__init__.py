"""Horizon refinement implementation modules."""

from geoseg.modules.segment_engines.horizon.coarse import _coarse_segment
from geoseg.modules.segment_engines.horizon.occlusion import (
    bridge_boundary_with_endpoint_tangents,
    detect_white_label_boxes,
    fit_boundary_endpoint,
    repair_white_label_occluded_boundaries,
)
from geoseg.modules.segment_engines.horizon.refine import refine_boundaries, refine_label_blur, segment

__all__ = [
    "_coarse_segment",
    "bridge_boundary_with_endpoint_tangents",
    "detect_white_label_boxes",
    "fit_boundary_endpoint",
    "refine_boundaries",
    "refine_label_blur",
    "repair_white_label_occluded_boundaries",
    "segment",
]
