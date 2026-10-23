"""Control-point interface editor (headless core).

The editor stores parametric interface curves with stable control-point IDs,
rasterizes them into labels without corrupting background, and is designed to
be driven equally by napari mouse gestures and agent commands.
"""

from geoseg.modules.curve_editor.background import background_roles
from geoseg.modules.curve_editor.commands import EditorCommands
from geoseg.modules.curve_editor.decisions import (
    BoundaryDecision,
    BoundaryDecisionResult,
    curve_background_role,
    decide_boundaries,
    refine_decisions,
)
from geoseg.modules.curve_editor.exact_boundaries import (
    ExactBoundary,
    ExactBoundaryDecision,
    boundary_union,
    closed_boundary_union,
    decide_exact_boundaries,
    draw_exact_decisions,
    extract_exact_boundaries,
)
from geoseg.modules.curve_editor.boundary_ops import (
    clip_polygon_to_rect,
    extend_open_curve,
    label_boundary_mask,
    self_intersects,
)
from geoseg.modules.curve_editor.extract import ExtractedCurve, boundary_mask, curves_from_labels
from geoseg.modules.curve_editor.io import (
    load_labels,
    load_session,
    load_shapes,
    model_from_labels,
    save_labels,
    save_session,
    save_shapes,
)
from geoseg.modules.curve_editor.model import ControlPoint, Interface, InterfaceModel
from geoseg.modules.curve_editor.partition import ComponentMerge, prune_small_components
from geoseg.modules.curve_editor.rasterize import draw_interfaces, rasterize_interfaces
from geoseg.modules.curve_editor.refine import (
    BoundaryRefinementConfig,
    RefinedBoundary,
    adjacent_labels,
    refine_curve,
    refine_curves,
)
from geoseg.modules.curve_editor.region import (
    SnappedRegion,
    close_and_resample,
    edge_strength,
    interior_point,
    select_control_points,
    snap_closed_curve,
    snap_region,
)
from geoseg.modules.curve_editor.spline import (
    BASIS_NAMES,
    curve_from_interface,
    curvature,
    curvature_peaks,
    fit_freehand,
    resample_by_arc_length,
)

__all__ = [
    "BASIS_NAMES",
    "BoundaryDecision",
    "BoundaryDecisionResult",
    "BoundaryRefinementConfig",
    "ControlPoint",
    "ComponentMerge",
    "EditorCommands",
    "ExactBoundary",
    "ExactBoundaryDecision",
    "ExtractedCurve",
    "Interface",
    "InterfaceModel",
    "RefinedBoundary",
    "SnappedRegion",
    "adjacent_labels",
    "background_roles",
    "boundary_mask",
    "boundary_union",
    "closed_boundary_union",
    "curve_background_role",
    "decide_boundaries",
    "decide_exact_boundaries",
    "draw_exact_decisions",
    "extract_exact_boundaries",
    "refine_decisions",
    "clip_polygon_to_rect",
    "close_and_resample",
    "curve_from_interface",
    "curvature",
    "curvature_peaks",
    "curves_from_labels",
    "draw_interfaces",
    "edge_strength",
    "extend_open_curve",
    "fit_freehand",
    "interior_point",
    "label_boundary_mask",
    "load_labels",
    "load_session",
    "load_shapes",
    "model_from_labels",
    "prune_small_components",
    "rasterize_interfaces",
    "refine_curve",
    "refine_curves",
    "resample_by_arc_length",
    "save_labels",
    "save_session",
    "save_shapes",
    "select_control_points",
    "self_intersects",
    "snap_closed_curve",
    "snap_region",
]
