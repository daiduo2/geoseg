"""Control-point interface editor (headless core).

The editor stores parametric interface curves with stable control-point IDs,
rasterizes them into labels without corrupting background, and is designed to
be driven equally by napari mouse gestures and agent commands.
"""

from geoseg.modules.curve_editor.commands import EditorCommands
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
from geoseg.modules.curve_editor.rasterize import draw_interfaces, rasterize_interfaces
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
    "ControlPoint",
    "EditorCommands",
    "ExtractedCurve",
    "Interface",
    "InterfaceModel",
    "SnappedRegion",
    "boundary_mask",
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
    "rasterize_interfaces",
    "resample_by_arc_length",
    "save_labels",
    "save_session",
    "save_shapes",
    "select_control_points",
    "self_intersects",
    "snap_closed_curve",
    "snap_region",
]
