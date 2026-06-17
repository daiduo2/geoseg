"""napari-based segmentation editor for geoseg pipeline."""

from geoseg.modules.editor.editor_core import (
    RegionProperties,
    compute_label_diff,
    extend_trim_line_to_mask,
    fill_boundary_gaps,
    labels_to_shapes,
    merge_labels,
    polygon_rasterize,
    shapes_to_labels,
    snap_line_endpoints,
    snap_path_endpoints,
    split_label_by_line,
)
from geoseg.modules.editor.editor_model import EditorModel

__all__ = [
    "EditorModel",
    "RegionProperties",
    "compute_label_diff",
    "extend_trim_line_to_mask",
    "fill_boundary_gaps",
    "labels_to_shapes",
    "merge_labels",
    "polygon_rasterize",
    "shapes_to_labels",
    "snap_line_endpoints",
    "snap_path_endpoints",
    "split_label_by_line",
]
