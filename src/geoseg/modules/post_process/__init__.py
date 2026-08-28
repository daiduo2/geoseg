"""Post-processing: labels → polygons / components / properties."""

from geoseg.modules.post_process.polygon import extract_components, labels_to_polygons, save_geojson
from geoseg.modules.post_process.properties import (
    DEFAULT_PROPERTIES,
    assign_properties,
    build_properties_template,
    load_properties_json,
    save_properties_json,
)
from geoseg.modules.post_process.merge import merge_warm_labels, merge_labels_by_ids
from geoseg.modules.post_process.split import (
    split_label_by_color_components,
    split_labels_by_red_boundaries,
)

__all__ = [
    "extract_components",
    "labels_to_polygons",
    "save_geojson",
    "DEFAULT_PROPERTIES",
    "assign_properties",
    "build_properties_template",
    "load_properties_json",
    "save_properties_json",
    "merge_warm_labels",
    "merge_labels_by_ids",
    "split_label_by_color_components",
    "split_labels_by_red_boundaries",
]
