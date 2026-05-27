"""Post-processing: labels → polygons / components / properties."""

from geoseg.modules.post_process.polygon import extract_components, labels_to_polygons, save_geojson
from geoseg.modules.post_process.properties import (
    DEFAULT_PROPERTIES,
    assign_properties,
    build_properties_template,
    load_properties_json,
    save_properties_json,
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
]
