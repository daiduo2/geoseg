"""Public API for geoseg.

This is the one module scripts, skills, notebooks and examples should import
from. It re-exports the stable entry points and contracts; deep imports into
``geoseg.modules.*`` are implementation details and may move.

Typical use::

    from geoseg import api

    result = api.run_pipeline(image_rgb, n_layers=5, output_dir="runs/out")
    labels = result["panels"][0]["properties"]  # see core.models for contracts

Sections below mirror the pipeline: inspect → segment → audit helpers →
post-process/export → edit → boundary geometry.
"""

from __future__ import annotations

from geoseg.core.image_ops import (
    adaptive_blur,
    create_overlay,
    detect_background_label,
    distinct_colors,
    estimate_noise_level,
    merge_small_regions,
    row_median_filter,
    saturation_ratio,
)
from geoseg.core.models import (
    BBoxXYWH,
    FigureClassification,
    PageOverview,
    PanelInput,
    RegionalAudit,
    RGBColor,
    SegmentationMeta,
    SegmentationResult,
)
from geoseg.modules.cv_detect.colorbar_extractor import extract_colorbar
from geoseg.modules.cv_detect.figure_classifier import classify
from geoseg.modules.cv_detect.panel_detector import detect_panels
from geoseg.modules.editor.editor_core import labels_to_shapes, shapes_to_labels
from geoseg.modules.exporter.specfem import (
    labels_to_grids,
    write_parfile_snippet,
    write_tomography_file,
)
from geoseg.modules.post_process import (
    assign_properties,
    extract_components,
    labels_to_polygons,
    load_properties_json,
    save_geojson,
    split_label_by_color_components,
)
from geoseg.modules.segment_engines import route_and_segment, run_engine
from geoseg.modules.segment_engines.edge_grow import segment as segment_edge_grow
from geoseg.modules.segment_engines.edge_guided import segment as segment_edge_guided
from geoseg.modules.segment_engines.ensemble import segment as segment_ensemble
from geoseg.modules.segment_engines.grayscale import (
    segment as segment_grayscale_agglomerative,
)
from geoseg.modules.segment_engines.horizon.occlusion import (
    bridge_boundary_with_endpoint_tangents,
    repair_white_label_occluded_boundaries,
)
from geoseg.modules.segment_engines.horizon_refinement import refine_boundaries
from geoseg.modules.segment_engines.kmeans_full import segment as segment_kmeans_full
from geoseg.modules.segment_engines.metrics import compute_all
from geoseg.modules.segment_engines.slic_kmeans import segment as segment_slic_kmeans
from geoseg.modules.segment_engines.strategy_memory import record_attempt
from geoseg.modules.segment_engines.v4_kmeans import segment as segment_v4_kmeans
from geoseg.modules.segment_engines.vlm_reps import (
    color_zones_to_reps,
    vertical_scan_reps,
)
from geoseg.modules.vlm_client.client import (
    classify_figure,
    review_page_overview,
    review_segmentation_quality,
)
from geoseg.pipeline.export import (
    export_segmented_panels,
    run_post_process_and_export,
)
from geoseg.pipeline.run import run_pipeline
from geoseg.pipeline.segment import run_segmentation_stage
from geoseg.session_state import load_session, save_session

__all__ = [
    # end-to-end pipeline
    "run_pipeline",
    "run_segmentation_stage",
    "run_post_process_and_export",
    "export_segmented_panels",
    # inspect / classify
    "classify",
    "classify_figure",
    "detect_panels",
    "extract_colorbar",
    "review_page_overview",
    "review_segmentation_quality",
    # segment engines
    "route_and_segment",
    "run_engine",
    "segment_edge_grow",
    "segment_edge_guided",
    "segment_ensemble",
    "segment_grayscale_agglomerative",
    "segment_kmeans_full",
    "segment_slic_kmeans",
    "segment_v4_kmeans",
    "refine_boundaries",
    "color_zones_to_reps",
    "vertical_scan_reps",
    "compute_all",
    "record_attempt",
    # image helpers
    "adaptive_blur",
    "create_overlay",
    "detect_background_label",
    "distinct_colors",
    "estimate_noise_level",
    "merge_small_regions",
    "row_median_filter",
    "saturation_ratio",
    # post-process / export
    "assign_properties",
    "extract_components",
    "labels_to_polygons",
    "load_properties_json",
    "save_geojson",
    "split_label_by_color_components",
    "labels_to_grids",
    "write_parfile_snippet",
    "write_tomography_file",
    # editing / boundary geometry
    "labels_to_shapes",
    "shapes_to_labels",
    "bridge_boundary_with_endpoint_tangents",
    "repair_white_label_occluded_boundaries",
    # session
    "load_session",
    "save_session",
    # contracts
    "BBoxXYWH",
    "FigureClassification",
    "PageOverview",
    "PanelInput",
    "RegionalAudit",
    "RGBColor",
    "SegmentationMeta",
    "SegmentationResult",
]
