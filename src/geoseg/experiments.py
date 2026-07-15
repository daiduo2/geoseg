"""Convenience facade for repository experiment scripts.

This module keeps scripts from binding directly to low-level module paths while
remaining explicit that these APIs are for experiments, audits, and ad-hoc
batch runs rather than product orchestration.
"""

from geoseg.modules.cv_detect.colorbar_extractor import extract_colorbar
from geoseg.modules.cv_detect.figure_classifier import classify
from geoseg.modules.cv_detect.panel_detector import detect_panels
from geoseg.modules.segment_engines import route_and_segment, run_engine
from geoseg.modules.segment_engines.metrics import compute_all
from geoseg.modules.segment_engines.strategy_memory import record_attempt
from geoseg.modules.segment_engines.edge_grow import segment as segment_edge_grow
from geoseg.modules.segment_engines.edge_guided import segment as segment_edge_guided
from geoseg.modules.segment_engines.ensemble import segment as segment_ensemble
from geoseg.modules.segment_engines.grayscale import (
    segment as segment_grayscale_agglomerative,
)
from geoseg.modules.segment_engines.horizon_refinement import refine_boundaries
from geoseg.modules.segment_engines.kmeans_full import segment as segment_kmeans_full
from geoseg.modules.segment_engines.slic_kmeans import segment as segment_slic_kmeans
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

__all__ = [
    "classify",
    "classify_figure",
    "color_zones_to_reps",
    "compute_all",
    "detect_panels",
    "extract_colorbar",
    "refine_boundaries",
    "record_attempt",
    "review_page_overview",
    "review_segmentation_quality",
    "route_and_segment",
    "run_engine",
    "segment_edge_grow",
    "segment_edge_guided",
    "segment_ensemble",
    "segment_grayscale_agglomerative",
    "segment_kmeans_full",
    "segment_slic_kmeans",
    "segment_v4_kmeans",
    "vertical_scan_reps",
]
