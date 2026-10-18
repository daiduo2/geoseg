"""Backward-compatible imports for pipeline contracts.

New code should import these contracts from ``geoseg.core.models``. This
module remains so existing scripts and tests can migrate gradually.
"""

from geoseg.core.models import (
    BBoxXYWH,
    FigureClassification,
    PageOverview,
    PanelDetector,
    PanelInput,
    PipelineStep,
    QualityReview,
    QualityReviewer,
    RGBColor,
    RegionalAudit,
    Segmenter,
    SegmentationMeta,
    SegmentationResult,
    coerce_bbox_xywh,
    empty_segmentation_result,
    make_panel_input,
    make_whole_image_panel,
    validate_segmentation_result,
)

__all__ = [
    "BBoxXYWH",
    "FigureClassification",
    "PageOverview",
    "PanelDetector",
    "PanelInput",
    "PipelineStep",
    "QualityReview",
    "QualityReviewer",
    "RGBColor",
    "RegionalAudit",
    "Segmenter",
    "SegmentationMeta",
    "SegmentationResult",
    "coerce_bbox_xywh",
    "empty_segmentation_result",
    "make_panel_input",
    "make_whole_image_panel",
    "validate_segmentation_result",
]
