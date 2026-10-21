"""Agent-driven audit-and-iterate segmentation infrastructure."""

from geoseg.modules.iteration.audit import (
    Action,
    AuditConfig,
    AuditReport,
    LabelAudit,
    audit_labels,
)
from geoseg.modules.iteration.cleanup import (
    TextPixelReport,
    clean_text_pixels,
    text_pixel_mask,
)
from geoseg.modules.iteration.preprocess import CropResult, crop_panel, detect_panel_bbox
from geoseg.modules.iteration.session import (
    AttemptRecord,
    IterationSession,
    compact_labels,
)

__all__ = [
    "Action",
    "AttemptRecord",
    "AuditConfig",
    "AuditReport",
    "CropResult",
    "IterationSession",
    "LabelAudit",
    "TextPixelReport",
    "audit_labels",
    "clean_text_pixels",
    "compact_labels",
    "crop_panel",
    "detect_panel_bbox",
    "text_pixel_mask",
]
