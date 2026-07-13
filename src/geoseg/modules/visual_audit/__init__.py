"""Visual audit module: agent-driven visual critic for segmentation quality.

Generates problem-exposing views and diagnostic signals for agent judgment.
This module does NOT make accept/reject decisions and does NOT enforce hard
gates. The agent inspects the overlay-with-legend and original panel, then
outputs a structured RegionalAudit that drives repair.
"""
from __future__ import annotations

from geoseg.modules.visual_audit.color_residual import (
    compute_color_residual_audit,
    compute_color_residual_map,
    compute_label_representative_colors,
    compute_label_residual_stats,
    create_color_residual_overlay,
    estimate_text_mask,
    find_high_deviation_regions,
)
from geoseg.modules.visual_audit.crops import create_audit_crops, save_crops
from geoseg.modules.visual_audit.report import create_audit_report
from geoseg.modules.visual_audit.semantic import compute_semantic_fidelity
from geoseg.modules.visual_audit.views import (
    create_audit_views,
    create_color_residual_heatmap,
    save_views,
)

__all__ = [
    "compute_color_residual_audit",
    "compute_color_residual_map",
    "compute_label_representative_colors",
    "compute_label_residual_stats",
    "create_color_residual_overlay",
    "estimate_text_mask",
    "find_high_deviation_regions",
    "compute_semantic_fidelity",
    "create_audit_crops",
    "save_crops",
    "create_audit_report",
    "create_audit_views",
    "create_color_residual_heatmap",
    "save_views",
]
