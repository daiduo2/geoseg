"""Preprocessing utilities for artifact removal and panel preparation."""
from __future__ import annotations

from geoseg.preprocessing.absorption import absorb_artifacts, visualize_mask_on_image
from geoseg.preprocessing.detectors import (
    detect_black_crosses,
    detect_red_boundaries,
    detect_red_lines,
    detect_text,
)
from geoseg.preprocessing.label_merge import merge_artifact_labels
from geoseg.preprocessing.panel_split import split_panels_colored_components
from geoseg.preprocessing.pipeline import (
    ArtifactAbsorptionConfig,
    process_image,
)

__all__ = [
    "absorb_artifacts",
    "detect_black_crosses",
    "detect_red_boundaries",
    "detect_red_lines",
    "detect_text",
    "merge_artifact_labels",
    "process_image",
    "split_panels_colored_components",
    "visualize_mask_on_image",
    "ArtifactAbsorptionConfig",
]
