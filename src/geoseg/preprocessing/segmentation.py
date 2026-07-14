"""Segmentation adapters used by preprocessing pipelines."""

from __future__ import annotations

from typing import Any

import numpy as np

from geoseg.modules.segment_engines import run_engine
from geoseg.modules.visual_audit.rendering import create_overlay_with_legend


def segment_artifact_baseline(
    image_rgb: np.ndarray,
    *,
    n_layers: int,
) -> dict[str, Any]:
    """Run the default artifact-absorption comparison segmenter."""
    return run_engine("v4_kmeans", image_rgb, None, None, n_layers)


def segment_artifact_colorbar_guided(
    image_rgb: np.ndarray,
    colorbar_rgb: np.ndarray,
    *,
    n_layers: int,
) -> dict[str, Any]:
    """Run colorbar-guided segmentation for artifact-absorption comparisons."""
    return run_engine("v4_kmeans_colorbar", image_rgb, None, colorbar_rgb, n_layers)


def create_audit_overlay(
    image_rgb: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Create the label-legend overlay expected by preprocessing audits."""
    return create_overlay_with_legend(image_rgb, labels)


__all__ = [
    "create_audit_overlay",
    "segment_artifact_baseline",
    "segment_artifact_colorbar_guided",
]
