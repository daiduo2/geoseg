"""Retry policy for segmentation routing."""

from __future__ import annotations

import numpy as np

from geoseg.modules.segment_engines.runner import run_engine
from geoseg.pipeline_interfaces import SegmentationResult


RETRY_CHAIN: dict[str, str] = {
    "grayscale_agglomerative": "v4_kmeans",
    "v4_kmeans_pastel": "v4_kmeans_colorbar",
    "v4_kmeans_colorbar": "kmeans_full",
    "v4_kmeans": "edge_guided",
    "edge_guided": "edge_grow",
    "edge_grow": "kmeans_full",
    "kmeans_full": "edge_guided",
}


def count_foreground_labels(labels: np.ndarray) -> int:
    """Count non-background labels in a segmentation array."""
    return len(set(labels.flatten()) - {0})


def retry_undersegmentation(
    seg: SegmentationResult,
    *,
    initial_engine: str,
    panel_rgb: np.ndarray,
    reps: list[dict] | None,
    colorbar_rgb: np.ndarray | None,
    n_layers: int,
    n_color_zones: int = 0,
) -> SegmentationResult:
    """Retry with the configured fallback engine if the result is undersegmented."""
    n_found = count_foreground_labels(seg["labels"])
    if n_found >= 2:
        return seg

    retry_engine = RETRY_CHAIN.get(initial_engine)
    if not retry_engine:
        return seg

    seg_retry = run_engine(
        retry_engine,
        panel_rgb,
        reps,
        colorbar_rgb,
        n_layers,
        n_color_zones=n_color_zones,
    )
    n_found_retry = count_foreground_labels(seg_retry["labels"])
    if n_found_retry <= n_found:
        return seg

    seg_retry["meta"]["retry_from"] = initial_engine
    seg_retry["meta"]["retry_reason"] = f"under_segmented_{n_found}_layers"
    return seg_retry


__all__ = ["RETRY_CHAIN", "count_foreground_labels", "retry_undersegmentation"]
