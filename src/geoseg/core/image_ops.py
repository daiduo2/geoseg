"""Stable image utility facade shared outside segmentation engine internals."""

from __future__ import annotations

import numpy as np

from geoseg.modules.segment_engines._shared import (
    _create_overlay as _engine_create_overlay,
    _distinct_colors as _engine_distinct_colors,
    saturation_ratio as _engine_saturation_ratio,
)


def create_overlay(
    panel_rgb: np.ndarray,
    labels: np.ndarray,
    seeds_rgb: np.ndarray | None,
    alpha: float = 0.65,
    boundary_mode: str = "thin",
    skip_background: bool = True,
    min_area_frac: float = 0.002,
    fill_mode: str = "blend",
    overlay_colors: np.ndarray | None = None,
) -> np.ndarray:
    """Create a segmentation overlay using the shared engine renderer."""
    if seeds_rgb is None:
        seeds_rgb = distinct_colors(int(labels.max()) + 1)
    return _engine_create_overlay(
        panel_rgb,
        labels,
        seeds_rgb,
        alpha=alpha,
        boundary_mode=boundary_mode,
        skip_background=skip_background,
        min_area_frac=min_area_frac,
        fill_mode=fill_mode,
        overlay_colors=overlay_colors,
    )


def distinct_colors(n: int, saturation: float = 0.88, value: float = 0.95) -> np.ndarray:
    """Generate perceptually distinct colors for labels or legends."""
    return _engine_distinct_colors(n, saturation=saturation, value=value)


def saturation_ratio(panel_rgb: np.ndarray, threshold: int = 80) -> float:
    """Return the fraction of pixels above the saturation threshold."""
    return _engine_saturation_ratio(panel_rgb, threshold=threshold)


__all__ = ["create_overlay", "distinct_colors", "saturation_ratio"]
