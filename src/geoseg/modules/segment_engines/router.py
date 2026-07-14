"""Compatibility facade for segmentation routing.

New code should use the smaller modules directly:
- ``policy.select_engine`` for routing decisions
- ``runner.run_engine`` for engine execution
- ``retry.retry_undersegmentation`` for retry policy
"""

from __future__ import annotations

import numpy as np

from geoseg.modules.segment_engines.policy import (
    _edge_density,
    _is_grayscale,
    _stable_panel_hash,
    select_engine,
)
from geoseg.modules.segment_engines.retry import RETRY_CHAIN as _RETRY_CHAIN
from geoseg.modules.segment_engines.retry import retry_undersegmentation
from geoseg.modules.segment_engines.runner import _normalize_result
from geoseg.modules.segment_engines.runner import run_engine as _run_engine
from geoseg.core.models import SegmentationResult


def route_and_segment(
    panel_rgb: np.ndarray,
    reps: list[dict] | None = None,
    colorbar_rgb: np.ndarray | None = None,
    n_layers: int = 5,
    quality_preference: str = "balanced",
    is_velocity_model: bool = True,
    retry_on_underseg: bool = True,
    n_color_zones: int = 0,
) -> SegmentationResult:
    """Route to an engine, run segmentation, and optionally retry undersegmentation."""
    engine = select_engine(
        panel_rgb,
        quality_preference=quality_preference,
        has_colorbar=colorbar_rgb is not None and colorbar_rgb.size > 0,
        is_velocity_model=is_velocity_model,
    )

    if engine == "skip":
        return {
            "labels": np.zeros(panel_rgb.shape[:2], dtype=np.int32),
            "overlay": panel_rgb.copy(),
            "meta": {
                "engine": "skip",
                "color_names": [],
                "n_layers": n_layers,
                "reason": "not_velocity_model",
            },
        }

    seg = _run_engine(
        engine,
        panel_rgb,
        reps,
        colorbar_rgb,
        n_layers,
        n_color_zones=n_color_zones,
    )

    if not retry_on_underseg:
        return seg

    return retry_undersegmentation(
        seg,
        initial_engine=engine,
        panel_rgb=panel_rgb,
        reps=reps,
        colorbar_rgb=colorbar_rgb,
        n_layers=n_layers,
        n_color_zones=n_color_zones,
    )


__all__ = [
    "_RETRY_CHAIN",
    "_edge_density",
    "_is_grayscale",
    "_normalize_result",
    "_run_engine",
    "_stable_panel_hash",
    "route_and_segment",
    "select_engine",
]
