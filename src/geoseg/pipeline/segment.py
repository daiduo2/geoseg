"""Segmentation stage wrapper.

This module owns the controller-facing segmentation stage. Algorithm selection
and engine execution remain inside ``modules.segment_engines``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from geoseg.modules.segment_engines.full_pipeline import process_figure


def run_segmentation_stage(
    img_rgb: np.ndarray,
    *,
    caption: str = "",
    text_blocks: list[dict] | None = None,
    n_layers: int = 5,
    quality_preference: str = "balanced",
    skip_non_velocity_model: bool = True,
    use_vlm: bool = True,
) -> dict[str, Any]:
    """Run classification, panel detection, and segmentation for one figure."""
    return process_figure(
        img_rgb,
        caption=caption,
        text_blocks=text_blocks,
        n_layers=n_layers,
        quality_preference=quality_preference,
        skip_non_velocity_model=skip_non_velocity_model,
        use_vlm=use_vlm,
    )


__all__ = ["run_segmentation_stage"]
