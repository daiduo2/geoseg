"""Legacy full segmentation pipeline facade."""

from __future__ import annotations

import numpy as np

from geoseg.pipeline.segment import run_segmentation_stage
from geoseg.pipeline.stages import (
    panel_complexity_score as _panel_complexity_score,
)


def process_figure(
    img_rgb: np.ndarray,
    caption: str = "",
    text_blocks: list[dict] | None = None,
    n_layers: int = 5,
    quality_preference: str = "balanced",
    skip_non_velocity_model: bool = True,
    use_vlm: bool = True,
    target_panel_id: int = -1,
) -> dict:
    """Process a raw extracted figure image through the compatibility facade."""
    return run_segmentation_stage(
        img_rgb,
        caption=caption,
        text_blocks=text_blocks,
        n_layers=n_layers,
        quality_preference=quality_preference,
        skip_non_velocity_model=skip_non_velocity_model,
        use_vlm=use_vlm,
        target_panel_id=target_panel_id,
    )


__all__ = ["_panel_complexity_score", "process_figure"]
