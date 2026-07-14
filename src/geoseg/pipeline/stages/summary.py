"""Segmentation pipeline summary helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from geoseg.core.image_ops import saturation_ratio


def summarize_pipeline_result(
    img_rgb: np.ndarray,
    *,
    classification: dict[str, Any],
    panel_results: list[dict[str, Any]],
    total_layers: int,
    engines_used: set[str],
    review_warnings: list[str],
    has_colorbar_hint: bool,
    target_panel_id: int,
) -> dict[str, Any]:
    """Build the figure-level segmentation result payload."""
    return {
        "classification": classification,
        "panels": panel_results,
        "summary": {
            "status": "ok",
            "n_panels": len(panel_results),
            "total_layers": total_layers,
            "engines_used": sorted(engines_used),
            "saturation_ratio": round(saturation_ratio(img_rgb), 4),
            "review_warnings": review_warnings,
            "vlm_has_colorbar": has_colorbar_hint,
            "vlm_target_panel_id": target_panel_id,
        },
    }


__all__ = ["summarize_pipeline_result"]
