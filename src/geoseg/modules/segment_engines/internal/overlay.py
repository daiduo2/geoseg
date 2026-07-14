"""Overlay rendering helpers for segmentation engines."""

from __future__ import annotations

import numpy as np

from geoseg.core.image_ops import create_overlay


def _create_overlay(
    panel_rgb: np.ndarray,
    labels: np.ndarray,
    seeds_rgb: np.ndarray,
    alpha: float = 0.65,
    boundary_mode: str = "thin",
    skip_background: bool = True,
    min_area_frac: float = 0.002,
    fill_mode: str = "blend",
    overlay_colors: np.ndarray | None = None,
) -> np.ndarray:
    """Create segmentation overlay with vivid, perceptually distinct region colors.

    Args:
        panel_rgb: Original RGB image.
        labels: Label map (int array).
        seeds_rgb: Color palette, one per label (used only for sizing if
            overlay_colors is not provided).
        alpha: Blending strength [0, 1].  Higher = more visible mask.
            Default 0.65 so VLM can clearly distinguish regions.
        boundary_mode: "thin" | "thick" | "inner".
        skip_background: If True, auto-detect and skip the background label.
        min_area_frac: Merge connected components smaller than this fraction
            before drawing the overlay.
        fill_mode:
            - "blend": alpha-blend distinct colors onto original (default).
            - "solid": strong alpha (0.85) blend, almost opaque.
            - "mask": pure mask with distinct colors, no original image.
        overlay_colors: Optional (n, 3) uint8 array of explicit colors.
            If None, auto-generates high-contrast distinct colors.
    """
    return create_overlay(
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


__all__ = ["_create_overlay"]
