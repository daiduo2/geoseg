"""Rendering helpers used by visual audit views."""

from __future__ import annotations

import numpy as np

from geoseg.modules.segment_engines.regional_fusion import (
    draw_legend,
    generate_overlay_with_legend,
)
from geoseg.modules.segment_engines.regions import reorder_labels_top_to_bottom


def create_overlay_with_legend(
    image_rgb: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Return a segmentation overlay with an embedded legend."""
    return generate_overlay_with_legend(image_rgb, labels)


def labels_ordered_top_to_bottom(labels: np.ndarray) -> np.ndarray:
    """Return labels remapped by vertical median position."""
    return reorder_labels_top_to_bottom(labels)


def draw_overlay_legend(
    overlay_rgb: np.ndarray,
    labels: np.ndarray,
    label_colors: dict[int, np.ndarray] | None = None,
    box_size: int = 12,
    font_size: int = 10,
) -> np.ndarray:
    """Draw a label legend on an existing overlay."""
    return draw_legend(
        overlay_rgb,
        labels,
        label_colors=label_colors,
        box_size=box_size,
        font_size=font_size,
    )
