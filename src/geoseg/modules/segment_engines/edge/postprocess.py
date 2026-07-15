"""Post-processing helpers for edge-based engines."""

from __future__ import annotations

import numpy as np

from geoseg.modules.segment_engines.internal.regions import (
    _merge_small_regions,
    _shape_filter,
)


def postprocess_edge_labels(
    labels: np.ndarray,
    min_area_frac: float = 0.003,
) -> np.ndarray:
    """Apply the standard edge-engine label cleanup chain."""
    labels = _shape_filter(labels)
    return _merge_small_regions(labels, min_area_frac=min_area_frac)


__all__ = ["postprocess_edge_labels"]
