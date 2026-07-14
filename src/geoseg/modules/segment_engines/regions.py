"""Public region-label helpers for segmentation engines."""

from __future__ import annotations

import numpy as np

from geoseg.modules.segment_engines.internal.regions import _reorder_labels_by_median_y


def reorder_labels_top_to_bottom(labels: np.ndarray) -> np.ndarray:
    """Return labels remapped so top regions receive lower IDs."""
    return _reorder_labels_by_median_y(labels)


__all__ = ["reorder_labels_top_to_bottom"]
