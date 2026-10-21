"""Background component semantics shared by exact boundary decisions.

Label 0 is not automatically "noise".  A zero component touching the panel
border (or occupying a large share of the panel) is exterior surface; a
confined component is an internal hole such as text or occlusion.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def background_roles(labels: np.ndarray) -> dict[int, str]:
    """Classify zero components as exterior background or internal holes."""
    labels = np.asarray(labels)
    zero = labels == 0
    components, n_components = ndimage.label(zero)
    h, w = labels.shape
    total = h * w
    roles: dict[int, str] = {}
    for component_id in range(1, n_components + 1):
        mask = components == component_id
        rows, cols = np.nonzero(mask)
        if len(rows) == 0:
            continue
        touches_border = (
            rows.min() == 0
            or cols.min() == 0
            or rows.max() == h - 1
            or cols.max() == w - 1
        )
        area_fraction = len(rows) / total
        bbox_fraction = (
            (rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1) / total
        )
        if touches_border or area_fraction >= 0.15 or bbox_fraction >= 0.5:
            roles[component_id] = "exterior"
        else:
            roles[component_id] = "internal_hole"
    return roles


__all__ = ["background_roles"]
