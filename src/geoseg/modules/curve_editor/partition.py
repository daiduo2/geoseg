"""Deterministic cleanup of tiny partition components.

The boundary is a function of the partition.  Therefore fragment cleanup must
merge small *label components* first and recompute the boundary afterwards; it
must not delete boundary pixels directly.

The rule here is intentionally conservative and deterministic:

* components are considered in increasing (area, label, row, column) order;
* a component is removable only if both its area and its maximum inscribed
  radius are small, so a long thin layer is not removed just because it is thin;
* it merges into the 4-neighbour label with the longest shared boundary;
* ties go to the smaller label id;
* background (label 0) is protected by default, and a component is not merged
  into background unless explicitly allowed.

Every merge is returned as a record so the operation can be audited or undone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

_CROSS = np.array(
    [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ],
    dtype=bool,
)


@dataclass(frozen=True)
class ComponentMerge:
    label: int
    area: int
    max_thickness: float
    target: int
    shared_boundary: int
    seed_rc: tuple[int, int]


def _component_seed(mask: np.ndarray) -> tuple[int, int]:
    rows, cols = np.nonzero(mask)
    return int(rows.min()), int(cols.min())


def _merge_target(
    labels: np.ndarray,
    component: np.ndarray,
    label: int,
    *,
    allow_background_merge: bool,
) -> tuple[int, int] | None:
    dilated = ndimage.binary_dilation(component, structure=_CROSS)
    neighbours = labels[dilated & ~component]
    neighbours = neighbours[neighbours != label]
    if neighbours.size == 0:
        return None
    values, counts = np.unique(neighbours, return_counts=True)
    candidates = [
        (int(value), int(count))
        for value, count in zip(values, counts)
        if allow_background_merge or int(value) != 0
    ]
    if not candidates:
        return None
    # Longest shared boundary first; ties go to the smaller label id.
    target, shared = max(candidates, key=lambda item: (item[1], -item[0]))
    return target, shared


def prune_small_components(
    labels: np.ndarray,
    *,
    min_area: int = 25,
    max_thickness: float = 2.5,
    protect_background: bool = True,
    allow_background_merge: bool = False,
    max_passes: int = 16,
) -> tuple[np.ndarray, list[ComponentMerge]]:
    """Merge tiny 4-connected label components into adjacent labels.

    Args:
        labels: 2-D integer partition.  Label 0 is background.
        min_area: Maximum pixel count of a removable component.
        max_thickness: Maximum distance-transform radius of a removable
            component.  This keeps long, thin layers even when their area is
            small.
        protect_background: Never remove a component whose label is 0.
        allow_background_merge: Allow a foreground fragment to merge into 0.
            Defaults to False because that erases a region into the exterior.
        max_passes: Fixed-point cap; a merge can expose another tiny component.

    Returns:
        ``(cleaned_labels, merges)``.  The merge order is deterministic.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"labels must be 2-D, got shape {labels.shape}")
    if min_area < 1:
        raise ValueError("min_area must be positive")
    if max_thickness <= 0:
        raise ValueError("max_thickness must be positive")

    out = labels.copy()
    merges: list[ComponentMerge] = []

    for _ in range(max_passes):
        changed = False
        for label in sorted(int(value) for value in np.unique(out)):
            if protect_background and label == 0:
                continue
            mask = out == label
            components, n_components = ndimage.label(mask, structure=_CROSS)
            for component_id in range(1, n_components + 1):
                component = components == component_id
                area = int(component.sum())
                if area == 0 or area >= min_area:
                    continue
                thickness = float(ndimage.distance_transform_edt(component).max())
                if thickness >= max_thickness:
                    continue
                target = _merge_target(
                    out,
                    component,
                    label,
                    allow_background_merge=allow_background_merge,
                )
                if target is None:
                    continue
                target_label, shared = target
                merges.append(
                    ComponentMerge(
                        label=label,
                        area=area,
                        max_thickness=thickness,
                        target=target_label,
                        shared_boundary=shared,
                        seed_rc=_component_seed(component),
                    )
                )
                out[component] = target_label
                changed = True
        if not changed:
            break
    return out, merges


__all__ = ["ComponentMerge", "prune_small_components"]
