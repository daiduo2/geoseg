"""Post-process segmentation labels by merging artifact residuals into neighbors.

Small or dark connected components are absorbed into the most common adjacent
label using boundary adjacency.  This is a zoning/partitioning operation: the
goal is to remove annotation residuals, not to create new geological meaning.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def merge_artifact_labels(
    labels: np.ndarray,
    image_rgb: np.ndarray | None = None,
    min_area_frac: float = 0.001,
    max_mean_brightness: int | None = None,
    connectivity: int = 2,
    artifact_labels: list[int] | None = None,
    artifact_mask: np.ndarray | None = None,
    mask_overlap_frac: float = 0.5,
) -> np.ndarray:
    """Merge small, dark, or mask-overlapping artifact components into neighbors.

    The function operates on connected components rather than whole labels, so
    a large geological label that contains a small artifact island can have only
    the island merged without losing the rest of the label.  When an
    ``artifact_mask`` is supplied, only the portion of a component that falls
    inside the mask is merged, leaving the unmasked portion intact.

    Args:
        labels: Integer label map.
        image_rgb: Optional original RGB image.  If provided, regions whose
            mean brightness is below ``max_mean_brightness`` are also merged.
        min_area_frac: Components whose area is below this fraction of the image
            are candidates for merging.
        max_mean_brightness: If provided, components with mean brightness below
            this value are also candidates.
        connectivity: 1 (4-neighbor) or 2 (8-neighbor).
        artifact_labels: Optional list of label IDs whose components should be
            merged regardless of size or brightness.
        artifact_mask: Optional boolean mask of detected artifacts.  Only the
            part of each component that overlaps this mask is merged.
        mask_overlap_frac: If a component overlaps the mask by at least this
            fraction, the *entire* component is merged.  Otherwise only the
            masked sub-portion is merged.

    Returns:
        Label map with artifact components merged.
    """
    out = labels.copy()
    h, w = out.shape
    total = h * w

    structure = ndimage.generate_binary_structure(2, connectivity)
    explicit = set(artifact_labels) if artifact_labels is not None else set()

    def _merge_region(region: np.ndarray, owner_lbl: int) -> bool:
        """Merge ``region`` into its most common neighbor label."""
        if not region.any():
            return False
        dilated = ndimage.binary_dilation(region, structure=structure)
        boundary = dilated & ~region
        neighbors = out[boundary]
        neighbors = neighbors[(neighbors != owner_lbl) & (neighbors >= 0)]
        if len(neighbors) == 0:
            return False
        target = int(np.bincount(neighbors).argmax())
        out[region] = target
        return True

    # Process from smallest to largest so merges cascade correctly.
    unique = np.unique(out)
    areas = {int(lbl): int((out == lbl).sum()) for lbl in unique}
    sorted_labels = sorted(unique, key=lambda lbl: areas[int(lbl)])

    for lbl in sorted_labels:
        if lbl < 0:
            continue
        mask = out == lbl
        if not mask.any():
            continue

        is_explicit = int(lbl) in explicit
        labeled_components, n_components = ndimage.label(mask, structure=structure)

        for comp_id in range(1, n_components + 1):
            comp = labeled_components == comp_id
            comp_area = int(comp.sum())
            if comp_area == 0:
                continue

            merge_whole = is_explicit
            if not merge_whole:
                comp_area_frac = comp_area / total
                is_small = comp_area_frac < min_area_frac

                is_dark = False
                if image_rgb is not None and max_mean_brightness is not None:
                    mean_brightness = float(image_rgb[comp].mean())
                    is_dark = mean_brightness < max_mean_brightness

                merge_whole = is_small or is_dark

            if not merge_whole and artifact_mask is not None:
                overlap = float((comp & artifact_mask).sum()) / comp_area
                if overlap >= mask_overlap_frac:
                    merge_whole = True

            if merge_whole:
                _merge_region(comp, lbl)
            elif artifact_mask is not None:
                inside = comp & artifact_mask
                if inside.any():
                    _merge_region(inside, lbl)

    return out
