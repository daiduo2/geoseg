"""Label split/merge helpers for regional repair."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def fuse_with_freeze(
    base_labels: np.ndarray,
    patch_labels: np.ndarray,
    freeze_mask: np.ndarray,
    seam_width: int = 3,
) -> np.ndarray:
    """Fuse two label arrays, keeping base where frozen and patch elsewhere.

    Seam smoothing resolves narrow label discontinuities at the freeze boundary
    by nearest-neighbor fill within a transition band.
    """
    if base_labels.shape != patch_labels.shape:
        raise ValueError(
            f"Shape mismatch: base {base_labels.shape} vs patch {patch_labels.shape}"
        )
    if base_labels.shape != freeze_mask.shape:
        raise ValueError(
            f"Shape mismatch: labels {base_labels.shape} vs mask {freeze_mask.shape}"
        )

    result = patch_labels.copy()
    result[freeze_mask] = base_labels[freeze_mask]

    if seam_width <= 0:
        return result

    struct = np.ones((3, 3), dtype=bool)
    dilated = ndimage.binary_dilation(freeze_mask, structure=struct)
    boundary = dilated & ~freeze_mask

    if not boundary.any():
        return result

    if seam_width > 1:
        wide_struct = np.ones((seam_width * 2 - 1, seam_width * 2 - 1), dtype=bool)
        transition = ndimage.binary_dilation(boundary, structure=wide_struct)
    else:
        transition = boundary

    if not transition.any():
        return result

    gap_mask = (result == 0) & transition
    if not gap_mask.any():
        return result

    nz_mask = result != 0
    _, indices = ndimage.distance_transform_edt(~nz_mask, return_indices=True)

    rr, cc = np.where(gap_mask)
    result[rr, cc] = result[indices[0][rr, cc], indices[1][rr, cc]]
    return result


__all__ = ["fuse_with_freeze"]
