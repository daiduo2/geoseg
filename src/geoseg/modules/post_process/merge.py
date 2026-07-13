"""Label merging utilities for post-processing segmentation results.

Handles common artifacts like gradient-induced label splitting (e.g. a plume
funnel being cut into two labels by k-means due to colour variation).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def _hsv_warm_mask(img_rgb: np.ndarray) -> np.ndarray:
    """Return boolean mask of warm-colour pixels (orange/yellow/red)."""
    import cv2

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    # Hue: red wraps around 0, orange ~15-30, yellow ~30-60
    # Also catch red wrap-around at 170-180
    return (
        ((hsv[:, :, 0] < 60) | (hsv[:, :, 0] > 170))
        & (hsv[:, :, 1] > 40)
        & (hsv[:, :, 2] > 80)
    )


def _fill_plume_background(
    result: np.ndarray,
    img_rgb: np.ndarray,
    warm_labels: list[int],
    color_distance_thresh: float = 3.0,
    max_dist: int = 30,
    center_column_ratio: float = 1.0,
) -> np.ndarray:
    """Fill background pixels that are colour-similar to the merged plume.

    Uses a spatially-constrained approach to avoid swallowing side crust:

    1. **Distance-constrained fill** — only background pixels within
       ``max_dist`` of the existing plume are considered.  This prevents
       filling far-away warm regions (e.g. side crust) that happen to share
       a similar colour.
    2. **Warm-colour check** — only warm background pixels are filled,
       avoiding cool holes that should remain background.
    3. **Center-column constraint** — when ``center_column_ratio < 1``,
       fill is restricted to the central portion of the image, keeping
       edge crust out of the plume.
    4. **Hole fill** — small enclosed background holes inside the plume
       (often left by k-means on gradient boundaries) are filled regardless
       of distance.
    """
    plume_mask = result == 1
    if not plume_mask.any():
        return result

    h, w = result.shape
    warm_mask = _hsv_warm_mask(img_rgb)
    bg_mask = result == 0

    # Build center-column constraint mask
    if center_column_ratio < 1.0:
        left = int(w * (1 - center_column_ratio) / 2)
        right = int(w * (1 + center_column_ratio) / 2)
        center_mask = np.zeros((h, w), dtype=bool)
        center_mask[:, left:right] = True
    else:
        center_mask = np.ones((h, w), dtype=bool)

    # Step 1: Distance-constrained fill for background near the plume
    dist = ndimage.distance_transform_edt(~plume_mask)
    near_plume = (dist <= max_dist) & bg_mask & center_mask & warm_mask

    if near_plume.any():
        # Compute warm-pixel statistics from the already-merged plume
        plume_pixels = img_rgb[plume_mask]
        mean = plume_pixels.mean(axis=0)
        std = plume_pixels.std(axis=0)

        rgb_near = img_rgb[near_plume].astype(np.float32)
        dists = np.abs(rgb_near - mean) / (std + 10)
        similar = dists.max(axis=1) < color_distance_thresh

        result_out = result.copy()
        fill_idx = np.where(near_plume)
        result_out[fill_idx[0][similar], fill_idx[1][similar]] = 1
        result = result_out

    # Step 2: Fill small enclosed holes inside the plume
    bg_mask = result == 0
    if bg_mask.any():
        labeled, num = ndimage.label(bg_mask)
        for i in range(1, num + 1):
            comp = labeled == i
            if comp.sum() > 1000:
                continue

            dilated = ndimage.binary_dilation(comp, iterations=2)
            neighbors = dilated & ~comp
            neighbor_labels = result[neighbors]
            non_bg = neighbor_labels[neighbor_labels > 0]

            if len(non_bg) > 0 and np.all(non_bg == 1):
                # Only fill if the hole is inside the center region
                comp_center = comp & center_mask
                if comp_center.sum() / comp.sum() > 0.5:
                    result[comp] = 1

    # Step 3: Vertical extension — fill downward in centre columns through
    # warm pixels to recover the narrow bottom neck of funnel-shaped plumes.
    plume_mask = result == 1
    if plume_mask.any() and center_column_ratio < 1.0:
        left = int(w * (1 - center_column_ratio) / 2)
        right = int(w * (1 + center_column_ratio) / 2)

        # Find current plume bottom in the centre strip
        bottom_y = 0
        for x in range(left, right):
            col = plume_mask[:, x]
            if col.any():
                bottom_y = max(bottom_y, int(np.where(col)[0].max()))

        # Fill downward through warm background pixels
        extension = np.zeros((h, w), dtype=bool)
        for x in range(left, right):
            for y in range(bottom_y + 1, h):
                if warm_mask[y, x] and result[y, x] == 0:
                    extension[y, x] = True
                else:
                    break
        if extension.any():
            result[extension] = 1

    return result


def _crop_mask_to_center(
    mask: np.ndarray,
    center_column_ratio: float,
) -> np.ndarray:
    """Return a copy of *mask* with only the centre columns retained."""
    h, w = mask.shape
    if center_column_ratio >= 1.0:
        return mask.copy()
    left = int(w * (1 - center_column_ratio) / 2)
    right = int(w * (1 + center_column_ratio) / 2)
    out = mask.copy()
    out[:, :left] = False
    out[:, right:] = False
    return out


def merge_warm_labels(
    labels: np.ndarray,
    img_rgb: np.ndarray,
    warm_threshold: float = 0.5,
    min_component_size: int = 100,
    fill_background: bool = False,
    max_width_ratio: float = 0.85,
    center_column_ratio: float = 0.5,
    fill_max_dist: int = 30,
    exclude_labels: list[int] | None = None,
    per_label_crop: dict[int, float] | None = None,
) -> np.ndarray:
    """Merge labels that are majority warm-colour into a single label.

    Useful when colour-gradient regions (e.g. 3D-rendered plume funnels)
    are split into multiple labels by clustering engines.

    To avoid swallowing side crust that shares the same warm colour,
    wide labels (spanning more than ``max_width_ratio`` of the image)
    are only merged in their central ``center_column_ratio`` portion.

    Args:
        labels: Integer label map.
        img_rgb: Original RGB image (same shape).
        warm_threshold: Fraction of warm pixels required to merge a label.
        min_component_size: Small non-warm components inside the merged
            region that are smaller than this are absorbed.
        fill_background: If True, background pixels (label 0) that are
            colour-similar to the merged plume and lie within
            ``fill_max_dist`` of it are assigned to the plume label.
        max_width_ratio: Labels wider than this fraction of the image
            width are only merged in their centre.
        center_column_ratio: Fraction of image width to retain when a
            label exceeds ``max_width_ratio``.
        fill_max_dist: Maximum distance (pixels) from the merged plume
            for background fill.
        exclude_labels: Label IDs that should *not* be merged even if they
            pass the warm-colour threshold.  Useful when a label contains
            warm-colour regions (e.g. clouds) that are not part of the
            target plume.
        per_label_crop: Override ``center_column_ratio`` on a per-label
            basis.  ``{label_id: ratio}`` where *ratio* is the fraction of
            the image width to retain for that label.  Labels not in this
            dict fall back to the global ``center_column_ratio``.

    Returns:
        New label map with warm labels merged and renumbered.
    """
    if labels.shape[:2] != img_rgb.shape[:2]:
        raise ValueError("Shape mismatch between labels and image")

    warm_mask = _hsv_warm_mask(img_rgb)
    h, w = labels.shape
    unique = sorted(set(labels.flatten()) - {0})
    exclude_set = set(exclude_labels or [])
    per_crop = per_label_crop or {}

    # Identify warm-dominant labels
    warm_labels: list[int] = []
    for lbl in unique:
        if lbl in exclude_set:
            continue
        lbl_mask = labels == lbl
        if lbl_mask.sum() == 0:
            continue
        warm_frac = np.logical_and(lbl_mask, warm_mask).sum() / lbl_mask.sum()
        if warm_frac >= warm_threshold:
            warm_labels.append(lbl)

    if len(warm_labels) == 0:
        return labels.copy()

    # Build merge map: warm labels -> 1, with spatial cropping for wide labels
    result = np.zeros_like(labels)
    widest_crop_ratio = 0.0  # track the widest crop used for fill centre-mask
    any_cropped = False

    for lbl in warm_labels:
        lbl_mask = labels == lbl
        ys, xs = np.where(lbl_mask)
        width_ratio = (xs.max() - xs.min() + 1) / w
        area_ratio = lbl_mask.sum() / (h * w)

        # Per-label override or heuristic default
        if lbl in per_crop:
            crop_ratio = per_crop[lbl]
            do_crop = crop_ratio < 1.0
        elif width_ratio >= max_width_ratio and min(h, w) > 50:
            # Large, wide labels get conservative cropping to avoid
            # swallowing side crust.
            crop_ratio = center_column_ratio
            do_crop = crop_ratio < 1.0
        else:
            crop_ratio = 1.0
            do_crop = False

        if do_crop:
            merge_mask = _crop_mask_to_center(lbl_mask, crop_ratio)
            result[merge_mask] = 1
            widest_crop_ratio = max(widest_crop_ratio, crop_ratio)
            any_cropped = True
        else:
            result[lbl_mask] = 1

    # Re-number non-warm labels compactly starting from 2
    next_lbl = 2
    for lbl in unique:
        if lbl in warm_labels:
            continue
        result[labels == lbl] = next_lbl
        next_lbl += 1

    # Clean small non-warm fragments inside the merged warm region
    warm_region = result == 1
    if warm_region.any():
        y_idx, x_idx = np.where(warm_region)
        y0, y1 = y_idx.min(), y_idx.max()
        x0, x1 = x_idx.min(), x_idx.max()

        roi = result[y0 : y1 + 1, x0 : x1 + 1]
        roi_non_warm = (roi > 0) & (roi != 1)
        if roi_non_warm.any():
            labeled, num = ndimage.label(roi_non_warm)
            for i in range(1, num + 1):
                comp = labeled == i
                if comp.sum() < min_component_size:
                    roi[comp] = 1
            result[y0 : y1 + 1, x0 : x1 + 1] = roi

    if fill_background:
        fill_crop_ratio = widest_crop_ratio if any_cropped else 1.0
        result = _fill_plume_background(
            result,
            img_rgb,
            warm_labels,
            max_dist=fill_max_dist,
            center_column_ratio=fill_crop_ratio,
        )
        # Re-run fragment cleanup after fill
        warm_region = result == 1
        if warm_region.any():
            y_idx, x_idx = np.where(warm_region)
            y0, y1 = y_idx.min(), y_idx.max()
            x0, x1 = x_idx.min(), x_idx.max()
            roi = result[y0 : y1 + 1, x0 : x1 + 1]
            roi_non_warm = (roi > 0) & (roi != 1)
            if roi_non_warm.any():
                labeled, num = ndimage.label(roi_non_warm)
                for i in range(1, num + 1):
                    comp = labeled == i
                    if comp.sum() < min_component_size:
                        roi[comp] = 1
                result[y0 : y1 + 1, x0 : x1 + 1] = roi

    return result


def merge_labels_by_ids(
    labels: np.ndarray,
    label_ids: list[int],
    target_id: int = 1,
) -> np.ndarray:
    """Merge specific label IDs into a single target ID.

    Args:
        labels: Integer label map.
        label_ids: Labels to merge.
        target_id: The new label value for the merged region.

    Returns:
        New label map.
    """
    result = labels.copy()
    mask = np.isin(labels, label_ids)
    result[mask] = target_id
    return result


def remove_labels_by_ids(
    labels: np.ndarray,
    label_ids: list[int],
    fill: str = "nearest",
) -> np.ndarray:
    """Remove specific label IDs and fill the vacated pixels.

    Useful for deleting text/annotation labels identified by the audit agent.
    Removed pixels are reassigned to the nearest remaining label so that
    underlying geology is preserved instead of becoming background.

    Args:
        labels: Integer label map.
        label_ids: Label IDs to remove.
        fill: "nearest" fills from nearest remaining label; "background"
            sets removed pixels to 0.

    Returns:
        New label map with the specified labels removed.
    """
    result = labels.copy()
    remove_mask = np.isin(labels, label_ids)
    if not remove_mask.any():
        return result

    if fill == "background":
        result[remove_mask] = 0
        return result

    valid_mask = (~remove_mask) & (labels != 0)
    if not valid_mask.any():
        result[remove_mask] = 0
        return result

    _, indices = ndimage.distance_transform_edt(~valid_mask, return_indices=True)
    rr, cc = np.where(remove_mask)
    result[rr, cc] = labels[indices[0][rr, cc], indices[1][rr, cc]]
    return result


def filter_small_components(
    labels: np.ndarray,
    min_area: int | None = None,
    min_area_ratio: float = 0.0005,
    fill: str = "nearest",
) -> np.ndarray:
    """Remove connected components that are smaller than a size threshold.

    Args:
        labels: Integer label map.
        min_area: Absolute pixel count threshold. If None, derived from
            min_area_ratio * image_area.
        min_area_ratio: Fraction of image area used when min_area is None.
        fill: "nearest" fills removed pixels from nearest remaining label;
            "background" sets them to 0.

    Returns:
        New label map with tiny components removed.
    """
    result = labels.copy()
    h, w = labels.shape
    total = h * w
    threshold = min_area if min_area is not None else int(total * min_area_ratio)

    tiny_mask = np.zeros_like(labels, dtype=bool)
    for lbl in sorted(set(labels.flatten()) - {0}):
        lbl_mask = labels == lbl
        labeled, num = ndimage.label(lbl_mask)
        for i in range(1, num + 1):
            comp = labeled == i
            if comp.sum() < threshold:
                tiny_mask[comp] = True

    if not tiny_mask.any():
        return result

    if fill == "background":
        result[tiny_mask] = 0
        return result

    valid_mask = (~tiny_mask) & (labels != 0)
    if not valid_mask.any():
        result[tiny_mask] = 0
        return result

    _, indices = ndimage.distance_transform_edt(~valid_mask, return_indices=True)
    rr, cc = np.where(tiny_mask)
    result[rr, cc] = labels[indices[0][rr, cc], indices[1][rr, cc]]
    return result
