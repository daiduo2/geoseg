"""Boundary extraction and label adjustment helpers for horizon refinement."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from geoseg.modules.segment_engines.horizon.coarse import _separator_mask


def _extract_boundary_points(
    panel_rgb: np.ndarray,
    coarse_labels: np.ndarray,
    layer_i: int,
    layer_j: int,
    label_blur_sigma: float = 5.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Phase B: sample boundary points between two adjacent layers.

    Creates a signed map from the coarse labels (+1 for layer_i, -1 for
    layer_j), applies Gaussian blur to smooth out fragmentation, then finds
    zero-crossing points per column. This is more robust than gradient-based
    sampling on the original RGB image because it averages out small fragments
    and finds the visual center of the transition band.
    """
    h, w = panel_rgb.shape[:2]

    # Signed map: layer_i = +1, layer_j = -1, others = 0
    signed = np.zeros((h, w), dtype=np.float32)
    signed[coarse_labels == layer_i] = 1.0
    signed[coarse_labels == layer_j] = -1.0

    # Gaussian blur to smooth out fragments and noise
    blurred = ndimage.gaussian_filter(signed, sigma=label_blur_sigma)

    xs = []
    ys = []

    for x in range(w):
        col = blurred[:, x]

        # Find zero-crossing: where col transitions from positive to negative
        pos_mask = col > 0
        neg_mask = col < 0

        if not pos_mask.any() or not neg_mask.any():
            continue

        # Find the crossing from + to - (top to bottom)
        found = False
        for y in range(h - 1):
            if col[y] > 0 and col[y + 1] <= 0:
                denom = col[y] - col[y + 1]
                if denom > 1e-6:
                    y_interp = y + col[y] / denom
                else:
                    y_interp = y + 0.5
                xs.append(x)
                ys.append(float(y_interp))
                found = True
                break

        if not found:
            # Fallback: use bottom edge of positive region
            pos_ys = np.where(pos_mask)[0]
            if len(pos_ys) > 0:
                xs.append(x)
                ys.append(float(pos_ys[-1]))

    if len(xs) < 3:
        return None

    ys_arr = np.array(ys, dtype=np.float32)

    # If the zero-crossing position varies wildly (MAD > h/10), the boundary
    # is not spatially coherent — this usually means the layers are not truly
    # adjacent or the coarse segmentation is on a smooth gradient. Skip fitting.
    mad = float(np.median(np.abs(ys_arr - np.median(ys_arr))))
    if mad > h / 10:
        return None

    return np.array(xs, dtype=np.int32), ys_arr


def _extract_boundary_dense(
    coarse_labels: np.ndarray,
    top_lbl: int,
    bot_lbl: int,
) -> np.ndarray:
    """Extract boundary candidates for non-touching fragmented layers.

    When two layers are so fragmented that they don't share any touching
    pixels, the label-blur zero-crossing method fails. Instead, we sample
    per-column using percentile-based edge detection on the raw label map:
    - top layer's lower edge = 50th percentile (median) of its pixels in the column
    - bottom layer's upper edge = 50th percentile (median) of its pixels in the column
    - boundary candidate = midpoint between these edges

    This treats each layer's fragments as an "archipelago" and finds the
    transition band between archipelagos.
    """
    h, w = coarse_labels.shape
    ys = np.full(w, np.nan)

    for x in range(w):
        col = coarse_labels[:, x]
        top_mask = col == top_lbl
        bot_mask = col == bot_lbl

        if not top_mask.any() or not bot_mask.any():
            continue

        top_ys = np.where(top_mask)[0]
        bot_ys = np.where(bot_mask)[0]

        # Use median (50th percentile) to locate the visual center of mass
        # of each fragmented layer. More accurate than 90/10 for highly
        # fragmented archipelagos where extreme percentiles are biased
        # by sparse outlier fragments.
        top_lower = float(np.percentile(top_ys, 50))
        bot_upper = float(np.percentile(bot_ys, 50))

        if bot_upper > top_lower:
            # Normal separation: boundary is in the gap
            ys[x] = (top_lower + bot_upper) / 2
        else:
            # Interleaved: layers overlap in this column
            transition_start = min(top_ys.min(), bot_ys.min())
            transition_end = max(top_ys.max(), bot_ys.max())
            ys[x] = (transition_start + transition_end) / 2

    return ys


def _adjust_boundaries(
    coarse_labels: np.ndarray,
    boundaries: list[np.ndarray],
    boundary_pairs: list[tuple[int, int]],
    blend_width: int = 5,
) -> np.ndarray:
    """Adjust only boundary-adjacent pixels, preserving coarse interior.

    For each fitted boundary between two layers, we identify the pixels in
    the coarse result that actually touch the adjacent layer (boundary pixels),
    dilate slightly to form an adjustment zone, and relabel pixels within
    that zone based on the smooth boundary position. Pixels far from the
    true boundary (interior of layers) are never touched.
    """
    h, w = coarse_labels.shape
    separator_mask = _separator_mask(coarse_labels)
    result = coarse_labels.copy()

    for boundary_y, (top_lbl, bot_lbl) in zip(boundaries, boundary_pairs):
        if len(boundary_y) != w:
            continue

        mask_top = coarse_labels == top_lbl
        mask_bot = coarse_labels == bot_lbl

        # Pixels of top_lbl that touch bot_lbl, and vice versa
        boundary_top = mask_top & ndimage.binary_dilation(mask_bot, iterations=1)
        boundary_bot = mask_bot & ndimage.binary_dilation(mask_top, iterations=1)

        # Dilate to create a narrow adjustment zone around the true boundary
        zone = ndimage.binary_dilation(boundary_top | boundary_bot, iterations=blend_width)

        for x in range(w):
            y_b = int(np.clip(round(boundary_y[x]), 0, h - 1))

            ys = np.where(zone[:, x])[0]
            if len(ys) == 0:
                continue

            for y in ys:
                if result[y, x] not in (top_lbl, bot_lbl):
                    continue
                if separator_mask[y, x]:
                    continue
                result[y, x] = top_lbl if y <= y_b else bot_lbl

    result[separator_mask] = 0
    return result


def _repartition_columns(
    coarse_labels: np.ndarray,
    spatial_order: list[int],
    boundaries: list[np.ndarray],
) -> np.ndarray:
    """Global column-wise repartitioning for severely fragmented images.

    When layer pairs are so fragmented they don't touch, local adjustment
    cannot reach the interior fragments. This function repartitions the
    ENTIRE image column-by-column using the fitted smooth boundaries:

    For each column x:
        y_0 = 0
        for each boundary i at position b_i[x]:
            assign pixels [y_{i-1}, b_i) to spatial_order[i]
        assign remaining pixels to spatial_order[-1]

    This treats each layer's fragments as an "archipelago" and redraws all
    maritime borders simultaneously. Foreign fragments (e.g. yellow pixels
    in the blue layer) are eliminated because every pixel is reassigned
    based on its vertical position relative to the smooth boundaries.

    Preserves: global layer ordering and identity.
    """
    h, w = coarse_labels.shape
    separator_mask = _separator_mask(coarse_labels)
    result = np.full_like(coarse_labels, -1)

    for x in range(w):
        boundary_ys = [int(np.clip(round(b[x]), 0, h - 1)) for b in boundaries]
        boundary_ys = sorted(boundary_ys)

        prev_y = 0
        for i, y_b in enumerate(boundary_ys):
            lbl = spatial_order[i]
            y_b = min(y_b, h)
            result[prev_y:y_b, x] = lbl
            prev_y = y_b
        # Last layer
        if len(spatial_order) > len(boundary_ys):
            result[prev_y:h, x] = spatial_order[len(boundary_ys)]

    result[separator_mask] = 0
    return result

__all__ = [
    "_adjust_boundaries",
    "_extract_boundary_dense",
    "_extract_boundary_points",
    "_repartition_columns",
]
