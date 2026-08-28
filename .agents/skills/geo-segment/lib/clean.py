"""Noise mask construction — simplified per advisor guidance.

Guidance (2026-05-14): do NOT aggressively strip text, lines, or marks from
the panel before segmentation.  K-means + the perimeter²/area post-filter in
segment.py handles thin 1-D elements naturally.  This module now only masks
out large non-data artefacts (scalebars, thick frames, big inset maps) that
could dominate a cluster centroid.

Public API:
    compose_clean_mask(panel_rgb, vlm_noise, panel_origin=(0,0)) -> mask
"""

from __future__ import annotations

import numpy as np
from skimage.color import rgb2gray
from skimage.measure import label, regionprops


# Element kinds that are large enough to distort K-means centroids
_SIGNIFICANT_KINDS = {"colorbar", "scalebar", "inset_map", "frame", "thick_overlay", "legend"}


def _bbox_from_element(elem: dict, origin: tuple[int, int]) -> tuple[int, int, int, int]:
    """Convert a VLM noise-element dict into a panel-local bbox."""
    ox, oy = origin
    if "approx_bbox" in elem:
        bx = elem["approx_bbox"]
        return bx[0] - ox, bx[1] - oy, bx[2] - ox, bx[3] - oy
    x = int(elem["x"]) - ox
    y = int(elem["y"]) - oy
    s = int(elem.get("size_px", 20))
    return x - s // 2, y - s // 2, x + s // 2, y + s // 2


def _is_significant(elem: dict) -> bool:
    """Return True for elements large enough to bias K-means."""
    kind = elem.get("kind", "").lower()
    if kind in _SIGNIFICANT_KINDS:
        return True
    size = elem.get("size_px", 20)
    return size >= 150  # large text blocks, big arrows, etc.


def compose_clean_mask(
    panel_rgb: np.ndarray,
    vlm_noise: list[dict],
    panel_origin: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Build a minimal mask covering only large non-data artefacts.

    Thin text, contour lines, and small marks are intentionally left UNMASKED
    so that the shape filter in segment.py can merge them into the correct
    colour zone.
    """
    h, w = panel_rgb.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    max_mask_area = h * w * 0.5  # reject single elements that cover >50% of panel
    for elem in vlm_noise:
        if not _is_significant(elem):
            continue
        x1, y1, x2, y2 = _bbox_from_element(elem, panel_origin)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        area = (x2 - x1) * (y2 - y1)
        if x2 > x1 and y2 > y1 and area <= max_mask_area:
            mask[y1:y2, x1:x2] = True
    return mask


__all__ = ["compose_clean_mask"]
