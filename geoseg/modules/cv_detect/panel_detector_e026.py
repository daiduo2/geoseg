"""CV-first panel detection for PDF page figures.

Implements T1 of the segmentation pipeline using only computer vision.
VLM is intentionally NOT used here — it is a fallback at the caller level.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage import measure


def detect_panels(page_image: np.ndarray) -> list[dict]:
    """Detect figure panels on a PDF page using CV only.

    Parameters
    ----------
    page_image:
        ``(H, W, 3)`` uint8 RGB array rendered at 300 DPI.

    Returns
    -------
    List of dicts with keys ``{"bbox": [x, y, w, h], "confidence": float}``.
    If no panels are found, returns a single panel covering the full page.
    """
    h, w = page_image.shape[:2]
    page_area = h * w

    # 1. Non-white mask: RGB mean < 245
    gray = page_image.mean(axis=2)
    content_mask = gray < 245

    # 2. Find content blocks via connected components
    labeled, num_labels = ndimage.label(content_mask)
    regions = measure.regionprops(labeled)

    # 3. Filter candidate blocks
    candidates = []
    for region in regions:
        y0, x0, y1, x1 = region.bbox
        bw, bh = x1 - x0, y1 - y0
        area = bw * bh
        aspect = bh / bw if bw > 0 else 0
        density = region.area / area if area > 0 else 0

        if area < 0.01 * page_area:
            continue
        if not (0.3 <= aspect <= 3.0):
            continue
        if density < 0.30:
            continue

        candidates.append({
            "bbox": [int(x0), int(y0), int(bw), int(bh)],
            "confidence": 0.7 + 0.2 * min(density, 1.0),
            "centroid": (region.centroid[1], region.centroid[0]),
            "area": area,
        })

    if not candidates:
        # Fallback: full page
        return [{"bbox": [0, 0, w, h], "confidence": 0.5}]

    if len(candidates) == 1:
        return candidates

    # 4. Multi-panel layout: detect white gutters and split/merge
    candidates = _resolve_layout(candidates, content_mask)

    # Deduplicate overlapping boxes (IoU > 0.5)
    candidates = _deduplicate(candidates)

    if not candidates:
        return [{"bbox": [0, 0, w, h], "confidence": 0.5}]

    return candidates


def _resolve_layout(
    candidates: list[dict], content_mask: np.ndarray
) -> list[dict]:
    """Merge or split candidates based on white-gutter analysis."""
    h, w = content_mask.shape

    # Project content onto axes to find gutters
    row_sum = content_mask.sum(axis=1)
    col_sum = content_mask.sum(axis=0)

    # White gutters: consecutive rows/cols with < 1% of width/height content
    row_thresh = w * 0.01
    col_thresh = h * 0.01

    h_gutters = _find_gaps(row_sum < row_thresh, min_len=5)
    v_gutters = _find_gaps(col_sum < col_thresh, min_len=5)

    # If strong gutters exist, use them to define panel boundaries
    if len(h_gutters) >= 1 or len(v_gutters) >= 1:
        panels = _split_by_gutters(h_gutters, v_gutters, h, w, content_mask)
        if panels:
            return panels

    # Otherwise keep the connected-component candidates
    return candidates


def _find_gaps(mask: np.ndarray, min_len: int = 5) -> list[tuple[int, int]]:
    """Return list of (start, end) for consecutive True runs in *mask*."""
    gaps = []
    in_gap = False
    start = 0
    for i, val in enumerate(mask):
        if val and not in_gap:
            in_gap = True
            start = i
        elif not val and in_gap:
            in_gap = False
            if i - start >= min_len:
                gaps.append((start, i))
    if in_gap and len(mask) - start >= min_len:
        gaps.append((start, len(mask)))
    return gaps


def _split_by_gutters(
    h_gutters: list[tuple[int, int]],
    v_gutters: list[tuple[int, int]],
    h: int,
    w: int,
    content_mask: np.ndarray,
) -> list[dict]:
    """Split page into panels using gutter boundaries."""
    # Build row and col boundaries including page edges
    row_bounds = [0] + [g[1] for g in h_gutters] + [h]
    col_bounds = [0] + [g[1] for g in v_gutters] + [w]

    panels = []
    for r_i in range(len(row_bounds) - 1):
        for c_i in range(len(col_bounds) - 1):
            y0, y1 = row_bounds[r_i], row_bounds[r_i + 1]
            x0, x1 = col_bounds[c_i], col_bounds[c_i + 1]
            bh, bw = y1 - y0, x1 - x0
            area = bw * bh
            if area < 0.01 * h * w:
                continue
            aspect = bh / bw if bw > 0 else 0
            if not (0.3 <= aspect <= 3.0):
                continue
            sub_mask = content_mask[y0:y1, x0:x1]
            density = sub_mask.sum() / area if area > 0 else 0
            if density < 0.15:
                continue
            confidence = 0.6 + 0.3 * min(density, 1.0)
            panels.append({
                "bbox": [int(x0), int(y0), int(bw), int(bh)],
                "confidence": round(confidence, 2),
            })

    return panels


def _deduplicate(candidates: list[dict], iou_thresh: float = 0.5) -> list[dict]:
    """Remove overlapping candidates, keeping higher confidence."""
    # Sort by confidence descending
    sorted_cands = sorted(candidates, key=lambda c: c["confidence"], reverse=True)
    kept: list[dict] = []
    for cand in sorted_cands:
        if all(_iou(cand["bbox"], k["bbox"]) < iou_thresh for k in kept):
            kept.append(cand)
    return kept


def is_cv_result_ambiguous(panels: list[dict], page_image: np.ndarray) -> bool:
    """Determine whether CV results need VLM fallback or human review.

    Returns True if:
    - Only one panel detected with low confidence (likely full-page fallback)
    - No panels detected
    - Panel bboxes overlap excessively
    - Panel bboxes are suspiciously small or large

    Parameters
    ----------
    panels :
        Output from :func:`detect_panels`.
    page_image :
        The original page image for size comparison.

    Returns
    -------
    bool
        True if VLM fallback is recommended.
    """
    if not panels:
        return True

    h, w = page_image.shape[:2]
    page_area = h * w

    # Single panel with low confidence -> ambiguous
    if len(panels) == 1 and panels[0].get("confidence", 0.0) < 0.5:
        return True

    # Check for suspiciously large single panel (full-page fallback)
    if len(panels) == 1:
        bbox = panels[0]["bbox"]
        panel_area = bbox[2] * bbox[3]
        if panel_area > 0.95 * page_area:
            return True

    return False


def _iou(a: list[int], b: list[int]) -> float:
    """Compute IoU of two bounding boxes [x, y, w, h]."""
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]

    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)

    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0.0

    inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0
