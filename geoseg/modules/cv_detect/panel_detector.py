"""Improved panel detection supporting horizontal, vertical, and grid layouts.

Replaces the simple vertical-gap heuristic with a layout-analysis approach:
1. Foreground extraction + noise filtering
2. Connected-component analysis
3. Layout clustering (rows/cols) to infer grid structure
4. Rectangular panel bbox extraction

Test scenario:
    >>> import numpy as np
    >>> img = np.full((300, 600, 3), 255, dtype=np.uint8)
    >>> img[20:120, 20:180] = 128  # panel 1
    >>> img[20:120, 220:380] = 100  # panel 2
    >>> img[160:280, 20:180] = 80   # panel 3
    >>> result = detect_panels(img)
    >>> assert len(result) == 3
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.color import rgb2gray
from skimage.measure import label, regionprops

from geoseg.pipeline_interfaces import PanelInput


def _detect_background_type(img_gray: np.ndarray, sample_size: int = 20) -> str:
    """Detect whether figure has black or white background by sampling corners.

    Returns "black" or "white".
    """
    h, w = img_gray.shape
    sz = min(sample_size, h // 4, w // 4)
    if sz < 5:
        # Too small to sample; default to white background (majority case)
        return "white"
    corners = [
        img_gray[:sz, :sz].mean(),
        img_gray[:sz, -sz:].mean(),
        img_gray[-sz:, :sz].mean(),
        img_gray[-sz:, -sz:].mean(),
    ]
    avg_corner = float(np.mean(corners))
    # Black background: corners are dark (< 0.25)
    # White background: corners are light (> 0.75)
    # Ambiguous middle: check which extreme the corners are closer to
    if avg_corner < 0.35:
        return "black"
    if avg_corner > 0.65:
        return "white"
    # Ambiguous: compare distance to extremes
    return "black" if avg_corner < 0.5 else "white"


def _binarize(img_gray: np.ndarray, white_threshold: float = 0.94) -> tuple[np.ndarray, str]:
    """Binary mask: True = foreground. Returns (mask, background_type)."""
    bg_type = _detect_background_type(img_gray)
    if bg_type == "black":
        # Black background: foreground is light
        return img_gray > (1.0 - white_threshold), bg_type
    # White background: foreground is dark
    return img_gray < white_threshold, bg_type


def _filter_noise_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove tiny connected components from binary mask."""
    labeled, num = label(mask, connectivity=2, return_num=True)
    if num == 0:
        return mask
    regions = regionprops(labeled)
    out = mask.copy()
    for r in regions:
        if r.area < min_area:
            out[labeled == r.label] = False
    return out


def _merge_overlapping_boxes(
    boxes: list[tuple[int, int, int, int]],
    overlap_threshold: float = 0.3,
) -> list[tuple[int, int, int, int]]:
    """Merge boxes with significant IoU overlap."""
    if not boxes:
        return []

    # Sort by area descending
    boxes_sorted = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    merged: list[tuple[int, int, int, int]] = []

    for bx, by, bw, bh in boxes_sorted:
        x1, y1, x2, y2 = bx, by, bx + bw, by + bh
        area = bw * bh
        to_merge = []
        for i, (mx, my, mw, mh) in enumerate(merged):
            mx1, my1, mx2, my2 = mx, my, mx + mw, my + mh
            inter_x1 = max(x1, mx1)
            inter_y1 = max(y1, my1)
            inter_x2 = min(x2, mx2)
            inter_y2 = min(y2, my2)
            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                continue
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            union_area = area + mw * mh - inter_area
            iou = inter_area / union_area if union_area > 0 else 0
            if iou > overlap_threshold:
                to_merge.append(i)

        if to_merge:
            # Merge all overlapping boxes
            all_x = [x1, x2]
            all_y = [y1, y2]
            for idx in sorted(to_merge, reverse=True):
                mx, my, mw, mh = merged.pop(idx)
                all_x.extend([mx, mx + mw])
                all_y.extend([my, my + mh])
            merged.append((min(all_x), min(all_y), max(all_x) - min(all_x), max(all_y) - min(all_y)))
        else:
            merged.append((bx, by, bw, bh))

    return merged


def _cluster_layout(
    boxes: list[tuple[int, int, int, int]],
    img_w: int,
    img_h: int,
) -> list[tuple[int, int, int, int]]:
    """Analyze box layout and merge boxes that belong to the same panel.

    Uses x/y coordinate clustering to detect grid arrangements.
    Only merges boxes that are spatially adjacent (overlap in x or y).
    """
    if len(boxes) <= 1:
        return boxes

    centers = np.array([(bx + bw / 2, by + bh / 2) for bx, by, bw, bh in boxes])
    areas = np.array([bw * bh for _, _, bw, bh in boxes])
    median_area = float(np.median(areas))

    # Heuristic: if boxes are roughly aligned in rows/cols, it's a grid
    x_hist, x_edges = np.histogram(centers[:, 0], bins=max(2, len(boxes) // 2))
    y_hist, y_edges = np.histogram(centers[:, 1], bins=max(2, len(boxes) // 2))

    n_cols = max(1, int((x_hist > 0).sum()))
    n_rows = max(1, int((y_hist > 0).sum()))

    # Only run grid merge if layout looks like a reasonable grid
    if n_cols * n_rows >= len(boxes) * 0.5 and n_cols * n_rows <= len(boxes) * 2:
        x_centers = [(x_edges[i] + x_edges[i + 1]) / 2 for i in range(len(x_edges) - 1) if x_hist[i] > 0]
        y_centers = [(y_edges[i] + y_edges[i + 1]) / 2 for i in range(len(y_edges) - 1) if y_hist[i] > 0]

        if not x_centers:
            x_centers = [img_w / 2]
        if not y_centers:
            y_centers = [img_h / 2]

        groups: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
        for box in boxes:
            cx = box[0] + box[2] / 2
            cy = box[1] + box[3] / 2
            col = min(range(len(x_centers)), key=lambda i: abs(cx - x_centers[i]))
            row = min(range(len(y_centers)), key=lambda i: abs(cy - y_centers[i]))
            groups.setdefault((row, col), []).append(box)

        merged = []
        for group_boxes in groups.values():
            if len(group_boxes) == 1:
                merged.append(group_boxes[0])
                continue
            # Sort by area descending; start with largest, only merge adjacent boxes
            group_boxes = sorted(group_boxes, key=lambda b: b[2] * b[3], reverse=True)
            seed = list(group_boxes[0])
            for bx, by, bw, bh in group_boxes[1:]:
                box_area = bw * bh
                # Skip tiny boxes that are < 1/5 of median (likely captions/labels)
                if box_area < median_area * 0.2:
                    merged.append((bx, by, bw, bh))
                    continue
                # Only merge if boxes are actually adjacent (small gap + overlap)
                sx1, sy1, sx2, sy2 = seed[0], seed[1], seed[0] + seed[2], seed[1] + seed[3]
                bx1, by1, bx2, by2 = bx, by, bx + bw, by + bh
                x_overlap = max(0, min(sx2, bx2) - max(sx1, bx1))
                y_overlap = max(0, min(sy2, by2) - max(sy1, by1))
                x_gap = max(0, max(sx1, bx1) - min(sx2, bx2))
                y_gap = max(0, max(sy1, by1) - min(sy2, by2))
                # Merge only if one dimension overlaps significantly AND the other is close
                x_ok = x_overlap >= min(bw, seed[2]) * 0.5
                y_ok = y_overlap >= min(bh, seed[3]) * 0.5
                x_close = x_gap < min(seed[2], bw) * 0.3
                y_close = y_gap < min(seed[3], bh) * 0.3
                if (x_ok and y_close) or (y_ok and x_close):
                    seed[0] = min(sx1, bx1)
                    seed[1] = min(sy1, by1)
                    seed[2] = max(sx2, bx2) - seed[0]
                    seed[3] = max(sy2, by2) - seed[1]
                else:
                    merged.append((bx, by, bw, bh))
            merged.append(tuple(seed))
        return merged

    return boxes


def detect_panels(
    img_rgb: np.ndarray,
    white_threshold: float = 0.94,
    min_area_ratio: float = 0.003,
    max_area_ratio: float = 0.6,
    gap_ratio: float = 0.15,
) -> list[PanelInput]:
    """Detect panel candidates in a figure image.

    Supports horizontal, vertical, and grid layouts.
    Implements the `PanelDetector` Protocol.

    Args:
        img_rgb: RGB uint8 array.
        white_threshold: Pixels with gray value >= this fraction of max are background.
        min_area_ratio: Minimum panel area as fraction of image area.
        max_area_ratio: Maximum panel area as fraction of image area.
        gap_ratio: Fraction of image width/height for gap detection fallback.

    Returns:
        List of PanelInput dicts:
            {"id": int, "bbox": (x, y, w, h), "source": "cv_detect", "confidence": float}
    """
    h, w = img_rgb.shape[:2]
    total_area = h * w
    min_area = max(100, int(total_area * min_area_ratio))
    # Allow single-panel figures to occupy most of the image
    max_area = int(total_area * 0.95)

    gray = rgb2gray(img_rgb)
    mask, bg_type = _binarize(gray, white_threshold)

    # Remove tiny noise
    mask = _filter_noise_components(mask, min_area // 5)

    # If the whole image is one big foreground blob, try to split it
    labeled, num = label(mask, connectivity=2, return_num=True)
    regions = regionprops(labeled)

    boxes = []
    for r in regions:
        if r.area < min_area or r.area > max_area:
            continue
        y1, x1, y2, x2 = r.bbox
        bw, bh = x2 - x1, y2 - y1
        if bw < 30 or bh < 30:
            continue
        boxes.append((x1, y1, bw, bh))

    # If no components found, fallback to gap-based splitting
    if not boxes:
        boxes = _gap_split(gray, gap_ratio, min_area, max_area, w, h, bg_type)

    # Merge overlapping boxes
    boxes = _merge_overlapping_boxes(boxes, overlap_threshold=0.3)

    # Layout analysis: if many small boxes, try to merge into grid cells
    if len(boxes) >= 3:
        boxes = _cluster_layout(boxes, w, h)
        boxes = _merge_overlapping_boxes(boxes, overlap_threshold=0.2)

    # Final filtering
    max_box_area = max((bw * bh for _, _, bw, bh in boxes), default=1)
    candidates = []
    for x, y, bw, bh in boxes:
        area = bw * bh
        if area < min_area or area > max_area:
            continue
        if bw < 30 or bh < 30:
            continue
        # Compute actual foreground area within bbox
        fg_area = int(mask[y : y + bh, x : x + bw].sum())
        if fg_area < min_area * 0.3:
            continue
        aspect = max(bw, bh) / min(bw, bh)

        # Filter out small caption/legend boxes that are much smaller than the largest panel
        # and roughly square (typical for legends)
        is_small_square = (
            max(bw, bh) < 150
            and 0.6 <= aspect <= 1.7
            and area < max_box_area * 0.15
        )
        if is_small_square:
            continue

        # Filter out caption/text strips (horizontal or vertical text bars / colorbars)
        is_caption_strip = min(bw, bh) < 110 and aspect > 3.5
        if is_caption_strip:
            continue

        # Filter out tiny letter icons / labels
        is_tiny_icon = max(bw, bh) < 100 and area < max_box_area * 0.05
        if is_tiny_icon:
            continue

        # Filter out small thumbnails / mini-panels in crowded multi-panel figures
        if len(boxes) >= 6:
            is_small_thumbnail = (
                max(bw, bh) < 200
                and 0.5 <= aspect <= 2.0
                and area < max_box_area * 0.6
            )
            if is_small_thumbnail:
                continue

        # Filter out extreme aspect ratio strips (colorbars, axis labels)
        if aspect > 8:
            continue
        candidates.append({
            "bbox": (x, y, bw, bh),
            "source": "cv_detect",
            "confidence": min(1.0, fg_area / min_area),
        })

    # Sort left-to-right, top-to-bottom
    candidates.sort(key=lambda c: (c["bbox"][1], c["bbox"][0]))
    for i, c in enumerate(candidates):
        c["id"] = i

    return candidates


def _extract_gaps(proj: np.ndarray, gap_thr: float, min_gap: int = 5) -> list[tuple[int, int]]:
    """Extract gap runs from a 1-D projection."""
    is_gap = proj < gap_thr
    gaps = []
    in_g = False
    start = 0
    for i in range(len(proj)):
        if is_gap[i] and not in_g:
            start = i
            in_g = True
        elif not is_gap[i] and in_g:
            gaps.append((start, i))
            in_g = False
    if in_g:
        gaps.append((start, len(proj)))
    return [(g1, g2) for g1, g2 in gaps if g2 - g1 >= min_gap]


def _gap_split(
    gray: np.ndarray,
    gap_ratio: float,
    min_area: int,
    max_area: int,
    w: int,
    h: int,
    bg_type: str = "white",
) -> list[tuple[int, int, int, int]]:
    """Fallback: split by vertical/horizontal gaps."""
    if bg_type == "black":
        mask = gray > 0.06
    else:
        mask = gray < 0.94

    # Vertical projection split
    vproj = mask.sum(axis=0)
    v_gaps = _extract_gaps(vproj, h * gap_ratio)

    v_regions = []
    prev = 0
    for g1, g2 in v_gaps:
        if g1 - prev >= 10:
            v_regions.append((prev, g1))
        prev = g2
    if w - prev >= 10:
        v_regions.append((prev, w))

    # Horizontal projection split
    hproj = mask.sum(axis=1)
    h_gaps = _extract_gaps(hproj, w * gap_ratio)

    h_regions = []
    prev = 0
    for g1, g2 in h_gaps:
        if g1 - prev >= 10:
            h_regions.append((prev, g1))
        prev = g2
    if h - prev >= 10:
        h_regions.append((prev, h))

    boxes = []
    # If both directions split, use grid intersection
    if len(v_regions) > 1 and len(h_regions) > 1:
        for x1, x2 in v_regions:
            for y1, y2 in h_regions:
                rw, rh = x2 - x1, y2 - y1
                if rw < 50 or rh < 30:
                    continue
                area = int(mask[y1:y2, x1:x2].sum())
                if min_area <= area <= max_area:
                    boxes.append((x1, y1, rw, rh))
    elif len(v_regions) > 1:
        for x1, x2 in v_regions:
            rw = x2 - x1
            if rw < 50:
                continue
            col_fg = mask[:, x1:x2]
            hproj_col = col_fg.sum(axis=1)
            content_rows = np.where(hproj_col > rw * 0.05)[0]
            if len(content_rows) == 0:
                continue
            y1 = int(content_rows.min())
            y2 = int(content_rows.max()) + 1
            rh = y2 - y1
            area = int(col_fg[y1:y2, :].sum())
            if min_area <= area <= max_area:
                boxes.append((x1, y1, rw, rh))
    elif len(h_regions) > 1:
        for y1, y2 in h_regions:
            rh = y2 - y1
            if rh < 30:
                continue
            row_fg = mask[y1:y2, :]
            vproj_row = row_fg.sum(axis=0)
            content_cols = np.where(vproj_row > rh * 0.05)[0]
            if len(content_cols) == 0:
                continue
            x1 = int(content_cols.min())
            x2 = int(content_cols.max()) + 1
            rw = x2 - x1
            area = int(row_fg[:, x1:x2].sum())
            if min_area <= area <= max_area:
                boxes.append((x1, y1, rw, rh))

    return boxes
