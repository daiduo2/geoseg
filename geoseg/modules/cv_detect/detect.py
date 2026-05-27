"""Panel candidate detection on extracted PDF images.

Uses connected-component analysis with area filtering to find panel bbox candidates.
Designed for geophysics cross-section figures with multiple panels arranged horizontally.
"""

import numpy as np
from scipy import ndimage


def find_panel_candidates(
    img: np.ndarray,
    white_threshold: int = 220,
    min_area_ratio: float = 0.02,
    max_area_ratio: float = 0.45,
    gap_ratio: float = 0.2,
    min_gap_width: int = 10,
) -> list:
    """Find panel candidate bounding boxes via vertical-gap segmentation.

    Algorithm:
    1. Binarize: non-white pixels as foreground.
    2. Vertical projection: find columns with few foreground pixels (gaps).
    3. Merge adjacent small gaps.
    4. Extract regions between gaps.
    5. Filter regions by width and area.
    6. Return bbox candidates sorted left-to-right.

    Args:
        img: RGB image array (H, W, 3).
        white_threshold: Pixels with mean >= this are considered background.
        min_area_ratio: Minimum panel area as fraction of image area.
        max_area_ratio: Maximum panel area as fraction of image area.
        gap_ratio: Fraction of image height below which a column is a gap.
        min_gap_width: Minimum gap width to consider (px).

    Returns:
        List of candidate dicts:
            {"id": int, "bbox": [x, y, w, h], "area": int, "confidence": float}
    """
    h, w = img.shape[:2]
    total_area = h * w
    min_area = int(total_area * min_area_ratio)
    max_area = int(total_area * max_area_ratio)

    # Binarize: foreground = non-white
    gray = img.mean(axis=2)
    foreground = gray < white_threshold

    # Vertical projection
    vproj = foreground.sum(axis=0)
    gap_thr = h * gap_ratio
    is_gap = vproj < gap_thr

    # Find gap regions
    gaps = []
    in_g = False
    start = 0
    for x in range(w):
        if is_gap[x] and not in_g:
            start = x
            in_g = True
        elif not is_gap[x] and in_g:
            gaps.append((start, x))
            in_g = False
    if in_g:
        gaps.append((start, w))

    # Filter wide gaps
    wide_gaps = [(g1, g2) for g1, g2 in gaps if g2 - g1 >= min_gap_width]

    # Find regions between wide gaps
    regions = []
    prev = 0
    for g1, g2 in wide_gaps:
        if g1 - prev >= 20:
            regions.append((prev, g1))
        prev = g2
    if w - prev >= 20:
        regions.append((prev, w))

    candidates = []
    for x1, x2 in regions:
        rw = x2 - x1
        if rw < 50:  # too narrow
            continue

        # Compute vertical extent of foreground in this region
        col_foreground = foreground[:, x1:x2]
        hproj = col_foreground.sum(axis=1)
        # Find top and bottom of content
        content_rows = np.where(hproj > rw * 0.1)[0]
        if len(content_rows) == 0:
            continue
        y1 = int(content_rows.min())
        y2 = int(content_rows.max()) + 1
        rh = y2 - y1
        area = int(col_foreground[y1:y2, :].sum())

        if area < min_area or area > max_area:
            continue

        candidates.append({
            "bbox": [x1, y1, rw, rh],
            "area": area,
            "confidence": min(1.0, area / min_area),
        })

    # Sort left-to-right and reassign IDs
    candidates.sort(key=lambda c: c["bbox"][0])
    for i, c in enumerate(candidates):
        c["id"] = i

    return candidates
