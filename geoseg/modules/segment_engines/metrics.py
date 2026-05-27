"""Objective segmentation metrics — NO physical priors.

These functions report FACTS about a segmentation, not JUDGMENTS.
They do NOT assume smooth=good, uniform=good, or connected=good.

Agent uses these for quick screening, then relies on VLM visual review
for the actual quality assessment.

All returns are raw counts/ratios, not 0-1 "scores".
"""

from __future__ import annotations

import numpy as np
from skimage import segmentation
from skimage.measure import label, regionprops
from skimage.filters import sobel


def _boundary_pixels(labels: np.ndarray) -> np.ndarray:
    """Return boolean mask of segmentation boundary pixels."""
    return segmentation.find_boundaries(labels, mode="thick")


def n_layers(labels: np.ndarray) -> int:
    """Count distinct layers (excluding background label 0)."""
    return len(set(labels.flatten()) - {0})


def boundary_alignment(labels: np.ndarray, img_rgb: np.ndarray) -> float:
    """Fraction of segmentation boundaries that overlap with image color edges.

    This is objective: it checks whether the white boundaries in the segmentation
    align with actual color transitions in the original image. It does NOT judge
    whether the boundaries are smooth or straight.

    Returns overlap ratio [0.0, 1.0]. Higher = more boundaries align with edges.
    """
    from skimage.color import rgb2gray

    gray = rgb2gray(img_rgb)
    edges = sobel(gray)
    edge_mask = np.abs(edges) > 0.05

    seg_boundaries = _boundary_pixels(labels)
    if not seg_boundaries.any():
        return 0.0

    aligned = (seg_boundaries & edge_mask).sum()
    total = seg_boundaries.sum()
    return float(aligned / total) if total > 0 else 0.0


def tiny_fragments(labels: np.ndarray, min_area_frac: float = 0.003) -> list[dict]:
    """List very small regions that may indicate over-segmentation.

    Returns list of {label_id, area, area_fraction} for regions smaller than
    min_area_frac of the image. These are WARNINGS, not errors — small regions
    may be legitimate thin layers or fault blocks.
    """
    h, w = labels.shape
    total_area = h * w
    min_area = max(30, int(total_area * min_area_frac))

    unique = sorted(set(labels.flatten()) - {0})
    fragments = []
    for lbl in unique:
        mask = labels == lbl
        cc = label(mask, connectivity=2)
        regions = regionprops(cc)
        for r in regions:
            if r.area < min_area:
                fragments.append({
                    "label_id": int(lbl),
                    "area": int(r.area),
                    "area_fraction": round(r.area / total_area, 5),
                })
    return fragments


def noise_detection(labels: np.ndarray, img_rgb: np.ndarray) -> dict:
    """Detect regions that may be non-layer elements (text, colorbar, axes).

    Uses heuristics: very high aspect ratio, touching image border, or
    located in typical caption/colorbar positions.

    Returns dict with suspect_regions list. These are WARNINGS for VLM review.
    """
    h, w = labels.shape
    unique = sorted(set(labels.flatten()) - {0})
    suspects = []

    for lbl in unique:
        mask = labels == lbl
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        bh, bw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        aspect = max(bh, bw) / max(min(bh, bw), 1)
        area = mask.sum()
        area_frac = area / (h * w)

        # Heuristic 1: extreme aspect ratio (text strips, colorbars)
        is_extreme_aspect = aspect > 8 and area_frac < 0.15

        # Heuristic 2: touches border and is narrow (axis labels)
        touches_left = xs.min() < w * 0.05
        touches_right = xs.max() > w * 0.95
        touches_top = ys.min() < h * 0.05
        touches_bottom = ys.max() > h * 0.95
        touches_border = touches_left or touches_right or touches_top or touches_bottom
        is_border_strip = touches_border and (aspect > 3 or area_frac < 0.05)

        if is_extreme_aspect or is_border_strip:
            suspects.append({
                "label_id": int(lbl),
                "bbox": [int(xs.min()), int(ys.min()), int(bw), int(bh)],
                "aspect_ratio": round(aspect, 2),
                "area_fraction": round(area_frac, 5),
                "reason": "extreme_aspect" if is_extreme_aspect else "border_strip",
            })

    return {
        "suspect_count": len(suspects),
        "suspect_regions": suspects,
    }


def region_stats(labels: np.ndarray) -> list[dict]:
    """Per-layer objective statistics.

    Returns list of {label_id, area, area_fraction, n_components, bbox}.
    n_components > 1 may indicate断层 or erosion — NOT necessarily an error.
    """
    h, w = labels.shape
    total = h * w
    unique = sorted(set(labels.flatten()) - {0})

    stats = []
    for lbl in unique:
        mask = labels == lbl
        if not mask.any():
            continue
        cc = label(mask, connectivity=2)
        n_components = int(cc.max())
        ys, xs = np.where(mask)
        stats.append({
            "label_id": int(lbl),
            "area": int(mask.sum()),
            "area_fraction": round(mask.sum() / total, 4),
            "n_components": n_components,
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
        })
    return stats


def compute_all(
    labels: np.ndarray,
    img_rgb: np.ndarray,
) -> dict:
    """Compute all objective metrics. Returns facts, not judgments.

    This dict is for agent quick-reference and VLM audit. It does NOT
    contain an 'overall_score' because quality assessment requires semantic
    understanding that only VLM visual review can provide.
    """
    ba = boundary_alignment(labels, img_rgb)
    fragments = tiny_fragments(labels)
    noise = noise_detection(labels, img_rgb)
    stats = region_stats(labels)
    n = n_layers(labels)

    return {
        "n_layers": n,
        "boundary_alignment": round(ba, 4),
        "tiny_fragments": fragments,
        "noise_warnings": noise,
        "region_stats": stats,
        # Summaries for quick glance
        "has_tiny_fragments": len(fragments) > 0,
        "has_noise_warnings": noise["suspect_count"] > 0,
        "total_fragment_area_fraction": round(
            sum(f["area_fraction"] for f in fragments), 5
        ) if fragments else 0.0,
    }
