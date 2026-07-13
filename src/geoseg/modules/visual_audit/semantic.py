"""Semantic fidelity metrics for visual audit.

These metrics attempt to measure whether the segmentation actually matches the
geological structure visible in the original image, rather than just checking
for artifacts like tiny islands or text leaks.
"""
from __future__ import annotations

import numpy as np
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.measure import label, regionprops

from geoseg.modules.visual_audit.color_residual import compute_color_residual_audit


def _boundary_mask(labels: np.ndarray) -> np.ndarray:
    """Boolean mask of label boundaries."""
    from skimage import segmentation

    return segmentation.find_boundaries(labels, mode="thick")


def compute_per_label_boundary_alignment(
    labels: np.ndarray,
    image_rgb: np.ndarray,
) -> dict[int, float]:
    """For each label, fraction of its boundary that aligns with image edges.

    High alignment means the segmentation boundary follows real color
    transitions. Low alignment suggests the boundary is arbitrary.
    """
    gray = rgb2gray(image_rgb)
    edges = np.abs(sobel(gray))
    edge_mask = edges > np.percentile(edges, 75)

    boundaries = _boundary_mask(labels)
    result: dict[int, float] = {}
    for lbl in sorted(set(labels.flatten()) - {0}):
        label_boundary = boundaries & (labels == lbl)
        if not label_boundary.any():
            result[lbl] = 0.0
            continue
        aligned = (label_boundary & edge_mask).sum()
        total = label_boundary.sum()
        result[lbl] = round(float(aligned / total), 4)
    return result


def compute_label_color_consistency(
    labels: np.ndarray,
    image_rgb: np.ndarray,
) -> dict[int, dict]:
    """Per-label color statistics.

    Returns mean RGB and coefficient of variation (std/mean) for each label.
    High variation within a label may indicate it covers multiple geological
    units.
    """
    result: dict[int, dict] = {}
    for lbl in sorted(set(labels.flatten()) - {0}):
        pixels = image_rgb[labels == lbl]
        mean = pixels.mean(axis=0)
        std = pixels.std(axis=0)
        cv = np.where(mean > 0, std / mean, 0.0)
        result[lbl] = {
            "mean_rgb": [round(float(v), 2) for v in mean],
            "color_cv": [round(float(v), 4) for v in cv],
            "area": int(pixels.shape[0]),
        }
    return result


def compute_layer_order_score(labels: np.ndarray) -> dict:
    """Check whether labels are ordered from top (shallow) to bottom (deep).

    Returns a mapping of label -> median y, and a score indicating whether the
    labels are monotonically ordered (no interleaving).
    """
    medians: dict[int, float] = {}
    for lbl in sorted(set(labels.flatten()) - {0}):
        ys = np.where(labels == lbl)[0]
        medians[lbl] = float(np.median(ys))

    labels_sorted = sorted(medians, key=medians.get)
    is_monotonic = all(
        medians[labels_sorted[i]] <= medians[labels_sorted[i + 1]]
        for i in range(len(labels_sorted) - 1)
    )

    return {
        "median_y": {int(k): round(v, 2) for k, v in medians.items()},
        "ordered_labels": [int(l) for l in labels_sorted],
        "monotonic": is_monotonic,
    }


def compute_plume_fidelity(
    labels: np.ndarray,
    panel_rgb: np.ndarray,
    gt_mask: np.ndarray | None = None,
) -> dict:
    """Evaluate how well the central plume/uplift is captured.

    If no GT mask is provided, uses a heuristic: the largest label in the
    central band that is lighter than surroundings. Reports IoU and whether
    the plume is split across multiple labels.
    """
    h, w = labels.shape
    center_band = labels[:, int(w * 0.35) : int(w * 0.65)]
    unique, counts = np.unique(center_band, return_counts=True)
    candidates = [(int(lbl), int(c)) for lbl, c in zip(unique, counts) if lbl != 0]
    if not candidates:
        return {"plume_label": None, "iou": 0.0, "split": True}

    candidates.sort(key=lambda x: x[1], reverse=True)
    plume_label = candidates[0][0]

    plume_mask = labels == plume_label
    cc = label(plume_mask, connectivity=2)
    n_components = int(cc.max())

    if gt_mask is not None and gt_mask.any():
        intersection = (plume_mask & gt_mask).sum()
        union = (plume_mask | gt_mask).sum()
        iou = float(intersection / union) if union > 0 else 0.0
    else:
        # No GT: report estimated coverage but do not claim IoU.
        center_area = center_band.size
        plume_frac = float(counts[unique.tolist().index(plume_label)] / center_area)
        iou = 0.0

    return {
        "plume_label": plume_label,
        "iou": round(iou, 4),
        "split": n_components > 1,
        "n_components": n_components,
    }


def compute_semantic_fidelity(
    labels: np.ndarray,
    panel_rgb: np.ndarray,
    gt_mask: np.ndarray | None = None,
) -> dict:
    """Aggregate semantic fidelity metrics.

    These metrics are meant to support agent judgment, not to replace it.
    """
    boundary_alignment = compute_per_label_boundary_alignment(labels, panel_rgb)
    color_consistency = compute_label_color_consistency(labels, panel_rgb)
    layer_order = compute_layer_order_score(labels)
    plume = compute_plume_fidelity(labels, panel_rgb, gt_mask)
    color_residual = compute_color_residual_audit(labels, panel_rgb)

    avg_boundary_alignment = round(
        sum(boundary_alignment.values()) / max(1, len(boundary_alignment)), 4
    )

    return {
        "avg_boundary_alignment": avg_boundary_alignment,
        "per_label_boundary_alignment": boundary_alignment,
        "color_consistency": color_consistency,
        "color_residual": color_residual,
        "layer_order": layer_order,
        "plume_fidelity": plume,
    }


def _find_manual_gt_mask(labels_path: str | None) -> np.ndarray | None:
    """Look for a manual GT plume mask in a sibling visuals directory."""
    from pathlib import Path

    if not labels_path:
        return None
    p = Path(labels_path)
    candidate = p.parent.parent / "visuals" / "manual_gt_mask.jpg"
    if candidate.exists():
        from PIL import Image

        img = np.array(Image.open(candidate).convert("L"))
        return img > 0
    return None
