"""Quantitative evaluation of segmentation against ground truth.

The geoseg project deliberately ships only UNSUPERVISED diagnostics
(`boundary_alignment`, `tiny_fragments`, ...) because most real-world
panels don't have ground truth. The synthetic benchmark is the one
place where we can compute supervised metrics:

  - pixel accuracy
  - mean Intersection-over-Union (mIoU) with Hungarian matching to
    handle label permutation (algo labels 3 may correspond to gt 7).
  - Boundary F1 (bfscore): precision/recall of boundary pixels.
  - Topology error: |n_connected_components(gt) - n_cc(pred)|
  - Fragment count from the existing diagnostics
  - Text-region "false positive" rate: fraction of text-mask pixels
    that got any non-background label.

All functions return plain Python data; no numpy structured arrays.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage import segmentation
from skimage.measure import label, regionprops

from geoseg.modules.segment_engines.diagnostics.metrics import (
    boundary_alignment,
    tiny_fragments,
)


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b != 0 else 0.0


def pixel_accuracy(pred: np.ndarray, gt: np.ndarray, ignore_mask: np.ndarray | None = None) -> float:
    """Fraction of pixels where pred == gt, optionally ignoring text regions."""
    if ignore_mask is not None:
        keep = ~ignore_mask
        if not keep.any():
            return 0.0
        return _safe_div((pred[keep] == gt[keep]).sum(), keep.sum())
    return _safe_div((pred == gt).sum(), pred.size)


def hungarian_match_iou(
    pred: np.ndarray,
    gt: np.ndarray,
    ignore_mask: np.ndarray | None = None,
) -> tuple[dict[int, int], dict[int, float]]:
    """Find the best 1-to-1 matching between predicted and gt labels.

    Returns:
      mapping  : {pred_label: gt_label} for matched pairs
      iou_each : {pred_label: iou value}

    Background label 0 is matched separately as a free pass — we just
    require the algorithm to leave text regions background.

    Why Hungarian? K-means and similar algorithms can output any
    permutation of the gt labels. Naive IoU averaging treats this as
    a failure even when the segmentation is geometrically perfect.
    """
    pred_labels = sorted(set(pred.flatten().tolist()))
    gt_labels = sorted(set(gt.flatten().tolist()))

    if not pred_labels or not gt_labels:
        return {}, {}

    keep = ~ignore_mask if ignore_mask is not None else np.ones_like(pred, dtype=bool)

    # Build cost matrix: rows = pred, cols = gt, cost = -IoU
    cost = np.zeros((len(pred_labels), len(gt_labels)), dtype=np.float64)
    for i, pl in enumerate(pred_labels):
        p_mask = (pred == pl) & keep
        p_size = int(p_mask.sum())
        if p_size == 0:
            continue
        for j, gl in enumerate(gt_labels):
            inter = int(((pred == pl) & (gt == gl) & keep).sum())
            union = p_size + int(((gt == gl) & keep).sum()) - inter
            iou_val = _safe_div(inter, union)
            cost[i, j] = -iou_val

    row_ind, col_ind = linear_sum_assignment(cost)

    mapping: dict[int, int] = {}
    iou_each: dict[int, float] = {}
    for i, j in zip(row_ind, col_ind):
        # Only accept the match if IoU > 0; otherwise pred-label is spurious.
        if cost[i, j] < 0:
            mapping[pred_labels[i]] = gt_labels[j]
            iou_each[pred_labels[i]] = float(-cost[i, j])
    return mapping, iou_each


def miou(
    pred: np.ndarray,
    gt: np.ndarray,
    ignore_mask: np.ndarray | None = None,
) -> float:
    """Mean IoU over Hungarian-matched label pairs (non-matched labels excluded)."""
    _, iou_each = hungarian_match_iou(pred, gt, ignore_mask)
    if not iou_each:
        return 0.0
    return float(sum(iou_each.values()) / len(iou_each))


def boundary_f1(
    pred: np.ndarray,
    gt: np.ndarray,
    ignore_mask: np.ndarray | None = None,
    thickness: int = 2,
) -> tuple[float, float, float]:
    """Precision/recall/F1 of segmentation boundaries vs gt boundaries.

    A boundary pixel is "matched" if there is a gt boundary pixel
    within `thickness` pixels (Chebyshev distance).
    """
    pred_b = segmentation.find_boundaries(pred, mode="thick")
    gt_b = segmentation.find_boundaries(gt, mode="thick")
    if ignore_mask is not None:
        pred_b = pred_b & ~ignore_mask
        gt_b = gt_b & ~ignore_mask

    if not pred_b.any() or not gt_b.any():
        return 0.0, 0.0, 0.0

    # Dilate gt boundary by `thickness` pixels
    from skimage.morphology import dilation

    kernel = np.ones((2 * thickness + 1, 2 * thickness + 1), dtype=bool)
    gt_dilated = dilation(gt_b, footprint=kernel)
    pred_dilated = dilation(pred_b, footprint=kernel)

    tp = int((pred_b & gt_dilated).sum())
    fp = int((pred_b & ~gt_dilated).sum())
    fn = int((gt_b & ~pred_dilated).sum())

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(f1)


def topology_error(pred: np.ndarray, gt: np.ndarray) -> dict[str, int]:
    """Difference in connected-component counts (overall and per label)."""
    h, w = pred.shape

    def per_label_counts(arr: np.ndarray) -> dict[int, int]:
        out = {}
        for lbl in sorted(set(arr.flatten().tolist()) - {0}):
            cc = label(arr == lbl, connectivity=2)
            out[lbl] = int(cc.max())
        return out

    pc = per_label_counts(pred)
    gc = per_label_counts(gt)
    common = set(pc) | set(gc)
    diffs = {lbl: abs(pc.get(lbl, 0) - gc.get(lbl, 0)) for lbl in common}
    return {
        "n_pred_components": sum(pc.values()),
        "n_gt_components": sum(gc.values()),
        "abs_diff_total": abs(sum(pc.values()) - sum(gc.values())),
        "abs_diff_per_label_sum": int(sum(diffs.values())),
    }


def text_region_metrics(
    pred: np.ndarray,
    text_mask: np.ndarray,
) -> dict[str, float]:
    """How badly the engine labels text-block pixels as regions."""
    if not text_mask.any():
        return {"text_pixel_total": 0, "text_false_positive_rate": 0.0}
    text_preds = pred[text_mask]
    fp = int((text_preds > 0).sum())
    return {
        "text_pixel_total": int(text_mask.sum()),
        "text_false_positive_rate": _safe_div(fp, text_mask.sum()),
    }


def evaluate(
    pred: np.ndarray,
    gt: np.ndarray,
    panel_rgb: np.ndarray,
    text_mask: np.ndarray | None = None,
    colorbar_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute the full set of supervised + unsupervised metrics.

    Returns a JSON-serialisable dict. `panel_rgb` is passed in only so
    we can reuse the existing `boundary_alignment` and `tiny_fragments`
    diagnostics from geoseg itself.

    `text_mask` and `colorbar_mask` are pixels to EXCLUDE from scoring
    (text is annotated overlay; the colorbar area is not part of the
    segmentation target even though it sits inside the panel image).
    They are unioned internally.
    """
    # Build combined ignore mask.
    ignore_mask: np.ndarray | None = None
    if text_mask is not None:
        ignore_mask = text_mask.astype(bool)
    if colorbar_mask is not None:
        cb = colorbar_mask.astype(bool)
        ignore_mask = cb if ignore_mask is None else (ignore_mask | cb)

    # 2. mIoU with Hungarian matching (label-permutation invariant)
    mapping, iou_each = hungarian_match_iou(pred, gt, ignore_mask=ignore_mask)
    mean_iou = float(sum(iou_each.values()) / max(len(iou_each), 1))

    # Remap pred to gt label space for label-ID-comparison metrics.
    # Unmapped pred labels keep their original value; that is fine for
    # boundary/topology/fragment counts because those don't depend on
    # the absolute label numbers.
    pred_aligned = np.copy(pred)
    for pl, gl in mapping.items():
        pred_aligned[pred == pl] = gl

    # 1. Pixel accuracy (after remap so permuted labels still match)
    pa = pixel_accuracy(pred_aligned, gt, ignore_mask=ignore_mask)

    # 3. Boundary F1
    p, r, f1 = boundary_f1(pred_aligned, gt, ignore_mask=ignore_mask)

    # 4. Topology
    topo = topology_error(pred_aligned, gt)

    # 5. Existing geoseg diagnostics (re-used) — also use aligned labels.
    # Boundary alignment is unsupervised on panel_rgb; tiny_fragments is
    # on labels only, unaffected by text vs colorbar.
    ba = boundary_alignment(pred_aligned, panel_rgb)
    frags = tiny_fragments(pred_aligned)

    # 6. Text-region metrics — only meaningful when text_mask is present
    text_metrics = (
        text_region_metrics(pred_aligned, text_mask) if text_mask is not None else {}
    )

    return {
        "pixel_accuracy": round(pa, 4),
        "mIoU": round(mean_iou, 4),
        "iou_per_matched_label": {int(k): round(v, 4) for k, v in iou_each.items()},
        "matched_label_count": len(iou_each),
        "label_mapping": {int(k): int(v) for k, v in mapping.items()},
        "boundary_precision": round(p, 4),
        "boundary_recall": round(r, 4),
        "boundary_f1": round(f1, 4),
        "topology": topo,
        "geoseg_boundary_alignment_unsupervised": round(ba, 4),
        "geoseg_tiny_fragment_count": len(frags),
        "geoseg_tiny_fragment_area_fraction": round(
            sum(f["area_fraction"] for f in frags), 5
        ) if frags else 0.0,
        "text_metrics": text_metrics,
    }
