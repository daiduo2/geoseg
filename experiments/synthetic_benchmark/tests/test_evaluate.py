"""Tests for the evaluation metrics."""

from __future__ import annotations

import numpy as np

from synthetic_benchmark.evaluate import (
    boundary_f1,
    evaluate,
    hungarian_match_iou,
    miou,
    pixel_accuracy,
    text_region_metrics,
    topology_error,
)


def _toy_panel(h: int = 32, w: int = 48, k: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a simple synthetic ground truth with k horizontal bands."""
    gt = np.zeros((h, w), dtype=np.int32)
    band = h // k
    for i in range(k):
        gt[i * band : (i + 1) * band, :] = i + 1
    rgb = np.dstack([gt * 50 % 256, gt * 80 % 256, gt * 110 % 256]).astype(np.uint8)
    return rgb, gt, rgb.copy()


def test_pixel_accuracy_perfect():
    _, gt, rgb = _toy_panel()
    assert pixel_accuracy(gt, gt) == 1.0


def test_pixel_accuracy_wrong():
    _, gt, rgb = _toy_panel()
    # Use a label that is guaranteed not to appear in gt (gt uses 0..k)
    pred = np.full_like(gt, 99)
    assert pixel_accuracy(pred, gt) == 0.0


def test_pixel_accuracy_with_ignore_mask():
    _, gt, rgb = _toy_panel()
    pred = np.ones_like(gt) * 99  # all wrong
    ignore = np.zeros_like(gt, dtype=bool)
    ignore[10:20, 10:40] = True
    # Without ignore: 0. Without ignored region we'd hit 0; with ignore
    # we only count the non-ignored pixels. Pixels outside ignore are wrong.
    pa = pixel_accuracy(pred, gt, ignore_mask=ignore)
    assert 0.0 <= pa <= 1.0
    # All non-ignored pixels are wrong
    nonignored = (~ignore).sum()
    assert pa == 0.0


def test_hungarian_match_finds_perfect_match_under_permutation():
    _, gt, rgb = _toy_panel(k=3)
    pred = gt.copy()
    pred[gt == 1] = 77
    pred[gt == 77] = 1
    pred[gt == 2] = 88
    pred[gt == 88] = 2
    mapping, iou_each = hungarian_match_iou(pred, gt)
    # Should match {77: 1, 88: 2, 3: 3} (or some equivalent)
    assert iou_each and all(v == 1.0 for v in iou_each.values())


def test_miou_perfect():
    _, gt, rgb = _toy_panel(k=3)
    assert miou(gt, gt) == 1.0


def test_miou_one_label():
    _, gt, rgb = _toy_panel(k=3)
    pred = np.ones_like(gt) * 1
    val = miou(pred, gt)
    # The single pred label gets matched to one of the gt labels
    # (the largest one, by Hungarian), so IoU for that match is < 1.0
    # and others are excluded. Expected IoU is area(largest_layer)/total.
    assert 0.0 < val < 1.0


def test_boundary_f1_perfect():
    _, gt, rgb = _toy_panel(k=3)
    p, r, f1 = boundary_f1(gt, gt)
    assert p == 1.0 and r == 1.0 and f1 == 1.0


def test_boundary_f1_all_wrong():
    _, gt, rgb = _toy_panel(k=3)
    pred = np.zeros_like(gt)
    p, r, f1 = boundary_f1(pred, gt)
    # pred has no boundary; gt does → precision undefined → returns 0
    assert p == 0.0 and r == 0.0 and f1 == 0.0


def test_topology_error_perfect():
    _, gt, rgb = _toy_panel(k=3)
    topo = topology_error(gt, gt)
    assert topo["n_pred_components"] == topo["n_gt_components"]
    assert topo["abs_diff_total"] == 0


def test_text_region_metrics():
    text_mask = np.zeros((10, 10), dtype=bool)
    text_mask[0:3, 0:5] = True
    pred = np.zeros((10, 10), dtype=np.int32)
    pred[0:3, 0:5] = 7  # all 15 text pixels marked as region 7
    out = text_region_metrics(pred, text_mask)
    assert out["text_pixel_total"] == 15
    assert out["text_false_positive_rate"] == 1.0


def test_evaluate_includes_all_metrics():
    _, gt, rgb = _toy_panel(k=3)
    metrics = evaluate(gt, gt, rgb)
    expected_keys = {
        "pixel_accuracy",
        "mIoU",
        "iou_per_matched_label",
        "matched_label_count",
        "label_mapping",
        "boundary_precision",
        "boundary_recall",
        "boundary_f1",
        "topology",
        "geoseg_boundary_alignment_unsupervised",
        "geoseg_tiny_fragment_count",
        "geoseg_tiny_fragment_area_fraction",
        "text_metrics",
    }
    assert expected_keys.issubset(metrics.keys())
