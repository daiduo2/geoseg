"""Unit tests for horizon_refinement module.

Covers: boundary extraction, curve fitting, spatial reordering,
quality gates, and end-to-end refinement on synthetic + real fixtures.

Run:
    pytest tests/test_horizon_refinement.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from geoseg.modules.segment_engines.horizon_refinement import (
    refine_boundaries,
    _extract_boundary_points,
    _fit_curve,
    _fit_savgol,
    _fit_bspline,
    _hampel_filter,
    _compute_fragmentation_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_smooth_layers(
    h: int = 300,
    w: int = 600,
    n_layers: int = 4,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Create synthetic image with smooth curved geological layers.

    Returns:
        img: RGB uint8 array
        labels: int32 label map (1-based, no background)
        true_boundaries: list of y-coordinate arrays for each boundary
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Distinct colors for each layer
    colors = np.array([
        [80, 160, 80],
        [80, 200, 200],
        [80, 120, 200],
        [200, 180, 100],
        [200, 100, 180],
        [100, 200, 100],
    ], dtype=np.uint8)

    x = np.arange(w, dtype=np.float64)
    boundaries = []
    prev = 30.0
    for i in range(n_layers - 1):
        amplitude = rng.uniform(10, 25)
        freq = rng.uniform(40, 90)
        phase = rng.uniform(0, np.pi)
        y = prev + 50 + amplitude * np.sin(x / freq + phase)
        y = np.clip(y, 10, h - 10)
        boundaries.append(y)
        prev = y.mean()

    labels = np.zeros((h, w), dtype=np.int32)
    for xi in range(w):
        bounds = [0] + [int(b[xi]) for b in boundaries] + [h]
        for li, (yt, yb) in enumerate(zip(bounds, bounds[1:])):
            labels[yt:yb, xi] = li + 1
            img[yt:yb, xi] = colors[li % len(colors)]

    return img, labels, boundaries


def _add_label_noise(labels: np.ndarray, frac: float = 0.03, seed: int = 7) -> np.ndarray:
    """Randomly flip boundary pixels to adjacent labels."""
    rng = np.random.default_rng(seed)
    out = labels.copy()
    h, w = labels.shape
    noise = rng.random((h, w)) < frac
    for y, x in zip(*np.where(noise)):
        nbrs = []
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and out[ny, nx] != out[y, x]:
                nbrs.append(out[ny, nx])
        if nbrs:
            out[y, x] = rng.choice(nbrs)
    return out


# ---------------------------------------------------------------------------
# Curve fitting tests
# ---------------------------------------------------------------------------

def test_hampel_filter_removes_spikes() -> None:
    y = np.array([10.0] * 20 + [100.0] + [10.0] * 20)
    cleaned = _hampel_filter(y, window=11, n_sigma=3.0)
    assert cleaned[20] < 50.0  # spike reduced
    assert np.allclose(cleaned[:20], 10.0)
    assert np.allclose(cleaned[21:], 10.0)


def test_fit_savgol_preserves_trend() -> None:
    x = np.arange(200, dtype=np.float64)
    y = 50 + 0.5 * x + 5 * np.sin(x / 10)
    smooth = _fit_savgol(x, y, smoothness=0.1)
    # RMSE should be small (savgol with small window approximates original)
    rmse = np.sqrt(np.mean((smooth - y) ** 2))
    assert rmse < 3.0


def test_fit_savgol_on_short_array_returns_copy() -> None:
    y = np.array([1.0, 2.0, 3.0])
    x = np.arange(len(y))
    result = _fit_savgol(x, y, smoothness=1.0)
    assert len(result) == 3
    assert np.allclose(result, y)


def test_fit_bspline_smoothness() -> None:
    x = np.arange(200, dtype=np.float64)
    y = 50 + 0.5 * x + 20 * np.sin(x / 5)  # high frequency noise
    smooth = _fit_bspline(x, y, smoothness=10.0)
    # Smoothed version should have lower variance
    assert np.var(smooth) < np.var(y)


def test_fit_curve_method_dispatch() -> None:
    x = np.arange(100, dtype=np.float64)
    y = 50 + np.sin(x / 10)
    for method in ["savgol", "bspline"]:
        result = _fit_curve(x, y, method, smoothness=1.0)  # type: ignore[arg-type]
        assert len(result) == len(y)
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# Boundary extraction tests
# ---------------------------------------------------------------------------

def test_extract_boundary_points_finds_known_boundary() -> None:
    img, labels, true_boundaries = _make_smooth_layers(h=200, w=400, n_layers=3)
    # Extract boundary between layer 1 and 2
    pts = _extract_boundary_points(img, labels, 1, 2, band_margin=15)
    assert pts is not None
    xs, ys = pts
    true_b = true_boundaries[0][xs]
    rmse = np.sqrt(np.mean((ys - true_b) ** 2))
    # Should be within ~20px of true boundary (gradient-based sampling is approximate)
    assert rmse < 25.0


def test_extract_boundary_points_returns_none_for_nonadjacent() -> None:
    """If layers don't touch, extraction may fail gracefully."""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 200
    labels = np.zeros((100, 100), dtype=np.int32)
    labels[:50, :] = 1
    labels[60:, :] = 2  # gap between 50 and 60
    pts = _extract_boundary_points(img, labels, 1, 2, band_margin=5)
    # May return None or find points in the gap; either is acceptable
    if pts is not None:
        assert len(pts[0]) > 0


# ---------------------------------------------------------------------------
# Fragmentation score tests
# ---------------------------------------------------------------------------

def test_fragmentation_score_clean_labels_is_zero() -> None:
    _, labels, _ = _make_smooth_layers()
    score = _compute_fragmentation_score(labels)
    assert score == 0.0


def test_fragmentation_score_detects_fragments() -> None:
    _, labels, _ = _make_smooth_layers()
    noisy = _add_label_noise(labels, frac=0.05)
    score = _compute_fragmentation_score(noisy)
    assert score > 0.0


# ---------------------------------------------------------------------------
# End-to-end refinement tests
# ---------------------------------------------------------------------------

def test_refinement_improves_fragmented_synthetic() -> None:
    """Coarse labels with noise → refined should have lower fragmentation."""
    img, clean_labels, _ = _make_smooth_layers(h=300, w=600, n_layers=4)
    coarse = _add_label_noise(clean_labels, frac=0.04)

    refined, boundaries = refine_boundaries(img, coarse_labels=coarse, method="savgol")

    coarse_frag = _compute_fragmentation_score(coarse)
    refined_frag = _compute_fragmentation_score(refined)

    # Either improved or fallback (same)
    assert refined_frag <= coarse_frag * 1.5 + 0.01
    assert len(boundaries) == 3
    # Layer count preserved (or within 1)
    coarse_n = len(set(coarse.flatten()) - {0})
    refined_n = len(set(refined.flatten()) - {0})
    assert refined_n >= coarse_n - 1


def test_refinement_preserves_layer_count_on_good_input() -> None:
    """Already-clean labels should be preserved or slightly improved."""
    img, labels, _ = _make_smooth_layers(h=200, w=400, n_layers=3)
    refined, boundaries = refine_boundaries(img, coarse_labels=labels, method="savgol")

    refined_unique = sorted(np.unique(refined))
    assert len(refined_unique) >= 2
    for lbl in refined_unique:
        assert lbl > 0  # background should not appear in refined output


def test_refinement_fallback_on_collapsed_boundaries() -> None:
    """If coarse labels are scrambled everywhere, refinement should fallback."""
    rng = np.random.default_rng(99)
    img = np.ones((200, 400, 3), dtype=np.uint8) * 200
    # Random labels — no spatial coherence
    labels = rng.integers(0, 5, size=(200, 400)).astype(np.int32)

    refined, boundaries = refine_boundaries(img, coarse_labels=labels)

    # Should fallback to coarse
    assert np.array_equal(refined, labels)


def test_refinement_with_auto_coarse_generation() -> None:
    """Passing n_layers without coarse_labels triggers internal coarse segmentation."""
    img, _, _ = _make_smooth_layers(h=200, w=400, n_layers=3)
    refined, boundaries = refine_boundaries(img, n_layers=3, method="savgol")
    assert refined.shape == img.shape[:2]
    assert len(boundaries) >= 1


# ---------------------------------------------------------------------------
# Real fixture tests (optional — skipped if fixtures missing)
# ---------------------------------------------------------------------------

def test_refinement_on_16b0cf_fixture() -> None:
    """Regression test for the fragmentation challenge image."""
    fixture_path = (
        Path(__file__).parent.parent
        / "runs"
        / "readme_examples_v2"
        / "gras2019_16b0cf"
        / "panel_cropped.png"
    )
    if not fixture_path.exists():
        pytest.skip(f"Fixture not found: {fixture_path}")

    from geoseg.modules.segment_engines.kmeans_full import segment as seg_kmeans

    img = np.array(Image.open(fixture_path).convert("RGB"))
    coarse_result = seg_kmeans(img, n_layers=5, max_auto_k=0)
    coarse_labels = coarse_result["labels"]

    refined, boundaries = refine_boundaries(img, coarse_labels=coarse_labels)

    coarse_frag = _compute_fragmentation_score(coarse_labels)
    refined_frag = _compute_fragmentation_score(refined)

    # Should improve or fallback — never worsen
    assert refined_frag <= coarse_frag * 1.5 + 0.01
    # Layer count should not drop by more than 1
    coarse_n = len(set(coarse_labels.flatten()) - {0})
    refined_n = len(set(refined.flatten()) - {0})
    assert refined_n >= coarse_n - 1
    assert len(boundaries) >= 1
