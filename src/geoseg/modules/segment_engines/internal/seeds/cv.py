"""Computer-vision seed proposal helpers."""

from __future__ import annotations

import numpy as np

from geoseg.modules.segment_engines.internal.color import (
    _estimate_background_color,
    _is_background_v2,
)


def _online_color_groups(
    panel_rgb: np.ndarray,
    tolerance: float = 120.0,
    max_groups: int = 15,
    max_samples: int = 5000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Online tolerance-based colour grouping."""
    pixels = panel_rgb.reshape(-1, 3)
    n = len(pixels)
    if n > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, max_samples, replace=False)
        sample = pixels[idx].astype(np.float32)
    else:
        sample = pixels.astype(np.float32)

    groups: list[tuple[np.ndarray, int]] = []
    tol_sq = tolerance * tolerance

    for px in sample:
        matched = False
        for i, (mean, count) in enumerate(groups):
            diff = px - mean
            if np.dot(diff, diff) <= tol_sq:
                new_mean = (mean * count + px) / (count + 1)
                groups[i] = (new_mean, count + 1)
                matched = True
                break
        if not matched:
            groups.append((px.copy(), 1))
            if len(groups) > max_groups * 2:
                groups.sort(key=lambda g: g[1], reverse=True)
                groups = groups[:max_groups]

    groups.sort(key=lambda g: g[1], reverse=True)
    groups = groups[:max_groups]

    centers = np.array([g[0] for g in groups], dtype=np.uint8)
    counts = np.array([g[1] for g in groups], dtype=np.int64)
    return centers, counts


def _histogram_peaks(
    panel_rgb: np.ndarray,
    n_bins: int = 25,
    min_peak_ratio: float = 0.02,
) -> np.ndarray:
    """Find foreground colour peaks via grayscale histogram."""
    gray = panel_rgb.mean(axis=2).astype(np.uint8)
    hist, bin_edges = np.histogram(gray.flatten(), bins=n_bins, range=(0, 256))

    bg_idx = int(np.argmax(hist))
    bg_val = (bin_edges[bg_idx] + bin_edges[bg_idx + 1]) / 2.0
    total = gray.size

    peaks = []
    for i, count in enumerate(hist):
        if count / total < min_peak_ratio:
            continue
        val = (bin_edges[i] + bin_edges[i + 1]) / 2.0
        if abs(val - bg_val) < 30:
            continue
        peaks.append((i, val, count))

    if not peaks:
        return np.empty((0, 3), dtype=np.uint8)

    peaks.sort(key=lambda p: p[2], reverse=True)

    centers = []
    for i, _, _ in peaks[:8]:
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (gray >= low) & (gray < high)
        if mask.sum() == 0:
            continue
        rep = np.median(panel_rgb[mask], axis=0).astype(np.uint8)
        centers.append(rep)

    return np.array(centers, dtype=np.uint8)


def _cv_seeds(
    panel_rgb: np.ndarray, k: int
) -> tuple[np.ndarray, list[str]]:
    """Compute multi-source CV seeds by combining online groups and histogram peaks."""
    bg = _estimate_background_color(panel_rgb)

    og_centers, og_counts = _online_color_groups(
        panel_rgb, tolerance=60, max_groups=20
    )
    hp_centers = _histogram_peaks(panel_rgb, n_bins=25)

    candidates: list[np.ndarray] = []
    tags: list[str] = []

    for c, count in zip(og_centers, og_counts):
        if _is_background_v2(c, bg, threshold=80):
            continue
        candidates.append(c)
        tags.append(f"online(count={count})")

    for c in hp_centers:
        if _is_background_v2(c, bg, threshold=80):
            continue
        if not candidates:
            candidates.append(c)
            tags.append("histogram")
            continue
        existing = np.array(candidates, dtype=np.float32)
        dists = np.linalg.norm(existing - c.astype(np.float32), axis=1)
        if dists.min() > 40:
            candidates.append(c)
            tags.append("histogram")

    if not candidates:
        return np.empty((0, 3), dtype=np.uint8), []

    seeds = np.array(candidates, dtype=np.uint8)
    if len(seeds) > k + 2:
        seeds = seeds[: k + 2]
        tags = tags[: k + 2]
    return seeds, tags

__all__ = ["_cv_seeds", "_histogram_peaks", "_online_color_groups"]
