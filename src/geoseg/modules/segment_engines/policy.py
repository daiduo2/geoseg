"""Routing policy for segmentation engines."""

from __future__ import annotations

import numpy as np

from geoseg.modules.segment_engines.internal.color import saturation_ratio


def _is_grayscale(panel_rgb: np.ndarray, threshold: float = 15.0) -> bool:
    """Check if panel is grayscale (low per-pixel saturation)."""
    diff = panel_rgb.max(axis=2).astype(np.float32) - panel_rgb.min(axis=2).astype(np.float32)
    return float(diff.mean()) < threshold


def _edge_density(panel_rgb: np.ndarray) -> float:
    """Estimate edge density via grayscale Sobel magnitude."""
    from skimage.filters import sobel

    gray = panel_rgb.mean(axis=2)
    edges = sobel(gray)
    return float((edges > 0.05).mean())


def _stable_panel_hash(panel_rgb: np.ndarray, mod: int) -> int:
    """Deterministic hash from scattered pixels for engine rotation."""
    h, w = panel_rgb.shape[:2]
    ys = np.linspace(h // 4, 3 * h // 4, 4).astype(int)
    xs = np.linspace(w // 4, 3 * w // 4, 4).astype(int)
    samples = panel_rgb[np.ix_(ys, xs)].flatten().astype(np.uint64)
    return int((np.sum(samples) + h * 7 + w * 13) % mod)


def select_engine(
    panel_rgb: np.ndarray,
    quality_preference: str = "balanced",
    has_colorbar: bool = False,
    is_velocity_model: bool = True,
) -> str:
    """Select a segmentation engine based on image features and preference."""
    if not is_velocity_model:
        return "skip"

    sat = saturation_ratio(panel_rgb)

    if sat < 0.005:
        return "grayscale_agglomerative"

    if sat < 0.1:
        return "v4_kmeans_colorbar" if has_colorbar else "v4_kmeans_pastel"

    if 0.1 <= sat < 0.5:
        return "v4_kmeans"

    if quality_preference == "fast":
        return "v4_kmeans"
    if quality_preference == "balanced":
        engines = ["kmeans_full", "edge_guided", "edge_grow"]
        return engines[_stable_panel_hash(panel_rgb, len(engines))]
    if quality_preference == "best":
        engines = ["ensemble", "kmeans_full", "edge_guided", "edge_grow"]
        return engines[_stable_panel_hash(panel_rgb, len(engines))]

    return "edge_guided"


__all__ = ["_edge_density", "_is_grayscale", "_stable_panel_hash", "select_engine"]
