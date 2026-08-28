"""Convert label maps to polygons.

For each color label, extract connected components, smooth their boundary,
simplify with Douglas-Peucker, and emit polygon vertices in the panel's
coordinate frame.

Output format is GeoJSON-like so SPECFEM downstream tooling can consume it
without bespoke parsing.

Public API:
    labels_to_polygons(labels, color_names, min_area=200, simplify_tol=2.0)
        -> dict (GeoJSON FeatureCollection)
    save_geojson(features, path)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from skimage import measure
from skimage.morphology import remove_small_holes, remove_small_objects


def _simplify(points: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker simplification using skimage's `measure.approximate_polygon`."""
    if len(points) < 4:
        return points
    return measure.approximate_polygon(points, tolerance=tol)


def labels_to_polygons(
    labels: np.ndarray,
    color_names: list[str],
    min_area: int = 200,
    simplify_tol: float = 2.0,
    hole_size: int = 50,
) -> dict:
    """Convert a (H, W) label map into a GeoJSON FeatureCollection.

    Each feature is a polygon for one connected component of one color label.
    Coordinates are pixel (x, y) in panel-local space; the caller can add
    georeferencing later.
    """
    features = []
    for idx, name in enumerate(color_names):
        mask = labels == idx
        if not mask.any():
            continue
        mask = remove_small_objects(mask, max_size=max(1, min_area - 1))
        mask = remove_small_holes(mask, max_size=max(1, hole_size - 1))
        if not mask.any():
            continue
        # find_contours returns (row, col) — convert to (x, y)
        contours = measure.find_contours(mask.astype(np.uint8), level=0.5)
        for cnt in contours:
            if len(cnt) < 4:
                continue
            simplified = _simplify(cnt, simplify_tol)
            # Convert (row, col) → (x, y); also close the ring
            xy = [[float(p[1]), float(p[0])] for p in simplified]
            if xy[0] != xy[-1]:
                xy.append(xy[0])
            features.append({
                "type": "Feature",
                "properties": {
                    "color_index": idx,
                    "color_name": name,
                    "n_vertices": len(xy),
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [xy],
                },
            })
    return {"type": "FeatureCollection", "features": features}


def save_geojson(features: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(features, ensure_ascii=False, indent=2), encoding="utf-8")


def render_label_overlay(panel_rgb: np.ndarray, labels: np.ndarray, palette: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Composite the label map onto the panel for visual QA.

    palette: (k, 3) uint8 colors.
    """
    overlay = np.zeros_like(panel_rgb)
    for idx in range(palette.shape[0]):
        overlay[labels == idx] = palette[idx]
    blended = (alpha * overlay + (1 - alpha) * panel_rgb).astype(np.uint8)
    return blended


__all__ = ["labels_to_polygons", "save_geojson", "render_label_overlay"]
