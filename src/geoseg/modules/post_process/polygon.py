"""Extract polygons and connected components from segmentation labels.

Bridges segment_engines output (labels array) to SPECFEM exporter input
(components + polygons).

Test scenario:
    >>> import numpy as np
    >>> labels = np.array([[0,0,1,1],[0,0,1,1],[2,2,2,0],[2,2,2,0]])
    >>> comps = extract_components(labels)
    >>> assert len(comps) == 2
    >>> assert all(k in comps[0] for k in ("id", "layer_id", "bbox", "area", "centroid"))
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from skimage import measure
from skimage.measure import regionprops
from skimage.morphology import remove_small_holes, remove_small_objects


def extract_components(
    labels: np.ndarray,
    min_area: int = 50,
) -> list[dict]:
    """Extract connected-component metadata from a label map.

    Each unique non-zero label is treated as a distinct layer.  For every
    connected component inside that layer, emits:

        {
            "id": int,
            "layer_id": int,
            "bbox": [x, y, w, h],
            "area": int,
            "centroid": [cx, cy],
        }

    Args:
        labels: (H, W) int array.  0 = background.
        min_area: Components smaller than this are discarded.

    Returns:
        List of component dicts, sorted by (layer_id, area descending).
    """
    components: list[dict] = []
    component_id = 0

    for layer_id in sorted(set(labels.flatten()) - {0}):
        mask = labels == layer_id
        mask = remove_small_objects(mask, max_size=max(1, min_area - 1))
        if not mask.any():
            continue

        for rp in regionprops(mask.astype(np.int32)):
            min_r, min_c, max_r, max_c = rp.bbox
            components.append({
                "id": component_id,
                "layer_id": int(layer_id),
                "bbox": [min_c, min_r, max_c - min_c, max_r - min_r],
                "area": int(rp.area),
                "centroid": [float(rp.centroid[1]), float(rp.centroid[0])],
            })
            component_id += 1

    components.sort(key=lambda c: (c["layer_id"], -c["area"]))
    # Re-assign sequential ids after sort
    for i, c in enumerate(components):
        c["id"] = i
    return components


def _simplify(points: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker simplification."""
    if len(points) < 4:
        return points
    return measure.approximate_polygon(points, tolerance=tol)


def labels_to_polygons(
    labels: np.ndarray,
    color_names: list[str] | None = None,
    min_area: int = 200,
    simplify_tol: float = 2.0,
    hole_size: int = 50,
) -> dict:
    """Convert a label map into a GeoJSON FeatureCollection.

    Args:
        labels: (H, W) int array. 0 = background.
        color_names: Optional list indexed by label id (label 0 ignored).
        min_area: Small objects are removed before contouring.
        simplify_tol: Douglas-Peucker tolerance in pixels.
        hole_size: Small holes are filled before contouring.

    Returns:
        GeoJSON FeatureCollection dict.
    """
    features = []
    unique_labels = sorted(set(labels.flatten()) - {0})

    for idx in unique_labels:
        name = (color_names[idx - 1] if color_names and idx - 1 < len(color_names) else f"layer_{idx}")
        mask = labels == idx
        if not mask.any():
            continue
        mask = remove_small_objects(mask, max_size=max(1, min_area - 1))
        mask = remove_small_holes(mask, max_size=max(1, hole_size - 1))
        if not mask.any():
            continue

        contours = measure.find_contours(mask.astype(np.uint8), level=0.5)
        for cnt in contours:
            if len(cnt) < 4:
                continue
            simplified = _simplify(cnt, simplify_tol)
            xy = [[float(p[1]), float(p[0])] for p in simplified]
            if xy and xy[0] != xy[-1]:
                xy.append(xy[0])
            features.append({
                "type": "Feature",
                "properties": {
                    "layer_id": int(idx),
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


__all__ = ["extract_components", "labels_to_polygons", "save_geojson"]
