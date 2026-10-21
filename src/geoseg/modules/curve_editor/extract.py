"""Extract editable interface curves from an existing label image.

The extraction targets the boundaries between labels, not region outlines:
the label-difference mask is skeletonized into a one-pixel graph, the graph is
split into simple paths at junctions/endpoints, and each path is simplified to
control points.  This turns a raster segmentation into the same curve model a
human would draw, including open horizons that end at the image border and
closed lenses/loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class ExtractedCurve:
    points: list[tuple[float, float]]
    closed: bool
    length: float


def boundary_mask(labels: np.ndarray) -> np.ndarray:
    """Pixels where a 4-neighbour pair has different labels."""
    labels = np.asarray(labels)
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
    return boundary


def _neighbors(pixel: tuple[int, int], skeleton: np.ndarray) -> Iterator[tuple[int, int]]:
    row, col = pixel
    h, w = skeleton.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if 0 <= nr < h and 0 <= nc < w and skeleton[nr, nc]:
                yield (nr, nc)


def _edge(a: tuple[int, int], b: tuple[int, int]) -> tuple:
    return (a, b) if a <= b else (b, a)


def skeleton_paths(skeleton: np.ndarray) -> list[tuple[list[tuple[int, int]], bool]]:
    """Split a one-pixel skeleton into open and closed pixel paths."""
    from scipy.ndimage import convolve

    skeleton = np.asarray(skeleton, dtype=bool)
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    degree = convolve(skeleton.astype(np.uint8), kernel, mode="constant")
    nodes = {
        (int(r), int(c))
        for r, c in zip(*np.nonzero(skeleton & (degree != 2)))
    }

    used_edges: set[tuple] = set()
    visited: set[tuple[int, int]] = set()
    paths: list[tuple[list[tuple[int, int]], bool]] = []

    def walk(start: tuple[int, int], nxt: tuple[int, int], stop: set[tuple[int, int]]) -> list[tuple[int, int]]:
        path = [start]
        used_edges.add(_edge(start, nxt))
        visited.add(start)
        previous, current = start, nxt
        for _ in range(int(skeleton.sum()) + 2):
            path.append(current)
            visited.add(current)
            if current in stop:
                break
            candidates = [
                n
                for n in _neighbors(current, skeleton)
                if n != previous and _edge(current, n) not in used_edges
            ]
            if not candidates:
                break
            nxt = sorted(candidates)[0]
            used_edges.add(_edge(current, nxt))
            previous, current = current, nxt
        return path

    for node in sorted(nodes):
        for neighbor in sorted(_neighbors(node, skeleton)):
            if _edge(node, neighbor) in used_edges:
                continue
            paths.append((walk(node, neighbor, nodes), False))

    for pixel in sorted(set(map(tuple, np.argwhere(skeleton))) - visited):
        if pixel in visited:
            continue
        pixel = (int(pixel[0]), int(pixel[1]))
        if pixel in visited:
            continue
        path = [pixel]
        visited.add(pixel)
        previous, current = None, pixel
        closed = False
        for _ in range(int(skeleton.sum()) + 2):
            candidates = [
                n
                for n in _neighbors(current, skeleton)
                if n != previous and _edge(current, n) not in used_edges
            ]
            if not candidates:
                break
            nxt = sorted(candidates)[0]
            used_edges.add(_edge(current, nxt))
            path.append(nxt)
            visited.add(nxt)
            previous, current = current, nxt
            if current == pixel:
                closed = True
                break
        paths.append((path, closed))

    return paths


def _simplify(
    points: np.ndarray,
    *,
    closed: bool,
    tolerance: float,
    max_points: int,
) -> np.ndarray:
    from skimage import measure

    from geoseg.modules.curve_editor.spline import fit_freehand

    if len(points) < 2:
        return points
    if closed:
        polygon = np.vstack([points, points[:1]])
        simplified = measure.approximate_polygon(polygon, tolerance=tolerance)
        simplified = simplified[:-1]
        if len(simplified) < 3:
            return points
    else:
        simplified = measure.approximate_polygon(points, tolerance=tolerance)
    if len(simplified) > max_points:
        simplified = np.asarray(
            fit_freehand(simplified, n_points=max_points, closed=closed), dtype=float
        )
    return simplified


def curves_from_labels(
    labels: np.ndarray,
    *,
    tolerance: float = 1.5,
    max_points: int = 28,
    min_length: float = 10.0,
) -> list[ExtractedCurve]:
    """Extract major label boundaries as open/closed spline control points."""
    from skimage.morphology import skeletonize

    mask = boundary_mask(labels)
    if not mask.any():
        return []
    skeleton = skeletonize(mask)
    curves: list[ExtractedCurve] = []
    for path, closed in skeleton_paths(skeleton):
        points = np.asarray(path, dtype=float)[:, ::-1]  # (row, col) -> (x, y)
        if closed and len(points) > 1 and np.allclose(points[0], points[-1]):
            points = points[:-1]
        if len(points) < (3 if closed else 2):
            continue
        length = float(
            np.hypot(*np.diff(np.vstack([points, points[:1]]) if closed else points, axis=0).T).sum()
        )
        if length < min_length:
            continue
        simplified = _simplify(
            points, closed=closed, tolerance=tolerance, max_points=max_points
        )
        if len(simplified) < (3 if closed else 2):
            continue
        curves.append(
            ExtractedCurve(
                points=[(float(x), float(y)) for x, y in simplified],
                closed=bool(closed),
                length=length,
            )
        )
    curves.sort(key=lambda curve: (-curve.length, curve.closed))
    return curves


__all__ = ["ExtractedCurve", "boundary_mask", "curves_from_labels", "skeleton_paths"]
