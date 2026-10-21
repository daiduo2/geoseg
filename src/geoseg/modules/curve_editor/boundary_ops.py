"""Boundary geometry helpers for drawn strokes.

Two cases from the annotation discussion are handled here:

- a freehand region stroke that runs past the image edge is clipped to the
  panel rectangle, so the closed polygon follows the image border instead of
  closing with an arbitrary chord;
- an open interface whose ends do not reach a boundary is extended (or trimmed
  if it overshot) along its tangent to the nearest existing label boundary or
  image edge.
"""

from __future__ import annotations

import numpy as np

from geoseg.modules.curve_editor.extract import boundary_mask


def label_boundary_mask(labels: np.ndarray, *, include_border: bool = True) -> np.ndarray:
    mask = boundary_mask(labels)
    if include_border:
        mask[0, :] = mask[-1, :] = True
        mask[:, 0] = mask[:, -1] = True
    return mask


def _dedupe(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return points
    if len(points) > 2 and np.allclose(points[0], points[-1]):
        points = points[:-1]
    keep = [points[0]]
    for point in points[1:]:
        if np.hypot(point[0] - keep[-1][0], point[1] - keep[-1][1]) > 1e-9:
            keep.append(point)
    return np.asarray(keep, dtype=float)


def clip_polygon_to_rect(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Sutherland-Hodgman clip of a closed polygon to the image rectangle."""
    points = _dedupe(np.asarray(points, dtype=float))
    if len(points) < 3:
        raise ValueError("a region polygon needs at least three points")
    h, w = int(shape[0]), int(shape[1])
    polygon = [(float(x), float(y)) for x, y in points]

    def clip(
        polygon_: list[tuple[float, float]],
        inside,
        intersect,
    ) -> list[tuple[float, float]]:
        if not polygon_:
            return []
        result: list[tuple[float, float]] = []
        previous = polygon_[-1]
        previous_inside = inside(previous)
        for current in polygon_:
            current_inside = inside(current)
            if current_inside:
                if not previous_inside:
                    result.append(intersect(previous, current))
                result.append(current)
            elif previous_inside:
                result.append(intersect(previous, current))
            previous, previous_inside = current, current_inside
        return result

    def vertical(x_value: float):
        def inside(point):
            return point[0] >= x_value

        def intersect(a, b):
            t = (x_value - a[0]) / (b[0] - a[0])
            return (x_value, a[1] + t * (b[1] - a[1]))

        return inside, intersect

    def horizontal(y_value: float):
        def inside(point):
            return point[1] >= y_value

        def intersect(a, b):
            t = (y_value - a[1]) / (b[1] - a[1])
            return (a[0] + t * (b[0] - a[0]), y_value)

        return inside, intersect

    for inside, intersect in (
        vertical(0.0),
        horizontal(0.0),
    ):
        polygon = clip(polygon, inside, intersect)
    for inside, intersect in (
        (lambda p: p[0] <= w - 1, vertical(w - 1)[1]),
        (lambda p: p[1] <= h - 1, horizontal(h - 1)[1]),
    ):
        polygon = clip(polygon, inside, intersect)

    if len(polygon) < 3:
        raise ValueError("stroke lies outside the image")
    clipped = _dedupe(np.asarray(polygon, dtype=float))
    if len(clipped) < 3:
        raise ValueError("stroke lies outside the image")
    return clipped


def self_intersects(points: np.ndarray) -> bool:
    """True if a closed polyline has a non-adjacent segment intersection."""
    points = _dedupe(np.asarray(points, dtype=float))
    n = len(points)
    if n < 4:
        return False
    segments = [(points[i], points[(i + 1) % n]) for i in range(n)]

    def orientation(a, b, c) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def intersects(a, b, c, d) -> bool:
        o1 = orientation(a, b, c)
        o2 = orientation(a, b, d)
        o3 = orientation(c, d, a)
        o4 = orientation(c, d, b)
        if o1 * o2 < 0 and o3 * o4 < 0:
            return True
        return False

    for i in range(n):
        for j in range(i + 1, n):
            if j == i or j == (i + 1) % n or (i == 0 and j == n - 1):
                continue
            if intersects(*segments[i], *segments[j]):
                return True
    return False


def _clip_to_image(point: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    return np.array(
        [np.clip(point[0], 0.0, w - 1.0), np.clip(point[1], 0.0, h - 1.0)],
        dtype=float,
    )


def _label_at(point: np.ndarray, labels: np.ndarray) -> int:
    col = int(round(point[0]))
    row = int(round(point[1]))
    if 0 <= row < labels.shape[0] and 0 <= col < labels.shape[1]:
        return int(labels[row, col])
    return 0


def _primary_label(points: np.ndarray, labels: np.ndarray) -> int:
    """Most common nonzero label along the drawn open curve."""
    counts: dict[int, int] = {}
    for start, end in zip(points[:-1], points[1:]):
        length = float(np.hypot(*(end - start)))
        steps = max(1, int(np.ceil(length)))
        for step in range(steps + 1):
            point = start + (end - start) * (step / steps)
            label = _label_at(point, labels)
            if label != 0:
                counts[label] = counts.get(label, 0) + 1
    if not counts:
        return 0
    midpoint = points[len(points) // 2]
    middle_label = _label_at(midpoint, labels)
    best = max(counts.values())
    if middle_label in counts and counts[middle_label] >= best * 0.8:
        return middle_label
    return max(counts, key=lambda label: counts[label])


def _image_edge_point(
    endpoint: np.ndarray, direction: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    h, w = shape
    direction = np.asarray(direction, dtype=float)
    length = float(np.hypot(*direction))
    if length < 1e-9:
        direction = np.array([1.0, 0.0])
    else:
        direction = direction / length
    candidates = []
    for axis in (0, 1):
        if abs(direction[axis]) > 1e-9:
            target = 0.0 if direction[axis] < 0 else (w - 1 if axis == 0 else h - 1)
            t = (target - endpoint[axis]) / direction[axis]
            if t > 0:
                candidates.append(endpoint + direction * t)
    if not candidates:
        return _clip_to_image(endpoint, shape)
    return _clip_to_image(
        min(candidates, key=lambda point: float(np.hypot(*(point - endpoint)))),
        shape,
    )


def _snap_endpoint_to_target(
    endpoint: np.ndarray,
    neighbor: np.ndarray,
    target_mask: np.ndarray,
    max_trim: float,
) -> np.ndarray:
    """Extend an inside endpoint to the target boundary, or trim an outside one."""
    h, w = target_mask.shape
    direction = endpoint - neighbor
    length = float(np.hypot(*direction))
    if length < 1e-9:
        direction = np.array([1.0, 0.0])
    else:
        direction = direction / length

    def sample(distance: float) -> tuple[bool, np.ndarray]:
        point = endpoint + direction * distance
        col = int(round(point[0]))
        row = int(round(point[1]))
        if 0 <= row < h and 0 <= col < w:
            return bool(target_mask[row, col]), point
        return False, point

    inside, _ = sample(0.0)
    if inside:
        diagonal = float(np.hypot(h, w))
        for distance in np.arange(0.5, diagonal + 0.5, 0.5):
            is_inside, point = sample(distance)
            if not is_inside:
                return _clip_to_image(point, target_mask.shape)
        return _image_edge_point(endpoint, direction, target_mask.shape)

    # Endpoint is outside the intended region (overshoot): trim to the nearest
    # crossing back into the region, searching both ways along the tangent.
    best: np.ndarray | None = None
    best_distance = float("inf")
    for sign in (1.0, -1.0):
        for distance in np.arange(0.5, max_trim + 0.5, 0.5):
            is_inside, point = sample(sign * distance)
            if is_inside:
                if distance < best_distance:
                    best_distance = distance
                    best = point
                break
    if best is not None:
        return _clip_to_image(best, target_mask.shape)
    return _clip_to_image(endpoint, target_mask.shape)


def _target_mask(primary: int, labels: np.ndarray) -> np.ndarray:
    if primary != 0:
        return labels == primary
    if (labels != 0).any():
        return labels == 0
    return np.ones_like(labels, dtype=bool)


def extend_open_curve(
    points: np.ndarray,
    labels: np.ndarray,
    *,
    max_trim: float = 60.0,
) -> np.ndarray:
    """Extend or trim an open interface until both ends hit a region boundary.

    The target region is the label covering most of the stroke.  Endpoints
    inside that region are extended outward to its boundary; endpoints that
    overshot into a neighbouring region/background are trimmed back to the
    nearest boundary along their tangent.  With no labels, both ends extend to
    the panel edge.
    """
    points = _dedupe(np.asarray(points, dtype=float))
    if len(points) < 2:
        raise ValueError("an open interface needs at least two points")
    labels = np.asarray(labels)
    primary = _primary_label(points, labels)
    target = _target_mask(primary, labels)
    start = _clip_to_image(points[0], labels.shape)
    end = _clip_to_image(points[-1], labels.shape)

    start = _snap_endpoint_to_target(start, points[1], target, max_trim)
    end = _snap_endpoint_to_target(end, points[-2], target, max_trim)

    if len(points) == 2:
        adjusted = np.vstack([start, end])
    else:
        adjusted = np.vstack([start, points[1:-1], end])
    adjusted = _dedupe(adjusted)
    if len(adjusted) < 2:
        raise ValueError("endpoint adjustment collapsed the curve")
    if float(np.hypot(*(adjusted[-1] - adjusted[0]))) < 1.0:
        raise ValueError("endpoint adjustment collapsed the curve")
    return adjusted


__all__ = [
    "clip_polygon_to_rect",
    "extend_open_curve",
    "label_boundary_mask",
    "self_intersects",
]
