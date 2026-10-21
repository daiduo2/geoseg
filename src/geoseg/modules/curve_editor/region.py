"""Freehand region mode: rough stroke -> image boundary -> filled region.

Advisor workflow this implements:

1. draw a rough closed stroke with the mouse;
2. snap the stroke to the strongest nearby image gradient (the real boundary);
3. click inside to say which side is the region;
4. fill that interior and parameterize the boundary by arc length;
5. keep control points at high-curvature / high-gradient locations.

The headless functions here are independent of napari so the snap quality and
control-point selection can be tested on synthetic and curated panels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks

from geoseg.modules.curve_editor.boundary_ops import clip_polygon_to_rect


@dataclass
class SnappedRegion:
    boundary: np.ndarray
    control_points: np.ndarray
    interior: tuple[float, float]
    mean_gradient: float


def edge_strength(image: np.ndarray, *, sigma: float = 1.0) -> np.ndarray:
    """Normalized color/grayscale gradient magnitude in [0, 1]."""
    from skimage.color import rgb2gray, rgb2lab
    from skimage.filters import sobel

    array = np.asarray(image)
    if array.ndim == 3:
        if array.dtype == np.uint8:
            rgb = array[..., :3].astype(float) / 255.0
        else:
            rgb = array[..., :3].astype(float)
            if rgb.max() > 1.5:
                rgb /= 255.0
        if array.shape[2] >= 3:
            lab = rgb2lab(rgb)
            channels = [lab[..., 0] / 100.0, lab[..., 1] / 128.0, lab[..., 2] / 128.0]
        else:
            channels = [rgb2gray(rgb)]
    else:
        gray = array.astype(float)
        if gray.max() > 1.5:
            gray /= 255.0
        channels = [gray]

    magnitude = None
    for channel in channels:
        smooth = ndimage.gaussian_filter(channel, sigma=sigma)
        gx = sobel(smooth, axis=1)
        gy = sobel(smooth, axis=0)
        current = np.hypot(gx, gy)
        magnitude = current if magnitude is None else magnitude + current
    if magnitude is None:
        return np.zeros(array.shape[:2], dtype=float)
    low, high = np.percentile(magnitude, [2, 98])
    if high <= low:
        return np.zeros_like(magnitude, dtype=float)
    return np.clip((magnitude - low) / (high - low), 0.0, 1.0)


def close_and_resample(points: np.ndarray, n: int = 192) -> np.ndarray:
    """Close a stroke and resample it at equal arc-length intervals."""
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        raise ValueError("a region stroke needs at least two points")
    if np.allclose(points[0], points[-1]):
        points = points[:-1]
    if len(points) < 3:
        raise ValueError("a closed region needs at least three distinct points")
    dense = np.vstack([points, points[:1]])
    diffs = np.diff(dense, axis=0)
    cumulative = np.concatenate([[0.0], np.cumsum(np.hypot(diffs[:, 0], diffs[:, 1]))])
    total = float(cumulative[-1])
    if total <= 0:
        raise ValueError("stroke has zero length")
    targets = np.linspace(0.0, total, int(n), endpoint=False)
    x = np.interp(targets, cumulative, dense[:, 0])
    y = np.interp(targets, cumulative, dense[:, 1])
    return np.column_stack([x, y])


def _normals(points: np.ndarray) -> np.ndarray:
    dx = np.roll(points[:, 0], -1) - np.roll(points[:, 0], 1)
    dy = np.roll(points[:, 1], -1) - np.roll(points[:, 1], 1)
    length = np.hypot(dx, dy)
    length[length == 0] = 1.0
    return np.column_stack([-dy / length, dx / length])


def snap_closed_curve(
    image: np.ndarray,
    stroke: np.ndarray,
    *,
    samples: int = 192,
    search_radius: float = 12.0,
    step: float = 0.5,
    iterations: int = 2,
    smooth_sigma: float = 2.5,
    distance_penalty: float = 0.12,
    min_gradient: float = 0.02,
) -> np.ndarray:
    """Snap a rough closed stroke to the strongest nearby image boundary.

    Each sample searches along its normal within ``search_radius``; the score
    is gradient magnitude minus a small distance penalty, so a weaker nearby
    edge is preferred over a strong edge far from the user's stroke.
    """
    edge = edge_strength(image)
    points = close_and_resample(clip_polygon_to_rect(stroke, edge.shape), samples)
    offsets = np.arange(-search_radius, search_radius + step / 2, step)

    for _ in range(max(1, iterations)):
        normals = _normals(points)
        xs = points[:, 0][None, :] + offsets[:, None] * normals[:, 0][None, :]
        ys = points[:, 1][None, :] + offsets[:, None] * normals[:, 1][None, :]
        values = ndimage.map_coordinates(
            edge, [ys, xs], order=1, mode="nearest"
        )
        score = values - distance_penalty * (np.abs(offsets)[:, None] / search_radius)
        best = np.argmax(score, axis=0)
        shift = offsets[best]
        shift[values.max(axis=0) < min_gradient] = 0.0
        points = points + shift[:, None] * normals
        points[:, 0] = np.clip(points[:, 0], 0, edge.shape[1] - 1)
        points[:, 1] = np.clip(points[:, 1], 0, edge.shape[0] - 1)
        points = ndimage.gaussian_filter1d(points, smooth_sigma, axis=0, mode="wrap")

    return close_and_resample(points, samples)


def _closed_curvature(points: np.ndarray) -> np.ndarray:
    x = np.asarray(points[:, 0], dtype=float)
    y = np.asarray(points[:, 1], dtype=float)
    dx = np.roll(x, -1) - np.roll(x, 1)
    dy = np.roll(y, -1) - np.roll(y, 1)
    ddx = np.roll(x, -1) - 2 * x + np.roll(x, 1)
    ddy = np.roll(y, -1) - 2 * y + np.roll(y, 1)
    denominator = np.power(dx * dx + dy * dy, 1.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        curvature = np.abs(dx * ddy - dy * ddx) / denominator
    return np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)


def _sample_edge(edge: np.ndarray, points: np.ndarray) -> np.ndarray:
    return ndimage.map_coordinates(
        edge, [points[:, 1], points[:, 0]], order=1, mode="nearest"
    )


def select_control_points(
    boundary: np.ndarray,
    edge: np.ndarray | None = None,
    *,
    samples: int = 256,
    min_spacing: float = 12.0,
    max_points: int = 24,
    min_points: int = 6,
) -> np.ndarray:
    """Pick control points at strong curvature / gradient locations.

    Curvature peaks dominate; gradient peaks are merged in when they are
    spatially separated.  If the boundary is too smooth, evenly spaced points
    are used so the spline still has enough freedom.
    """
    points = close_and_resample(boundary, samples)
    curvature = ndimage.gaussian_filter1d(_closed_curvature(points), 2.0, mode="wrap")
    spacing = max(1, int(round(min_spacing / max(_perimeter(points) / samples, 1e-6))))

    prominence = max(float(curvature.max()) * 0.08, 1e-6)
    curvature_peaks, _ = find_peaks(
        curvature, distance=spacing, prominence=prominence
    )
    candidates = set(int(index) for index in curvature_peaks)

    if edge is not None:
        gradient = ndimage.gaussian_filter1d(_sample_edge(edge, points), 2.0, mode="wrap")
        gradient_prominence = max(float(gradient.max()) * 0.12, 1e-6)
        gradient_peaks, _ = find_peaks(
            gradient, distance=spacing, prominence=gradient_prominence
        )
        candidates.update(int(index) for index in gradient_peaks)

    if len(candidates) < min_points:
        for index in np.linspace(0, samples - 1, min_points, endpoint=False):
            candidates.add(int(round(index)))

    ordered = sorted(candidates)
    scores = {index: float(curvature[index]) for index in ordered}
    if edge is not None:
        gradient = _sample_edge(edge, points)
        for index in ordered:
            scores[index] += 0.25 * float(gradient[index])

    selected: list[int] = []
    for index in sorted(ordered, key=lambda item: scores[item], reverse=True):
        if len(selected) >= max_points:
            break
        if all(_circular_distance(index, other, samples) >= spacing for other in selected):
            selected.append(index)
    if len(selected) < min_points:
        for index in ordered:
            if index not in selected:
                selected.append(index)
            if len(selected) >= min_points:
                break
    selected.sort()
    return points[selected]


def _perimeter(points: np.ndarray) -> float:
    dense = np.vstack([points, points[:1]])
    return float(np.hypot(*np.diff(dense, axis=0).T).sum())


def _circular_distance(a: int, b: int, n: int) -> int:
    difference = abs(a - b)
    return min(difference, n - difference)


def interior_point(
    boundary: np.ndarray,
    shape: tuple[int, int],
    preferred: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Return ``preferred`` if inside the polygon, else the deepest interior point."""
    from skimage.draw import polygon2mask

    rows = np.clip(np.rint(boundary[:, 1]).astype(int), 0, shape[0] - 1)
    cols = np.clip(np.rint(boundary[:, 0]).astype(int), 0, shape[1] - 1)
    mask = polygon2mask(shape, np.column_stack([rows, cols]))
    if not mask.any():
        return (float(shape[1] / 2), float(shape[0] / 2))

    distance = ndimage.distance_transform_edt(mask)
    if preferred is not None:
        col = int(np.clip(round(preferred[0]), 0, shape[1] - 1))
        row = int(np.clip(round(preferred[1]), 0, shape[0] - 1))
        if mask[row, col] and distance[row, col] >= 1.0:
            return (float(col), float(row))
        # Click outside, or too close to the boundary for the spline to keep
        # the seed inside: move to the nearest comfortably-interior pixel.
        safe = distance >= min(2.0, float(distance.max()))
        if safe.any():
            _, indices = ndimage.distance_transform_edt(~safe, return_indices=True)
            row, col = int(indices[0, row, col]), int(indices[1, row, col])
            return (float(col), float(row))
        if mask[row, col]:
            return (float(col), float(row))

    row, col = np.unravel_index(int(np.argmax(distance)), mask.shape)
    return (float(col), float(row))


def snap_region(
    image: np.ndarray,
    stroke: np.ndarray,
    *,
    interior: tuple[float, float] | None = None,
    samples: int = 192,
    search_radius: float = 12.0,
    **kwargs,
) -> SnappedRegion:
    """Full headless workflow: stroke -> boundary -> control points + interior."""
    boundary = snap_closed_curve(
        image, stroke, samples=samples, search_radius=search_radius, **kwargs
    )
    edge = edge_strength(image)
    control_points = select_control_points(boundary, edge=edge)
    interior = interior_point(boundary, image.shape[:2], preferred=interior)
    mean_gradient = float(_sample_edge(edge, boundary).mean())
    return SnappedRegion(
        boundary=boundary,
        control_points=control_points,
        interior=interior,
        mean_gradient=mean_gradient,
    )


__all__ = [
    "SnappedRegion",
    "close_and_resample",
    "edge_strength",
    "interior_point",
    "select_control_points",
    "snap_closed_curve",
    "snap_region",
]
