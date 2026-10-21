"""Boundary refinement with label and curvature priors.

Raw skeleton paths follow every label edge, including text holes, background
outlines and tiny annotation loops.  This layer treats those as evidence, not
truth:

- a boundary is kept only when its two sides are two different foreground
  labels (or background is explicitly allowed);
- the ordered samples are fit with a smoothing spline, with endpoints pinned;
- smoothing is increased until the curvature stays under a limit, or the
  curve is rejected;
- short curves are discarded.

The output is still a dense polyline; control-point selection happens later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import splprep, splev

from geoseg.modules.curve_editor.extract import ExtractedCurve


@dataclass
class BoundaryRefinementConfig:
    smoothing: float = 5.0
    endpoint_weight: float = 100.0
    min_length: float = 25.0
    max_curvature: float = 0.35
    max_smoothing_scale: float = 8.0
    ignore_background: bool = True
    side_sample_distance: float = 3.0
    samples: int = 240


@dataclass
class RefinedBoundary:
    points: np.ndarray
    closed: bool
    length: float
    labels: tuple[int, int] | None
    max_curvature: float
    raw_length: float
    smoothing: float


def _resample(points: np.ndarray, closed: bool, n: int) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if closed and len(points) > 1 and np.allclose(points[0], points[-1]):
        points = points[:-1]
    dense = np.vstack([points, points[:1]]) if closed else points
    diffs = np.diff(dense, axis=0)
    cumulative = np.concatenate([[0.0], np.cumsum(np.hypot(diffs[:, 0], diffs[:, 1]))])
    total = float(cumulative[-1])
    if total <= 0:
        return points
    targets = np.linspace(0.0, total, n, endpoint=not closed)
    x = np.interp(targets, cumulative, dense[:, 0])
    y = np.interp(targets, cumulative, dense[:, 1])
    return np.column_stack([x, y])


def _curvature(points: np.ndarray, closed: bool) -> float:
    if len(points) < 3:
        return 0.0
    dense = np.vstack([points, points[:1]]) if closed else points
    x = dense[:, 0]
    y = dense[:, 1]
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denominator = np.power(dx * dx + dy * dy, 1.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.abs(dx * ddy - dy * ddx) / denominator
    return float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).max())


def _sample_label(labels: np.ndarray, point: np.ndarray) -> int:
    h, w = labels.shape
    col = int(round(point[0]))
    row = int(round(point[1]))
    if 0 <= row < h and 0 <= col < w:
        return int(labels[row, col])
    return 0


def adjacent_labels(
    labels: np.ndarray,
    points: np.ndarray,
    *,
    closed: bool,
    sample_distance: float = 3.0,
    sample_count: int = 24,
) -> tuple[int, int] | None:
    """Majority labels on the two sides of an ordered boundary curve."""
    dense = _resample(points, closed, 128)
    if len(dense) < 3:
        return None
    dx = np.roll(dense[:, 0], -1) - np.roll(dense[:, 0], 1)
    dy = np.roll(dense[:, 1], -1) - np.roll(dense[:, 1], 1)
    norm = np.hypot(dx, dy)
    norm[norm == 0] = 1.0
    normals = np.column_stack([-dy / norm, dx / norm])
    indices = np.linspace(0, len(dense) - 1, min(sample_count, len(dense))).astype(int)
    side_a: list[int] = []
    side_b: list[int] = []
    for index in indices:
        point = dense[index]
        normal = normals[index]
        side_a.append(_sample_label(labels, point + normal * sample_distance))
        side_b.append(_sample_label(labels, point - normal * sample_distance))

    def majority(values: list[int]) -> int:
        counts = np.bincount(values)
        return int(counts.argmax())

    return (majority(side_a), majority(side_b))


def boundary_curvature(points: np.ndarray, closed: bool) -> float:
    """Public max discrete curvature helper for the decision layer."""
    return _curvature(np.asarray(points, dtype=float), closed)


def refine_curve(
    curve: ExtractedCurve,
    labels: np.ndarray,
    *,
    config: BoundaryRefinementConfig | None = None,
) -> RefinedBoundary | None:
    """Refine one raw boundary path; return ``None`` if it is not a valid interface."""
    config = config or BoundaryRefinementConfig()
    points = np.asarray(curve.points, dtype=float)
    if len(points) < 2 or curve.length < config.min_length:
        return None

    pair = adjacent_labels(
        labels,
        points,
        closed=curve.closed,
        sample_distance=config.side_sample_distance,
    )
    if pair is None or pair[0] == pair[1]:
        return None
    if config.ignore_background and 0 in pair:
        return None

    n = max(64, len(points) * 4)
    dense = _resample(points, curve.closed, n)
    if curve.closed:
        dense = dense[:-1] if np.allclose(dense[0], dense[-1]) else dense
    u = np.linspace(0.0, 1.0, len(dense))
    weights = np.ones(len(dense))
    if not curve.closed:
        weights[0] = weights[-1] = config.endpoint_weight

    smoothing = float(config.smoothing)
    refined = dense
    max_curvature = _curvature(refined, curve.closed)
    for _ in range(8):
        try:
            tck, _ = splprep(
                [dense[:, 0], dense[:, 1]],
                u=u,
                w=weights,
                s=smoothing * len(dense),
                k=min(3, len(dense) - 1),
                per=curve.closed,
            )
            t = np.linspace(0.0, 1.0, config.samples)
            refined = np.column_stack(splev(t, tck))
            if not curve.closed:
                refined[0] = dense[0]
                refined[-1] = dense[-1]
        except Exception:
            return None
        max_curvature = _curvature(refined, curve.closed)
        if max_curvature <= config.max_curvature:
            break
        if smoothing >= config.smoothing * config.max_smoothing_scale:
            return None
        smoothing *= 2.0

    if max_curvature > config.max_curvature:
        return None
    length = float(
        np.hypot(*np.diff(np.vstack([refined, refined[:1]]) if curve.closed else refined, axis=0).T).sum()
    )
    return RefinedBoundary(
        points=refined,
        closed=curve.closed,
        length=length,
        labels=pair,
        max_curvature=max_curvature,
        raw_length=curve.length,
        smoothing=smoothing,
    )


def refine_curves(
    curves: list[ExtractedCurve],
    labels: np.ndarray,
    *,
    config: BoundaryRefinementConfig | None = None,
) -> list[RefinedBoundary]:
    config = config or BoundaryRefinementConfig()
    out: list[RefinedBoundary] = []
    for curve in curves:
        refined = refine_curve(curve, labels, config=config)
        if refined is not None:
            out.append(refined)
    return out


__all__ = [
    "BoundaryRefinementConfig",
    "RefinedBoundary",
    "adjacent_labels",
    "boundary_curvature",
    "refine_curve",
    "refine_curves",
]
