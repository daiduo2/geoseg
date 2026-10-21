"""Spline evaluation for interface curves.

The model stores control points in pixel coordinates ``(x, y)``.  This module
converts an :class:`~geoseg.modules.curve_editor.model.Interface` into a dense
polyline using ``splinebox`` (B-spline, Catmull-Rom, exponential bases), with a
polyline fallback when the dependency or point count is insufficient.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from geoseg.modules.curve_editor.model import Interface

BASIS_NAMES = ("B1", "B3", "CatmullRom", "Exponential")


def _basis(name: str, m: int):
    import splinebox

    return splinebox.basis_functions.basis_function_from_name(name, M=m)


def _as_array(interface: Interface) -> np.ndarray:
    return np.asarray([[p.x, p.y] for p in interface.points], dtype=float)


def _polyline(points: np.ndarray, closed: bool) -> np.ndarray:
    if len(points) < 2:
        return points.copy()
    if closed:
        return np.vstack([points, points[:1]])
    return points.copy()


def curve_from_interface(interface: Interface, samples: int | None = None) -> np.ndarray:
    """Return a dense ``(N, 2)`` polyline for an interface.

    ``interface.point_mode`` selects whether stored points are spline knots
    (interpolation, curve passes through them) or control points (B-spline
    approximation).  Degenerate cases fall back to a polyline.
    """
    points = _as_array(interface)
    if len(points) < 2:
        return points
    if len(points) < 4:
        return _polyline(points, interface.closed)

    try:
        import splinebox
    except ImportError:
        return _polyline(points, interface.closed)

    basis_name = interface.basis if interface.basis in BASIS_NAMES else "B3"
    m = len(points)
    try:
        basis = _basis(basis_name, m)
        if interface.point_mode == "control_points" and not interface.closed:
            m -= 2 * (math.ceil(basis.support / 2) - 1)
        if m < 2:
            return _polyline(points, interface.closed)
        basis = _basis(basis_name, m)
        if basis.support > m:
            return _polyline(points, interface.closed)
        spline = splinebox.spline_curves.Spline(
            M=m, basis_function=basis, closed=interface.closed
        )
        if interface.point_mode == "control_points":
            spline.control_points = points
        else:
            spline.knots = points
        n = samples or max(32, len(points) * 8)
        # splinebox uses [0, M-1] for open knot-interpolating splines and
        # [0, M] for closed ones (t=M wraps to t=0).
        t = np.linspace(
            0.0,
            float(m if interface.closed else m - 1),
            n,
            endpoint=not interface.closed,
        )
        curve = np.asarray(spline(t), dtype=float)
        if interface.closed:
            curve = np.vstack([curve, curve[:1]])
        return curve
    except Exception:
        return _polyline(points, interface.closed)


def resample_by_arc_length(
    interface: Interface, n: int, *, samples: int | None = None
) -> np.ndarray:
    """Resample the curve at ``n`` approximately equal arc-length positions."""
    curve = curve_from_interface(interface, samples=samples or max(256, n * 16))
    if len(curve) < 2:
        return curve
    diffs = np.diff(curve, axis=0)
    cumulative = np.concatenate([[0.0], np.cumsum(np.hypot(diffs[:, 0], diffs[:, 1]))])
    total = float(cumulative[-1])
    if total <= 0:
        return curve[:n]
    targets = np.linspace(0.0, total, n, endpoint=not interface.closed)
    x = np.interp(targets, cumulative, curve[:, 0])
    y = np.interp(targets, cumulative, curve[:, 1])
    return np.column_stack([x, y])


def fit_freehand(
    points: np.ndarray,
    *,
    n_points: int = 8,
    closed: bool = False,
    basis: str = "B3",
) -> list[tuple[float, float]]:
    """Reduce a freehand trace to ``n_points`` arc-length knots.

    The returned points are stored as ``knots`` so the spline interpolates the
    trace.  This is the deterministic, testable half of freehand drawing.
    """
    dense = np.asarray(points, dtype=float)
    if len(dense) < 2:
        return [tuple(map(float, p)) for p in dense]
    diffs = np.diff(dense, axis=0)
    cumulative = np.concatenate([[0.0], np.cumsum(np.hypot(diffs[:, 0], diffs[:, 1]))])
    total = float(cumulative[-1])
    if total <= 0:
        return [tuple(map(float, dense[0]))]
    targets = np.linspace(0.0, total, max(2, n_points), endpoint=not closed)
    x = np.interp(targets, cumulative, dense[:, 0])
    y = np.interp(targets, cumulative, dense[:, 1])
    return list(zip(x.tolist(), y.tolist()))


def curvature(interface: Interface, samples: int = 400) -> np.ndarray:
    """Discrete curvature ``|x'y'' - y'x''| / (x'^2 + y'^2)^1.5``."""
    curve = curve_from_interface(interface, samples=samples)
    if len(curve) < 3:
        return np.zeros(len(curve))
    x, y = curve[:, 0], curve[:, 1]
    dx, dy = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    denom = np.power(dx * dx + dy * dy, 1.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.abs(dx * ddy - dy * ddx) / denom
    return np.nan_to_num(k, nan=0.0, posinf=0.0, neginf=0.0)


def curvature_peaks(interface: Interface, *, samples: int = 400, min_distance: int = 12, prominence: float = 0.002) -> np.ndarray:
    """Indices of significant curvature peaks along the sampled curve."""
    from scipy.signal import find_peaks

    k = curvature(interface, samples=samples)
    if len(k) < 3:
        return np.array([], dtype=int)
    peaks, _ = find_peaks(k, distance=min_distance, prominence=prominence)
    return peaks


__all__ = [
    "BASIS_NAMES",
    "curve_from_interface",
    "curvature",
    "curvature_peaks",
    "fit_freehand",
    "resample_by_arc_length",
]
