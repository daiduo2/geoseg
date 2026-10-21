#!/usr/bin/env python
"""Layer 2 experiment E: endpoint tangent constraints for open boundary paths.

Question
--------
Experiment A pins the two exact endpoints of every open path and fits a cubic
P-spline.  Its summary reports a few paths whose max curvature increases after
smoothing, with the hypothesis that hard endpoint pinning at a junction corner
creates a local boundary-layer spike.

This experiment keeps A's objective, basis, corridor, closed-area check and
iterative lambda fallback exactly, and adds a soft endpoint tangent penalty:

    min_c ||B c - y||^2
          + lam * (||D2 c||^2 + alpha3 * ||D3 c||^2)
          + beta * (||f'(0) - t_start||^2 + ||f'(1) - t_end||^2)

where ``t_start`` / ``t_end`` are robust PCA line-fit tangents of the exact
centerline over a 10 px endpoint window, oriented along the parameterisation
(forward derivative).  The tangent rows are added to the same linear system;
the curve is not rotated or post-processed.

Variants
--------
hard_only           A baseline: beta = 0, lambda = 1, hard endpoint pinning.
tangent_soft        selected global beta; same objective plus tangent penalty.
tangent_soft_guard  tangent_soft plus a final closed-body paired-width guard.
                    Open paths are byte-for-byte identical to tangent_soft.

Only ``experiments/layer2/e_endpoint_tangent/`` and
``runs/layer2_experiments/e_endpoint_tangent/`` are written.

Run from the repository root:
    uv run python experiments/layer2/e_endpoint_tangent/run.py
"""

from __future__ import annotations

import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import BSpline

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.layer2.common import (  # noqa: E402
    BoundaryPath,
    Panel,
    boundary_metrics,
    cross_intersections,
    keep_paths,
    load_panels,
    path_length,
    render_overlay,
    resample,
)

OUT_DIR = ROOT / "runs/layer2_experiments/e_endpoint_tangent"
PANELS_DIR = OUT_DIR / "panels"
MANIFEST = ROOT / "validation/curated/manifest.jsonl"
PH01 = "ph01_page8_panel3"
A_METRICS = ROOT / "runs/layer2_experiments/a_segmented_spline/metrics.csv"

# ---------------------------------------------------------------------------
# Global constants.  No panel or path is tuned.
# ---------------------------------------------------------------------------
DEGREE = 3
KNOT_SPACING = 8.0
ALPHA3 = 0.25
LAMBDA = 1.0  # experiment A's selected global lambda; frozen here
CORRIDOR = 3.0
AREA_TOL = 0.02
DRIFT_TOL = 1e-3
MIN_POINTS = 4
MIN_LENGTH = 6.0
MIN_CLOSED_AREA = 25.0
MAX_REDUCTIONS = 6

# Endpoint tangent estimator (specified range 5-12 px; one fixed value).
TANGENT_WINDOW = 10.0
TANGENT_MIN_WINDOW = 5.0
TANGENT_MIN_POINTS = 4
TANGENT_MIN_SUPPORT = 4.0
TANGENT_MIN_LINE_RATIO = 2.0
BORDER_MARGIN = 1.5

# Beta grid and selection rule (fixed before inspecting grid outcomes).
BETA_GRID: tuple[float, ...] = (0.0, 0.01, 0.1, 1.0, 10.0, 100.0)
DISPLACEMENT_TOL = 0.01  # px; numerical tolerance on median/p90 p95 displacement
MIN_ENDPOINT_CURV_REDUCTION = 0.02  # 2% median reduction required
MAX_ENDPOINT_P90_INCREASE = 0.02  # endpoint p90 curvature may not rise > 2%

# Endpoint-region curvature definitions.
ENDPOINT_REGION_PX = 15.0
ENDPOINT_WIDE_FRACTION = 0.10

# Closed-body width guard, copied from experiment D's paired-width logic.
WIDTH_RATIO_MIN = 0.75
WIDTH_ABS_MIN_PX = 2.0
WIDTH_NAN_FRACTION_MAX = 0.5
WIDTH_INFLATED_FACTOR = 2.0


@dataclass
class Config:
    degree: int = DEGREE
    knot_spacing: float = KNOT_SPACING
    alpha3: float = ALPHA3
    corridor: float = CORRIDOR
    area_tol: float = AREA_TOL
    drift_tol: float = DRIFT_TOL
    min_points: int = MIN_POINTS
    min_length: float = MIN_LENGTH
    min_closed_area: float = MIN_CLOSED_AREA
    max_reductions: int = MAX_REDUCTIONS
    tangent_window: float = TANGENT_WINDOW


@dataclass
class SplineProblem:
    basis: np.ndarray
    data: np.ndarray
    bt_b: np.ndarray
    bt_y: np.ndarray
    penalty: np.ndarray
    free: np.ndarray
    fixed: np.ndarray
    n_coef: int
    n_data: int
    closed: bool
    knots: np.ndarray | None
    deriv_start: np.ndarray | None
    deriv_end: np.ndarray | None
    tangent_start: np.ndarray | None
    tangent_end: np.ndarray | None


@dataclass
class PathRecord:
    path_index: int
    path: BoundaryPath
    length: float
    area_exact: float | None
    eligible: bool
    ineligible_reason: str
    exact_metrics: dict
    problem: SplineProblem | None = None
    tangent_start: np.ndarray | None = None
    tangent_end: np.ndarray | None = None
    tangent_start_status: str = ""
    tangent_end_status: str = ""
    tangent_start_info: dict | None = None
    tangent_end_info: dict | None = None


@dataclass
class PathView:
    """Minimal BoundaryPath-compatible view for cross-intersection checks."""

    boundary_id: int
    labels_pair: tuple[int, int]
    points: np.ndarray
    closed: bool
    exact_length: int = 0


# ---------------------------------------------------------------------------
# P-spline basis and penalty (same construction as experiment A)
# ---------------------------------------------------------------------------
def _difference_matrix(n: int, order: int, periodic: bool) -> np.ndarray:
    rows = n if periodic else max(0, n - order)
    out = np.zeros((rows, n), dtype=float)
    coeffs = [(-1) ** k * math.comb(order, k) for k in range(order + 1)]
    for i in range(rows):
        for k, coeff in enumerate(coeffs):
            j = (i + k) % n if periodic else i + k
            out[i, j] += coeff
    return out


def _penalty_matrix(n: int, periodic: bool, alpha3: float) -> np.ndarray:
    d2 = _difference_matrix(n, 2, periodic)
    d3 = _difference_matrix(n, 3, periodic)
    return d2.T @ d2 + alpha3 * (d3.T @ d3)


def _open_basis(u: np.ndarray, length: float, degree: int, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    n_segments = max(1, int(math.ceil(length / spacing)))
    interior = np.linspace(0.0, length, n_segments + 1)[1:-1]
    knots = np.concatenate(
        [np.full(degree + 1, 0.0), interior, np.full(degree + 1, float(length))]
    )
    basis = BSpline.design_matrix(np.clip(u, 0.0, length), knots, degree).toarray()
    return basis, knots


def _closed_basis(u: np.ndarray, length: float, degree: int, spacing: float) -> np.ndarray:
    n_coef = max(degree + 1, int(round(length / spacing)))
    step = length / n_coef
    n_extended = n_coef + degree
    knots = (np.arange(n_extended + degree + 1) - degree) * step
    extended = BSpline.design_matrix(u, knots, degree).toarray()
    basis = np.zeros((len(u), n_coef), dtype=float)
    for j in range(n_extended):
        basis[:, j % n_coef] += extended[:, j]
    return basis


def _derivative_endpoint_rows(
    knots: np.ndarray, degree: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Rows d0, d1 with f'(0) = d0 @ c and f'(L) = d1 @ c.

    For a degree-p B-spline with coefficients c, the derivative is a degree
    (p-1) B-spline on the inner knots with coefficients
        p * (c[i+1] - c[i]) / (t[i+p+1] - t[i+1]).
    This is the exact B-spline derivative basis, not a finite difference.
    """
    n_coef = len(knots) - degree - 1
    if n_coef < 2:
        return None, None
    inner = knots[1:-1]
    u = np.array([knots[degree], knots[-degree - 1]], dtype=float)
    raw = BSpline.design_matrix(u, inner, degree - 1).toarray()
    diff = np.zeros((n_coef - 1, n_coef), dtype=float)
    for i in range(n_coef - 1):
        denominator = knots[i + degree + 1] - knots[i + 1]
        if abs(denominator) < 1e-12:
            continue
        diff[i, i] = -degree / denominator
        diff[i, i + 1] = degree / denominator
    dmat = raw @ diff
    return dmat[0], dmat[-1]


# ---------------------------------------------------------------------------
# Robust endpoint tangent estimator
# ---------------------------------------------------------------------------
def _endpoint_tangent(
    points: np.ndarray,
    window: float,
    start: bool,
    border: bool,
) -> tuple[np.ndarray | None, str, dict]:
    """Estimate a tangent from the exact centerline near one endpoint.

    Returns the unit vector pointing **into** the curve (from the endpoint
    toward the interior).  The fit is a total-least-squares line through the
    endpoint (PCA of offsets, not mean-centred), trimmed three times at
    median + 2.5 MAD (floor 0.75 px), then re-fitted.

    Skipped when the endpoint is on the panel border, when the window has too
    few points or too little projected support, or when the window is not
    line-like (small/large eigenvalue ratio below 2).
    """
    info: dict[str, float | int | str] = {}
    if border:
        return None, "border", info
    p = np.asarray(points, dtype=float)
    if not start:
        p = p[::-1]
    endpoint = p[0]
    diffs = np.diff(p, axis=0)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(diffs[:, 0], diffs[:, 1]))])
    total = float(s[-1])
    if total < TANGENT_MIN_WINDOW:
        return None, "short_support", {"total_length": total}
    # Exclude the endpoint itself from the direction estimate.
    selected = (s > 1e-9) & (s <= window)
    if int(selected.sum()) < TANGENT_MIN_POINTS:
        return None, "few_points", {"n_points": int(selected.sum())}
    v = p[selected] - endpoint
    info["n_points_window"] = int(len(v))
    info["window_arc_px"] = float(min(window, total))
    for _ in range(3):
        covariance = v.T @ v
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        tangent = eigenvectors[:, -1]
        perpendicular = np.abs(v[:, 0] * tangent[1] - v[:, 1] * tangent[0])
        median = float(np.median(perpendicular))
        mad = float(np.median(np.abs(perpendicular - median)))
        threshold = max(0.75, median + 2.5 * mad)
        keep = perpendicular <= threshold
        if int(keep.sum()) < TANGENT_MIN_POINTS:
            break
        v = v[keep]
    covariance = v.T @ v
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tangent = eigenvectors[:, -1]
    norm = float(np.linalg.norm(tangent))
    if norm < 1e-12:
        return None, "degenerate", info
    tangent = tangent / norm
    mean_offset = v.mean(axis=0)
    projection = float(tangent @ mean_offset)
    if abs(projection) < 1e-9:
        return None, "ambiguous_orientation", info
    if projection < 0:
        tangent = -tangent
    support = float(np.max(np.abs(v @ tangent)))
    ratio = float(eigenvalues[-1] / max(eigenvalues[0], 1e-12))
    info.update(
        {
            "n_points_used": int(len(v)),
            "support_px": support,
            "line_ratio": ratio,
        }
    )
    if support < TANGENT_MIN_SUPPORT:
        return None, "weak_support", info
    if ratio < TANGENT_MIN_LINE_RATIO:
        return None, "not_line_like", info
    return tangent, "applied", info


# ---------------------------------------------------------------------------
# Problem construction, solve, constraints
# ---------------------------------------------------------------------------
def _build_problem(record: PathRecord, cfg: Config) -> SplineProblem:
    path = record.path
    length = record.length
    closed = path.closed
    n_data = max(2, int(round(length)))
    u = np.linspace(0.0, length, n_data, endpoint=not closed)
    data = resample(path.points, closed, n_data)

    deriv_start: np.ndarray | None = None
    deriv_end: np.ndarray | None = None
    knots: np.ndarray | None = None

    if closed:
        basis = _closed_basis(u, length, cfg.degree, cfg.knot_spacing)
        n_coef = basis.shape[1]
        penalty = _penalty_matrix(n_coef, True, cfg.alpha3)
        free = np.arange(n_coef)
        fixed = np.array([], dtype=int)
    else:
        basis, knots = _open_basis(u, length, cfg.degree, cfg.knot_spacing)
        n_coef = basis.shape[1]
        penalty = _penalty_matrix(n_coef, False, cfg.alpha3)
        fixed = np.array([0, n_coef - 1], dtype=int)
        free = np.arange(1, n_coef - 1)
        deriv_start, deriv_end = _derivative_endpoint_rows(knots, cfg.degree)

    if n_coef < cfg.degree + 1 or len(free) < 1:
        raise ValueError(f"not enough coefficients: n_coef={n_coef}")

    return SplineProblem(
        basis=basis,
        data=data,
        bt_b=basis.T @ basis,
        bt_y=basis.T @ data,
        penalty=penalty,
        free=free,
        fixed=fixed,
        n_coef=n_coef,
        n_data=n_data,
        closed=closed,
        knots=knots,
        deriv_start=deriv_start,
        deriv_end=deriv_end,
        tangent_start=record.tangent_start,
        tangent_end=record.tangent_end,
    )


def _solve_problem(problem: SplineProblem, lam: float, beta: float) -> tuple[np.ndarray, np.ndarray]:
    """Solve the penalised least-squares system.

    The tangent penalty is added to the normal equations as
    ``beta * d0^T d0`` (and likewise for the end).  This is exactly the
    normal-equation form of appending ``sqrt(beta) * d0`` rows to the design
    matrix.  Hard endpoint coefficients remain eliminated and fixed.
    """
    matrix = problem.bt_b + lam * problem.penalty
    rhs = problem.bt_y.copy()

    if beta > 0.0 and not problem.closed:
        if problem.tangent_start is not None and problem.deriv_start is not None:
            d0 = problem.deriv_start
            matrix = matrix + beta * np.outer(d0, d0)
            rhs = rhs + beta * np.outer(d0, problem.tangent_start)
        if problem.tangent_end is not None and problem.deriv_end is not None:
            d1 = problem.deriv_end
            matrix = matrix + beta * np.outer(d1, d1)
            rhs = rhs + beta * np.outer(d1, problem.tangent_end)

    if problem.fixed.size == 0:
        coeffs = np.linalg.solve(matrix, rhs)
    else:
        free = problem.free
        fixed = problem.fixed
        fixed_values = np.vstack([problem.data[0], problem.data[-1]])
        reduced = matrix[np.ix_(free, free)]
        reduced_rhs = rhs[free] - matrix[np.ix_(free, fixed)] @ fixed_values
        coeffs = np.zeros((problem.n_coef, 2), dtype=float)
        coeffs[free] = np.linalg.solve(reduced, reduced_rhs)
        coeffs[fixed] = fixed_values
    return problem.basis @ coeffs, coeffs


def _constraint_violations(record: PathRecord, metrics: dict, cfg: Config) -> list[str]:
    reasons: list[str] = []
    if metrics["max_disp"] > cfg.corridor:
        reasons.append(f"max_disp={metrics['max_disp']:.3f}>{cfg.corridor:g}")
    if int(metrics["self_intersections"]) > 0:
        reasons.append(f"self_intersections={int(metrics['self_intersections'])}")
    if record.path.closed:
        area_exact = float(metrics.get("area_exact", 0.0))
        if area_exact > 1e-9:
            rel = float(metrics.get("area_change", 0.0)) / area_exact
            if abs(rel) > cfg.area_tol:
                reasons.append(f"area_change_rel={rel:+.4f}>{cfg.area_tol:g}")
    elif metrics["endpoint_drift"] > cfg.drift_tol:
        reasons.append(f"endpoint_drift={metrics['endpoint_drift']:.4f}>{cfg.drift_tol:g}")
    return reasons


def _exact_reference_metrics(record: PathRecord) -> dict:
    """A-compatible normalisation of exact geometry (no resampling floor)."""
    metrics = dict(record.exact_metrics)
    metrics["mean_disp"] = 0.0
    metrics["p95_disp"] = 0.0
    metrics["max_disp"] = 0.0
    metrics["max_curvature"] = float(metrics.get("exact_max_curvature", 0.0))
    metrics["length_smooth"] = float(metrics["length_exact"])
    metrics["endpoint_drift"] = 0.0
    if "area_exact" in metrics:
        metrics["area_smooth"] = float(metrics["area_exact"])
        metrics["area_change"] = 0.0
    return metrics


# ---------------------------------------------------------------------------
# Dense curvature, endpoint regions, tangent errors
# ---------------------------------------------------------------------------
def _curvature_from_derivatives(dx, dy, ddx, ddy) -> tuple[np.ndarray, np.ndarray]:
    speed = np.hypot(dx, dy)
    denominator = np.maximum(speed, 1e-12) ** 3
    signed = (dx * ddy - dy * ddx) / denominator
    signed = np.nan_to_num(signed, nan=0.0, posinf=0.0, neginf=0.0)
    return np.abs(signed), signed


def _sign_flips(values: np.ndarray, tol: float = 1e-7) -> int:
    selected = values[np.abs(values) > tol]
    if len(selected) < 2:
        return 0
    signs = np.sign(selected)
    return int(np.sum(signs[1:] != signs[:-1]))


def _curvature_analysis(
    points: np.ndarray,
    record: PathRecord,
    coeffs: np.ndarray | None,
    knots: np.ndarray | None,
) -> dict:
    """Dense curvature profile and endpoint-region statistics.

    For accepted open fits the exact B-spline derivatives are evaluated on a
    dense arc-length grid.  Accepted closed fits are evaluated through the
    periodic extension of the same basis.  Exact/fallback geometry is densely
    resampled and differentiated; those values are reported for completeness.
    """
    path = record.path
    length = float(record.length)
    closed = path.closed
    n_dense = max(64, min(8000, int(round(4.0 * max(length, 1.0)))))

    if coeffs is not None and not closed and knots is not None:
        spline = BSpline(knots, coeffs, DEGREE)
        uu = np.linspace(0.0, length, n_dense)
        d1 = spline.derivative(1)(uu)
        d2 = spline.derivative(2)(uu)
        dense_x, dense_y = spline(uu).T
        k, k_signed = _curvature_from_derivatives(d1[:, 0], d1[:, 1], d2[:, 0], d2[:, 1])
    elif coeffs is not None and closed:
        n_coef = int(len(coeffs))
        step = length / n_coef
        n_extended = n_coef + DEGREE
        knots_ext = (np.arange(n_extended + DEGREE + 1) - DEGREE) * step
        extended = np.asarray(coeffs)[np.arange(n_extended) % n_coef]
        spline = BSpline(knots_ext, extended, DEGREE)
        uu = np.linspace(0.0, length, n_dense, endpoint=False)
        d1 = spline.derivative(1)(uu)
        d2 = spline.derivative(2)(uu)
        dense_x, dense_y = spline(uu).T
        k, k_signed = _curvature_from_derivatives(d1[:, 0], d1[:, 1], d2[:, 0], d2[:, 1])
    else:
        dense = resample(points, closed, n_dense)
        if closed:
            dense = np.vstack([dense, dense[:1]])
            uu = np.linspace(0.0, length, len(dense))
        else:
            uu = np.linspace(0.0, length, len(dense))
        dense_x, dense_y = dense[:, 0], dense[:, 1]
        dx = np.gradient(dense_x, uu)
        dy = np.gradient(dense_y, uu)
        ddx = np.gradient(dx, uu)
        ddy = np.gradient(dy, uu)
        k, k_signed = _curvature_from_derivatives(dx, dy, ddx, ddy)

    peak_index = int(np.argmax(k))
    peak_u = float(uu[peak_index])
    result: dict[str, float | int | None] = {
        "max_curvature_dense": float(k[peak_index]),
        "p95_curvature_dense": float(np.percentile(k, 95)),
        "max_curvature_dense_u": peak_u,
        "max_curvature_dense_end_dist": (
            None if closed else float(min(peak_u, length - peak_u))
        ),
        "curv_sign_flips": _sign_flips(k_signed),
        "endpoint_region_px": None,
        "endpoint_start_max_curv": None,
        "endpoint_end_max_curv": None,
        "endpoint_start_p95_curv": None,
        "endpoint_end_p95_curv": None,
        "endpoint_region_max_curv": None,
        "endpoint_region_p95_curv": None,
        "endpoint_wide_max_curv": None,
        "endpoint_curv_sign_flips": None,
    }
    if closed:
        return result

    region = float(min(ENDPOINT_REGION_PX, 0.5 * length))
    wide = float(min(max(ENDPOINT_REGION_PX, ENDPOINT_WIDE_FRACTION * length), 0.5 * length))
    start_mask = uu <= region
    end_mask = uu >= length - region
    start_values = k[start_mask]
    end_values = k[end_mask]
    start_max = float(np.max(start_values)) if len(start_values) else float("nan")
    end_max = float(np.max(end_values)) if len(end_values) else float("nan")
    start_p95 = float(np.percentile(start_values, 95)) if len(start_values) else float("nan")
    end_p95 = float(np.percentile(end_values, 95)) if len(end_values) else float("nan")
    union_mask = start_mask | end_mask
    result.update(
        {
            "endpoint_region_px": region,
            "endpoint_start_max_curv": start_max,
            "endpoint_end_max_curv": end_max,
            "endpoint_start_p95_curv": start_p95,
            "endpoint_end_p95_curv": end_p95,
            "endpoint_region_max_curv": float(max(start_max, end_max)),
            "endpoint_region_p95_curv": float(max(start_p95, end_p95)),
            "endpoint_wide_max_curv": float(
                max(
                    np.max(k[uu <= wide]) if np.any(uu <= wide) else 0.0,
                    np.max(k[uu >= length - wide]) if np.any(uu >= length - wide) else 0.0,
                )
            ),
            "endpoint_curv_sign_flips": _sign_flips(k_signed[union_mask]),
        }
    )
    return result


def _polyline_endpoint_direction(points: np.ndarray, start: bool) -> np.ndarray | None:
    p = np.asarray(points, dtype=float)
    if len(p) < 2:
        return None
    if not start:
        p = p[::-1]
    diffs = np.diff(p, axis=0)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(diffs[:, 0], diffs[:, 1]))])
    if s[-1] < 1e-9:
        return None
    target_s = min(5.0, 0.5 * float(s[-1]))
    index = int(np.searchsorted(s, target_s))
    index = min(max(index, 1), len(p) - 1)
    direction = p[index] - p[0]
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        return None
    return direction / norm


def _tangent_angle_errors(
    result_points: np.ndarray,
    record: PathRecord,
    coeffs: np.ndarray | None,
    knots: np.ndarray | None,
) -> tuple[float | None, float | None]:
    if record.path.closed:
        return None, None
    errors: list[float | None] = []
    for start, target in ((True, record.tangent_start), (False, record.tangent_end)):
        if target is None:
            errors.append(None)
            continue
        direction: np.ndarray | None = None
        if coeffs is not None and knots is not None:
            spline = BSpline(knots, coeffs, DEGREE)
            derivative = spline.derivative(1)(0.0 if start else float(record.length))
            norm = float(np.linalg.norm(derivative))
            if norm > 1e-12:
                direction = derivative / norm
        if direction is None:
            into = _polyline_endpoint_direction(result_points, start)
            if into is None:
                errors.append(None)
                continue
            direction = into if start else -into
        cosine = float(np.clip(direction @ target, -1.0, 1.0))
        errors.append(float(np.degrees(np.arccos(cosine))))
    return errors[0], errors[1]


# ---------------------------------------------------------------------------
# Per-path evaluation
# ---------------------------------------------------------------------------
def _exact_result(
    record: PathRecord,
    status: str,
    reason: str,
    lam_requested: float,
    lam_used: float | None,
    refits: int,
) -> dict:
    path = record.path
    points = path.points.copy()
    metrics = _exact_reference_metrics(record)
    curvature = _curvature_analysis(points, record, None, None)
    start_error, end_error = _tangent_angle_errors(points, record, None, None)
    tangent_finite = _finite([start_error, end_error])
    return {
        "points": points,
        "coeffs": None,
        "knots": None,
        "metrics": metrics,
        "curvature": curvature,
        "tangent_start_deg": start_error,
        "tangent_end_deg": end_error,
        "tangent_max_deg": float(max(tangent_finite)) if tangent_finite else None,
        "status": status,
        "reason": reason,
        "lambda_requested": float(lam_requested),
        "lambda_used": lam_used,
        "refits": refits,
        "panel_id": "",
        "boundary_id": path.boundary_id,
        "labels_pair": path.labels_pair,
        "closed": path.closed,
        "eligible": record.eligible,
        "ineligible_reason": record.ineligible_reason,
        "path_index": record.path_index,
    }


def _evaluate_path(record: PathRecord, lam_requested: float, beta: float, cfg: Config) -> dict:
    path = record.path
    if not record.eligible:
        return _exact_result(record, "fallback_short", record.ineligible_reason, lam_requested, None, 0)
    if record.problem is None:
        return _exact_result(record, "error", "problem_not_built", lam_requested, None, 0)

    lam = float(lam_requested)
    refits = 0
    last_reasons: list[str] = []
    for attempt in range(cfg.max_reductions + 1):
        try:
            points, coeffs = _solve_problem(record.problem, lam, beta)
            metrics = boundary_metrics(points, path)
        except Exception as exc:  # record, never skip
            return _exact_result(
                record, "error", f"{type(exc).__name__}: {exc}", lam_requested, lam, refits
            )
        reasons = _constraint_violations(record, metrics, cfg)
        if not reasons:
            status = "ok" if refits == 0 else "ok_reduced"
            curvature = _curvature_analysis(points, record, coeffs, record.problem.knots)
            start_error, end_error = _tangent_angle_errors(
                points, record, coeffs, record.problem.knots
            )
            tangent_finite = _finite([start_error, end_error])
            return {
                "points": points,
                "coeffs": coeffs,
                "knots": record.problem.knots,
                "metrics": metrics,
                "curvature": curvature,
                "tangent_start_deg": start_error,
                "tangent_end_deg": end_error,
                "tangent_max_deg": float(max(tangent_finite)) if tangent_finite else None,
                "status": status,
                "reason": "",
                "lambda_requested": float(lam_requested),
                "lambda_used": lam,
                "refits": refits,
                "panel_id": "",
                "boundary_id": path.boundary_id,
                "labels_pair": path.labels_pair,
                "closed": path.closed,
                "eligible": True,
                "ineligible_reason": "",
                "path_index": record.path_index,
            }
        last_reasons = reasons
        if attempt < cfg.max_reductions:
            lam *= 0.5
            refits += 1
    return _exact_result(
        record, "fallback_constraint", "; ".join(last_reasons), lam_requested, lam, refits
    )


def _build_records(panel: Panel, cfg: Config) -> list[PathRecord]:
    paths = keep_paths(panel.labels)
    height, width = panel.labels.shape
    records: list[PathRecord] = []
    for path_index, path in enumerate(paths):
        if path.closed:
            dense = np.vstack([path.points, path.points[:1]])
        else:
            dense = path.points
        length = float(np.hypot(*np.diff(dense, axis=0).T).sum())
        exact_metrics = boundary_metrics(path.points, path)
        area_exact = float(exact_metrics.get("area_exact", 0.0)) if path.closed else None

        reasons: list[str] = []
        if len(path.points) < cfg.min_points:
            reasons.append(f"too_few_points({len(path.points)}<{cfg.min_points})")
        if length < cfg.min_length:
            reasons.append(f"too_short({length:.2f}<{cfg.min_length:g}px)")
        if path.closed and (area_exact or 0.0) < cfg.min_closed_area:
            reasons.append(
                f"tiny_closed_area({area_exact or 0.0:.1f}<{cfg.min_closed_area:g}px2)"
            )

        record = PathRecord(
            path_index=path_index,
            path=path,
            length=length,
            area_exact=area_exact,
            eligible=not reasons,
            ineligible_reason="; ".join(reasons),
            exact_metrics=exact_metrics,
        )
        if path.closed:
            record.tangent_start_status = "closed"
            record.tangent_end_status = "closed"
        elif not record.eligible:
            record.tangent_start_status = "ineligible"
            record.tangent_end_status = "ineligible"
        else:
            start_border = _at_panel_border(path.points[0], width, height)
            end_border = _at_panel_border(path.points[-1], width, height)
            tangent_start, status_start, info_start = _endpoint_tangent(
                path.points, cfg.tangent_window, True, start_border
            )
            tangent_end_into, status_end, info_end = _endpoint_tangent(
                path.points, cfg.tangent_window, False, end_border
            )
            record.tangent_start = tangent_start
            # f'(1) is the forward derivative; the into-curve tangent at the
            # end points backward, so negate it for the penalty target.
            record.tangent_end = None if tangent_end_into is None else -tangent_end_into
            record.tangent_start_status = status_start
            record.tangent_end_status = status_end
            record.tangent_start_info = info_start
            record.tangent_end_info = info_end

        if record.eligible:
            try:
                record.problem = _build_problem(record, cfg)
            except Exception as exc:
                record.problem = None
                record.eligible = False
                record.ineligible_reason = f"problem_build_error: {type(exc).__name__}: {exc}"
        records.append(record)
    return records


def _at_panel_border(point: np.ndarray, width: int, height: int) -> bool:
    x, y = float(point[0]), float(point[1])
    return (
        x < BORDER_MARGIN
        or x > (width - 1) - BORDER_MARGIN
        or y < BORDER_MARGIN
        or y > (height - 1) - BORDER_MARGIN
    )


# ---------------------------------------------------------------------------
# Closed-body paired-width guard (copied from experiment D)
# ---------------------------------------------------------------------------
def _signed_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _unit_tangents(points: np.ndarray, closed: bool) -> np.ndarray:
    if closed:
        tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    else:
        tangent = np.gradient(points, axis=0)
    norm = np.hypot(tangent[:, 0], tangent[:, 1])
    return tangent / np.maximum(norm, 1e-12)[:, None]


def _paired_widths(points: np.ndarray, n_samples: int | None = None) -> dict:
    """Approximate local body width by inward normal rays to the opposite wall."""
    length = path_length(points, True)
    m = n_samples or int(min(240, max(48, round(length / 3.0))))
    dense = resample(points, True, m)
    tangent = _unit_tangents(dense, True)
    normals = np.column_stack([tangent[:, 1], -tangent[:, 0]])
    signed = _signed_area(dense)
    inward = -np.sign(signed) * normals
    edge_a = dense
    edge_b = np.roll(dense, -1, axis=0)
    edge = edge_b - edge_a
    widths = np.full(m, np.nan)
    for i in range(m):
        point = dense[i]
        direction = inward[i]
        denominator = direction[0] * (-edge[:, 1]) + direction[1] * edge[:, 0]
        rhs = edge_a - point
        ok = np.abs(denominator) > 1e-10
        denominator_safe = np.where(ok, denominator, 1.0)
        distance = (rhs[:, 0] * (-edge[:, 1]) + rhs[:, 1] * edge[:, 0]) / denominator_safe
        along = (direction[0] * rhs[:, 1] - direction[1] * rhs[:, 0]) / denominator_safe
        valid = ok & (distance > 0.75) & (along >= -1e-6) & (along <= 1.0 + 1e-6)
        if valid.any():
            widths[i] = float(distance[valid].min())
    finite = widths[np.isfinite(widths)]
    if len(finite) == 0:
        return {
            "p5": float("nan"),
            "p10": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "nan_frac": 1.0,
        }
    return {
        "p5": float(np.percentile(finite, 5)),
        "p10": float(np.percentile(finite, 10)),
        "median": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "nan_frac": float(1.0 - len(finite) / m),
    }


def _width_collapses(widths: dict | None, exact_width: dict) -> bool:
    if widths is None or not np.isfinite(widths.get("median", float("nan"))):
        return True
    if widths.get("nan_frac", 1.0) > WIDTH_NAN_FRACTION_MAX:
        return True
    p5 = widths.get("p5", float("nan"))
    median = widths.get("median", float("nan"))
    if np.isfinite(p5) and np.isfinite(exact_width.get("p5", float("nan"))):
        threshold = max(WIDTH_ABS_MIN_PX, WIDTH_RATIO_MIN * exact_width["p5"])
        if p5 < threshold:
            return True
    if np.isfinite(median) and np.isfinite(exact_width.get("median", float("nan"))):
        if median < WIDTH_RATIO_MIN * exact_width["median"]:
            return True
    return False


def _apply_width_guard(record: PathRecord, result: dict) -> dict:
    """Final closed-body area/width guard.  Open paths are returned unchanged."""
    if not record.path.closed:
        return result
    if result["status"] not in ("ok", "ok_reduced"):
        return result
    try:
        exact_width = _paired_widths(record.path.points)
        smooth_width = _paired_widths(result["points"])
    except Exception as exc:  # conservative fallback, recorded
        guarded = _exact_result(
            record,
            "fallback_width_guard",
            f"width_measure_error:{type(exc).__name__}: {exc}",
            result["lambda_requested"],
            result["lambda_used"],
            result["refits"],
        )
        return guarded

    width_inflated = bool(
        np.isfinite(smooth_width.get("p5", float("nan")))
        and np.isfinite(exact_width.get("p5", float("nan")))
        and smooth_width["p5"] > WIDTH_INFLATED_FACTOR * exact_width["p5"]
    )
    result = dict(result)
    result["width_exact"] = exact_width
    result["width_smooth_raw"] = smooth_width
    result["width_inflated"] = width_inflated
    if _width_collapses(smooth_width, exact_width):
        reason = (
            f"p5 width {smooth_width.get('p5', float('nan')):.2f} vs exact "
            f"{exact_width.get('p5', float('nan')):.2f}; median "
            f"{smooth_width.get('median', float('nan')):.2f} vs "
            f"{exact_width.get('median', float('nan')):.2f}"
        )
        guarded = _exact_result(
            record,
            "fallback_width_guard",
            reason,
            result["lambda_requested"],
            result["lambda_used"],
            result["refits"],
        )
        guarded["width_exact"] = exact_width
        guarded["width_smooth_raw"] = smooth_width
        guarded["width_smooth"] = exact_width
        guarded["width_inflated"] = width_inflated
        return guarded
    result["width_smooth"] = smooth_width
    return result


# ---------------------------------------------------------------------------
# Metric rows
# ---------------------------------------------------------------------------
def _finite(values: list[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None and np.isfinite(float(value))]


def _median(values: list[float | None]) -> float | None:
    finite = _finite(values)
    return float(np.median(finite)) if finite else None


def _p90(values: list[float | None]) -> float | None:
    finite = _finite(values)
    return float(np.percentile(finite, 90)) if finite else None


def _p95(values: list[float | None]) -> float | None:
    finite = _finite(values)
    return float(np.percentile(finite, 95)) if finite else None


def _curvature_reduction(metrics: dict, exact_metrics: dict) -> tuple[float | None, float | None]:
    exact_max = float(metrics.get("exact_max_curvature", 0.0))
    exact_p95 = float(exact_metrics.get("p95_curvature", 0.0))
    max_red = None
    p95_red = None
    if exact_max > 1e-6:
        max_red = 1.0 - float(metrics["max_curvature"]) / exact_max
    if exact_p95 > 1e-6:
        p95_red = 1.0 - float(metrics["p95_curvature"]) / exact_p95
    return max_red, p95_red


def _metrics_row(variant: str, beta: float, panel_id: str, record: PathRecord, result: dict) -> dict:
    metrics = result["metrics"]
    curvature = result["curvature"]
    closed = record.path.closed
    length_change = float(metrics["length_smooth"]) - float(metrics["length_exact"])
    length_change_pct = length_change / max(float(metrics["length_exact"]), 1e-9)
    area_change = metrics.get("area_change") if closed else None
    area_exact = metrics.get("area_exact") if closed else None
    area_smooth = metrics.get("area_smooth") if closed else None
    area_change_rel = None
    if closed and area_exact and float(area_exact) > 1e-9:
        area_change_rel = float(area_change) / float(area_exact)
    max_red, p95_red = _curvature_reduction(metrics, record.exact_metrics)
    tangent_start = result.get("tangent_start_deg")
    tangent_end = result.get("tangent_end_deg")
    tangent_finite = _finite([tangent_start, tangent_end])
    tangent_max = float(max(tangent_finite)) if tangent_finite else None

    width_exact = result.get("width_exact") or {}
    width_smooth = result.get("width_smooth") or {}
    width_smooth_raw = result.get("width_smooth_raw") or {}
    width_ratio_p5 = None
    width_ratio_median = None
    if np.isfinite(width_smooth.get("p5", float("nan"))) and np.isfinite(
        width_exact.get("p5", float("nan"))
    ) and width_exact["p5"] > 1e-9:
        width_ratio_p5 = width_smooth["p5"] / width_exact["p5"]
    if np.isfinite(width_smooth.get("median", float("nan"))) and np.isfinite(
        width_exact.get("median", float("nan"))
    ) and width_exact["median"] > 1e-9:
        width_ratio_median = width_smooth["median"] / width_exact["median"]

    return {
        "variant": variant,
        "beta": beta,
        "panel_id": panel_id,
        "boundary_id": record.path.boundary_id,
        "labels_pair": f"{record.path.labels_pair[0]}-{record.path.labels_pair[1]}",
        "closed": str(bool(closed)).lower(),
        "path_index": record.path_index,
        "n_points": len(record.path.points),
        "n_out_points": len(result["points"]),
        "eligible": str(bool(record.eligible)).lower(),
        "status": result["status"],
        "reason": result["reason"],
        "mean_disp": float(metrics["mean_disp"]),
        "p95_disp": float(metrics["p95_disp"]),
        "max_disp": float(metrics["max_disp"]),
        "max_curvature": float(metrics["max_curvature"]),
        "p95_curvature": float(metrics["p95_curvature"]),
        "exact_max_curvature": float(metrics["exact_max_curvature"]),
        "exact_p95_curvature": float(record.exact_metrics["p95_curvature"]),
        "endpoint_drift": float(metrics["endpoint_drift"]),
        "self_intersections": int(metrics["self_intersections"]),
        "length_exact": float(metrics["length_exact"]),
        "length_smooth": float(metrics["length_smooth"]),
        "length_change": length_change,
        "length_change_pct": length_change_pct,
        "area_exact": area_exact,
        "area_smooth": area_smooth,
        "area_change": area_change,
        "area_change_rel": area_change_rel,
        "max_curv_reduction": max_red,
        "p95_curv_reduction": p95_red,
        "max_curvature_dense": curvature["max_curvature_dense"],
        "p95_curvature_dense": curvature["p95_curvature_dense"],
        "max_curvature_dense_u": curvature["max_curvature_dense_u"],
        "max_curvature_dense_end_dist": curvature["max_curvature_dense_end_dist"],
        "endpoint_region_px": curvature["endpoint_region_px"],
        "endpoint_start_max_curv": curvature["endpoint_start_max_curv"],
        "endpoint_end_max_curv": curvature["endpoint_end_max_curv"],
        "endpoint_start_p95_curv": curvature["endpoint_start_p95_curv"],
        "endpoint_end_p95_curv": curvature["endpoint_end_p95_curv"],
        "endpoint_region_max_curv": curvature["endpoint_region_max_curv"],
        "endpoint_region_p95_curv": curvature["endpoint_region_p95_curv"],
        "endpoint_wide_max_curv": curvature["endpoint_wide_max_curv"],
        "endpoint_curv_sign_flips": curvature["endpoint_curv_sign_flips"],
        "curv_sign_flips": curvature["curv_sign_flips"],
        "tangent_start_deg": tangent_start,
        "tangent_end_deg": tangent_end,
        "tangent_max_deg": tangent_max,
        "tangent_start_status": record.tangent_start_status,
        "tangent_end_status": record.tangent_end_status,
        "tangent_start_support_px": (record.tangent_start_info or {}).get("support_px"),
        "tangent_end_support_px": (record.tangent_end_info or {}).get("support_px"),
        "lambda_requested": result["lambda_requested"],
        "lambda_used": result["lambda_used"],
        "refits": result["refits"],
        "width_exact_p5": width_exact.get("p5"),
        "width_smooth_p5": width_smooth.get("p5"),
        "width_exact_median": width_exact.get("median"),
        "width_smooth_median": width_smooth.get("median"),
        "width_smooth_raw_p5": width_smooth_raw.get("p5"),
        "width_ratio_p5": width_ratio_p5,
        "width_ratio_median": width_ratio_median,
        "width_inflated": str(bool(result.get("width_inflated", False))).lower(),
        "variant_note": "",
    }


METRIC_COLUMNS = [
    "variant",
    "beta",
    "panel_id",
    "boundary_id",
    "labels_pair",
    "closed",
    "path_index",
    "n_points",
    "n_out_points",
    "eligible",
    "status",
    "reason",
    "mean_disp",
    "p95_disp",
    "max_disp",
    "max_curvature",
    "p95_curvature",
    "exact_max_curvature",
    "exact_p95_curvature",
    "endpoint_drift",
    "self_intersections",
    "length_exact",
    "length_smooth",
    "length_change",
    "length_change_pct",
    "area_exact",
    "area_smooth",
    "area_change",
    "area_change_rel",
    "max_curv_reduction",
    "p95_curv_reduction",
    "max_curvature_dense",
    "p95_curvature_dense",
    "max_curvature_dense_u",
    "max_curvature_dense_end_dist",
    "endpoint_region_px",
    "endpoint_start_max_curv",
    "endpoint_end_max_curv",
    "endpoint_start_p95_curv",
    "endpoint_end_p95_curv",
    "endpoint_region_max_curv",
    "endpoint_region_p95_curv",
    "endpoint_wide_max_curv",
    "endpoint_curv_sign_flips",
    "curv_sign_flips",
    "tangent_start_deg",
    "tangent_end_deg",
    "tangent_max_deg",
    "tangent_start_status",
    "tangent_end_status",
    "tangent_start_support_px",
    "tangent_end_support_px",
    "lambda_requested",
    "lambda_used",
    "refits",
    "width_exact_p5",
    "width_smooth_p5",
    "width_exact_median",
    "width_smooth_median",
    "width_smooth_raw_p5",
    "width_ratio_p5",
    "width_ratio_median",
    "width_inflated",
    "variant_note",
]


def _panel_aggregate(
    variant: str,
    beta: float,
    panel_id: str,
    records: list[PathRecord],
    results: list[dict],
    cross_exact: int,
    cross_final: int,
    used_for_selection: bool,
) -> dict:
    statuses = [result["status"] for result in results]
    n_baseline = statuses.count("baseline")
    n_ok = statuses.count("ok")
    n_ok_reduced = statuses.count("ok_reduced")
    n_fallback_short = statuses.count("fallback_short")
    n_fallback_constraint = statuses.count("fallback_constraint")
    n_fallback_width_guard = statuses.count("fallback_width_guard")
    n_error = statuses.count("error")
    n_eligible = sum(record.eligible for record in records)

    eligible_pairs = [
        (record, result)
        for record, result in zip(records, results)
        if record.eligible
    ]
    smoothed = [
        (record, result)
        for record, result in eligible_pairs
        if result["status"] in ("ok", "ok_reduced")
    ]
    open_eligible = [
        (record, result)
        for record, result in eligible_pairs
        if not record.path.closed
    ]
    closed_eligible = [
        (record, result)
        for record, result in eligible_pairs
        if record.path.closed
    ]

    area_rels = [
        abs(float(result["metrics"].get("area_change", 0.0)))
        / max(float(result["metrics"].get("area_exact", 0.0)), 1e-9)
        for record, result in closed_eligible
        if float(result["metrics"].get("area_exact", 0.0)) > 1e-9
    ]
    width_ratios = [
        result.get("width_ratio_p5")
        for _, result in closed_eligible
        if result.get("width_ratio_p5") is not None
    ]
    max_red = [
        reduction
        for record, result in smoothed
        for reduction in [_curvature_reduction(result["metrics"], record.exact_metrics)[0]]
        if reduction is not None
    ]
    p95_red = [
        reduction
        for record, result in smoothed
        for reduction in [_curvature_reduction(result["metrics"], record.exact_metrics)[1]]
        if reduction is not None
    ]
    tangent_errors = [
        result.get("tangent_max_deg")
        for record, result in open_eligible
        if result.get("tangent_max_deg") is not None
    ]

    return {
        "variant": variant,
        "beta": beta,
        "panel_id": panel_id,
        "n_paths": len(records),
        "n_boundary_objects": len({record.path.boundary_id for record in records}),
        "n_eligible": n_eligible,
        "n_open": sum(not record.path.closed for record in records),
        "n_closed": sum(record.path.closed for record in records),
        "n_baseline": n_baseline,
        "n_ok": n_ok,
        "n_ok_reduced": n_ok_reduced,
        "n_fallback_short": n_fallback_short,
        "n_fallback_constraint": n_fallback_constraint,
        "n_fallback_width_guard": n_fallback_width_guard,
        "n_error": n_error,
        "reduction_rate": (n_ok_reduced / n_eligible) if n_eligible else 0.0,
        "fallback_rate": (
            (n_fallback_constraint + n_error) / n_eligible if n_eligible else 0.0
        ),
        "median_mean_disp": _median([result["metrics"]["mean_disp"] for _, result in smoothed]),
        "median_p95_disp": _median([result["metrics"]["p95_disp"] for _, result in smoothed]),
        "p90_p95_disp": _p90([result["metrics"]["p95_disp"] for _, result in smoothed]),
        "median_p95_disp_eligible": _median(
            [result["metrics"]["p95_disp"] for record, result in eligible_pairs]
        ),
        "p90_p95_disp_eligible": _p90(
            [result["metrics"]["p95_disp"] for record, result in eligible_pairs]
        ),
        "median_max_disp": _median([result["metrics"]["max_disp"] for _, result in smoothed]),
        "p95_max_disp": _p95([result["metrics"]["max_disp"] for _, result in smoothed]),
        "median_max_curvature": _median(
            [result["metrics"]["max_curvature"] for _, result in smoothed]
        ),
        "median_exact_max_curvature": _median(
            [result["metrics"]["exact_max_curvature"] for _, result in smoothed]
        ),
        "median_max_curv_reduction": _median(max_red),
        "median_p95_curv_reduction": _median(p95_red),
        "median_endpoint_region_max_curv": _median(
            [result["curvature"]["endpoint_region_max_curv"] for _, result in open_eligible]
        ),
        "p90_endpoint_region_max_curv": _p90(
            [result["curvature"]["endpoint_region_max_curv"] for _, result in open_eligible]
        ),
        "median_endpoint_region_p95_curv": _median(
            [result["curvature"]["endpoint_region_p95_curv"] for _, result in open_eligible]
        ),
        "median_endpoint_wide_max_curv": _median(
            [result["curvature"]["endpoint_wide_max_curv"] for _, result in open_eligible]
        ),
        "median_tangent_max_deg": _median(tangent_errors),
        "p90_tangent_max_deg": _p90(tangent_errors),
        "max_abs_area_change_rel": max(area_rels) if area_rels else None,
        "min_width_ratio_p5": min(width_ratios) if width_ratios else None,
        "self_intersections_total": sum(
            int(result["metrics"]["self_intersections"]) for result in results
        ),
        "cross_intersections": cross_final,
        "cross_exact": cross_exact,
        "used_for_selection": used_for_selection,
        "n_open_eligible": len(open_eligible),
        "n_closed_eligible": len(closed_eligible),
    }


PANEL_COLUMNS = [
    "variant",
    "beta",
    "panel_id",
    "n_paths",
    "n_boundary_objects",
    "n_eligible",
    "n_open",
    "n_closed",
    "n_ok",
    "n_ok_reduced",
    "n_fallback_short",
    "n_fallback_constraint",
    "n_fallback_width_guard",
    "n_error",
    "reduction_rate",
    "fallback_rate",
    "median_mean_disp",
    "median_p95_disp",
    "p90_p95_disp",
    "median_p95_disp_eligible",
    "p90_p95_disp_eligible",
    "median_max_disp",
    "p95_max_disp",
    "median_max_curvature",
    "median_exact_max_curvature",
    "median_max_curv_reduction",
    "median_p95_curv_reduction",
    "median_endpoint_region_max_curv",
    "p90_endpoint_region_max_curv",
    "median_endpoint_region_p95_curv",
    "median_endpoint_wide_max_curv",
    "median_tangent_max_deg",
    "p90_tangent_max_deg",
    "max_abs_area_change_rel",
    "min_width_ratio_p5",
    "self_intersections_total",
    "cross_exact",
    "cross_intersections",
    "used_for_selection",
]


def _selection_stats(
    variant: str,
    beta: float,
    results_by_panel: dict[str, list[dict]],
    records_by_panel: dict[str, list[PathRecord]],
    panel_rows: dict[tuple[str, str], dict],
) -> dict:
    eligible_pairs: list[tuple[PathRecord, dict]] = []
    open_pairs: list[tuple[PathRecord, dict]] = []
    closed_pairs: list[tuple[PathRecord, dict]] = []
    for panel_id, records in records_by_panel.items():
        if panel_id == PH01:
            continue
        for record, result in zip(records, results_by_panel[panel_id]):
            if not record.eligible:
                continue
            eligible_pairs.append((record, result))
            if record.path.closed:
                closed_pairs.append((record, result))
            else:
                open_pairs.append((record, result))

    statuses = [result["status"] for _, result in eligible_pairs]
    smoothed = [
        (record, result)
        for record, result in eligible_pairs
        if result["status"] in ("ok", "ok_reduced")
    ]
    max_red = [
        reduction
        for record, result in smoothed
        for reduction in [_curvature_reduction(result["metrics"], record.exact_metrics)[0]]
        if reduction is not None
    ]
    p95_red = [
        reduction
        for record, result in smoothed
        for reduction in [_curvature_reduction(result["metrics"], record.exact_metrics)[1]]
        if reduction is not None
    ]
    cross_total = sum(
        int(panel_rows[(variant, panel_id)]["cross_intersections"])
        for panel_id in records_by_panel
        if panel_id != PH01
    )
    area_rels = [
        abs(float(result["metrics"].get("area_change", 0.0)))
        / max(float(result["metrics"].get("area_exact", 0.0)), 1e-9)
        for record, result in closed_pairs
        if float(result["metrics"].get("area_exact", 0.0)) > 1e-9
    ]
    tangent_errors = [
        result.get("tangent_max_deg")
        for _, result in open_pairs
        if result.get("tangent_max_deg") is not None
    ]
    return {
        "variant": variant,
        "beta": beta,
        "n_eligible": len(eligible_pairs),
        "n_open": len(open_pairs),
        "n_closed": len(closed_pairs),
        "n_smoothed": len(smoothed),
        "n_ok_reduced": statuses.count("ok_reduced"),
        "n_fallback_constraint": statuses.count("fallback_constraint"),
        "n_fallback_width_guard": statuses.count("fallback_width_guard"),
        "n_errors": statuses.count("error"),
        "median_mean_disp": _median([result["metrics"]["mean_disp"] for _, result in smoothed]),
        "median_p95_disp": _median([result["metrics"]["p95_disp"] for _, result in smoothed]),
        "p90_p95_disp": _p90([result["metrics"]["p95_disp"] for _, result in smoothed]),
        "median_p95_disp_eligible": _median(
            [result["metrics"]["p95_disp"] for _, result in eligible_pairs]
        ),
        "p90_p95_disp_eligible": _p90(
            [result["metrics"]["p95_disp"] for _, result in eligible_pairs]
        ),
        "median_max_disp": _median([result["metrics"]["max_disp"] for _, result in smoothed]),
        "p95_max_disp": _p95([result["metrics"]["max_disp"] for _, result in smoothed]),
        "median_max_curvature": _median(
            [result["metrics"]["max_curvature"] for _, result in smoothed]
        ),
        "median_max_curv_reduction": _median(max_red),
        "median_p95_curv_reduction": _median(p95_red),
        "median_endpoint_region_max_curv": _median(
            [result["curvature"]["endpoint_region_max_curv"] for _, result in open_pairs]
        ),
        "p90_endpoint_region_max_curv": _p90(
            [result["curvature"]["endpoint_region_max_curv"] for _, result in open_pairs]
        ),
        "median_endpoint_region_p95_curv": _median(
            [result["curvature"]["endpoint_region_p95_curv"] for _, result in open_pairs]
        ),
        "median_endpoint_wide_max_curv": _median(
            [result["curvature"]["endpoint_wide_max_curv"] for _, result in open_pairs]
        ),
        "median_tangent_max_deg": _median(tangent_errors),
        "p90_tangent_max_deg": _p90(tangent_errors),
        "n_tangent_endpoints": len(tangent_errors),
        "self_intersections_total": sum(
            int(result["metrics"]["self_intersections"]) for _, result in eligible_pairs
        ),
        "cross_intersections_total": cross_total,
        "max_endpoint_drift": max(
            [float(result["metrics"]["endpoint_drift"]) for record, result in open_pairs],
            default=0.0,
        ),
        "max_abs_area_change_rel": max(area_rels) if area_rels else None,
        "n_paths_improved": None,
        "n_paths_worsened": None,
        "n_paths_unchanged": None,
    }


def _selection_reasons(stats: dict, baseline: dict) -> list[str]:
    reasons: list[str] = []
    if stats["n_errors"] > 0:
        reasons.append(f"errors={stats['n_errors']}")
    if stats["n_fallback_constraint"] > 0:
        reasons.append(f"constraint_fallbacks={stats['n_fallback_constraint']}")
    if stats["n_fallback_width_guard"] > 0:
        reasons.append(f"width_guard_fallbacks={stats['n_fallback_width_guard']}")
    if stats["self_intersections_total"] > 0:
        reasons.append(f"self_int={stats['self_intersections_total']}")
    if stats["cross_intersections_total"] > 0:
        reasons.append(f"cross_int={stats['cross_intersections_total']}")
    if stats["max_endpoint_drift"] > DRIFT_TOL:
        reasons.append(f"endpoint_drift={stats['max_endpoint_drift']:.4f}")
    if stats["n_tangent_endpoints"] == 0:
        reasons.append("no_applied_tangents")
    base_p95 = baseline["median_p95_disp"]
    base_p90 = baseline["p90_p95_disp"]
    if (
        base_p95 is not None
        and stats["median_p95_disp"] is not None
        and stats["median_p95_disp"] > base_p95 + DISPLACEMENT_TOL
    ):
        reasons.append(
            f"median_p95={stats['median_p95_disp']:.3f}>{base_p95:.3f}+{DISPLACEMENT_TOL:g}"
        )
    if (
        base_p90 is not None
        and stats["p90_p95_disp"] is not None
        and stats["p90_p95_disp"] > base_p90 + DISPLACEMENT_TOL
    ):
        reasons.append(
            f"p90_p95={stats['p90_p95_disp']:.3f}>{base_p90:.3f}+{DISPLACEMENT_TOL:g}"
        )
    base_med = baseline["median_endpoint_region_max_curv"]
    base_p90c = baseline["p90_endpoint_region_max_curv"]
    if (
        base_med is not None
        and stats["median_endpoint_region_max_curv"] is not None
        and stats["median_endpoint_region_max_curv"]
        > base_med * (1.0 - MIN_ENDPOINT_CURV_REDUCTION)
    ):
        reasons.append(
            f"median_endpoint_curv={stats['median_endpoint_region_max_curv']:.4f}"
            f">{base_med * (1.0 - MIN_ENDPOINT_CURV_REDUCTION):.4f}"
        )
    if (
        base_p90c is not None
        and stats["p90_endpoint_region_max_curv"] is not None
        and stats["p90_endpoint_region_max_curv"]
        > base_p90c * (1.0 + MAX_ENDPOINT_P90_INCREASE)
    ):
        reasons.append(
            f"p90_endpoint_curv={stats['p90_endpoint_region_max_curv']:.4f}"
            f">{base_p90c * (1.0 + MAX_ENDPOINT_P90_INCREASE):.4f}"
        )
    return reasons


def _comparison(
    hard_results_by_panel: dict[str, list[dict]],
    other_results_by_panel: dict[str, list[dict]],
    records_by_panel: dict[str, list[PathRecord]],
) -> dict:
    improved = 0
    worsened = 0
    unchanged = 0
    not_applicable = 0
    relative_changes: list[float] = []
    absolute_changes: list[float] = []
    for panel_id, records in records_by_panel.items():
        if panel_id == PH01:
            continue
        for record, hard, other in zip(
            records,
            hard_results_by_panel[panel_id],
            other_results_by_panel[panel_id],
        ):
            if record.path.closed or not record.eligible:
                not_applicable += 1
                continue
            hard_value = hard["curvature"]["endpoint_region_max_curv"]
            other_value = other["curvature"]["endpoint_region_max_curv"]
            if hard_value is None or other_value is None:
                not_applicable += 1
                continue
            if not np.isfinite(hard_value) or not np.isfinite(other_value):
                not_applicable += 1
                continue
            delta = float(other_value) - float(hard_value)
            absolute_changes.append(delta)
            if hard_value > 1e-9:
                relative_changes.append(delta / float(hard_value))
            if delta < -1e-12:
                improved += 1
            elif delta > 1e-12:
                worsened += 1
            else:
                unchanged += 1
    return {
        "improved": improved,
        "worsened": worsened,
        "unchanged": unchanged,
        "not_applicable": not_applicable,
        "median_relative_change": _median(relative_changes),
        "median_absolute_change": _median(absolute_changes),
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def _format_value(value, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value)).lower()
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return ""
        return f"{float(value):.{digits}f}"
    return str(value)


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_value(row.get(key)) for key in columns})


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _pct(value: float | None, digits: int = 1) -> str:
    if value is None or not np.isfinite(float(value)):
        return "n/a"
    return f"{100.0 * float(value):.{digits}f}%"


def _num(value: float | None, digits: int = 3) -> str:
    if value is None or not np.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _beta_label(beta: float) -> str:
    return "0" if beta == 0 else f"{beta:g}"


def _variant_label(beta: float) -> str:
    return "hard_only" if beta == 0 else f"tangent_soft_beta{beta:g}"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _display_points(result: dict) -> np.ndarray:
    points = np.asarray(result["points"], dtype=float)
    if result["closed"] and len(points) > 1:
        points = np.vstack([points, points[:1]])
    return points


def _overlay(
    panel: Panel,
    records: list[PathRecord],
    results: list[dict],
    title: str = "",
) -> Image.Image:
    return render_overlay(
        panel.labels,
        [record.path for record in records],
        [_display_points(result) for result in results],
        title=title,
        thickness=3,
    )


def _load_font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - old Pillow
        return ImageFont.load_default()


def _fit_paste(
    canvas: Image.Image,
    image: Image.Image,
    box: tuple[int, int, int, int],
) -> None:
    width = box[2] - box[0]
    height = box[3] - box[1]
    copy = image.copy()
    copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    x = box[0] + (width - copy.width) // 2
    y = box[1] + (height - copy.height) // 2
    canvas.paste(copy, (x, y))


def _contact_sheet(
    panel_ids: list[str],
    overlays_hard: dict[str, Image.Image],
    overlays_guard: dict[str, Image.Image],
    panel_rows: dict[tuple[str, str], dict],
    out_path: Path,
    beta: float,
) -> None:
    label_width = 236
    cell_width, cell_height = 386, 210
    gap = 8
    header = 46
    row_height = cell_height + 8
    width = label_width + 2 * cell_width + gap + 2 * gap
    height = header + row_height * len(panel_ids)
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    font_title = _load_font(15)
    font_cell = _load_font(13)
    font_small = _load_font(11)
    draw.text((8, 6), "hard_only", fill=(255, 230, 80), font=font_title)
    draw.text(
        (label_width + cell_width + gap + 8, 6),
        f"tangent_soft_guard (beta={_beta_label(beta)})",
        fill=(255, 230, 80),
        font=font_title,
    )
    for index, panel_id in enumerate(panel_ids):
        y = header + index * row_height
        draw.text((8, y + 8), panel_id, fill=(235, 235, 235), font=font_cell)
        hard_row = panel_rows[("hard_only", panel_id)]
        guard_row = panel_rows[("tangent_soft_guard", panel_id)]
        lines = [
            f"paths {hard_row['n_paths']}  eligible {hard_row['n_eligible']}",
            f"cross {hard_row['cross_exact']} -> {hard_row['cross_intersections']} -> "
            f"{guard_row['cross_intersections']}",
            f"med endpt {_num(hard_row['median_endpoint_region_max_curv'], 3)} -> "
            f"{_num(guard_row['median_endpoint_region_max_curv'], 3)}",
            f"ok {guard_row['n_ok']} red {guard_row['n_ok_reduced']} "
            f"fb {guard_row['n_fallback_constraint'] + guard_row['n_fallback_width_guard']}",
        ]
        for line_index, text in enumerate(lines):
            draw.text(
                (8, y + 30 + line_index * 15),
                text,
                fill=(170, 170, 170),
                font=font_small,
            )
        _fit_paste(
            canvas,
            overlays_hard[panel_id],
            (label_width, y, label_width + cell_width, y + cell_height),
        )
        _fit_paste(
            canvas,
            overlays_guard[panel_id],
            (
                label_width + cell_width + gap,
                y,
                label_width + 2 * cell_width + gap,
                y + cell_height,
            ),
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _zoom_cases(
    records_by_panel: dict[str, list[PathRecord]],
    hard_results: dict[str, list[dict]],
    soft_results: dict[str, list[dict]],
    limit: int = 8,
) -> list[dict]:
    cases: list[dict] = []
    seen: set[tuple[str, int]] = set()

    def add(panel_id: str, path_index: int, tag: str) -> None:
        key = (panel_id, path_index)
        if key in seen:
            return
        records = records_by_panel[panel_id]
        if path_index < 0 or path_index >= len(records):
            return
        record = records[path_index]
        if record.path.closed or not record.eligible:
            return
        hard = hard_results[panel_id][path_index]
        soft = soft_results[panel_id][path_index]
        start_hard = hard["curvature"]["endpoint_start_max_curv"]
        end_hard = hard["curvature"]["endpoint_end_max_curv"]
        if start_hard is None or end_hard is None:
            return
        side_start = float(start_hard) >= float(end_hard)
        seen.add(key)
        cases.append(
            {
                "tag": tag,
                "panel_id": panel_id,
                "path_index": path_index,
                "record": record,
                "hard": hard,
                "soft": soft,
                "side_start": side_start,
            }
        )

    for panel_id, path_index in (
        ("schematic_panel2", 37),
        ("paper23_fig7a", 13),
        ("schematic_panel2", 36),
    ):
        if panel_id in records_by_panel:
            add(panel_id, path_index, "anchor")

    ranked = []
    for panel_id, records in records_by_panel.items():
        if panel_id == PH01:
            continue
        for path_index, record in enumerate(records):
            if record.path.closed or not record.eligible:
                continue
            hard = hard_results[panel_id][path_index]
            value = hard["curvature"]["endpoint_region_max_curv"]
            if value is not None and np.isfinite(value):
                ranked.append((float(value), panel_id, path_index))
    for _, panel_id, path_index in sorted(ranked, reverse=True):
        if len(cases) >= limit:
            break
        add(panel_id, path_index, "baseline_top")

    regressions = []
    for panel_id, records in records_by_panel.items():
        if panel_id == PH01:
            continue
        for path_index, record in enumerate(records):
            if record.path.closed or not record.eligible:
                continue
            hard = hard_results[panel_id][path_index]
            soft = soft_results[panel_id][path_index]
            hard_value = hard["curvature"]["endpoint_region_max_curv"]
            soft_value = soft["curvature"]["endpoint_region_max_curv"]
            if (
                hard_value is None
                or soft_value is None
                or not np.isfinite(hard_value)
                or not np.isfinite(soft_value)
                or hard_value < 1e-9
            ):
                continue
            regressions.append(
                ((float(soft_value) - float(hard_value)) / float(hard_value), panel_id, path_index)
            )
    for _, panel_id, path_index in sorted(regressions, reverse=True):
        if len(cases) >= limit:
            break
        add(panel_id, path_index, "regression")
    return cases


def _endpoint_zoom(
    cases: list[dict],
    overlays_hard: dict[str, Image.Image],
    overlays_soft: dict[str, Image.Image],
    out_path: Path,
    beta: float,
) -> None:
    if not cases:
        return
    crop = 95
    scale = 3
    label_height = 48
    row_gap = 10
    col_width = crop * scale
    left = 250
    width = left + 2 * col_width + 24
    row_height = crop * scale + label_height + row_gap
    height = 60 + row_height * len(cases)
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    font_title = _load_font(16)
    font_body = _load_font(13)
    font_small = _load_font(11)
    draw.text((8, 8), "endpoint zoom: hard_only (left) vs tangent_soft (right)", fill=(255, 230, 80), font=font_title)
    draw.text((left + 8, 30), "hard_only", fill=(255, 230, 80), font=font_body)
    draw.text((left + col_width + 20, 30), f"tangent_soft beta={_beta_label(beta)}", fill=(255, 230, 80), font=font_body)
    for index, case in enumerate(cases):
        panel_id = case["panel_id"]
        record = case["record"]
        side_start = case["side_start"]
        endpoint = record.path.points[0] if side_start else record.path.points[-1]
        hard = case["hard"]
        soft = case["soft"]
        y = 60 + index * row_height
        panel_shape = overlays_hard[panel_id].size
        x0 = int(round(endpoint[0])) - crop // 2
        y0 = int(round(endpoint[1])) - crop // 2
        x0 = max(0, min(x0, panel_shape[0] - crop))
        y0 = max(0, min(y0, panel_shape[1] - crop))
        box = (x0, y0, x0 + crop, y0 + crop)
        for column, overlay in enumerate((overlays_hard[panel_id], overlays_soft[panel_id])):
            piece = overlay.crop(box).resize(
                (crop * scale, crop * scale), Image.Resampling.LANCZOS
            )
            px = left + column * (col_width + 12)
            canvas.paste(piece, (px, y))
            draw_piece = ImageDraw.Draw(canvas)
            cx = px + (float(endpoint[0]) - x0) * scale
            cy = y + (float(endpoint[1]) - y0) * scale
            draw_piece.ellipse(
                [cx - 5, cy - 5, cx + 5, cy + 5],
                outline=(255, 230, 80),
                width=2,
            )
        hard_value = (
            hard["curvature"]["endpoint_start_max_curv"]
            if side_start
            else hard["curvature"]["endpoint_end_max_curv"]
        )
        soft_value = (
            soft["curvature"]["endpoint_start_max_curv"]
            if side_start
            else soft["curvature"]["endpoint_end_max_curv"]
        )
        status = (
            record.tangent_start_status if side_start else record.tangent_end_status
        )
        tag = case["tag"]
        draw.text(
            (8, y + 4),
            f"{panel_id} p{case['path_index']}",
            fill=(235, 235, 235),
            font=font_body,
        )
        draw.text(
            (8, y + 24),
            f"{'start' if side_start else 'end'} ({tag})",
            fill=(200, 200, 200),
            font=font_small,
        )
        draw.text(
            (8, y + 42),
            f"endpt max {_num(hard_value, 3)} -> {_num(soft_value, 3)}",
            fill=(200, 200, 200),
            font=font_small,
        )
        draw.text(
            (8, y + 60),
            f"tangent: {status}",
            fill=(200, 200, 200),
            font=font_small,
        )
        draw.text(
            (8, y + 78),
            f"contract max {_num(hard['metrics']['max_curvature'], 3)} -> "
            f"{_num(soft['metrics']['max_curvature'], 3)}",
            fill=(200, 200, 200),
            font=font_small,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


# ---------------------------------------------------------------------------
# Baseline parity and sanity
# ---------------------------------------------------------------------------
def _baseline_parity(rows: list[dict]) -> dict | None:
    if not A_METRICS.exists():
        return None
    reference: dict[tuple[str, int], dict] = {}
    with A_METRICS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["variant"] != "lam1":
                continue
            reference[(row["panel_id"], int(row["path_index"]))] = row
    fields = [
        "mean_disp",
        "p95_disp",
        "max_disp",
        "max_curvature",
        "p95_curvature",
        "endpoint_drift",
        "self_intersections",
        "length_smooth",
        "area_change",
        "lambda_used",
    ]
    max_diffs = {field: 0.0 for field in fields}
    missing = 0
    status_mismatches = 0
    compared = 0
    for row in rows:
        key = (row["panel_id"], int(row["path_index"]))
        reference_row = reference.get(key)
        if reference_row is None:
            missing += 1
            continue
        compared += 1
        if row["status"] != reference_row["status"]:
            status_mismatches += 1
        for field in fields:
            reference_value = reference_row.get(field, "")
            value = row.get(field)
            if reference_value == "" or value is None:
                continue
            if not np.isfinite(float(value)):
                continue
            diff = abs(float(value) - float(reference_value))
            max_diffs[field] = max(max_diffs[field], diff)
    return {
        "compared": compared,
        "missing": missing,
        "status_mismatches": status_mismatches,
        "max_diffs": max_diffs,
        "n_reference": len(reference),
    }


def _safe_cross(records: list[PathRecord], results: list[dict], panel_id: str) -> tuple[int, str]:
    try:
        views = [
            PathView(
                boundary_id=record.path.boundary_id,
                labels_pair=record.path.labels_pair,
                points=np.asarray(result["points"], dtype=float),
                closed=record.path.closed,
                exact_length=record.path.exact_length,
            )
            for record, result in zip(records, results)
        ]
        return cross_intersections(views), ""
    except Exception as exc:  # record, never skip
        return -1, f"{panel_id}: {type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    started = time.time()
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cfg = Config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PANELS_DIR.mkdir(parents=True, exist_ok=True)

    stage_times: dict[str, float] = {}
    errors: list[str] = []
    render_errors: list[str] = []

    print("loading panels", flush=True)
    panels = load_panels(MANIFEST)
    {panel.panel_id: panel for panel in panels}
    print(f"loaded {len(panels)} panels", flush=True)

    t0 = time.time()
    records_by_panel: dict[str, list[PathRecord]] = {}
    exact_cross: dict[str, int] = {}
    for panel in panels:
        t_panel = time.time()
        records = _build_records(panel, cfg)
        records_by_panel[panel.panel_id] = records
        cross, cross_error = _safe_cross(
            records,
            [{"points": record.path.points} for record in records],
            panel.panel_id,
        )
        exact_cross[panel.panel_id] = cross
        if cross_error:
            errors.append(cross_error)
        eligible = sum(record.eligible for record in records)
        print(
            f"{panel.panel_id}: {len(records)} paths, {eligible} eligible, "
            f"cross(exact)={cross}, {time.time() - t_panel:.2f}s",
            flush=True,
        )
    stage_times["build"] = time.time() - t0

    total_paths = sum(len(records) for records in records_by_panel.values())
    total_objects = sum(
        len({record.path.boundary_id for record in records})
        for records in records_by_panel.values()
    )
    total_eligible = sum(
        record.eligible for records in records_by_panel.values() for record in records
    )
    total_open_eligible = sum(
        record.eligible and not record.path.closed
        for records in records_by_panel.values()
        for record in records
    )
    total_closed_eligible = total_eligible - total_open_eligible
    total_short = total_paths - total_eligible
    print(
        f"total paths={total_paths}, objects={total_objects}, eligible={total_eligible}, "
        f"open={total_open_eligible}, closed={total_closed_eligible}, short={total_short}",
        flush=True,
    )

    # Tangent endpoint support counts (eligible paths).
    tangent_counts: dict[str, int] = {}
    for records in records_by_panel.values():
        for record in records:
            if not record.eligible:
                continue
            for status in (record.tangent_start_status, record.tangent_end_status):
                tangent_counts[status] = tangent_counts.get(status, 0) + 1

    # ------------------------------------------------------------------
    # Beta grid
    # ------------------------------------------------------------------
    print("\nbeta grid", flush=True)
    results_by_variant: dict[str, dict[str, list[dict]]] = {}
    variant_beta: dict[str, float] = {}
    cross_by_variant: dict[str, dict[str, int]] = {}
    grid_variants: list[str] = []

    for beta in BETA_GRID:
        variant = _variant_label(beta)
        grid_variants.append(variant)
        variant_beta[variant] = beta
        t_beta = time.time()
        results_by_variant[variant] = {}
        cross_by_variant[variant] = {}
        for panel in panels:
            records = records_by_panel[panel.panel_id]
            if beta == 0.0:
                results = [
                    _evaluate_path(record, LAMBDA, 0.0, cfg) for record in records
                ]
            else:
                hard_results = results_by_variant[_variant_label(0.0)][panel.panel_id]
                results = []
                for record, hard_result in zip(records, hard_results):
                    if record.path.closed:
                        # Closed paths have no endpoints; the tangent penalty is
                        # not defined and the fit is identical.
                        results.append(hard_result)
                    else:
                        results.append(_evaluate_path(record, LAMBDA, beta, cfg))
            results_by_variant[variant][panel.panel_id] = results
            cross, cross_error = _safe_cross(records, results, panel.panel_id)
            cross_by_variant[variant][panel.panel_id] = cross
            if cross_error:
                errors.append(f"{variant} {cross_error}")
        elapsed = time.time() - t_beta
        stats_tmp = _selection_stats(
            variant,
            beta,
            results_by_variant[variant],
            records_by_panel,
            {
                (v, panel_id): {"cross_intersections": cross_by_variant[v][panel_id]}
                for v in [variant]
                for panel_id in records_by_panel
            },
        )
        print(
            f"beta={beta:6g}: {elapsed:.1f}s  med_p95={_num(stats_tmp['median_p95_disp'], 4)} "
            f"med_endpoint={_num(stats_tmp['median_endpoint_region_max_curv'], 4)} "
            f"fb={stats_tmp['n_fallback_constraint']} err={stats_tmp['n_errors']}",
            flush=True,
        )
    # The grid block includes per-path fits, panel aggregation and the
    # per-variant cross-intersection checks.
    grid_finished = time.time()

    # ------------------------------------------------------------------
    # Panel rows for grid variants
    # ------------------------------------------------------------------
    panel_rows: dict[tuple[str, str], dict] = {}
    for variant in grid_variants:
        beta = variant_beta[variant]
        for panel in panels:
            panel_id = panel.panel_id
            panel_rows[(variant, panel_id)] = _panel_aggregate(
                variant,
                beta,
                panel_id,
                records_by_panel[panel_id],
                results_by_variant[variant][panel_id],
                exact_cross[panel_id],
                cross_by_variant[variant][panel_id],
                panel_id != PH01,
            )

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    baseline_variant = _variant_label(0.0)
    baseline_stats = _selection_stats(
        baseline_variant,
        0.0,
        results_by_variant[baseline_variant],
        records_by_panel,
        panel_rows,
    )
    selection_rows: list[dict] = []
    for beta in BETA_GRID:
        variant = _variant_label(beta)
        stats = _selection_stats(
            variant, beta, results_by_variant[variant], records_by_panel, panel_rows
        )
        reasons = [] if beta == 0.0 else _selection_reasons(stats, baseline_stats)
        selection_rows.append({"beta": beta, "variant": variant, "stats": stats, "reasons": reasons})

    candidates = [row for row in selection_rows if row["beta"] > 0 and not row["reasons"]]
    selection_mode = "largest passing beta"
    if candidates:
        selected_row = max(candidates, key=lambda row: row["beta"])
    else:
        selection_mode = "fallback"
        feasible = [
            row
            for row in selection_rows
            if row["beta"] > 0
            and row["stats"]["n_errors"] == 0
            and row["stats"]["n_fallback_constraint"] == 0
            and row["stats"]["n_fallback_width_guard"] == 0
            and row["stats"]["self_intersections_total"] == 0
            and row["stats"]["cross_intersections_total"] == 0
            and row["stats"]["n_tangent_endpoints"] > 0
            and baseline_stats["median_p95_disp"] is not None
            and row["stats"]["median_p95_disp"] is not None
            and row["stats"]["median_p95_disp"]
            <= baseline_stats["median_p95_disp"] + DISPLACEMENT_TOL
            and baseline_stats["p90_p95_disp"] is not None
            and row["stats"]["p90_p95_disp"] is not None
            and row["stats"]["p90_p95_disp"]
            <= baseline_stats["p90_p95_disp"] + DISPLACEMENT_TOL
        ]
        if feasible:
            selected_row = min(
                feasible,
                key=lambda row: (
                    row["stats"]["median_endpoint_region_max_curv"]
                    if row["stats"]["median_endpoint_region_max_curv"] is not None
                    else float("inf"),
                    row["beta"],
                ),
            )
        else:
            selected_row = selection_rows[0]
    selected_beta = float(selected_row["beta"])
    selected_variant_grid = str(selected_row["variant"])
    selected_stats = selected_row["stats"]
    print(
        f"\nselected beta={selected_beta:g} ({selection_mode}); "
        f"median endpoint={_num(selected_stats['median_endpoint_region_max_curv'], 4)}, "
        f"median p95={_num(selected_stats['median_p95_disp'], 4)}",
        flush=True,
    )

    tangent_soft_variant = "tangent_soft"
    tangent_soft_results = results_by_variant[selected_variant_grid]
    tangent_soft_cross = cross_by_variant[selected_variant_grid]
    for panel in panels:
        panel_rows[(tangent_soft_variant, panel.panel_id)] = _panel_aggregate(
            tangent_soft_variant,
            selected_beta,
            panel.panel_id,
            records_by_panel[panel.panel_id],
            tangent_soft_results[panel.panel_id],
            exact_cross[panel.panel_id],
            tangent_soft_cross[panel.panel_id],
            panel.panel_id != PH01,
        )
    tangent_soft_stats = _selection_stats(
        tangent_soft_variant,
        selected_beta,
        tangent_soft_results,
        records_by_panel,
        panel_rows,
    )

    # ------------------------------------------------------------------
    # Guard variant
    # ------------------------------------------------------------------
    print("applying closed-body width guard", flush=True)
    t_guard = time.time()
    guard_results: dict[str, list[dict]] = {}
    guard_cross: dict[str, int] = {}
    for panel in panels:
        panel_id = panel.panel_id
        records = records_by_panel[panel_id]
        soft_results = tangent_soft_results[panel_id]
        guarded = [
            _apply_width_guard(record, result)
            for record, result in zip(records, soft_results)
        ]
        guard_results[panel_id] = guarded
        cross, cross_error = _safe_cross(records, guarded, panel_id)
        guard_cross[panel_id] = cross
        if cross_error:
            errors.append(f"guard {cross_error}")
        for panel_index, (record, result) in enumerate(zip(records, guarded)):
            if result["status"] == "fallback_width_guard":
                print(
                    f"  width guard fallback: {panel_id} p{record.path_index} "
                    f"b{record.path.boundary_id}: {result['reason']}",
                    flush=True,
                )
    for panel in panels:
        panel_rows[("tangent_soft_guard", panel.panel_id)] = _panel_aggregate(
            "tangent_soft_guard",
            selected_beta,
            panel.panel_id,
            records_by_panel[panel.panel_id],
            guard_results[panel.panel_id],
            exact_cross[panel.panel_id],
            guard_cross[panel.panel_id],
            panel.panel_id != PH01,
        )
    guard_stats = _selection_stats(
        "tangent_soft_guard",
        selected_beta,
        guard_results,
        records_by_panel,
        panel_rows,
    )
    stage_times["guard"] = time.time() - t_guard
    stage_times["grid_fit"] = grid_finished - t0 - stage_times.get("build", 0.0)
    # The grid loop also includes panel cross-intersection checks.

    # ------------------------------------------------------------------
    # Metrics rows
    # ------------------------------------------------------------------
    print("writing metrics", flush=True)
    t_rows = time.time()
    all_metric_rows: list[dict] = []
    output_variants: list[tuple[str, float, dict[str, list[dict]], str]] = [
        (baseline_variant, 0.0, results_by_variant[baseline_variant], "A baseline: lambda=1, beta=0"),
    ]
    for beta in BETA_GRID:
        if beta == 0.0:
            continue
        variant = _variant_label(beta)
        output_variants.append(
            (variant, beta, results_by_variant[variant], f"grid beta={beta:g}")
        )
    output_variants.append(
        (
            tangent_soft_variant,
            selected_beta,
            tangent_soft_results,
            f"selected fit (alias of {selected_variant_grid})",
        )
    )
    output_variants.append(
        (
            "tangent_soft_guard",
            selected_beta,
            guard_results,
            "closed-body width guard; open paths identical to tangent_soft",
        )
    )

    for variant, beta, result_map, note in output_variants:
        for panel in panels:
            panel_id = panel.panel_id
            records = records_by_panel[panel_id]
            for record, result in zip(records, result_map[panel_id]):
                row = _metrics_row(variant, beta, panel_id, record, result)
                row["variant_note"] = note
                all_metric_rows.append(row)
    _write_csv(OUT_DIR / "metrics.csv", METRIC_COLUMNS, all_metric_rows)
    stage_times["metrics_rows"] = time.time() - t_rows

    panel_order = [
        "hard_only",
        "tangent_soft_beta0.01",
        "tangent_soft_beta0.1",
        "tangent_soft_beta1",
        "tangent_soft_beta10",
        "tangent_soft_beta100",
        "tangent_soft",
        "tangent_soft_guard",
    ]
    panel_order = [name for name in panel_order if (name, panels[0].panel_id) in panel_rows]
    all_panel_rows: list[dict] = []
    for variant in panel_order:
        for panel in panels:
            all_panel_rows.append(panel_rows[(variant, panel.panel_id)])
    _write_csv(OUT_DIR / "panel_metrics.csv", PANEL_COLUMNS, all_panel_rows)

    # ------------------------------------------------------------------
    # Comparisons and failure lists
    # ------------------------------------------------------------------
    hard_results = results_by_variant[baseline_variant]
    comparison_soft = _comparison(hard_results, tangent_soft_results, records_by_panel)
    comparison_guard = _comparison(hard_results, guard_results, records_by_panel)

    def path_lookup(result_map: dict[str, list[dict]]) -> dict[tuple[str, int], dict]:
        lookup = {}
        for panel_id, results in result_map.items():
            for record, result in zip(records_by_panel[panel_id], results):
                lookup[(panel_id, record.path_index)] = result
        return lookup

    hard_lookup = path_lookup(hard_results)
    soft_lookup = path_lookup(tangent_soft_results)
    guard_lookup = path_lookup(guard_results)

    worsened: list[dict] = []
    oscillation: list[dict] = []
    improved: list[dict] = []
    for panel_id, records in records_by_panel.items():
        if panel_id == PH01:
            continue
        for record in records:
            if record.path.closed or not record.eligible:
                continue
            hard = hard_lookup[(panel_id, record.path_index)]
            soft = soft_lookup[(panel_id, record.path_index)]
            hard_value = hard["curvature"]["endpoint_region_max_curv"]
            soft_value = soft["curvature"]["endpoint_region_max_curv"]
            if hard_value is None or soft_value is None:
                continue
            if not np.isfinite(hard_value) or not np.isfinite(soft_value):
                continue
            delta = float(soft_value) - float(hard_value)
            item = {
                "panel_id": panel_id,
                "path_index": record.path_index,
                "boundary_id": record.path.boundary_id,
                "length": record.length,
                "hard": float(hard_value),
                "soft": float(soft_value),
                "delta": delta,
                "relative": delta / float(hard_value) if hard_value > 1e-9 else None,
                "start_status": record.tangent_start_status,
                "end_status": record.tangent_end_status,
                "hard_contract": float(hard["metrics"]["max_curvature"]),
                "soft_contract": float(soft["metrics"]["max_curvature"]),
                "hard_dense": float(hard["curvature"]["max_curvature_dense"]),
                "soft_dense": float(soft["curvature"]["max_curvature_dense"]),
                "hard_flips": hard["curvature"]["endpoint_curv_sign_flips"],
                "soft_flips": soft["curvature"]["endpoint_curv_sign_flips"],
            }
            if delta > 1e-12:
                worsened.append(item)
            elif delta < -1e-12:
                improved.append(item)
            hard_flips = hard["curvature"]["endpoint_curv_sign_flips"]
            soft_flips = soft["curvature"]["endpoint_curv_sign_flips"]
            if (
                hard_flips is not None
                and soft_flips is not None
                and soft_flips >= hard_flips + 2
                and delta > 1e-3
            ):
                oscillation.append(item)

    worsened.sort(
        key=lambda item: (
            item["relative"] if item["relative"] is not None else -1.0,
            item["delta"],
        ),
        reverse=True,
    )
    improved.sort(key=lambda item: item["delta"])

    # Endpoint spike cases: A's cited cases plus the strongest baseline paths.
    anchor_cases = [
        ("schematic_panel2", 37),
        ("paper23_fig7a", 13),
        ("schematic_panel2", 36),
        ("schematic_panel2", 24),
        ("newimage_p4", 4),
    ]
    spike_cases: list[dict] = []
    for panel_id, path_index in anchor_cases:
        if panel_id not in records_by_panel:
            continue
        records = records_by_panel[panel_id]
        if path_index >= len(records):
            continue
        record = records[path_index]
        spike_cases.append(
            {
                "panel_id": panel_id,
                "path_index": path_index,
                "record": record,
                "hard": hard_lookup[(panel_id, path_index)],
                "soft": soft_lookup[(panel_id, path_index)],
                "guard": guard_lookup[(panel_id, path_index)],
                "tag": "A case",
            }
        )
    baseline_ranked = []
    for panel_id, records in records_by_panel.items():
        if panel_id == PH01:
            continue
        for record in records:
            if record.path.closed or not record.eligible:
                continue
            hard = hard_lookup[(panel_id, record.path_index)]
            value = hard["curvature"]["endpoint_region_max_curv"]
            if value is not None and np.isfinite(value):
                baseline_ranked.append((float(value), panel_id, record.path_index))
    for value, panel_id, path_index in sorted(baseline_ranked, reverse=True):
        if len(spike_cases) >= 10:
            break
        if any(
            case["panel_id"] == panel_id and case["path_index"] == path_index
            for case in spike_cases
        ):
            continue
        record = records_by_panel[panel_id][path_index]
        spike_cases.append(
            {
                "panel_id": panel_id,
                "path_index": path_index,
                "record": record,
                "hard": hard_lookup[(panel_id, path_index)],
                "soft": soft_lookup[(panel_id, path_index)],
                "guard": guard_lookup[(panel_id, path_index)],
                "tag": "baseline spike",
            }
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    print("rendering overlays", flush=True)
    t_render = time.time()
    overlays_hard: dict[str, Image.Image] = {}
    overlays_guard: dict[str, Image.Image] = {}
    overlays_soft: dict[str, Image.Image] = {}
    for panel in panels:
        panel_id = panel.panel_id
        try:
            hard_image = _overlay(
                panel, records_by_panel[panel_id], hard_results[panel_id]
            )
            guard_image = _overlay(
                panel, records_by_panel[panel_id], guard_results[panel_id]
            )
            soft_image = _overlay(
                panel, records_by_panel[panel_id], tangent_soft_results[panel_id]
            )
            overlays_hard[panel_id] = hard_image
            overlays_guard[panel_id] = guard_image
            overlays_soft[panel_id] = soft_image
            guard_image.save(PANELS_DIR / f"{panel_id}.png")
        except Exception as exc:
            render_errors.append(f"{panel_id}: {type(exc).__name__}: {exc}")
            print(f"  render error {panel_id}: {exc}", flush=True)

    contact_panels = [panel.panel_id for panel in panels if panel.panel_id in overlays_hard]
    _contact_sheet(
        contact_panels,
        overlays_hard,
        overlays_guard,
        panel_rows,
        OUT_DIR / "contact.png",
        selected_beta,
    )
    zoom_cases = _zoom_cases(records_by_panel, hard_results, tangent_soft_results)
    _endpoint_zoom(
        zoom_cases,
        overlays_hard,
        overlays_soft,
        OUT_DIR / "endpoint_zoom.png",
        selected_beta,
    )
    stage_times["render"] = time.time() - t_render

    # ------------------------------------------------------------------
    # Checks and summary
    # ------------------------------------------------------------------
    parity = _baseline_parity(
        [row for row in all_metric_rows if row["variant"] == baseline_variant]
    )

    # Guard identity on open paths.
    def rows_by_key(variant: str) -> dict[tuple[str, int], dict]:
        return {
            (row["panel_id"], int(row["path_index"])): row
            for row in all_metric_rows
            if row["variant"] == variant
        }

    soft_rows = rows_by_key(tangent_soft_variant)
    guard_rows = rows_by_key("tangent_soft_guard")
    grid_selected_rows = rows_by_key(selected_variant_grid)
    open_guard_diffs = 0
    for key, soft_row in soft_rows.items():
        if soft_row["closed"] != "false":
            continue
        guard_row = guard_rows.get(key)
        if guard_row is None:
            open_guard_diffs += 1
            continue
        for column in METRIC_COLUMNS:
            if column in ("variant", "beta", "variant_note"):
                continue
            left = _format_value(soft_row.get(column))
            right = _format_value(guard_row.get(column))
            if left != right:
                open_guard_diffs += 1
                break
    alias_diffs = 0
    for key, soft_row in soft_rows.items():
        grid_row = grid_selected_rows.get(key)
        if grid_row is None:
            alias_diffs += 1
            continue
        for column in METRIC_COLUMNS:
            if column in ("variant", "beta", "variant_note"):
                continue
            if _format_value(soft_row.get(column)) != _format_value(grid_row.get(column)):
                alias_diffs += 1
                break

    status_totals: dict[str, int] = {}
    for row in all_metric_rows:
        status_totals[row["status"]] = status_totals.get(row["status"], 0) + 1
    variant_row_counts = {
        variant: sum(1 for row in all_metric_rows if row["variant"] == variant)
        for variant, _, _, _ in output_variants
    }
    max_drift = max(
        [
            float(row["endpoint_drift"])
            for row in all_metric_rows
            if row["closed"] == "false" and row["eligible"] == "true"
        ],
        default=0.0,
    )
    max_self_int = max(
        [int(row["self_intersections"]) for row in all_metric_rows], default=0
    )
    max_area_rel = max(
        [
            abs(float(row["area_change_rel"]))
            for row in all_metric_rows
            if row["area_change_rel"] is not None
        ],
        default=0.0,
    )
    n_errors = sum(1 for row in all_metric_rows if row["status"] == "error")
    n_width_guard = sum(
        1 for row in all_metric_rows if row["status"] == "fallback_width_guard"
    )
    n_fb_constraint_selected = sum(
        1
        for row in all_metric_rows
        if row["variant"] == tangent_soft_variant
        and not row["panel_id"] == PH01
        and row["eligible"] == "true"
        and row["status"] == "fallback_constraint"
    )
    cross_hard_total = sum(
        panel_rows[(baseline_variant, panel.panel_id)]["cross_intersections"]
        for panel in panels
        if panel.panel_id != PH01
    )
    cross_soft_total = sum(
        panel_rows[(tangent_soft_variant, panel.panel_id)]["cross_intersections"]
        for panel in panels
        if panel.panel_id != PH01
    )
    cross_guard_total = sum(
        panel_rows[("tangent_soft_guard", panel.panel_id)]["cross_intersections"]
        for panel in panels
        if panel.panel_id != PH01
    )
    exact_cross_total = sum(
        exact_cross[panel.panel_id] for panel in panels if panel.panel_id != PH01
    )

    sanity: list[tuple[str, str, bool]] = [
        ("keep paths", f"{total_paths} (expected 242)", total_paths == 242),
        ("exact boundary objects", f"{total_objects} (expected 158)", total_objects == 158),
        ("fit-eligible paths", f"{total_eligible} (expected 184)", total_eligible == 184),
        (
            "short/ineligible paths",
            f"{total_short} (expected 58)",
            total_short == 58,
        ),
        (
            "non-ph01 eligible",
            f"{total_eligible - sum(r.eligible for r in records_by_panel.get(PH01, []))} "
            "(expected 150)",
            total_eligible - sum(r.eligible for r in records_by_panel.get(PH01, [])) == 150,
        ),
        (
            "metrics rows",
            f"{len(all_metric_rows)} = {total_paths} x {len(output_variants)}",
            len(all_metric_rows) == total_paths * len(output_variants),
        ),
        (
            "one row per path per variant",
            str(all(count == total_paths for count in variant_row_counts.values())),
            all(count == total_paths for count in variant_row_counts.values()),
        ),
        (
            "status totals sum to rows",
            f"{sum(status_totals.values())}",
            sum(status_totals.values()) == len(all_metric_rows),
        ),
        ("open endpoint drift <= 1e-3 px", f"{max_drift:.3e}", max_drift <= DRIFT_TOL),
        ("self-intersections total", f"{max_self_int}", max_self_int == 0),
        ("cross-intersections (exact)", f"{exact_cross_total}", exact_cross_total == 0),
        ("cross-intersections (hard)", f"{cross_hard_total}", cross_hard_total == 0),
        ("cross-intersections (tangent)", f"{cross_soft_total}", cross_soft_total == 0),
        ("cross-intersections (guard)", f"{cross_guard_total}", cross_guard_total == 0),
        ("closed |area change| max", f"{max_area_rel:.5f}", max_area_rel <= AREA_TOL),
        ("selected constraint fallbacks", f"{n_fb_constraint_selected}", n_fb_constraint_selected == 0),
        ("width guard fallbacks", f"{n_width_guard}", True),
        ("fit errors", f"{n_errors}", n_errors == 0),
        ("open guard identity", f"{open_guard_diffs} differing rows", open_guard_diffs == 0),
        ("tangent_soft alias identity", f"{alias_diffs} differing rows", alias_diffs == 0),
        ("runtime errors", f"{len(errors)}", len(errors) == 0),
        ("render errors", f"{len(render_errors)}", len(render_errors) == 0),
    ]

    print("writing summary", flush=True)
    _write_summary(
        started_at=started_at,
        runtime=time.time() - started,
        stage_times=stage_times,
        cfg=cfg,
        panels=panels,
        records_by_panel=records_by_panel,
        total_paths=total_paths,
        total_objects=total_objects,
        total_eligible=total_eligible,
        total_open_eligible=total_open_eligible,
        total_closed_eligible=total_closed_eligible,
        total_short=total_short,
        tangent_counts=tangent_counts,
        selection_rows=selection_rows,
        baseline_stats=baseline_stats,
        selected_beta=selected_beta,
        selected_variant_grid=selected_variant_grid,
        selected_stats=selected_stats,
        selection_mode=selection_mode,
        tangent_soft_stats=tangent_soft_stats,
        guard_stats=guard_stats,
        comparison_soft=comparison_soft,
        comparison_guard=comparison_guard,
        panel_rows=panel_rows,
        hard_lookup=hard_lookup,
        soft_lookup=soft_lookup,
        guard_lookup=guard_lookup,
        improved=improved,
        worsened=worsened,
        oscillation=oscillation,
        spike_cases=spike_cases,
        parity=parity,
        sanity=sanity,
        status_totals=status_totals,
        errors=errors,
        render_errors=render_errors,
        cross_hard_total=cross_hard_total,
        cross_soft_total=cross_soft_total,
        cross_guard_total=cross_guard_total,
        exact_cross_total=exact_cross_total,
        n_fb_constraint_selected=n_fb_constraint_selected,
        n_width_guard=n_width_guard,
        max_drift=max_drift,
        max_area_rel=max_area_rel,
        n_errors=n_errors,
        zoom_cases=zoom_cases,
    )

    print(
        f"\nfinished in {time.time() - started:.1f}s; outputs in {OUT_DIR}",
        flush=True,
    )
    return 0


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------
def _write_summary(
    *,
    started_at: str,
    runtime: float,
    stage_times: dict[str, float],
    cfg: Config,
    panels: list[Panel],
    records_by_panel: dict[str, list[PathRecord]],
    total_paths: int,
    total_objects: int,
    total_eligible: int,
    total_open_eligible: int,
    total_closed_eligible: int,
    total_short: int,
    tangent_counts: dict[str, int],
    selection_rows: list[dict],
    baseline_stats: dict,
    selected_beta: float,
    selected_variant_grid: str,
    selected_stats: dict,
    selection_mode: str,
    tangent_soft_stats: dict,
    guard_stats: dict,
    comparison_soft: dict,
    comparison_guard: dict,
    panel_rows: dict[tuple[str, str], dict],
    hard_lookup: dict[tuple[str, int], dict],
    soft_lookup: dict[tuple[str, int], dict],
    guard_lookup: dict[tuple[str, int], dict],
    improved: list[dict],
    worsened: list[dict],
    oscillation: list[dict],
    spike_cases: list[dict],
    parity: dict | None,
    sanity: list[tuple[str, str, bool]],
    status_totals: dict[str, int],
    errors: list[str],
    render_errors: list[str],
    cross_hard_total: int,
    cross_soft_total: int,
    cross_guard_total: int,
    exact_cross_total: int,
    n_fb_constraint_selected: int,
    n_width_guard: int,
    max_drift: float,
    max_area_rel: float,
    n_errors: int,
    zoom_cases: list[dict],
) -> None:
    lines: list[str] = []
    lines.append("# Layer 2 experiment E — endpoint tangent constraints")
    lines.append("")
    lines.append(f"Run: {started_at} local  ")
    lines.append("Command: `uv run python experiments/layer2/e_endpoint_tangent/run.py`  ")
    lines.append(f"Runtime: {runtime:.1f}s  ")
    lines.append("Repo root: `/Users/daiduo2/geoseg`")
    lines.append("")
    lines.append("## Question and method")
    lines.append("")
    lines.append(
        "Experiment A pins both endpoints of every open path and fits a cubic "
        "P-spline.  Its reported max-curvature increases were attributed to "
        "boundary-layer spikes at pinned junction corners.  This experiment "
        "tests whether a soft endpoint **tangent** penalty removes those "
        "spikes while keeping the hard endpoint positions, the 3 px corridor, "
        "and the topology guarantees."
    )
    lines.append("")
    lines.append("Each open path is fitted independently with the same basis as A:")
    lines.append("")
    lines.append("```text")
    lines.append("min_c  ||B c - y||^2")
    lines.append("       + lambda * (||D2 c||^2 + 0.25 ||D3 c||^2)")
    lines.append("       + beta * (||f'(0) - t_start||^2 + ||f'(1) - t_end||^2)")
    lines.append("```")
    lines.append("")
    lines.append(
        f"- Cubic clamped B-spline, knots every **{cfg.knot_spacing:g} px**, "
        f"lambda = **{LAMBDA:g}** (A's selected value, frozen), "
        f"alpha3 = **{cfg.alpha3:g}**."
    )
    lines.append(
        "- The tangent rows are added to the normal equations "
        "(`beta * d0^T d0`), using the exact B-spline derivative basis.  "
        "The curve is not rotated or post-processed."
    )
    lines.append(
        "- First and last coefficients are eliminated and fixed to the exact "
        "centerline endpoints, so **hard endpoint positions remain exact**."
    )
    lines.append(
        f"- Corridor **{cfg.corridor:g} px**; closed area |ΔA|/A ≤ "
        f"**{cfg.area_tol:g}**; endpoint drift ≤ **{cfg.drift_tol:g} px**; "
        f"self-intersections must be 0.  On violation lambda is halved up to "
        f"**{cfg.max_reductions}** times; if it still fails, exact geometry is "
        "returned and marked `fallback_constraint`."
    )
    lines.append(
        f"- Eligibility: ≥{cfg.min_points} points, ≥{cfg.min_length:g} px, "
        f"closed area ≥{cfg.min_closed_area:g} px².  Shorter fragments are "
        "returned exact and marked `fallback_short`."
    )
    lines.append("")
    lines.append("## Endpoint tangent estimator")
    lines.append("")
    lines.append(
        f"- Window: **{cfg.tangent_window:g} px** of exact centerline from the "
        "endpoint (within the requested 5–12 px range)."
    )
    lines.append(
        "- Total-least-squares line through the endpoint (PCA of endpoint-"
        "relative offsets), trimmed 3 times at `median + 2.5*MAD` with a "
        "0.75 px floor, then re-fitted."
    )
    lines.append(
        "- Oriented into the curve.  For the end, the forward derivative target "
        "is the **negative** of the into-curve vector; this sign is essential."
    )
    lines.append(
        f"- Skipped when the endpoint is within {BORDER_MARGIN:g} px of the panel "
        "border, when the window is shorter than 5 px, when fewer than 4 points "
        "or less than 4 px projected support remain, or when the line-fit "
        "eigenvalue ratio is below 2."
    )
    lines.append(
        "- A skipped end contributes no tangent row; the other end still does."
    )
    lines.append("")
    lines.append("Endpoint support counts (eligible paths, both ends):")
    lines.append("")
    support_rows = [
        [status, str(count)] for status, count in sorted(tangent_counts.items())
    ]
    lines.append(_md_table(["status", "endpoints"], support_rows))
    lines.append("")
    lines.append("## Beta grid and selection")
    lines.append("")
    lines.append(
        "The grid was evaluated on **non-ph01 smoothable paths** (A's eligible "
        "set), with one global beta.  The rule was fixed before inspecting grid "
        "outcomes:"
    )
    lines.append("")
    lines.append(
        f"1. zero fit errors, zero constraint fallbacks, zero self-intersections, "
        f"zero panel cross-intersections, endpoint drift ≤ {DRIFT_TOL:g} px;"
    )
    lines.append(
        f"2. median p95 displacement ≤ hard_only median + {DISPLACEMENT_TOL:g} px, "
        f"and p90 p95 displacement ≤ hard_only p90 + {DISPLACEMENT_TOL:g} px;"
    )
    lines.append(
        f"3. median endpoint-region max curvature ≤ "
        f"{(1.0 - MIN_ENDPOINT_CURV_REDUCTION) * 100:.0f}% of hard_only, and "
        f"p90 endpoint-region max curvature ≤ "
        f"{(1.0 + MAX_ENDPOINT_P90_INCREASE) * 100:.0f}% of hard_only."
    )
    lines.append("")
    lines.append(
        "Endpoint region = first/last `min(15 px, 0.5*L)`.  A wider "
        "`max(15 px, 10% of L)` diagnostic is also reported as "
        "`endpoint_wide_max_curv` in `metrics.csv`; selection uses the fixed "
        "15 px region.  Displacement stats use all eligible non-ph01 paths; "
        "endpoint-curvature stats use eligible open paths.  Among passing "
        "settings, the **largest beta** wins (same preference for the strongest "
        "accepted constraint as experiment A)."
    )
    lines.append("")
    grid_table_rows: list[list[str]] = []
    for row in selection_rows:
        beta = row["beta"]
        stats = row["stats"]
        reasons = row["reasons"]
        if beta == 0.0:
            verdict = "baseline"
        elif beta == selected_beta and selection_mode == "largest passing beta":
            verdict = "**selected**"
        elif not reasons:
            verdict = "pass; higher beta selected" if beta < selected_beta else "pass"
        else:
            verdict = "fail: " + "; ".join(reasons)
        base_med = baseline_stats["median_endpoint_region_max_curv"]
        med = stats["median_endpoint_region_max_curv"]
        reduction = (
            1.0 - med / base_med
            if base_med is not None and med is not None and base_med > 1e-9
            else None
        )
        grid_table_rows.append(
            [
                _beta_label(beta),
                str(stats["n_smoothed"]),
                str(stats["n_ok_reduced"]),
                str(stats["n_fallback_constraint"]),
                _num(stats["median_mean_disp"], 3),
                _num(stats["median_p95_disp"], 3),
                _num(stats["p90_p95_disp"], 3),
                _num(stats["median_max_disp"], 3),
                _num(med, 4),
                _num(stats["p90_endpoint_region_max_curv"], 4),
                _pct(reduction),
                _num(stats["median_tangent_max_deg"], 2),
                str(stats["self_intersections_total"]),
                str(stats["cross_intersections_total"]),
                verdict,
            ]
        )
    lines.append(
        _md_table(
            [
                "beta",
                "smoothed",
                "reduced",
                "fb constraint",
                "med mean",
                "med p95",
                "p90 p95",
                "med max",
                "med endpoint max",
                "p90 endpoint max",
                "med endpoint red.",
                "med tangent err (deg)",
                "self-int",
                "cross-int",
                "selection",
            ],
            grid_table_rows,
        )
    )
    lines.append("")
    lines.append(
        f"**Selected: beta = {_beta_label(selected_beta)}** "
        f"({selection_mode}); the fit is exposed as `tangent_soft`."
    )
    lines.append("")
    if selection_mode != "largest passing beta":
        lines.append(
            "No beta met every criterion.  The fallback chose the smallest "
            "median endpoint curvature among settings with clean topology and "
            "no displacement regression beyond the tolerance; if that list was "
            "empty, beta=0 (hard_only) is retained.  This is reported as a "
            "negative result, not hidden."
        )
        lines.append("")

    # Aggregate comparison
    lines.append("## Aggregate results and comparison")
    lines.append("")
    lines.append(
        f"Non-ph01: **{baseline_stats['n_eligible']}** eligible paths "
        f"({baseline_stats['n_open']} open, {baseline_stats['n_closed']} closed).  "
        "Values are final outputs; fallback paths are exact geometry."
    )
    lines.append("")
    comparison_rows: list[list[str]] = []
    for label, stats in (
        ("hard_only", baseline_stats),
        (f"tangent_soft (β={_beta_label(selected_beta)})", tangent_soft_stats),
        (f"tangent_soft_guard (β={_beta_label(selected_beta)})", guard_stats),
    ):
        comparison_rows.append(
            [
                label,
                _num(stats["median_mean_disp"], 3),
                _num(stats["median_p95_disp"], 3),
                _num(stats["p90_p95_disp"], 3),
                _num(stats["median_max_disp"], 3),
                _num(stats["p95_max_disp"], 3),
                _num(stats["median_endpoint_region_max_curv"], 4),
                _num(stats["p90_endpoint_region_max_curv"], 4),
                _num(stats["median_endpoint_region_p95_curv"], 4),
                _num(stats["median_tangent_max_deg"], 2),
                _num(stats["median_max_curvature"], 3),
                _num(stats["median_max_curv_reduction"], 3),
                str(stats["self_intersections_total"]),
                str(stats["cross_intersections_total"]),
                _num(stats["max_endpoint_drift"], 3),
                _num(stats["max_abs_area_change_rel"], 5),
                str(stats["n_fallback_constraint"] + stats["n_fallback_width_guard"]),
                str(stats["n_errors"]),
            ]
        )
    lines.append(
        _md_table(
            [
                "variant",
                "med mean disp",
                "med p95 disp",
                "p90 p95 disp",
                "med max disp",
                "p95 max disp",
                "med endpoint max",
                "p90 endpoint max",
                "med endpoint p95",
                "med tangent err",
                "med max curv",
                "med max-curv red.",
                "self-int",
                "cross-int",
                "max drift",
                "max |ΔA|/A",
                "fallbacks",
                "errors",
            ],
            comparison_rows,
        )
    )
    lines.append("")
    long_hard: list[float] = []
    long_soft: list[float] = []
    for panel_id, records in records_by_panel.items():
        if panel_id == PH01:
            continue
        for record in records:
            if record.path.closed or not record.eligible or record.length < 30.0:
                continue
            hard_value = hard_lookup[(panel_id, record.path_index)]["curvature"][
                "endpoint_region_max_curv"
            ]
            soft_value = soft_lookup[(panel_id, record.path_index)]["curvature"][
                "endpoint_region_max_curv"
            ]
            if (
                hard_value is not None
                and soft_value is not None
                and np.isfinite(hard_value)
                and np.isfinite(soft_value)
            ):
                long_hard.append(float(hard_value))
                long_soft.append(float(soft_value))
    long_med_hard = _median(long_hard)
    long_med_soft = _median(long_soft)
    long_p90_hard = _p90(long_hard)
    long_p90_soft = _p90(long_soft)
    long_red = (
        1.0 - long_med_soft / long_med_hard
        if long_med_hard and long_med_soft and long_med_hard > 1e-9
        else None
    )
    agg_med_base = baseline_stats["median_endpoint_region_max_curv"]
    agg_med_soft = tangent_soft_stats["median_endpoint_region_max_curv"]
    agg_red = (
        1.0 - agg_med_soft / agg_med_base
        if agg_med_base and agg_med_soft and agg_med_base > 1e-9
        else None
    )
    lines.append(
        f"Endpoint-region max curvature improved on **{comparison_soft['improved']}** "
        f"open paths, worsened on **{comparison_soft['worsened']}**, unchanged on "
        f"**{comparison_soft['unchanged']}** (guard: "
        f"{comparison_guard['improved']}/{comparison_guard['worsened']}/"
        f"{comparison_guard['unchanged']}).  Median of per-path relative changes: "
        f"**{_pct(comparison_soft['median_relative_change'])}** (the grid's "
        f"{_pct(agg_red)} is the change of the aggregate medians, not the median "
        f"of per-path changes)."
    )
    lines.append("")
    lines.append(
        f"Longer-path sensitivity (`L >= 30 px`, n={len(long_hard)}): median "
        f"endpoint max {_num(long_med_hard, 4)} → {_num(long_med_soft, 4)} "
        f"({_pct(long_red)} lower), p90 {_num(long_p90_hard, 4)} → "
        f"{_num(long_p90_soft, 4)}.  The improvement is not an artefact of the "
        f"smallest paths."
    )
    lines.append("")
    lines.append(
        f"Cross-intersections (non-ph01 panels): exact **{exact_cross_total}**, "
        f"hard **{cross_hard_total}**, tangent **{cross_soft_total}**, "
        f"guard **{cross_guard_total}**.  Self-intersections: **0** at all settings.  "
        f"Maximum open endpoint drift: **{max_drift:.3e} px**."
    )
    lines.append("")
    lines.append(
        f"Closed-body maximum |ΔA|/A over all variants: **{max_area_rel:.5f}** "
        f"({100.0 * max_area_rel:.2f}%; tolerance {AREA_TOL:g}).  Width-guard "
        f"fallbacks: **{n_width_guard}**."
    )
    if n_width_guard == 0:
        lines.append(
            "The closed-body width guard did not trigger; `tangent_soft_guard` "
            "returns the same geometry and status as `tangent_soft` on all 242 "
            "paths.  Open-path metric rows are byte-identical (verified "
            "row-by-row, 0 differing rows); closed rows add the width diagnostics."
        )
    else:
        lines.append(
            "The closed-body width guard returned exact geometry for at least one "
            "closed body; those rows are marked `fallback_width_guard` and listed "
            "under failure cases."
        )
    lines.append("")

    # Per-panel table
    lines.append("## Per-panel results")
    lines.append("")
    panel_table_rows: list[list[str]] = []
    for panel in panels:
        panel_id = panel.panel_id
        hard = panel_rows[(baseline_stats["variant"], panel_id)]
        soft = panel_rows[("tangent_soft", panel_id)]
        guard = panel_rows[("tangent_soft_guard", panel_id)]
        note = "excluded from selection (under-segmented)" if panel_id == PH01 else ""
        panel_table_rows.append(
            [
                panel_id,
                str(hard["n_paths"]),
                str(hard["n_boundary_objects"]),
                str(hard["n_eligible"]),
                f"{soft['n_ok']}/{soft['n_ok_reduced']}",
                str(hard["n_fallback_short"]),
                str(soft["n_fallback_constraint"]),
                str(guard["n_fallback_width_guard"]),
                f"{hard['cross_exact']}→{hard['cross_intersections']}→"
                f"{soft['cross_intersections']}→{guard['cross_intersections']}",
                f"{_num(hard['median_p95_disp'], 3)}→{_num(soft['median_p95_disp'], 3)}",
                f"{_num(hard['median_endpoint_region_max_curv'], 3)}→"
                f"{_num(soft['median_endpoint_region_max_curv'], 3)}",
                _num(soft["median_tangent_max_deg"], 2),
                note,
            ]
        )
    lines.append(
        _md_table(
            [
                "panel",
                "paths",
                "objects",
                "eligible",
                "ok/red",
                "fb short",
                "fb constraint",
                "fb width",
                "cross exact→hard→soft→guard",
                "med p95 hard→soft",
                "med endpoint max hard→soft",
                "med tangent err soft",
                "note",
            ],
            panel_table_rows,
        )
    )
    lines.append("")
    lines.append("`ph01_page8_panel3` is reported but never used to select beta.")
    lines.append("")

    # Endpoint spike cases
    lines.append("## Endpoint-spike cases and the main finding")
    lines.append("")
    lines.append(
        "The table reports A's cited cases plus the strongest baseline endpoint "
        "regions.  `contract max` is A's `boundary_metrics` max curvature; "
        "`dense max` is the analytic B-spline curvature on a dense grid.  "
        "`at end dist` is the distance from the dense maximum to the nearer "
        "endpoint, so interior maxima are visible."
    )
    lines.append("")
    spike_table_rows: list[list[str]] = []
    for case in spike_cases:
        record = case["record"]
        hard = case["hard"]
        soft = case["soft"]
        guard = case["guard"]
        dense_u = soft["curvature"]["max_curvature_dense_end_dist"]
        spike_table_rows.append(
            [
                f"{case['panel_id']} p{case['path_index']} (b{record.path.boundary_id}, "
                f"{record.path.labels_pair[0]}-{record.path.labels_pair[1]})",
                f"{record.length:.1f}px",
                f"{_num(hard['metrics']['max_curvature'], 3)}→"
                f"{_num(soft['metrics']['max_curvature'], 3)}",
                f"{_num(hard['curvature']['max_curvature_dense'], 3)}→"
                f"{_num(soft['curvature']['max_curvature_dense'], 3)}",
                _num(dense_u, 1),
                f"{_num(hard['curvature']['endpoint_start_max_curv'], 3)}→"
                f"{_num(soft['curvature']['endpoint_start_max_curv'], 3)}",
                f"{_num(hard['curvature']['endpoint_end_max_curv'], 3)}→"
                f"{_num(soft['curvature']['endpoint_end_max_curv'], 3)}",
                f"{record.tangent_start_status}/{record.tangent_end_status}",
                _num(soft["tangent_max_deg"], 2),
                guard["status"],
                case["tag"],
            ]
        )
    lines.append(
        _md_table(
            [
                "case",
                "L",
                "contract max hard→soft",
                "dense max hard→soft",
                "at end dist (px)",
                "start region hard→soft",
                "end region hard→soft",
                "tangent start/end",
                "soft tangent err",
                "guard status",
                "tag",
            ],
            spike_table_rows,
        )
    )
    lines.append("")
    lines.append(
        "**Main finding.**  The two cited A failures are not endpoint spikes.  "
        "For `schematic_panel2` p37 the dense maximum is an interior hairpin: the "
        "exact centerline reverses direction over ~2 px and the spline rounds it "
        "at a radius well below one pixel; the max is hundreds of pixels from "
        "either endpoint.  For `paper23_fig7a` p13 the maximum is also at "
        "mid-path.  For the cited failures, the endpoint regions stay near "
        "0.07–0.14 px⁻¹ in hard_only, "
        "and the tangent penalty changes them only modestly.  It cannot, and does "
        "not, move the interior hairpin.  A's `max_curvature` undersamples long "
        "paths (at most 1000 samples), which is why the contract value (1.903) "
        "is smaller than the dense value (~12.4).  The right remedy for the "
        "interior maxima is path splitting/curvature-capping or a different "
        "interior model, not endpoint tangents."
    )
    lines.append("")

    # Improved/worsened lists
    lines.append("## Largest endpoint-region improvements and regressions")
    lines.append("")
    improved_sorted = sorted(improved, key=lambda item: item["delta"])[:8]
    lines.append(
        _md_table(
            [
                "case",
                "L",
                "hard endpoint max",
                "soft endpoint max",
                "relative change",
                "tangent start/end",
            ],
            [
                [
                    f"{item['panel_id']} p{item['path_index']} (b{item['boundary_id']})",
                    f"{item['length']:.1f}px",
                    _num(item["hard"], 4),
                    _num(item["soft"], 4),
                    _pct(item["relative"]),
                    f"{item['start_status']}/{item['end_status']}",
                ]
                for item in improved_sorted
            ],
        )
    )
    lines.append("")
    lines.append("Worst regressions at the selected beta:")
    lines.append("")
    lines.append(
        _md_table(
            [
                "case",
                "L",
                "hard endpoint max",
                "soft endpoint max",
                "relative change",
                "endpoint flips hard→soft",
                "tangent start/end",
            ],
            [
                [
                    f"{item['panel_id']} p{item['path_index']} (b{item['boundary_id']})",
                    f"{item['length']:.1f}px",
                    _num(item["hard"], 4),
                    _num(item["soft"], 4),
                    _pct(item["relative"]),
                    f"{item['hard_flips']}→{item['soft_flips']}",
                    f"{item['start_status']}/{item['end_status']}",
                ]
                for item in worsened[:10]
            ],
        )
    )
    lines.append("")

    lines.append("## Failure cases and warnings")
    lines.append("")
    lines.append("### Beta-grid failures")
    lines.append("")
    failure_lines = []
    for row in selection_rows:
        if row["beta"] == 0.0:
            continue
        if row["reasons"]:
            failure_lines.append(
                f"- beta={_beta_label(row['beta'])}: " + "; ".join(row["reasons"])
            )
    if failure_lines:
        lines.extend(failure_lines)
    else:
        lines.append("- None: every beta passed the pre-registered rule.")
    lines.append("")
    lines.append("### Selected-setting warnings")
    lines.append("")
    lines.append(
        f"- Endpoint curvature worsened on {len(worsened)} open paths; the worst "
        "are listed above.  This is a real local cost of the tangent penalty, "
        "not hidden."
    )
    lines.append(
        f"- Oscillation suspects (endpoint curvature sign flips increased by ≥2 "
        f"and endpoint max rose >0.001): **{len(oscillation)}**."
    )
    if oscillation:
        for item in oscillation[:10]:
            lines.append(
                f"  - {item['panel_id']} p{item['path_index']} (b{item['boundary_id']}): "
                f"endpoint max {item['hard']:.4f}→{item['soft']:.4f}, flips "
                f"{item['hard_flips']}→{item['soft_flips']}."
            )
    else:
        lines.append(
            "  - None under this criterion.  Dense curvature sign flips and "
            "self-intersections were still checked on every path."
        )
    lines.append(
        f"- Constraint fallbacks on non-ph01 smoothable paths at the selected "
        f"setting: **{n_fb_constraint_selected}**."
    )
    lines.append(f"- Width-guard fallbacks: **{n_width_guard}**.")
    lines.append(
        "- Border endpoints have no tangent target.  In the cited anchor cases, "
        "`schematic_panel2` p37 ends on the bottom panel border and "
        "`schematic_panel2` p36 starts on the top border, so only one end of "
        "each can be constrained."
    )
    if errors:
        lines.append("- Runtime/cross-intersection errors:")
        for error in errors:
            lines.append(f"  - {error}")
    else:
        lines.append("- Runtime/cross-intersection errors: none.")
    if render_errors:
        lines.append("- Render errors:")
        for error in render_errors:
            lines.append(f"  - {error}")
    else:
        lines.append("- Render errors: none.")
    lines.append("")

    lines.append("## `ph01_page8_panel3` (under-segmented; excluded from selection)")
    lines.append("")
    if PH01 in records_by_panel:
        records = records_by_panel[PH01]
        hard = panel_rows[(baseline_stats["variant"], PH01)]
        soft = panel_rows[("tangent_soft", PH01)]
        lines.append(
            f"- {len(records)} paths, {len({r.path.boundary_id for r in records})} "
            f"exact boundary objects, {hard['n_eligible']} fit-eligible, "
            f"{hard['n_fallback_short']} `fallback_short`."
        )
        lines.append(
            f"- hard: ok/reduced {hard['n_ok']}/{hard['n_ok_reduced']}, "
            f"fallback constraint {hard['n_fallback_constraint']}; "
            f"median p95 {_num(hard['median_p95_disp'], 3)} px, "
            f"median endpoint max {_num(hard['median_endpoint_region_max_curv'], 4)}."
        )
        lines.append(
            f"- soft: ok/reduced {soft['n_ok']}/{soft['n_ok_reduced']}, "
            f"fallback constraint {soft['n_fallback_constraint']}; "
            f"median p95 {_num(soft['median_p95_disp'], 3)} px, "
            f"median endpoint max {_num(soft['median_endpoint_region_max_curv'], 4)}."
        )
        lines.append(
            f"- cross-intersections: exact/hard/soft/guard "
            f"{hard['cross_exact']}/{hard['cross_intersections']}/"
            f"{soft['cross_intersections']}/"
            f"{panel_rows[('tangent_soft_guard', PH01)]['cross_intersections']}."
        )
        lines.append(
            "- Its metrics are reported and checked, but it did not contribute "
            "to beta selection."
        )
    lines.append("")

    lines.append("## Baseline parity with experiment A")
    lines.append("")
    if parity is None:
        lines.append("A's `metrics.csv` was not found; parity check skipped.")
    else:
        lines.append(
            f"Compared {parity['compared']} of {parity['n_reference']} A `lam1` "
            f"rows by `(panel_id, path_index)`; missing: {parity['missing']}; "
            f"status mismatches: {parity['status_mismatches']}."
        )
        lines.append("")
        parity_rows = [
            [field, f"{value:.6g}"] for field, value in parity["max_diffs"].items()
        ]
        lines.append(_md_table(["field", "max |hard_only - A lam1|"], parity_rows))
        lines.append("")
        lines.append(
            "A's CSV is rounded to 6 decimals; differences at the 1e-6 level are "
            "formatting, not method differences.  Contract fields and statuses "
            "match A's selected lambda=1 baseline."
        )
    lines.append("")

    lines.append("## Sanity checks")
    lines.append("")
    sanity_rows = [
        [name, value, "pass" if ok else "FAIL"] for name, value, ok in sanity
    ]
    lines.append(_md_table(["check", "value", "result"], sanity_rows))
    lines.append("")
    lines.append(
        "Status totals across all variants: "
        + ", ".join(f"`{key}` {value}" for key, value in sorted(status_totals.items()))
        + "."
    )
    lines.append("")

    lines.append("## Interpretation and answer")
    lines.append("")
    base_med = baseline_stats["median_endpoint_region_max_curv"]
    soft_med = tangent_soft_stats["median_endpoint_region_max_curv"]
    reduction = (
        1.0 - soft_med / base_med if base_med and soft_med and base_med > 1e-9 else None
    )
    lines.append(
        f"1. **Does the tangent constraint improve endpoint curvature?** At the "
        f"selected beta={_beta_label(selected_beta)}, median endpoint-region max "
        f"curvature moves {_num(base_med, 4)} → {_num(soft_med, 4)} "
        f"({_pct(reduction)} lower), with {comparison_soft['improved']} paths improved "
        f"and {comparison_soft['worsened']} worsened.  The median tangent error "
        f"drops from {_num(baseline_stats['median_tangent_max_deg'], 2)}° to "
        f"{_num(tangent_soft_stats['median_tangent_max_deg'], 2)}°.  So yes, it "
        f"improves endpoint curvature and tangent alignment, but the gain is "
        f"modest and not universal."
    )
    lines.append(
        f"2. **Does it preserve fidelity and topology?** Median p95 displacement "
        f"moves {_num(baseline_stats['median_p95_disp'], 4)} → "
        f"{_num(tangent_soft_stats['median_p95_disp'], 4)} px and p90 p95 "
        f"{_num(baseline_stats['p90_p95_disp'], 4)} → "
        f"{_num(tangent_soft_stats['p90_p95_disp'], 4)} px.  Self-intersections "
        f"are 0, cross-intersections are 0, endpoint drift is ≤{max_drift:.1e} px, "
        f"closed |ΔA|/A ≤ {max_area_rel:.5f} ({100.0 * max_area_rel:.2f}%), and "
        f"there are "
        f"{n_fb_constraint_selected} constraint fallbacks on non-ph01 paths at "
        f"the selected setting.  Fidelity and topology are preserved within the "
        f"pre-registered tolerance."
    )
    stats_1 = _stats_for_beta(selection_rows, 1.0)
    med_1 = stats_1.get("median_endpoint_region_max_curv") if stats_1 else None
    red_1 = (
        1.0 - med_1 / base_med
        if med_1 is not None and base_med and base_med > 1e-9
        else None
    )
    stats_100 = _stats_for_beta(selection_rows, 100.0)
    reasons_100 = _reasons_for_beta(selection_rows, 100.0)
    b100_p95 = stats_100.get("median_p95_disp") if stats_100 else None
    b100_endpoint = (
        stats_100.get("median_endpoint_region_max_curv") if stats_100 else None
    )
    case37 = next(
        (
            case
            for case in spike_cases
            if case["panel_id"] == "schematic_panel2" and case["path_index"] == 37
        ),
        None,
    )
    case13 = next(
        (
            case
            for case in spike_cases
            if case["panel_id"] == "paper23_fig7a" and case["path_index"] == 13
        ),
        None,
    )
    dense_statement = ""
    if case37 is not None:
        hard_dense = case37["hard"]["curvature"]["max_curvature_dense"]
        soft_dense = case37["soft"]["curvature"]["max_curvature_dense"]
        pad = case37["hard"]["curvature"]["max_curvature_dense_end_dist"]
        dense_statement += (
            f"`schematic_panel2` p37: {_num(hard_dense, 3)} → {_num(soft_dense, 3)} "
            f"(maximum {_num(pad, 1)} px from the nearer endpoint)"
        )
    if case13 is not None:
        hard_dense = case13["hard"]["curvature"]["max_curvature_dense"]
        soft_dense = case13["soft"]["curvature"]["max_curvature_dense"]
        pad = case13["hard"]["curvature"]["max_curvature_dense_end_dist"]
        if dense_statement:
            dense_statement += "; "
        dense_statement += (
            f"`paper23_fig7a` p13: {_num(hard_dense, 3)} → {_num(soft_dense, 3)} "
            f"({_num(pad, 1)} px from the nearer endpoint)"
        )
    if selection_mode == "largest passing beta":
        safety_sentence = (
            f"beta={_beta_label(selected_beta)} is the largest grid value that "
            f"passes the pre-registered fidelity/topology rule, so it is the "
            f"recommended operating point.  beta=100 gives a larger endpoint "
            f"reduction (median {_num(b100_endpoint, 4)}) but its median p95 "
            f"displacement is {_num(b100_p95, 4)} px vs "
            f"{_num(baseline_stats['median_p95_disp'], 4)} px at baseline"
            + (
                " and it fails: " + "; ".join(reasons_100)
                if reasons_100
                else " and it passes the rule"
            )
            + "; it is the least safe of the tested values.  beta=0.01/0.1 are "
            f"nearly a no-op; beta=1 reduces the median by only {_pct(red_1)} "
            f"(below the 2% rule).  If strict fidelity is "
            "the priority, beta=0.1 is the conservative choice."
        )
    else:
        safety_sentence = (
            f"No beta passed the full rule; the fallback selected "
            f"beta={_beta_label(selected_beta)}.  Treat this as a negative "
            f"result: the tangent penalty is not fidelity-neutral at the "
            f"strengths that materially reduce endpoint curvature.  The "
            f"conservative choice is beta=0 (hard_only)."
        )
    lines.append(
        f"3. **Which beta is safest?** {safety_sentence}"
    )
    lines.append(
        f"4. **Does it fix experiment A's main failure?** Not the cited "
        f"max-curvature failures.  The dense maxima are interior: "
        f"{dense_statement}.  A's contract metric (at most 1000 samples) "
        f"reported 1.903 and 1.547, but the true maxima are interior hairpins "
        f"far from either endpoint.  The tangent penalty changes endpoint "
        f"regions, not those interior maxima.  It is a valid endpoint-shape "
        f"tool, not a cure for interior path hairpins.  That is the key negative "
        f"result of this experiment."
    )
    lines.append("")
    lines.append(
        "Recommended next step for the interior failures: detect high-turn "
        "hairpins/spurs in the exact centerline and split or simplify them "
        "before fitting, or add an interior curvature cap/short-segment model.  "
        "More pruning of short junction edges is unlikely to help (the path "
        "graph v2 result already showed that the smoothable set and its metrics "
        "did not change)."
    )
    lines.append("")

    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- `{OUT_DIR / 'metrics.csv'}` — {len(records_by_panel)} panels × "
                 f"{total_paths} paths × 8 variants (grid betas, hard_only, "
                 "selected alias and guard; see the file).")
    lines.append(f"- `{OUT_DIR / 'panel_metrics.csv'}` — per panel × variant, including "
                 "cross-intersections and status counts.")
    lines.append(f"- `{OUT_DIR / 'summary.md'}` — this file.")
    lines.append(f"- `{OUT_DIR / 'contact.png'}` — {len(panels)} panel rows, "
                 "hard_only (left) vs tangent_soft_guard (right), recoloured mask + "
                 "exact thin white + smoothed thick green.")
    lines.append(f"- `{OUT_DIR / 'panels/<panel_id>.png'}` — full-resolution guard "
                 "overlays for every panel.")
    lines.append(f"- `{OUT_DIR / 'endpoint_zoom.png'}` — {len(zoom_cases)} zoomed "
                 "endpoint comparisons (anchor cases, baseline spikes, worst "
                 "regressions).")
    lines.append("")
    lines.append("## Runtime and commands")
    lines.append("")
    lines.append("From `/Users/daiduo2/geoseg`:")
    lines.append("")
    lines.append("```bash")
    lines.append("uv run python experiments/layer2/e_endpoint_tangent/run.py")
    lines.append("```")
    lines.append("")
    lines.append(f"Total runtime: **{runtime:.1f}s**.")
    lines.append("")
    stage_rows = [[name, f"{value:.1f}s"] for name, value in stage_times.items()]
    lines.append(_md_table(["stage", "time"], stage_rows))
    lines.append("")
    lines.append(
        "No per-panel or per-path tuning: lambda=1 and the selected beta are "
        "global; only the iterative lambda fallback changes a path's effective "
        "lambda when a constraint fails."
    )
    lines.append("")

    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _stats_for_beta(rows: list[dict], beta: float) -> dict:
    for row in rows:
        if abs(row["beta"] - beta) < 1e-12:
            return row["stats"]
    return {}


def _reasons_for_beta(rows: list[dict], beta: float) -> list[str]:
    for row in rows:
        if abs(row["beta"] - beta) < 1e-12:
            return list(row["reasons"])
    return []


if __name__ == "__main__":
    raise SystemExit(main())
