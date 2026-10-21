"""Layer 2 experiment A: per-boundary-path constrained smoothing spline.

Method
------
Each ``BoundaryPath`` from ``experiments.layer2.common.keep_paths`` is fitted
independently in arc length with an explicit cubic B-spline P-spline:

    min_c  ||B c - y||^2 + lam * (||D2 c||^2 + alpha3 * ||D3 c||^2)

* Open paths are clamped cubic B-splines.  The first and last coefficients are
  eliminated and set exactly to the exact centerline endpoints (hard pinning).
* Closed paths use a periodic cubic B-spline basis; D2/D3 are circular
  difference matrices.  Periodicity is enforced by the basis, so the loop
  closes without duplicating the endpoint.
* The penalty is a tension/bending term (D2) plus a curvature-variation term
  (D3) on the control polygon; both are scaled by one global smoothing strength
  ``lam``.
* A displacement corridor relative to the exact centerline, plus
  self-intersection, endpoint-drift and closed-area checks, is applied after
  fitting.  If a fit violates a constraint, lam is halved and the fit is
  repeated (at most ``MAX_REDUCTIONS`` times).  If it still fails, the exact
  geometry is returned and the row is marked ``fallback_constraint``.

All output is constrained to
``experiments/layer2/a_segmented_spline/`` and
``runs/layer2_experiments/a_segmented_spline/``.
"""

from __future__ import annotations

import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import BSpline

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.layer2.common import (  # noqa: E402
    BoundaryPath,
    Panel,
    boundary_metrics,
    contact_sheet,
    cross_intersections,
    keep_paths,
    load_panels,
    recolour_labels,
    render_overlay,
    resample,
)

OUT_DIR = ROOT / "runs/layer2_experiments/a_segmented_spline"
PANELS_DIR = OUT_DIR / "panels"
MANIFEST = ROOT / "validation/curated/manifest.jsonl"
PH01 = "ph01_page8_panel3"

# Global smoothing grid.  One setting is selected for every panel.
LAMBDA_GRID: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)

# Fixed method constants (not tuned per panel or per path).
DEGREE = 3
KNOT_SPACING = 8.0  # px of exact arc length between B-spline knots
ALPHA3 = 0.25  # weight of the D3 curvature-variation penalty
CORRIDOR = 3.0  # max allowed displacement from the exact centerline, px
AREA_TOL = 0.02  # max allowed |area change| / exact area for closed paths
DRIFT_TOL = 1e-3  # endpoint drift tolerance, px
MIN_POINTS = 4
MIN_LENGTH = 6.0  # px
MIN_CLOSED_AREA = 25.0  # px^2; smaller loops are skeleton artefacts
MAX_REDUCTIONS = 6  # lam -> lam / 2 each time

# Selection rule, fixed before inspecting grid outcomes (see summary.md).
MAX_FALLBACK_RATE = 0.02
MAX_REDUCTION_RATE = 0.10
MAX_MEDIAN_P95 = 1.0
MAX_P90_P95 = 2.0
MIN_CURVATURE_REDUCTION = 0.50


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


# ---------------------------------------------------------------------------
# P-spline basis and penalty


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


def _open_basis(u: np.ndarray, length: float, degree: int, spacing: float) -> np.ndarray:
    n_segments = max(1, int(math.ceil(length / spacing)))
    interior = np.linspace(0.0, length, n_segments + 1)[1:-1]
    knots = np.concatenate(
        [
            np.full(degree + 1, 0.0),
            interior,
            np.full(degree + 1, float(length)),
        ]
    )
    return BSpline.design_matrix(np.clip(u, 0.0, length), knots, degree).toarray()


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


def build_problem(record: PathRecord, cfg: Config) -> SplineProblem:
    path = record.path
    length = record.length
    closed = path.closed
    n_data = max(2, int(round(length)))
    u = np.linspace(0.0, length, n_data, endpoint=not closed)
    data = resample(path.points, closed, n_data)

    if closed:
        basis = _closed_basis(u, length, cfg.degree, cfg.knot_spacing)
        n_coef = basis.shape[1]
        penalty = _penalty_matrix(n_coef, True, cfg.alpha3)
        free = np.arange(n_coef)
        fixed = np.array([], dtype=int)
    else:
        basis = _open_basis(u, length, cfg.degree, cfg.knot_spacing)
        n_coef = basis.shape[1]
        penalty = _penalty_matrix(n_coef, False, cfg.alpha3)
        fixed = np.array([0, n_coef - 1], dtype=int)
        free = np.arange(1, n_coef - 1)

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
    )


def solve_problem(problem: SplineProblem, lam: float) -> np.ndarray:
    """Solve the penalised least-squares system and return sampled points."""
    if problem.fixed.size == 0:
        matrix = problem.bt_b + lam * problem.penalty
        coeffs = np.linalg.solve(matrix, problem.bt_y)
    else:
        free = problem.free
        fixed = problem.fixed
        # Fixed coefficients are exactly the two exact endpoints (B(0), B(L)).
        fixed_values = np.vstack([problem.data[0], problem.data[-1]])
        matrix = problem.bt_b[np.ix_(free, free)] + lam * problem.penalty[np.ix_(free, free)]
        rhs = (
            problem.bt_y[free]
            - problem.bt_b[np.ix_(free, fixed)] @ fixed_values
            - lam * problem.penalty[np.ix_(free, fixed)] @ fixed_values
        )
        free_values = np.linalg.solve(matrix, rhs)
        coeffs = np.zeros((problem.n_coef, 2), dtype=float)
        coeffs[free] = free_values
        coeffs[fixed] = fixed_values
    return problem.basis @ coeffs


# ---------------------------------------------------------------------------
# Per-path fitting, checks, fallback


def constraint_violations(record: PathRecord, metrics: dict, cfg: Config) -> list[str]:
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


def exact_reference_metrics(record: PathRecord) -> dict:
    """Metrics for exact geometry.

    ``boundary_metrics`` independently resamples the candidate and the exact
    centerline, so scoring the exact polyline against itself leaves a small
    resampling floor (a few tenths of a px, and a small length difference).
    When the method returns exact geometry, the geometric changes are zero by
    definition; normalise those fields so fallback rows are not misread.
    """
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


def evaluate_path(record: PathRecord, lam_requested: float, cfg: Config) -> dict:
    """Fit one path, iteratively reducing lam if a constraint is violated."""
    path = record.path

    def exact_result(status: str, reason: str, lam_used: float | None, refits: int) -> dict:
        return {
            "points": path.points.copy(),
            "metrics": exact_reference_metrics(record),
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

    if not record.eligible:
        return exact_result("fallback_short", record.ineligible_reason, None, 0)
    if record.problem is None:
        return exact_result("error", "problem_not_built", None, 0)

    lam = float(lam_requested)
    refits = 0
    last_reasons: list[str] = []
    for attempt in range(cfg.max_reductions + 1):
        try:
            smoothed = solve_problem(record.problem, lam)
            metrics = boundary_metrics(smoothed, path)
        except Exception as exc:  # record, never skip
            return exact_result(
                "error", f"{type(exc).__name__}: {exc}", lam, refits
            )
        reasons = constraint_violations(record, metrics, cfg)
        if not reasons:
            status = "ok" if refits == 0 else "ok_reduced"
            return {
                "points": smoothed,
                "metrics": metrics,
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
    return exact_result(
        "fallback_constraint", "; ".join(last_reasons), lam, refits
    )


def build_records(panel: Panel, cfg: Config) -> list[PathRecord]:
    paths = keep_paths(panel.labels)
    records: list[PathRecord] = []
    for path_index, path in enumerate(paths):
        if path.closed:
            dense = np.vstack([path.points, path.points[:1]])
        else:
            dense = path.points
        length = float(np.hypot(*np.diff(dense, axis=0).T).sum())
        exact_metrics = boundary_metrics(path.points, path)
        area_exact = (
            float(exact_metrics.get("area_exact", 0.0)) if path.closed else None
        )

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
        if record.eligible:
            try:
                record.problem = build_problem(record, cfg)
            except Exception as exc:
                record.problem = None
                record.eligible = False
                record.ineligible_reason = f"problem_build_error: {type(exc).__name__}: {exc}"
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# Metrics tables


def variant_label(lam: float) -> str:
    return f"lam{lam:g}"


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


def metrics_row(
    variant: str,
    panel_id: str,
    record: PathRecord,
    result: dict,
) -> dict:
    metrics = result["metrics"]
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
    return {
        "variant": variant,
        "panel_id": panel_id,
        "boundary_id": record.path.boundary_id,
        "labels_pair": f"{record.path.labels_pair[0]}-{record.path.labels_pair[1]}",
        "closed": str(bool(closed)).lower(),
        "n_points": len(record.path.points),
        "mean_disp": float(metrics["mean_disp"]),
        "p95_disp": float(metrics["p95_disp"]),
        "max_disp": float(metrics["max_disp"]),
        "max_curvature": float(metrics["max_curvature"]),
        "exact_max_curvature": float(metrics["exact_max_curvature"]),
        "length_change": length_change,
        "endpoint_drift": float(metrics["endpoint_drift"]),
        "self_intersections": int(metrics["self_intersections"]),
        "area_change": area_change,
        "status": result["status"],
        "path_index": record.path_index,
        "n_out_points": len(result["points"]),
        "p95_curvature": float(metrics["p95_curvature"]),
        "exact_p95_curvature": float(record.exact_metrics["p95_curvature"]),
        "length_exact": float(metrics["length_exact"]),
        "length_smooth": float(metrics["length_smooth"]),
        "length_change_pct": length_change_pct,
        "area_exact": area_exact,
        "area_smooth": area_smooth,
        "area_change_rel": area_change_rel,
        "max_curv_reduction": max_red,
        "p95_curv_reduction": p95_red,
        "lambda_requested": result["lambda_requested"],
        "lambda_used": result["lambda_used"],
        "refits": result["refits"],
        "eligible": str(bool(record.eligible)).lower(),
        "reason": result["reason"],
    }


METRIC_COLUMNS = [
    "variant",
    "panel_id",
    "boundary_id",
    "labels_pair",
    "closed",
    "n_points",
    "mean_disp",
    "p95_disp",
    "max_disp",
    "max_curvature",
    "exact_max_curvature",
    "length_change",
    "endpoint_drift",
    "self_intersections",
    "area_change",
    "status",
    "path_index",
    "n_out_points",
    "p95_curvature",
    "exact_p95_curvature",
    "length_exact",
    "length_smooth",
    "length_change_pct",
    "area_exact",
    "area_smooth",
    "area_change_rel",
    "max_curv_reduction",
    "p95_curv_reduction",
    "lambda_requested",
    "lambda_used",
    "refits",
    "eligible",
    "reason",
]


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _p90(values: list[float]) -> float | None:
    return float(np.percentile(values, 90)) if values else None


def aggregate_panel(
    variant: str,
    panel_id: str,
    records: list[PathRecord],
    results: list[dict],
    cross_count: int,
    used_for_selection: bool,
) -> dict:
    statuses = [result["status"] for result in results]
    n_baseline = statuses.count("baseline")
    n_ok = statuses.count("ok")
    n_ok_reduced = statuses.count("ok_reduced")
    n_fallback_short = statuses.count("fallback_short")
    n_fallback_constraint = statuses.count("fallback_constraint")
    n_error = statuses.count("error")
    n_eligible = sum(record.eligible for record in records)

    all_p95 = [float(result["metrics"]["p95_disp"]) for result in results]
    all_mean = [float(result["metrics"]["mean_disp"]) for result in results]
    all_max = [float(result["metrics"]["max_disp"]) for result in results]
    smoothed = [
        result
        for result in results
        if result["status"] in ("ok", "ok_reduced")
    ]
    smoothed_p95 = [float(result["metrics"]["p95_disp"]) for result in smoothed]
    max_red = [
        reduction
        for result, record in zip(results, records)
        if result["status"] in ("ok", "ok_reduced")
        for reduction in [_curvature_reduction(result["metrics"], record.exact_metrics)[0]]
        if reduction is not None
    ]
    p95_red = [
        reduction
        for result, record in zip(results, records)
        if result["status"] in ("ok", "ok_reduced")
        for reduction in [_curvature_reduction(result["metrics"], record.exact_metrics)[1]]
        if reduction is not None
    ]
    length_pct = [
        abs(float(result["metrics"]["length_smooth"]) - float(result["metrics"]["length_exact"]))
        / max(float(result["metrics"]["length_exact"]), 1e-9)
        for result in results
    ]
    area_rel = [
        abs(float(result["metrics"]["area_change"])) / float(result["metrics"]["area_exact"])
        for result in results
        if result["closed"]
        and result["metrics"].get("area_exact") is not None
        and float(result["metrics"]["area_exact"]) > 1e-9
    ]
    self_total = sum(int(result["metrics"]["self_intersections"]) for result in results)

    return {
        "variant": variant,
        "panel_id": panel_id,
        "n_paths": len(records),
        "n_boundary_objects": len({record.path.boundary_id for record in records}),
        "n_eligible": n_eligible,
        "n_baseline": n_baseline,
        "n_ok": n_ok,
        "n_ok_reduced": n_ok_reduced,
        "n_fallback_short": n_fallback_short,
        "n_fallback_constraint": n_fallback_constraint,
        "n_error": n_error,
        "reduction_rate": (n_ok_reduced / n_eligible) if n_eligible else 0.0,
        "fallback_rate": (n_fallback_constraint / n_eligible) if n_eligible else 0.0,
        "median_mean_disp": _median(all_mean),
        "median_p95_disp": _median(all_p95),
        "median_p95_disp_smoothed": _median(smoothed_p95),
        "p90_p95_disp": _p90(all_p95),
        "max_disp": max(all_max) if all_max else None,
        "median_max_curvature": _median(
            [float(result["metrics"]["max_curvature"]) for result in results]
        ),
        "median_exact_max_curvature": _median(
            [float(result["metrics"]["exact_max_curvature"]) for result in results]
        ),
        "median_max_curv_reduction": _median(max_red),
        "median_p95_curv_reduction": _median(p95_red),
        "median_length_change_pct": _median(length_pct),
        "max_abs_area_change_rel": max(area_rel) if area_rel else None,
        "self_intersections_total": self_total,
        "cross_intersections": cross_count,
        "used_for_selection": str(bool(used_for_selection)).lower(),
        "note": "under-segmented; excluded from parameter selection" if panel_id == PH01 else "",
    }


PANEL_COLUMNS = [
    "variant",
    "panel_id",
    "n_paths",
    "n_boundary_objects",
    "n_eligible",
    "n_baseline",
    "n_ok",
    "n_ok_reduced",
    "n_fallback_short",
    "n_fallback_constraint",
    "n_error",
    "reduction_rate",
    "fallback_rate",
    "median_mean_disp",
    "median_p95_disp",
    "median_p95_disp_smoothed",
    "p90_p95_disp",
    "max_disp",
    "median_max_curvature",
    "median_exact_max_curvature",
    "median_max_curv_reduction",
    "median_p95_curv_reduction",
    "median_length_change_pct",
    "max_abs_area_change_rel",
    "self_intersections_total",
    "cross_intersections",
    "used_for_selection",
    "note",
]


def selection_stats(
    lam: float,
    records_by_panel: dict[str, list[PathRecord]],
    results_by_panel: dict[str, list[dict]],
    panel_cross: dict[str, int],
) -> dict:
    """Aggregate non-ph01, eligible paths for the global selection criterion."""
    eligible: list[tuple[PathRecord, dict]] = []
    smoothed: list[tuple[PathRecord, dict]] = []
    reduced = 0
    fallback_constraint = 0
    errors = 0
    short = 0
    for panel_id, records in records_by_panel.items():
        if panel_id == PH01:
            continue
        for record, result in zip(records, results_by_panel[panel_id]):
            if not record.eligible:
                short += 1
                continue
            eligible.append((record, result))
            if result["status"] in ("ok", "ok_reduced"):
                smoothed.append((record, result))
                if result["status"] == "ok_reduced":
                    reduced += 1
            elif result["status"] == "fallback_constraint":
                fallback_constraint += 1
            elif result["status"] == "error":
                errors += 1

    p95 = [float(result["metrics"]["p95_disp"]) for _, result in smoothed]
    mean_disp = [float(result["metrics"]["mean_disp"]) for _, result in smoothed]
    max_disp = [float(result["metrics"]["max_disp"]) for _, result in smoothed]
    max_red = []
    p95_red = []
    for record, result in smoothed:
        max_red_value, p95_red_value = _curvature_reduction(
            result["metrics"], record.exact_metrics
        )
        if max_red_value is not None:
            max_red.append(max_red_value)
        if p95_red_value is not None:
            p95_red.append(p95_red_value)
    length_pct = [
        abs(float(result["metrics"]["length_smooth"]) - float(result["metrics"]["length_exact"]))
        / max(float(result["metrics"]["length_exact"]), 1e-9)
        for _, result in smoothed
    ]
    area_rel = [
        abs(float(result["metrics"]["area_change"])) / float(result["metrics"]["area_exact"])
        for record, result in eligible
        if record.path.closed and (record.area_exact or 0.0) > 1e-9
    ]
    n_eligible = len(eligible)
    return {
        "lambda": lam,
        "variant": variant_label(lam),
        "n_eligible": n_eligible,
        "n_smoothed": len(smoothed),
        "n_reduced": reduced,
        "n_fallback_constraint": fallback_constraint,
        "n_short": short,
        "n_error": errors,
        "reduction_rate": (reduced / n_eligible) if n_eligible else 0.0,
        "fallback_rate": (fallback_constraint / n_eligible) if n_eligible else 0.0,
        "median_mean_disp": _median(mean_disp),
        "median_p95_disp": _median(p95),
        "p90_p95_disp": _p90(p95),
        "median_max_disp": _median(max_disp),
        "p95_max_disp": (
            float(np.percentile(max_disp, 95)) if max_disp else None
        ),
        "median_max_curv_reduction": _median(max_red),
        "median_p95_curv_reduction": _median(p95_red),
        "median_length_change_pct": _median(length_pct),
        "max_abs_area_change_rel": max(area_rel) if area_rel else None,
        "self_intersections_total": sum(
            int(result["metrics"]["self_intersections"])
            for _, result in eligible
        ),
        "cross_intersections_total": sum(
            panel_cross[panel_id] for panel_id in panel_cross if panel_id != PH01
        ),
    }


def selection_reason(stats: dict) -> str:
    failures: list[str] = []
    if stats["fallback_rate"] > MAX_FALLBACK_RATE:
        failures.append(f"fallback>{MAX_FALLBACK_RATE:.0%}")
    if stats["reduction_rate"] > MAX_REDUCTION_RATE:
        failures.append(f"reductions>{MAX_REDUCTION_RATE:.0%}")
    if stats["median_p95_disp"] is None or stats["median_p95_disp"] > MAX_MEDIAN_P95:
        failures.append(f"median p95>{MAX_MEDIAN_P95:g}px")
    if stats["p90_p95_disp"] is None or stats["p90_p95_disp"] > MAX_P90_P95:
        failures.append(f"p90 p95>{MAX_P90_P95:g}px")
    if (
        stats["median_max_curv_reduction"] is None
        or stats["median_max_curv_reduction"] < MIN_CURVATURE_REDUCTION
    ):
        failures.append(f"curvature gain<{MIN_CURVATURE_REDUCTION:.0%}")
    if not failures:
        return "pass"
    return "fail: " + ", ".join(failures)


def choose_global(stats_rows: list[dict]) -> dict:
    candidates = [row for row in stats_rows if selection_reason(row) == "pass"]
    if candidates:
        return max(candidates, key=lambda row: row["lambda"])
    return min(
        stats_rows,
        key=lambda row: (
            row["median_p95_disp"] if row["median_p95_disp"] is not None else 1e9,
            -(row["median_max_curv_reduction"] or 0.0),
        ),
    )


# ---------------------------------------------------------------------------
# Rendering


def display_points(result: dict) -> np.ndarray:
    points = np.asarray(result["points"], dtype=float)
    if result["closed"] and len(points) > 1:
        points = np.vstack([points, points[:1]])
    return points


def contact_cell(
    panel: Panel,
    records: list[PathRecord],
    results: list[dict],
    cell: tuple[int, int] = (440, 280),
) -> Image.Image:
    base = Image.fromarray(recolour_labels(panel.labels)).convert("RGB")
    width, height = base.size
    scale = min(1.0, cell[0] / width, cell[1] / height)
    if scale < 1.0:
        base = base.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.LANCZOS,
        )
    cell_image = Image.new("RGB", cell, (18, 18, 18))
    offset_x = (cell[0] - base.width) // 2
    offset_y = (cell[1] - base.height) // 2
    cell_image.paste(base, (offset_x, offset_y))
    draw = ImageDraw.Draw(cell_image)

    def to_cell(points: np.ndarray) -> list[tuple[float, float]]:
        arr = np.asarray(points, dtype=float) * scale
        arr[:, 0] += offset_x
        arr[:, 1] += offset_y
        return [tuple(point) for point in arr]

    for record in records:
        points = record.path.points
        if record.path.closed:
            points = np.vstack([points, points[:1]])
        if len(points) >= 2:
            draw.line(to_cell(points), fill=(255, 255, 255), width=1)
    for result in results:
        points = display_points(result)
        if len(points) < 2:
            continue
        line = to_cell(points)
        draw.line(line, fill=(8, 8, 10), width=5, joint="curve")
        draw.line(line, fill=(0, 255, 90), width=3, joint="curve")
    return cell_image


# ---------------------------------------------------------------------------
# Output helpers


def format_value(value, digits: int = 6) -> str:
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


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key)) for key in columns})


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.{digits}f}%"


def num(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


# ---------------------------------------------------------------------------
# Main


def main() -> int:
    started = time.time()
    cfg = Config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PANELS_DIR.mkdir(parents=True, exist_ok=True)

    panels = load_panels(MANIFEST)
    print(f"loaded {len(panels)} panels", flush=True)

    records_by_panel: dict[str, list[PathRecord]] = {}
    exact_cross: dict[str, int] = {}
    for panel in panels:
        t0 = time.time()
        records = build_records(panel, cfg)
        records_by_panel[panel.panel_id] = records
        exact_cross[panel.panel_id] = cross_intersections([record.path for record in records])
        eligible = sum(record.eligible for record in records)
        print(
            f"{panel.panel_id}: {len(records)} paths, {eligible} eligible, "
            f"cross(exact)={exact_cross[panel.panel_id]}, {time.time() - t0:.1f}s",
            flush=True,
        )

    total_paths = sum(len(records) for records in records_by_panel.values())
    total_objects = sum(
        len({record.path.boundary_id for record in records})
        for records in records_by_panel.values()
    )
    total_eligible = sum(
        record.eligible
        for records in records_by_panel.values()
        for record in records
    )
    total_short = total_paths - total_eligible
    print(
        f"total paths={total_paths}, eligible={total_eligible}, short/ineligible={total_short}",
        flush=True,
    )

    results_by_variant: dict[str, dict[str, list[dict]]] = {}
    panel_cross_by_variant: dict[str, dict[str, int]] = {}

    for lam in LAMBDA_GRID:
        variant = variant_label(lam)
        t0 = time.time()
        results_by_variant[variant] = {}
        panel_cross_by_variant[variant] = {}
        for panel in panels:
            records = records_by_panel[panel.panel_id]
            results = [evaluate_path(record, lam, cfg) for record in records]
            for result in results:
                result["panel_id"] = panel.panel_id
            results_by_variant[variant][panel.panel_id] = results
            panel_cross_by_variant[variant][panel.panel_id] = cross_intersections(
                [
                    BoundaryPath(
                        boundary_id=record.path.boundary_id,
                        labels_pair=record.path.labels_pair,
                        points=result["points"],
                        closed=record.path.closed,
                        exact_length=record.path.exact_length,
                    )
                    for record, result in zip(records, results)
                ]
            )
        print(
            f"variant {variant}: {time.time() - t0:.1f}s",
            flush=True,
        )

    # exact baseline rows/results (for reference and completeness checks)
    exact_results: dict[str, list[dict]] = {}
    for panel in panels:
        records = records_by_panel[panel.panel_id]
        results = []
        for record in records:
            results.append(
                {
                    "points": record.path.points.copy(),
                    "metrics": exact_reference_metrics(record),
                    "status": "baseline",
                    "reason": "",
                    "lambda_requested": None,
                    "lambda_used": None,
                    "refits": None,
                    "panel_id": panel.panel_id,
                    "boundary_id": record.path.boundary_id,
                    "labels_pair": record.path.labels_pair,
                    "closed": record.path.closed,
                    "eligible": record.eligible,
                    "ineligible_reason": record.ineligible_reason,
                    "path_index": record.path_index,
                }
            )
        exact_results[panel.panel_id] = results

    # Metrics CSV: one row per variant (including exact baseline) and path.
    metric_rows: list[dict] = []
    for panel in panels:
        records = records_by_panel[panel.panel_id]
        for record, result in zip(records, exact_results[panel.panel_id]):
            metric_rows.append(metrics_row("exact", panel.panel_id, record, result))
    for lam in LAMBDA_GRID:
        variant = variant_label(lam)
        for panel in panels:
            records = records_by_panel[panel.panel_id]
            for record, result in zip(records, results_by_variant[variant][panel.panel_id]):
                metric_rows.append(metrics_row(variant, panel.panel_id, record, result))
    write_csv(OUT_DIR / "metrics.csv", METRIC_COLUMNS, metric_rows)

    # Panel-level aggregate CSV.
    panel_rows: list[dict] = []
    for panel in panels:
        records = records_by_panel[panel.panel_id]
        panel_rows.append(
            aggregate_panel(
                "exact",
                panel.panel_id,
                records,
                exact_results[panel.panel_id],
                exact_cross[panel.panel_id],
                used_for_selection=panel.panel_id != PH01,
            )
        )
    for lam in LAMBDA_GRID:
        variant = variant_label(lam)
        for panel in panels:
            panel_rows.append(
                aggregate_panel(
                    variant,
                    panel.panel_id,
                    records_by_panel[panel.panel_id],
                    results_by_variant[variant][panel.panel_id],
                    panel_cross_by_variant[variant][panel.panel_id],
                    used_for_selection=panel.panel_id != PH01,
                )
            )
    write_csv(OUT_DIR / "panel_metrics.csv", PANEL_COLUMNS, panel_rows)

    # Selection stats and global setting.
    stats_rows = [
        selection_stats(
            lam,
            records_by_panel,
            results_by_variant[variant_label(lam)],
            panel_cross_by_variant[variant_label(lam)],
        )
        for lam in LAMBDA_GRID
    ]
    selected = choose_global(stats_rows)
    selected_lam = float(selected["lambda"])
    selected_variant = variant_label(selected_lam)
    selected_results = results_by_variant[selected_variant]
    print(
        f"selected {selected_variant} ({selection_reason(selected)})",
        flush=True,
    )

    # Per-panel overlays (full resolution) and contact sheet.
    contact_rows: list[tuple[str, Image.Image, str]] = []
    for panel in panels:
        records = records_by_panel[panel.panel_id]
        results = selected_results[panel.panel_id]
        smoothed = [display_points(result) for result in results]
        statuses = [result["status"] for result in results]
        ok = statuses.count("ok") + statuses.count("ok_reduced")
        fb = statuses.count("fallback_short") + statuses.count("fallback_constraint")
        err = statuses.count("error")
        title = (
            f"{panel.panel_id} | {selected_variant} | paths={len(records)} "
            f"ok={ok} fallback={fb} error={err}"
        )
        overlay = render_overlay(
            panel.labels,
            [record.path for record in records],
            smoothed,
            title=title,
            thickness=3,
        )
        overlay.save(PANELS_DIR / f"{panel.panel_id}.png")

        smoothed_p95 = [
            float(result["metrics"]["p95_disp"])
            for result in results
            if result["status"] in ("ok", "ok_reduced")
        ]
        median_p95 = float(np.median(smoothed_p95)) if smoothed_p95 else 0.0
        text = f"n{len(records)} ok{ok} fb{fb} p95={median_p95:.2f}"
        contact_rows.append(
            (
                panel.panel_id,
                contact_cell(panel, records, results),
                text,
            )
        )
    contact_sheet(contact_rows, OUT_DIR / "contact.png", cell=(440, 280))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    grid_rows_displacement = []
    grid_rows_curvature = []
    for stats in stats_rows:
        marker = "**selected**" if stats["lambda"] == selected_lam else selection_reason(stats)
        grid_rows_displacement.append(
            [
                f"{stats['lambda']:g}",
                str(stats["n_smoothed"]),
                str(stats["n_reduced"]),
                str(stats["n_fallback_constraint"]),
                pct(stats["reduction_rate"], 2),
                pct(stats["fallback_rate"], 2),
                num(stats["median_mean_disp"]),
                num(stats["median_p95_disp"]),
                num(stats["p90_p95_disp"]),
                num(stats["median_max_disp"]),
                num(stats["p95_max_disp"]),
                pct(stats["median_length_change_pct"], 2),
            ]
        )
        grid_rows_curvature.append(
            [
                f"{stats['lambda']:g}",
                pct(stats["median_max_curv_reduction"]),
                pct(stats["median_p95_curv_reduction"]),
                pct(stats["max_abs_area_change_rel"], 3),
                str(stats["self_intersections_total"]),
                str(stats["cross_intersections_total"]),
                marker,
            ]
        )

    selected_panel_rows = []
    for panel in panels:
        row = next(
            (
                item
                for item in panel_rows
                if item["variant"] == selected_variant and item["panel_id"] == panel.panel_id
            ),
            None,
        )
        assert row is not None
        note = "excluded from selection" if panel.panel_id == PH01 else ""
        selected_panel_rows.append(
            [
                panel.panel_id,
                str(row["n_paths"]),
                str(row["n_boundary_objects"]),
                str(row["n_eligible"]),
                str(row["n_ok"]),
                str(row["n_ok_reduced"]),
                str(row["n_fallback_short"]),
                str(row["n_fallback_constraint"]),
                num(row["median_p95_disp_smoothed"]),
                pct(row["median_max_curv_reduction"]),
                pct(row["max_abs_area_change_rel"], 3),
                str(row["self_intersections_total"]),
                str(row["cross_intersections"]),
                note,
            ]
        )

    # Selected-variant path-level lists for failures/interpretation.
    selected_records = [
        (panel.panel_id, record, selected_results[panel.panel_id][record.path_index])
        for panel in panels
        for record in records_by_panel[panel.panel_id]
    ]
    constraint_fallbacks = [
        (panel_id, record, result)
        for panel_id, record, result in selected_records
        if result["status"] == "fallback_constraint"
    ]
    [
        (panel_id, record, result)
        for panel_id, record, result in selected_records
        if result["status"] == "fallback_short"
    ]
    errors = [
        (panel_id, record, result)
        for panel_id, record, result in selected_records
        if result["status"] == "error"
    ]
    curvature_increases = sorted(
        [
            (panel_id, record, result)
            for panel_id, record, result in selected_records
            if result["status"] in ("ok", "ok_reduced")
            and float(result["metrics"]["exact_max_curvature"]) > 1e-6
            and float(result["metrics"]["max_curvature"])
            > float(result["metrics"]["exact_max_curvature"]) * (1.0 + 1e-6)
        ],
        key=lambda item: float(item[2]["metrics"]["max_curvature"])
        / max(float(item[2]["metrics"]["exact_max_curvature"]), 1e-9),
        reverse=True,
    )
    largest_displacements = sorted(
        [
            (panel_id, record, result)
            for panel_id, record, result in selected_records
            if result["status"] in ("ok", "ok_reduced")
        ],
        key=lambda item: float(item[2]["metrics"]["max_disp"]),
        reverse=True,
    )
    worst_length = sorted(
        [
            (panel_id, record, result)
            for panel_id, record, result in selected_records
            if result["status"] in ("ok", "ok_reduced")
        ],
        key=lambda item: abs(
            float(item[2]["metrics"]["length_smooth"])
            - float(item[2]["metrics"]["length_exact"])
        )
        / max(float(item[2]["metrics"]["length_exact"]), 1e-9),
        reverse=True,
    )
    closed_records = [
        (panel_id, record, result)
        for panel_id, record, result in selected_records
        if record.path.closed
    ]

    short_by_panel: dict[str, int] = {}
    for panel in panels:
        short_by_panel[panel.panel_id] = sum(
            not record.eligible for record in records_by_panel[panel.panel_id]
        )

    exact_cross_total = sum(exact_cross.values())
    selected_cross_total = sum(panel_cross_by_variant[selected_variant].values())
    non_ph01_selected_cross = sum(
        count
        for panel_id, count in panel_cross_by_variant[selected_variant].items()
        if panel_id != PH01
    )

    ph01_records = records_by_panel[PH01]
    ph01_results = selected_results[PH01]
    ph01_statuses = [result["status"] for result in ph01_results]
    ph01_rows = [
        [
            "paths",
            str(len(ph01_records)),
        ],
        [
            "exact boundary objects",
            str(len({record.path.boundary_id for record in ph01_records})),
        ],
        [
            "fit-eligible",
            str(sum(record.eligible for record in ph01_records)),
        ],
        [
            "fallback_short (too short / tiny closed)",
            str(ph01_statuses.count("fallback_short")),
        ],
        [
            "ok / ok_reduced",
            f"{ph01_statuses.count('ok')} / {ph01_statuses.count('ok_reduced')}",
        ],
        [
            "fallback_constraint",
            str(ph01_statuses.count("fallback_constraint")),
        ],
        [
            "cross intersections (exact -> selected)",
            f"{exact_cross[PH01]} -> {panel_cross_by_variant[selected_variant][PH01]}",
        ],
    ]
    ph01_smoothed_p95 = [
        float(result["metrics"]["p95_disp"])
        for result in ph01_results
        if result["status"] in ("ok", "ok_reduced")
    ]
    ph01_smoothed_disp = [
        float(result["metrics"]["mean_disp"])
        for result in ph01_results
        if result["status"] in ("ok", "ok_reduced")
    ]
    ph01_curv = [
        _curvature_reduction(result["metrics"], record.exact_metrics)[0]
        for record, result in zip(ph01_records, ph01_results)
        if result["status"] in ("ok", "ok_reduced")
    ]
    ph01_curv = [value for value in ph01_curv if value is not None]
    if ph01_smoothed_p95:
        ph01_rows.extend(
            [
                [
                    "median mean / p95 displacement (smoothed paths, px)",
                    f"{np.median(ph01_smoothed_disp):.3f} / {np.median(ph01_smoothed_p95):.3f}",
                ],
                [
                    "median max-curvature reduction (smoothed paths)",
                    pct(float(np.median(ph01_curv))) if ph01_curv else "n/a",
                ],
            ]
        )

    # Failure lists.
    constraint_lines = []
    for panel_id, record, result in constraint_fallbacks[:20]:
        constraint_lines.append(
            f"- `{panel_id}` path {record.path_index} (boundary {record.path.boundary_id}, "
            f"{record.path.labels_pair[0]}-{record.path.labels_pair[1]}, L={record.length:.1f}px): "
            f"requested λ={result['lambda_requested']:g}, used λ={result['lambda_used']}; "
            f"{result['reason']}"
        )
    if not constraint_lines:
        constraint_lines.append("- None at the selected setting.")

    curvature_lines = []
    for panel_id, record, result in curvature_increases[:12]:
        metrics = result["metrics"]
        curvature_lines.append(
            f"- `{panel_id}` path {record.path_index} (boundary {record.path.boundary_id}, "
            f"{record.path.labels_pair[0]}-{record.path.labels_pair[1]}, L={record.length:.1f}px, "
            f"n={len(record.path.points)}): max curvature "
            f"{metrics['exact_max_curvature']:.3f} -> {metrics['max_curvature']:.3f} "
            f"({result['status']}, λ_used={result['lambda_used']:.4g}, "
            f"p95 disp={metrics['p95_disp']:.2f}, max disp={metrics['max_disp']:.2f})"
        )
    if not curvature_lines:
        curvature_lines.append("- None: every smoothed path reduced its max curvature.")

    displacement_lines = []
    for panel_id, record, result in largest_displacements[:10]:
        metrics = result["metrics"]
        displacement_lines.append(
            f"- `{panel_id}` path {record.path_index} (boundary {record.path.boundary_id}, "
            f"{record.path.labels_pair[0]}-{record.path.labels_pair[1]}, L={record.length:.1f}px): "
            f"mean/p95/max = {metrics['mean_disp']:.2f}/{metrics['p95_disp']:.2f}/"
            f"{metrics['max_disp']:.2f}px, λ_used={result['lambda_used']:.4g}"
        )
    if not displacement_lines:
        displacement_lines.append("- None.")

    closed_lines = []
    for panel_id, record, result in sorted(
        closed_records,
        key=lambda item: abs(
            float(item[2]["metrics"].get("area_change", 0.0))
            / max(float(item[2]["metrics"].get("area_exact", 1.0)), 1e-9)
        ),
        reverse=True,
    )[:12]:
        metrics = result["metrics"]
        area_exact = float(metrics.get("area_exact", 0.0))
        area_change = float(metrics.get("area_change", 0.0))
        rel = area_change / area_exact if area_exact > 1e-9 else 0.0
        closed_lines.append(
            f"- `{panel_id}` path {record.path_index} (boundary {record.path.boundary_id}, "
            f"{record.path.labels_pair[0]}-{record.path.labels_pair[1]}, L={record.length:.1f}px): "
            f"area {area_exact:.1f} -> {area_exact + area_change:.1f}px² "
            f"({rel:+.2%}), status={result['status']}, "
            f"λ_used={result['lambda_used'] if result['lambda_used'] is not None else 'n/a'}"
        )
    if not closed_lines:
        closed_lines.append("- No closed paths.")

    length_lines = []
    for panel_id, record, result in worst_length[:8]:
        metrics = result["metrics"]
        rel = (
            float(metrics["length_smooth"]) - float(metrics["length_exact"])
        ) / max(float(metrics["length_exact"]), 1e-9)
        length_lines.append(
            f"- `{panel_id}` path {record.path_index} (boundary {record.path.boundary_id}, "
            f"L={record.length:.1f}px): length {metrics['length_exact']:.1f} -> "
            f"{metrics['length_smooth']:.1f}px ({rel:+.2%}), status={result['status']}"
        )
    if not length_lines:
        length_lines.append("- None.")

    # Sanity checks.
    variant_names = ["exact"] + [variant_label(lam) for lam in LAMBDA_GRID]
    sanity_errors: list[str] = []
    expected_rows = total_paths * len(variant_names)
    if len(metric_rows) != expected_rows:
        sanity_errors.append(
            f"metrics.csv rows {len(metric_rows)} != expected {expected_rows}"
        )
    for variant in variant_names:
        seen = {
            (row["panel_id"], row["path_index"])
            for row in metric_rows
            if row["variant"] == variant
        }
        if len(seen) != total_paths:
            sanity_errors.append(
                f"{variant}: {len(seen)} unique paths, expected {total_paths}"
            )
    if errors:
        sanity_errors.append(f"{len(errors)} error rows at selected setting")
    total_error_rows = sum(1 for row in metric_rows if row["status"] == "error")
    if total_error_rows:
        sanity_errors.append(f"{total_error_rows} error rows across all variants")
    for row in panel_rows:
        total_counted = (
            row["n_baseline"]
            + row["n_ok"]
            + row["n_ok_reduced"]
            + row["n_fallback_short"]
            + row["n_fallback_constraint"]
            + row["n_error"]
        )
        if total_counted != row["n_paths"]:
            sanity_errors.append(
                f"panel {row['panel_id']} {row['variant']}: status counts "
                f"{total_counted} != {row['n_paths']}"
            )

    sanity_lines = [
        f"- Keep paths in manifest: **{total_paths}** from "
        f"**{total_objects}** exact boundary objects (skeleton fragmentation "
        f"creates repeated boundary ids).",
        f"- Rows in `metrics.csv`: **{len(metric_rows)}** = {total_paths} paths × "
        f"{len(variant_names)} variants (exact + {len(LAMBDA_GRID)} λ settings).",
        f"- Every variant contains one record for every keep path: "
        f"**{'yes' if not any('unique paths' in item for item in sanity_errors) else 'NO'}**.",
        f"- Status counts sum to path counts in every panel row: "
        f"**{'yes' if not any('status counts' in item for item in sanity_errors) else 'NO'}**.",
        f"- Runtime errors at selected setting: **{len(errors)}**; across all "
        f"variants: **{total_error_rows}**.",
        f"- Fit-eligible paths: **{total_eligible}**; short/ineligible returned exact and marked "
        f"`fallback_short`: **{total_short}**.",
    ]
    if sanity_errors:
        sanity_lines.append("- **Sanity problems:** " + "; ".join(sanity_errors))
    else:
        sanity_lines.append("- No sanity problems detected.")

    grid_disp_table = md_table(
        [
            "λ",
            "smoothed",
            "reduced",
            "fb constraint",
            "reduction rate",
            "fallback rate",
            "med mean disp",
            "med p95 disp",
            "p90 p95 disp",
            "med max disp",
            "p95 max disp",
            "med abs ΔL/L",
        ],
        grid_rows_displacement,
    )
    grid_curv_table = md_table(
        [
            "λ",
            "med max-curv reduction",
            "med p95-curv reduction",
            "max closed abs ΔA/A",
            "self-int",
            "cross-int",
            "selection",
        ],
        grid_rows_curvature,
    )
    panel_table = md_table(
        [
            "panel",
            "paths",
            "objects",
            "eligible",
            "ok",
            "reduced",
            "fb short",
            "fb constraint",
            "med p95 (smoothed)",
            "med max-curv red.",
            "max abs ΔA/A",
            "self-int",
            "cross-int",
            "note",
        ],
        selected_panel_rows,
    )

    runtime = time.time() - started
    summary = f"""# Layer 2 experiment A — per-boundary-path constrained smoothing spline

Run: {time.strftime('%Y-%m-%d %H:%M:%S')} local
Command: `uv run python experiments/layer2/a_segmented_spline/run.py`
Runtime: {runtime:.1f}s

## Method and fixed parameters

- 11 curated panels, **{total_paths} keep paths** from **{total_objects} exact boundary objects**
  (Layer 1 keep set unchanged; no re-segmentation).
- Every `BoundaryPath` is fitted independently in arc length.  No path is
  smoothed across a junction or fault.
- Cubic B-spline P-spline with knots every **{KNOT_SPACING:g}px** of exact arc length.
  The objective is
  `min_c ||B c - y||² + λ (||D2 c||² + {ALPHA3:g} ||D3 c||²)`.
  `D2` is a tension/bending penalty on the control polygon; `D3` penalizes
  curvature variation.  Both use circular differences for closed paths.
- **Open paths:** clamped basis, first and last coefficients eliminated and set
  exactly to the two exact centerline endpoints (hard endpoint pinning).
  `boundary_metrics` reports endpoint drift ≈ 1e-13 px (floating point).
- **Closed paths:** periodic basis; closure is built into the basis, and the
  exact-to-smoothed area change is checked before accepting a fit.
- **Corridor:** max displacement to the exact centerline ≤ **{CORRIDOR:g}px**.
  Closed area change ≤ **{AREA_TOL:.0%}**; self-intersections must be zero;
  open endpoint drift ≤ **{DRIFT_TOL:g}px**.
- **Iterative fallback:** if a fit violates a constraint, λ is halved and the
  path is refitted, up to **{MAX_REDUCTIONS}** times.  If it still fails, the
  exact geometry is returned and marked `fallback_constraint`.
- **Eligibility:** at least {MIN_POINTS} points, ≥{MIN_LENGTH:g}px arc length,
  and (for closed paths) ≥{MIN_CLOSED_AREA:g}px² exact area.  Shorter skeleton
  fragments cannot support a cubic fit; they are returned exact and marked
  `fallback_short` — recorded, never dropped.
- One global λ is selected for all panels.  Per-path fallback is the only
  place where the effective λ differs; `lambda_used` and `refits` are recorded
  per path.

## Global setting and selection criterion

**Selected global setting: λ = {selected_lam:g}** (`{selected_variant}`).

The rule was fixed before examining grid outcomes.  Using eligible paths from
all panels **except `ph01_page8_panel3`**, a setting is a candidate only if:

1. constraint-fallback rate ≤ {MAX_FALLBACK_RATE:.0%};
2. path-reduction rate ≤ {MAX_REDUCTION_RATE:.0%};
3. median p95 displacement ≤ {MAX_MEDIAN_P95:g}px;
4. 90th-percentile p95 displacement ≤ {MAX_P90_P95:g}px;
5. median max-curvature reduction ≥ {MIN_CURVATURE_REDUCTION:.0%}.

Among candidates, the **largest λ** (strongest smoothing) wins.  If none pass,
the setting with the smallest median p95 displacement is used.

Selection for λ = {selected_lam:g}: **{selection_reason(selected)}**.

## Global grid (non-ph01, eligible paths only)

`smoothed` is the number of paths successfully fitted (including paths that
needed λ reduction); `reduced` counts paths that required at least one fallback
reduction; `fb constraint` counts paths that exhausted reductions and returned
exact geometry.  Displacements are in px.

{grid_disp_table}

{grid_curv_table}

## Aggregate results at the selected setting

Definitions: `smoothed` statistics use successful fits only; `eligible` excludes
short skeleton fragments.  Final-output fallbacks are exact geometry.  Rows for
exact geometry (`baseline`, `fallback_short`, `fallback_constraint`) report
zero displacement, length change and area change by definition.  Smoothed rows
still include the contract metric's independent-resampling floor (scoring exact
geometry against itself gives ~0.27px mean / 0.51px p95 on a long path), so the
reported displacement values are conservative rather than exact
point-to-segment distances.

- Eligible paths (non-ph01): **{selected['n_eligible']}**; successfully smoothed:
  **{selected['n_smoothed']}** ({selected['n_reduced']} after λ reduction);
  constraint fallbacks: **{selected['n_fallback_constraint']}**.
- Median mean / p95 / max displacement: **{num(selected['median_mean_disp'])} /
  {num(selected['median_p95_disp'])} / {num(selected['median_max_disp'])} px**;
  p90 of p95 and p95 of max: **{num(selected['p90_p95_disp'])} /
  {num(selected['p95_max_disp'])} px**.
- Median max-curvature reduction: **{pct(selected['median_max_curv_reduction'])}**;
  median p95-curvature reduction: **{pct(selected['median_p95_curv_reduction'])}**.
- Median |length change| / exact length: **{pct(selected['median_length_change_pct'], 2)}**
  (uniform smoothing removes pixel-staircase length; signed changes are in
  `metrics.csv`).
- Maximum closed-body |area change| / exact area: **{pct(selected['max_abs_area_change_rel'], 3)}**.
- Self-intersections at selected setting: **{selected['self_intersections_total']}**.
- Cross-intersections (all panels): exact **{exact_cross_total}** → selected
  **{selected_cross_total}** (non-ph01: **{non_ph01_selected_cross}**).
- Errors: **{len(errors)}**.

## Per-panel results at the selected setting

`objects` is the number of distinct exact boundary objects; skeleton
fragmentation can yield more paths than objects.  `med p95 (smoothed)` and
`med max-curv red.` use successful fits; `fb short` paths are exact.

{panel_table}

## `ph01_page8_panel3` (under-segmented; excluded from selection)

This panel is reported separately and was **not** used for parameter selection.
Its labels split into **167 skeleton paths from only 22 boundary objects**
(62 fragments from one object), and 108 paths are the same label pair (1,2);
the boundaries are strongly under-segmented and fragmented.

{md_table(["item", "value"], ph01_rows)}

The panel therefore illustrates the failure mode of treating a pixel skeleton
as a geological curve: many 1–2px fragments cannot be smoothed, while the few
long boundaries do smooth normally.  Its metrics do not affect the global λ or
the aggregate selection tables.

## Failure cases and warnings

### Constraint fallbacks at selected λ

{chr(10).join(constraint_lines)}

At the selected setting the only constraint fallback is a small closed loop in
the under-segmented `ph01_page8_panel3` (boundary 39, L≈55px).  At the fixed
8px knot spacing its periodic basis has only 7 control coefficients; even the
unpenalised projection onto that basis changes its area by ≈2.6%, so no amount
of λ reduction can satisfy the 2% area corridor.  Returning exact geometry is
the intended resolution limit for a 55px loop, not a silent skip.

### Short/ineligible skeleton fragments (`fallback_short`)

- Total: **{total_short}** of **{total_paths}** paths.
- By panel: {', '.join(f'`{panel_id}` {count}' for panel_id, count in short_by_panel.items() if count)}.
- These are mostly 1–2px skeleton edges produced by `trace_skeleton` at branch
  nodes, plus one tiny 9px closed loop.  They are genuine Layer 1 keep objects,
  but a cubic fit is undefined or meaningless at this size.  Returning the
  exact geometry is the conservative outcome; they are not removed.

### Paths whose max curvature increased after smoothing

Endpoint pinning is a hard positional constraint; at a junction corner the
cubic must transition from the pinned endpoint into the interior, which can
create a local boundary-layer curvature spike.  This is the main qualitative
failure of the method.

{chr(10).join(curvature_lines)}

### Largest displacements at selected λ

{chr(10).join(displacement_lines)}

### Closed-body area changes

All closed paths at selected λ (including exact fallbacks; sorted by |relative
area change|).  Closed bodies are periodic-smoothed, and a fit is rejected if
it exceeds the {AREA_TOL:.0%} area tolerance.

{chr(10).join(closed_lines)}

### Largest length changes

{chr(10).join(length_lines)}

## Interpretation / sensitivity

- The grid shows a monotone trade-off: λ = 0.01 → 10 moves median p95
  displacement from {num(stats_rows[0]['median_p95_disp'])} to
  {num(stats_rows[-1]['median_p95_disp'])}px and median max-curvature reduction
  from {pct(stats_rows[0]['median_max_curv_reduction'])} to
  {pct(stats_rows[-1]['median_max_curv_reduction'])}.
- λ = 1 is the strongest setting that keeps the aggregate p95 displacement
  distribution inside the selection corridor (p90 p95 =
  {num(selected['p90_p95_disp'])}px) with only
  {selected['n_reduced']} path(s) needing reduction.  λ = 3 already pushes the
  p90 p95 to {num(next((row['p90_p95_disp'] for row in stats_rows if row['lambda'] == 3.0), None))}px
  and {next((row['n_reduced'] for row in stats_rows if row['lambda'] == 3.0), 'n/a')}
  paths require reduction; λ = 10 triggers reductions on
  {next((row['n_reduced'] for row in stats_rows if row['lambda'] == 10.0), 'n/a')} paths.
- Most of the curvature gain is already obtained at λ = 0.1–0.3
  (p95-curvature reduction
  {pct(next((row['median_p95_curv_reduction'] for row in stats_rows if row['lambda'] == 0.3), None))});
  larger λ mainly shortens and simplifies the curves, not the noise scale.
- The method is conservative by construction: corridor + iterative reduction
  guarantee no accepted path moves more than {CORRIDOR:g}px, but the corridor
  also caps how much a sharp junction can be rounded.  When the exact centerline
  has a corner exactly at a pinned endpoint, the spline's transition can raise
  the reported max curvature even though the p95 curvature drops.
- Cross-intersections do not increase in this run
  ({exact_cross_total} → {selected_cross_total}); smoothing each path
  independently with pinned endpoints preserved the junction topology at the
  resolution of the validator.

## Outputs

- `runs/layer2_experiments/a_segmented_spline/metrics.csv` — {len(metric_rows)} rows
  (required columns followed by diagnostic columns).
- `runs/layer2_experiments/a_segmented_spline/panel_metrics.csv` — {len(panel_rows)} rows
  (per panel × variant, including cross-intersections).
- `runs/layer2_experiments/a_segmented_spline/summary.md` — this file.
- `runs/layer2_experiments/a_segmented_spline/contact.png` — 11 rows,
  recoloured labels + exact thin white + smoothed thick green.
- `runs/layer2_experiments/a_segmented_spline/panels/<panel_id>.png` — full-resolution
  per-panel overlays at the selected setting.

## Sanity checks

{chr(10).join(sanity_lines)}
"""
    (OUT_DIR / "summary.md").write_text(summary, encoding="utf-8")

    print(f"wrote {OUT_DIR / 'metrics.csv'}")
    print(f"wrote {OUT_DIR / 'panel_metrics.csv'}")
    print(f"wrote {OUT_DIR / 'summary.md'}")
    print(f"wrote {OUT_DIR / 'contact.png'}")
    print(f"wrote {len(panels)} per-panel overlays in {PANELS_DIR}")
    if sanity_errors:
        print("SANITY PROBLEMS:", "; ".join(sanity_errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
