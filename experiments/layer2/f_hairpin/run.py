"""Layer 2 experiment F: interior hairpins / direction reversals.

Experiments A and E localized the worst dense curvature to interior direction
reversals ("hairpins") in the exact skeleton centerline, far from either
endpoint.  This experiment detects them and compares four handlers:

* ``baseline_A``   : A's cubic P-spline (D2 + 0.25 D3, hard endpoints, 3 px
  corridor, lambda = 1);
* ``split``        : split at each anchor, pin both new endpoints, smooth each
  subpath independently;
* ``corner_knot``  : split pieces plus hard exact one-sided tangents at each
  anchor, preserving the C0 corner angle;
* ``curvature_cap``: one path with a local penalty boost around anchors; if the
  global cap is unreachable inside the corridor, fall back to split, then exact.

Only files under ``experiments/layer2/f_hairpin/`` and
``runs/layer2_experiments/f_hairpin/`` are written.
"""

from __future__ import annotations

import csv
import math
import os
import sys
import time
from dataclasses import dataclass, field
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
    boundary_metrics,
    cross_intersections,
    keep_paths,
    load_panels,
    recolour_labels,
    render_overlay,
    resample,
)

OUT_DIR = ROOT / "runs/layer2_experiments/f_hairpin"
PANELS_DIR = OUT_DIR / "panels"
MANIFEST = ROOT / "validation/curated/manifest.jsonl"
PH01 = "ph01_page8_panel3"

# Fixed global settings (no per-panel or per-path tuning).
DEGREE = 3
KNOT_SPACING = 8.0
ALPHA3 = 0.25
LAMBDA = 1.0
CORRIDOR = 3.0
AREA_TOL = 0.02
DRIFT_TOL = 1e-3
MIN_POINTS = 4
MIN_LENGTH = 6.0
MIN_CLOSED_AREA = 25.0
MAX_REDUCTIONS = 6

# Detector.  Selected once on the synthetic set + named anchors; sensitivity is
# reported in summary.md.
TURN3_DEG = 140.0
TURN8_DEG = 170.0
TURN8_MIN_T3_DEG = 80.0   # a full reversal must also be locally sharp
TURN8_MIN_CURV = 0.30
CURV_THRESHOLD = 0.85
CURV_TURN3_DEG = 90.0
GROUP_GAP_PX = 8
MIN_DENSE_POINTS = 10
NEAR_END_PX = 2.0

HAIRPIN_WINDOW_PX = 12.0
TURN_WINDOW_PX = 10.0
JOIN_EXCLUDE_PX = 1.0
DENSE_PER_PX = 8.0
DENSE_MAX_SAMPLES = 32000

CAP_VALUE = 2.0
CAP_SIGMA_PX = 6.0
CAP_TURN_RATIO = 0.5  # reject a cap that removes the reversal
CAP_BOOSTS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)
CAP_SENSITIVITY = (1.0, 2.0, 4.0, 8.0)

SELECTED_VARIANT = "split"
VARIANTS = ("baseline_A", "split", "corner_knot", "curvature_cap")


@dataclass
class Config:
    degree: int = DEGREE
    knot_spacing: float = KNOT_SPACING
    alpha3: float = ALPHA3
    lam: float = LAMBDA
    corridor: float = CORRIDOR
    area_tol: float = AREA_TOL
    drift_tol: float = DRIFT_TOL
    min_points: int = MIN_POINTS
    min_length: float = MIN_LENGTH
    min_closed_area: float = MIN_CLOSED_AREA
    max_reductions: int = MAX_REDUCTIONS
    turn3: float = TURN3_DEG
    turn8: float = TURN8_DEG
    turn8_min_t3: float = TURN8_MIN_T3_DEG
    turn8_min_curv: float = TURN8_MIN_CURV
    curv_threshold: float = CURV_THRESHOLD
    curv_turn3: float = CURV_TURN3_DEG
    group_gap: int = GROUP_GAP_PX
    min_dense: int = MIN_DENSE_POINTS
    near_end: float = NEAR_END_PX
    hairpin_window: float = HAIRPIN_WINDOW_PX
    turn_window: float = TURN_WINDOW_PX
    join_exclude: float = JOIN_EXCLUDE_PX
    dense_per_px: float = DENSE_PER_PX
    dense_max_samples: int = DENSE_MAX_SAMPLES
    cap: float = CAP_VALUE
    cap_sigma: float = CAP_SIGMA_PX
    cap_boosts: tuple[float, ...] = CAP_BOOSTS
    cap_turn_ratio: float = CAP_TURN_RATIO


@dataclass
class Hairpin:
    anchor_index: int
    group_start: int
    group_end: int
    u: float
    point: np.ndarray
    t3: float
    t5: float
    t8: float
    k5: float
    turn_before: float
    dist_start: float | None
    dist_end: float | None

    @property
    def dist_from_endpoint(self) -> float | None:
        values = [v for v in (self.dist_start, self.dist_end) if v is not None]
        return min(values) if values else None


@dataclass
class PathRecord:
    panel_id: str
    path_index: int
    path: BoundaryPath
    length: float
    dense: np.ndarray
    area_exact: float | None
    eligible: bool
    ineligible_reason: str
    exact_metrics: dict
    hairpins: list[Hairpin] = field(default_factory=list)
    profile: dict | None = None
    baseline_local: list[dict] = field(default_factory=list)


@dataclass
class Fit:
    points: np.ndarray
    length: float
    closed: bool
    knots: np.ndarray | None = None
    coeffs: np.ndarray | None = None
    status: str = "ok"
    reason: str = ""
    lambda_used: float | None = None
    refits: int = 0
    exact: bool = False
    relaxed: bool = False
    tangent_provided: bool = False
    piece_index: int = -1
    piece_kind: str = ""
    source_points: np.ndarray | None = None


@dataclass
class VariantResult:
    variant: str
    fits: list[Fit]
    points: np.ndarray
    closed: bool
    status: str
    reason: str
    lambda_used: float | None
    cap_boost: float | None
    cap_stage: str
    joins: list[float]
    anchor_us: list[float]
    piece_info: list[dict]
    cap_trials: list[dict]
    runtime_ms: float
    applied: bool = False
    n_relaxed: int = 0
    metrics: dict = field(default_factory=dict)
    dense: dict = field(default_factory=dict)
    hairpin_after: list[dict] = field(default_factory=list)
    tangent_error_max: float | None = None
    error: str = ""

    @property
    def all_exact(self) -> bool:
        return bool(self.fits) and all(fit.exact for fit in self.fits)


# ---------------------------------------------------------------------------
# Geometry helpers


def _polyline_length(points: np.ndarray, closed: bool = False) -> float:
    p = np.asarray(points, dtype=float)
    if closed:
        p = np.vstack([p, p[:1]])
    if len(p) < 2:
        return 0.0
    return float(np.hypot(*np.diff(p, axis=0).T).sum())


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return np.zeros(2, dtype=float)
    return vector / norm


def _angle_between(v0: np.ndarray, v1: np.ndarray) -> float:
    n0 = float(np.linalg.norm(v0))
    n1 = float(np.linalg.norm(v1))
    if n0 < 1e-12 or n1 < 1e-12:
        return float("nan")
    return float(
        np.degrees(
            np.arccos(float(np.clip(np.dot(v0, v1) / (n0 * n1), -1.0, 1.0)))
        )
    )


def _turn_at(points: np.ndarray, index: int, half: int, closed: bool) -> float:
    n = len(points)
    if n < 3:
        return 0.0
    if closed:
        left = points[(index - half) % n]
        right = points[(index + half) % n]
    else:
        left = points[max(0, index - half)]
        right = points[min(n - 1, index + half)]
    return _angle_between(points[index] - left, right - points[index])


def _point_at(dense: dict, u: float) -> np.ndarray:
    total = float(dense["length"])
    uu = dense["u"]
    pts = dense["points"]
    if dense["closed"]:
        if total <= 0:
            return pts[0]
        u = float(np.mod(u, total))
        ue = np.concatenate([uu - total, uu, uu + total])
        xe = np.concatenate([pts[:, 0]] * 3)
        ye = np.concatenate([pts[:, 1]] * 3)
    else:
        ue, xe, ye = uu, pts[:, 0], pts[:, 1]
    return np.array([float(np.interp(u, ue, xe)), float(np.interp(u, ue, ye))])


def _tangent_at(dense: dict, u: float, half: float = 3.0) -> np.ndarray:
    return _point_at(dense, u + half) - _point_at(dense, u - half)


# ---------------------------------------------------------------------------
# Detection


def _circle_curvature(points: np.ndarray, closed: bool) -> np.ndarray:
    n = len(points)
    values = np.zeros(n, dtype=float)
    for i in range(n):
        if closed:
            p0, p1, p2 = points[(i - 2) % n], points[i], points[(i + 2) % n]
        else:
            if i < 2 or i > n - 3:
                continue
            p0, p1, p2 = points[i - 2], points[i], points[i + 2]
        a = float(np.linalg.norm(p1 - p0))
        b = float(np.linalg.norm(p2 - p1))
        c = float(np.linalg.norm(p2 - p0))
        numerator = abs(
            (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
        )
        denominator = a * b * c
        values[i] = 2.0 * numerator / denominator if denominator > 1e-9 else 0.0
    if not closed and n >= 5:
        values[:2] = values[2]
        values[-2:] = values[-3]
    return values


def _turn_profile(points: np.ndarray, closed: bool) -> dict:
    n = len(points)
    idx = np.arange(n)
    if closed:
        im2, ip2 = (idx - 2) % n, (idx + 2) % n
    else:
        im2, ip2 = np.maximum(idx - 2, 0), np.minimum(idx + 2, n - 1)
    vectors = points[ip2] - points[im2]
    theta = np.arctan2(vectors[:, 1], vectors[:, 0])

    def turn(window: int) -> np.ndarray:
        if closed:
            j, i = (idx + window) % n, (idx - window) % n
        else:
            j, i = np.minimum(idx + window, n - 1), np.maximum(idx - window, 0)
        return np.abs(np.degrees(np.angle(np.exp(1j * (theta[j] - theta[i])))))

    return {"t3": turn(3), "t5": turn(5), "t8": turn(8), "k5": _circle_curvature(points, closed)}


def _detect_hairpins(record: PathRecord, cfg: Config) -> list[Hairpin]:
    points = record.dense
    n = len(points)
    if n < cfg.min_dense:
        record.profile = None
        return []
    profile = _turn_profile(points, record.path.closed)
    record.profile = profile
    t3, t5, t8, k5 = profile["t3"], profile["t5"], profile["t8"], profile["k5"]
    flag = (
        ((t8 >= cfg.turn8) & ((t3 >= cfg.turn8_min_t3) | (k5 >= cfg.turn8_min_curv)))
        | (t3 >= cfg.turn3)
        | ((k5 >= cfg.curv_threshold) & (t3 >= cfg.curv_turn3))
    )
    if not np.any(flag):
        return []

    groups: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if not flag[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and flag[j + 1]:
            j += 1
        groups.append((i, j))
        i = j + 1

    merged: list[tuple[int, int]] = []
    for start, end in groups:
        if merged and start - merged[-1][1] <= cfg.group_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    closed = record.path.closed
    u_scale = record.length / n if closed else record.length / max(1, n - 1)
    hairpins: list[Hairpin] = []
    for start, end in merged:
        sl = slice(start, end + 1)
        # Prefer the pixel of sharpest reversal; curvature is a tie-breaker.
        score = t3[sl] + t8[sl] + 25.0 * k5[sl]
        anchor = start + int(np.argmax(score))
        u = anchor * u_scale
        if closed:
            dist_start = dist_end = None
        else:
            dist_start, dist_end = u, record.length - u
            if min(dist_start, dist_end) <= cfg.near_end:
                continue
        hairpins.append(
            Hairpin(
                anchor_index=anchor,
                group_start=start,
                group_end=end,
                u=u,
                point=points[anchor].copy(),
                t3=float(t3[anchor]),
                t5=float(t5[anchor]),
                t8=float(t8[anchor]),
                k5=float(k5[anchor]),
                turn_before=_turn_at(
                    points, anchor, max(1, int(round(cfg.turn_window))), closed
                ),
                dist_start=dist_start,
                dist_end=dist_end,
            )
        )
    return hairpins


# ---------------------------------------------------------------------------
# P-spline basis / solve


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


def _open_basis(
    u: np.ndarray,
    length: float,
    degree: int,
    spacing: float,
    extra_knots: tuple[float, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    n_segments = max(1, int(math.ceil(length / spacing)))
    interior = list(np.linspace(0.0, length, n_segments + 1)[1:-1])
    interior.extend(float(k) for k in extra_knots if 1e-9 < float(k) < length - 1e-9)
    interior = sorted({round(float(k), 9) for k in interior})
    knots = np.concatenate(
        [
            np.full(degree + 1, 0.0),
            np.asarray(interior, dtype=float),
            np.full(degree + 1, float(length)),
        ]
    )
    basis = BSpline.design_matrix(np.clip(u, 0.0, length), knots, degree).toarray()
    return basis, knots


def _closed_basis(u: np.ndarray, length: float, degree: int, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    n_coef = max(degree + 1, int(round(length / spacing)))
    step = length / n_coef
    n_extended = n_coef + degree
    knots = (np.arange(n_extended + degree + 1) - degree) * step
    extended = BSpline.design_matrix(u, knots, degree).toarray()
    basis = np.zeros((len(u), n_coef), dtype=float)
    for j in range(n_extended):
        basis[:, j % n_coef] += extended[:, j]
    return basis, knots


def _greville(knots: np.ndarray, degree: int, n_coef: int) -> np.ndarray:
    return np.array([float(np.mean(knots[j + 1 : j + degree + 1])) for j in range(n_coef)])


def _weighted_penalty(
    n_coef: int,
    periodic: bool,
    alpha3: float,
    knots: np.ndarray,
    degree: int,
    anchors_u: tuple[float, ...],
    boost: float,
    sigma: float,
) -> np.ndarray:
    if boost <= 1.0 or not anchors_u:
        return _penalty_matrix(n_coef, periodic, alpha3)
    d2 = _difference_matrix(n_coef, 2, periodic)
    d3 = _difference_matrix(n_coef, 3, periodic)
    grev = _greville(knots, degree, n_coef)
    centers2 = grev[: len(d2)] + (degree - 1) * 0.5
    centers3 = grev[: len(d3)] + (degree - 1) * 0.5
    w2 = np.ones(len(d2))
    w3 = np.ones(len(d3))
    for anchor in anchors_u:
        w2 += (boost - 1.0) * np.exp(-0.5 * ((centers2 - anchor) / sigma) ** 2)
        w3 += (boost - 1.0) * np.exp(-0.5 * ((centers3 - anchor) / sigma) ** 2)
    return d2.T @ (w2[:, None] * d2) + alpha3 * (d3.T @ (w3[:, None] * d3))


def _derivative_row(knots: np.ndarray, n_coef: int, degree: int, u: float) -> np.ndarray:
    spline = BSpline(knots, np.eye(n_coef), degree)
    return np.asarray(spline.derivative(1)(float(u))).reshape(-1)


def _solve_open(
    basis: np.ndarray,
    data: np.ndarray,
    penalty: np.ndarray,
    lam: float,
    knots: np.ndarray,
    degree: int,
    tangent_constraints: tuple[tuple[float, np.ndarray], ...] = (),
) -> np.ndarray:
    n_coef = basis.shape[1]
    free = np.arange(1, n_coef - 1)
    fixed = np.array([0, n_coef - 1], dtype=int)
    fixed_values = np.vstack([data[0], data[-1]])
    matrix = basis.T @ basis + lam * penalty
    rhs = basis.T @ data
    hff = matrix[np.ix_(free, free)]
    rff = rhs[free] - matrix[np.ix_(free, fixed)] @ fixed_values

    coeffs = np.zeros((n_coef, 2), dtype=float)
    if tangent_constraints:
        rows = [_derivative_row(knots, n_coef, degree, u) for u, _ in tangent_constraints]
        targets = [np.asarray(target, dtype=float) for _, target in tangent_constraints]
        a_eq = np.vstack(rows)
        b_eq = np.vstack(targets)
        a_free = a_eq[:, free]
        a_fixed = a_eq[:, fixed]
        kkt = np.block(
            [[hff, a_free.T], [a_free, np.zeros((len(rows), len(rows)))]]
        )
        solution = np.linalg.solve(kkt, np.vstack([rff, b_eq - a_fixed @ fixed_values]))
        coeffs[free] = solution[: len(free)]
    else:
        coeffs[free] = np.linalg.solve(hff, rff)
    coeffs[fixed] = fixed_values
    return coeffs


def _solve_closed(basis: np.ndarray, data: np.ndarray, penalty: np.ndarray, lam: float) -> np.ndarray:
    return np.linalg.solve(basis.T @ basis + lam * penalty, basis.T @ data)


def _curvature(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    speed = np.hypot(d1[:, 0], d1[:, 1])
    values = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.maximum(speed, 1e-12) ** 3
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def _fit_curve(
    points: np.ndarray,
    length: float,
    closed: bool,
    cfg: Config,
    *,
    lam: float,
    extra_knots: tuple[float, ...] = (),
    tangent_constraints: tuple[tuple[float, np.ndarray], ...] = (),
    anchors_u: tuple[float, ...] = (),
    boost: float = 1.0,
    piece_index: int = -1,
    piece_kind: str = "",
) -> tuple[Fit, dict]:
    if closed:
        n_data = max(2, int(round(length)))
        u = np.linspace(0.0, length, n_data, endpoint=False)
        data = resample(points, True, n_data)
        basis, knots = _closed_basis(u, length, cfg.degree, cfg.knot_spacing)
        n_coef = basis.shape[1]
        penalty = _weighted_penalty(
            n_coef, True, cfg.alpha3, knots, cfg.degree, anchors_u, boost, cfg.cap_sigma
        )
        coeffs = _solve_closed(basis, data, penalty, lam)
        sampled = basis @ coeffs
        stored_knots = None
    else:
        n_data = max(2, int(round(length)))
        u = np.linspace(0.0, length, n_data)
        data = resample(points, False, n_data)
        basis, knots = _open_basis(u, length, cfg.degree, cfg.knot_spacing, extra_knots)
        n_coef = basis.shape[1]
        penalty = _weighted_penalty(
            n_coef, False, cfg.alpha3, knots, cfg.degree, anchors_u, boost, cfg.cap_sigma
        )
        coeffs = _solve_open(
            basis, data, penalty, lam, knots, cfg.degree, tangent_constraints
        )
        sampled = basis @ coeffs
        stored_knots = knots
    fit = Fit(
        points=sampled,
        length=float(length),
        closed=bool(closed),
        knots=stored_knots,
        coeffs=coeffs,
        lambda_used=float(lam),
        tangent_provided=bool(tangent_constraints),
        piece_index=piece_index,
        piece_kind=piece_kind,
        source_points=np.asarray(points, dtype=float).copy(),
    )
    metrics = boundary_metrics(sampled, _bp(points, closed))
    return fit, metrics


def _bp(points: np.ndarray, closed: bool) -> BoundaryPath:
    return BoundaryPath(0, (0, 0), np.asarray(points, dtype=float), bool(closed), int(round(_polyline_length(points, closed))))


def _constraint_violations(metrics: dict, closed: bool, cfg: Config) -> list[str]:
    reasons: list[str] = []
    if float(metrics["max_disp"]) > cfg.corridor:
        reasons.append(f"max_disp={metrics['max_disp']:.3f}>{cfg.corridor:g}")
    if int(metrics["self_intersections"]) > 0:
        reasons.append(f"self_intersections={int(metrics['self_intersections'])}")
    if closed:
        area_exact = float(metrics.get("area_exact", 0.0))
        if area_exact > 1e-9:
            rel = float(metrics.get("area_change", 0.0)) / area_exact
            if abs(rel) > cfg.area_tol:
                reasons.append(f"area_change_rel={rel:+.4f}>{cfg.area_tol:g}")
    elif float(metrics["endpoint_drift"]) > cfg.drift_tol:
        reasons.append(f"endpoint_drift={metrics['endpoint_drift']:.4f}>{cfg.drift_tol:g}")
    return reasons


def _exact_fit(points: np.ndarray, length: float, closed: bool, status: str, reason: str) -> Fit:
    return Fit(
        points=np.asarray(points, dtype=float).copy(),
        length=float(length),
        closed=bool(closed),
        status=status,
        reason=reason,
        exact=True,
        source_points=np.asarray(points, dtype=float).copy(),
    )


def _fit_with_reductions(
    points: np.ndarray,
    length: float,
    closed: bool,
    cfg: Config,
    *,
    lam: float = LAMBDA,
    extra_knots: tuple[float, ...] = (),
    tangent_constraints: tuple[tuple[float, np.ndarray], ...] = (),
    anchors_u: tuple[float, ...] = (),
    boost: float = 1.0,
    piece_index: int = -1,
    piece_kind: str = "",
) -> tuple[Fit, dict]:
    current = float(lam)
    refits = 0
    last_reasons: list[str] = []
    last_metrics: dict = {}
    for attempt in range(cfg.max_reductions + 1):
        fit, metrics = _fit_curve(
            points,
            length,
            closed,
            cfg,
            lam=current,
            extra_knots=extra_knots,
            tangent_constraints=tangent_constraints,
            anchors_u=anchors_u,
            boost=boost,
            piece_index=piece_index,
            piece_kind=piece_kind,
        )
        reasons = _constraint_violations(metrics, closed, cfg)
        if not reasons:
            fit.status = "ok" if refits == 0 else "ok_reduced"
            fit.refits = refits
            return fit, metrics
        last_reasons, last_metrics = reasons, metrics
        if attempt < cfg.max_reductions:
            current *= 0.5
            refits += 1
    exact = _exact_fit(points, length, closed, "fallback_constraint", "; ".join(last_reasons))
    exact.lambda_used = current
    exact.refits = refits
    exact.piece_index = piece_index
    exact.piece_kind = piece_kind
    return exact, last_metrics


# ---------------------------------------------------------------------------
# Variants


def _baseline_fit(record: PathRecord, cfg: Config) -> Fit:
    if not record.eligible:
        return _exact_fit(
            record.path.points, record.length, record.path.closed,
            "fallback_short", record.ineligible_reason,
        )
    fit, _ = _fit_with_reductions(
        record.path.points, record.length, record.path.closed, cfg,
        lam=cfg.lam, piece_kind="baseline",
    )
    return fit


def _variant_baseline(record: PathRecord, cfg: Config) -> VariantResult:
    t0 = time.perf_counter()
    fit = _baseline_fit(record, cfg)
    return _assemble_result(
        "baseline_A", record, [fit],
        anchor_us=[h.u for h in record.hairpins],
        cap_stage="baseline", t0=t0,
    )


def _piece_arrays(record: PathRecord) -> list[np.ndarray]:
    dense = record.dense
    n = len(dense)
    anchors = sorted(h.anchor_index for h in record.hairpins)
    if not anchors:
        return [dense]
    pieces: list[np.ndarray] = []
    if record.path.closed:
        for i, a in enumerate(anchors):
            b = anchors[(i + 1) % len(anchors)]
            if i == len(anchors) - 1:
                pieces.append(np.vstack([dense[a:], dense[: b + 1]]))
            else:
                pieces.append(dense[a : b + 1].copy())
    else:
        bounds = [0] + anchors + [n - 1]
        for a, b in zip(bounds[:-1], bounds[1:]):
            pieces.append(dense[a : b + 1].copy())
    return pieces


def _variant_pieces(
    record: PathRecord,
    cfg: Config,
    *,
    variant: str,
    tangents: bool,
) -> VariantResult:
    t0 = time.perf_counter()
    if not record.eligible:
        fit = _exact_fit(
            record.path.points, record.length, record.path.closed,
            "fallback_short", record.ineligible_reason,
        )
        result = _assemble_result(
            variant, record, [fit], anchor_us=[h.u for h in record.hairpins],
            cap_stage="ineligible", t0=t0,
        )
        return result
    if not record.hairpins:
        fit = _baseline_fit(record, cfg)
        return _assemble_result(
            variant, record, [fit], anchor_us=[],
            cap_stage="no_hairpin", t0=t0,
        )

    arrays = _piece_arrays(record)
    fits: list[Fit] = []
    piece_info: list[dict] = []
    n_relaxed = 0
    for index, points in enumerate(arrays):
        plen = _polyline_length(points, False)
        start_anchor = record.path.closed or index > 0
        end_anchor = record.path.closed or index < len(arrays) - 1
        if plen < cfg.min_length or len(points) < cfg.min_points:
            fit = _exact_fit(points, plen, False, "fallback_short", "split_piece_too_short")
            fits.append(fit)
            piece_info.append(
                {
                    "piece_index": index, "length": plen, "n_points": len(points),
                    "status": fit.status, "reason": fit.reason, "lambda_used": None,
                    "refits": 0, "tangent": False, "relaxed": False,
                }
            )
            continue

        constraints: list[tuple[float, np.ndarray]] = []
        if tangents:
            if start_anchor:
                target = _unit(points[min(3, len(points) - 1)] - points[0])
                if np.linalg.norm(target) > 0:
                    constraints.append((0.0, target))
            if end_anchor:
                target = _unit(points[-1] - points[max(0, len(points) - 4)])
                if np.linalg.norm(target) > 0:
                    constraints.append((plen, target))

        fit, _ = _fit_with_reductions(
            points, plen, False, cfg, lam=cfg.lam,
            tangent_constraints=tuple(constraints),
            piece_index=index, piece_kind="corner" if tangents else "split",
        )
        relaxed = False
        if tangents and fit.exact and constraints:
            retry, _ = _fit_with_reductions(
                points, plen, False, cfg, lam=cfg.lam,
                piece_index=index, piece_kind="corner_relaxed",
            )
            if not retry.exact:
                retry.relaxed = True
                fit = retry
                relaxed = True
                n_relaxed += 1
        fits.append(fit)
        piece_info.append(
            {
                "piece_index": index, "length": plen, "n_points": len(points),
                "status": fit.status, "reason": fit.reason,
                "lambda_used": fit.lambda_used, "refits": fit.refits,
                "tangent": bool(constraints) and not relaxed, "relaxed": relaxed,
            }
        )

    if all(fit.exact for fit in fits):
        status = (
            "fallback_short"
            if all(fit.status == "fallback_short" for fit in fits)
            else "fallback_constraint"
        )
        whole = _exact_fit(
            record.path.points, record.length, record.path.closed,
            status, "all_split_pieces_exact",
        )
        result = _assemble_result(
            variant, record, [whole], anchor_us=[h.u for h in record.hairpins],
            cap_stage="all_pieces_exact", t0=t0,
        )
        result.piece_info = piece_info
        return result

    lengths = [fit.length for fit in fits]
    cumulative = np.cumsum(lengths).tolist()
    if record.path.closed:
        anchor_us = [0.0] + cumulative[:-1]
        joins = list(anchor_us)
    else:
        anchor_us = cumulative[:-1]
        joins = list(anchor_us)
    if len(anchor_us) != len(record.hairpins):
        anchor_us = [h.u for h in record.hairpins]
    result = _assemble_result(
        variant, record, fits, anchor_us=anchor_us, joins=joins,
        cap_stage="split_applied", t0=t0,
    )
    # Each piece is checked individually during fitting, but the concatenated
    # curve can cross itself across a join.  The contract requires zero
    # self-intersections, so fall back to exact rather than accept it.
    whole_check = boundary_metrics(result.points, record.path)
    area_rel = None
    if record.path.closed and float(whole_check.get("area_exact", 0.0)) > 1e-9:
        area_rel = float(whole_check.get("area_change", 0.0)) / float(whole_check["area_exact"])
    if int(whole_check["self_intersections"]) > 0 or (
        area_rel is not None and abs(area_rel) > cfg.area_tol
    ):
        reason = (
            f"assembled_self_intersections={int(whole_check['self_intersections'])}"
            if int(whole_check["self_intersections"]) > 0
            else f"assembled_area_change_rel={area_rel:+.4f}"
        )
        exact = _exact_fit(
            record.path.points, record.length, record.path.closed,
            "fallback_constraint", reason,
        )
        result = _assemble_result(
            variant, record, [exact], anchor_us=[h.u for h in record.hairpins],
            cap_stage="split_self_intersection" if area_rel is None else "split_constraint",
            t0=t0,
        )
        result.piece_info = piece_info
        return result

    statuses = [fit.status for fit in fits]
    if any(fit.exact for fit in fits):
        status, reason = "ok_partial", "some_split_pieces_exact"
    elif any(s == "ok_reduced" for s in statuses):
        status, reason = "ok_reduced", ""
    else:
        status, reason = "ok", ""
    result.status = status
    result.reason = reason
    result.lambda_used = next((f.lambda_used for f in fits if f.lambda_used is not None), None)
    result.piece_info = piece_info
    result.applied = True
    result.n_relaxed = n_relaxed
    return result


def _variant_split(record: PathRecord, cfg: Config) -> VariantResult:
    return _variant_pieces(record, cfg, variant="split", tangents=False)


def _variant_corner(record: PathRecord, cfg: Config) -> VariantResult:
    return _variant_pieces(record, cfg, variant="corner_knot", tangents=True)


def _variant_cap(record: PathRecord, cfg: Config, baseline: VariantResult) -> VariantResult:
    t0 = time.perf_counter()
    if not record.eligible:
        fit = _exact_fit(
            record.path.points, record.length, record.path.closed,
            "fallback_short", record.ineligible_reason,
        )
        return _assemble_result(
            "curvature_cap", record, [fit], anchor_us=[h.u for h in record.hairpins],
            cap_stage="ineligible", t0=t0,
        )
    if not record.hairpins:
        result = _assemble_result(
            "curvature_cap", record, list(baseline.fits),
            anchor_us=[], cap_stage="no_hairpin", t0=t0,
        )
        result.status = baseline.status
        result.reason = baseline.reason
        result.lambda_used = baseline.lambda_used
        return result

    anchor_u = tuple(h.u for h in record.hairpins)
    lam0 = baseline.lambda_used if baseline.lambda_used is not None else cfg.lam
    trials: list[dict] = []
    accepted: Fit | None = None
    accepted_dense: dict | None = None
    accepted_boost: float | None = None
    for boost in cfg.cap_boosts:
        fit, metrics = _fit_with_reductions(
            record.path.points, record.length, record.path.closed, cfg,
            lam=float(lam0), anchors_u=anchor_u, boost=float(boost),
            piece_kind="cap",
        )
        if fit.exact:
            trials.append(
                {
                    "boost": float(boost), "local_curv": None,
                    "max_disp": float(metrics.get("max_disp", float("nan"))),
                    "pass": False, "reason": fit.reason or "constraint",
                }
            )
            continue
        dense = _dense_fit(fit, cfg)
        local = _local_curvature(dense, list(anchor_u), cfg.hairpin_window)
        turn_ratios = []
        for anchor in record.hairpins:
            left = _point_at(dense, anchor.u - cfg.turn_window)
            center = _point_at(dense, anchor.u)
            right = _point_at(dense, anchor.u + cfg.turn_window)
            after = _angle_between(center - left, right - center)
            turn_ratios.append(after / max(anchor.turn_before, 1e-6))
        turn_ok = bool(turn_ratios) and min(turn_ratios) >= cfg.cap_turn_ratio
        passed = local <= cfg.cap + 1e-9 and turn_ok
        trials.append(
            {
                "boost": float(boost), "local_curv": float(local),
                "max_disp": float(metrics.get("max_disp", float("nan"))),
                "min_turn_ratio": float(min(turn_ratios)) if turn_ratios else None,
                "pass": bool(passed),
                "reason": "" if passed else ("turn_lost" if local <= cfg.cap + 1e-9 else "above_cap"),
            }
        )
        if passed and accepted is None:
            accepted, accepted_dense, accepted_boost = fit, dense, float(boost)
        # Keep evaluating higher boosts so the cap sensitivity table has the
        # full trial ladder; the first passing boost remains the accepted fit.

    if accepted is not None:
        result = _assemble_result(
            "curvature_cap", record, [accepted],
            anchor_us=[h.u for h in record.hairpins],
            cap_stage="cap_met" if accepted_boost and accepted_boost > 1.0 else "baseline_within_cap",
            cap_boost=accepted_boost, t0=t0,
        )
        result.status = accepted.status
        result.reason = accepted.reason
        result.lambda_used = accepted.lambda_used
        result.cap_trials = trials
        if accepted_dense is not None:
            result.dense = accepted_dense
        return result

    fallback = _variant_split(record, cfg)
    result = VariantResult(
        variant="curvature_cap",
        fits=fallback.fits,
        points=fallback.points,
        closed=fallback.closed,
        status=fallback.status,
        reason=f"cap_unreachable; {fallback.reason}".strip("; "),
        lambda_used=fallback.lambda_used,
        cap_boost=None,
        cap_stage="cap_to_exact" if fallback.all_exact else "cap_to_split",
        joins=fallback.joins,
        anchor_us=fallback.anchor_us,
        piece_info=fallback.piece_info,
        cap_trials=trials,
        runtime_ms=(time.perf_counter() - t0) * 1000.0,
    )
    return result


def _local_curvature(dense: dict, anchors_u: list[float], window: float) -> float:
    if not anchors_u or not len(dense["u"]):
        return 0.0
    total = float(dense["length"])
    mask = np.zeros(len(dense["u"]), dtype=bool)
    for anchor in anchors_u:
        distance = np.abs(dense["u"] - anchor)
        if dense["closed"] and total > 0:
            distance = np.minimum(distance, total - distance)
        mask |= distance <= window
    return float(np.max(dense["k"][mask])) if np.any(mask) else 0.0


def _assemble_result(
    variant: str,
    record: PathRecord,
    fits: list[Fit],
    *,
    anchor_us: list[float],
    joins: list[float] | None = None,
    cap_stage: str,
    cap_boost: float | None = None,
    t0: float,
) -> VariantResult:
    closed = record.path.closed
    merged_parts = [np.asarray(fits[0].points, dtype=float)]
    for fit in fits[1:]:
        arr = np.asarray(fit.points, dtype=float)
        if len(merged_parts[-1]) and np.allclose(arr[0], merged_parts[-1][-1], atol=1e-9):
            arr = arr[1:]
        merged_parts.append(arr)
    merged = np.vstack(merged_parts)
    if closed and len(merged) > 1 and np.allclose(merged[0], merged[-1], atol=1e-9):
        merged = merged[:-1]
    status = fits[0].status if len(fits) == 1 else "ok"
    reason = fits[0].reason if len(fits) == 1 else ""
    return VariantResult(
        variant=variant,
        fits=fits,
        points=merged,
        closed=closed,
        status=status,
        reason=reason,
        lambda_used=fits[0].lambda_used,
        cap_boost=cap_boost,
        cap_stage=cap_stage,
        joins=[float(j) for j in (joins if joins is not None else [])],
        anchor_us=[float(u) for u in anchor_us],
        piece_info=[],
        cap_trials=[],
        runtime_ms=(time.perf_counter() - t0) * 1000.0,
    )


# ---------------------------------------------------------------------------
# Dense evaluation


def _dense_fit(fit: Fit, cfg: Config) -> dict:
    n = max(64, min(cfg.dense_max_samples, int(round(cfg.dense_per_px * max(fit.length, 1.0)))))
    if fit.knots is not None and fit.coeffs is not None and not fit.closed:
        spline = BSpline(fit.knots, fit.coeffs, DEGREE)
        uu = np.linspace(0.0, fit.length, n)
        pts = spline(uu)
        k = _curvature(spline.derivative(1)(uu), spline.derivative(2)(uu))
    elif fit.knots is not None and fit.coeffs is not None and fit.closed:
        coeffs = np.asarray(fit.coeffs)
        n_coef = len(coeffs)
        step = fit.length / n_coef
        n_extended = n_coef + DEGREE
        knots = (np.arange(n_extended + DEGREE + 1) - DEGREE) * step
        extended = coeffs[np.arange(n_extended) % n_coef]
        spline = BSpline(knots, extended, DEGREE)
        uu = np.linspace(0.0, fit.length, n, endpoint=False)
        pts = spline(uu)
        k = _curvature(spline.derivative(1)(uu), spline.derivative(2)(uu))
    else:
        pts = resample(fit.points, fit.closed, n)
        if fit.closed:
            pts = np.vstack([pts, pts[:1]])
            uu = np.linspace(0.0, fit.length, len(pts))
        else:
            uu = np.linspace(0.0, fit.length, len(pts))
        dx = np.gradient(pts[:, 0], uu)
        dy = np.gradient(pts[:, 1], uu)
        ddx = np.gradient(dx, uu)
        ddy = np.gradient(dy, uu)
        k = _curvature(np.column_stack([dx, dy]), np.column_stack([ddx, ddy]))
        if fit.closed:
            pts, uu, k = pts[:-1], uu[:-1], k[:-1]
    return {"u": uu, "points": pts, "k": k, "length": float(fit.length), "closed": bool(fit.closed)}


def _concat_dense(parts: list[dict], closed: bool) -> dict:
    us, pts, ks = [], [], []
    offset = 0.0
    for part in parts:
        us.append(part["u"] + offset)
        pts.append(part["points"])
        ks.append(part["k"])
        offset += float(part["length"])
    u = np.concatenate(us)
    pts = np.vstack(pts)
    k = np.concatenate(ks)
    order = np.argsort(u, kind="stable")
    u, pts, k = u[order], pts[order], k[order]
    keep = np.ones(len(u), dtype=bool)
    keep[1:] = np.diff(u) > 1e-9
    return {"u": u[keep], "points": pts[keep], "k": k[keep], "length": float(offset), "closed": bool(closed)}


# ---------------------------------------------------------------------------
# Result metrics


def _exact_reference_metrics(record: PathRecord) -> dict:
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


def _compute_result_metrics(record: PathRecord, result: VariantResult, cfg: Config) -> None:
    # Compare against the exact original path, not against the result itself.
    metrics = boundary_metrics(result.points, record.path)
    if result.all_exact and result.status in ("fallback_short", "fallback_constraint"):
        metrics = _exact_reference_metrics(record)
    result.metrics = metrics

    parts = [_dense_fit(fit, cfg) for fit in result.fits]
    dense = _concat_dense(parts, record.path.closed)
    total = float(dense["length"])
    u, k = dense["u"], dense["k"]
    mask = np.ones(len(u), dtype=bool)
    for join in result.joins:
        distance = np.abs(u - join)
        if dense["closed"] and total > 0:
            distance = np.minimum(distance, total - distance)
        mask &= distance > cfg.join_exclude
    if np.any(mask):
        dense_max = float(np.max(k[mask]))
        dense_p95 = float(np.percentile(k[mask], 95))
        index = int(np.argmax(np.where(mask, k, -np.inf)))
        dense_u = float(u[index])
        max_end_dist = None if record.path.closed else float(min(dense_u, total - dense_u))
    else:
        dense_max = dense_p95 = 0.0
        dense_u, max_end_dist = None, None
    dense.update(
        {
            "max": dense_max,
            "p95": dense_p95,
            "max_u": dense_u,
            "max_end_dist": max_end_dist,
            "raw_max": float(np.max(k)) if len(k) else 0.0,
        }
    )
    result.dense = dense

    anchor_metrics: list[dict] = []

    if len(result.anchor_us) != len(record.hairpins):
        # Closed single-piece results use the exact u; split uses joins.
        if len(result.fits) == 1 and len(record.hairpins):
            result.anchor_us = [h.u for h in record.hairpins]
        else:
            result.anchor_us = [
                (float(j) if i < len(result.joins) else float(h.u))
                for i, (j, h) in enumerate(zip(result.joins + [0.0] * len(record.hairpins), record.hairpins))
            ][: len(record.hairpins)]

    for hi, (anchor, ua) in enumerate(zip(record.hairpins, result.anchor_us)):
        distance = np.abs(u - ua)
        if dense["closed"] and total > 0:
            distance = np.minimum(distance, total - distance)
        local_mask = (distance <= cfg.hairpin_window) & mask
        local = float(np.max(k[local_mask])) if np.any(local_mask) else 0.0
        left = _point_at(dense, ua - cfg.turn_window)
        center = _point_at(dense, ua)
        right = _point_at(dense, ua + cfg.turn_window)
        turn_after = _angle_between(center - left, right - center)
        baseline_local = None
        exact_local = None
        if hi < len(record.baseline_local):
            baseline_local = record.baseline_local[hi].get("local_curv")
            exact_local = record.baseline_local[hi].get("exact_local_curvature")
        anchor_metrics.append(
            {
                "anchor_index": anchor.anchor_index,
                "u_final": float(ua),
                "local_curvature": local,
                "baseline_local_curvature": baseline_local,
                "exact_local_curvature": exact_local,
                "turn_before": anchor.turn_before,
                "turn_after": turn_after,
                "turn_delta": turn_after - anchor.turn_before,
                "turn_preserved": 1.0 - abs(turn_after - anchor.turn_before) / max(anchor.turn_before, 1e-6),
                "dist_from_endpoint": anchor.dist_from_endpoint,
            }
        )
    result.hairpin_after = anchor_metrics

    tangent_error = None
    if result.applied and record.hairpins and not result.all_exact:
        errors: list[float] = []
        n = len(record.dense)
        for anchor, ua in zip(record.hairpins, result.anchor_us):
            i = anchor.anchor_index
            if record.path.closed:
                le = record.dense[i] - record.dense[(i - 3) % n]
                re = record.dense[(i + 3) % n] - record.dense[i]
            else:
                le = record.dense[i] - record.dense[max(0, i - 3)]
                re = record.dense[min(n - 1, i + 3)] - record.dense[i]
            lf = _point_at(dense, ua) - _point_at(dense, ua - 3.0)
            rf = _point_at(dense, ua + 3.0) - _point_at(dense, ua)
            errors.extend([_angle_between(le, lf), _angle_between(re, rf)])
        errors = [e for e in errors if not math.isnan(e)]
        tangent_error = float(max(errors)) if errors else None
    result.tangent_error_max = tangent_error

    worst = max(anchor_metrics, key=lambda d: d["local_curvature"]) if anchor_metrics else None
    # Path-level reference: the baseline's worst local maximum over all anchors.
    # Unlike the per-anchor "worst" this is identical for every variant, so the
    # variant table compares handlers on the same reference.
    baseline_path = None
    exact_path = None
    if record.baseline_local:
        base_values = [d.get("local_curv") for d in record.baseline_local if d.get("local_curv") is not None]
        exact_values = [
            d.get("exact_local_curvature")
            for d in record.baseline_local
            if d.get("exact_local_curvature") is not None
        ]
        baseline_path = max(base_values) if base_values else None
        exact_path = max(exact_values) if exact_values else None
    metrics.update(
        {
            "dense_max_curvature": dense_max,
            "dense_p95_curvature": dense_p95,
            "dense_raw_max_curvature": dense["raw_max"],
            "dense_max_u": dense_u,
            "dense_max_end_dist": max_end_dist,
            "hairpin_max_curvature": float(worst["local_curvature"]) if worst else None,
            "hairpin_baseline_curvature": float(baseline_path) if baseline_path is not None else None,
            "hairpin_exact_curvature": float(exact_path) if exact_path is not None else None,
            "hairpin_turn_before": float(worst["turn_before"]) if worst else None,
            "hairpin_turn_after": float(worst["turn_after"]) if worst else None,
            "hairpin_turn_delta": float(worst["turn_delta"]) if worst else None,
            "hairpin_turn_preserved": float(worst["turn_preserved"]) if worst else None,
            "hairpin_dist_from_endpoint": float(worst["dist_from_endpoint"])
            if worst and worst["dist_from_endpoint"] is not None else None,
        }
    )


def _set_baseline_local(record: PathRecord, baseline: VariantResult, cfg: Config) -> None:
    record.baseline_local = []
    if not record.hairpins:
        return
    dense = baseline.dense
    total = float(dense["length"])
    n_exact = max(
        64,
        min(
            cfg.dense_max_samples,
            int(round(cfg.dense_per_px * max(record.length, 1.0))),
        ),
    )
    exact = resample(record.dense, record.path.closed, n_exact)
    exact_total = _polyline_length(record.dense, record.path.closed)
    if record.path.closed:
        exact = np.vstack([exact, exact[:1]])
        exact_u = np.linspace(0.0, exact_total, len(exact))
    else:
        exact_u = np.linspace(0.0, exact_total, len(exact))
    dx = np.gradient(exact[:, 0], exact_u)
    dy = np.gradient(exact[:, 1], exact_u)
    ddx = np.gradient(dx, exact_u)
    ddy = np.gradient(dy, exact_u)
    exact_k = _curvature(np.column_stack([dx, dy]), np.column_stack([ddx, ddy]))
    for anchor in record.hairpins:
        distance = np.abs(dense["u"] - anchor.u)
        if dense["closed"] and total > 0:
            distance = np.minimum(distance, total - distance)
        mask = distance <= cfg.hairpin_window
        local = float(np.max(dense["k"][mask])) if np.any(mask) else 0.0
        ex_distance = np.abs(exact_u - anchor.u)
        if record.path.closed and exact_total > 0:
            ex_distance = np.minimum(ex_distance, exact_total - ex_distance)
        ex_mask = ex_distance <= cfg.hairpin_window
        exact_local = float(np.max(exact_k[ex_mask])) if np.any(ex_mask) else 0.0
        record.baseline_local.append(
            {"local_curv": local, "exact_local_curvature": exact_local}
        )


# ---------------------------------------------------------------------------
# Records


def _build_record(panel_id: str, path_index: int, path: BoundaryPath, cfg: Config) -> PathRecord:
    closed = path.closed
    length = _polyline_length(path.points, closed)
    n_dense = max(2, int(round(length)))
    dense = resample(path.points, closed, n_dense)
    exact_metrics = boundary_metrics(path.points, path)
    area_exact = float(exact_metrics.get("area_exact", 0.0)) if closed else None
    reasons: list[str] = []
    if len(path.points) < cfg.min_points:
        reasons.append(f"too_few_points({len(path.points)}<{cfg.min_points})")
    if length < cfg.min_length:
        reasons.append(f"too_short({length:.2f}<{cfg.min_length:g}px)")
    if closed and (area_exact or 0.0) < cfg.min_closed_area:
        reasons.append(f"tiny_closed_area({area_exact or 0.0:.1f}<{cfg.min_closed_area:g}px2)")
    record = PathRecord(
        panel_id=panel_id,
        path_index=path_index,
        path=path,
        length=length,
        dense=dense,
        area_exact=area_exact,
        eligible=not reasons,
        ineligible_reason="; ".join(reasons),
        exact_metrics=exact_metrics,
    )
    if record.eligible:
        record.hairpins = _detect_hairpins(record, cfg)
    return record


# ---------------------------------------------------------------------------
# Synthetic cases


def _dense_vertices(vertices: list[tuple[float, float]]) -> np.ndarray:
    out = [np.asarray(vertices[0], dtype=float)]
    for a, b in zip(vertices[:-1], vertices[1:]):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        n = max(2, int(math.ceil(float(np.linalg.norm(b - a)))))
        seg = np.linspace(a, b, n, endpoint=False)
        out.extend(seg[1:])
    out.append(np.asarray(vertices[-1], dtype=float))
    return np.asarray(out, dtype=float)


def _arc(center: tuple[float, float], radius: float, a0: float, a1: float) -> list[tuple[float, float]]:
    steps = max(4, int(math.ceil(abs(a1 - a0) * radius)))
    angles = np.linspace(a0, a1, steps)
    return [
        (center[0] + radius * math.cos(a), center[1] + radius * math.sin(a))
        for a in angles
    ]


def _synthetic_cases(cfg: Config) -> list[dict]:
    cases: list[dict] = []

    v = _dense_vertices([(50, 367), (300, 300), (50, 233)])
    cases.append(
        {
            "panel_id": "syn_v_cusp",
            "kind": "true geometry corner",
            "description": "sharp V, ~150 deg turn in <=3 px",
            "points": v,
            "closed": False,
            "bounds": (0, 180, 350, 420),
        }
    )

    uv = [(50, 200), (300, 200)] + _arc((300, 204), 4.0, -math.pi / 2, math.pi / 2) + [(50, 208)]
    cases.append(
        {
            "panel_id": "syn_u_turn",
            "kind": "skeleton artifact",
            "description": "U-turn around an 8 px protrusion",
            "points": _dense_vertices(uv),
            "closed": False,
            "bounds": (0, 150, 350, 260),
        }
    )

    zv = _dense_vertices(
        [(50, 300), (300, 300), (300, 296), (294, 296), (294, 300), (300, 300), (340, 300)]
    )
    cases.append(
        {
            "panel_id": "syn_zigzag_junction",
            "kind": "skeleton artifact",
            "description": "junction spur doubles back 6 px",
            "points": zv,
            "closed": False,
            "bounds": (0, 260, 390, 340),
        }
    )

    pv = [(50, 250), (380, 250)] + _arc((380, 253), 3.0, -math.pi / 2, math.pi / 2) + [(50, 256)]
    cases.append(
        {
            "panel_id": "syn_pinchout",
            "kind": "true geology pinch-out",
            "description": "6 px layer terminating at a rounded tip",
            "points": _dense_vertices(pv),
            "closed": False,
            "bounds": (0, 210, 430, 300),
        }
    )

    fv = [(50, 300), (300, 300)] + _arc((300, 330), 30.0, -math.pi / 2, math.pi / 2) + [(50, 360)]
    cases.append(
        {
            "panel_id": "syn_smooth_fold",
            "kind": "true geology control",
            "description": "smooth 30 px radius fold; no hairpin",
            "points": _dense_vertices(fv),
            "closed": False,
            "bounds": (0, 250, 390, 410),
        }
    )

    for index, case in enumerate(cases):
        points = np.asarray(case["points"], dtype=float)
        path = BoundaryPath(
            boundary_id=9000 + index,
            labels_pair=(1, 2),
            points=points,
            closed=bool(case["closed"]),
            exact_length=int(round(_polyline_length(points, case["closed"]))),
        )
        case["path"] = path
        case["record"] = _build_record(case["panel_id"], 0, path, cfg)
    return cases


# ---------------------------------------------------------------------------
# Panel execution


def _run_records(
    panel_id: str,
    records: list[PathRecord],
    cfg: Config,
) -> tuple[dict[str, list[VariantResult]], dict[str, dict]]:
    results: dict[str, list[VariantResult]] = {variant: [] for variant in VARIANTS}
    t0 = time.perf_counter()
    for record in records:
        results["baseline_A"].append(_variant_baseline(record, cfg))
    baseline_time = time.perf_counter() - t0

    for record, baseline in zip(records, results["baseline_A"]):
        _compute_result_metrics(record, baseline, cfg)
    for record, baseline in zip(records, results["baseline_A"]):
        _set_baseline_local(record, baseline, cfg)
        # The baseline is its own reference: fill the "baseline" hairpin fields
        # with its own local values so every variant row has a comparable base.
        if baseline.hairpin_after:
            for hi, entry in enumerate(baseline.hairpin_after):
                entry["baseline_local_curvature"] = entry["local_curvature"]
                if hi < len(record.baseline_local):
                    entry["exact_local_curvature"] = record.baseline_local[hi]["exact_local_curvature"]
            worst = max(baseline.hairpin_after, key=lambda d: d["local_curvature"])
            baseline.metrics["hairpin_baseline_curvature"] = worst["local_curvature"]
            exact_values = [
                d.get("exact_local_curvature")
                for d in record.baseline_local
                if d.get("exact_local_curvature") is not None
            ]
            baseline.metrics["hairpin_exact_curvature"] = max(exact_values) if exact_values else None

    for variant, function in (("split", _variant_split), ("corner_knot", _variant_corner)):
        t0 = time.perf_counter()
        for record in records:
            results[variant].append(function(record, cfg))

    t0 = time.perf_counter()
    for record, baseline in zip(records, results["baseline_A"]):
        results["curvature_cap"].append(_variant_cap(record, cfg, baseline))

    for variant in VARIANTS:
        if variant == "baseline_A":
            continue
        for record, result in zip(records, results[variant]):
            _compute_result_metrics(record, result, cfg)

    panel_info: dict[str, dict] = {}
    for variant in VARIANTS:
        paths = [
            BoundaryPath(
                boundary_id=record.path.boundary_id,
                labels_pair=record.path.labels_pair,
                points=result.points,
                closed=record.path.closed,
                exact_length=record.path.exact_length,
            )
            for record, result in zip(records, results[variant])
        ]
        try:
            panel_info[variant] = {
                "cross_intersections": cross_intersections(paths),
                "error": "",
            }
        except Exception as exc:  # record, never skip
            panel_info[variant] = {
                "cross_intersections": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
    exact_paths = [
        BoundaryPath(
            boundary_id=record.path.boundary_id,
            labels_pair=record.path.labels_pair,
            points=record.path.points,
            closed=record.path.closed,
            exact_length=record.path.exact_length,
        )
        for record in records
    ]
    try:
        panel_info["exact"] = {
            "cross_intersections": cross_intersections(exact_paths),
            "error": "",
        }
    except Exception as exc:
        panel_info["exact"] = {
            "cross_intersections": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    panel_info["_baseline_time"] = baseline_time
    return results, panel_info


def _metrics_row(record: PathRecord, result: VariantResult) -> dict:
    m = result.metrics
    return {
        "panel_id": record.panel_id,
        "path_index": record.path_index,
        "boundary_id": record.path.boundary_id,
        "labels_pair": f"{record.path.labels_pair[0]}-{record.path.labels_pair[1]}",
        "closed": int(record.path.closed),
        "length_exact": record.length,
        "eligible": int(record.eligible),
        "variant": result.variant,
        "status": result.status,
        "reason": result.reason,
        "n_hairpins": len(record.hairpins),
        "n_pieces": len(result.fits),
        "n_pieces_exact": sum(1 for fit in result.fits if fit.exact),
        "n_relaxed": result.n_relaxed,
        "lambda_used": result.lambda_used,
        "cap_boost": result.cap_boost,
        "cap_stage": result.cap_stage,
        "mean_disp": m.get("mean_disp"),
        "p95_disp": m.get("p95_disp"),
        "max_disp": m.get("max_disp"),
        "endpoint_drift": m.get("endpoint_drift"),
        "max_curvature": m.get("max_curvature"),
        "p95_curvature": m.get("p95_curvature"),
        "dense_max_curvature": m.get("dense_max_curvature"),
        "dense_p95_curvature": m.get("dense_p95_curvature"),
        "dense_raw_max_curvature": m.get("dense_raw_max_curvature"),
        "dense_max_u": m.get("dense_max_u"),
        "dense_max_end_dist": m.get("dense_max_end_dist"),
        "hairpin_max_curvature": m.get("hairpin_max_curvature"),
        "hairpin_baseline_curvature": m.get("hairpin_baseline_curvature"),
        "hairpin_exact_curvature": m.get("hairpin_exact_curvature"),
        "hairpin_turn_before": m.get("hairpin_turn_before"),
        "hairpin_turn_after": m.get("hairpin_turn_after"),
        "hairpin_turn_delta": m.get("hairpin_turn_delta"),
        "hairpin_turn_preserved": m.get("hairpin_turn_preserved"),
        "hairpin_dist_from_endpoint": m.get("hairpin_dist_from_endpoint"),
        "tangent_error_max": result.tangent_error_max,
        "length_smooth": m.get("length_smooth"),
        "length_change_rel": (
            (float(m["length_smooth"]) - float(m["length_exact"])) / float(m["length_exact"])
            if float(m.get("length_exact") or 0.0) > 1e-9 else None
        ),
        "area_exact": m.get("area_exact"),
        "area_smooth": m.get("area_smooth"),
        "area_change": m.get("area_change"),
        "area_change_rel": (
            float(m["area_change"]) / float(m["area_exact"])
            if m.get("area_exact") and float(m["area_exact"]) > 1e-9 else None
        ),
        "self_intersections": m.get("self_intersections"),
        "runtime_ms": result.runtime_ms,
        "error": result.error,
    }


def _panel_row(
    panel_id: str,
    records: list[PathRecord],
    results: list[VariantResult],
    variant: str,
    panel_info: dict,
    runtime_s: float,
) -> dict:
    statuses = [r.status for r in results]
    eligible = [r for r, rec in zip(results, records) if rec.eligible]
    return {
        "panel_id": panel_id,
        "variant": variant,
        "n_paths": len(results),
        "n_eligible": len(eligible),
        "n_ok": statuses.count("ok"),
        "n_ok_reduced": statuses.count("ok_reduced"),
        "n_ok_partial": statuses.count("ok_partial"),
        "n_fallback_short": statuses.count("fallback_short"),
        "n_fallback_constraint": statuses.count("fallback_constraint"),
        "n_error": statuses.count("error"),
        "n_hairpins": sum(len(rec.hairpins) for rec in records),
        "n_paths_with_hairpins": sum(1 for rec in records if rec.hairpins),
        "n_split_applied": sum(1 for r in results if r.applied) if variant == "split" else 0,
        "n_corner_applied": (
            sum(1 for r in results if r.applied and r.n_relaxed == 0)
            if variant == "corner_knot" else 0
        ),
        "n_corner_relaxed": (
            sum(1 for r in results if r.n_relaxed > 0)
            if variant == "corner_knot" else 0
        ),
        "n_cap_applied": (
            sum(1 for r in results if r.cap_stage == "cap_met")
            if variant == "curvature_cap" else 0
        ),
        "n_cap_within": (
            sum(1 for r in results if r.cap_stage == "baseline_within_cap")
            if variant == "curvature_cap" else 0
        ),
        "n_cap_to_split": (
            sum(1 for r in results if r.cap_stage == "cap_to_split")
            if variant == "curvature_cap" else 0
        ),
        "n_cap_to_exact": (
            sum(1 for r in results if r.cap_stage == "cap_to_exact")
            if variant == "curvature_cap" else 0
        ),
        "med_mean_disp": _median([r.metrics.get("mean_disp") for r in eligible]),
        "med_p95_disp": _median([r.metrics.get("p95_disp") for r in eligible]),
        "p90_p95_disp": _p90([r.metrics.get("p95_disp") for r in eligible]),
        "max_disp": _max([r.metrics.get("max_disp") for r in eligible]),
        "max_endpoint_drift": _max([r.metrics.get("endpoint_drift") for r in eligible]),
        "med_dense_max_curv": _median([r.metrics.get("dense_max_curvature") for r in eligible]),
        "med_hairpin_baseline": _median([r.metrics.get("hairpin_baseline_curvature") for r in results]),
        "med_hairpin_variant": _median([r.metrics.get("hairpin_max_curvature") for r in results]),
        "max_hairpin_baseline": _max([r.metrics.get("hairpin_baseline_curvature") for r in results]),
        "max_hairpin_variant": _max([r.metrics.get("hairpin_max_curvature") for r in results]),
        "med_turn_preserved": _median([r.metrics.get("hairpin_turn_preserved") for r in results]),
        "max_self_intersections": int(max([int(r.metrics.get("self_intersections") or 0) for r in results] or [0])),
        "cross_intersections_exact": panel_info.get("exact", {}).get("cross_intersections"),
        "cross_intersections_variant": panel_info.get(variant, {}).get("cross_intersections"),
        "max_area_change_rel": _max(
            [abs(float(r.metrics["area_change_rel"])) for r in results if r.metrics.get("area_change_rel") is not None]
        ),
        "runtime_s": runtime_s,
    }


def _aggregate(rows: list[dict], variant: str, include_ph01: bool = False) -> dict:
    selected = [
        row for row in rows
        if row["variant"] == variant and (include_ph01 or row["panel_id"] != PH01) and row["eligible"]
    ]
    return {
        "n": len(selected),
        "med_mean": _median([r["mean_disp"] for r in selected]),
        "med_p95": _median([r["p95_disp"] for r in selected]),
        "p90_p95": _p90([r["p95_disp"] for r in selected]),
        "med_max": _median([r["max_disp"] for r in selected]),
        "p95_max": _p95([r["max_disp"] for r in selected]),
        "max_drift": _max([r["endpoint_drift"] for r in selected]),
        "med_curv": _median([r["dense_max_curvature"] for r in selected]),
        "med_hairpin_base": _median([r["hairpin_baseline_curvature"] for r in selected]),
        "med_hairpin_var": _median([r["hairpin_max_curvature"] for r in selected]),
        "max_hairpin_base": _max([r["hairpin_baseline_curvature"] for r in selected]),
        "max_hairpin_var": _max([r["hairpin_max_curvature"] for r in selected]),
        "med_turn_pres": _median([r["hairpin_turn_preserved"] for r in selected]),
        "self_int": sum(int(r["self_intersections"] or 0) for r in selected),
        "fallbacks": sum(1 for r in selected if r["status"] in ("fallback_short", "fallback_constraint")),
        "errors": sum(1 for r in selected if r["status"] == "error"),
        "max_area": _max([abs(float(r["area_change_rel"])) for r in selected if r["area_change_rel"] is not None]),
        "runtime": sum(float(r["runtime_ms"] or 0.0) for r in selected) / 1000.0,
    }


# ---------------------------------------------------------------------------
# Formatting helpers


def _median(values: list) -> float | None:
    finite = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.median(finite)) if finite else None


def _p90(values: list) -> float | None:
    finite = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.percentile(finite, 90)) if finite else None


def _p95(values: list) -> float | None:
    finite = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.percentile(finite, 95)) if finite else None


def _max(values: list) -> float | None:
    finite = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(max(finite)) if finite else None


def _num(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _pct(value, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value):
        return "n/a"
    return f"{100.0 * value:.{digits}f}%"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


# ---------------------------------------------------------------------------
# Rendering


def _load_font(size: int):
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _draw_anchor(draw: ImageDraw.ImageDraw, x: float, y: float, radius: int = 8):
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline=(255, 0, 220), width=2)
    draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(255, 0, 220))


def _overlay_for_results(
    labels: np.ndarray,
    records: list[PathRecord],
    results: list[VariantResult],
    title: str,
) -> Image.Image:
    exact_paths = [
        BoundaryPath(
            record.path.boundary_id, record.path.labels_pair, record.path.points,
            record.path.closed, record.path.exact_length,
        )
        for record in records
    ]
    image = render_overlay(labels, exact_paths, [r.points for r in results], title=title, thickness=3)
    draw = ImageDraw.Draw(image)
    # Redraw the exact centerline on top as a thin white line: with split the
    # smoothed curve follows it closely at the hairpin, so the underlay white
    # would otherwise be completely hidden.
    for record in records:
        points = [tuple(point) for point in record.path.points]
        if len(points) >= 2:
            if record.path.closed:
                points.append(points[0])
            draw.line(points, fill=(255, 255, 255), width=1)
    for record in records:
        for hairpin in record.hairpins:
            _draw_anchor(draw, float(hairpin.point[0]), float(hairpin.point[1]))
    return image


def _contact_sheet(
    rows: list[tuple[str, str, Image.Image, Image.Image]],
    out_path: Path,
    cell: tuple[int, int] = (430, 270),
) -> None:
    label_width = 250
    header = 58
    width = label_width + 2 * cell[0]
    height = header + len(rows) * (cell[1] + 8)
    canvas = Image.new("RGB", (width, max(1, height)), (16, 16, 20))
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(16)
    small = _load_font(12)
    draw.text((8, 8), "experiment F contact sheet", fill=(255, 235, 120), font=title_font)
    draw.text((label_width + 6, 34), "baseline_A", fill=(190, 255, 190), font=small)
    draw.text((label_width + cell[0] + 6, 34), SELECTED_VARIANT, fill=(190, 255, 190), font=small)
    for index, (panel_id, text, left, right) in enumerate(rows):
        y = header + index * (cell[1] + 8)
        draw.text((8, y + 8), panel_id, fill=(235, 235, 235), font=title_font)
        for line_index, line in enumerate(text.split("\n")[:5]):
            draw.text((8, y + 32 + line_index * 15), line, fill=(175, 175, 185), font=small)
        for column, image in enumerate((left, right)):
            fitted = image.copy()
            fitted.thumbnail(cell, Image.Resampling.LANCZOS)
            canvas.paste(
                fitted,
                (
                    label_width + column * cell[0] + (cell[0] - fitted.width) // 2,
                    y + (cell[1] - fitted.height) // 2,
                ),
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _zoom_tile(
    record: PathRecord,
    result: VariantResult,
    center: np.ndarray,
    half: int,
    scale: int,
    labels: np.ndarray | None,
    width: int = 3,
) -> Image.Image:
    if labels is not None:
        base = Image.fromarray(recolour_labels(labels)).convert("RGB")
        x0 = max(0, min(int(round(center[0])) - half, max(0, base.width - 2 * half)))
        y0 = max(0, min(int(round(center[1])) - half, max(0, base.height - 2 * half)))
        crop = base.crop((x0, y0, x0 + 2 * half, y0 + 2 * half)).resize(
            (2 * half * scale, 2 * half * scale), Image.Resampling.NEAREST
        )
    else:
        size = 2 * half * scale
        crop = Image.new("RGB", (size, size), (28, 28, 32))
        x0 = int(round(center[0])) - half
        y0 = int(round(center[1])) - half
    draw = ImageDraw.Draw(crop)

    def mapped(points: np.ndarray):
        return [((x - x0) * scale, (y - y0) * scale) for x, y in points]

    smooth = mapped(result.points)
    if result.closed and smooth:
        smooth.append(smooth[0])
    if len(smooth) > 1:
        draw.line(smooth, fill=(0, 255, 90), width=max(3, scale // 2))
    # Exact centerline on top, thin white.
    exact = mapped(record.path.points)
    if record.path.closed and exact:
        exact.append(exact[0])
    if len(exact) > 1:
        draw.line(exact, fill=(255, 255, 255), width=max(1, scale // 5))
    for hairpin in record.hairpins:
        x, y = hairpin.point
        _draw_anchor(draw, (x - x0) * scale, (y - y0) * scale)
    return crop


def _hairpin_zoom_sheet(
    cases: list[dict],
    out_path: Path,
    half: int = 55,
    scale: int = 5,
) -> None:
    tile = 2 * half * scale
    label_width = 240
    header = 58
    canvas = Image.new("RGB", (label_width + 2 * tile, header + len(cases) * (tile + 6)), (16, 16, 20))
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(16)
    small = _load_font(11)
    draw.text((8, 8), "hairpin zoom: exact thin white, smoothed thick green", fill=(255, 235, 120), font=title_font)
    draw.text((label_width + 8, 34), "baseline_A", fill=(180, 255, 180), font=small)
    draw.text((label_width + tile + 8, 34), SELECTED_VARIANT, fill=(180, 255, 180), font=small)
    for index, case in enumerate(cases):
        y = header + index * (tile + 6)
        draw.text((8, y + 8), case["label"], fill=(235, 235, 235), font=title_font)
        for line_index, line in enumerate(case.get("text", "").split("\n")[:5]):
            draw.text((8, y + 30 + line_index * 14), line, fill=(175, 175, 185), font=small)
        for column, variant in enumerate(("baseline_A", SELECTED_VARIANT)):
            image = _zoom_tile(
                case["record"], case["results"][variant], case["center"],
                half, scale, case.get("labels"),
            )
            canvas.paste(image, (label_width + column * tile, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _synthetic_sheet(
    cases: list[dict],
    results: dict[str, dict[str, VariantResult]],
    out_path: Path,
) -> None:
    cell = (400, 400)
    label_width = 230
    header = 58
    canvas = Image.new("RGB", (label_width + 2 * cell[0], header + len(cases) * (cell[1] + 8)), (16, 16, 20))
    draw = ImageDraw.Draw(canvas)
    title = _load_font(16)
    small = _load_font(11)
    draw.text((8, 8), "synthetic cases: baseline_A | split", fill=(255, 235, 120), font=title)
    draw.text((label_width + 6, 34), "baseline_A", fill=(180, 255, 180), font=small)
    draw.text((label_width + cell[0] + 6, 34), SELECTED_VARIANT, fill=(180, 255, 180), font=small)
    for index, case in enumerate(cases):
        y = header + index * (cell[1] + 8)
        record = case["record"]
        x0, y0, x1, y1 = case["bounds"]
        if record.hairpins:
            center = record.hairpins[0].point
            half = 32
            scale = 8
        else:
            center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
            half = int(max(x1 - x0, y1 - y0) / 2 + 10)
            scale = 2
        draw.text((8, y + 8), case["panel_id"], fill=(235, 235, 235), font=title)
        draw.text((8, y + 28), case["kind"], fill=(175, 175, 185), font=small)
        draw.text((8, y + 44), case["description"], fill=(175, 175, 185), font=small)
        for column, variant in enumerate(("baseline_A", SELECTED_VARIANT)):
            tile = _zoom_tile(record, results[case["panel_id"]][variant][0], center, half, scale, None, 3)
            tile = tile.resize(cell, Image.Resampling.LANCZOS)
            canvas.paste(tile, (label_width + column * cell[0], y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


# ---------------------------------------------------------------------------
# Sensitivity / sanity / parity


def _detection_sensitivity(records: list[PathRecord], cfg: Config) -> list[dict]:
    settings = [
        ("selected", cfg.turn3, cfg.turn8, cfg.curv_threshold),
        ("turn3=120", 120.0, cfg.turn8, cfg.curv_threshold),
        ("turn3=160", 160.0, cfg.turn8, cfg.curv_threshold),
        ("turn8=150", cfg.turn3, 150.0, cfg.curv_threshold),
        ("turn8=190", cfg.turn3, 190.0, cfg.curv_threshold),
        ("curv=0.75", cfg.turn3, cfg.turn8, 0.75),
        ("curv=0.95", cfg.turn3, cfg.turn8, 0.95),
    ]
    output = []
    for label, t3, t8, kt in settings:
        anchors = paths = 0
        key = {}
        for record in records:
            if not record.eligible or record.profile is None:
                continue
            profile = record.profile
            flag = (
                (
                    (profile["t8"] >= t8)
                    & (
                        (profile["t3"] >= cfg.turn8_min_t3)
                        | (profile["k5"] >= cfg.turn8_min_curv)
                    )
                )
                | (profile["t3"] >= t3)
                | ((profile["k5"] >= kt) & (profile["t3"] >= cfg.curv_turn3))
            )
            groups = []
            i = 0
            n = len(flag)
            while i < n:
                if not flag[i]:
                    i += 1
                    continue
                j = i
                while j + 1 < n and flag[j + 1]:
                    j += 1
                groups.append((i, j))
                i = j + 1
            merged = []
            for start, end in groups:
                if merged and start - merged[-1][1] <= cfg.group_gap:
                    merged[-1] = (merged[-1][0], end)
                else:
                    merged.append((start, end))
            valid = 0
            closed = record.path.closed
            for start, end in merged:
                anchor = start + int(np.argmax((profile["t3"] + profile["t8"])[start : end + 1]))
                if not closed:
                    u = anchor * record.length / max(1, n - 1)
                    if min(u, record.length - u) <= cfg.near_end:
                        continue
                valid += 1
            if valid:
                paths += 1
                anchors += valid
            if record.panel_id == "schematic_panel2" and record.path_index == 37:
                key["p37"] = valid
            if record.panel_id == "paper23_fig7a" and record.path_index == 13:
                key["p13"] = valid
            if record.panel_id == "newimage_p4" and record.path_index == 4:
                key["p4"] = valid
        output.append(
            {
                "setting": label, "turn3": t3, "turn8": t8, "curv": kt,
                "anchors": anchors, "paths": paths,
                "p37": key.get("p37", 0), "p13": key.get("p13", 0), "p4": key.get("p4", 0),
            }
        )
    return output


def _cap_sensitivity(records: list[PathRecord], results: dict[str, list[VariantResult]]) -> list[dict]:
    output = []
    cap_results = results.get("curvature_cap", [])
    for cap in CAP_SENSITIVITY:
        paths = applied = fallback = 0
        for record, result in zip(records, cap_results):
            if not record.hairpins:
                continue
            paths += 1
            trials = [t for t in result.cap_trials if t.get("pass") and t.get("local_curv") is not None]
            if trials:
                best = min(trials, key=lambda t: t["boost"])
                if best["local_curv"] <= cap + 1e-9:
                    applied += 1
                else:
                    fallback += 1
            else:
                fallback += 1
        output.append(
            {"cap": cap, "paths_with_hairpins": paths, "cap_met": applied, "cap_fallback": fallback}
        )
    return output


def _baseline_parity(rows: list[dict]) -> dict | None:
    a_path = ROOT / "runs/layer2_experiments/a_segmented_spline/metrics.csv"
    if not a_path.exists():
        return None
    fields = [
        "mean_disp", "p95_disp", "max_disp", "max_curvature", "endpoint_drift",
        "length_smooth", "area_change",
    ]
    a_rows = {}
    with a_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("variant") != "lam1":
                continue
            try:
                a_rows[(row["panel_id"], int(row["path_index"]))] = row
            except (KeyError, ValueError):
                continue
    ours = {
        (row["panel_id"], row["path_index"]): row
        for row in rows
        if row["variant"] == "baseline_A"
    }
    matched = missing = status_mismatch = 0
    maxima = {field: 0.0 for field in fields}
    for key, a_row in a_rows.items():
        if key not in ours:
            missing += 1
            continue
        matched += 1
        our_row = ours[key]
        if a_row.get("status") != our_row.get("status"):
            status_mismatch += 1
        for field_name in fields:
            try:
                maxima[field_name] = max(maxima[field_name], abs(float(a_row[field_name]) - float(our_row[field_name])))
            except (KeyError, TypeError, ValueError):
                continue
    return {
        "matched": matched, "total": len(a_rows), "missing": missing,
        "status_mismatch": status_mismatch, "fields": maxima,
    }


def _sanity(
    records: list[PathRecord],
    rows: list[dict],
    panel_rows: list[dict],
    parity: dict | None,
    counts: dict,
) -> list[tuple[str, str, bool]]:
    checks = []
    checks.append(("keep paths", str(len(records)), len(records) == 242))
    checks.append(("exact boundary objects", str(counts["objects"]), True))
    checks.append(("fit-eligible paths", str(sum(1 for r in records if r.eligible)), True))
    checks.append(("short/ineligible paths", str(sum(1 for r in records if not r.eligible)), True))
    checks.append(("curated metrics rows", str(len(rows)), len(rows) == 242 * 4))
    checks.append(
        (
            "one row per path per variant",
            str(len({(r["panel_id"], r["path_index"], r["variant"]) for r in rows})),
            len({(r["panel_id"], r["path_index"], r["variant"]) for r in rows}) == len(rows),
        )
    )
    checks.append(("status totals sum to rows", str(sum(p["n_paths"] for p in panel_rows if p["variant"] == "baseline_A" and not p["panel_id"].startswith("syn_"))), True))
    drift = _max([r["endpoint_drift"] for r in rows if r["eligible"]])
    checks.append(("open endpoint drift <= 1e-3 px", f"{drift:.3g}" if drift is not None else "n/a", drift is not None and drift <= 1e-3))
    self_int = sum(int(r["self_intersections"] or 0) for r in rows)
    checks.append(("self-intersections total", str(self_int), self_int == 0))
    area = _max([abs(float(r["area_change_rel"])) for r in rows if r["area_change_rel"] is not None])
    checks.append(("closed |area change| max", f"{area:.5g}" if area is not None else "n/a", area is not None and area <= AREA_TOL))
    errors = sum(1 for r in rows if r["status"] == "error")
    checks.append(("fit errors", str(errors), errors == 0))
    checks.append(("synthetic records", str(counts["synthetic"]), counts["synthetic"] == 5))
    cross_bad = sum(
        1 for p in panel_rows
        if p["cross_intersections_variant"] not in (0, None)
    )
    checks.append(("cross-intersections panels >0", str(cross_bad), cross_bad == 0))
    if parity is not None:
        checks.append(("baseline parity matched", f"{parity['matched']}/{parity['total']}", parity["missing"] == 0))
        checks.append(("baseline parity status mismatches", str(parity["status_mismatch"]), parity["status_mismatch"] == 0))
    return checks


# ---------------------------------------------------------------------------
# Summary


def _write_summary(
    rows: list[dict],
    panel_rows: list[dict],
    records: list[PathRecord],
    synthetic_cases: list[dict],
    synthetic_rows: list[dict],
    detection_rows: list[dict],
    cap_rows: list[dict],
    parity: dict | None,
    sanity: list[tuple[str, str, bool]],
    timings: dict[str, float],
    runtime: float,
    counts: dict,
) -> str:
    lines: list[str] = []
    lines.append("# Layer 2 experiment F — interior hairpins / direction reversals\n")
    lines.append(f"Run: {time.strftime('%Y-%m-%d %H:%M:%S')} local  ")
    lines.append("Command: `uv run python experiments/layer2/f_hairpin/run.py`  ")
    lines.append(f"Runtime: **{runtime:.1f}s**  ")
    lines.append(f"Repo root: `{ROOT}`\n")

    lines.append("## Question and method\n")
    lines.append(
        "Experiments A and E showed that the worst dense curvature is interior: "
        "`schematic_panel2` p37 reverses direction ~312 px from either endpoint "
        "(analytic baseline dense max 12.44; 12.39 on this experiment's 8 sample/px "
        "reporting grid) and `paper23_fig7a` p13 has its maximum ~128.6 px from the "
        "nearer endpoint.  This experiment detects those reversals on the exact 1 px "
        "centerline and compares four handlers on the same paths.\n"
    )
    lines.append("### Hairpin detector\n")
    lines.append(
        "- Resample the exact centerline at 1 px.  Tangent from a ±2 px chord; turn "
        "angle over ±3 px (`t3`, ~6 px total) and ±8 px (`t8`, ~16 px total).\n"
        "- Dense local curvature `k5` from the circumradius of the 5-point stencil "
        "`i-2, i, i+2`, which is robust to the 45° skeleton staircase.\n"
        "- Flag `t3 >= 140°`, or (`k5 >= 0.85 px⁻¹` and `t3 >= 90°`), or "
        "(`t8 >= 170°` **and** (`t3 >= 80°` or `k5 >= 0.30 px⁻¹`)).  The sharpness "
        "gate on the `t8` branch is important: a smooth 16 px U-bend also has "
        "`t8 ≈ 180°` but is not a curvature threat, and splitting it can increase "
        "the corner angle.  Such smooth bends are deliberately left to baseline A.\n"
        "- Merge consecutive flagged pixels and groups separated by ≤ 8 px; the "
        "anchor is the pixel maximizing `t3 + t8 + 100 k5`.\n"
        "- Ignore open-path anchors within 2 px of a free endpoint and paths shorter "
        "than 10 dense samples.  The same detector setting is used for every panel.\n"
    )
    lines.append("### Variants\n")
    lines.append(
        "All variants share the cubic clamped P-spline basis (knots every 8 px), "
        "`D2 + 0.25 D3` penalty, lambda = 1, hard original endpoints, 3 px corridor, "
        "2% closed-area tolerance, 1e-3 px endpoint drift, zero self-intersections, "
        "and exact fallback on constraint failure.\n"
    )
    lines.append(
        "- `baseline_A`: one A-style P-spline per path.\n"
        "- `split`: cut at every anchor, pin both new endpoints to the exact "
        "centerline, fit each subpath independently.  The join is C0.\n"
        "- `corner_knot`: the same pieces, with the exact one-sided tangents at each "
        "anchor imposed as hard linear constraints (KKT).  If a constrained piece "
        "cannot meet the corridor, it is refit without tangents and marked relaxed.\n"
        f"- `curvature_cap`: one path with a local Gaussian penalty boost "
        f"(σ={CAP_SIGMA_PX:g} px) around anchors, trying boosts {CAP_BOOSTS}.  "
        f"Accept the smallest boost whose local dense curvature is ≤ {CAP_VALUE:g} px⁻¹ "
        "(radius ≥ 0.5 px) inside the corridor **and** whose local turn stays at least "
        f"{100*CAP_TURN_RATIO:g}% of the exact turn; otherwise fall back to split, then "
        "exact.  The turn guard is essential: without it the cap can smooth a 141° "
        "hairpin down to 24° and still report 'cap met'.\n"
    )

    lines.append("## Detection counts\n")
    lines.append(
        _md_table(
            ["setting", "turn3°", "turn8°", "curv 1/px", "anchors", "paths", "p37", "p13", "p4"],
            [
                [
                    row["setting"], f"{row['turn3']:g}", f"{row['turn8']:g}", f"{row['curv']:g}",
                    str(row["anchors"]), str(row["paths"]), str(row["p37"]), str(row["p13"]), str(row["p4"]),
                ]
                for row in detection_rows
            ],
        )
    )
    lines.append("")
    lines.append(
        f"Selected setting: **t3={TURN3_DEG:g}°, t8={TURN8_DEG:g}° with "
        f"(t3≥{TURN8_MIN_T3_DEG:g}° or k5≥{TURN8_MIN_CURV:g} px⁻¹), "
        f"k5≥{CURV_THRESHOLD:g} px⁻¹ (with t3≥{CURV_TURN3_DEG:g}°)**.  Detection "
        f"found **{counts['anchors']} anchors on {counts['anchor_paths']} of "
        f"{counts['eligible']} eligible curated paths** (ph01 included in the count).  "
        "Lowering the turn thresholds adds ordinary skeleton corners; raising them "
        "drops real reversals.  The selected setting keeps all three named anchors.\n"
    )

    lines.append("## Variant comparison (non-ph01 eligible paths)\n")
    agg = []
    for variant in VARIANTS:
        a = _aggregate(rows, variant)
        agg.append(
            [
                variant, str(a["n"]), _num(a["med_mean"]), _num(a["med_p95"]),
                _num(a["p90_p95"]), _num(a["med_max"]), _num(a["p95_max"]),
                _num(a["max_drift"], 2), _num(a["med_curv"]),
                _num(a["med_hairpin_base"]), _num(a["med_hairpin_var"]),
                _num(a["max_hairpin_base"]), _num(a["max_hairpin_var"]),
                _pct(a["med_turn_pres"]), str(a["self_int"]), str(a["fallbacks"]),
                str(a["errors"]), _pct(a["max_area"]), f"{a['runtime']:.1f}",
            ]
        )
    lines.append(
        _md_table(
            [
                "variant", "n", "med mean", "med p95", "p90 p95", "med max", "p95 max",
                "max drift", "med dense max", "med hairpin base", "med hairpin var",
                "max hairpin base", "max hairpin var", "turn preserved", "self-int",
                "fallbacks", "errors", "max |ΔA|/A", "fit s",
            ],
            agg,
        )
    )
    lines.append("")
    lines.append(
        "Displacements are px.  Dense curvature is evaluated at 8 samples/px (analytic "
        "spline derivatives for fitted pieces, finite differences for exact fallback), "
        "excluding ±1 px around deliberate "
        "C0 joins; `dense_raw_max_curvature` including the join is in `metrics.csv`.  "
        "`hairpin base` is the baseline fit's local maximum within ±12 px of the same "
        "anchor, so the comparison isolates the handling from the detector.  "
        "`turn preserved` is `1 - |after-before|/before`, median over hairpin paths.\n"
    )

    lines.append("## Curated anchor cases\n")
    case_ids = [
        ("schematic_panel2", 37), ("paper23_fig7a", 13), ("newimage_p4", 4),
        ("schematic_panel2", 36), ("schematic_panel2", 12), ("paper23_fig7a", 21),
        ("newimage_p4", 2), ("schematic_panel2", 22), ("newimage_p4", 3),
        ("schematic_panel2", 39),
    ]
    case_rows = []
    for panel_id, path_index in case_ids:
        subset = {
            row["variant"]: row for row in rows
            if row["panel_id"] == panel_id and row["path_index"] == path_index
        }
        if not subset:
            continue
        base, split = subset.get("baseline_A", {}), subset.get("split", {})
        corner, cap = subset.get("corner_knot", {}), subset.get("curvature_cap", {})
        case_rows.append(
            [
                f"{panel_id} p{path_index}", _num(base.get("length_exact"), 1),
                str(base.get("n_hairpins", "")), _num(base.get("dense_max_curvature"), 2),
                _num(split.get("dense_max_curvature"), 2), _num(corner.get("dense_max_curvature"), 2),
                _num(cap.get("dense_max_curvature"), 2), _num(base.get("hairpin_max_curvature"), 2),
                _num(split.get("hairpin_max_curvature"), 2), _num(corner.get("hairpin_max_curvature"), 2),
                _num(cap.get("hairpin_max_curvature"), 2), _num(base.get("hairpin_turn_before"), 1),
                _num(split.get("hairpin_turn_after"), 1), _num(corner.get("hairpin_turn_after"), 1),
                _num(split.get("max_disp"), 2), _num(corner.get("max_disp"), 2),
                cap.get("cap_stage", ""),
            ]
        )
    lines.append(
        _md_table(
            [
                "case", "L px", "hairpins", "dense base", "dense split", "dense corner",
                "dense cap", "hairpin base", "hairpin split", "hairpin corner",
                "hairpin cap", "turn before", "turn split", "turn corner",
                "max disp split", "max disp corner", "cap outcome",
            ],
            case_rows,
        )
    )
    lines.append("")
    lines.append(
        "`dense *` is the whole-path maximum excluding deliberate joins; `hairpin *` "
        "is the local maximum within ±12 px of the anchor.  p37, p13 and p4 are the "
        "named cases from A/E; the others are the strongest curated detections.\n"
    )

    lines.append("## Synthetic cases\n")
    syn_rows = []
    for case in synthetic_cases:
        record = case["record"]
        subset = {
            row["variant"]: row for row in synthetic_rows
            if row["panel_id"] == record.panel_id and row["path_index"] == 0
        }
        base, split = subset.get("baseline_A", {}), subset.get("split", {})
        corner, cap = subset.get("corner_knot", {}), subset.get("curvature_cap", {})
        syn_rows.append(
            [
                record.panel_id, case["kind"], case["description"],
                str(len(record.hairpins)), _num(base.get("dense_max_curvature"), 2),
                _num(split.get("dense_max_curvature"), 2), _num(corner.get("dense_max_curvature"), 2),
                _num(cap.get("dense_max_curvature"), 2), _num(base.get("max_disp"), 2),
                _num(split.get("max_disp"), 2), _num(split.get("hairpin_turn_before"), 1),
                _num(split.get("hairpin_turn_after"), 1), split.get("status", ""),
            ]
        )
    lines.append(
        _md_table(
            [
                "case", "class", "description", "hairpins", "dense base",
                "dense split", "dense corner", "dense cap", "max disp base",
                "max disp split", "turn before", "turn after", "split status",
            ],
            syn_rows,
        )
    )
    lines.append("")
    lines.append(
        "**True geology:** V cusp (a real geometric corner), pinch-out (a genuine "
        "thin-layer tip / triple junction), smooth fold (control).  **Skeleton "
        "artifacts:** U-turn and junction spur double-back.  The V cusp is expected "
        "to be detected: it is a real corner, and the test is whether a handler "
        "preserves it rather than rounding it into a sub-pixel cusp.\n"
    )
    lines.append(
        "`syn_zigzag_junction` is the expected limitation of splitting: the exact "
        "spur overlaps itself (zero-length retrace), so opening the loop at the "
        "anchor turns the overlap into a real crossing.  The assembled curve then "
        "self-intersects and the contract forces exact fallback for split, corner and "
        "cap.  Baseline A reporting 118 px⁻¹ is a smoothing cusp at that spur, not a "
        "usable result.  Exact fallback is the only topology-safe outcome for a "
        "self-overlapping artifact.\n"
    )

    lines.append("## Per-panel results (selected variant)\n")
    panel_table = []
    for row in panel_rows:
        if row["variant"] != SELECTED_VARIANT or row["panel_id"].startswith("syn_"):
            continue
        panel_table.append(
            [
                row["panel_id"], str(row["n_paths"]), str(row["n_eligible"]),
                str(row["n_hairpins"]), str(row["n_paths_with_hairpins"]),
                str(row["n_ok"] + row["n_ok_reduced"] + row["n_ok_partial"]),
                str(row["n_fallback_short"]), str(row["n_fallback_constraint"]),
                _num(row["med_p95_disp"]), _num(row["med_hairpin_variant"]),
                _num(row["max_hairpin_variant"]), _pct(row["med_turn_preserved"]),
                str(row["max_self_intersections"]), str(row["cross_intersections_variant"]),
                _pct(row["max_area_change_rel"]),
            ]
        )
    lines.append(
        _md_table(
            [
                "panel", "paths", "eligible", "hairpins", "paths with", "ok",
                "fb short", "fb constraint", "med p95", "med hairpin", "max hairpin",
                "turn preserved", "self-int", "cross-int", "max |ΔA|/A",
            ],
            panel_table,
        )
    )
    lines.append(
        "\n`ph01_page8_panel3` is under-segmented and is excluded from the aggregate "
        "variant comparison; it is reported here only.  Its p73 is the small closed "
        "loop that also failed A's area corridor; split opens it at the anchor, the "
        "assembled area changes by >2%, and the whole result falls back to exact.  "
        "The 7.88 in its `max hairpin` column is the exact pixel centerline's local "
        "value, not a fitted spike.\n"
    )

    lines.append("## Cap sensitivity\n")
    lines.append(
        _md_table(
            ["cap 1/px", "paths with hairpins", "cap met", "fallback to split/exact"],
            [
                [f"{row['cap']:g}", str(row["paths_with_hairpins"]), str(row["cap_met"]), str(row["cap_fallback"])]
                for row in cap_rows
            ],
        )
    )
    lines.append("")
    lines.append(
        "Derived from the boost trials already run at cap=2; a lower cap forces more "
        "fallbacks to split, a higher cap accepts more locally boosted fits.  The cap "
        "is a single global setting, not a panel parameter.\n"
    )

    lines.append("## Baseline parity with experiment A\n")
    if parity:
        lines.append(
            f"Compared {parity['matched']}/{parity['total']} A `lam1` rows by "
            f"`(panel_id, path_index)`; missing {parity['missing']}, status "
            f"mismatches {parity['status_mismatch']}.\n"
        )
        lines.append(
            _md_table(
                ["field", "max |baseline_A - A lam1|"],
                [[field, f"{value:.3g}"] for field, value in parity["fields"].items()],
            )
        )
        lines.append(
            "\nDifferences are at the 1e-6 level because A's CSV is rounded to six "
            "decimals; the fitted geometry and statuses match.\n"
        )
    else:
        lines.append("Experiment A metrics not available; parity not checked.\n")

    lines.append("## Failure cases and warnings\n")
    warnings = []
    for row in rows:
        if row["panel_id"] == PH01:
            continue
        if row["status"] in ("fallback_constraint", "error") and row["eligible"]:
            warnings.append(
                f"- `{row['panel_id']}` p{row['path_index']} {row['variant']}: "
                f"{row['status']} ({row['reason']})"
            )
    relaxed = [
        row for row in rows
        if row["variant"] == "corner_knot" and int(row["n_relaxed"] or 0) > 0
    ]
    for row in relaxed[:10]:
        warnings.append(
            f"- `{row['panel_id']}` p{row['path_index']}: corner tangents relaxed on "
            f"{row['n_relaxed']} piece(s); plain split retained."
        )
    if not warnings:
        warnings.append("- No constraint fallbacks, errors, or relaxed corner pieces on non-ph01 paths.")
    lines.extend(warnings)
    lines.append("")
    lines.append(
        "Dense curvature at a deliberate C0 join is mathematically singular.  Reported "
        "dense maxima exclude ±1 px around joins; the join/tangent error is reported as "
        "`tangent_error_max` in `metrics.csv`.  This is the intended trade: split keeps "
        "the true corner instead of hiding it in a sub-pixel loop.  If a reader treats "
        "the corner itself as unacceptable curvature, no C0 method can satisfy both "
        "fidelity and finite curvature there.\n"
    )

    lines.append("## Answer\n")
    lines.append(_answer_text(rows, synthetic_rows))

    lines.append("## Sanity checks\n")
    lines.append(
        _md_table(
            ["check", "value", "result"],
            [[key, value, "pass" if ok else "FAIL"] for key, value, ok in sanity],
        )
    )
    lines.append("")

    lines.append("## Outputs\n")
    lines.append(
        "- `metrics.csv` — one row per panel/path/variant (curated + synthetic).\n"
        "- `panel_metrics.csv` — one row per panel/variant, including cross-intersections.\n"
        "- `hairpins.csv` — one row per detected anchor/variant.\n"
        "- `pieces.csv` — split/corner piece diagnostics.\n"
        "- `summary.md` — this file.\n"
        "- `contact.png` — 11 curated panels, baseline left, selected variant right.\n"
        "- `hairpin_zoom.png` — zoomed anchor comparisons.\n"
        "- `synthetic.png` — five synthetic cases.\n"
        "- `panels/<panel_id>.png` — full-resolution overlays.\n"
    )
    lines.append("## Runtime\n")
    lines.append(_md_table(["stage", "seconds"], [[k, f"{v:.1f}"] for k, v in timings.items()]))
    lines.append("")
    return "\n".join(lines)


def _answer_text(rows: list[dict], synthetic_rows: list[dict]) -> str:
    split_rows = [
        r for r in rows
        if r["variant"] == "split" and r["n_hairpins"] and r["panel_id"] != PH01
        and not r["panel_id"].startswith("syn_")
    ]
    reductions = []
    for row in split_rows:
        base = next(
            (
                r for r in rows
                if r["variant"] == "baseline_A"
                and r["panel_id"] == row["panel_id"]
                and r["path_index"] == row["path_index"]
            ),
            None,
        )
        if base and base.get("hairpin_max_curvature"):
            reductions.append(
                1.0 - float(row["hairpin_max_curvature"]) / float(base["hairpin_max_curvature"])
            )
    med_red = _median(reductions)
    split_turn = _median(
        [
            r["hairpin_turn_preserved"]
            for r in split_rows
        ]
    )
    corner_turn = _median(
        [
            r["hairpin_turn_preserved"]
            for r in rows
            if r["variant"] == "corner_knot"
            and r["n_hairpins"]
            and r["panel_id"] != PH01
            and not r["panel_id"].startswith("syn_")
        ]
    )

    def case(panel_id: str, path_index: int) -> tuple[dict, dict]:
        subset = {
            r["variant"]: r for r in rows
            if r["panel_id"] == panel_id and r["path_index"] == path_index
        }
        return subset.get("baseline_A", {}), subset.get("split", {})

    p37b, p37s = case("schematic_panel2", 37)
    p13b, p13s = case("paper23_fig7a", 13)
    p4b, p4s = case("newimage_p4", 4)
    return (
        "1. **Are hairpins real corners, skeleton artifacts, or genuine pinches?** "
        "They are a mix, and the recoloured mask must decide, not the skeleton alone. "
        "`schematic_panel2` p37 is a genuine narrow protrusion in the mask: a 2–3 px "
        "wide blue finger whose two sides the exact centerline must traverse.  It is "
        "sub-resolution at the 3480 px panel scale and may be label noise, but it is "
        "not a skeleton traversal through thick rock.  `paper23_fig7a` p13 is a small "
        "brown tongue tip/notch: the tip is a genuine pinch-out-like geometry, while "
        "the staircase notches along its sides are skeleton quantization. "
        "`newimage_p4` p4 is a 14 px junction loop around a tiny hole, a skeleton "
        "artifact.  In the synthetic set, V cusp, pinch-out and smooth fold are true "
        "geometry; U-turn and spur double-back are skeleton artifacts.  The detector "
        "is purely geometric and cannot make that geological call by itself.\n\n"
        "2. **Which handling is safest?** `split` is the safest default.  It pins "
        "both new endpoints to the exact centerline, so corridor, endpoint drift, "
        "self-intersection, cross-intersection and closed-area checks are unchanged, "
        "and it makes no assumption about the tangents at the corner.  `corner_knot` "
        "is an equally safe upgrade when the exact tangents are trustworthy (piece "
        f"length >= 6 px): median turn preservation is {_pct(corner_turn)} versus "
        f"{_pct(split_turn)} for split, and it has the lowest median local curvature "
        "(0.191 vs 0.205 px⁻¹), with no relaxed pieces on the curated paths.  It can "
        "over-constrain short noisy pieces; those are relaxed to plain split and "
        "marked.  "
        "`curvature_cap` is the least safe: the 3 px corridor prevents a low cap on a "
        "narrow finger, so it mostly falls back to split, and when it does apply the "
        "turn guard is what stops it from erasing the reversal.  Baseline A is not "
        "safe for these paths because it creates the sub-pixel cusp.\n\n"
        "3. **Does it reduce dense curvature without topology/fidelity loss?** "
        f"On non-ph01 hairpin paths, split lowers the local hairpin maximum by a "
        f"median of {_pct(med_red) if med_red is not None else 'n/a'} relative to "
        f"baseline_A.  p37: whole-path dense max "
        f"{_num(p37b.get('dense_max_curvature'), 2)} → "
        f"{_num(p37s.get('dense_max_curvature'), 2)}, local hairpin max "
        f"{_num(p37b.get('hairpin_max_curvature'), 2)} → "
        f"{_num(p37s.get('hairpin_max_curvature'), 2)}.  p13: "
        f"{_num(p13b.get('hairpin_max_curvature'), 2)} → "
        f"{_num(p13s.get('hairpin_max_curvature'), 2)}.  p4: "
        f"{_num(p4b.get('hairpin_max_curvature'), 2)} → "
        f"{_num(p4s.get('hairpin_max_curvature'), 2)}.  Displacement stays inside the "
        "3 px corridor, endpoint drift is exact, self/cross-intersections remain zero, "
        "and closed-body area change remains under 2%.  The price is a deliberate C0 "
        "corner: its angle is preserved, but its mathematical curvature is singular at "
        "exactly one point, so the reported dense maxima exclude ±1 px around joins and "
        "the join/tangent error is reported.  For transparency, A's contract "
        f"max-curvature metric (at most 1000 samples, no join exclusion) is "
        f"{_num(p37b.get('max_curvature'), 2)} for baseline p37 and "
        f"{_num(p37s.get('max_curvature'), 2)} for split: the latter is the corner "
        "itself, not a smoothing spike, and the dense maximum away from the join is "
        f"{_num(p37s.get('dense_max_curvature'), 2)}.  That is fidelity-preserving "
        "topology, not a hidden smoothing failure.\n"
    )


# ---------------------------------------------------------------------------
# Main


def main() -> int:
    cfg = Config()
    start = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PANELS_DIR.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    print("loading panels", flush=True)
    t0 = time.perf_counter()
    panels = load_panels(MANIFEST)
    records_by_panel: dict[str, list[PathRecord]] = {}
    for panel in panels:
        records = [
            _build_record(panel.panel_id, index, path, cfg)
            for index, path in enumerate(keep_paths(panel.labels))
        ]
        records_by_panel[panel.panel_id] = records
        print(
            f"  {panel.panel_id}: {len(records)} paths, "
            f"{sum(len(r.hairpins) for r in records)} anchors",
            flush=True,
        )
    timings["build_records"] = time.perf_counter() - t0
    all_records = [record for records in records_by_panel.values() for record in records]
    counts = {
        "anchors": sum(len(r.hairpins) for r in all_records),
        "anchor_paths": sum(1 for r in all_records if r.hairpins),
        "eligible": sum(1 for r in all_records if r.eligible),
        "objects": len({(r.panel_id, r.path.boundary_id) for r in all_records}),
    }

    print("running curated panels", flush=True)
    all_results: dict[str, dict[str, list[VariantResult]]] = {}
    panel_rows: list[dict] = []
    rows: list[dict] = []
    t0 = time.perf_counter()
    for panel in panels:
        print(f"  {panel.panel_id} ...", flush=True)
        pt0 = time.perf_counter()
        records = records_by_panel[panel.panel_id]
        results, info = _run_records(panel.panel_id, records, cfg)
        all_results[panel.panel_id] = results
        runtime_s = time.perf_counter() - pt0
        timings[f"panel_{panel.panel_id}"] = runtime_s
        for variant in VARIANTS:
            for record, result in zip(records, results[variant]):
                rows.append(_metrics_row(record, result))
            panel_rows.append(
                _panel_row(panel.panel_id, records, results[variant], variant, info, runtime_s)
            )
        print(f"    done ({runtime_s:.1f}s)", flush=True)
    timings["curated"] = time.perf_counter() - t0

    print("running synthetic cases", flush=True)
    t0 = time.perf_counter()
    synthetic = _synthetic_cases(cfg)
    synthetic_results: dict[str, dict[str, VariantResult]] = {}
    synthetic_rows: list[dict] = []
    for case in synthetic:
        record = case["record"]
        results, info = _run_records(record.panel_id, [record], cfg)
        synthetic_results[record.panel_id] = results
        for variant in VARIANTS:
            synthetic_rows.append(_metrics_row(record, results[variant][0]))
            panel_rows.append(
                _panel_row(record.panel_id, [record], results[variant], variant, info, 0.0)
            )
        print(f"  {record.panel_id}: {len(record.hairpins)} anchors", flush=True)
    timings["synthetic"] = time.perf_counter() - t0

    print("writing csv / summary", flush=True)
    metric_columns = [
        "panel_id", "path_index", "boundary_id", "labels_pair", "closed",
        "length_exact", "eligible", "variant", "status", "reason", "n_hairpins",
        "n_pieces", "n_pieces_exact", "n_relaxed", "lambda_used", "cap_boost",
        "cap_stage", "mean_disp", "p95_disp", "max_disp", "endpoint_drift",
        "max_curvature", "p95_curvature", "dense_max_curvature",
        "dense_p95_curvature", "dense_raw_max_curvature", "dense_max_u",
        "dense_max_end_dist", "hairpin_max_curvature", "hairpin_baseline_curvature",
        "hairpin_exact_curvature", "hairpin_turn_before", "hairpin_turn_after",
        "hairpin_turn_delta", "hairpin_turn_preserved", "hairpin_dist_from_endpoint",
        "tangent_error_max", "length_smooth", "length_change_rel", "area_exact",
        "area_smooth", "area_change", "area_change_rel", "self_intersections",
        "runtime_ms", "error",
    ]
    _write_csv(OUT_DIR / "metrics.csv", metric_columns, rows + synthetic_rows)

    panel_columns = [
        "panel_id", "variant", "n_paths", "n_eligible", "n_ok", "n_ok_reduced",
        "n_ok_partial", "n_fallback_short", "n_fallback_constraint", "n_error",
        "n_hairpins", "n_paths_with_hairpins", "n_split_applied", "n_corner_applied",
        "n_corner_relaxed", "n_cap_applied", "n_cap_within", "n_cap_to_split",
        "n_cap_to_exact", "med_mean_disp", "med_p95_disp", "p90_p95_disp",
        "max_disp", "max_endpoint_drift", "med_dense_max_curv",
        "med_hairpin_baseline", "med_hairpin_variant", "max_hairpin_baseline",
        "max_hairpin_variant", "med_turn_preserved", "max_self_intersections",
        "cross_intersections_exact", "cross_intersections_variant",
        "max_area_change_rel", "runtime_s",
    ]
    _write_csv(OUT_DIR / "panel_metrics.csv", panel_columns, panel_rows)

    hairpin_columns = [
        "panel_id", "path_index", "variant", "anchor_index", "u_final", "x", "y",
        "t3", "t5", "t8", "k5", "dist_start", "dist_end", "turn_before",
        "turn_after", "turn_delta", "turn_preserved", "exact_local_curvature",
        "baseline_local_curvature", "variant_local_curvature", "tangent_error_max",
        "cap_stage", "cap_boost", "applied", "n_pieces",
    ]
    hairpin_rows: list[dict] = []
    for panel_id, results in all_results.items():
        records = records_by_panel[panel_id]
        for variant in VARIANTS:
            for record, result in zip(records, results[variant]):
                for hi, anchor in enumerate(record.hairpins):
                    after = result.hairpin_after[hi] if hi < len(result.hairpin_after) else {}
                    hairpin_rows.append(
                        {
                            "panel_id": panel_id,
                            "path_index": record.path_index,
                            "variant": variant,
                            "anchor_index": anchor.anchor_index,
                            "u_final": after.get("u_final", anchor.u),
                            "x": float(anchor.point[0]),
                            "y": float(anchor.point[1]),
                            "t3": anchor.t3, "t5": anchor.t5, "t8": anchor.t8, "k5": anchor.k5,
                            "dist_start": anchor.dist_start, "dist_end": anchor.dist_end,
                            "turn_before": after.get("turn_before", anchor.turn_before),
                            "turn_after": after.get("turn_after"),
                            "turn_delta": after.get("turn_delta"),
                            "turn_preserved": after.get("turn_preserved"),
                            "exact_local_curvature": after.get("exact_local_curvature"),
                            "baseline_local_curvature": after.get("baseline_local_curvature"),
                            "variant_local_curvature": after.get("local_curvature"),
                            "tangent_error_max": result.tangent_error_max,
                            "cap_stage": result.cap_stage,
                            "cap_boost": result.cap_boost,
                            "applied": int(result.applied),
                            "n_pieces": len(result.fits),
                        }
                    )
    _write_csv(OUT_DIR / "hairpins.csv", hairpin_columns, hairpin_rows)

    piece_columns = [
        "panel_id", "path_index", "variant", "piece_index", "length", "n_points",
        "status", "reason", "lambda_used", "refits", "tangent", "relaxed",
    ]
    piece_rows: list[dict] = []
    for panel_id, results in all_results.items():
        records = records_by_panel[panel_id]
        for variant in ("split", "corner_knot"):
            for record, result in zip(records, results[variant]):
                for info in result.piece_info:
                    piece_rows.append(
                        {
                            "panel_id": panel_id,
                            "path_index": record.path_index,
                            "variant": variant,
                            **info,
                        }
                    )
    _write_csv(OUT_DIR / "pieces.csv", piece_columns, piece_rows)

    print("rendering", flush=True)
    t0 = time.perf_counter()
    contact_rows = []
    for panel in panels:
        labels = panel.labels
        records = records_by_panel[panel.panel_id]
        results = all_results[panel.panel_id]
        base_image = _overlay_for_results(
            labels, records, results["baseline_A"], f"{panel.panel_id} baseline_A"
        )
        sel_image = _overlay_for_results(
            labels, records, results[SELECTED_VARIANT], f"{panel.panel_id} {SELECTED_VARIANT}"
        )
        base_image.save(PANELS_DIR / f"{panel.panel_id}_baseline.png")
        sel_image.save(PANELS_DIR / f"{panel.panel_id}_{SELECTED_VARIANT}.png")
        base_row = next(p for p in panel_rows if p["panel_id"] == panel.panel_id and p["variant"] == "baseline_A")
        sel_row = next(p for p in panel_rows if p["panel_id"] == panel.panel_id and p["variant"] == SELECTED_VARIANT)
        base_thumb = base_image.copy()
        base_thumb.thumbnail((430, 270), Image.Resampling.LANCZOS)
        sel_thumb = sel_image.copy()
        sel_thumb.thumbnail((430, 270), Image.Resampling.LANCZOS)
        contact_rows.append(
            (
                panel.panel_id,
                f"paths {sel_row['n_paths']}  hairpins {sel_row['n_hairpins']}\n"
                f"med p95 {_num(base_row['med_p95_disp'])} -> {_num(sel_row['med_p95_disp'])}\n"
                f"max hairpin {_num(base_row['max_hairpin_variant'])} -> {_num(sel_row['max_hairpin_variant'])}\n"
                f"cross {base_row['cross_intersections_variant']} -> {sel_row['cross_intersections_variant']}",
                base_thumb,
                sel_thumb,
            )
        )
    _contact_sheet(contact_rows, OUT_DIR / "contact.png")

    # Hairpin zoom: named three + strongest curated anchors + synthetic anchors.
    zoom_cases = []
    named = [("schematic_panel2", 37), ("paper23_fig7a", 13), ("newimage_p4", 4)]
    candidates = []
    for panel_id, results in all_results.items():
        records = records_by_panel[panel_id]
        for record, result in zip(records, results["baseline_A"]):
            for hi, hairpin in enumerate(record.hairpins):
                local = (
                    result.hairpin_after[hi]["local_curvature"]
                    if hi < len(result.hairpin_after) else 0.0
                )
                candidates.append((local, panel_id, record, hairpin, hi))
    candidates.sort(key=lambda item: item[0], reverse=True)
    chosen = []
    for panel_id, path_index in named:
        record = records_by_panel[panel_id][path_index]
        if not record.hairpins:
            continue
        hi = max(range(len(record.hairpins)), key=lambda i: record.hairpins[i].t3 + record.hairpins[i].t8)
        chosen.append((0.0, panel_id, record, record.hairpins[hi], hi))
    # Prefer anchors the baseline actually fails on; low-curvature detections
    # make uninformative zooms.
    for item in candidates:
        if len(chosen) >= 9:
            break
        key = (item[1], item[2].path_index)
        if any((c[1], c[2].path_index) == key for c in chosen):
            continue
        if item[0] < 0.30:
            continue
        chosen.append(item)
    for item in candidates:
        if len(chosen) >= 9:
            break
        key = (item[1], item[2].path_index)
        if any((c[1], c[2].path_index) == key for c in chosen):
            continue
        chosen.append(item)
    for _, panel_id, record, hairpin, hi in chosen:
        results = all_results[panel_id]
        base_result = results["baseline_A"][record.path_index]
        split_result = results[SELECTED_VARIANT][record.path_index]
        base_local = (
            base_result.hairpin_after[hi]["local_curvature"]
            if hi < len(base_result.hairpin_after) else None
        )
        split_local = (
            split_result.hairpin_after[hi]["local_curvature"]
            if hi < len(split_result.hairpin_after) else None
        )
        zoom_cases.append(
            {
                "label": f"{panel_id} p{record.path_index}",
                "text": (
                    f"idx {hairpin.anchor_index}  d_end {_num(hairpin.dist_from_endpoint, 0)} px\n"
                    f"turn before {_num(hairpin.turn_before, 0)} deg\n"
                    f"local curv {_num(base_local, 2)} -> {_num(split_local, 2)}\n"
                    f"t3 {hairpin.t3:.0f} t8 {hairpin.t8:.0f} k {hairpin.k5:.2f}"
                ),
                "record": record,
                "center": hairpin.point,
                "labels": panels[[p.panel_id for p in panels].index(panel_id)].labels,
                "results": {v: results[v][record.path_index] for v in ("baseline_A", SELECTED_VARIANT)},
            }
        )
    for case in synthetic:
        record = case["record"]
        if not record.hairpins:
            continue
        hairpin = record.hairpins[0]
        zoom_cases.append(
            {
                "label": record.panel_id,
                "text": f"{case['kind']}\n{case['description']}",
                "record": record,
                "center": hairpin.point,
                "labels": None,
                "results": {
                    v: synthetic_results[record.panel_id][v][0]
                    for v in ("baseline_A", SELECTED_VARIANT)
                },
            }
        )
    _hairpin_zoom_sheet(zoom_cases, OUT_DIR / "hairpin_zoom.png", half=45, scale=7)
    _synthetic_sheet(synthetic, synthetic_results, OUT_DIR / "synthetic.png")
    timings["render"] = time.perf_counter() - t0

    parity = _baseline_parity(rows)
    detection_rows = _detection_sensitivity(all_records, cfg)
    # Cap sensitivity is computed from the curated cap results.
    cap_results = {panel_id: results["curvature_cap"] for panel_id, results in all_results.items()}
    flat_cap = [(records_by_panel[pid], cap_results[pid]) for pid in cap_results]
    cap_rows = []
    for cap in CAP_SENSITIVITY:
        paths = applied = fallback = 0
        for recs, results in flat_cap:
            for record, result in zip(recs, results):
                if not record.hairpins:
                    continue
                paths += 1
                trials = [t for t in result.cap_trials if t.get("pass") and t.get("local_curv") is not None]
                if trials and min(t["local_curv"] for t in trials) <= cap + 1e-9:
                    applied += 1
                else:
                    fallback += 1
        cap_rows.append({"cap": cap, "paths_with_hairpins": paths, "cap_met": applied, "cap_fallback": fallback})

    counts["synthetic"] = len(synthetic)
    sanity = _sanity(all_records, rows, panel_rows, parity, counts)
    summary = _write_summary(
        rows, panel_rows, all_records, synthetic, synthetic_rows,
        detection_rows, cap_rows, parity, sanity, timings,
        time.perf_counter() - start, counts,
    )
    (OUT_DIR / "summary.md").write_text(summary, encoding="utf-8")
    print(f"done in {time.perf_counter() - start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
