#!/usr/bin/env python
"""Layer 2 experiment B: smoothing-kernel benchmark.

Compares five arc-length smoothers on the exact same Layer 1 keep paths:

  * ``splprep``       per-path smoothing spline (``splprep``), strong endpoint
                      weights, endpoints reset afterwards.
  * ``bspline_lsq``   cubic B-spline least squares on a fixed coarse knot grid
                      with an explicit integrated-curvature penalty.
  * ``pspline_diff2`` cubic B-spline (P-spline) on a dense fixed knot grid with
                      a second-difference penalty on control coefficients.
  * ``savgol``        Savitzky-Golay along arc length, fixed window/order.
  * ``knot_quintic``  sliding quintic local regression along arc length
                      (arc-length equivalent of a quintic y(x) smoother; a
                      genuine y(x) method cannot represent closed loops or
                      multi-valued curves).

All variants receive the same arc-length resampling, the same closed/open
handling, and the same post-fit endpoint reset.  Parameters are selected from
one global grid over all non-ph01 panels, then frozen for every panel.

Run from the repository root:

    uv run python experiments/layer2/b_kernel_benchmark/run.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import BSpline, splprep, splev
from scipy.linalg import solve
from scipy.signal import savgol_filter

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.layer2.common import (  # noqa: E402
    BoundaryPath,
    boundary_metrics,
    closed_area,
    cross_intersections,
    curvature,
    displacement,
    keep_paths,
    load_panels,
    render_overlay,
    resample,
    self_intersections,
)

# ---------------------------------------------------------------------------
# Global, panel-independent constants
# ---------------------------------------------------------------------------

MIN_SMOOTH_LENGTH_PX = 8.0
MIN_SAMPLES = 8
MAX_SAMPLES = 1200

SPLPREP_ENDPOINT_WEIGHT = 100.0
BSPLINE_LSQ_KNOT_SPACING = 24.0
PSPLINE_KNOT_SPACING = 4.0
SAVGOL_POLYORDER = 3
QUINTIC_DEGREE = 5

# Global parameter grids.  A single value per kernel is selected on all
# non-ph01 panels and then frozen for the final run.
PARAM_GRIDS: dict[str, tuple[float, ...]] = {
    "splprep": (0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2),
    "bspline_lsq": (1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0),
    "pspline_diff2": (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0),
    "savgol": (9, 13, 17, 21, 29, 41, 61, 81, 101),
    "knot_quintic": (6, 8, 11, 15, 22, 32, 45, 64, 90),
}

VARIANT_ORDER = ("splprep", "bspline_lsq", "pspline_diff2", "savgol", "knot_quintic")
RENDER_ORDER = ("exact",) + VARIANT_ORDER
VARIANT_DESCRIPTIONS = {
    "exact": "exact skeleton (baseline)",
    "splprep": "per-path smoothing spline, pinned ends",
    "bspline_lsq": "cubic B-spline LSQ, coarse knots + curvature penalty",
    "pspline_diff2": "P-spline, dense knots + second-difference penalty",
    "savgol": "Savitzky-Golay along arc length",
    "knot_quintic": "sliding quintic local regression along arc length",
}

SPECIAL_PANEL = "ph01_page8_panel3"

ADVERSE_FLAGS = {
    "area_change_gt_5pct",
    "length_change_gt_8pct",
    "new_self_intersections",
    "displacement_gt_5px",
    "curvature_spike",
    "endpoint_moved",
    "degenerate_collapse",
}


def _adverse_flags(flag_string: str) -> list[str]:
    return [flag for flag in flag_string.split(";") if flag in ADVERSE_FLAGS]

# Global selection corridor.  Selection is done once on the pooled non-ph01
# keep paths; no panel ever gets its own parameter.  Length change is reported
# as an outcome, not used to select: short paths make a percentage corridor
# unfairly sensitive to sub-pixel changes.
SELECT_MEAN_P95_DISP = 1.5
SELECT_P95_MAX_DISP = 4.5
SELECT_MAX_FALLBACK_FRACTION = 0.02


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


@dataclass
class PreparedPath:
    panel_id: str
    boundary_id: int
    path_index: int
    labels_pair: tuple[int, int]
    closed: bool
    raw: np.ndarray
    exact_resampled: np.ndarray
    s: np.ndarray
    length: float
    smoothable: bool
    exact_self_intersections: int
    key: str


@dataclass
class FittedPath:
    value: np.ndarray
    runtime_ms: float
    status: str
    flags: list[str]


def _polyline_length(points: np.ndarray, closed: bool) -> float:
    dense = np.vstack([points, points[:1]]) if closed else points
    if len(dense) < 2:
        return 0.0
    return float(np.hypot(*np.diff(dense, axis=0).T).sum())


def _resample_count(length: float) -> int:
    return int(np.clip(int(round(length)), MIN_SAMPLES, MAX_SAMPLES))


def prepare_paths(panel) -> list[PreparedPath]:
    raw_paths = keep_paths(panel.labels)
    prepared: list[PreparedPath] = []
    for index, path in enumerate(raw_paths):
        length = _polyline_length(path.points, path.closed)
        n = _resample_count(length)
        exact = resample(path.points, path.closed, n)
        if path.closed:
            s = np.linspace(0.0, length, len(exact), endpoint=False)
        else:
            s = np.linspace(0.0, length, len(exact))
        key = f"{panel.panel_id}:b{path.boundary_id}:p{index}"
        prepared.append(
            PreparedPath(
                panel_id=panel.panel_id,
                boundary_id=path.boundary_id,
                path_index=index,
                labels_pair=path.labels_pair,
                closed=path.closed,
                raw=np.asarray(path.points, dtype=float),
                exact_resampled=np.asarray(exact, dtype=float),
                s=s,
                length=length,
                smoothable=length >= MIN_SMOOTH_LENGTH_PX,
                exact_self_intersections=self_intersections(
                    np.asarray(exact, dtype=float), path.closed
                ),
                key=key,
            )
        )
    return prepared


# ---------------------------------------------------------------------------
# B-spline machinery
# ---------------------------------------------------------------------------


def _open_knots(n_ctrl: int, degree: int = 3) -> np.ndarray:
    interior = n_ctrl - degree - 1
    if interior > 0:
        inner = np.linspace(0.0, 1.0, interior + 2)[1:-1]
        return np.concatenate(
            [np.zeros(degree + 1), inner, np.ones(degree + 1)]
        )
    return np.concatenate([np.zeros(degree + 1), np.ones(degree + 1)])


def _open_design(t: np.ndarray, n_ctrl: int, degree: int = 3, second: bool = False):
    knots = _open_knots(n_ctrl, degree)
    basis = BSpline(knots, np.eye(n_ctrl), degree, extrapolate=True)
    matrix = np.asarray(basis(t), dtype=float)
    if not second:
        return matrix, None
    return matrix, np.asarray(basis.derivative(2)(t), dtype=float)


def _cubic_cardinal(z: np.ndarray) -> np.ndarray:
    a = np.abs(z)
    return np.where(
        a <= 1.0,
        2.0 / 3.0 - a * a + 0.5 * a**3,
        np.where(a < 2.0, (2.0 - a) ** 3 / 6.0, 0.0),
    )


def _cubic_cardinal_second(z: np.ndarray) -> np.ndarray:
    a = np.abs(z)
    return np.where(
        a <= 1.0,
        -2.0 + 3.0 * a,
        np.where(a < 2.0, 2.0 - a, 0.0),
    )


def _periodic_design(t: np.ndarray, n_ctrl: int):
    """Periodic cubic cardinal B-spline basis and its second derivative in t."""
    z = t[:, None] * n_ctrl - np.arange(n_ctrl)[None, :]
    z = np.mod(z + n_ctrl / 2.0, n_ctrl) - n_ctrl / 2.0
    return _cubic_cardinal(z), _cubic_cardinal_second(z)


def _second_difference(n_ctrl: int, periodic: bool) -> np.ndarray:
    if periodic:
        d = np.zeros((n_ctrl, n_ctrl), dtype=float)
        for i in range(n_ctrl):
            d[i, i] += 1.0
            d[i, (i + 1) % n_ctrl] += -2.0
            d[i, (i + 2) % n_ctrl] += 1.0
        return d
    return np.diff(np.eye(n_ctrl), n=2, axis=0)


def _solve_penalized(a: np.ndarray, penalty: np.ndarray, rhs: np.ndarray, lam: float):
    normal = a.T @ a
    scale = max(1.0, float(np.mean(np.diag(normal))))
    system = normal + lam * penalty + 1e-9 * scale * np.eye(normal.shape[0])
    return solve(system, rhs, assume_a="sym")


# ---------------------------------------------------------------------------
# Smoothing kernels
# ---------------------------------------------------------------------------


def fit_splprep(path: PreparedPath, param: float) -> tuple[np.ndarray, dict]:
    points = path.exact_resampled
    n = len(points)
    if path.closed:
        # splprep's periodic convention makes the first and last data
        # parameters coincide.  Supply a duplicated seam point with u in
        # [0, 1], then evaluate on n unique samples of the period.
        fit_points = np.vstack([points, points[:1]])
        u_fit = np.linspace(0.0, 1.0, n + 1)
        u_eval = np.arange(n, dtype=float) / n
        weights = np.ones(n + 1)
        per = 1
    else:
        fit_points = points
        u_fit = np.linspace(0.0, 1.0, n)
        u_eval = u_fit
        weights = np.ones(n)
        weights[0] = SPLPREP_ENDPOINT_WEIGHT
        weights[-1] = SPLPREP_ENDPOINT_WEIGHT
        per = 0
    tck, _ = splprep(
        [fit_points[:, 0], fit_points[:, 1]],
        u=u_fit,
        w=weights,
        s=param * len(fit_points),
        per=per,
        k=3,
    )
    out = np.column_stack(splev(u_eval, tck))
    return np.asarray(out, dtype=float), {}


def fit_bspline_lsq(path: PreparedPath, param: float) -> tuple[np.ndarray, dict]:
    """Coarse fixed-knot cubic B-spline with integrated-curvature penalty."""
    points = path.exact_resampled
    n = len(points)
    length = max(path.length, 1e-9)
    if path.closed:
        n_ctrl = max(8, int(round(length / BSPLINE_LSQ_KNOT_SPACING)))
        n_ctrl = min(n_ctrl, n)
        t = np.arange(n, dtype=float) / n
        a, a2_z = _periodic_design(t, n_ctrl)
        # d2/dt2 = n_ctrl**2 * d2/dz2, and ds = L dt.
        penalty = (n_ctrl**4) / (n * length**3) * (a2_z.T @ a2_z)
    else:
        n_ctrl = max(4, int(round(length / BSPLINE_LSQ_KNOT_SPACING)) + 3)
        n_ctrl = min(n_ctrl, n)
        t = np.linspace(0.0, 1.0, n)
        a, a2 = _open_design(t, n_ctrl, 3, second=True)
        # Integral of |C_ss|^2 ds evaluated by the sample mean of |C_tt|^2.
        penalty = (a2.T @ a2) / (n * length**3)
    coefficients = _solve_penalized(a, penalty, a.T @ points, param)
    return np.asarray(a @ coefficients, dtype=float), {}


def fit_pspline(path: PreparedPath, param: float) -> tuple[np.ndarray, dict]:
    """Dense B-spline basis with second-difference control-point penalty."""
    points = path.exact_resampled
    n = len(points)
    length = max(path.length, 1e-9)
    if path.closed:
        n_ctrl = max(8, int(round(length / PSPLINE_KNOT_SPACING)))
        n_ctrl = min(n_ctrl, n)
        t = np.arange(n, dtype=float) / n
        a, _ = _periodic_design(t, n_ctrl)
        d = _second_difference(n_ctrl, periodic=True)
    else:
        n_ctrl = max(4, int(round(length / PSPLINE_KNOT_SPACING)) + 3)
        n_ctrl = min(n_ctrl, n)
        t = np.linspace(0.0, 1.0, n)
        a, _ = _open_design(t, n_ctrl, 3)
        d = _second_difference(n_ctrl, periodic=False)
    penalty = d.T @ d
    coefficients = _solve_penalized(a, penalty, a.T @ points, param)
    return np.asarray(a @ coefficients, dtype=float), {}


def fit_savgol(path: PreparedPath, param: float) -> tuple[np.ndarray, dict]:
    points = path.exact_resampled
    n = len(points)
    window = int(round(param))
    if window % 2 == 0:
        window += 1
    info: dict = {}
    if path.closed:
        half = window // 2
        indices = np.arange(-half, n + half) % n
        extended = points[indices]
        smoothed = np.column_stack(
            [
                savgol_filter(extended[:, 0], window, SAVGOL_POLYORDER, mode="interp"),
                savgol_filter(extended[:, 1], window, SAVGOL_POLYORDER, mode="interp"),
            ]
        )
        out = smoothed[half : half + n]
        info["periodic_wrap"] = True
    else:
        if n < window:
            left = (window - n) // 2 + 2
            right = max(0, window - n - left + 2)
            extended = np.pad(points, ((left, right), (0, 0)), mode="symmetric")
            info["edge_padded"] = True
        else:
            left = 0
            extended = points
        smoothed = np.column_stack(
            [
                savgol_filter(extended[:, 0], window, SAVGOL_POLYORDER, mode="interp"),
                savgol_filter(extended[:, 1], window, SAVGOL_POLYORDER, mode="interp"),
            ]
        )
        out = smoothed[left : left + n]
    return np.asarray(out, dtype=float), info


def _tricube_kernel_weights(offsets: np.ndarray, half: int) -> np.ndarray:
    distance = np.clip(np.abs(offsets) / float(half), 0.0, 1.0)
    return (1.0 - distance**3) ** 3


def _local_quintic_value(values: np.ndarray, offsets: np.ndarray, half: int) -> float:
    """Weighted local polynomial evaluated at offset 0.

    Degree drops only if a truncated edge window cannot support the quintic
    (the global grid starts at half-window 6, so this is a defensive branch).
    """
    degree = min(QUINTIC_DEGREE, max(1, len(values) - 1))
    x = offsets.astype(float) / float(half)
    v = np.vander(x, degree + 1, increasing=True)
    w = _tricube_kernel_weights(offsets, half)
    design = v.T @ (w[:, None] * v)
    rhs = v.T @ (w * values)
    return float(np.linalg.solve(design, rhs)[0])


def fit_knot_quintic(path: PreparedPath, param: float) -> tuple[np.ndarray, dict]:
    """Sliding local quintic regression on x(s), y(s).

    This is the arc-length equivalent of a quintic y(x) smoother.  A true
    y(x) implementation is not applicable to closed loops, overhangs or
    vertical segments, so both coordinates are smoothed against arc length.
    """
    points = path.exact_resampled
    n = len(points)
    half = max(3, int(round(param)))
    degree = QUINTIC_DEGREE
    offsets = np.arange(-half, half + 1)
    x = offsets / float(half)
    v = np.vander(x, degree + 1, increasing=True)
    w = _tricube_kernel_weights(offsets, half)
    design = v.T @ (w[:, None] * v)
    # First row of (V^T W V)^-1 V^T W: the intercept functional, evaluated at
    # the window centre.  This is a fixed convolution kernel in the interior.
    kernel = np.linalg.solve(design, (v.T * w[None, :]))[0]
    out = np.zeros_like(points)
    info: dict = {}
    if path.closed:
        indices = (np.arange(n)[:, None] + offsets[None, :]) % n
        out[:, 0] = points[indices, 0] @ kernel
        out[:, 1] = points[indices, 1] @ kernel
    else:
        info["edge_truncated"] = True
        for j in range(n):
            lo = max(0, j - half)
            hi = min(n, j + half + 1)
            if hi - lo >= degree + 1 and lo == j - half and hi == j + half + 1:
                window = points[lo:hi]
                out[j, 0] = float(kernel @ window[:, 0])
                out[j, 1] = float(kernel @ window[:, 1])
            else:
                local_offsets = np.arange(lo, hi) - j
                out[j, 0] = _local_quintic_value(points[lo:hi, 0], local_offsets, half)
                out[j, 1] = _local_quintic_value(points[lo:hi, 1], local_offsets, half)
    return out, info


FITTERS = {
    "splprep": fit_splprep,
    "bspline_lsq": fit_bspline_lsq,
    "pspline_diff2": fit_pspline,
    "savgol": fit_savgol,
    "knot_quintic": fit_knot_quintic,
}


def _degenerate(path: PreparedPath, value: np.ndarray) -> bool:
    """Reject fits that collapsed the geometry instead of smoothing it."""
    smooth_length = _polyline_length(value, path.closed)
    if smooth_length < 0.25 * max(path.length, 1e-9):
        return True
    if path.closed:
        exact_area = closed_area(path.exact_resampled)
        smooth_area = closed_area(value)
        if exact_area >= 2.0 and smooth_area < 0.10 * exact_area:
            return True
    return False


def fit_variant(variant: str, path: PreparedPath, param: float) -> FittedPath:
    if variant == "exact":
        return FittedPath(path.exact_resampled, 0.0, "exact", [])
    if not path.smoothable:
        return FittedPath(path.exact_resampled, 0.0, "short_passthrough", [])
    started = time.perf_counter()
    try:
        value, info = FITTERS[variant](path, param)
    except Exception as error:  # pragma: no cover - defensive fallback
        elapsed = (time.perf_counter() - started) * 1000.0
        return FittedPath(
            path.exact_resampled,
            elapsed,
            f"error_passthrough:{type(error).__name__}",
            [str(error)[:160]],
        )
    elapsed = (time.perf_counter() - started) * 1000.0
    value = np.asarray(value, dtype=float)
    if value.shape != path.exact_resampled.shape or not np.isfinite(value).all():
        return FittedPath(
            path.exact_resampled,
            elapsed,
            "error_passthrough:bad_shape_or_nonfinite",
            [],
        )
    # The same endpoint rule for every variant: open paths pin both endpoints
    # to the exact resampled endpoints.  Closed paths are periodic and have no
    # free endpoint.
    if not path.closed:
        value = value.copy()
        value[0] = path.exact_resampled[0]
        value[-1] = path.exact_resampled[-1]
    if _degenerate(path, value):
        return FittedPath(
            path.exact_resampled,
            elapsed,
            "degenerate_passthrough",
            ["degenerate_collapse"],
        )
    flags: list[str] = []
    if info.get("edge_padded"):
        flags.append("edge_padded")
    if info.get("edge_truncated"):
        flags.append("edge_truncated")
    return FittedPath(value, elapsed, "ok", flags)


# ---------------------------------------------------------------------------
# Parameter selection
# ---------------------------------------------------------------------------


def param_label(variant: str, value: float) -> str:
    if variant == "splprep":
        return f"s_per_point={value:g}"
    if variant == "bspline_lsq":
        return f"lambda={value:g};knot={BSPLINE_LSQ_KNOT_SPACING:g}px"
    if variant == "pspline_diff2":
        return f"lambda={value:g};knot={PSPLINE_KNOT_SPACING:g}px"
    if variant == "savgol":
        return f"window={int(value)};order={SAVGOL_POLYORDER}"
    if variant == "knot_quintic":
        return f"half_window={int(value)};degree={QUINTIC_DEGREE}"
    return f"param={value:g}"


def param_short(variant: str, value: float) -> str:
    if variant == "splprep":
        return f"s={value:g}"
    if variant in ("bspline_lsq", "pspline_diff2"):
        return f"lam={value:g}"
    if variant == "savgol":
        return f"w={int(value)}"
    if variant == "knot_quintic":
        return f"h={int(value)}"
    return f"{value:g}"


def _aggregate_config(rows: list[dict]) -> dict:
    measured = [
        row
        for row in rows
        if not row["status"].startswith(("degenerate", "error"))
    ]
    statuses: dict[str, int] = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    if not measured:
        return {
            "n_paths": len(rows),
            "n_measured": 0,
            "mean_p95_disp": float("inf"),
            "mean_disp": float("inf"),
            "p95_max_disp": float("inf"),
            "max_disp": float("inf"),
            "mean_p95_curvature": float("inf"),
            "mean_curvature_reduction": 0.0,
            "mean_abs_length_change_pct": float("inf"),
            "mean_length_change_pct": float("inf"),
            "n_error": 0,
            "n_degenerate": 0,
            "statuses": statuses,
        }
    p95_disp = np.array([row["p95_disp"] for row in measured], dtype=float)
    p95_curv = np.array([row["p95_curvature"] for row in measured], dtype=float)
    exact_p95_curv = np.array(
        [row["exact_p95_curvature"] for row in measured], dtype=float
    )
    valid = exact_p95_curv > 1e-3
    reduction = 1.0 - p95_curv[valid] / exact_p95_curv[valid]
    length_pct = np.array(
        [row["length_change_pct"] for row in measured], dtype=float
    )
    return {
        "n_paths": len(rows),
        "n_measured": len(measured),
        "mean_p95_disp": float(np.mean(p95_disp)),
        "mean_disp": float(np.mean([row["mean_disp"] for row in measured])),
        "p95_max_disp": float(np.percentile([row["max_disp"] for row in measured], 95)),
        "max_disp": float(np.max([row["max_disp"] for row in measured])),
        "mean_p95_curvature": float(np.mean(p95_curv)),
        "mean_curvature_reduction": float(np.mean(reduction)) if len(reduction) else 0.0,
        "mean_abs_length_change_pct": float(np.mean(np.abs(length_pct))),
        "mean_length_change_pct": float(np.mean(length_pct)),
        "n_error": int(
            sum(count for status, count in statuses.items() if status.startswith("error"))
        ),
        "n_degenerate": int(statuses.get("degenerate_passthrough", 0)),
        "statuses": statuses,
    }


def run_grid(
    panels, prepared_by_panel
) -> tuple[list[dict], dict[str, float], dict[str, dict]]:
    grid_rows: list[dict] = []
    selected: dict[str, float] = {}
    selection_notes: dict[str, dict] = {}
    pool = [
        path
        for panel in panels
        if panel.panel_id != SPECIAL_PANEL
        for path in prepared_by_panel[panel.panel_id]
        if path.smoothable
    ]
    print(f"[grid] {len(pool)} smoothable non-ph01 paths", flush=True)
    for variant in VARIANT_ORDER:
        for value in PARAM_GRIDS[variant]:
            started = time.perf_counter()
            rows: list[dict] = []
            for path in pool:
                fitted = fit_variant(variant, path, value)
                move = displacement(fitted.value, path.exact_resampled)
                smooth_curv = curvature(fitted.value, path.closed)
                exact_curv = curvature(path.exact_resampled, path.closed)
                length_change = _polyline_length(
                    fitted.value, path.closed
                ) - _polyline_length(path.exact_resampled, path.closed)
                length_pct = 100.0 * length_change / max(path.length, 1e-9)
                rows.append(
                    {
                        "mean_disp": move["mean"],
                        "p95_disp": move["p95"],
                        "max_disp": move["max"],
                        "p95_curvature": smooth_curv["p95"],
                        "exact_p95_curvature": exact_curv["p95"],
                        "length_change_pct": length_pct,
                        "status": fitted.status,
                    }
                )
            aggregate = _aggregate_config(rows)
            aggregate.update(
                {
                    "variant": variant,
                    "param": value,
                    "parameters": param_label(variant, value),
                    "grid_seconds": (time.perf_counter() - started),
                }
            )
            grid_rows.append(aggregate)
            print(
                f"[grid] {variant:14s} {param_short(variant, value):10s} "
                f"p95disp={aggregate['mean_p95_disp']:.3f} "
                f"maxp95={aggregate['p95_max_disp']:.3f} "
                f"curv={aggregate['mean_curvature_reduction']:.3f} "
                f"len%={aggregate['mean_abs_length_change_pct']:.2f} "
                f"err={aggregate['n_error']} degen={aggregate['n_degenerate']}",
                flush=True,
            )

        candidates = [row for row in grid_rows if row["variant"] == variant]
        max_fallbacks = int(np.floor(SELECT_MAX_FALLBACK_FRACTION * len(pool)))
        eligible = [
            row
            for row in candidates
            if row["n_error"] == 0
            and row["n_degenerate"] <= max_fallbacks
            and row["mean_p95_disp"] <= SELECT_MEAN_P95_DISP
            and row["p95_max_disp"] <= SELECT_P95_MAX_DISP
        ]
        if eligible:
            chosen = max(eligible, key=lambda row: row["mean_curvature_reduction"])
            rule = (
                "max curvature reduction inside displacement corridor "
                f"(<= {max_fallbacks} fallbacks allowed)"
            )
        elif any(row["n_error"] == 0 for row in candidates):
            chosen = min(
                candidates,
                key=lambda row: (
                    max(0, row["n_degenerate"] - max_fallbacks),
                    row["n_error"],
                    max(0.0, row["mean_p95_disp"] - SELECT_MEAN_P95_DISP),
                    max(0.0, row["p95_max_disp"] - SELECT_P95_MAX_DISP),
                    -row["mean_curvature_reduction"],
                ),
            )
            rule = "no config inside corridor; chose closest with fewest fallbacks"
        else:
            chosen = min(
                candidates,
                key=lambda row: (
                    row["n_error"],
                    row["n_degenerate"],
                    max(0.0, row["mean_p95_disp"] - SELECT_MEAN_P95_DISP),
                    max(0.0, row["p95_max_disp"] - SELECT_P95_MAX_DISP),
                    -row["mean_curvature_reduction"],
                ),
            )
            rule = "some configs had errors; chose the cleanest"
        selected[variant] = float(chosen["param"])
        selection_notes[variant] = {
            "rule": rule,
            "config": chosen,
            "n_eligible": len(eligible),
            "n_grid": len(candidates),
        }
        print(
            f"[select] {variant}: {param_label(variant, selected[variant])} "
            f"({rule}, {len(eligible)}/{len(candidates)} eligible)",
            flush=True,
        )
    return grid_rows, selected, selection_notes


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------


def _boundary_path(path: PreparedPath, points: np.ndarray) -> BoundaryPath:
    return BoundaryPath(
        boundary_id=path.boundary_id,
        labels_pair=path.labels_pair,
        points=np.asarray(points, dtype=float),
        closed=path.closed,
        exact_length=max(1, int(round(path.length))),
    )


def _flags_for(path: PreparedPath, fitted: FittedPath, metrics: dict, smooth_self: int):
    flags = list(fitted.flags)
    if path.closed and "area_change_pct" in metrics:
        if abs(metrics["area_change_pct"]) > 5.0:
            flags.append("area_change_gt_5pct")
    if abs(metrics["length_change_pct"]) > 8.0:
        flags.append("length_change_gt_8pct")
    if smooth_self > path.exact_self_intersections:
        flags.append("new_self_intersections")
    if metrics["max_disp"] > 5.0:
        flags.append("displacement_gt_5px")
    if not path.closed and metrics["endpoint_drift"] > 1e-6:
        flags.append("endpoint_moved")
    return sorted(set(flags))


def _metric_row(
    variant: str,
    parameters: str,
    path: PreparedPath,
    fitted: FittedPath,
    exact_area: float | None,
) -> dict:
    exact_boundary = _boundary_path(path, path.raw)
    metrics = boundary_metrics(fitted.value, exact_boundary)
    smooth_curv = curvature(fitted.value, path.closed)
    exact_curv = curvature(path.exact_resampled, path.closed)
    smooth_self = int(metrics["self_intersections"])
    length_change = float(metrics["length_smooth"] - metrics["length_exact"])
    length_change_pct = 100.0 * length_change / max(float(metrics["length_exact"]), 1e-9)
    exact_max = float(metrics["exact_max_curvature"])
    curvature_ratio = (
        float(metrics["max_curvature"]) / exact_max if exact_max > 1e-9 else float("nan")
    )
    exact_p95_common = float(exact_curv["p95"])
    curvature_reduction = (
        1.0 - float(smooth_curv["p95"]) / exact_p95_common
        if exact_p95_common > 1e-9
        else float("nan")
    )
    area_change_pct = (
        100.0 * float(metrics["area_change"]) / max(exact_area, 1e-9)
        if path.closed and exact_area is not None
        else ""
    )
    flags = _flags_for(
        path,
        fitted,
        {
            "length_change_pct": length_change_pct,
            "area_change_pct": area_change_pct,
            "max_disp": metrics["max_disp"],
            "endpoint_drift": metrics["endpoint_drift"],
        },
        smooth_self,
    )
    if (
        variant != "exact"
        and exact_curv["max"] > 1e-6
        and smooth_curv["max"] > 50.0 * exact_curv["max"]
        and smooth_curv["max"] > 5.0
    ):
        flags.append("curvature_spike")
    row = {
        "variant": variant,
        "parameters": parameters,
        "panel_id": path.panel_id,
        "path_id": path.key,
        "boundary_id": path.boundary_id,
        "path_index": path.path_index,
        "labels_pair": f"{path.labels_pair[0]}-{path.labels_pair[1]}",
        "closed": int(path.closed),
        "n_points": len(fitted.value),
        "exact_length_px": round(path.length, 4),
        "mean_disp": metrics["mean_disp"],
        "p95_disp": metrics["p95_disp"],
        "max_disp": metrics["max_disp"],
        "max_curvature": metrics["max_curvature"],
        "p95_curvature": metrics["p95_curvature"],
        "exact_max_curvature": metrics["exact_max_curvature"],
        "curvature_ratio": curvature_ratio,
        "max_curvature_common": smooth_curv["max"],
        "p95_curvature_common": smooth_curv["p95"],
        "exact_p95_curvature_common": exact_p95_common,
        "exact_max_curvature_common": exact_curv["max"],
        "curvature_reduction_p95": curvature_reduction,
        "length_exact": metrics["length_exact"],
        "length_smooth": metrics["length_smooth"],
        "length_change": length_change,
        "length_change_pct": length_change_pct,
        "endpoint_drift": metrics["endpoint_drift"],
        "self_intersections": smooth_self,
        "self_intersections_exact": path.exact_self_intersections,
        "area_exact": metrics.get("area_exact", ""),
        "area_smooth": metrics.get("area_smooth", ""),
        "area_change": metrics.get("area_change", ""),
        "area_change_pct": area_change_pct,
        "status": fitted.status,
        "flags": ";".join(sorted(set(flags))),
        "runtime_ms": fitted.runtime_ms,
    }
    return row


def _cross_count(paths: list[PreparedPath], points_by_key: dict[str, np.ndarray]) -> int:
    boundaries = [_boundary_path(path, points_by_key[path.key]) for path in paths]
    return int(cross_intersections(boundaries))


def run_full(panels, prepared_by_panel, selected, out_dir):
    variants = ("exact",) + VARIANT_ORDER
    rows: list[dict] = []
    panel_rows: list[dict] = []
    exact_cross_by_panel: dict[str, int] = {}
    outputs: dict[str, dict[str, np.ndarray]] = {variant: {} for variant in variants}

    for panel in panels:
        paths = prepared_by_panel[panel.panel_id]
        print(f"[run] {panel.panel_id}: {len(paths)} paths", flush=True)
        exact_points = {path.key: path.exact_resampled for path in paths}
        exact_cross = _cross_count(paths, exact_points)
        exact_cross_by_panel[panel.panel_id] = exact_cross
        exact_area_by_key: dict[str, float | None] = {}
        for path in paths:
            exact_area_by_key[path.key] = (
                closed_area(path.exact_resampled) if path.closed else None
            )

        for variant in variants:
            parameters = "identity" if variant == "exact" else param_label(variant, selected[variant])
            variant_rows: list[dict] = []
            runtime_s = 0.0
            for path in paths:
                if variant == "exact":
                    fitted = FittedPath(path.exact_resampled, 0.0, "exact", [])
                else:
                    fitted = fit_variant(variant, path, selected[variant])
                outputs[variant][path.key] = fitted.value
                runtime_s += fitted.runtime_ms / 1000.0
                row = _metric_row(
                    variant,
                    parameters,
                    path,
                    fitted,
                    exact_area_by_key[path.key],
                )
                rows.append(row)
                variant_rows.append(row)
            smooth_cross = _cross_count(paths, {path.key: outputs[variant][path.key] for path in paths})
            panel_row = _panel_row(
                panel.panel_id,
                variant,
                parameters,
                variant_rows,
                exact_cross,
                smooth_cross,
                runtime_s,
            )
            panel_rows.append(panel_row)
            print(
                f"[run]   {variant:14s} mean_p95={panel_row['mean_p95_disp']:.3f} "
                f"cross {exact_cross}->{smooth_cross} "
                f"flags={panel_row['flag_count']}",
                flush=True,
            )
    return rows, panel_rows, outputs, exact_cross_by_panel


def _panel_row(
    panel_id: str,
    variant: str,
    parameters: str,
    rows: list[dict],
    exact_cross: int,
    smooth_cross: int,
    runtime_s: float,
) -> dict:
    measured = [
        row
        for row in rows
        if not row["status"].startswith(("degenerate", "error"))
    ]
    if not measured:
        measured = [row for row in rows if row["status"] == "short_passthrough"]
    p95 = np.array([row["p95_disp"] for row in measured], dtype=float)
    exact_p95_curv = np.array(
        [row["exact_p95_curvature_common"] for row in measured], dtype=float
    )
    smooth_p95_curv = np.array(
        [row["p95_curvature_common"] for row in measured], dtype=float
    )
    valid = exact_p95_curv > 1e-3
    reductions = 1.0 - smooth_p95_curv[valid] / exact_p95_curv[valid]
    closed_rows = [
        row for row in measured if row["closed"] and row["area_change_pct"] != ""
    ]
    significant_area = [
        row for row in closed_rows if float(row["area_exact"]) >= 25.0
    ]
    area_pcts = [abs(float(row["area_change_pct"])) for row in significant_area]
    flags = [
        flag for row in rows for flag in _adverse_flags(row["flags"])
    ]
    return {
        "panel_id": panel_id,
        "variant": variant,
        "parameters": parameters,
        "n_paths": len(rows),
        "n_open": sum(1 for row in rows if not row["closed"]),
        "n_closed": sum(1 for row in rows if row["closed"]),
        "n_short_passthrough": sum(1 for row in rows if row["status"] == "short_passthrough"),
        "n_degenerate": sum(1 for row in rows if row["status"] == "degenerate_passthrough"),
        "n_error": sum(1 for row in rows if row["status"].startswith("error")),
        "mean_disp": float(np.mean([row["mean_disp"] for row in measured])),
        "mean_p95_disp": float(np.mean(p95)),
        "p95_disp": float(np.percentile(p95, 95)),
        "max_disp": float(np.max([row["max_disp"] for row in measured])),
        "mean_curvature_reduction": float(np.mean(reductions)) if len(reductions) else 0.0,
        "mean_abs_length_change_pct": float(
            np.mean([abs(row["length_change_pct"]) for row in measured])
        ),
        "max_endpoint_drift": float(np.max([row["endpoint_drift"] for row in measured])),
        "self_intersections_total": int(
            sum(row["self_intersections"] for row in measured)
        ),
        "self_intersections_exact_total": int(
            sum(row["self_intersections_exact"] for row in measured)
        ),
        "area_mean_abs_pct": float(np.mean(area_pcts)) if area_pcts else 0.0,
        "area_max_abs_pct": float(np.max(area_pcts)) if area_pcts else 0.0,
        "area_signed_mean_pct": (
            float(np.mean([float(row["area_change_pct"]) for row in significant_area]))
            if significant_area
            else 0.0
        ),
        "n_area_measured": len(significant_area),
        "cross_intersections_exact": exact_cross,
        "cross_intersections_smooth": smooth_cross,
        "cross_intersection_delta": smooth_cross - exact_cross,
        "flag_count": len(flags),
        "flags": ";".join(sorted(set(flags))),
        "runtime_s": runtime_s,
    }


# ---------------------------------------------------------------------------
# Aggregates / rankings
# ---------------------------------------------------------------------------


def aggregate_variants(
    rows: list[dict],
    panel_rows: list[dict],
    *,
    exclude_special: bool = False,
    only_ok: bool = False,
) -> dict[str, dict]:
    aggregate: dict[str, dict] = {}
    exact_rows = [row for row in rows if row["variant"] == "exact"]
    for variant in VARIANT_ORDER:
        variant_rows = [row for row in rows if row["variant"] == variant]
        variant_panels = [row for row in panel_rows if row["variant"] == variant]
        if exclude_special:
            variant_rows = [
                row for row in variant_rows if row["panel_id"] != SPECIAL_PANEL
            ]
            variant_panels = [
                row for row in variant_panels if row["panel_id"] != SPECIAL_PANEL
            ]
        if only_ok:
            variant_rows = [row for row in variant_rows if row["status"] == "ok"]
        measured = [
            row
            for row in variant_rows
            if not row["status"].startswith(("degenerate", "error"))
        ]
        measured_keys = {row["path_id"] for row in measured}
        exact_self = sum(
            row["self_intersections"]
            for row in exact_rows
            if row["path_id"] in measured_keys
        )
        closed_rows = [
            row for row in measured if row["closed"] and row["area_change_pct"] != ""
        ]
        significant_area = [
            row for row in closed_rows if float(row["area_exact"]) >= 25.0
        ]
        area_pcts = [abs(float(row["area_change_pct"])) for row in significant_area]
        reductions = [
            float(row["curvature_reduction_p95"])
            for row in measured
            if row["curvature_reduction_p95"] != ""
            and np.isfinite(row["curvature_reduction_p95"])
        ]
        curv_ratios = [
            float(row["curvature_ratio"])
            for row in measured
            if np.isfinite(float(row["curvature_ratio"]))
        ]
        full_metrics = _aggregate_config(
            [
                {
                    "mean_disp": row["mean_disp"],
                    "p95_disp": row["p95_disp"],
                    "max_disp": row["max_disp"],
                    "p95_curvature": row["p95_curvature_common"],
                    "exact_p95_curvature": row["exact_p95_curvature_common"],
                    "length_change_pct": row["length_change_pct"],
                    "status": row["status"],
                }
                for row in variant_rows
            ]
        )
        self_total = int(sum(row["self_intersections"] for row in measured))
        cross_delta = int(sum(row["cross_intersection_delta"] for row in variant_panels))
        aggregate[variant] = {
            **full_metrics,
            "n_open": sum(1 for row in variant_rows if not row["closed"]),
            "n_closed": sum(1 for row in variant_rows if row["closed"]),
            "n_short": sum(
                1 for row in variant_rows if row["status"] == "short_passthrough"
            ),
            "n_error": sum(
                1 for row in variant_rows if row["status"].startswith("error")
            ),
            "n_degenerate": sum(
                1 for row in variant_rows if row["status"] == "degenerate_passthrough"
            ),
            "n_area_measured": len(significant_area),
            "area_mean_abs_pct": float(np.mean(area_pcts)) if area_pcts else 0.0,
            "area_max_abs_pct": float(np.max(area_pcts)) if area_pcts else 0.0,
            "area_signed_mean_pct": (
                float(np.mean([float(row["area_change_pct"]) for row in significant_area]))
                if significant_area
                else 0.0
            ),
            "mean_curvature_reduction_final": float(np.mean(reductions)) if reductions else 0.0,
            "mean_curvature_ratio": float(np.mean(curv_ratios)) if curv_ratios else float("nan"),
            "max_endpoint_drift": float(
                np.max([row["endpoint_drift"] for row in measured])
            ),
            "self_intersections_total": self_total,
            "self_intersections_exact_total": exact_self,
            "self_intersection_delta": self_total - exact_self,
            "cross_intersections_exact": int(
                sum(row["cross_intersections_exact"] for row in variant_panels)
            ),
            "cross_intersections_smooth": int(
                sum(row["cross_intersections_smooth"] for row in variant_panels)
            ),
            "cross_intersection_delta": cross_delta,
            "topology_violation_delta": max(0, cross_delta) + max(0, self_total - exact_self),
            "runtime_s": float(sum(row["runtime_s"] for row in variant_panels)),
            "flag_count": int(sum(row["flag_count"] for row in variant_panels)),
        }
    return aggregate


def build_rankings(aggregate: dict[str, dict]) -> dict[str, list[str]]:
    variants = list(VARIANT_ORDER)
    return {
        "displacement": sorted(variants, key=lambda v: aggregate[v]["mean_p95_disp"]),
        "curvature": sorted(
            variants, key=lambda v: -aggregate[v]["mean_curvature_reduction_final"]
        ),
        "endpoint": sorted(variants, key=lambda v: aggregate[v]["max_endpoint_drift"]),
        "topology": sorted(variants, key=lambda v: aggregate[v]["topology_violation_delta"]),
        "area": sorted(variants, key=lambda v: aggregate[v]["area_mean_abs_pct"]),
        "runtime": sorted(variants, key=lambda v: aggregate[v]["runtime_s"]),
    }


def ranking_value(aggregate: dict[str, dict], criterion: str, variant: str) -> str:
    values = {
        "displacement": f"{aggregate[variant]['mean_p95_disp']:.3f}",
        "curvature": f"{aggregate[variant]['mean_curvature_reduction_final']:.3f}",
        "endpoint": f"{aggregate[variant]['max_endpoint_drift']:.3f}",
        "topology": f"{aggregate[variant]['topology_violation_delta']}",
        "area": f"{aggregate[variant]['area_mean_abs_pct']:.2f}",
        "runtime": f"{aggregate[variant]['runtime_s']:.2f}",
    }
    return values[criterion]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _font(size: int = 14):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNS.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _panel_composite(
    panel,
    prepared_paths: list[PreparedPath],
    outputs: dict[str, dict[str, np.ndarray]],
    selected: dict[str, float],
    panel_rows: list[dict],
    cell: tuple[int, int] = (380, 240),
) -> Image.Image:
    label_width = 170
    header_height = 64
    caption_height = 30
    cell_width, cell_max_height = cell
    thumb_height = int(
        round(cell_width * panel.labels.shape[0] / max(1, panel.labels.shape[1]))
    )
    cell_height = max(80, min(cell_max_height, thumb_height))
    columns = list(RENDER_ORDER)
    width = label_width + cell_width * len(columns)
    height = header_height + cell_height + caption_height
    canvas = Image.new("RGB", (width, height), (16, 16, 18))
    draw = ImageDraw.Draw(canvas)
    font = _font(14)
    small = _font(11)
    exact_paths = [
        BoundaryPath(
            boundary_id=path.boundary_id,
            labels_pair=path.labels_pair,
            points=path.raw,
            closed=path.closed,
            exact_length=max(1, int(round(path.length))),
        )
        for path in prepared_paths
    ]
    draw.text((8, 8), panel.panel_id, fill=(240, 240, 240), font=font)
    draw.text(
        (8, 28),
        f"{len(prepared_paths)} keep paths | "
        f"{sum(1 for p in prepared_paths if p.closed)} closed",
        fill=(170, 170, 170),
        font=small,
    )
    for column_index, variant in enumerate(columns):
        x = label_width + column_index * cell_width
        title = variant
        if variant != "exact":
            title += f"  [{param_short(variant, selected[variant])}]"
        draw.text((x + 6, 10), title, fill=(220, 220, 120), font=small)
        draw.text(
            (x + 6, 28),
            VARIANT_DESCRIPTIONS[variant],
            fill=(150, 150, 150),
            font=small,
        )
        if variant == "exact":
            smoothed = []
        else:
            smoothed = []
            for path in prepared_paths:
                points = outputs[variant][path.key]
                if path.closed:
                    points = np.vstack([points, points[:1]])
                smoothed.append(points)
        image = render_overlay(panel.labels, exact_paths, smoothed)
        image.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
        paste_x = x + max(0, (cell_width - image.width) // 2)
        canvas.paste(image, (paste_x, header_height))
        panel_row = next(
            row for row in panel_rows if row["panel_id"] == panel.panel_id and row["variant"] == variant
        )
        caption = (
            f"p95disp={panel_row['mean_p95_disp']:.2f} "
            f"curvR={panel_row['mean_curvature_reduction']:.2f} "
            f"cross {panel_row['cross_intersections_exact']}->"
            f"{panel_row['cross_intersections_smooth']}"
        )
        draw.text((x + 6, header_height + cell_height + 4), caption, fill=(180, 220, 180), font=small)
    return canvas


def _contact_from_composites(
    composites: list[tuple[str, Image.Image]],
    rankings: dict[str, list[str]],
    aggregate: dict[str, dict],
) -> Image.Image:
    width = max(image.width for _, image in composites)
    title_height = 44
    ranking_height = 30 + 6 * 30
    height = title_height + sum(image.height for _, image in composites) + ranking_height
    canvas = Image.new("RGB", (width, height), (14, 14, 16))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(22)
    font = _font(18)
    draw.text(
        (10, 8),
        "Layer 2 experiment B - smoothing kernels on identical Layer 1 keep paths",
        fill=(250, 230, 90),
        font=title_font,
    )
    y = title_height
    for _, image in composites:
        canvas.paste(image, (0, y))
        y += image.height
    y += 10
    draw.text(
        (10, y),
        "Aggregate ranking (1 = best; pool: non-ph01 smoothed paths). Displacement: mean path p95 px; "
        "curvature: mean p95 reduction; endpoint: max px; topology: cross/self violations; "
        "area: mean |%| closed; runtime: total s.",
        fill=(200, 200, 200),
        font=font,
    )
    y += 30
    labels = [
        ("Displacement", "displacement"),
        ("Curvature reduction", "curvature"),
        ("Endpoint drift", "endpoint"),
        ("Topology violations", "topology"),
        ("Area change", "area"),
        ("Runtime", "runtime"),
    ]
    for label, criterion in labels:
        order = rankings[criterion]
        text = "  ".join(
            f"{index + 1}.{variant}={ranking_value(aggregate, criterion, variant)}"
            for index, variant in enumerate(order)
        )
        draw.text((10, y), f"{label:22s} {text}", fill=(190, 235, 190), font=font)
        y += 30
    return canvas


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _md_cell(value) -> str:
    return str(value).replace("|", "\\|")


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(_md_cell(header) for header in headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_md_cell(value) for value in row) + " |")
    return "\n".join(lines)


def _fmt(value, digits: int = 3) -> str:
    if value == "" or value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "nan"
    return f"{number:.{digits}f}"


def build_summary(
    panels,
    prepared_by_panel,
    selected,
    selection_notes,
    grid_rows,
    rows,
    panel_rows,
    aggregate,
    aggregate_all,
    rankings,
    out_dir: Path,
) -> str:
    lines: list[str] = []
    total_paths = sum(len(prepared_by_panel[panel.panel_id]) for panel in panels)
    total_open = sum(
        1
        for panel in panels
        for path in prepared_by_panel[panel.panel_id]
        if not path.closed
    )
    total_closed = total_paths - total_open
    short_paths = sum(
        1
        for panel in panels
        for path in prepared_by_panel[panel.panel_id]
        if not path.smoothable
    )
    lines.append("# Layer 2 experiment B: smoothing-kernel benchmark")
    lines.append("")
    lines.append(
        "All five kernels run on the **same Layer 1 keep set**, the same "
        "arc-length resampling, the same open/closed convention, and the same "
        "post-fit endpoint reset.  No image diffusion, level set, graph cut or "
        "re-segmentation is used."
    )
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Panels: **{len(panels)}**, keep paths: **{total_paths}** "
                 f"({total_open} open, {total_closed} closed).")
    lines.append(
        f"- Resampling: uniform arc length, `round(length)` samples clamped to "
        f"[{MIN_SAMPLES}, {MAX_SAMPLES}] (about 1 sample/pixel).  Open paths "
        "include both endpoints; closed paths are periodic and do not duplicate "
        "the endpoint."
    )
    lines.append(
        f"- Paths shorter than **{MIN_SMOOTH_LENGTH_PX:g} px** cannot be smoothed "
        f"meaningfully; they are passed through and marked `short_passthrough` "
        f"({short_paths} paths).  They remain in every table; nothing is silently skipped."
    )
    lines.append(
        "- Open-path endpoints are reset to the exact resampled endpoints after "
        "every variant.  `splprep` additionally uses endpoint weights "
        f"{SPLPREP_ENDPOINT_WEIGHT:g}; `savgol` uses edge padding/mode `interp`; "
        "`knot_quintic` uses truncated edge windows.  Closed paths use periodic/wrap "
        "handling and are never endpoint-pinned."
    )
    lines.append(
        "- `knot_quintic` is a sliding **quintic local regression on x(s), y(s)**.  "
        "A true y(x) quintic smoother is not applicable to arbitrary curves "
        "(closed loops, overhangs, vertical segments), so this is the explicit "
        "arc-length equivalent.  It is not a y(x) method."
    )
    lines.append(
        "- Metric note: the harness `displacement` is nearest-vertex distance, not "
        "point-to-segment distance, and the exact/smoothed curves are resampled "
        "at different densities.  The `exact` rows therefore have a small "
        "non-zero displacement baseline (typically a few tenths of a pixel).  "
        "This offset is common to all kernels and the exact rows are retained in "
        "`metrics.csv`; compare kernels against each other and against the exact "
        "row, not against a literal zero."
    )
    lines.append(
        f"- Structural constants: `bspline_lsq` knot spacing "
        f"{BSPLINE_LSQ_KNOT_SPACING:g} px; `pspline_diff2` knot spacing "
        f"{PSPLINE_KNOT_SPACING:g} px; Savitzky-Golay order {SAVGOL_POLYORDER}; "
        f"quintic degree {QUINTIC_DEGREE}."
    )
    lines.append("")
    lines.append("## Global parameter selection")
    lines.append("")
    lines.append(
        "One parameter per kernel was selected on a single global grid pooled "
        f"over all **non-`{SPECIAL_PANEL}`** smoothable paths "
        f"({len([p for panel in panels if panel.panel_id != SPECIAL_PANEL for p in prepared_by_panel[panel.panel_id] if p.smoothable])} paths).  "
        "The selection rule is global and data-independent per panel: among "
        f"configs with mean path p95 displacement <= {SELECT_MEAN_P95_DISP:g} px, "
        f"p95 of path max displacement <= {SELECT_P95_MAX_DISP:g} px and no "
        "error fallback, with at most "
        f"{SELECT_MAX_FALLBACK_FRACTION:.0%} of paths degenerate-restored; choose "
        "the one with the largest mean p95 curvature reduction.  Length change "
        "is reported as an outcome rather than used as a selection gate, because "
        "percentage changes on very short paths would dominate.  If no config is "
        "inside the corridor, choose the one closest to it.  The selected value "
        "is then frozen for all 11 panels.  "
        f"`{SPECIAL_PANEL}` is excluded from selection and reported separately below."
    )
    lines.append("")
    selected_rows = []
    for variant in VARIANT_ORDER:
        note = selection_notes[variant]
        selected_rows.append(
            [
                variant,
                param_label(variant, selected[variant]),
                f"{note['n_eligible']}/{note['n_grid']}",
                note["rule"],
                _fmt(aggregate[variant]["mean_p95_disp"]),
                _fmt(aggregate[variant]["mean_curvature_reduction_final"]),
            ]
        )
    lines.append(
        _md_table(
            ["kernel", "selected parameter", "eligible", "selection rule", "final mean p95 disp", "final curv. reduction"],
            selected_rows,
        )
    )
    lines.append("")
    lines.append(
        "The values in the last two columns are measured on the **ranking pool**: "
        f"non-`{SPECIAL_PANEL}` paths whose fit status is `ok`.  This is the same "
        "population used for parameter selection.  The all-panel/all-path table "
        "below additionally includes short passthrough fragments, which are "
        "identical for every kernel and dilute absolute displacement/curvature "
        "values without changing the ordering."
    )
    lines.append("")

    lines.append("## Aggregate ranking")
    lines.append("")
    lines.append(
        "Lower is better for displacement, endpoint drift, topology violations, "
        "area change and runtime; higher is better for curvature reduction.  "
        "Topology violations count increases in cross-intersections plus "
        "increases in self-intersections over the exact skeleton.  Endpoint "
        "drift is zero for every kernel by construction (endpoints are pinned), "
        "so that criterion is a tie by design rather than a measured advantage."
    )
    lines.append("")
    criteria_rows = []
    labels = [
        ("Displacement", "displacement"),
        ("Curvature reduction", "curvature"),
        ("Endpoint drift", "endpoint"),
        ("Topology violations", "topology"),
        ("Area change (closed)", "area"),
        ("Runtime", "runtime"),
    ]
    for label, criterion in labels:
        order = rankings[criterion]
        criteria_rows.append(
            [
                label,
                " > ".join(
                    f"{variant} ({ranking_value(aggregate, criterion, variant)})"
                    for variant in order
                ),
            ]
        )
    lines.append(_md_table(["criterion", "ranking (best to worst)"], criteria_rows))
    lines.append("")

    most_conservative = rankings["displacement"][0]
    smooths_most = rankings["curvature"][0]
    shortest_area = rankings["area"][0]
    cleanest_topology = rankings["topology"][0]
    fastest = rankings["runtime"][0]
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        f"- **Most conservative:** `{most_conservative}` has the smallest mean "
        f"path p95 displacement ({_fmt(aggregate[most_conservative]['mean_p95_disp'])} px) "
        "at its selected global setting.  It is the least disruptive to the "
        "Layer 1 geometry."
    )
    lines.append(
        f"- **Smooths most:** `{smooths_most}` has the largest mean p95 curvature "
        f"reduction ({_fmt(aggregate[smooths_most]['mean_curvature_reduction_final'])}) "
        f"at a mean p95 displacement of "
        f"{_fmt(aggregate[smooths_most]['mean_p95_disp'])} px.  Aggressive settings "
        "were allowed by the common displacement corridor, so this is a "
        "smoothing-vs-fidelity trade-off, not a free win."
    )
    lines.append(
        f"- **Area preservation (non-ph01 pool):** `{shortest_area}` has the "
        f"smallest mean absolute closed-body area change "
        f"({_fmt(aggregate[shortest_area]['area_mean_abs_pct'], 2)} %), while the "
        f"largest is `{rankings['area'][-1]}` "
        f"({_fmt(aggregate[rankings['area'][-1]]['area_mean_abs_pct'], 2)} %).  "
        "This pool excludes ph01, whose large closed loop is a much more severe "
        "area failure (see the special-panel and failure-case sections): the "
        "closed-body ranking is therefore not a global area guarantee."
    )
    lines.append(
        f"- **Topology:** `{cleanest_topology}` has the fewest added cross/self "
        f"intersections ({aggregate[cleanest_topology]['topology_violation_delta']}); "
        f"the worst is `{rankings['topology'][-1]}` "
        f"({aggregate[rankings['topology'][-1]]['topology_violation_delta']})."
    )
    lines.append(
        f"- **Runtime:** on the ranking pool, `{fastest}` is fastest at "
        f"{_fmt(aggregate[fastest]['runtime_s'], 2)} s; "
        f"`{rankings['runtime'][-1]}` is slowest at "
        f"{_fmt(aggregate[rankings['runtime'][-1]]['runtime_s'], 2)} s.  Runtime "
        "is measured for fitting only, not for metric or rendering work."
    )
    lines.append(
        "- **Endpoint drift:** every kernel reports exactly zero because open "
        "endpoints are pinned after fitting.  This metric checks the harness, "
        "not the kernel; the meaningful endpoint question is shape distortion "
        "just inside the ends, visible in the composite images."
    )
    lines.append("")

    lines.append("## Per-variant aggregate (all 11 panels, all paths)")
    lines.append("")
    agg_rows = []
    for variant in VARIANT_ORDER:
        values = aggregate_all[variant]
        agg_rows.append(
            [
                variant,
                values["n_paths"],
                values["n_measured"],
                values["n_short"],
                values["n_degenerate"],
                _fmt(values["mean_p95_disp"]),
                _fmt(values["mean_disp"]),
                _fmt(values["max_disp"]),
                _fmt(values["mean_curvature_reduction_final"]),
                _fmt(values["mean_curvature_ratio"]),
                _fmt(values["mean_abs_length_change_pct"], 2),
                _fmt(values["max_endpoint_drift"], 4),
                values["self_intersections_total"],
                values["cross_intersection_delta"],
                _fmt(values["area_mean_abs_pct"], 2),
                _fmt(values["runtime_s"], 2),
            ]
        )
    lines.append(
        _md_table(
            [
                "kernel",
                "paths",
                "measured",
                "short",
                "degen",
                "mean p95 disp",
                "mean disp",
                "max disp",
                "curv. reduction",
                "max curv. ratio",
                "|len change| %",
                "endpoint drift",
                "self-int",
                "cross delta",
                "|area| %",
                "runtime s",
            ],
            agg_rows,
        )
    )
    lines.append("")
    lines.append(
        "`measured` excludes degenerate/error fallbacks from the metric means; "
        "`degen` counts paths where a fit collapsed the geometry and the exact "
        "polyline was restored.  `curv. reduction` is "
        "`1 - p95 curvature(smoothed)/p95 curvature(exact)` "
        "on the same resampled points (negative means the smoother added "
        "curvature).  `max curv. ratio` is the per-path `max_curvature` ratio "
        "averaged over paths; it is outlier-sensitive and is reported for "
        "completeness, while the p95 reduction is the more stable curvature "
        "summary.  `|area| %` is the mean absolute area change over "
        "closed paths with exact area >= 25 px^2; small degenerate loops are "
        "excluded from that average but remain in the CSV."
    )
    lines.append("")
    pool_rows = []
    for variant in VARIANT_ORDER:
        values = aggregate[variant]
        pool_rows.append(
            [
                variant,
                values["n_paths"],
                _fmt(values["mean_p95_disp"]),
                _fmt(values["mean_disp"]),
                _fmt(values["max_disp"]),
                _fmt(values["mean_curvature_reduction_final"]),
                _fmt(values["mean_abs_length_change_pct"], 2),
                values["self_intersections_total"],
                values["cross_intersection_delta"],
                _fmt(values["area_mean_abs_pct"], 2),
                _fmt(values["runtime_s"], 3),
            ]
        )
    lines.append("## Per-variant aggregate (ranking pool)")
    lines.append("")
    lines.append(
        f"Non-`{SPECIAL_PANEL}` smoothable paths with `status == ok`: **150 "
        "smoothable paths** per kernel, of which **149 are measured** after the "
        "shared tiny-loop degenerate fallback.  This is the population used for "
        "the ranking and interpretation above.  Cross-intersection deltas here "
        "exclude ph01; ph01 is tabulated separately."
    )
    lines.append("")
    lines.append(
        _md_table(
            [
                "kernel",
                "paths",
                "mean p95 disp",
                "mean disp",
                "max disp",
                "curv. reduction",
                "|len change| %",
                "self-int",
                "cross delta",
                "|area| %",
                "runtime s",
            ],
            pool_rows,
        )
    )
    lines.append("")
    lines.append(
        "The all-path table and this ranking-pool table use the same per-path "
        "metrics; the difference is only the inclusion of short passthrough "
        "fragments and the special panel.  Rankings are reported on the pool "
        "because short fragments cannot be smoothed by any kernel and would "
        "otherwise dominate the average with identical zero-displacement rows."
    )
    lines.append("")

    lines.append("## Per-panel displacement and topology")
    lines.append("")
    lines.append("Mean path p95 displacement (px):")
    lines.append("")
    panel_ids = [panel.panel_id for panel in panels]
    disp_rows = []
    for panel_id in panel_ids:
        row = [panel_id]
        for variant in VARIANT_ORDER:
            match = next(
                item
                for item in panel_rows
                if item["panel_id"] == panel_id and item["variant"] == variant
            )
            row.append(_fmt(match["mean_p95_disp"]))
        disp_rows.append(row)
    lines.append(_md_table(["panel"] + list(VARIANT_ORDER), disp_rows))
    lines.append("")
    lines.append("Cross-intersection delta vs exact (smoothed minus exact):")
    lines.append("")
    cross_rows = []
    for panel_id in panel_ids:
        row = [panel_id]
        for variant in VARIANT_ORDER:
            match = next(
                item
                for item in panel_rows
                if item["panel_id"] == panel_id and item["variant"] == variant
            )
            row.append(
                f"{match['cross_intersections_smooth']} ({match['cross_intersection_delta']:+d})"
            )
        cross_rows.append(row)
    lines.append(_md_table(["panel"] + list(VARIANT_ORDER), cross_rows))
    lines.append("")

    lines.append(f"## Special panel: `{SPECIAL_PANEL}`")
    lines.append("")
    special_prepared = prepared_by_panel[SPECIAL_PANEL]
    lines.append(
        f"This panel is known under-segmented.  It contributes "
        f"**{len(special_prepared)}** keep paths, of which "
        f"**{sum(1 for p in special_prepared if not p.smoothable)}** are shorter "
        f"than {MIN_SMOOTH_LENGTH_PX:g} px and are passed through.  It is excluded "
        "from parameter selection; the values below use the same global settings."
    )
    lines.append("")
    special_rows = []
    for variant in VARIANT_ORDER:
        values = next(
            row
            for row in panel_rows
            if row["panel_id"] == SPECIAL_PANEL and row["variant"] == variant
        )
        special_rows.append(
            [
                variant,
                values["n_paths"],
                values["n_short_passthrough"],
                _fmt(values["mean_p95_disp"]),
                _fmt(values["mean_curvature_reduction"]),
                _fmt(values["area_mean_abs_pct"], 2),
                f"{values['cross_intersections_exact']}->{values['cross_intersections_smooth']}",
                values["flag_count"],
            ]
        )
    lines.append(
        _md_table(
            [
                "kernel",
                "paths",
                "short",
                "mean p95 disp",
                "curv. reduction",
                "|area| %",
                "cross exact->smooth",
                "flags",
            ],
            special_rows,
        )
    )
    lines.append("")
    lines.append(
        "Because most boundaries are tiny fragments between under-segmented "
        "labels, displacement and curvature on this panel are dominated by "
        "passthrough paths and a few long boundaries; it is useful as a stress "
        "case, not as a quality target."
    )
    lines.append("")
    ph01_closed = [
        row
        for row in rows
        if row["panel_id"] == SPECIAL_PANEL
        and row["variant"] != "exact"
        and row["closed"]
        and row["area_change_pct"] != ""
        and float(row["area_exact"]) >= 25.0
    ]
    if ph01_closed:
        worst = min(ph01_closed, key=lambda row: float(row["area_change_pct"]))
        lines.append(
            f"The one large closed loop on this panel (path `{worst['path_id'].split(':')[-1]}`, "
            f"{_fmt(worst['area_exact'], 1)} px^2) loses up to "
            f"**{_fmt(abs(float(worst['area_change_pct'])), 1)}% of its area** "
            f"under `{worst['variant']}`.  This is a genuine closed-body failure "
            "on under-segmented geometry, not a metric artifact."
        )
        lines.append("")

    lines.append("## Failure cases and diagnostics")
    lines.append("")
    closed_paths = [
        path
        for panel in panels
        for path in prepared_by_panel[panel.panel_id]
        if path.closed
    ]
    lines.append(
        f"- **Closed loops:** {len(closed_paths)} closed keep paths.  Periodic "
        "handling is used for all kernels, so endpoints cannot drift, but a "
        "smoother can still shrink or inflate the enclosed body.  The worst "
        "absolute area changes are listed below."
    )
    area_candidates = [
        row
        for row in rows
        if row["variant"] != "exact"
        and row["closed"]
        and row["area_change_pct"] != ""
        and float(row["area_exact"]) >= 25.0
    ]
    area_candidates.sort(key=lambda row: -abs(float(row["area_change_pct"])))
    area_rows = [
        [
            row["panel_id"],
            row["path_id"].split(":")[-1],
            row["labels_pair"],
            row["variant"],
            _fmt(row["area_exact"], 1),
            _fmt(row["area_change_pct"], 2),
            _fmt(row["max_disp"]),
        ]
        for row in area_candidates[:8]
    ]
    if area_rows:
        lines.append("")
        lines.append(
            _md_table(
                ["panel", "path", "labels", "kernel", "exact area px^2", "area change %", "max disp px"],
                area_rows,
            )
        )
    else:
        lines.append("  No closed path with area >= 25 px^2 was measured.")

    degenerate_rows = [
        row
        for row in rows
        if row["variant"] != "exact" and row["status"] == "degenerate_passthrough"
    ]
    lines.append("")
    lines.append(
        f"- **Degenerate-collapse fallbacks:** {len(degenerate_rows)} path/kernel "
        "fits collapsed the geometry and were restored to the exact polyline.  "
        "These rows are excluded from the metric means and are visible here:"
    )
    lines.append("")
    if degenerate_rows:
        lines.append(
            _md_table(
                ["panel", "path", "labels", "kernel", "closed", "length px", "exact area px^2"],
                [
                    [
                        row["panel_id"],
                        row["path_id"].split(":")[-1],
                        row["labels_pair"],
                        row["variant"],
                        "yes" if row["closed"] else "no",
                        _fmt(row["exact_length_px"], 1),
                        _fmt(row["area_exact"], 2),
                    ]
                    for row in degenerate_rows
                ],
            )
        )
    else:
        lines.append("  None: no selected setting collapsed a path.")
    lines.append("")
    lines.append(
        "  The dominant case is a tiny closed loop (`newimage_p2:b6:p5`, about "
        "9 px long and 5 px^2 in area): any global smoothing length larger than "
        "the loop itself collapses it.  The fallback is intentional and explicit; "
        "it is not a silent deletion and it does not change the Layer 1 keep set."
    )

    worst_disp = sorted(
        [row for row in rows if row["variant"] != "exact"],
        key=lambda row: -float(row["max_disp"]),
    )[:8]
    lines.append("")
    lines.append("- **Thin / highly curved paths:** the largest maximum displacements are:")
    lines.append("")
    lines.append(
        _md_table(
            ["panel", "path", "labels", "kernel", "exact length px", "max disp", "p95 disp", "flags"],
            [
                [
                    row["panel_id"],
                    row["path_id"].split(":")[-1],
                    row["labels_pair"],
                    row["variant"],
                    _fmt(row["exact_length_px"], 1),
                    _fmt(row["max_disp"]),
                    _fmt(row["p95_disp"]),
                    ";".join(_adverse_flags(row["flags"])),
                ]
                for row in worst_disp
            ],
        )
    )

    new_self = [
        row
        for row in rows
        if row["variant"] != "exact"
        and int(row["self_intersections"]) > int(row["self_intersections_exact"])
    ]
    new_self.sort(key=lambda row: -int(row["self_intersections"]))
    lines.append("")
    lines.append(
        f"- **Topology:** {len(new_self)} path/variant rows gained at least one "
        "self-intersection.  Cross-intersection changes by panel are tabulated "
        "above; the largest positive deltas are:"
    )
    cross_candidates = sorted(
        [row for row in panel_rows if row["variant"] != "exact"],
        key=lambda row: -row["cross_intersection_delta"],
    )[:8]
    lines.append("")
    lines.append(
        _md_table(
            ["panel", "kernel", "exact", "smoothed", "delta"],
            [
                [
                    row["panel_id"],
                    row["variant"],
                    row["cross_intersections_exact"],
                    row["cross_intersections_smooth"],
                    f"{row['cross_intersection_delta']:+d}",
                ]
                for row in cross_candidates
            ],
        )
    )
    fault_cross = [
        row
        for row in panel_rows
        if row["panel_id"] == "paper23_fault_guided"
        and row["variant"] != "exact"
        and row["cross_intersection_delta"] > 0
    ]
    if fault_cross:
        lines.append("")
        lines.append(
            "  On the faulted `paper23_fault_guided` panel, "
            + ", ".join(
                f"`{row['variant']}` adds {row['cross_intersection_delta']}"
                for row in fault_cross
            )
            + " cross-intersections between nearby fault blocks.  This is the "
            "most important non-ph01 topology failure: smoothing is per path, "
            "but paths from different label pairs can still be pushed through "
            "each other by aggressive corner cutting."
        )
    lines.append("")
    lines.append(
        "- **Short fragments:** paths below the smoothing length threshold "
        f"({MIN_SMOOTH_LENGTH_PX:g} px) are identity passthrough for every kernel. "
        "They bias the displacement aggregate toward zero by the same amount for "
        "all kernels, so comparisons remain fair, but panel-level means on "
        "fragment-heavy panels should be read with that in mind."
    )
    lines.append("")
    lines.append(
        f"- **`{SPECIAL_PANEL}`:** under-segmented labels produce many 2-3 px "
        "skeleton fragments and a handful of long boundaries; it is the clearest "
        "failure case for any smoothing benchmark and is therefore not used for "
        "tuning."
    )
    lines.append("")

    lines.append("## Parameter sensitivity (global grid)")
    lines.append("")
    lines.append(
        "All values below are pooled over the same non-ph01 smoothable paths.  "
        "The selected row is marked with `*`.  This is the sensitivity report; "
        "there are no per-panel parameter values."
    )
    lines.append("")
    for variant in VARIANT_ORDER:
        lines.append(f"### `{variant}`")
        lines.append("")
        sensitivity_rows = []
        for row in grid_rows:
            if row["variant"] != variant:
                continue
            marker = "*" if abs(float(row["param"]) - selected[variant]) < 1e-12 else ""
            sensitivity_rows.append(
                [
                    f"{param_label(variant, row['param'])}{marker}",
                    row["n_paths"],
                    _fmt(row["mean_p95_disp"]),
                    _fmt(row["p95_max_disp"]),
                    _fmt(row["mean_curvature_reduction"]),
                    _fmt(row["mean_abs_length_change_pct"], 2),
                    row["n_error"],
                    row["n_degenerate"],
                    _fmt(row["grid_seconds"], 2),
                ]
            )
        lines.append(
            _md_table(
                [
                    "parameter",
                    "paths",
                    "mean p95 disp",
                    "p95 max disp",
                    "mean curv. reduction",
                    "mean |len change| %",
                    "errors",
                    "degen",
                    "grid s",
                ],
                sensitivity_rows,
            )
        )
        lines.append("")

    lines.append("## Sanity checks")
    lines.append("")
    lines.append(
        f"- Expected keep paths: **{total_paths}** per variant; see `sanity.json` "
        "for the exact key-set comparison.  The run aborts if any variant is "
        "missing or has extra paths."
    )
    lines.append(
        "- Every variant produces one row per path per panel in `metrics.csv`; "
        "`panel_metrics.csv` has one row per panel and variant."
    )
    lines.append(
        "- Errors are never silently dropped: a failed fit falls back to the "
        "exact geometry and is marked `error_passthrough:*`.  Fits that collapse "
        "the geometry are likewise restored and marked "
        "`degenerate_passthrough`.  Both statuses are counted in "
        "`sanity.json`; only `short_passthrough`, `ok`, `degenerate_passthrough` "
        "and `exact` should appear, and every variant must cover the full path "
        "key set."
    )
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- `metrics.csv` - one row per panel/path/variant.")
    lines.append("- `panel_metrics.csv` - one row per panel/variant, including cross-intersections.")
    lines.append("- `sensitivity.csv` - the global parameter grid.")
    lines.append("- `selected_params.json` - frozen parameters and selection notes.")
    lines.append("- `sanity.json` - path coverage and status counts.")
    lines.append("- `contact.png` - representative panel grid plus aggregate rankings.")
    lines.append("- `panels/<panel_id>.png` - full per-panel composite for all 11 panels.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_sanity(
    path: Path,
    panels,
    prepared_by_panel,
    rows: list[dict],
    selected: dict[str, float],
    out_dir: Path,
) -> dict:
    expected = {
        path.key
        for panel in panels
        for path in prepared_by_panel[panel.panel_id]
    }
    by_variant: dict[str, dict] = {}
    problems: list[str] = []
    for variant in ("exact",) + VARIANT_ORDER:
        keys = {row["path_id"] for row in rows if row["variant"] == variant}
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        statuses: dict[str, int] = {}
        for row in rows:
            if row["variant"] == variant:
                statuses[row["status"]] = statuses.get(row["status"], 0) + 1
        if missing or extra:
            problems.append(
                f"{variant}: missing={len(missing)} extra={len(extra)}"
            )
        by_variant[variant] = {
            "paths": len(keys),
            "expected": len(expected),
            "missing": missing[:20],
            "extra": extra[:20],
            "statuses": statuses,
        }
    payload = {
        "expected_paths_total": len(expected),
        "panels": {panel.panel_id: len(prepared_by_panel[panel.panel_id]) for panel in panels},
        "variants": by_variant,
        "selected_parameters": {variant: param_label(variant, value) for variant, value in selected.items()},
        "out_dir": str(out_dir),
        "problems": problems,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if problems:
        raise RuntimeError("path coverage sanity check failed: " + "; ".join(problems))
    return payload


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="runs/layer2_experiments/b_kernel_benchmark",
        help="output directory (default: runs/layer2_experiments/b_kernel_benchmark)",
    )
    parser.add_argument(
        "--grid-only",
        action="store_true",
        help="run the global parameter grid and write sensitivity.csv, then exit",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panels_dir = out_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    print("[load] panels", flush=True)
    panels = load_panels()
    prepared_by_panel: dict[str, list[PreparedPath]] = {}
    for panel in panels:
        prepared_by_panel[panel.panel_id] = prepare_paths(panel)
        print(
            f"[load] {panel.panel_id}: {len(prepared_by_panel[panel.panel_id])} paths",
            flush=True,
        )

    grid_rows, selected, selection_notes = run_grid(panels, prepared_by_panel)
    sensitivity_fields = [
        "variant",
        "param",
        "parameters",
        "n_paths",
        "mean_p95_disp",
        "mean_disp",
        "p95_max_disp",
        "max_disp",
        "mean_p95_curvature",
        "mean_curvature_reduction",
        "mean_abs_length_change_pct",
        "mean_length_change_pct",
        "n_error",
        "n_degenerate",
        "grid_seconds",
    ]
    write_csv(out_dir / "sensitivity.csv", grid_rows, sensitivity_fields)
    (out_dir / "selected_params.json").write_text(
        json.dumps(
            {
                "selection_corridor": {
                    "mean_p95_disp_px": SELECT_MEAN_P95_DISP,
                    "p95_max_disp_px": SELECT_P95_MAX_DISP,
                    "max_degenerate_fraction": SELECT_MAX_FALLBACK_FRACTION,
                },
                "selection_pool": "all smoothable keep paths except " + SPECIAL_PANEL,
                "selected": {
                    variant: {
                        "value": value,
                        "parameters": param_label(variant, value),
                        "note": selection_notes[variant]["rule"],
                        "n_eligible": selection_notes[variant]["n_eligible"],
                        "n_grid": selection_notes[variant]["n_grid"],
                    }
                    for variant, value in selected.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if args.grid_only:
        print("[done] grid-only mode", flush=True)
        return

    rows, panel_rows, outputs, exact_cross = run_full(
        panels, prepared_by_panel, selected, out_dir
    )

    metric_fields = [
        "variant",
        "parameters",
        "panel_id",
        "path_id",
        "boundary_id",
        "path_index",
        "labels_pair",
        "closed",
        "n_points",
        "exact_length_px",
        "mean_disp",
        "p95_disp",
        "max_disp",
        "max_curvature",
        "p95_curvature",
        "exact_max_curvature",
        "curvature_ratio",
        "max_curvature_common",
        "p95_curvature_common",
        "exact_p95_curvature_common",
        "exact_max_curvature_common",
        "curvature_reduction_p95",
        "length_exact",
        "length_smooth",
        "length_change",
        "length_change_pct",
        "endpoint_drift",
        "self_intersections",
        "self_intersections_exact",
        "area_exact",
        "area_smooth",
        "area_change",
        "area_change_pct",
        "status",
        "flags",
        "runtime_ms",
    ]
    panel_fields = [
        "panel_id",
        "variant",
        "parameters",
        "n_paths",
        "n_open",
        "n_closed",
        "n_short_passthrough",
        "n_degenerate",
        "n_error",
        "mean_disp",
        "mean_p95_disp",
        "p95_disp",
        "max_disp",
        "mean_curvature_reduction",
        "mean_abs_length_change_pct",
        "max_endpoint_drift",
        "self_intersections_total",
        "self_intersections_exact_total",
        "area_mean_abs_pct",
        "area_max_abs_pct",
        "area_signed_mean_pct",
        "n_area_measured",
        "cross_intersections_exact",
        "cross_intersections_smooth",
        "cross_intersection_delta",
        "flag_count",
        "flags",
        "runtime_s",
    ]
    write_csv(out_dir / "metrics.csv", rows, metric_fields)
    write_csv(out_dir / "panel_metrics.csv", panel_rows, panel_fields)
    write_sanity(out_dir / "sanity.json", panels, prepared_by_panel, rows, selected, out_dir)

    aggregate_all = aggregate_variants(rows, panel_rows)
    aggregate = aggregate_variants(
        rows, panel_rows, exclude_special=True, only_ok=True
    )
    rankings = build_rankings(aggregate)

    print("[render] per-panel composites", flush=True)
    composites: list[tuple[str, Image.Image]] = []
    for panel in panels:
        composite = _panel_composite(
            panel,
            prepared_by_panel[panel.panel_id],
            outputs,
            selected,
            panel_rows,
        )
        composite.save(panels_dir / f"{panel.panel_id}.png")
        composites.append((panel.panel_id, composite))
        print(f"[render]   {panel.panel_id}", flush=True)

    representative = [
        "newimage_p0",
        "newimage_p2",
        "paper22_fig6",
        "paper23_fig7a",
        "schematic_panel2",
        SPECIAL_PANEL,
    ]
    contact_sources = [
        (panel_id, image) for panel_id, image in composites if panel_id in representative
    ]
    contact = _contact_from_composites(contact_sources, rankings, aggregate)
    contact.save(out_dir / "contact.png")

    summary = build_summary(
        panels,
        prepared_by_panel,
        selected,
        selection_notes,
        grid_rows,
        rows,
        panel_rows,
        aggregate,
        aggregate_all,
        rankings,
        out_dir,
    )
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(f"[done] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
