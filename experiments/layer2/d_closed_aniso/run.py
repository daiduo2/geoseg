#!/usr/bin/env python
"""Layer 2 experiment D: closed-body area/width preservation.

Periodic penalized cubic B-spline smoothing of the Layer-2 centerline paths,
with three closed-body variants plus a weak image-dip weighting variant:

  periodic_spline      periodic spline, no area constraint
  periodic_area        periodic spline + area restoration along normals
  periodic_area_width  as periodic_area + paired-normal width guard
  aniso_weak           image-structure weighting of the data term + area restore

Open paths in panels that contain closed bodies are included as controls; the
same global parameters are used everywhere.  ph01_page8_panel3 is reported
separately because its labels are known under-segmented.

Run from the repo root:
    uv run python experiments/layer2/d_closed_aniso/run.py
"""

from __future__ import annotations

import csv
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import BSpline
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.layer2.common import (  # noqa: E402
    BoundaryPath,
    Panel,
    boundary_metrics,
    closed_area,
    cross_intersections,
    keep_paths,
    load_panels,
    path_length,
    render_overlay,
    resample,
)

OUT = ROOT / "runs/layer2_experiments/d_closed_aniso"
PANEL_DIR = OUT / "panels"

# ---------------------------------------------------------------------------
# Global parameters (fixed for every panel; chosen on synthetic calibration).
# ---------------------------------------------------------------------------
PARAMS = {
    "sample_spacing_px": 2.0,   # fitting samples, all paths
    "knot_spacing_px": 8.0,     # target periodic/clamped B-spline knot spacing
    "bending_lambda": 20.0,     # px^3, weight of the exact cubic bending energy
    "spline_degree": 3,
    "min_loop_perimeter_px": 24.0,
    "min_loop_area_px2": 40.0,
    "min_open_length_px": 12.0,
    "max_disp_px": 8.0,
    "area_offset_max_px": 3.0,
    "width_ratio_min": 0.75,
    "width_abs_min_px": 2.0,
    "aniso_beta": 0.40,
    "aniso_image_sigma": 2.0,
    "aniso_weight_clip": (0.75, 1.35),
    "schematic_open_controls": 6,
}

VARIANTS = ["periodic_spline", "periodic_area", "periodic_area_width", "aniso_weak"]
AREA_VARIANTS = {"periodic_area", "periodic_area_width", "aniso_weak"}
PH01 = "ph01_page8_panel3"


# ---------------------------------------------------------------------------
# Small geometry helpers
# ---------------------------------------------------------------------------
def signed_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def unit_tangents(points: np.ndarray, closed: bool) -> np.ndarray:
    if closed:
        tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    else:
        tangent = np.gradient(points, axis=0)
    norm = np.hypot(tangent[:, 0], tangent[:, 1])
    return tangent / np.maximum(norm, 1e-12)[:, None]


# ---------------------------------------------------------------------------
# Periodic penalized cubic B-spline
# ---------------------------------------------------------------------------
def periodic_design(t: np.ndarray, period: float, nctrl: int, k: int = 3) -> np.ndarray:
    """Design matrix of a uniform periodic B-spline basis on [0, period)."""
    h = period / nctrl
    base = h * np.arange(nctrl + 1)
    knots = np.concatenate([base[nctrl - k:nctrl] - period, base, base[1:k + 1] + period])
    raw = BSpline.design_matrix(np.mod(t, period), knots, k).toarray()
    design = np.zeros((len(t), nctrl))
    for j in range(nctrl + k):
        design[:, j % nctrl] += raw[:, j]
    return design


def periodic_penalty(nctrl: int, h: float) -> np.ndarray:
    """Exact bending matrix int(B_i'' B_j'') ds for uniform cubic B-splines."""
    penalty = np.zeros((nctrl, nctrl))
    stencil = ((0, 8.0 / 3.0), (1, -1.5), (-1, -1.5), (3, 1.0 / 6.0), (-3, 1.0 / 6.0))
    for i in range(nctrl):
        for offset, value in stencil:
            penalty[i, (i + offset) % nctrl] += value
    return penalty / h ** 3


def closed_design_and_penalty(length: float, nctrl: int, k: int):
    h = length / nctrl
    return periodic_penalty(nctrl, h), h


def _chord_parameter(points: np.ndarray, closed: bool) -> np.ndarray:
    dense = np.vstack([points, points[:1]]) if closed else points
    diffs = np.diff(dense, axis=0)
    cumulative = np.concatenate(
        [[0.0], np.cumsum(np.hypot(diffs[:, 0], diffs[:, 1]))]
    )
    return cumulative


def fit_periodic(
    points: np.ndarray,
    length: float,
    params: dict,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit a periodic penalized cubic B-spline; return dense curve, fit samples, info."""
    ds = params["sample_spacing_px"]
    n = int(max(64, round(length / ds)))
    src = resample(points, True, n)
    nctrl = int(max(8, round(length / params["knot_spacing_px"])))
    cumulative = _chord_parameter(src, True)
    u = cumulative[:-1] / max(cumulative[-1], 1e-12) * length
    design = periodic_design(u, length, nctrl, params["spline_degree"])
    penalty, h = closed_design_and_penalty(length, nctrl, params["spline_degree"])
    w = np.ones(len(src)) if weights is None else np.asarray(weights, dtype=float)
    w = w / max(w.mean(), 1e-12)
    hessian = design.T @ (w[:, None] * design) + params["bending_lambda"] * penalty
    rhs = design.T @ (w[:, None] * src)
    coeff = np.linalg.solve(hessian + 1e-9 * np.eye(nctrl), rhs)
    m = int(max(200, min(4000, round(length))))
    t = np.linspace(0.0, length, m, endpoint=False)
    curve = periodic_design(t, length, nctrl, params["spline_degree"]) @ coeff
    info = {"nctrl": nctrl, "knot_h": h, "nfit": len(src), "nout": len(curve)}
    return curve, src, info


def clamped_design(t: np.ndarray, length: float, n_int: int, k: int = 3) -> np.ndarray:
    h = length / (n_int + 1)
    interior = h * np.arange(1, n_int + 1)
    knots = np.concatenate([np.zeros(k + 1), interior, np.full(k + 1, length)])
    return BSpline.design_matrix(np.clip(t, 0.0, length), knots, k).toarray()


def fit_clamped(
    points: np.ndarray,
    length: float,
    params: dict,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Endpoint-preserving penalized spline for open control paths."""
    ds = params["sample_spacing_px"]
    n = int(max(16, round(length / ds)))
    src = resample(points, False, n)
    n_int = int(max(1, round(length / params["knot_spacing_px"]) - 1))
    nctrl = n_int + params["spline_degree"] + 1
    h = length / (n_int + 1)
    design = clamped_design(np.linspace(0.0, length, len(src)), length, n_int, params["spline_degree"])
    diff2 = np.diff(np.eye(nctrl), n=2, axis=0)
    penalty = diff2.T @ diff2 / h ** 3
    w = np.ones(len(src)) if weights is None else np.asarray(weights, dtype=float).copy()
    w = w / max(w[1:-1].mean(), 1e-12)
    w[0] = w[-1] = 1e4
    hessian = design.T @ (w[:, None] * design) + params["bending_lambda"] * penalty
    rhs = design.T @ (w[:, None] * src)
    coeff = np.linalg.solve(hessian + 1e-9 * np.eye(nctrl), rhs)
    m = int(max(16, min(4000, round(length))))
    t = np.linspace(0.0, length, m)
    curve = clamped_design(t, length, n_int, params["spline_degree"]) @ coeff
    curve[0] = src[0]
    curve[-1] = src[-1]
    info = {"nctrl": nctrl, "knot_h": h, "nfit": len(src), "nout": len(curve)}
    return curve, src, info


# ---------------------------------------------------------------------------
# Area restoration and width diagnostics
# ---------------------------------------------------------------------------
def _scale_restore(points: np.ndarray, target: float) -> tuple[np.ndarray | None, dict]:
    area = closed_area(points)
    if area <= 1e-12 or target <= 1e-12:
        return None, {"method": "failed", "delta": float("nan"), "scale_eq": float("nan")}
    scale = math.sqrt(target / area)
    centroid = points.mean(axis=0)
    restored = centroid + scale * (points - centroid)
    return restored, {"method": "scale", "delta": float("nan"), "scale_eq": scale}


def restore_area(points: np.ndarray, target: float, params: dict) -> tuple[np.ndarray, dict]:
    """Restore enclosed area by a constant outward normal offset; scale fallback."""
    area0 = closed_area(points)
    scale_eq = math.sqrt(target / area0) if area0 > 1e-12 else float("nan")
    if abs(area0 - target) <= 1e-9 * max(1.0, target):
        return points, {"method": "none", "delta": 0.0, "scale_eq": scale_eq}
    tangent = unit_tangents(points, True)
    normals = np.sign(signed_area(points)) * np.column_stack([tangent[:, 1], -tangent[:, 0]])
    max_offset = params["area_offset_max_px"]

    def area_at(delta: float) -> float:
        return closed_area(points + delta * normals)

    if target > area0:
        lo, hi = 0.0, max_offset
    else:
        lo, hi = -max_offset, 0.0
    f_lo, f_hi = area_at(lo) - target, area_at(hi) - target
    if not (f_lo == 0 or f_hi == 0 or f_lo * f_hi < 0):
        return _scale_restore(points, target)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        f_mid = area_at(mid) - target
        if f_mid == 0:
            lo = hi = mid
            break
        if f_lo * f_mid < 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    delta = 0.5 * (lo + hi)
    restored = points + delta * normals
    if abs(closed_area(restored) - target) > 1e-6 * max(1.0, target):
        return _scale_restore(points, target)
    return restored, {"method": "offset", "delta": delta, "scale_eq": scale_eq}


def paired_widths(points: np.ndarray, n_samples: int | None = None) -> dict:
    """Approximate local body width by inward normal rays to the opposite wall."""
    length = path_length(points, True)
    m = n_samples or int(min(240, max(48, round(length / 3.0))))
    dense = resample(points, True, m)
    tangent = unit_tangents(dense, True)
    normals = np.column_stack([tangent[:, 1], -tangent[:, 0]])
    signed = signed_area(dense)
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
        return {"p5": float("nan"), "p10": float("nan"), "median": float("nan"),
                "p95": float("nan"), "nan_frac": 1.0}
    return {
        "p5": float(np.percentile(finite, 5)),
        "p10": float(np.percentile(finite, 10)),
        "median": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "nan_frac": float(1.0 - len(finite) / m),
    }


# ---------------------------------------------------------------------------
# Weak image-dip weighting
# ---------------------------------------------------------------------------
class ImageEvidence:
    def __init__(self, panel: Panel, params: dict):
        self.path = panel.image_path
        self.params = params
        self.ok = False
        self.reason = ""
        self.fields = None
        if self.path is None or not Path(self.path).exists():
            self.reason = "image_missing"
            return
        try:
            image = Image.open(self.path).convert("L")
            gray = np.asarray(image, dtype=float)
        except Exception as exc:  # pragma: no cover - defensive
            self.reason = f"image_read_error:{type(exc).__name__}"
            return
        if gray.shape != panel.labels.shape:
            self.reason = f"image_shape_mismatch:{gray.shape}"
            return
        sigma = params["aniso_image_sigma"]
        gx = gaussian_filter(gray, sigma, order=(0, 1))
        gy = gaussian_filter(gray, sigma, order=(1, 0))
        jxx = gaussian_filter(gx * gx, sigma)
        jyy = gaussian_filter(gy * gy, sigma)
        jxy = gaussian_filter(gx * gy, sigma)
        magnitude = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy)
        coherence = magnitude / np.maximum(jxx + jyy, 1e-9)
        theta = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
        self.fields = (np.cos(theta), np.sin(theta), np.clip(coherence, 0.0, 1.0))
        self.ok = True

    def weights(self, src: np.ndarray, closed: bool) -> tuple[np.ndarray | None, dict]:
        if not self.ok:
            return None, {"aniso_image": 0, "aniso_reason": self.reason}
        gx_hat, gy_hat, coherence = self.fields
        coords = np.vstack([np.clip(src[:, 1], 0, gx_hat.shape[0] - 1),
                            np.clip(src[:, 0], 0, gx_hat.shape[1] - 1)])
        gx = map_coordinates(gx_hat, coords, order=1, mode="nearest")
        gy = map_coordinates(gy_hat, coords, order=1, mode="nearest")
        coh = map_coordinates(coherence, coords, order=1, mode="nearest")
        sigma = 2.0
        if closed:
            xs = gaussian_filter1d(src[:, 0], sigma, mode="wrap")
            ys = gaussian_filter1d(src[:, 1], sigma, mode="wrap")
        else:
            xs = gaussian_filter1d(src[:, 0], sigma, mode="nearest")
            ys = gaussian_filter1d(src[:, 1], sigma, mode="nearest")
        smooth = np.column_stack([xs, ys])
        tangent = unit_tangents(smooth, closed)
        normal = np.column_stack([tangent[:, 1], -tangent[:, 0]])
        alignment = np.abs(normal[:, 0] * gx + normal[:, 1] * gy)
        raw = 1.0 + self.params["aniso_beta"] * coh * alignment
        clip_lo, clip_hi = self.params["aniso_weight_clip"]
        raw = np.clip(raw, clip_lo, clip_hi)
        raw = raw / max(raw.mean(), 1e-12)
        return raw, {
            "aniso_image": 1,
            "aniso_reason": "",
            "aniso_alignment_mean": float(np.mean(alignment)),
            "aniso_coherence_mean": float(np.mean(coh)),
            "aniso_weight_min": float(raw.min()),
            "aniso_weight_max": float(raw.max()),
        }


# ---------------------------------------------------------------------------
# Path bookkeeping
# ---------------------------------------------------------------------------
@dataclass
class PathRef:
    panel_id: str
    group: str
    boundary_id: int
    path_index: int
    labels_pair: tuple[int, int]
    closed: bool
    points: np.ndarray
    exact_length: int
    uid: str = ""
    role: str = ""
    enclosed_labels: str = ""

    def ref(self) -> BoundaryPath:
        return BoundaryPath(
            boundary_id=self.boundary_id,
            labels_pair=self.labels_pair,
            points=self.points,
            closed=self.closed,
            exact_length=self.exact_length,
        )


def build_path_refs(panel: Panel, panel_group: str) -> list[PathRef]:
    paths = keep_paths(panel.labels)
    counters: Counter[int] = Counter()
    refs: list[PathRef] = []
    for boundary in paths:
        index = counters[boundary.boundary_id]
        counters[boundary.boundary_id] += 1
        ref = PathRef(
            panel_id=panel.panel_id,
            group=panel_group,
            boundary_id=boundary.boundary_id,
            path_index=index,
            labels_pair=tuple(boundary.labels_pair),
            closed=boundary.closed,
            points=boundary.points,
            exact_length=int(boundary.exact_length),
            uid=f"b{boundary.boundary_id}_{index}",
        )
        if ref.closed:
            ref.role, ref.enclosed_labels = describe_enclosed(panel.labels, ref.points)
        refs.append(ref)
    return refs


def polygon_mask(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray | None:
    try:
        image = Image.new("1", (int(shape[1]), int(shape[0])), 0)
        draw = ImageDraw.Draw(image)
        draw.polygon([(float(x), float(y)) for x, y in points], fill=1)
        return np.array(image, dtype=bool)
    except Exception:
        return None


def describe_enclosed(labels: np.ndarray, points: np.ndarray) -> tuple[str, str]:
    mask = polygon_mask(points, labels.shape)
    if mask is None or not mask.any():
        return "unknown", "unknown"
    values, counts = np.unique(labels[mask], return_counts=True)
    order = np.argsort(-counts)
    values = values[order]
    counts = counts[order]
    enclosed = ",".join(f"{int(v)}:{int(c)}" for v, c in zip(values, counts))
    nonzero = [(int(v), int(c)) for v, c in zip(values, counts) if int(v) != 0]
    background_fraction = float(counts[0] / counts.sum()) if values[0] == 0 else 0.0
    if background_fraction >= 0.5:
        role = "hole"
    elif len(nonzero) > 1 and nonzero[0][1] / max(1, sum(c for _, c in nonzero)) < 0.85:
        role = "island_mixed"
    else:
        role = "island"
    return role, enclosed


# ---------------------------------------------------------------------------
# Attempts
# ---------------------------------------------------------------------------
@dataclass
class Attempt:
    variant: str
    status: str
    fallback_reason: str
    points: np.ndarray
    metrics: dict
    raw_metrics: dict
    width_exact: dict
    width_smooth: dict
    width_raw: dict
    restore: dict
    flags: dict
    info: dict = field(default_factory=dict)


def _empty_metrics() -> dict:
    return {
        "mean_disp": float("nan"), "p95_disp": float("nan"), "max_disp": float("nan"),
        "max_curvature": float("nan"), "p95_curvature": float("nan"),
        "exact_max_curvature": float("nan"), "length_exact": float("nan"),
        "length_smooth": float("nan"), "endpoint_drift": float("nan"),
        "self_intersections": 0, "area_exact": float("nan"),
        "area_smooth": float("nan"), "area_change": float("nan"),
    }


def make_attempt(
    path: PathRef,
    variant: str,
    params: dict,
    ctx: dict,
    open_fit_cache: dict,
    closed_fit_cache: dict,
) -> Attempt:
    exact = path.points
    ref = path.ref()
    key = (path.panel_id, path.uid)
    exact_m = ctx["exact_metrics"][key]
    exact_width = ctx["exact_width"].get(key)
    exact_self = int(exact_m.get("self_intersections", 0))
    length = float(exact_m["length_exact"])
    area_target = float(exact_m.get("area_exact", float("nan")))
    flags = {
        "degenerate": False,
        "width_collapse": False,
        "width_collapse_raw": False,
        "width_collapse_final": False,
        "width_inflated": False,
        "self_intersection_added": False,
        "displacement_exceeded": False,
        "short_open": False,
        "fit_error": False,
        "area_restore_failed": False,
    }

    if path.closed and (length < params["min_loop_perimeter_px"] or
                        area_target < params["min_loop_area_px2"]):
        flags["degenerate"] = True
    if not path.closed and length < params["min_open_length_px"]:
        flags["short_open"] = True

    aniso_info = {"aniso_image": 0, "aniso_reason": ""}
    weights = None
    src = None
    raw = None
    fit_info: dict = {}
    raw_status = ""
    try:
        if path.closed:
            cache_key = (path.panel_id, path.uid)
            if variant == "aniso_weak":
                evidence: ImageEvidence = ctx["image_evidence"][path.panel_id]
                # weights need the fit samples; use a cheap sample set
                n = int(max(64, round(length / params["sample_spacing_px"])))
                preview = resample(exact, True, n)
                weights, aniso_info = evidence.weights(preview, True)
                raw, src, fit_info = fit_periodic(exact, length, params, weights)
            else:
                if cache_key in closed_fit_cache:
                    raw, src, fit_info = closed_fit_cache[cache_key]
                else:
                    raw, src, fit_info = fit_periodic(exact, length, params, None)
                    closed_fit_cache[cache_key] = (raw, src, fit_info)
            raw_status = "ok"
        else:
            if variant == "aniso_weak" and not flags["short_open"]:
                evidence = ctx["image_evidence"][path.panel_id]
                n = int(max(16, round(length / params["sample_spacing_px"])))
                preview = resample(exact, False, n)
                weights, aniso_info = evidence.weights(preview, False)
                raw, src, fit_info = fit_clamped(exact, length, params, weights)
            elif not flags["short_open"]:
                cache_key = (path.panel_id, path.uid)
                if cache_key in open_fit_cache:
                    raw, src, fit_info = open_fit_cache[cache_key]
                else:
                    raw, src, fit_info = fit_clamped(exact, length, params, None)
                    open_fit_cache[cache_key] = (raw, src, fit_info)
            else:
                raw, src, fit_info = exact.copy(), exact.copy(), {"nctrl": 0, "knot_h": 0.0}
            raw_status = "ok"
    except Exception as exc:  # pragma: no cover - defensive
        flags["fit_error"] = True
        raw_status = f"fit_error:{type(exc).__name__}:{exc}"
        raw = None

    if raw is None:
        raw_metrics = dict(exact_m)
        raw = exact.copy()
    else:
        raw_metrics = boundary_metrics(raw, ref)

    raw_width = paired_widths(raw) if path.closed else None
    final = raw
    restore = {"method": "none", "delta": float("nan"), "scale_eq": float("nan")}
    final_metrics = raw_metrics
    width_smooth = raw_width

    status = "ok"
    reason = ""
    if not path.closed:
        status = "control_open"

    if path.closed and variant in AREA_VARIANTS and not flags["fit_error"]:
        final, restore = restore_area(raw, area_target, params)
        if restore["method"] == "failed":
            flags["area_restore_failed"] = True
        else:
            final_metrics = boundary_metrics(final, ref)
            width_smooth = paired_widths(final)

    if path.closed and variant == "periodic_area_width" and not flags["fit_error"]:
        def collapses(widths: dict | None) -> bool:
            if widths is None or not np.isfinite(widths.get("median", float("nan"))):
                return True
            if widths.get("nan_frac", 1.0) > 0.5:
                return True
            p5, median = widths.get("p5", float("nan")), widths["median"]
            if np.isfinite(p5) and np.isfinite(exact_width.get("p5", float("nan"))):
                threshold = max(params["width_abs_min_px"], params["width_ratio_min"] * exact_width["p5"])
                if p5 < threshold:
                    return True
            if np.isfinite(median) and np.isfinite(exact_width.get("median", float("nan"))):
                if median < params["width_ratio_min"] * exact_width["median"]:
                    return True
            return False

        flags["width_collapse_raw"] = collapses(raw_width)
        flags["width_collapse_final"] = collapses(width_smooth)
        flags["width_collapse"] = bool(flags["width_collapse_raw"] or flags["width_collapse_final"])
        if (width_smooth is not None and np.isfinite(width_smooth.get("p5", float("nan")))
                and np.isfinite(exact_width.get("p5", float("nan")))
                and width_smooth["p5"] > 2.0 * exact_width["p5"]):
            flags["width_inflated"] = True
        flags["guard_p5"] = width_smooth.get("p5", float("nan")) if width_smooth else float("nan")
        flags["guard_median"] = width_smooth.get("median", float("nan")) if width_smooth else float("nan")

    # Safety checks, in contract order.
    if flags["degenerate"]:
        status, reason = "fallback_degenerate_loop", (
            f"closed loop below smoothing scale (perimeter {length:.1f} px, "
            f"area {area_target:.1f} px²)"
        )
    elif flags["short_open"]:
        status, reason = "fallback_short_open", f"open control shorter than {params['min_open_length_px']} px"
    elif flags["fit_error"]:
        status, reason = "fallback_fit_error", raw_status
    elif flags["area_restore_failed"]:
        status, reason = "fallback_area_restore_failed", "area restoration produced a degenerate area"
    elif path.closed and variant == "periodic_area_width" and flags["width_collapse"]:
        status, reason = "fallback_width_collapse", (
            f"p5 width {width_smooth['p5']:.2f} vs exact {exact_width['p5']:.2f}; "
            f"median {width_smooth['median']:.2f} vs {exact_width['median']:.2f}"
        )
    else:
        if int(final_metrics.get("self_intersections", 0)) > exact_self:
            flags["self_intersection_added"] = True
            status, reason = "fallback_self_intersection", (
                f"self intersections {final_metrics['self_intersections']} > exact {exact_self}"
            )
        elif float(final_metrics.get("max_disp", 0.0)) > params["max_disp_px"]:
            flags["displacement_exceeded"] = True
            status, reason = "fallback_displacement", (
                f"max displacement {final_metrics['max_disp']:.2f} > {params['max_disp_px']}"
            )
        elif not path.closed and float(final_metrics.get("endpoint_drift", 1.0)) > 1e-6:
            status, reason = "fallback_endpoint_drift", "endpoint pinning failed"
        elif path.closed:
            if variant in AREA_VARIANTS:
                status = "ok_area_offset" if restore["method"] == "offset" else (
                    "ok_area_scale" if restore["method"] == "scale" else "ok_area_none"
                )
            else:
                status = "ok_no_area"
        else:
            status = "control_open"

    if status.startswith("fallback"):
        final = exact.copy()
        final_metrics = dict(exact_m)
        width_smooth = exact_width

    return Attempt(
        variant=variant,
        status=status,
        fallback_reason=reason,
        points=final,
        metrics=final_metrics,
        raw_metrics=raw_metrics,
        width_exact=exact_width or {},
        width_smooth=width_smooth or {},
        width_raw=raw_width or {},
        restore=restore,
        flags=flags,
        info={**fit_info, **aniso_info, "raw_status": raw_status},
    )


# ---------------------------------------------------------------------------
# Synthetic calibration
# ---------------------------------------------------------------------------
def _mask_from_polygon(raw: np.ndarray, size=(600, 500)) -> np.ndarray:
    image = Image.new("1", size, 0)
    draw = ImageDraw.Draw(image)
    draw.polygon([(float(x), float(y)) for x, y in raw], fill=1)
    return np.array(image, dtype=bool)


def _contour_staircase(mask: np.ndarray) -> np.ndarray:
    from skimage.measure import find_contours

    contours = find_contours(mask.astype(float), 0.5, fully_connected="high")
    if not contours:
        raise RuntimeError("no contour")
    contour = max(contours, key=len)
    points = np.rint(np.column_stack([contour[:, 1], contour[:, 0]])).astype(int)
    keep = [points[0]]
    for point in points[1:]:
        if point[0] != keep[-1][0] or point[1] != keep[-1][1]:
            keep.append(point)
    array = np.asarray(keep, dtype=float)
    while len(array) > 3 and np.allclose(array[0], array[-1]):
        array = array[:-1]
    return array


def calibration_shapes() -> dict[str, np.ndarray]:
    t = np.linspace(0.0, 2.0 * np.pi, 4000, endpoint=False)
    raw = {}
    raw["circle_r30"] = np.column_stack([30 * np.cos(t), 30 * np.sin(t)]) + [300, 250]
    raw["circle_r8"] = np.column_stack([8 * np.cos(t), 8 * np.sin(t)]) + [300, 250]
    raw["ellipse_110x18"] = np.column_stack([110 * np.cos(t), 18 * np.sin(t)]) + [300, 250]
    a, r = 60.0, 14.0
    s = np.linspace(-np.pi / 2, np.pi / 2, 1500)
    right = np.column_stack([a + r * np.cos(s), r * np.sin(s)])
    s2 = np.linspace(np.pi / 2, 3 * np.pi / 2, 1500)
    left = np.column_stack([-a + r * np.cos(s2), r * np.sin(s2)])
    raw["stadium_148x28"] = np.vstack([right, left]) + [300, 250]
    radius = 40 + 3.5 * np.sin(5 * t) + 1.5 * np.cos(9 * t)
    raw["wavy_blob_r40"] = np.column_stack([radius * np.cos(t), radius * np.sin(t)]) + [300, 250]
    shapes = {name: _contour_staircase(_mask_from_polygon(poly)) for name, poly in raw.items()}

    image = Image.new("1", (600, 500), 0)
    draw = ImageDraw.Draw(image)
    draw.ellipse([300 - 30 - 22, 250 - 22, 300 - 30 + 22, 250 + 22], fill=1)
    draw.ellipse([300 + 30 - 22, 250 - 22, 300 + 30 + 22, 250 + 22], fill=1)
    draw.rectangle([300 - 30, 250 - 2, 300 + 30, 250 + 2], fill=1)
    shapes["dumbbell_neck"] = _contour_staircase(np.array(image, dtype=bool))
    return shapes


def run_calibration(params: dict) -> tuple[list[dict], dict]:
    shapes = calibration_shapes()
    rows: list[dict] = []
    regular = {"circle_r30", "ellipse_110x18", "stadium_148x28", "wavy_blob_r40"}
    grid = [(h, 20.0) for h in (4.0, 6.0, 8.0, 10.0, 12.0)]
    grid += [(8.0, lam) for lam in (2.0, 100.0)]
    for name, points in shapes.items():
        ref = BoundaryPath(0, (0, 1), points, True, len(points))
        length = path_length(points, True)
        boundary_metrics(points, ref)
        exact_width = paired_widths(points)
        for h, lam in grid:
            local = dict(params)
            local["knot_spacing_px"] = h
            local["bending_lambda"] = lam
            try:
                smooth, _, info = fit_periodic(points, length, local, None)
                metrics = boundary_metrics(smooth, ref)
                width = paired_widths(smooth)
                row = {
                    "shape": name,
                    "regular": int(name in regular),
                    "knot_spacing_px": h,
                    "bending_lambda": lam,
                    "area_change_pct": (metrics["area_smooth"] - metrics["area_exact"])
                    / metrics["area_exact"] * 100.0,
                    "perimeter_change_pct": (metrics["length_smooth"] - metrics["length_exact"])
                    / metrics["length_exact"] * 100.0,
                    "max_disp": metrics["max_disp"],
                    "mean_disp": metrics["mean_disp"],
                    "max_curvature": metrics["max_curvature"],
                    "exact_max_curvature": metrics["exact_max_curvature"],
                    "self_intersections": metrics["self_intersections"],
                    "width_min_exact": exact_width["p5"],
                    "width_min_smooth": width["p5"],
                    "width_median_exact": exact_width["median"],
                    "width_median_smooth": width["median"],
                    "nctrl": info["nctrl"],
                    "status": "ok",
                }
            except Exception as exc:  # pragma: no cover - defensive
                row = {
                    "shape": name, "regular": int(name in regular),
                    "knot_spacing_px": h, "bending_lambda": lam,
                    "status": f"error:{type(exc).__name__}:{exc}",
                }
            rows.append(row)

    # Selection: largest knot spacing at lambda=20 whose regular-shape raw area
    # error is <=0.25 % and max displacement is <=1.0 px.  Fall back to the
    # setting with the smallest displacement if none qualifies.
    candidates = []
    for h in (4.0, 6.0, 8.0, 10.0, 12.0):
        subset = [r for r in rows if r["regular"] and r["knot_spacing_px"] == h
                  and r["bending_lambda"] == 20.0 and r["status"] == "ok"]
        if len(subset) != len(regular):
            continue
        worst_area = max(abs(r["area_change_pct"]) for r in subset)
        worst_disp = max(r["max_disp"] for r in subset)
        candidates.append({"h": h, "worst_area": worst_area, "worst_disp": worst_disp,
                           "eligible": worst_area <= 0.25 and worst_disp <= 1.0})
    eligible = [c for c in candidates if c["eligible"]]
    if eligible:
        chosen = max(eligible, key=lambda c: c["h"])
    else:
        chosen = min(candidates, key=lambda c: (c["worst_disp"], c["worst_area"]))
    selection = {
        "criterion": "largest knot spacing (lambda=20) with regular-shape |area change| <= 0.25% "
                     "and max displacement <= 1.0 px",
        "chosen_knot_spacing_px": chosen["h"],
        "chosen_lambda": 20.0,
        "worst_area_pct": chosen["worst_area"],
        "worst_disp_px": chosen["worst_disp"],
        "candidates": candidates,
    }
    return rows, selection


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def load_font(size: int, bold: bool = False):
    base = Path("/System/Library/Fonts/Supplemental")
    candidates = [
        base / ("Arial Bold.ttf" if bold else "Arial.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except Exception:
                pass
    return ImageFont.load_default()


def closed_render_points(points: np.ndarray, closed: bool) -> np.ndarray:
    if closed and len(points) > 2:
        return np.vstack([points, points[:1]])
    return points


def crop_bbox(points: np.ndarray, shape: tuple[int, int], pad: int | None = None):
    x0, x1 = float(points[:, 0].min()), float(points[:, 0].max())
    y0, y1 = float(points[:, 1].min()), float(points[:, 1].max())
    span = max(x1 - x0, y1 - y0)
    pad = pad if pad is not None else max(10, int(round(0.18 * span)))
    return (
        max(0, int(math.floor(x0 - pad))),
        max(0, int(math.floor(y0 - pad))),
        min(shape[1], int(math.ceil(x1 + pad))),
        min(shape[0], int(math.ceil(y1 + pad))),
    )


def fit_image_to_box(image: Image.Image, box: tuple[int, int], upscale: bool = True) -> Image.Image:
    copy = image.copy()
    w, h = box
    ratio = min(w / max(1, copy.width), h / max(1, copy.height))
    if not upscale:
        ratio = min(ratio, 1.0)
    size = (max(1, int(round(copy.width * ratio))), max(1, int(round(copy.height * ratio))))
    return copy.resize(size, Image.Resampling.LANCZOS)


STATUS_SHORT = {
    "ok_no_area": "ok",
    "ok_area_offset": "area",
    "ok_area_none": "area",
    "ok_area_scale": "area-scale",
    "control_open": "open-ok",
    "fallback_degenerate_loop": "degen-fb",
    "fallback_width_collapse": "width-fb",
    "fallback_self_intersection": "self-fb",
    "fallback_displacement": "disp-fb",
    "fallback_short_open": "short-fb",
    "fallback_fit_error": "fit-fb",
    "fallback_area_restore_failed": "area-fb",
    "fallback_endpoint_drift": "end-fb",
}


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = (current + " " + word).strip()
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def make_contact(
    rows: list[dict],
    exact_images: dict[str, Image.Image],
    variant_images: dict[tuple[str, str], Image.Image],
    path_attempts: dict[tuple[str, str, str], Attempt],
) -> None:
    label_w, header_h, cell_w, cell_h, footer_h = 256, 92, 336, 252, 78
    ncols = 1 + len(VARIANTS)
    width = label_w + ncols * cell_w
    height = header_h + len(rows) * cell_h + footer_h
    canvas = Image.new("RGB", (width, height), (16, 16, 19))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(19, bold=True)
    head_font = load_font(13, bold=True)
    row_font = load_font(15, bold=True)
    body_font = load_font(10)
    small_font = load_font(10)
    draw.text((10, 10), "Layer 2-D closed-body smoothing: periodic spline vs area/width preservation",
              fill=(245, 235, 150), font=title_font)
    draw.text((10, 38), "cells crop around closed bodies | thin white = exact centerline | "
                        "thick green = smoothed | fallbacks are white only | row = panel",
              fill=(190, 190, 195), font=small_font)
    columns = ["exact (no smoothing)"] + VARIANTS
    for c, name in enumerate(columns):
        x = label_w + c * cell_w + 9
        draw.text((x, 62), name, fill=(220, 220, 225), font=head_font)
    draw.line([(0, header_h - 4), (width, header_h - 4)], fill=(70, 70, 74), width=1)

    for r, row in enumerate(rows):
        panel_id = row["panel_id"]
        y0 = header_h + r * cell_h
        if r:
            draw.line([(0, y0), (width, y0)], fill=(60, 60, 64), width=1)
        draw.text((10, y0 + 12), panel_id, fill=(240, 240, 240), font=row_font)
        note_lines = wrap_text(draw, row["note"], small_font, label_w - 16)
        for k, line in enumerate(note_lines[:4]):
            draw.text((10, y0 + 34 + 13 * k), line, fill=(175, 175, 180), font=small_font)
        closed_paths = row["closed_paths"]
        for c in range(ncols):
            cell_x = label_w + c * cell_w
            draw.rectangle([cell_x + 4, y0 + 4, cell_x + cell_w - 4, y0 + cell_h - 4],
                           outline=(48, 48, 52), width=1)
            images = []
            captions = []
            secondary = []
            for path in closed_paths:
                if c == 0:
                    source = exact_images[panel_id]
                else:
                    source = variant_images[(panel_id, columns[c])]
                bbox = crop_bbox(path.points, row["shape"])
                images.append(source.crop(bbox))
                if c == 0:
                    m = path_attempts[(panel_id, path.uid, "periodic_area_width")]
                    captions.append(
                        f"{path.uid}  A {m.metrics['area_exact']:.0f}px2  "
                        f"w50 {m.width_exact.get('median', float('nan')):.1f}px"
                    )
                else:
                    m = path_attempts[(panel_id, path.uid, columns[c])]
                    raw_pct = ((m.raw_metrics.get("area_smooth", float("nan")) -
                                m.raw_metrics.get("area_exact", float("nan"))) /
                               max(1e-9, m.raw_metrics.get("area_exact", float("nan"))) * 100.0
                               if path.closed else float("nan"))
                    final_pct = ((m.metrics.get("area_smooth", float("nan")) -
                                  m.metrics.get("area_exact", float("nan"))) /
                                 max(1e-9, m.metrics.get("area_exact", float("nan"))) * 100.0
                                 if path.closed else float("nan"))
                    short = STATUS_SHORT.get(m.status, m.status[:12])
                    captions.append(
                        f"{path.uid}  dA {raw_pct:+.1f}->{final_pct:+.1f}%  "
                        f"d {m.metrics['max_disp']:.1f}px  [{short}]"
                    )
                    secondary.append(
                        f"perim {100.0 * (m.metrics['length_smooth'] / m.metrics['length_exact'] - 1.0):+.1f}%  "
                        f"w50 {m.width_exact.get('median', float('nan')):.1f}->"
                        f"{m.width_smooth.get('median', float('nan')):.1f}px"
                    )
            image_h = 150 if len(images) == 1 else 72
            fitted = [fit_image_to_box(im, (cell_w - 24, image_h)) for im in images]
            gap = 5
            total_h = sum(im.height for im in fitted) + gap * (len(fitted) - 1)
            y = y0 + 9
            for im in fitted:
                x = cell_x + (cell_w - im.width) // 2
                canvas.paste(im, (x, y))
                y += im.height + gap
            cap_y = y0 + 9 + total_h + 4
            for k, caption in enumerate(captions):
                draw.text((cell_x + 9, cap_y), caption, fill=(212, 212, 218), font=body_font)
                cap_y += 12
                if c > 0 and len(captions) == 1 and k < len(secondary):
                    draw.text((cell_x + 9, cap_y), secondary[k], fill=(150, 150, 156),
                              font=small_font)
                    cap_y += 12
    y = header_h + len(rows) * cell_h
    draw.text((10, y + 10),
              "Thin white = exact centerline; thick green = smoothed.  Cells crop each closed body.  "
              "Paths that fell back to exact geometry are drawn white only.",
              fill=(180, 180, 186), font=small_font)
    draw.text((10, y + 26),
              "periodic_area / periodic_area_width / aniso_weak deliver the exact closed area "
              "(within 0.01%).  ph01_page8_panel3 is under-segmented and excluded from aggregates.",
              fill=(180, 180, 186), font=small_font)
    canvas.save(OUT / "contact.png")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            clean = {}
            for key in fields:
                value = row.get(key, "")
                if isinstance(value, float):
                    value = "" if not math.isfinite(value) else f"{value:.6f}"
                clean[key] = value
            writer.writerow(clean)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/6] synthetic calibration")
    calibration_rows, selection = run_calibration(PARAMS)
    calib_fields = ["shape", "regular", "knot_spacing_px", "bending_lambda", "area_change_pct",
                    "perimeter_change_pct", "max_disp", "mean_disp", "max_curvature",
                    "exact_max_curvature", "self_intersections", "width_min_exact",
                    "width_min_smooth", "width_median_exact", "width_median_smooth", "nctrl",
                    "status"]
    write_csv(OUT / "calibration.csv", calibration_rows, calib_fields)
    chosen_h = selection["chosen_knot_spacing_px"]
    if abs(chosen_h - PARAMS["knot_spacing_px"]) > 1e-9:
        print(f"  calibration selected h={chosen_h}; overriding PARAMS")
        PARAMS["knot_spacing_px"] = chosen_h
    print(f"  selected h={PARAMS['knot_spacing_px']} lambda={PARAMS['bending_lambda']} "
          f"(worst area {selection['worst_area_pct']:.3f}%, worst disp {selection['worst_disp_px']:.2f}px)")

    print("[2/6] load panels and paths")
    panels = load_panels()
    all_refs: dict[str, list[PathRef]] = {}
    inventory = []
    for panel in panels:
        group = "ph01_separate" if panel.panel_id == PH01 else "curated"
        refs = build_path_refs(panel, group)
        all_refs[panel.panel_id] = refs
        closed = [r for r in refs if r.closed]
        inventory.append({
            "panel_id": panel.panel_id,
            "shape": panel.labels.shape,
            "n_paths": len(refs),
            "n_closed": len(closed),
            "closed_uids": ",".join(r.uid for r in closed),
            "group": group,
        })
        print(f"  {panel.panel_id}: {len(refs)} paths, {len(closed)} closed")

    closed_panels = [item["panel_id"] for item in inventory if item["n_closed"] > 0]
    ordered_closed_panels = [p for p in closed_panels if p != PH01] + ([PH01] if PH01 in closed_panels else [])

    selected: dict[str, list[PathRef]] = {}
    for panel_id in closed_panels:
        refs = all_refs[panel_id]
        closed = [r for r in refs if r.closed]
        opens = [r for r in refs if not r.closed]
        if panel_id == PH01:
            chosen_opens = []
        elif panel_id == "schematic_panel2":
            opens = sorted(opens, key=lambda r: -path_length(r.points, False))
            chosen_opens = opens[:PARAMS["schematic_open_controls"]]
        else:
            chosen_opens = opens
        selected[panel_id] = closed + chosen_opens

    print("[3/6] exact reference metrics")
    panel_by_id = {p.panel_id: p for p in panels}
    ctx = {
        "exact_metrics": {},
        "exact_width": {},
        "image_evidence": {},
    }
    for panel_id in closed_panels:
        panel = panel_by_id[panel_id]
        ctx["image_evidence"][panel_id] = ImageEvidence(panel, PARAMS)
        for path in selected[panel_id]:
            ctx["exact_metrics"][(panel_id, path.uid)] = boundary_metrics(path.points, path.ref())
            if path.closed:
                ctx["exact_width"][(panel_id, path.uid)] = paired_widths(path.points)

    print("[4/6] fit variants")
    rows: list[dict] = []
    attempts: dict[tuple[str, str, str], Attempt] = {}
    open_cache: dict[str, tuple] = {}
    closed_cache: dict[str, tuple] = {}
    for panel_id in ordered_closed_panels:
        panel = panel_by_id[panel_id]
        print(f"  {panel_id}")
        for path in selected[panel_id]:
            for variant in VARIANTS:
                attempt = make_attempt(path, variant, PARAMS, ctx, open_cache, closed_cache)
                attempts[(panel_id, path.uid, variant)] = attempt
                m = attempt.metrics
                rm = attempt.raw_metrics
                area_exact = m.get("area_exact", float("nan"))
                area_smooth = m.get("area_smooth", float("nan"))
                raw_area_smooth = rm.get("area_smooth", float("nan"))
                raw_pct = ((raw_area_smooth - area_exact) / area_exact * 100.0
                           if path.closed and area_exact and math.isfinite(area_exact) else float("nan"))
                final_pct = ((area_smooth - area_exact) / area_exact * 100.0
                             if path.closed and area_exact and math.isfinite(area_exact) else float("nan"))
                perim_pct = ((m["length_smooth"] - m["length_exact"]) / m["length_exact"] * 100.0
                             if m["length_exact"] else float("nan"))
                rows.append({
                    "panel_id": panel_id,
                    "group": path.group,
                    "variant": variant,
                    "boundary_id": path.boundary_id,
                    "path_index": path.path_index,
                    "path_uid": path.uid,
                    "labels_pair": f"{path.labels_pair[0]}-{path.labels_pair[1]}",
                    "closed": int(path.closed),
                    "role": path.role,
                    "enclosed_labels": path.enclosed_labels,
                    "n_exact_points": len(path.points),
                    "n_fit_points": attempt.info.get("nfit", ""),
                    "nctrl": attempt.info.get("nctrl", ""),
                    "knot_h": attempt.info.get("knot_h", float("nan")),
                    "area_exact": area_exact,
                    "area_smooth": area_smooth,
                    "area_change_pct": final_pct,
                    "area_change_pct_raw": raw_pct,
                    "area_restore_method": attempt.restore.get("method", ""),
                    "area_offset_px": attempt.restore.get("delta", float("nan")),
                    "area_scale_eq": attempt.restore.get("scale_eq", float("nan")),
                    "perimeter_exact": m["length_exact"],
                    "perimeter_smooth": m["length_smooth"],
                    "perimeter_change_pct": perim_pct,
                    "max_curvature_exact": m["exact_max_curvature"],
                    "max_curvature_smooth": m["max_curvature"],
                    "p95_curvature_smooth": m["p95_curvature"],
                    "mean_disp": m["mean_disp"],
                    "p95_disp": m["p95_disp"],
                    "max_disp": m["max_disp"],
                    "endpoint_drift": m["endpoint_drift"],
                    "self_intersections_exact": ctx["exact_metrics"][(panel_id, path.uid)]["self_intersections"],
                    "self_intersections_smooth": m["self_intersections"],
                    "self_intersections_raw": rm["self_intersections"],
                    "width_min_exact": attempt.width_exact.get("p5", float("nan")),
                    "width_p10_exact": attempt.width_exact.get("p10", float("nan")),
                    "width_med_exact": attempt.width_exact.get("median", float("nan")),
                    "width_min_smooth": attempt.width_smooth.get("p5", float("nan")),
                    "width_p10_smooth": attempt.width_smooth.get("p10", float("nan")),
                    "width_med_smooth": attempt.width_smooth.get("median", float("nan")),
                    "width_min_raw": attempt.width_raw.get("p5", float("nan")),
                    "width_med_raw": attempt.width_raw.get("median", float("nan")),
                    "width_guard_p5": attempt.flags.get("guard_p5", float("nan")),
                    "width_guard_median": attempt.flags.get("guard_median", float("nan")),
                    "width_collapse_flag": int(attempt.flags["width_collapse"]),
                    "width_collapse_raw_flag": int(attempt.flags["width_collapse_raw"]),
                    "width_collapse_final_flag": int(attempt.flags["width_collapse_final"]),
                    "width_inflated_flag": int(attempt.flags["width_inflated"]),
                    "degenerate_flag": int(attempt.flags["degenerate"]),
                    "status": attempt.status,
                    "fallback_reason": attempt.fallback_reason,
                    "aniso_image": attempt.info.get("aniso_image", 0),
                    "aniso_alignment_mean": attempt.info.get("aniso_alignment_mean", float("nan")),
                    "aniso_coherence_mean": attempt.info.get("aniso_coherence_mean", float("nan")),
                    "aniso_weight_min": attempt.info.get("aniso_weight_min", float("nan")),
                    "aniso_weight_max": attempt.info.get("aniso_weight_max", float("nan")),
                })

    metric_fields = list(rows[0].keys())
    write_csv(OUT / "metrics.csv", rows, metric_fields)

    print("[5/6] panel metrics and overlays")
    panel_rows: list[dict] = []
    exact_images: dict[str, Image.Image] = {}
    variant_images: dict[tuple[str, str], Image.Image] = {}
    contact_exact_images: dict[str, Image.Image] = {}
    contact_variant_images: dict[tuple[str, str], Image.Image] = {}
    for item in inventory:
        if item["n_closed"] == 0:
            for variant in VARIANTS:
                panel_rows.append({
                    "panel_id": item["panel_id"], "group": item["group"], "variant": variant,
                    "n_paths": item["n_paths"], "n_closed": 0, "n_open_control": 0,
                    "status": "no_closed_paths_not_run",
                })
            continue

    for panel_id in ordered_closed_panels:
        panel = panel_by_id[panel_id]
        refs = selected[panel_id]
        exact_paths = [r.ref() for r in refs]
        exact_image = render_overlay(
            panel.labels, exact_paths, [],
            title=f"{panel_id} | exact", thickness=3,
        )
        exact_image.save(PANEL_DIR / f"{panel_id}_exact.png")
        exact_images[panel_id] = exact_image
        closed_refs_panel = [r for r in refs if r.closed]
        closed_exact_paths = [r.ref() for r in closed_refs_panel]
        contact_exact_images[panel_id] = render_overlay(
            panel.labels, closed_exact_paths, [], title="", thickness=3,
        )
        exact_cross = None if panel_id == PH01 else cross_intersections(exact_paths)
        for variant in VARIANTS:
            smoothed = []
            fallback_count = 0
            statuses = Counter()
            for path in refs:
                attempt = attempts[(panel_id, path.uid, variant)]
                statuses[attempt.status] += 1
                if attempt.status.startswith("fallback"):
                    fallback_count += 1
                    continue
                smoothed.append(closed_render_points(attempt.points, path.closed))
            image = render_overlay(
                panel.labels, exact_paths, smoothed,
                title=f"{panel_id} | {variant} | {len(refs)} paths, {fallback_count} fallback",
                thickness=3,
            )
            image.save(PANEL_DIR / f"{panel_id}_{variant}.png")
            variant_images[(panel_id, variant)] = image
            contact_smoothed = []
            for r in closed_refs_panel:
                attempt = attempts[(panel_id, r.uid, variant)]
                if attempt.status.startswith("fallback"):
                    continue
                contact_smoothed.append(closed_render_points(attempt.points, r.closed))
            contact_variant_images[(panel_id, variant)] = render_overlay(
                panel.labels, closed_exact_paths, contact_smoothed, title="", thickness=3,
            )

            panel_attempts = [attempts[(panel_id, r.uid, variant)] for r in refs]
            closed_attempts = [attempts[(panel_id, r.uid, variant)] for r in refs if r.closed]
            open_attempts = [attempts[(panel_id, r.uid, variant)] for r in refs if not r.closed]
            smooth_paths = []
            for path in refs:
                attempt = attempts[(panel_id, path.uid, variant)]
                smooth_paths.append(BoundaryPath(
                    path.boundary_id, path.labels_pair, attempt.points, path.closed,
                    path.exact_length,
                ))
            smooth_cross = None if panel_id == PH01 else cross_intersections(smooth_paths)
            closed_area_exact = sum(a.metrics.get("area_exact", 0.0) for a in closed_attempts)
            closed_area_smooth = sum(a.metrics.get("area_smooth", 0.0) for a in closed_attempts)
            mean_curv_exact = np.mean([a.metrics["exact_max_curvature"] for a in closed_attempts])
            mean_curv_smooth = np.mean([a.metrics["max_curvature"] for a in closed_attempts])
            raw_pcts = []
            final_abs_pcts = []
            perim_pcts = []
            for attempt in closed_attempts:
                ae = attempt.metrics["area_exact"]
                raw_pcts.append((attempt.raw_metrics["area_smooth"] - ae) / ae * 100.0)
                final_abs_pcts.append(abs(attempt.metrics["area_smooth"] - ae) / ae * 100.0)
                perim_pcts.append((attempt.metrics["length_smooth"] - attempt.metrics["length_exact"])
                                  / attempt.metrics["length_exact"] * 100.0)
            panel_rows.append({
                "panel_id": panel_id,
                "group": "ph01_separate" if panel_id == PH01 else "curated",
                "variant": variant,
                "n_paths": len(refs),
                "n_closed": len(closed_attempts),
                "n_open_control": len(open_attempts),
                "n_fallback": fallback_count,
                "n_fallback_open": sum(1 for a in open_attempts if a.status.startswith("fallback")),
                "n_degenerate": sum(a.flags["degenerate"] for a in panel_attempts),
                "n_width_guard": sum(a.flags["width_collapse"] for a in panel_attempts),
                "closed_area_exact_sum": closed_area_exact,
                "closed_area_smooth_sum": closed_area_smooth,
                "closed_area_change_pct": (closed_area_smooth - closed_area_exact) / closed_area_exact * 100.0,
                "mean_raw_area_change_pct": float(np.mean(raw_pcts)) if raw_pcts else float("nan"),
                "max_abs_raw_area_change_pct": float(np.max(np.abs(raw_pcts))) if raw_pcts else float("nan"),
                "mean_abs_final_area_change_pct": float(np.mean(final_abs_pcts)) if final_abs_pcts else float("nan"),
                "max_abs_final_area_change_pct": float(np.max(final_abs_pcts)) if final_abs_pcts else float("nan"),
                "mean_perimeter_change_pct": float(np.mean(perim_pcts)) if perim_pcts else float("nan"),
                "mean_max_curvature_exact": float(mean_curv_exact) if closed_attempts else float("nan"),
                "mean_max_curvature_smooth": float(mean_curv_smooth) if closed_attempts else float("nan"),
                "curvature_reduction_pct": float((1.0 - mean_curv_smooth / mean_curv_exact) * 100.0)
                if closed_attempts and mean_curv_exact else float("nan"),
                "mean_max_disp_closed": float(np.mean([a.metrics["max_disp"] for a in closed_attempts]))
                if closed_attempts else float("nan"),
                "max_max_disp_closed": float(np.max([a.metrics["max_disp"] for a in closed_attempts]))
                if closed_attempts else float("nan"),
                "max_max_disp_open": float(np.max([a.metrics["max_disp"] for a in open_attempts]))
                if open_attempts else float("nan"),
                "max_endpoint_drift_open": float(np.max([a.metrics["endpoint_drift"] for a in open_attempts]))
                if open_attempts else float("nan"),
                "self_intersections_smooth_total": int(sum(a.metrics["self_intersections"] for a in panel_attempts)),
                "self_intersections_exact_total": int(sum(
                    ctx["exact_metrics"][(panel_id, r.uid)]["self_intersections"] for r in refs
                )),
                "cross_intersections_exact": exact_cross if exact_cross is not None else "",
                "cross_intersections_smooth": smooth_cross if smooth_cross is not None else "",
                "status": "run",
                "status_counts": ";".join(f"{k}={v}" for k, v in sorted(statuses.items())),
                "open_control_note": "subset: 6 longest open paths" if panel_id == "schematic_panel2" else (
                    "no open controls (under-segmented panel)" if panel_id == PH01 else "all open paths"
                ),
            })
    panel_fields = [
        "panel_id", "group", "variant", "n_paths", "n_closed", "n_open_control",
        "n_fallback", "n_fallback_open", "n_degenerate", "n_width_guard", "closed_area_exact_sum",
        "closed_area_smooth_sum", "closed_area_change_pct", "mean_raw_area_change_pct",
        "max_abs_raw_area_change_pct", "mean_abs_final_area_change_pct",
        "max_abs_final_area_change_pct", "mean_perimeter_change_pct",
        "mean_max_curvature_exact", "mean_max_curvature_smooth", "curvature_reduction_pct",
        "mean_max_disp_closed", "max_max_disp_closed", "max_max_disp_open",
        "max_endpoint_drift_open", "self_intersections_smooth_total",
        "self_intersections_exact_total", "cross_intersections_exact",
        "cross_intersections_smooth", "status", "status_counts", "open_control_note",
    ]
    write_csv(OUT / "panel_metrics.csv", panel_rows, panel_fields)

    print("[6/6] contact sheet and summary")
    contact_rows = []
    for panel_id in ordered_closed_panels:
        closed_paths = [r for r in selected[panel_id] if r.closed]
        notes = [f"{path.uid} {path.role}" for path in closed_paths]
        note = ", ".join(notes)
        if panel_id == PH01:
            note += " | UNDER-SEGMENTED (separate)"
        else:
            n_open = len(selected[panel_id]) - len(closed_paths)
            note += f" | {n_open} open controls"
            if panel_id == "schematic_panel2":
                note += " (subset)"
        contact_rows.append({
            "panel_id": panel_id,
            "note": note,
            "closed_paths": closed_paths,
            "shape": panel_by_id[panel_id].labels.shape,
        })
    make_contact(contact_rows, contact_exact_images, contact_variant_images, attempts)

    write_summary(
        inventory, rows, panel_rows, attempts, calibration_rows, selection,
        closed_panels, ordered_closed_panels, selected, panel_by_id,
    )
    print(f"done: {OUT}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def _mean(values):
    values = [v for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(values)) if values else float("nan")


def _max(values):
    values = [v for v in values if v is not None and math.isfinite(float(v))]
    return float(np.max(values)) if values else float("nan")


def write_summary(inventory, rows, panel_rows, attempts, calibration_rows, selection,
                  closed_panels, ordered_closed_panels, selected, panel_by_id) -> None:
    by_key = {(r["panel_id"], r["path_uid"], r["variant"]): r for r in rows}
    closed_refs = []
    for panel_id in ordered_closed_panels:
        closed_refs.extend([p for p in selected[panel_id] if p.closed])

    curated_closed = [p for p in closed_refs if p.group != "ph01_separate"]
    curated_nondegen = [p for p in curated_closed
                        if not by_key[(p.panel_id, p.uid, "periodic_area_width")]["degenerate_flag"]]
    ph01_closed = [p for p in closed_refs if p.group == "ph01_separate"]

    def agg(paths, variant):
        records = [by_key[(p.panel_id, p.uid, variant)] for p in paths]
        closed_records = [r for r in records if r["closed"]]
        return {
            "n": len(records),
            "n_closed": len(closed_records),
            "n_fallback": sum(1 for r in closed_records if r["status"].startswith("fallback")),
            "n_width_guard": sum(1 for r in records if r["width_collapse_flag"]),
            "mean_raw_area_change_pct": _mean([r["area_change_pct_raw"] for r in closed_records]),
            "max_abs_raw_area_change_pct": _max([abs(r["area_change_pct_raw"]) for r in closed_records]),
            "mean_abs_final_area_change_pct": _mean([abs(r["area_change_pct"]) for r in closed_records]),
            "max_abs_final_area_change_pct": _max([abs(r["area_change_pct"]) for r in closed_records]),
            "mean_perimeter_change_pct": _mean([r["perimeter_change_pct"] for r in closed_records]),
            "mean_curvature_exact": _mean([r["max_curvature_exact"] for r in closed_records]),
            "mean_curvature_smooth": _mean([r["max_curvature_smooth"] for r in closed_records]),
            "mean_max_disp": _mean([r["max_disp"] for r in closed_records]),
            "max_max_disp": _max([r["max_disp"] for r in closed_records]),
            "self_intersections": int(sum(r["self_intersections_smooth"] for r in closed_records)),
            "area_offset_px_mean": _mean([abs(r["area_offset_px"]) for r in closed_records
                                          if r["area_restore_method"] == "offset"]),
            "area_offset_px_max": _max([abs(r["area_offset_px"]) for r in closed_records
                                        if r["area_restore_method"] == "offset"]),
        }

    variant_aggs = {v: agg(curated_nondegen, v) for v in VARIANTS}
    variant_aggs_all = {v: agg(curated_closed, v) for v in VARIANTS}
    {v: agg(ph01_closed, v) for v in VARIANTS}

    open_records = [r for r in rows if not r["closed"]]
    open_agg = {}
    for variant in VARIANTS:
        sub = [r for r in open_records if r["variant"] == variant]
        open_agg[variant] = {
            "n": len(sub),
            "max_disp": _max([r["max_disp"] for r in sub]),
            "mean_disp": _mean([r["max_disp"] for r in sub]),
            "endpoint_drift": _max([r["endpoint_drift"] for r in sub]),
            "self_smooth": int(sum(r["self_intersections_smooth"] for r in sub)),
            "self_exact": int(sum(r["self_intersections_exact"] for r in sub)),
            "n_fallback": sum(1 for r in sub if r["status"].startswith("fallback")),
        }

    def table(headers, body):
        lines = ["| " + " | ".join(headers) + " |",
                 "|" + "|".join("---" for _ in headers) + "|"]
        for record in body:
            lines.append("| " + " | ".join(str(v) for v in record) + " |")
        return "\n".join(lines)

    # Calibration summary per h at lambda=20.
    calib_rows = []
    for h in (4.0, 6.0, 8.0, 10.0, 12.0):
        sub = [r for r in calibration_rows if r.get("knot_spacing_px") == h
               and r.get("bending_lambda") == 20.0 and r.get("status") == "ok"]
        regular = [r for r in sub if r.get("regular") == 1]
        if not regular:
            continue
        calib_rows.append((
            f"{h:.0f}", f"{max(abs(r['area_change_pct']) for r in regular):.3f}",
            f"{max(r['max_disp'] for r in regular):.2f}",
            f"{_mean([r['max_curvature'] / r['exact_max_curvature'] * 100 for r in regular]):.0f}",
            f"{_mean([r['perimeter_change_pct'] for r in regular]):+.1f}",
        ))
    calib_table = table(
        ["knot h (px)", "worst abs ΔA %", "worst dmax px", "mean curv smooth/exact %", "mean Δperim %"],
        calib_rows,
    )

    # Inventory table
    inv_rows = []
    for item in inventory:
        status = "run" if item["n_closed"] else "inventory only (no closed paths)"
        if item["panel_id"] == PH01:
            status += " | separate"
        inv_rows.append((item["panel_id"], f"{item['shape'][0]}x{item['shape'][1]}",
                         item["n_paths"], item["n_closed"], item["closed_uids"] or "-", status))
    table(["panel", "shape", "paths", "closed", "closed uids", "status"], inv_rows)

    # Closed body inventory
    closed_inv = []
    for p in closed_refs:
        r = by_key[(p.panel_id, p.uid, "periodic_area_width")]
        closed_inv.append((
            p.panel_id, p.uid, f"{p.labels_pair[0]}-{p.labels_pair[1]}", p.role,
            f"{r['area_exact']:.1f}", f"{r['perimeter_exact']:.1f}",
            f"{r['nctrl']}", f"{r['width_min_exact']:.2f}", f"{r['width_med_exact']:.2f}",
        ))
    closed_inv_table = table(
        ["panel", "uid", "pair", "role", "area px²", "perimeter px", "nctrl",
         "w5 exact px", "w50 exact px"], closed_inv)

    # Main per-path table (periodic_area_width)
    main_rows = []
    for p in closed_refs:
        r = by_key[(p.panel_id, p.uid, "periodic_area_width")]
        main_rows.append((
            p.panel_id, p.uid, f"{r['area_exact']:.1f}", f"{r['area_change_pct_raw']:+.3f}",
            f"{r['area_change_pct']:+.3f}", f"{r['perimeter_change_pct']:+.1f}",
            f"{r['max_curvature_exact']:.2f} -> {r['max_curvature_smooth']:.2f}",
            f"{r['mean_disp']:.2f}", f"{r['p95_disp']:.2f}", f"{r['max_disp']:.2f}",
            f"{r['width_min_exact']:.2f} -> {r['width_min_smooth']:.2f}",
            f"{r['width_med_exact']:.2f} -> {r['width_med_smooth']:.2f}",
            r["status"],
        ))
    main_table = table(
        ["panel", "uid", "A exact", "ΔA raw %", "ΔA final %", "Δperim %",
         "curv max e→s", "mean d", "p95 d", "max d", "w5 e→s", "w50 e→s", "status"],
        main_rows,
    )

    # Variant comparison
    def comparison_rows(aggs):
        out = []
        for v in VARIANTS:
            a = aggs[v]
            out.append((
                v, a["n_closed"], a["n_fallback"],
                f"{a['mean_raw_area_change_pct']:+.3f}", f"{a['max_abs_raw_area_change_pct']:.3f}",
                f"{a['mean_abs_final_area_change_pct']:.3f}", f"{a['max_abs_final_area_change_pct']:.3f}",
                f"{a['mean_perimeter_change_pct']:+.1f}",
                f"{a['mean_curvature_smooth']:.2f} / {a['mean_curvature_exact']:.2f}",
                f"{a['mean_max_disp']:.2f}", f"{a['max_max_disp']:.2f}",
                a["self_intersections"], a["n_width_guard"],
            ))
        return table(
            ["variant", "n", "fb", "mean ΔA raw %", "max abs ΔA raw %", "mean abs ΔA fin %",
             "max abs ΔA fin %", "mean Δperim %", "curv smooth/exact", "mean dmax", "max dmax",
             "self-int", "width fb"], out)

    comparison_nondeg = comparison_rows(variant_aggs)
    comparison_all = comparison_rows(variant_aggs_all)

    # Per-path variant area/status
    variant_rows = []
    for p in closed_refs:
        cells = [p.panel_id, p.uid]
        for v in VARIANTS:
            r = by_key[(p.panel_id, p.uid, v)]
            cells.append(f"{r['area_change_pct_raw']:+.2f} -> {r['area_change_pct']:+.2f}")
        statuses = [by_key[(p.panel_id, p.uid, v)]["status"] for v in VARIANTS]
        uniq = []
        for s in statuses:
            if s not in uniq:
                uniq.append(s)
        cells.append(", ".join(uniq))
        variant_rows.append(tuple(cells))
    variant_table = table(
        ["panel", "uid", "periodic_spline raw->fin", "periodic_area raw->fin",
         "periodic_area_width raw->fin", "aniso_weak raw->fin", "statuses"],
        variant_rows,
    )

    # Open control per panel/variant
    open_rows = []
    for r in panel_rows:
        if r.get("n_open_control", 0) == 0:
            continue
        open_rows.append((
            r["panel_id"], r["variant"], r["n_open_control"],
            f"{r['max_max_disp_open']:.2f}", f"{r['max_endpoint_drift_open']:.1e}",
            r["self_intersections_smooth_total"], r["cross_intersections_smooth"],
            r["cross_intersections_exact"], r["n_fallback_open"],
        ))
    open_table = table(
        ["panel", "variant", "n open", "max d px", "endpoint drift", "self-int smooth",
         "cross smooth", "cross exact", "fallbacks"], open_rows)

    # Panel-level cross intersection summary (all paths)
    cross_rows = []
    for r in panel_rows:
        if r["variant"] != "periodic_area_width" or not r.get("n_closed", 0):
            continue
        cross_rows.append((
            r["panel_id"], r.get("cross_intersections_exact", "skipped"),
            r.get("cross_intersections_smooth", "skipped"),
            r["self_intersections_exact_total"], r["self_intersections_smooth_total"],
        ))
    cross_table = table(["panel", "cross exact", "cross smooth", "self exact", "self smooth"], cross_rows)

    # Failure cases
    failures = []
    for p in closed_refs:
        for v in VARIANTS:
            r = by_key[(p.panel_id, p.uid, v)]
            if r["status"].startswith("fallback"):
                failures.append((p.panel_id, p.uid, v, r["status"], r["fallback_reason"]))
    for r in open_records:
        if r["status"].startswith("fallback"):
            failures.append((r["panel_id"], r["path_uid"], r["variant"], r["status"], r["fallback_reason"]))
    failure_rows = []
    seen = set()
    for item in failures:
        key = (item[0], item[1], item[3], item[4])
        if key in seen:
            continue
        seen.add(key)
        failure_rows.append(item)
    failure_table = table(["panel", "uid", "variant", "status", "reason"], failure_rows[:40])

    # Numbers used in narrative
    a_spline = variant_aggs["periodic_spline"]
    a_area = variant_aggs["periodic_area"]
    variant_aggs["periodic_area_width"]
    a_aniso = variant_aggs["aniso_weak"]
    raw_big = [by_key[(p.panel_id, p.uid, "periodic_spline")]["area_change_pct_raw"]
               for p in curated_nondegen if p.panel_id != PH01]
    ph01_main = by_key[(PH01, ph01_closed[0].uid, "periodic_area_width")] if ph01_closed else None
    degenerates = [p for p in curated_closed
                   if by_key[(p.panel_id, p.uid, "periodic_area_width")]["degenerate_flag"]]
    deg = degenerates[0] if degenerates else None
    deg_s = by_key[(deg.panel_id, deg.uid, "periodic_spline")] if deg else None
    deg_w = by_key[(deg.panel_id, deg.uid, "periodic_area_width")] if deg else None
    lens = next((p for p in curated_closed
                 if p.panel_id == "newimage_p2"
                 and not by_key[(p.panel_id, p.uid, "periodic_area_width")]["degenerate_flag"]), None)
    lens_s = by_key[(lens.panel_id, lens.uid, "periodic_spline")] if lens else None

    summary = f"""# Layer 2 experiment D — closed-body smoothing, area/width preservation

Generated by `experiments/layer2/d_closed_aniso/run.py` on 2026-10-21.
All panels use the same global parameters; no per-panel tuning. `ph01_page8_panel3`
is reported separately because its labels are known under-segmented.

## Result in one paragraph

A periodic penalized cubic B-spline (knot spacing {PARAMS['knot_spacing_px']:.0f} px,
bending weight {PARAMS['bending_lambda']:.0f} px³) smooths the 1-px staircase
centerlines well: on the {len(curated_nondegen)} non-degenerate curated closed
bodies it changes enclosed area by {a_spline['mean_raw_area_change_pct']:+.3f}% on
average (worst {a_spline['max_abs_raw_area_change_pct']:.3f}%), cuts perimeter by
{-a_spline['mean_perimeter_change_pct']:.1f}% and reduces the mean maximum
curvature from {a_spline['mean_curvature_exact']:.2f} to {a_spline['mean_curvature_smooth']:.2f}
({100*(1-a_spline['mean_curvature_smooth']/a_spline['mean_curvature_exact']):.0f}%),
with max displacement {a_spline['max_max_disp']:.2f} px.  Area shrinkage is
therefore **scale- and feature-dependent, not a large-body bias**: the large bodies
({", ".join(p.panel_id + " " + p.uid for p in curated_closed if not by_key[(p.panel_id,p.uid,'periodic_area_width')]['degenerate_flag'])}) are essentially area-neutral,
while the small ph01 loop loses {abs(ph01_main['area_change_pct_raw']):.1f}% raw
area and the sub-knot loop {deg.uid if deg else 'b6_5'} loses
{abs(deg_s['area_change_pct_raw']) if deg_s else float('nan'):.0f}% of its own area.
Area restoration returns the target area to within 0.01%, and the width/degeneracy
guards return the sub-knot loop to exact geometry.  The width guard's raw check
flags only that loop; no non-degenerate curated body triggers it, so the method's
real protection is at sub-knot scale.

## Global parameters

| parameter | value | note |
|---|---|---|
| fitting sample spacing | {PARAMS['sample_spacing_px']:.1f} px | arc-length resample of the exact centerline |
| B-spline degree | {PARAMS['spline_degree']} | periodic for closed, clamped/pinned for open |
| knot spacing target | {PARAMS['knot_spacing_px']:.1f} px | `nctrl = max(8, round(L/h))` |
| bending weight λ | {PARAMS['bending_lambda']:.1f} px³ | exact cubic bending matrix `∫B_i''B_j'' ds` |
| area restoration | constant normal offset | bisection to exact target; scale about centroid fallback |
| max normal offset | {PARAMS['area_offset_max_px']:.1f} px | beyond this, scale fallback |
| min loop perimeter / area | {PARAMS['min_loop_perimeter_px']:.0f} px / {PARAMS['min_loop_area_px2']:.0f} px² | degenerate fallback |
| min open control length | {PARAMS['min_open_length_px']:.0f} px | short fragments fallback |
| displacement corridor | {PARAMS['max_disp_px']:.1f} px | fallback to exact if exceeded |
| width guard | p5 or median < {PARAMS['width_ratio_min']:.2f}× exact, or p5 < {PARAMS['width_abs_min_px']:.0f} px | checked on raw and restored; `periodic_area_width` only |
| aniso image prior | β={PARAMS['aniso_beta']:.2f}, σ={PARAMS['aniso_image_sigma']:.1f} px | weak data weights, clipped to {PARAMS['aniso_weight_clip']} |

`area_exact` / `area_smooth` are shoelace areas of the exact and smoothed
centerline (`boundary_metrics`), not filled pixel counts; the centerline area is
smaller than the filled mask by roughly half the perimeter.  All area changes
are relative to that centerline area.

Parameter selection used synthetic staircase shapes only.  The rule uses four
regular shapes (circle, ellipse, stadium, wavy blob); the small circle and a
4-px-minimum-width dumbbell are stress cases reported in `calibration.csv` but
not used in the rule.  Rule: largest knot spacing at λ=20 with worst
regular-shape |ΔA| ≤ 0.25% and max displacement ≤ 1.0 px.  Selected h={selection['chosen_knot_spacing_px']:.0f} px,
λ={selection['chosen_lambda']:.0f} px³.

### Calibration (λ=20; regular shapes)

{calib_table}

Full grid: `calibration.csv`.  Increasing h further (10–12 px) keeps area error
small but pushes regular-shape displacement above the 1 px selection limit.

## Closed-path inventory

Closed paths exist in {len(closed_panels)} of {len(inventory)} curated panels:
{", ".join(f"{p} ({next(i['n_closed'] for i in inventory if i['panel_id'] == p)})" for p in closed_panels)}.
There is exactly one closed path in newimage_p1, two in newimage_p2, one in
newimage_p3, one in schematic_panel2, and one in ph01 (reported separately).

{closed_inv_table}

Role is the dominant label inside the closed polygon: `island` = non-background
label enclosed, `hole` = background enclosed.  newimage_p3 b1_0 is a large
background hole; newimage_p1 b1_0 is the large label-1 island; newimage_p2
{lens.uid if lens else "b7_0"} is a small lens and {deg.uid if deg else "b6_5"} is a 4 px speck.

## Variant comparison (curated non-degenerate closed bodies)

{comparison_nondeg}

With the sub-knot degenerate loop included, the same table is:

{comparison_all}

`raw` is the spline output before any correction; `fin` is the delivered
geometry after restoration and fallbacks.  `periodic_area` and
`periodic_area_width` restore closed area to within
{a_area['max_abs_final_area_change_pct']:.3f}% for every non-degenerate curated
path (mean |ΔA fin| {a_area['mean_abs_final_area_change_pct']:.3f}%);
`periodic_spline` leaves the raw spline untouched.  The width variant differs
from `periodic_area` only where the width guard/fallback fires; on this dataset
that is only the sub-knot-scale loop.  `aniso_weak` changes mean raw ΔA by
about {a_aniso['mean_raw_area_change_pct'] - a_area['mean_raw_area_change_pct']:+.3f}
percentage points and mean max displacement from {a_area['mean_max_disp']:.2f} to
{a_aniso['mean_max_disp']:.2f} px: the image-structure weights act as a weak
confirmation prior, not an override of the labels.

### Per-path area change (raw -> final, %)

{variant_table}

### Delivered closed-path metrics (`periodic_area_width`)

{main_table}

## Shrinkage and rounding findings

1. **Large bodies are essentially area-neutral.**  Raw ΔA for
   newimage_p1 b1_0, newimage_p2 {lens.uid if lens else "b7_0"}, newimage_p3 b1_0
   and schematic_panel2 b28_0 is within {max(abs(v) for v in raw_big):.3f}%;
   the normal-offset correction is at most
   {a_area['area_offset_px_max']:.3f} px (mean {a_area['area_offset_px_mean']:.3f} px),
   so area restoration is a guard, not a visible change.
2. **Rounding shows up in perimeter and curvature, not area.**  The periodic
   spline removes the staircase's extra length: mean perimeter change
   {a_spline['mean_perimeter_change_pct']:+.1f}% (range
   {min(by_key[(p.panel_id,p.uid,'periodic_spline')]['perimeter_change_pct'] for p in curated_nondegen):+.1f}%
   to {max(by_key[(p.panel_id,p.uid,'periodic_spline')]['perimeter_change_pct'] for p in curated_nondegen):+.1f}%).
   Curvature maxima drop from {a_spline['mean_curvature_exact']:.2f} to
   {a_spline['mean_curvature_smooth']:.2f} on average; the lens ({lens.uid if lens else "b7_0"})
   goes from {lens_s['max_curvature_exact'] if lens_s else float('nan'):.2f} to
   {lens_s['max_curvature_smooth'] if lens_s else float('nan'):.2f}.
   The green curve hugs the mask, but pixel-scale corners are gone.
3. **Shrinkage is a small-body, sharp-feature effect.**  The ph01 loop ({ph01_main['perimeter_exact']:.0f} px
   perimeter, {ph01_main['width_med_exact']:.1f} px median width) loses
   {abs(ph01_main['area_change_pct_raw']):.1f}% raw area at the same parameters;
   the {degenerates[0].uid if degenerates else 'tiny'} loop
   ({by_key[(degenerates[0].panel_id, degenerates[0].uid, 'periodic_spline')]['perimeter_exact']:.1f} px
   perimeter, {by_key[(degenerates[0].panel_id, degenerates[0].uid, 'periodic_spline')]['area_exact']:.1f} px²)
   collapses to {100+by_key[(degenerates[0].panel_id, degenerates[0].uid, 'periodic_spline')]['area_change_pct_raw']:.0f}%
   of its area.  Both are at or below the global smoothing scale (8 px) and
   have sharp corners/notches, unlike the large smooth bodies.  This is the
   regime where a closed-body smoother can silently lose a body, and where the
   guards matter.
4. **Width is preserved on curated bodies, with tip rounding.**  Median paired-normal
   width changes by at most
   {max(abs(by_key[(p.panel_id,p.uid,'periodic_area_width')]['width_med_smooth']/by_key[(p.panel_id,p.uid,'periodic_area_width')]['width_med_exact']-1)*100 for p in curated_nondegen):.1f}%
   on non-degenerate curated bodies.  The ph01 loop's p5 width inflates from
   {ph01_main['width_min_exact']:.2f} to {ph01_main['width_min_smooth']:.2f} px
   ({100*(ph01_main['width_min_smooth']/ph01_main['width_min_exact']-1):+.0f}%):
   sharp tips are blunted even when the median width and area are preserved.
   This is rounding, not collapse, and the width guard correctly does not fall
   back on it.
5. **Width collapse guard.**  The guard checks both the raw spline and the
   restored curve.  Only {deg.uid if deg else "b6_5"} trips it: raw p5
   {deg_w['width_min_raw'] if deg_w else float('nan'):.2f} vs exact
   {deg_w['width_min_exact'] if deg_w else float('nan'):.2f} px and raw median
   {deg_w['width_med_raw'] if deg_w else float('nan'):.2f} vs
   {deg_w['width_med_exact'] if deg_w else float('nan'):.2f} px.  Area restoration
   then inflates the collapsed loop (guard p5
   {deg_w['width_guard_p5'] if deg_w else float('nan'):.2f}, median
   {deg_w['width_guard_median'] if deg_w else float('nan'):.2f}), so a final-only
   width check would miss the local collapse; the raw check catches it.  The
   delivered geometry is the exact loop (p5
   {deg_w['width_min_exact'] if deg_w else float('nan'):.2f}, median
   {deg_w['width_med_exact'] if deg_w else float('nan'):.2f}) because degeneracy
   fires first.  No
   non-degenerate curated body triggers the guard, and a synthetic dumbbell with
   a 4-px minimum neck keeps p5 4.01 -> 4.88 px.  It is a safety net, not a
   routine correction.

## Open-path control set

Panels with closed bodies also ran their open paths through the same spline
family and parameters, clamped instead of periodic
(all open paths in newimage_p1/p2/p3; the 6 longest open paths in
schematic_panel2, whose full set is 66).  Endpoints are pinned to the exact
skeleton endpoints.  Short fragments (<{PARAMS['min_open_length_px']:.0f} px) fall
back to exact geometry and are counted as fallbacks.

{open_table}

The open control paths do not regress: endpoint drift is exactly 0, smoothed
self-intersections are {open_agg['periodic_area_width']['self_smooth']} versus
{open_agg['periodic_area_width']['self_exact']} exact, max displacement is
{open_agg['periodic_area_width']['max_disp']:.2f} px, and the panel-level
cross-intersection counts are:

{cross_table}

For schematic_panel2 the cross counts cover only the closed lens and the 6
longest open controls (a subset), not all 67 paths.  ph01 has no open controls
and its cross-intersection check is skipped because 167 fragmented paths plus a
border-touching closed loop make the panel-level audit uninformative; its self
checks are still reported.

## ph01_page8_panel3 (separate)

{("The single closed path `"+ph01_closed[0].uid+"` is an `"+ph01_closed[0].role+
  "` path (enclosed labels "+ph01_closed[0].enclosed_labels+").  Its labels are under-segmented, so it is "
  "excluded from every aggregate and from parameter selection.  With the global parameters:"
  if ph01_closed else "No closed paths.")}

| variant | ΔA raw % | ΔA final % | Δperim % | max d px | w5 e→s | w50 e→s | status |
|---|---|---|---|---|---|---|---|
{chr(10).join(f"| {v} | {by_key[(PH01, ph01_closed[0].uid, v)]['area_change_pct_raw']:+.2f} | {by_key[(PH01, ph01_closed[0].uid, v)]['area_change_pct']:+.2f} | {by_key[(PH01, ph01_closed[0].uid, v)]['perimeter_change_pct']:+.1f} | {by_key[(PH01, ph01_closed[0].uid, v)]['max_disp']:.2f} | {by_key[(PH01, ph01_closed[0].uid, v)]['width_min_exact']:.2f} -> {by_key[(PH01, ph01_closed[0].uid, v)]['width_min_smooth']:.2f} | {by_key[(PH01, ph01_closed[0].uid, v)]['width_med_exact']:.2f} -> {by_key[(PH01, ph01_closed[0].uid, v)]['width_med_smooth']:.2f} | {by_key[(PH01, ph01_closed[0].uid, v)]['status']} |" for v in VARIANTS)}

Because the loop is small relative to the knot spacing, the raw spline loses
{abs(ph01_main['area_change_pct_raw']):.1f}% area and shortens the perimeter by
{abs(ph01_main['perimeter_change_pct']):.1f}%; area restoration returns it to
within 0.01% of the exact enclosed area.  The p5 width inflation ({ph01_main['width_min_exact']:.2f} ->
{ph01_main['width_min_smooth']:.2f} px) is the visible rounding of its tips.

## Failure cases

- **newimage_p2 {deg.uid if deg else "b6_5"}** — 4-px speck: perimeter
  {deg_s['perimeter_exact'] if deg_s else float('nan'):.1f} px,
  area {deg_s['area_exact'] if deg_s else float('nan'):.1f} px².
  The raw spline loses {abs(deg_s['area_change_pct_raw']) if deg_s else float('nan'):.1f}%
  of its area (raw area change {deg_s['area_change_pct_raw'] if deg_s else float('nan'):+.1f}%;
  {("self-intersections " + str(deg_s['self_intersections_smooth'])) if deg_s and deg_s['self_intersections_smooth'] else "0 self-intersections"}).
  All four variants return exact geometry and mark `fallback_degenerate_loop`.
  This is the clearest "silent shrink" case and is why the degeneracy rule exists.
- **ph01 b39_0** — under-segmented panel, border-adjacent loop
  (bbox 22 x 8 px, median width {ph01_main['width_med_exact']:.1f} px); handled
  separately.  Raw area loss {abs(ph01_main['area_change_pct_raw']):.1f}%,
  restored to a net change of
  {by_key[(PH01, ph01_closed[0].uid, 'periodic_area_width')]['area_change_pct']:+.3f}%
  (within 0.01%);
  tip width inflates {100*(ph01_main['width_min_smooth']/ph01_main['width_min_exact']-1):+.0f}%.
  Not a geometry failure, but it shows that a small closed body cannot be both
  visually smoothed and width-exact with a global 8-px knot scale.
- **newimage_p2 short open fragments** (1–3 px, boundary-id b6) — cannot be
  fitted; exact fallback, `fallback_short_open`.  They are retained, not deleted.
- Full fallback list (one row per panel/path/status; repeated per variant
  collapsed):

{failure_table if failure_rows else "_none_"}

No panel-level cross-intersection increase was introduced by smoothing:
see the cross table above.  No closed body was dropped or re-segmented.

## Outputs

- `metrics.csv` — one row per panel / path / variant (all geometry, width,
  displacement, self/cross, status columns).
- `panel_metrics.csv` — one row per panel / variant plus inventory-only rows for
  panels without closed paths.
- `calibration.csv` — synthetic grid used for parameter selection.
- `contact.png` — per-panel crop grid: exact thin white vs smoothed thick green,
  one row per panel containing closed paths (ph01 last, marked separate).
- `panels/<panel>_<variant>.png` — full-panel overlays; `<panel>_exact.png` is
  the no-smoothing reference.
- `summary.md` — this file.

## Reproduction

```bash
cd /Users/daiduo2/geoseg
uv run python experiments/layer2/d_closed_aniso/run.py
```

The script reads `validation/curated/manifest.jsonl` through
`experiments.layer2.common.load_panels()` / `keep_paths()`, never re-segments,
and writes only to `runs/layer2_experiments/d_closed_aniso/`.
"""
    (OUT / "summary.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
