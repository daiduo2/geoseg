"""Global topology validation for Layer 2 experiment C.

The validator is deliberately independent of the smoother: it consumes the
final smoothed polylines plus the per-path metrics and reports violations.
The fallback driver in ``run_experiment.py`` uses these violations to reduce
smoothing iteratively and, as a last resort, to return exact geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy.spatial import cKDTree

from experiments.layer2.common import BoundaryPath, resample
from experiments.layer2.c_fault_topology.smoothing import clean_points


@dataclass
class Violation:
    code: str
    path_indices: tuple[int, ...]
    detail: str


@dataclass
class PanelAudit:
    violations: list[Violation] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.violations)

    @property
    def path_indices(self) -> set[int]:
        return {index for violation in self.violations for index in violation.path_indices}

    def codes_for(self, index: int) -> list[str]:
        return [violation.code for violation in self.violations if index in violation.path_indices]


def _polyline(points: np.ndarray, closed: bool, samples: int = 100) -> np.ndarray:
    return resample(clean_points(points), closed, min(samples, max(8, len(points))))


def _cross_counts(
    smoothed: list[np.ndarray],
    paths: list[BoundaryPath],
    *,
    samples: int = 100,
) -> dict[tuple[int, int], int]:
    """Count proper segment crossings for every path pair.

    Vectorised version of ``common._segments_intersect`` with the same strict
    interior rule (0.01 < t, u < 0.99), so results are directly comparable
    while avoiding millions of Python calls on large panels.
    """
    polylines = [_polyline(points, path.closed, samples) for points, path in zip(smoothed, paths)]
    counts: dict[tuple[int, int], int] = {}
    for i, j in combinations(range(len(paths)), 2):
        if paths[i].labels_pair == paths[j].labels_pair:
            continue
        a, b = polylines[i], polylines[j]
        count = _segment_cross_count(a, b)
        if count:
            counts[(i, j)] = count
    return counts


def _segment_cross_count(a: np.ndarray, b: np.ndarray) -> int:
    if len(a) < 2 or len(b) < 2:
        return 0
    a0, a1 = a[:-1], a[1:]
    b0, b1 = b[:-1], b[1:]
    r = a1 - a0
    s = b1 - b0
    denominator = r[:, None, 0] * s[None, :, 1] - r[:, None, 1] * s[None, :, 0]
    qp = b0[None, :, :] - a0[:, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qp[..., 0] * s[None, :, 1] - qp[..., 1] * s[None, :, 0]) / denominator
        u = (qp[..., 0] * r[:, None, 1] - qp[..., 1] * r[:, None, 0]) / denominator
    valid = (
        (np.abs(denominator) > 1e-12)
        & (t > 0.01)
        & (t < 0.99)
        & (u > 0.01)
        & (u < 0.99)
    )
    return int(np.count_nonzero(valid))


def _min_separations(
    smoothed: list[np.ndarray],
    paths: list[BoundaryPath],
) -> dict[tuple[int, int], float]:
    trees = [cKDTree(clean_points(points)) for points in smoothed]
    output: dict[tuple[int, int], float] = {}
    for i, j in combinations(range(len(paths)), 2):
        if paths[i].labels_pair == paths[j].labels_pair:
            continue
        if len(smoothed[i]) == 0 or len(smoothed[j]) == 0:
            continue
        distances, _ = trees[j].query(smoothed[i], k=1)
        output[(i, j)] = float(np.min(distances))
    return output


def audit_panel(
    smoothed: list[np.ndarray],
    paths: list[BoundaryPath],
    metrics: list[dict[str, float | int]],
    *,
    exact_cross: dict[tuple[int, int], int],
    exact_min_separation: dict[tuple[int, int], float],
    exact_self: list[int] | None = None,
    max_disp: float = 3.0,
    p95_disp: float = 2.5,
    endpoint_drift: float = 0.75,
    area_change_fraction: float = 0.02,
    min_separation_px: float = 0.5,
    samples: int = 100,
) -> PanelAudit:
    """Validate a panel's smoothed geometry against exact topology."""
    audit = PanelAudit()
    len(paths)

    for index, (path, metric) in enumerate(zip(paths, metrics)):
        if metric.get("max_disp", 0.0) > max_disp:
            audit.violations.append(
                Violation(
                    "displacement_corridor",
                    (index,),
                    f"max displacement {metric['max_disp']:.2f} px > {max_disp:.2f}",
                )
            )
        if metric.get("p95_disp", 0.0) > p95_disp:
            audit.violations.append(
                Violation(
                    "p95_displacement_corridor",
                    (index,),
                    f"p95 displacement {metric['p95_disp']:.2f} px > {p95_disp:.2f}",
                )
            )
        if not path.closed and metric.get("endpoint_drift", 0.0) > endpoint_drift:
            audit.violations.append(
                Violation(
                    "endpoint_drift",
                    (index,),
                    f"endpoint drift {metric['endpoint_drift']:.2f} px > {endpoint_drift:.2f}",
                )
            )
        if int(metric.get("self_intersections", 0)) > (exact_self[index] if exact_self else 0):
            audit.violations.append(
                Violation(
                    "self_intersection",
                    (index,),
                    f"{int(metric['self_intersections'])} self-intersections",
                )
            )
        if path.closed and "area_exact" in metric and metric["area_exact"] > 0:
            change = abs(float(metric["area_smooth"]) - float(metric["area_exact"]))
            if change / float(metric["area_exact"]) > area_change_fraction:
                audit.violations.append(
                    Violation(
                        "closed_area",
                        (index,),
                        f"closed area changed by {100 * change / metric['area_exact']:.1f}%",
                    )
                )

    smooth_cross = _cross_counts(smoothed, paths, samples=samples)
    for (i, j), count in smooth_cross.items():
        baseline = exact_cross.get((i, j), 0)
        if count > baseline:
            audit.violations.append(
                Violation(
                    "cross_intersection",
                    (i, j),
                    f"new cross-intersections {count} (exact {baseline})",
                )
            )

    smooth_sep = _min_separations(smoothed, paths)
    for pair, distance in smooth_sep.items():
        exact_distance = exact_min_separation.get(pair, float("inf"))
        if distance < min_separation_px and exact_distance > 2.0:
            audit.violations.append(
                Violation(
                    "min_separation",
                    pair,
                    f"separation collapsed from {exact_distance:.1f} to {distance:.2f} px",
                )
            )

    # Drop exact duplicate (code, paths, detail) entries.
    seen = set()
    unique: list[Violation] = []
    for violation in audit.violations:
        key = (violation.code, tuple(sorted(violation.path_indices)), violation.detail)
        if key in seen:
            continue
        seen.add(key)
        unique.append(violation)
    audit.violations = unique
    return audit


__all__ = ["PanelAudit", "Violation", "audit_panel"]
