"""Legacy curve-level retention preview for the interactive editor.

Layer 1 audit and retention must use :mod:`exact_boundaries`, which never
smooths.  This module is kept for in-editor preview of curve objects and must
not be used to decide which partition closures survive.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from scipy import ndimage

from geoseg.modules.curve_editor.background import background_roles
from geoseg.modules.curve_editor.extract import ExtractedCurve
from geoseg.modules.curve_editor.refine import (
    BoundaryRefinementConfig,
    RefinedBoundary,
    adjacent_labels,
    boundary_curvature,
    refine_curve,
)

DecisionAction = Literal["keep", "review", "remove"]


@dataclass
class BoundaryDecision:
    curve: ExtractedCurve
    labels_pair: tuple[int, int] | None
    background_role: str
    max_curvature: float
    action: DecisionAction
    reasons: list[str]


@dataclass
class BoundaryDecisionResult:
    decisions: list[BoundaryDecision]
    kept: list[RefinedBoundary]
    review: list[ExtractedCurve]
    removed: list[ExtractedCurve]


def _resample(points: np.ndarray, closed: bool, n: int = 128) -> np.ndarray:
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
    return np.column_stack(
        [
            np.interp(targets, cumulative, dense[:, 0]),
            np.interp(targets, cumulative, dense[:, 1]),
        ]
    )


def curve_background_role(
    labels: np.ndarray,
    points: np.ndarray,
    *,
    closed: bool,
    zero_components: np.ndarray,
    roles: dict[int, str],
    sample_distance: float = 3.0,
) -> str:
    dense = _resample(points, closed, 96)
    if len(dense) < 3:
        return "unknown"
    dx = np.roll(dense[:, 0], -1) - np.roll(dense[:, 0], 1)
    dy = np.roll(dense[:, 1], -1) - np.roll(dense[:, 1], 1)
    length = np.hypot(dx, dy)
    length[length == 0] = 1.0
    normals = np.column_stack([-dy / length, dx / length])
    found: list[str] = []
    h, w = labels.shape
    for point, normal in zip(dense, normals):
        for sign in (1.0, -1.0):
            sample = point + normal * sign * sample_distance
            col = int(round(sample[0]))
            row = int(round(sample[1]))
            if not (0 <= row < h and 0 <= col < w):
                continue
            if labels[row, col] != 0:
                continue
            component_id = int(zero_components[row, col])
            if component_id in roles:
                found.append(roles[component_id])
    if not found:
        return "unknown"
    if "exterior" in found:
        return "exterior"
    return "internal_hole"


def decide_boundaries(
    curves: list[ExtractedCurve],
    labels: np.ndarray,
    *,
    min_exterior_length: float = 200.0,
    max_exterior_curvature: float = 0.35,
    min_boundary_length: float = 25.0,
) -> list[BoundaryDecision]:
    """Apply conservative keep/review decisions; never auto-remove."""
    zero = labels == 0
    zero_components, _ = ndimage.label(zero)
    roles = background_roles(labels)
    decisions: list[BoundaryDecision] = []
    for curve in curves:
        points = np.asarray(curve.points, dtype=float)
        pair = adjacent_labels(labels, points, closed=curve.closed)
        curvature = boundary_curvature(points, curve.closed)
        reasons: list[str] = []

        if pair is None or pair[0] == pair[1]:
            action: DecisionAction = "review"
            role = "unknown"
            reasons.append("no distinct label pair")
        elif 0 not in pair:
            if curve.length < min_boundary_length:
                action = "review"
                reasons.append("short foreground boundary; review instead of delete")
            else:
                action = "keep"
                reasons.append("foreground-foreground boundary")
            role = "none"
        else:
            role = curve_background_role(
                labels,
                points,
                closed=curve.closed,
                zero_components=zero_components,
                roles=roles,
            )
            if role == "exterior":
                if (
                    curve.length >= min_exterior_length
                    and curvature <= max_exterior_curvature
                ):
                    action = "keep"
                    reasons.append("long exterior/surface boundary")
                else:
                    action = "review"
                    reasons.append("short or high-curvature background boundary")
            elif role == "internal_hole":
                action = "review"
                reasons.append("internal background hole; do not auto-delete")
            else:
                action = "review"
                reasons.append("unknown background role")

        decisions.append(
            BoundaryDecision(
                curve=curve,
                labels_pair=pair,
                background_role=role,
                max_curvature=curvature,
                action=action,
                reasons=reasons,
            )
        )
    return decisions


def refine_decisions(
    decisions: list[BoundaryDecision],
    labels: np.ndarray,
    *,
    config: BoundaryRefinementConfig | None = None,
) -> BoundaryDecisionResult:
    """Smooth only kept boundaries; review boundaries stay raw and visible."""
    config = config or BoundaryRefinementConfig()
    kept: list[RefinedBoundary] = []
    review: list[ExtractedCurve] = []
    removed: list[ExtractedCurve] = []
    smooth_config = replace(config, ignore_background=False)
    for decision in decisions:
        if decision.action == "keep":
            refined = refine_curve(decision.curve, labels, config=smooth_config)
            if refined is not None:
                kept.append(refined)
            else:
                review.append(decision.curve)
        elif decision.action == "review":
            review.append(decision.curve)
        else:
            removed.append(decision.curve)
    return BoundaryDecisionResult(
        decisions=decisions,
        kept=kept,
        review=review,
        removed=removed,
    )


__all__ = [
    "BoundaryDecision",
    "BoundaryDecisionResult",
    "DecisionAction",
    "background_roles",
    "curve_background_role",
    "decide_boundaries",
    "refine_decisions",
]
