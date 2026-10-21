"""Exact partition-closure boundaries.

Boundary extraction is deliberately separate from fitting/smoothing.  A
boundary object here is the set of pixels where two labels meet; no spline, no
RDP, no curvature smoothing.  Layer 1 retention decisions operate on these
objects.  Geometry fitting is a later, independent step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import ndimage

from geoseg.modules.curve_editor.background import background_roles

DecisionAction = Literal["keep", "review", "remove"]


@dataclass
class ExactBoundary:
    id: int
    labels_pair: tuple[int, int]
    pixels: np.ndarray  # (N, 2) x, y
    length: int


@dataclass
class ExactBoundaryDecision:
    boundary: ExactBoundary
    background_role: str
    action: DecisionAction
    reasons: list[str]


def _label_pairs(labels: np.ndarray) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for left, right in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
        mask = left != right
        if not mask.any():
            continue
        for a, b in zip(left[mask].tolist(), right[mask].tolist()):
            if a == b:
                continue
            pairs.add((min(int(a), int(b)), max(int(a), int(b))))
    return pairs


def _pair_mask(labels: np.ndarray, a: int, b: int) -> np.ndarray:
    mask = np.zeros(labels.shape, dtype=bool)
    mask[:, :-1] |= (labels[:, :-1] == a) & (labels[:, 1:] == b)
    mask[:, 1:] |= (labels[:, 1:] == a) & (labels[:, :-1] == b)
    mask[:-1, :] |= (labels[:-1, :] == a) & (labels[1:, :] == b)
    mask[1:, :] |= (labels[1:, :] == a) & (labels[:-1, :] == b)
    return mask


def extract_exact_boundaries(labels: np.ndarray) -> list[ExactBoundary]:
    """Return exact, un-smoothed boundary components for each adjacent label pair."""
    labels = np.asarray(labels, dtype=np.int32)
    objects: list[ExactBoundary] = []
    next_id = 1
    for a, b in sorted(_label_pairs(labels)):
        mask = _pair_mask(labels, a, b)
        components, n_components = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
        for component_id in range(1, n_components + 1):
            rows, cols = np.nonzero(components == component_id)
            if len(rows) == 0:
                continue
            pixels = np.column_stack([cols, rows]).astype(float)
            objects.append(
                ExactBoundary(
                    id=next_id,
                    labels_pair=(int(a), int(b)),
                    pixels=pixels,
                    length=int(len(rows)),
                )
            )
            next_id += 1
    return objects


def decide_exact_boundaries(
    labels: np.ndarray,
    boundaries: list[ExactBoundary] | None = None,
    *,
    min_boundary_pixels: int = 25,
    min_exterior_pixels: int = 200,
) -> list[ExactBoundaryDecision]:
    """Conservative decisions on exact boundaries; never auto-remove."""
    labels = np.asarray(labels, dtype=np.int32)
    if boundaries is None:
        boundaries = extract_exact_boundaries(labels)
    zero = labels == 0
    zero_components, _ = ndimage.label(zero)
    roles = background_roles(labels)
    decisions: list[ExactBoundaryDecision] = []
    for boundary in boundaries:
        a, b = boundary.labels_pair
        reasons: list[str] = []
        if a == b:
            action: DecisionAction = "review"
            role = "unknown"
            reasons.append("same label on both sides")
        elif a != 0 and b != 0:
            if boundary.length < min_boundary_pixels:
                action = "review"
                reasons.append("short foreground boundary; review instead of delete")
            else:
                action = "keep"
                reasons.append("foreground-foreground closure boundary")
            role = "none"
        else:
            # Determine zero component role from pixels adjacent to the object.
            rows = np.rint(boundary.pixels[:, 1]).astype(int)
            cols = np.rint(boundary.pixels[:, 0]).astype(int)
            found_roles: list[str] = []
            for row, col in zip(rows, cols):
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = row + dr, col + dc
                    if not (0 <= rr < labels.shape[0] and 0 <= cc < labels.shape[1]):
                        continue
                    if labels[rr, cc] != 0:
                        continue
                    component_id = int(zero_components[rr, cc])
                    if component_id in roles:
                        found_roles.append(roles[component_id])
            role = "exterior" if "exterior" in found_roles else (
                "internal_hole" if found_roles else "unknown"
            )
            if role == "exterior" and boundary.length >= min_exterior_pixels:
                action = "keep"
                reasons.append("long exterior/surface closure boundary")
            elif role == "exterior":
                action = "review"
                reasons.append("short exterior boundary; review")
            else:
                action = "review"
                reasons.append("internal background hole or unknown role; review")
        decisions.append(
            ExactBoundaryDecision(
                boundary=boundary,
                background_role=role,
                action=action,
                reasons=reasons,
            )
        )
    return decisions


def draw_exact_decisions(
    image: np.ndarray,
    decisions: list[ExactBoundaryDecision],
    *,
    scale: int = 1,
) -> np.ndarray:
    """Draw exact boundary pixels without smoothing (keep=green, review=yellow)."""
    out = np.asarray(image).copy()
    colours = {
        "keep": np.array([40, 210, 80], dtype=np.uint8),
        "review": np.array([255, 200, 20], dtype=np.uint8),
        "remove": np.array([255, 40, 40], dtype=np.uint8),
    }
    for decision in decisions:
        pixels = decision.boundary.pixels
        rows = np.rint(pixels[:, 1]).astype(int)
        cols = np.rint(pixels[:, 0]).astype(int)
        valid = (
            (rows >= 0)
            & (rows < out.shape[0])
            & (cols >= 0)
            & (cols < out.shape[1])
        )
        out[rows[valid], cols[valid]] = colours[decision.action]
        if scale > 1:
            mask = np.zeros(out.shape[:2], dtype=bool)
            mask[rows[valid], cols[valid]] = True
            dilated = ndimage.binary_dilation(mask, iterations=scale - 1)
            out[dilated] = colours[decision.action]
    return out


__all__ = [
    "DecisionAction",
    "ExactBoundary",
    "ExactBoundaryDecision",
    "decide_exact_boundaries",
    "draw_exact_decisions",
    "extract_exact_boundaries",
]
