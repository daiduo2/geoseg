"""Fault/junction-aware smoothing primitives for Layer 2 experiment C.

The smoother is a discrete second-difference (Whittaker/P-spline) penalty
solved as a sparse linear system.  This gives an exact, deterministic
per-path smoother whose only tuning knob is ``lam``.  Junction/fault handling
is expressed as hard pin weights and as an explicit split of the path into
smoothing domains; no image diffusion or re-segmentation is used.

All coordinates are (x, y) in panel pixel units.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from experiments.layer2.common import BoundaryPath, resample

PIN_WEIGHT = 1e6
FREEZE_WEIGHT = 1e4
KINK_ANGLE_DEG = 32.0
FAULT_BAND_PX = 2.5
JUNCTION_NEAR_PX = 2.5
GAP_NEAR_PX = 12.0
KINK_NEAR_OTHER_PX = 6.0


# ---------------------------------------------------------------------------
# Sparse smoother
# ---------------------------------------------------------------------------


def _second_difference(n: int, closed: bool) -> sparse.csr_matrix:
    if closed:
        d = sparse.diags(
            [np.ones(n), -2.0 * np.ones(n), np.ones(n)],
            [-1, 0, 1],
            shape=(n, n),
            format="lil",
        )
        d[0, n - 1] = 1.0
        d[n - 1, 0] = 1.0
        return d.tocsr()
    return sparse.diags(
        [np.ones(n - 2), -2.0 * np.ones(n - 2), np.ones(n - 2)],
        [0, 1, 2],
        shape=(n - 2, n),
        format="csr",
    )


def whittaker_smooth(
    points: np.ndarray,
    lam: float,
    *,
    closed: bool = False,
    pin_indices: Iterable[int] = (),
    freeze_indices: Iterable[int] = (),
) -> np.ndarray:
    """Second-difference penalised least squares, per coordinate.

    ``pin_indices`` are effectively hard constraints; ``freeze_indices`` are
    strongly (but not exactly) held.  For closed curves the penalty is
    periodic and pins are ignored.
    """
    p = clean_points(points)
    n = len(p)
    if n < 4 or lam <= 0:
        return p.copy()
    weights = np.ones(n)
    if not closed:
        pins = sorted({int(i) for i in pin_indices if 0 <= int(i) < n})
        freezes = sorted({int(i) for i in freeze_indices if 0 <= int(i) < n})
        if pins:
            weights[pins] = PIN_WEIGHT
        if freezes:
            weights[freezes] = np.maximum(weights[freezes], FREEZE_WEIGHT)
    d = _second_difference(n, closed)
    a = sparse.diags(weights, offsets=0, format="csc") + lam * (d.T @ d)
    rhs = (weights[:, None] * p).astype(float)
    solved = spsolve(a, rhs)
    solved = np.asarray(solved, dtype=float)
    if solved.ndim == 1:
        solved = solved.reshape(n, 2)
    if not np.all(np.isfinite(solved)):
        return p.copy()
    if not closed:
        # Hard pin: remove numerical residue.
        for index in sorted({int(i) for i in pin_indices if 0 <= int(i) < n}):
            solved[index] = p[index]
    return solved


def unit_resample(points: np.ndarray, closed: bool) -> np.ndarray:
    points = clean_points(points)
    if len(points) < 2:
        return points
    length = float(np.hypot(*np.diff(points, axis=0).T).sum())
    return resample(points, closed, max(3, int(round(length))))


def path_length(points: np.ndarray, closed: bool) -> float:
    dense = np.vstack([points, points[:1]]) if closed else points
    if len(dense) < 2:
        return 0.0
    return float(np.hypot(*np.diff(dense, axis=0).T).sum())


def clean_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return points.copy()
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.any(np.abs(np.diff(points, axis=0)) > 1e-9, axis=1)
    return points[keep]


# ---------------------------------------------------------------------------
# Feature detection
# ---------------------------------------------------------------------------


@dataclass
class PathFeatures:
    """Junction/fault evidence for one traced boundary path."""

    open_anchor_indices: list[int] = field(default_factory=list)  # path endpoints
    split_indices: list[int] = field(default_factory=list)  # interior domain cuts
    pin_indices: list[int] = field(default_factory=list)
    freeze_indices: list[int] = field(default_factory=list)
    evidence: str = "none"  # explicit_fault | explicit_fault_zone | inferred_junction | inferred_kink_junction | ambiguous_gap | none
    fault_frozen: bool = False
    review: bool = False
    endpoint_evidence: tuple[str, str] = ("none", "none")
    endpoint_distances: tuple[float, float] = (float("inf"), float("inf"))
    notes: list[str] = field(default_factory=list)

    @property
    def has_fault_signal(self) -> bool:
        return self.evidence in {
            "explicit_fault",
            "explicit_fault_zone",
            "inferred_kink_junction",
            "ambiguous_gap",
        }

    @property
    def has_junction_signal(self) -> bool:
        return self.evidence in {
            "inferred_junction",
            "inferred_kink_junction",
            "explicit_fault_zone",
        }


def _nearest_other(
    path_index: int,
    paths: list[BoundaryPath],
    trees: list[cKDTree | None],
    point: np.ndarray,
) -> tuple[float, int]:
    best = (float("inf"), -1)
    for other_index, (other, tree) in enumerate(zip(paths, trees)):
        if tree is None or other_index == path_index:
            continue
        if other.boundary_id == paths[path_index].boundary_id:
            continue
        distance = float(tree.query(point, k=1)[0])
        if distance < best[0]:
            best = (distance, other_index)
    return best


def _local_tangent(points: np.ndarray, index: int, window: int = 8) -> np.ndarray:
    a = max(0, index - window)
    b = min(len(points) - 1, index + window)
    vector = points[b] - points[a]
    norm = float(np.hypot(*vector))
    return vector / norm if norm > 0 else np.array([1.0, 0.0])


def turning_angles(points: np.ndarray, window: int = 6) -> np.ndarray:
    """Interior turning angle in degrees (0 = straight, 180 = reversal)."""
    n = len(points)
    angles = np.zeros(n)
    if n < 2 * window + 3:
        return angles
    for i in range(window, n - window):
        before = points[i] - points[i - window]
        after = points[i + window] - points[i]
        norm_before = float(np.hypot(*before))
        norm_after = float(np.hypot(*after))
        if norm_before <= 0 or norm_after <= 0:
            continue
        cos = float(np.clip(np.dot(before, after) / (norm_before * norm_after), -1.0, 1.0))
        angles[i] = float(np.degrees(np.arccos(cos)))
    return angles


def _fault_distance_map(fault_mask: np.ndarray | None) -> np.ndarray | None:
    if fault_mask is None:
        return None
    return ndimage.distance_transform_edt(~np.asarray(fault_mask, dtype=bool))


def _select_peaks(
    values: np.ndarray,
    cluster: list[int],
    *,
    threshold: float = 45.0,
    min_distance: int = 5,
    max_peaks: int = 3,
) -> list[int]:
    """Greedy non-maximum suppression of turning-angle peaks in a cluster."""
    candidates = sorted(
        (int(i) for i in cluster if float(values[i]) >= threshold),
        key=lambda i: float(values[i]),
        reverse=True,
    )
    chosen: list[int] = []
    for index in candidates:
        if all(abs(index - other) >= min_distance for other in chosen):
            chosen.append(index)
            if len(chosen) >= max_peaks:
                break
    return sorted(chosen)


def _cluster_indices(indices: list[int], gap: int = 2) -> list[list[int]]:
    if not indices:
        return []
    clusters = [[indices[0]]]
    for index in indices[1:]:
        if index - clusters[-1][-1] <= gap:
            clusters[-1].append(index)
        else:
            clusters.append([index])
    return clusters


def detect_features(
    path: BoundaryPath,
    path_index: int,
    paths: list[BoundaryPath],
    trees: list[cKDTree | None],
    *,
    fault_distance: np.ndarray | None = None,
    fault_explicit: bool = False,
    curated: bool = False,
) -> PathFeatures:
    """Classify endpoints and interior kinks and derive smoothing anchors.

    ``fault_explicit`` is only true for synthetic cases with a known fault
    mask/label.  On curated panels no fault label exists, so the only things
    reported are *inferred* junction/kink evidence; any geometry gap evidence
    is marked ambiguous and the path is kept exact by the caller.
    """
    features = PathFeatures()
    points = clean_points(path.points)
    n = len(points)
    if n < 3:
        features.evidence = "none"
        return features

    if path.closed:
        if fault_distance is not None:
            near = fault_distance[np.rint(points[:, 1]).astype(int), np.rint(points[:, 0]).astype(int)]
            fraction = float(np.mean(near <= FAULT_BAND_PX + 0.5))
            if fraction >= 0.5:
                features.fault_frozen = True
                features.evidence = "explicit_fault_zone"
                features.notes.append(f"closed body {fraction:.0%} within fault band; frozen exact")
            elif fraction > 0.0:
                features.evidence = "explicit_fault"
                features.notes.append(f"closed body {fraction:.1%} within fault band; periodic smoothing")
        if features.evidence == "none":
            features.evidence = "none"
        return features

    ends = (points[0], points[-1])
    evidence: list[str] = []
    distances: list[float] = []
    for end_index, end in enumerate(ends):
        other_distance, other_index = _nearest_other(path_index, paths, trees, end)
        distances.append(other_distance)
        if fault_distance is not None:
            row, col = int(round(end[1])), int(round(end[0]))
            if 0 <= row < fault_distance.shape[0] and 0 <= col < fault_distance.shape[1]:
                fd = float(fault_distance[row, col])
            else:
                fd = float("inf")
        else:
            fd = float("inf")
        if fault_explicit and fd <= FAULT_BAND_PX + 0.5:
            evidence.append("explicit_fault_endpoint")
        elif other_distance <= JUNCTION_NEAR_PX:
            other = paths[other_index]
            tangent_a = _local_tangent(points, 0 if end_index == 0 else n - 1)
            # local tangent of the other path at the nearest point
            _, nearest = trees[other_index].query(end, k=1)
            other_points = clean_points(other.points)
            nearest_index = int(nearest)
            tangent_b = _local_tangent(other_points, nearest_index)
            angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(tangent_a, tangent_b)), 0.0, 1.0))))
            if angle < 25.0 and other_distance > 1.5:
                evidence.append("ambiguous_gap")
                features.notes.append(
                    f"endpoint near parallel boundary {other.labels_pair} at {other_distance:.1f} px (angle {angle:.0f} deg)"
                )
            else:
                evidence.append("inferred_junction")
        elif other_distance <= GAP_NEAR_PX and curated:
            evidence.append("ambiguous_gap")
            features.notes.append(
                f"endpoint {other_distance:.1f} px from a different boundary (possible offset)"
            )
        else:
            evidence.append("isolated")
    features.endpoint_evidence = (evidence[0], evidence[1])
    features.endpoint_distances = (distances[0], distances[1])

    if "explicit_fault_endpoint" in evidence:
        features.evidence = "explicit_fault"
    elif "ambiguous_gap" in evidence:
        features.evidence = "ambiguous_gap"
        features.review = True
    elif "inferred_junction" in evidence:
        # A junction is a geometric fact, not a fault claim; pinning it is the
        # conservative default.  Only kink splits and ambiguous gaps are
        # flagged for review as uncertain fault evidence.
        features.evidence = "inferred_junction"
        features.review = False
    else:
        features.evidence = "none"

    features.open_anchor_indices = [0, n - 1]
    features.pin_indices.extend([0, n - 1])

    # ----- interior fault bands (synthetic only) ---------------------------
    if fault_distance is not None:
        distances_px = fault_distance[np.rint(points[:, 1]).astype(int), np.rint(points[:, 0]).astype(int)]
        near_mask = distances_px <= FAULT_BAND_PX
        near_fraction = float(np.mean(near_mask))
        angles = turning_angles(points)
        if near_fraction >= 0.5:
            # The path runs along the fault zone.  Smoothing *along* the trace
            # does not smooth across the fault; split only at its kinks and
            # pin the endpoints so the fault trace is not displaced.
            features.evidence = "explicit_fault_zone"
            features.notes.append(
                f"{near_fraction:.0%} of path lies along the fault zone; smoothed along trace"
            )
            candidates = [int(i) for i in np.nonzero(angles >= 40.0)[0] if 2 <= i < n - 2]
            peaks = _select_peaks(
                angles, candidates, threshold=40.0, min_distance=8, max_peaks=3
            )
            for index in peaks:
                features.split_indices.append(index)
                features.pin_indices.append(index)
                features.freeze_indices.extend(
                    [i for i in range(n) if 0 < i < n and abs(i - index) <= 2]
                )
                features.notes.append(
                    f"fault-zone kink at index {index} (angle {angles[index]:.0f} deg)"
                )
        else:
            near_indices = np.nonzero(near_mask)[0].tolist()
            for cluster in _cluster_indices(near_indices):
                if len(cluster) == 0:
                    continue
                # Preserve strong corners inside the fault band; if the path
                # only crosses the fault transversely, pin the crossing point.
                corners = _select_peaks(angles, cluster, threshold=45.0, min_distance=5, max_peaks=3)
                if corners:
                    chosen = corners
                else:
                    chosen = [cluster[len(cluster) // 2]]
                for index in chosen:
                    if 1 <= index < n - 1:
                        features.split_indices.append(index)
                        features.pin_indices.append(index)
                        features.freeze_indices.extend(
                            [i for i in range(n) if 0 < i < n and abs(i - index) <= 2]
                        )
                        features.notes.append(
                            f"fault-adjacent anchor at index {index} (angle {angles[index]:.0f} deg)"
                        )
            if features.split_indices:
                features.evidence = "explicit_fault"

    # ----- interior kinks near other boundaries (curated) ------------------
    angles = turning_angles(points)
    candidate_indices = [int(i) for i in np.nonzero(angles >= KINK_ANGLE_DEG)[0]]
    peak_indices = _select_peaks(
        angles, candidate_indices, threshold=KINK_ANGLE_DEG, min_distance=6, max_peaks=4
    )
    for index in peak_indices:
        if not (2 <= index < n - 2):
            continue
        other_distance, _ = _nearest_other(path_index, paths, trees, points[index])
        if other_distance <= KINK_NEAR_OTHER_PX:
            # Merge a run of adjacent peak indices to one anchor.
            if features.split_indices and abs(index - features.split_indices[-1]) <= 3:
                continue
            features.split_indices.append(int(index))
            features.pin_indices.append(int(index))
            features.notes.append(
                f"kink {angles[index]:.0f} deg within {other_distance:.1f} px of another boundary"
            )
            if curated and features.evidence in {"none", "inferred_junction"}:
                features.evidence = "inferred_kink_junction"
                features.review = True

    features.split_indices = sorted(set(i for i in features.split_indices if 0 < i < n - 1))
    features.pin_indices = sorted(set(i for i in features.pin_indices if 0 <= i < n))
    features.freeze_indices = sorted(set(i for i in features.freeze_indices if 0 <= i < n))
    return features


# ---------------------------------------------------------------------------
# Variant smoothers
# ---------------------------------------------------------------------------


def smooth_global(points: np.ndarray, closed: bool, lam: float) -> np.ndarray:
    base = unit_resample(points, closed)
    return whittaker_smooth(base, lam, closed=closed)


def smooth_junction_pinned(points: np.ndarray, closed: bool, lam: float) -> np.ndarray:
    base = unit_resample(points, closed)
    if closed:
        return whittaker_smooth(base, lam, closed=True)
    return whittaker_smooth(base, lam, closed=False, pin_indices=[0, len(base) - 1])


def _arc_length(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.zeros(len(points))
    diffs = np.diff(points, axis=0)
    return np.concatenate([[0.0], np.cumsum(np.hypot(diffs[:, 0], diffs[:, 1]))])


def _map_feature_indices(
    original: np.ndarray,
    resampled: np.ndarray,
    indices: Iterable[int],
) -> list[int]:
    """Map original pixel-path indices onto a resampled (unit-spacing) path.

    Feature indices are computed on the exact centerline; the smoother runs on
    a unit-spacing resampling whose index count differs.  Arc length is the
    stable mapping.  The resampled anchor is snapped back to the exact feature
    point so a fault corner is reproduced exactly.
    """
    original = clean_points(original)
    if len(original) < 2 or len(resampled) < 2:
        return []
    s_original = _arc_length(original)
    s_resampled = _arc_length(resampled)
    mapped: list[int] = []
    for value in indices:
        index = int(value)
        if not 0 < index < len(original) - 1:
            continue
        target = s_original[index]
        candidate = int(np.argmin(np.abs(s_resampled - target)))
        if 0 < candidate < len(resampled) - 1:
            resampled[candidate] = original[index]
            mapped.append(candidate)
    return sorted(set(mapped))


def smooth_fault_aware(
    points: np.ndarray,
    closed: bool,
    lam: float,
    features: PathFeatures,
    *,
    pin_open_endpoints: bool = True,
) -> tuple[np.ndarray, bool]:
    """Split at fault/junction anchors, pin anchors and freeze across faults.

    Returns (smoothed points, exact_when_frozen).
    """
    if features.fault_frozen:
        return clean_points(points), True
    original = clean_points(points)
    base = unit_resample(original, closed)
    if closed:
        return whittaker_smooth(base, lam, closed=True), False
    if len(base) < 4:
        return base, True

    n = len(base)
    anchors = _map_feature_indices(original, base, features.split_indices)
    cut_points = [0] + anchors + [n - 1]
    pieces: list[np.ndarray] = []
    for start, stop in zip(cut_points[:-1], cut_points[1:]):
        if stop - start < 1:
            continue
        piece = base[start : stop + 1]
        if len(piece) < 4:
            pieces.append(piece.copy())
            continue
        pins = {0, len(piece) - 1} if pin_open_endpoints or anchors else set()
        # Freeze a small stencil next to any fault/junction anchor.
        freeze = set()
        for anchor in anchors:
            if start <= anchor <= stop:
                offset = anchor - start
                for delta in range(-3, 4):
                    if 0 <= offset + delta < len(piece):
                        freeze.add(offset + delta)
        smoothed = whittaker_smooth(
            piece,
            lam,
            closed=False,
            pin_indices=sorted(pins),
            freeze_indices=sorted(freeze),
        )
        pieces.append(smoothed)
    merged = pieces[0]
    for piece in pieces[1:]:
        merged = np.vstack([merged, piece[1:]])
    return merged, False


__all__ = [
    "PathFeatures",
    "detect_features",
    "smooth_fault_aware",
    "smooth_global",
    "smooth_junction_pinned",
    "turning_angles",
    "unit_resample",
    "whittaker_smooth",
]
