"""Continue layer boundaries through occlusions such as white annotation boxes.

The functions here operate on a boundary trace ``y(x)``: one value per column,
NaN where the boundary is hidden. Two capabilities are provided:

- :func:`bridge_boundary_with_endpoint_tangents` fits endpoint tangents on both
  sides of a gap and fills it with a cubic Hermite segment, rejecting bridges
  whose overshoot or curvature change looks non-geological.
- :func:`repair_white_label_occluded_boundaries` finds compact white-backed
  labels, identifies the most supported label pair crossing each box, and
  writes that boundary through the box.

The tangent prior reuses the horizon engine's quintic fit, so both live in the
same package.
"""

from __future__ import annotations

import cv2
import numpy as np

from geoseg.modules.segment_engines.horizon.fitting import _fit_quintic


def detect_white_label_boxes(
    panel: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    """Detect compact white-backed labels such as C1/C2/C3 and Moho.

    Args:
        panel: (H, W, 3) uint8 RGB panel.

    Returns:
        List of ``(x0, y0, x1, y1)`` boxes, exclusive on the right/bottom.
    """
    near_white = (panel.min(axis=2) > 242).astype(np.uint8) * 255
    closed = cv2.morphologyEx(
        near_white,
        cv2.MORPH_CLOSE,
        np.ones((11, 11), dtype=np.uint8),
        iterations=1,
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        closed, connectivity=8
    )
    boxes: list[tuple[int, int, int, int]] = []
    for component in range(1, count):
        x, y, width, height, area = map(int, stats[component])
        x1, y1 = x + width, y + height
        white_fraction = float(near_white[y:y1, x:x1].mean() / 255.0)
        rectangularity = area / max(width * height, 1)
        if not (
            25 <= width <= 180
            and 16 <= height <= 105
            and rectangularity > 0.55
            and white_fraction > 0.35
            and y > 10
            and y1 < panel.shape[0] - 10
        ):
            continue
        boxes.append((x, y, x1, y1))
    return boxes


def fit_boundary_endpoint(
    trace: np.ndarray,
    *,
    edge_x: int,
    from_left: bool,
    max_samples: int = 24,
) -> tuple[float, float, int] | None:
    """Estimate boundary position and tangent at one side of an occlusion.

    Returns:
        ``(y_at_edge, slope, support)`` or ``None`` when fewer than five
        samples exist on the requested side.
    """
    valid_x = np.flatnonzero(~np.isnan(trace))
    if len(valid_x) < 5:
        return None
    side_x = valid_x[valid_x < edge_x] if from_left else valid_x[valid_x > edge_x]
    if len(side_x) < 5:
        return None
    side_x = side_x[-max_samples:] if from_left else side_x[:max_samples]
    side_y = trace[side_x]

    # Reuse the horizon engine's curvature prior to suppress pixel stair-steps
    # before estimating the local endpoint tangent.
    local_trace = np.full(side_x[-1] - side_x[0] + 1, np.nan, dtype=np.float64)
    local_trace[side_x - side_x[0]] = side_y
    if len(side_x) >= 10:
        local_y = _fit_quintic(local_trace, smoothness=0.02)[
            side_x - side_x[0]
        ]
    else:
        local_y = side_y

    centered_x = side_x.astype(np.float64) - float(edge_x)
    slope, intercept = np.polyfit(centered_x, local_y, deg=1)
    residual = local_y - (slope * centered_x + intercept)
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median)))
    if mad > 0:
        keep = np.abs(residual - median) <= 3.0 * 1.4826 * mad
        if keep.sum() >= 5:
            slope, intercept = np.polyfit(
                centered_x[keep], local_y[keep], deg=1
            )
    return float(intercept), float(np.clip(slope, -0.75, 0.75)), len(side_x)


def bridge_boundary_with_endpoint_tangents(
    trace: np.ndarray,
    *,
    start: int,
    stop: int,
) -> tuple[np.ndarray, dict[str, float | int]] | None:
    """Bridge ``[start, stop)`` with a tangent-constrained cubic Hermite curve.

    Args:
        trace: Boundary y(x) trace with NaN in the gap.
        start: First column of the gap.
        stop: One past the last column of the gap.

    Returns:
        ``(segment, metrics)`` where ``segment`` has length ``stop - start``,
        or ``None`` when either side lacks support or the bridge looks
        non-geological (overshoot or curvature change too large).
    """
    if stop <= start:
        return None
    left = fit_boundary_endpoint(trace, edge_x=start, from_left=True)
    right = fit_boundary_endpoint(trace, edge_x=stop - 1, from_left=False)
    if left is None or right is None:
        return None
    left_y, left_slope, left_support = left
    right_y, right_slope, right_support = right

    width = stop - start
    t = np.linspace(0.0, 1.0, width, dtype=np.float64)
    span = float(max(width - 1, 1))
    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2
    segment = (
        h00 * left_y
        + h10 * span * left_slope
        + h01 * right_y
        + h11 * span * right_slope
    )

    # Tangents can be noisy when an annotation touches a visible boundary.
    # Permit mild geological curvature, but reject spline-like excursions.
    allowance = max(2.0, 0.15 * abs(right_y - left_y))
    lower = min(left_y, right_y) - allowance
    upper = max(left_y, right_y) + allowance
    overshoot = float(
        max(np.max(lower - segment), np.max(segment - upper), 0.0)
    )
    max_curvature_change = (
        float(np.max(np.abs(np.diff(segment, n=2))))
        if len(segment) >= 3
        else 0.0
    )
    if overshoot > 0.5 or max_curvature_change > 0.35:
        return None
    return segment, {
        "left_endpoint_y": round(left_y, 3),
        "right_endpoint_y": round(right_y, 3),
        "left_endpoint_slope": round(left_slope, 5),
        "right_endpoint_slope": round(right_slope, 5),
        "left_endpoint_support": left_support,
        "right_endpoint_support": right_support,
        "overshoot_pixels": round(overshoot, 5),
        "max_curvature_change": round(max_curvature_change, 5),
    }


def repair_white_label_occluded_boundaries(
    labels: np.ndarray,
    panel: np.ndarray,
    body: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, object]],
]:
    """Continue label boundaries through white boxes with smooth tangents.

    Args:
        labels: (H, W) int label map; modified copy is returned.
        panel: (H, W, 3) uint8 RGB panel used to locate white boxes.
        body: (H, W) bool mask of the valid panel body; pixels outside are
            set to ``-1``.

    Returns:
        ``(labels, box_mask, change_mask, curve_mask, records)``.
    """
    result = labels.copy()
    box_mask = np.zeros(body.shape, dtype=bool)
    change_mask = np.zeros(body.shape, dtype=bool)
    curve_mask = np.zeros(body.shape, dtype=bool)
    records: list[dict[str, object]] = []

    for box in detect_white_label_boxes(panel):
        x0, y0, x1, y1 = box
        box_mask[y0:y1, x0:x1] = True
        pad_x = max(20, (x1 - x0) // 2)
        pad_y = max(12, (y1 - y0) // 2)
        sample_x0 = max(0, x0 - pad_x)
        sample_x1 = min(labels.shape[1], x1 + pad_x)
        sample_y0 = max(0, y0 - pad_y)
        sample_y1 = min(labels.shape[0] - 1, y1 + pad_y)

        side_counts: list[dict[tuple[int, int], int]] = []
        for side_x0, side_x1 in (
            (sample_x0, x0),
            (x1, sample_x1),
        ):
            counts: dict[tuple[int, int], int] = {}
            for x in range(side_x0, side_x1):
                column = result[sample_y0 : sample_y1 + 1, x]
                transitions = np.where(column[:-1] != column[1:])[0]
                for offset in transitions:
                    pair = (
                        int(column[offset]),
                        int(column[offset + 1]),
                    )
                    if pair[0] < 0 or pair[1] < 0:
                        continue
                    counts[pair] = counts.get(pair, 0) + 1
            side_counts.append(counts)

        common = [
            (min(side_counts[0][pair], side_counts[1][pair]), pair)
            for pair in side_counts[0].keys() & side_counts[1].keys()
            if min(side_counts[0][pair], side_counts[1][pair]) >= 5
        ]
        if not common:
            records.append({"bbox": list(box), "status": "no_common_boundary"})
            continue

        support, pair = max(common)
        trace = np.full(sample_x1 - sample_x0, np.nan, dtype=np.float64)
        side_valid = [0, 0]
        for side_index, (side_x0, side_x1) in enumerate(
            ((sample_x0, x0), (x1, sample_x1))
        ):
            for x in range(side_x0, side_x1):
                column = result[sample_y0 : sample_y1 + 1, x]
                transitions = np.where(
                    (column[:-1] == pair[0])
                    & (column[1:] == pair[1])
                )[0]
                if not len(transitions):
                    continue
                center_y = (y0 + y1) / 2.0
                offset = transitions[
                    np.argmin(np.abs(sample_y0 + transitions - center_y))
                ]
                trace[x - sample_x0] = sample_y0 + offset + 0.5
                side_valid[side_index] += 1

        if min(side_valid) < 5:
            records.append({"bbox": list(box), "status": "insufficient_side_samples"})
            continue

        bridge = bridge_boundary_with_endpoint_tangents(
            trace,
            start=x0 - sample_x0,
            stop=x1 - sample_x0,
        )
        if bridge is None:
            records.append(
                {"bbox": list(box), "status": "endpoint_bridge_failed"}
            )
            continue
        segment, bridge_metrics = bridge
        inside_fraction = float(((segment >= y0) & (segment < y1)).mean())
        median_y = float(np.median(segment))
        if not (
            inside_fraction >= 0.5
            and y0 <= median_y < y1
        ):
            records.append(
                {
                    "bbox": list(box),
                    "status": "boundary_does_not_cross_box",
                    "label_pair": list(pair),
                    "inside_fraction": round(inside_fraction, 4),
                }
            )
            continue

        before = result[y0:y1, x0:x1].copy()
        for local_x, x in enumerate(range(x0, x1)):
            boundary_y = int(round(segment[local_x]))
            boundary_y = int(np.clip(boundary_y, y0, y1 - 1))
            result[y0 : boundary_y + 1, x] = pair[0]
            result[boundary_y + 1 : y1, x] = pair[1]
            curve_mask[boundary_y, x] = True
        changed = before != result[y0:y1, x0:x1]
        change_mask[y0:y1, x0:x1] |= changed
        records.append(
            {
                "bbox": list(box),
                "status": "repaired",
                "label_pair": list(pair),
                "side_support": int(support),
                "boundary_y_min_median_max": [
                    round(float(segment.min()), 3),
                    round(median_y, 3),
                    round(float(segment.max()), 3),
                ],
                **bridge_metrics,
                "changed_pixels": int(changed.sum()),
            }
        )

    box_mask &= body
    change_mask &= body
    curve_mask &= body
    result[~body] = -1
    return result, box_mask, change_mask, curve_mask, records


__all__ = [
    "bridge_boundary_with_endpoint_tangents",
    "detect_white_label_boxes",
    "fit_boundary_endpoint",
    "repair_white_label_occluded_boundaries",
]
