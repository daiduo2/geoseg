"""Core algorithms for Shapes-primary segmentation editor.

Topology principle: single-connected regions are labels.
All user operations happen on Shapes (boundaries); Labels are computed.

All functions are pure (immutable) — they return new arrays, never mutate inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from skimage.draw import line, polygon_perimeter
from skimage.measure import find_contours


# ---------------------------------------------------------------------------
# shapes_to_labels  (Shapes → Topology → Labels)
# ---------------------------------------------------------------------------


def shapes_to_labels(
    shapes_data: list[np.ndarray],
    shape_types: list[str],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Rasterize boundary shapes and label connected regions.

    Args:
        shapes_data: List of (N, 2) arrays, each vertex is [y, x] (row, col).
        shape_types: List of shape type strings ('polygon', 'line', 'path', ...).
        image_shape: (H, W) of the output label array.

    Returns:
        (H, W) int32 array where 0 = boundary, 1..N = regions.
    """
    boundary = np.zeros(image_shape, dtype=bool)

    for vertices, shape_type in zip(shapes_data, shape_types):
        vertices = np.asarray(vertices)
        if len(vertices) < 2:
            continue

        rr: np.ndarray
        cc: np.ndarray

        if shape_type == "polygon":
            # Draw polygon perimeter segment by segment, including closing edge
            rr_list, cc_list = [], []
            n = len(vertices)
            for i in range(n):
                r0 = int(np.rint(vertices[i, 0]))
                c0 = int(np.rint(vertices[i, 1]))
                r1 = int(np.rint(vertices[(i + 1) % n, 0]))
                c1 = int(np.rint(vertices[(i + 1) % n, 1]))
                seg_rr, seg_cc = line(r0, c0, r1, c1)
                rr_list.extend(seg_rr.tolist())
                cc_list.extend(seg_cc.tolist())
            rr = np.array(rr_list, dtype=int)
            cc = np.array(cc_list, dtype=int)
        else:
            # Open line / path: draw segment by segment
            rr_list, cc_list = [], []
            for i in range(len(vertices) - 1):
                r0, c0 = int(vertices[i, 0]), int(vertices[i, 1])
                r1, c1 = int(vertices[i + 1, 0]), int(vertices[i + 1, 1])
                seg_rr, seg_cc = line(r0, c0, r1, c1)
                rr_list.extend(seg_rr.tolist())
                cc_list.extend(seg_cc.tolist())
            if not rr_list:
                continue
            rr = np.array(rr_list, dtype=int)
            cc = np.array(cc_list, dtype=int)

        valid = (
            (rr >= 0)
            & (rr < image_shape[0])
            & (cc >= 0)
            & (cc < image_shape[1])
        )
        boundary[rr[valid], cc[valid]] = True

    # Background = boundaries + image border
    background = boundary.copy()
    background[0, :] = True
    background[-1, :] = True
    background[:, 0] = True
    background[:, -1] = True

    # Label connected components of non-background
    fillable = ~background
    labels, _ = ndimage.label(fillable)

    result = labels.astype(np.int32)
    result[boundary] = 0
    return result


# ---------------------------------------------------------------------------
# labels_to_shapes  (Labels → Shapes)
# ---------------------------------------------------------------------------


def labels_to_shapes(labels: np.ndarray) -> list[np.ndarray]:
    """Extract boundary contours from a labels array.

    Args:
        labels: (H, W) int32 array where 0 = boundary/separator.

    Returns:
        List of (N, 2) contour arrays, each vertex is [y, x] (row, col).
        Suitable for napari Shapes layer data.
    """
    shapes: list[np.ndarray] = []
    unique = np.unique(labels)
    unique = unique[unique != 0]

    h, w = labels.shape
    for label_id in unique:
        mask = labels == label_id
        if not mask.any():
            continue
        # Pad to avoid open contours at image edges; find_contours then
        # returns fully closed contours inside the padded region.
        padded = np.pad(mask, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        contours = find_contours(padded, level=0.5)
        for contour in contours:
            # contour is (N, 2) of [row, col] = [y, x]
            if len(contour) >= 3:
                # Adjust back to original image coords
                contour[:, 0] -= 1
                contour[:, 1] -= 1
                contour[:, 0] = np.clip(contour[:, 0], 0, h - 1)
                contour[:, 1] = np.clip(contour[:, 1], 0, w - 1)
                shapes.append(contour)

    return shapes


# ---------------------------------------------------------------------------
# RegionProperties  (stable property binding independent of temporary IDs)
# ---------------------------------------------------------------------------


@dataclass
class RegionProperties:
    """Bind properties to regions via geometric fingerprint.

    Label IDs are temporary (recomputed on every shapes change).
    Properties persist via region fingerprint (centroid + area).
    """

    _props: dict[str, dict] = field(default_factory=dict)

    @staticmethod
    def _fingerprint(labels: np.ndarray, label_id: int) -> str:
        mask = labels == label_id
        if not mask.any():
            return ""
        cy, cx = ndimage.center_of_mass(mask)
        area = int(mask.sum())
        # Use 3-decimal precision + bbox aspect ratio to reduce collision risk
        yy, xx = np.where(mask)
        h = int(yy.max() - yy.min()) + 1 if len(yy) else 1
        w = int(xx.max() - xx.min()) + 1 if len(xx) else 1
        aspect = round(h / max(w, 1), 3)
        return f"{cy:.3f},{cx:.3f},{area},{aspect}"

    def get(self, labels: np.ndarray, label_id: int) -> dict | None:
        fp = self._fingerprint(labels, label_id)
        return self._props.get(fp) if fp else None

    def set(self, labels: np.ndarray, label_id: int, props: dict) -> None:
        fp = self._fingerprint(labels, label_id)
        if not fp:
            return
        if fp in self._props:
            # Collision guard: warn but allow overwrite
            import warnings
            warnings.warn(
                f"RegionProperties fingerprint collision: {fp}. "
                "Overwriting existing properties.",
                stacklevel=2,
            )
        self._props[fp] = props

    def remove(self, labels: np.ndarray, label_id: int) -> None:
        fp = self._fingerprint(labels, label_id)
        if fp and fp in self._props:
            del self._props[fp]

    def to_dict(self) -> dict[str, dict]:
        return dict(self._props)

    @classmethod
    def from_dict(cls, data: dict[str, dict]) -> RegionProperties:
        return cls(_props=dict(data))


# ---------------------------------------------------------------------------
# Legacy utilities (kept for optional line extension / advanced features)
# ---------------------------------------------------------------------------


def extend_trim_line_to_mask(
    mask: np.ndarray,
    p1: tuple[float, float],
    p2: tuple[float, float],
    *,
    oversample: int = 4,
    extend_ratio: float = 0.3,
    min_segment_length: float = 5.0,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Snap user-drawn line to mask boundary (extend + trim).

    Args:
        mask: (H, W) bool array. True = inside target region.
        p1: (x, y) start point (rough, need not be on boundary).
        p2: (x, y) end point (rough, need not be on boundary).
        oversample: Samples per pixel along the line.
        extend_ratio: How far beyond endpoints to sample.
        min_segment_length: Reject segments shorter than this.

    Returns:
        (b1, b2) where each is (x, y) on mask boundary, or None.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = np.hypot(dx, dy)
    if length < 1e-6:
        return None

    n_samples = max(int(length * oversample), 100)
    t = np.linspace(-extend_ratio, 1.0 + extend_ratio, n_samples)

    xs = p1[0] + t * dx
    ys = p1[1] + t * dy

    xi = np.clip(np.rint(xs).astype(int), 0, mask.shape[1] - 1)
    yi = np.clip(np.rint(ys).astype(int), 0, mask.shape[0] - 1)
    inside = mask[yi, xi]

    segments = _extract_contiguous_segments(inside)
    if not segments:
        return None

    best = _select_best_segment(segments, t)
    if best is None:
        return None

    s_start, s_end = best
    seg_len = np.hypot(
        xs[s_end - 1] - xs[s_start],
        ys[s_end - 1] - ys[s_start],
    )
    if seg_len < min_segment_length:
        return None

    b1 = _refine_boundary(mask, xs, ys, xi, yi, s_start, inside, going_in=True)
    b2 = _refine_boundary(mask, xs, ys, xi, yi, s_end - 1, inside, going_in=False)
    return (b1, b2)


def _extract_contiguous_segments(inside: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    in_seg = False
    start = 0
    for i, val in enumerate(inside):
        if val and not in_seg:
            in_seg = True
            start = i
        elif not val and in_seg:
            in_seg = False
            segments.append((start, i))
    if in_seg:
        segments.append((start, len(inside)))
    return segments


def _select_best_segment(
    segments: list[tuple[int, int]],
    t: np.ndarray,
) -> tuple[int, int] | None:
    mid_t = 0.5
    for start, end in segments:
        if t[start] <= mid_t <= t[end - 1]:
            return (start, end)
    return max(segments, key=lambda s: s[1] - s[0])


def _refine_boundary(
    mask: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    xi: np.ndarray,
    yi: np.ndarray,
    idx: int,
    inside: np.ndarray,
    going_in: bool,
) -> tuple[float, float]:
    if going_in:
        if idx > 0 and not inside[idx - 1]:
            return _interpolate_crossing(xs, ys, xi, yi, idx - 1, idx, mask)
    else:
        if idx + 1 < len(inside) and not inside[idx + 1]:
            return _interpolate_crossing(xs, ys, xi, yi, idx, idx + 1, mask)
    return (float(xs[idx]), float(ys[idx]))


def _interpolate_crossing(
    xs: np.ndarray,
    ys: np.ndarray,
    xi: np.ndarray,
    yi: np.ndarray,
    i0: int,
    i1: int,
    mask: np.ndarray,
) -> tuple[float, float]:
    x0, y0 = xs[i0], ys[i0]
    x1, y1 = xs[i1], ys[i1]
    for _ in range(4):
        mx = (x0 + x1) / 2
        my = (y0 + y1) / 2
        mxi = int(np.clip(np.rint(mx), 0, mask.shape[1] - 1))
        myi = int(np.clip(np.rint(my), 0, mask.shape[0] - 1))
        if mask[myi, mxi]:
            x1, y1 = mx, my
        else:
            x0, y0 = mx, my
    return ((x0 + x1) / 2, (y0 + y1) / 2)


# ---------------------------------------------------------------------------
# Unified endpoint snapping (threshold-circle + tangent extension)
# ---------------------------------------------------------------------------


def snap_line_endpoints(
    mask: np.ndarray,
    p1: tuple[float, float],
    p2: tuple[float, float],
    *,
    threshold: float = 25.0,
    min_length: float = 5.0,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Snap line endpoints to mask boundary if a boundary lies within threshold.

    For each endpoint, searches along the line's tangent direction (both ways)
    for the nearest mask boundary crossing. If the closest crossing is within
    ``threshold`` distance, the endpoint is snapped to that boundary point.

    This unifies "extend" (endpoint inside mask, boundary ahead) and
    "trim" (endpoint outside mask, boundary behind) into a single operation.

    Args:
        mask: (H, W) bool array. True = inside target region.
        p1, p2: (x, y) endpoints.
        threshold: Search radius (pixels). Only snap if boundary is within this
            distance from the endpoint along the line direction.
        min_length: Reject resulting segments shorter than this.

    Returns:
        (new_p1, new_p2) in (x, y), or None if no snapping occurred.
    """
    new_p1 = snap_endpoint_to_boundary(p1, p2, mask, threshold)
    new_p2 = snap_endpoint_to_boundary(p2, p1, mask, threshold)

    if new_p1 is None and new_p2 is None:
        return None

    b1 = new_p1 if new_p1 is not None else p1
    b2 = new_p2 if new_p2 is not None else p2

    seg_len = np.hypot(b2[0] - b1[0], b2[1] - b1[1])
    if seg_len < min_length:
        return None

    # Avoid duplicate (no change)
    if np.isclose(b1[0], p1[0], atol=0.5) and np.isclose(b1[1], p1[1], atol=0.5):
        if np.isclose(b2[0], p2[0], atol=0.5) and np.isclose(b2[1], p2[1], atol=0.5):
            return None

    return (b1, b2)


def snap_endpoint_to_boundary(
    endpoint: tuple[float, float],
    other: tuple[float, float],
    mask: np.ndarray,
    threshold: float,
) -> tuple[float, float] | None:
    """Snap a single endpoint to the nearest mask boundary along line direction.

    Searches both forward and backward along the line's tangent from the
    endpoint. Returns the closest boundary crossing within ``threshold``.
    """
    ex, ey = endpoint
    ox, oy = other
    dx, dy = ox - ex, oy - ey
    length = np.hypot(dx, dy)
    if length < 1e-6:
        return None

    vx, vy = dx / length, dy / length

    n = max(int(threshold * 4), 80)
    t = np.linspace(-threshold, threshold, n)
    xs = ex + t * vx
    ys = ey + t * vy

    xi = np.clip(np.rint(xs).astype(int), 0, mask.shape[1] - 1)
    yi = np.clip(np.rint(ys).astype(int), 0, mask.shape[0] - 1)
    inside = mask[yi, xi]

    if inside.all() or not inside.any():
        return None

    endpoint_idx = n // 2
    start_inside = inside[endpoint_idx]

    best_pt = None
    best_dist = float("inf")

    # Forward
    for i in range(endpoint_idx + 1, len(inside)):
        if inside[i] != start_inside:
            pt = _interpolate_crossing(xs, ys, xi, yi, i - 1, i, mask)
            dist = (pt[0] - ex) ** 2 + (pt[1] - ey) ** 2
            if dist < best_dist:
                best_dist = dist
                best_pt = pt
            break

    # Backward
    for i in range(endpoint_idx - 1, -1, -1):
        if inside[i] != start_inside:
            pt = _interpolate_crossing(xs, ys, xi, yi, i, i + 1, mask)
            dist = (pt[0] - ex) ** 2 + (pt[1] - ey) ** 2
            if dist < best_dist:
                best_dist = dist
                best_pt = pt
            break

    # Only snap if boundary is within threshold
    if best_pt is not None and best_dist > threshold ** 2:
        return None

    return best_pt


def snap_path_endpoints(
    mask: np.ndarray,
    vertices: np.ndarray,
    *,
    threshold: float = 25.0,
    min_segment_length: float = 5.0,
) -> np.ndarray | None:
    """Snap open-path endpoints to mask boundary.

    Only the first and last vertices are snapped; intermediate vertices are
    preserved. Each endpoint is snapped against the mask independently.

    Args:
        mask: (H, W) bool array. True = inside target region.
        vertices: (N, 2) array of [y, x] vertices.
        threshold: Max search distance for boundary snapping.
        min_segment_length: Reject if the snapped first or last segment becomes
            shorter than this.

    Returns:
        New (N, 2) vertex array if any endpoint changed, else None.
    """
    v = np.asarray(vertices, dtype=float)
    if len(v) < 2:
        return None

    changed = False

    # Snap first endpoint (direction: toward second vertex)
    new_start = snap_endpoint_to_boundary(
        (float(v[0, 1]), float(v[0, 0])),
        (float(v[1, 1]), float(v[1, 0])),
        mask,
        threshold,
    )
    if new_start is not None:
        v[0, 1] = new_start[0]
        v[0, 0] = new_start[1]
        changed = True

    # Snap last endpoint (direction: toward second-to-last vertex)
    new_end = snap_endpoint_to_boundary(
        (float(v[-1, 1]), float(v[-1, 0])),
        (float(v[-2, 1]), float(v[-2, 0])),
        mask,
        threshold,
    )
    if new_end is not None:
        v[-1, 1] = new_end[0]
        v[-1, 0] = new_end[1]
        changed = True

    if not changed:
        return None

    # Guard against collapsed end segments
    first_seg = np.hypot(v[1, 1] - v[0, 1], v[1, 0] - v[0, 0])
    last_seg = np.hypot(v[-1, 1] - v[-2, 1], v[-1, 0] - v[-2, 0])
    if first_seg < min_segment_length or last_seg < min_segment_length:
        return None

    return v


# ---------------------------------------------------------------------------
# Legacy: split / merge / rasterize (kept for backward compatibility)
# ---------------------------------------------------------------------------


def split_label_by_line(
    labels: np.ndarray,
    target_label: int,
    line_start: tuple[float, float],
    line_end: tuple[float, float],
    new_label: int,
    *,
    line_width: int = 1,
    min_area: int = 50,
) -> np.ndarray | None:
    """Split a region by drawing a cut line through it."""
    mask = labels == target_label
    if not mask.any():
        return None

    trimmed = extend_trim_line_to_mask(mask, line_start, line_end)
    if trimmed is None:
        return None

    b1, b2 = trimmed

    new_labels = labels.copy()
    _draw_cut_line(new_labels, target_label, b1, b2, line_width)

    target_mask = new_labels == target_label
    labeled, n = ndimage.label(target_mask)

    if n <= 1:
        return None

    return _assign_split_labels(new_labels, labeled, n, target_label, new_label, min_area)


def _draw_cut_line(
    labels: np.ndarray,
    target_label: int,
    line_start: tuple[float, float],
    line_end: tuple[float, float],
    line_width: int,
) -> None:
    y0, x0 = int(round(line_start[1])), int(round(line_start[0]))
    y1, x1 = int(round(line_end[1])), int(round(line_end[0]))
    rr, cc = line(y0, x0, y1, x1)

    valid = (
        (rr >= 0) & (rr < labels.shape[0]) & (cc >= 0) & (cc < labels.shape[1])
    )
    rr, cc = rr[valid], cc[valid]
    on_target = labels[rr, cc] == target_label
    labels[rr[on_target], cc[on_target]] = 0

    if line_width > 1:
        _widen_line(labels, rr[on_target], cc[on_target], target_label, line_width)


def _widen_line(
    labels: np.ndarray,
    rr: np.ndarray,
    cc: np.ndarray,
    target_label: int,
    width: int,
) -> None:
    radius = width // 2
    for r, c in zip(rr, cc, strict=False):
        r0 = max(0, r - radius)
        r1 = min(labels.shape[0], r + radius + 1)
        c0 = max(0, c - radius)
        c1 = min(labels.shape[1], c + radius + 1)
        patch = labels[r0:r1, c0:c1]
        patch[patch == target_label] = 0


def _assign_split_labels(
    labels: np.ndarray,
    labeled: np.ndarray,
    n_components: int,
    target_label: int,
    new_label: int,
    min_area: int,
) -> np.ndarray:
    component_ids = list(range(1, n_components + 1))
    sizes = [int((labeled == cid).sum()) for cid in component_ids]

    if n_components > 2:
        indexed = sorted(enumerate(sizes), key=lambda x: x[1], reverse=True)
        keep = {indexed[0][0] + 1, indexed[1][0] + 1}
        for cid in component_ids:
            if cid not in keep:
                frag_mask = labeled == cid
                labels[frag_mask] = 0

    remaining = [cid for cid in component_ids if (labeled == cid).any()]
    if len(remaining) < 2:
        return labels

    sizes_rem = [(cid, int((labeled == cid).sum())) for cid in remaining]
    sizes_rem.sort(key=lambda x: x[1], reverse=True)

    labels[labeled == sizes_rem[0][0]] = target_label
    labels[labeled == sizes_rem[1][0]] = new_label

    for cid, _ in sizes_rem[2:]:
        frag_mask = labeled == cid
        if frag_mask.sum() < min_area:
            labels[frag_mask] = sizes_rem[0][0]
        else:
            labels[frag_mask] = new_label

    return labels


def merge_labels(
    labels: np.ndarray,
    label_a: int,
    label_b: int,
) -> np.ndarray:
    """Merge label_b into label_a. Returns new labels array."""
    if label_a == label_b:
        return labels.copy()
    new_labels = labels.copy()
    new_labels[new_labels == label_b] = label_a
    return new_labels


def polygon_rasterize(
    labels: np.ndarray,
    vertices: np.ndarray,
    new_label: int,
) -> np.ndarray:
    """Fill polygon interior with new_label. Returns new labels array."""
    from skimage.draw import polygon

    new_labels = labels.copy()
    if len(vertices) < 3:
        return new_labels

    rr, cc = polygon(vertices[:, 0], vertices[:, 1], shape=labels.shape)
    new_labels[rr, cc] = new_label
    return new_labels


def fill_boundary_gaps(labels: np.ndarray) -> np.ndarray:
    """Fill interior label-0 boundary pixels by nearest-neighbour interpolation.

    After napari editing, label 0 represents thin boundary lines (separators)
    between labeled regions. Background regions that are connected to the image
    border are preserved. A 0-pixel is considered a boundary if it has at least
    two different non-zero labels among its 8-neighbours.

    Returns a new labels array with boundary pixels filled.
    """
    filled = labels.copy()

    # Maximum non-zero label in the 8-neighbourhood (center is ignored for 0s).
    max_neighbor = ndimage.maximum_filter(labels, size=3)

    # Minimum non-zero label in the 8-neighbourhood: replace 0 with a large
    # sentinel so minimum_filter ignores it.
    large = np.iinfo(labels.dtype).max
    nonzero_min = ndimage.minimum_filter(
        np.where(labels == 0, large, labels), size=3
    )

    # Boundary = 0 pixel with at least two different non-zero neighbours.
    boundary = (
        (labels == 0)
        & (max_neighbor > 0)
        & (nonzero_min < large)
        & (max_neighbor != nonzero_min)
    )

    if not boundary.any():
        return filled

    # Nearest-neighbour fill for boundary pixels
    non_zero = labels != 0
    _, indices = ndimage.distance_transform_edt(~non_zero, return_indices=True)
    rr, cc = np.where(boundary)
    filled[rr, cc] = filled[indices[0][rr, cc], indices[1][rr, cc]]
    return filled


def compute_label_diff(
    before: np.ndarray,
    after: np.ndarray,
) -> np.ndarray:
    """Compute diff between two label arrays for undo.

    Returns (N, 3) array of [y, x, before_label] for pixels that changed.
    """
    changed = before != after
    yy, xx = np.where(changed)
    return np.column_stack([yy, xx, before[changed]])


__all__ = [
    "RegionProperties",
    "compute_label_diff",
    "extend_trim_line_to_mask",
    "labels_to_shapes",
    "merge_labels",
    "polygon_rasterize",
    "shapes_to_labels",
    "snap_line_endpoints",
    "snap_path_endpoints",
    "split_label_by_line",
]
