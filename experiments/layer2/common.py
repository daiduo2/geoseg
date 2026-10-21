"""Shared harness for Layer 2 smoothing experiments.

This module is read-only for experiment agents.  It fixes the comparison
contract: the same curated panels, the same Layer 1 keep set, the same
centerline extraction (only for fitting; retention is still the exact closure),
the same geometric metrics, and the same render style.

It deliberately does not implement any smoother.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from geoseg import api


@dataclass
class Panel:
    panel_id: str
    labels: np.ndarray
    image_path: Path | None = None


@dataclass
class BoundaryPath:
    boundary_id: int
    labels_pair: tuple[int, int]
    points: np.ndarray  # (N, 2) x, y, ordered skeleton centerline
    closed: bool
    exact_length: int


def load_panels(
    manifest: Path | str = Path("validation/curated/manifest.jsonl"),
    root: Path | str = Path("validation/curated"),
    panel_ids: list[str] | None = None,
) -> list[Panel]:
    manifest = Path(manifest)
    root = Path(root)
    wanted = set(panel_ids or [])
    panels: list[Panel] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if wanted and record["panel_id"] not in wanted:
            continue
        labels = np.load(root / record["files"]["labels"])["labels"]
        panels.append(
            Panel(
                panel_id=record["panel_id"],
                labels=labels,
                image_path=root / record["files"]["image"],
            )
        )
    return panels


def _neighbours(point: tuple[int, int], coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
    row, col = point
    return [
        (row + dr, col + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if (dr or dc) and (row + dr, col + dc) in coords
    ]


def trace_skeleton(mask: np.ndarray) -> list[tuple[np.ndarray, bool]]:
    """Split a 1-px skeleton into ordered open/closed pixel chains."""
    coords = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    if not coords:
        return []
    adjacency = {point: _neighbours(point, coords) for point in coords}
    nodes = {point for point, nbrs in adjacency.items() if len(nbrs) != 2}
    used: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    paths: list[tuple[np.ndarray, bool]] = []

    def walk(start, first):
        pixels = [start, first]
        used.add((start, first))
        used.add((first, start))
        previous, current = start, first
        while current not in nodes:
            nxt = [point for point in adjacency[current] if point != previous]
            if not nxt:
                break
            step = nxt[0]
            if (current, step) in used:
                break
            used.add((current, step))
            used.add((step, current))
            pixels.append(step)
            previous, current = current, step
            if current == start:
                break
        closed = current == start and len(pixels) > 3
        return pixels, closed

    def add(pixels, closed):
        if len(pixels) < 2:
            return
        points = np.array([[col, row] for row, col in pixels], dtype=float)
        paths.append((points, closed))

    # Walk out of every branch/endpoint.  Each undirected edge is used once.
    for node in sorted(nodes):
        for neighbour in adjacency[node]:
            if (node, neighbour) not in used:
                add(*walk(node, neighbour))

    # Remaining components are pure cycles (all degree 2).
    remaining = {
        (point, neighbour)
        for point, nbrs in adjacency.items()
        for neighbour in nbrs
        if (point, neighbour) not in used
    }
    while remaining:
        start, first = next(iter(remaining))
        add(*walk(start, first))
        remaining = {
            (point, neighbour)
            for point, nbrs in adjacency.items()
            for neighbour in nbrs
            if (point, neighbour) not in used
        }
    return paths


def _adjacency(coords: set[tuple[int, int]]) -> dict[tuple[int, int], list[tuple[int, int]]]:
    return {point: _neighbours(point, coords) for point in coords}


def prune_skeleton_spurs(mask: np.ndarray, max_length: int = 5) -> np.ndarray:
    """Remove short endpoint-to-branch spurs created by skeletonization.

    Layer 1 keeps boundaries by exact pixel length, not by skeleton degree.
    At junctions, a 2-px-thick exact closure can produce 1-2 px skeleton
    branches that are not geological boundaries.  Prune only endpoint-to-branch
    walks no longer than ``max_length``; never prune a path between two
    junctions or a standalone short boundary.
    """
    coords = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    while True:
        adjacency = _adjacency(coords)
        endpoints = {point for point, nbrs in adjacency.items() if len(nbrs) == 1}
        branches = {point for point, nbrs in adjacency.items() if len(nbrs) > 2}
        if not endpoints or not branches:
            break
        remove: set[tuple[int, int]] = set()
        for endpoint in sorted(endpoints):
            path = [endpoint]
            previous = None
            current = endpoint
            while True:
                nxt = [point for point in adjacency[current] if point != previous]
                if len(nxt) != 1:
                    break
                previous, current = current, nxt[0]
                path.append(current)
                if len(path) - 1 > max_length:
                    break
            if len(path) - 1 <= max_length and current in branches:
                remove.update(path[:-1])
        if not remove:
            break
        coords -= remove

    pruned = np.zeros_like(mask, dtype=bool)
    if coords:
        rows, cols = np.array(sorted(coords)).T
        pruned[rows, cols] = True
    return pruned


def _drop_short_junction_edges(
    paths: list[BoundaryPath],
    *,
    max_length: int = 8,
    tolerance: float = 2.0,
) -> tuple[list[BoundaryPath], list[BoundaryPath]]:
    """Drop 1-8 px open edges whose two ends touch longer paths.

    These are skeleton cycles/clusters at a junction, not separate geological
    boundaries.  Closed loops and short paths with a free end (panel border or
    a genuine termination) are retained.  The dropped paths are returned for
    diagnostics; the exact boundary remains in Layer 1.
    """
    long_paths = [path for path in paths if path.closed or len(path.points) > max_length]
    kept: list[BoundaryPath] = []
    dropped: list[BoundaryPath] = []
    for path in paths:
        if path.closed or len(path.points) > max_length:
            kept.append(path)
            continue
        ends = (path.points[0], path.points[-1])
        hits = []
        for end in ends:
            hit = any(
                float(np.min(np.linalg.norm(other.points - end, axis=1))) <= tolerance
                for other in long_paths
                if other is not path
            )
            hits.append(hit)
        if all(hits):
            dropped.append(path)
        else:
            kept.append(path)
    return kept, dropped


def keep_paths(
    labels: np.ndarray,
    *,
    spur_length: int = 5,
    drop_junction_edges: bool = True,
) -> list[BoundaryPath]:
    """Exact Layer 1 keep objects, converted to centerlines for Layer 2 fitting.

    Centerline extraction prunes short skeleton spurs and, by default, drops
    short open junction-cluster edges.  Those pixels remain part of the Layer 1
    exact boundary; they are simply not treated as independent curves.
    """
    labels = np.asarray(labels)
    boundaries = api.extract_exact_boundaries(labels)
    decisions = api.decide_exact_boundaries(labels, boundaries)
    output: list[BoundaryPath] = []
    for decision in decisions:
        if decision.action != "keep":
            continue
        boundary = decision.boundary
        mask = np.zeros(labels.shape, dtype=bool)
        rows = np.rint(boundary.pixels[:, 1]).astype(int)
        cols = np.rint(boundary.pixels[:, 0]).astype(int)
        mask[rows, cols] = True
        skeleton = skeletonize(mask)
        skeleton = prune_skeleton_spurs(skeleton, max_length=spur_length)
        for points, closed in trace_skeleton(skeleton):
            output.append(
                BoundaryPath(
                    boundary_id=boundary.id,
                    labels_pair=boundary.labels_pair,
                    points=points,
                    closed=closed,
                    exact_length=boundary.length,
                )
            )
    if drop_junction_edges:
        output, _ = _drop_short_junction_edges(output)
    return output


def resample(points: np.ndarray, closed: bool, n: int | None = None) -> np.ndarray:
    """Arc-length resample a polyline (closed curves do not duplicate the endpoint)."""
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return points
    if closed:
        dense = np.vstack([points, points[:1]])
    else:
        dense = points
    diffs = np.diff(dense, axis=0)
    cumulative = np.concatenate([[0.0], np.cumsum(np.hypot(diffs[:, 0], diffs[:, 1]))])
    total = float(cumulative[-1])
    if total <= 0:
        return points
    count = n or max(2, int(round(total)))
    targets = np.linspace(0.0, total, count, endpoint=not closed)
    return np.column_stack(
        [
            np.interp(targets, cumulative, dense[:, 0]),
            np.interp(targets, cumulative, dense[:, 1]),
        ]
    )


def displacement(points: np.ndarray, exact_points: np.ndarray) -> dict[str, float]:
    tree = cKDTree(np.asarray(exact_points, dtype=float))
    distances, _ = tree.query(np.asarray(points, dtype=float), k=1)
    return {
        "mean": float(np.mean(distances)),
        "p95": float(np.percentile(distances, 95)),
        "max": float(np.max(distances)),
    }


def curvature(points: np.ndarray, closed: bool) -> dict[str, float]:
    dense = resample(points, closed, max(16, min(1000, len(points) * 2)))
    if len(dense) < 3:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    if closed:
        dense = np.vstack([dense, dense[:1]])
    dx = np.gradient(dense[:, 0])
    dy = np.gradient(dense[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denominator = np.power(dx * dx + dy * dy, 1.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.abs(dx * ddy - dy * ddx) / denominator
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def path_length(points: np.ndarray, closed: bool) -> float:
    dense = np.vstack([points, points[:1]]) if closed else points
    return float(np.hypot(*np.diff(dense, axis=0).T).sum())


def _segments_intersect(a, b, c, d) -> bool:
    def proper(t, u):
        return 0.01 < t < 0.99 and 0.01 < u < 0.99

    r = b - a
    s = d - c
    denominator = r[0] * s[1] - r[1] * s[0]
    if abs(denominator) < 1e-12:
        return False
    qp = c - a
    t = (qp[0] * s[1] - qp[1] * s[0]) / denominator
    u = (qp[0] * r[1] - qp[1] * r[0]) / denominator
    return proper(t, u)


def self_intersections(points: np.ndarray, closed: bool) -> int:
    dense = resample(points, closed, min(300, max(16, len(points))))
    n = len(dense)
    if n < 4:
        return 0
    count = 0
    for i in range(n if closed else n - 1):
        a, b = dense[i], dense[(i + 1) % n]
        for j in range(i + 2, n if closed else n - 1):
            if closed and i == 0 and j == n - 2:
                continue
            c, d = dense[j], dense[(j + 1) % n]
            if _segments_intersect(a, b, c, d):
                count += 1
    return count


def cross_intersections(paths: list[BoundaryPath], samples: int = 120) -> int:
    count = 0
    polylines = [resample(path.points, path.closed, min(samples, max(8, len(path.points)))) for path in paths]
    for i in range(len(polylines)):
        a = polylines[i]
        for j in range(i + 1, len(polylines)):
            if paths[i].labels_pair == paths[j].labels_pair:
                continue
            b = polylines[j]
            for k in range(len(a) - 1):
                for m in range(len(b) - 1):
                    if _segments_intersect(a[k], a[k + 1], b[m], b[m + 1]):
                        count += 1
    return count


def closed_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def boundary_metrics(
    smoothed: np.ndarray,
    exact: BoundaryPath,
    *,
    closed: bool | None = None,
) -> dict[str, float | int]:
    closed = exact.closed if closed is None else closed
    exact_dense = resample(exact.points, exact.closed, max(16, len(exact.points)))
    smooth_dense = resample(smoothed, closed, max(16, min(1000, len(smoothed) * 2)))
    move = displacement(smooth_dense, exact_dense)
    curv = curvature(smooth_dense, closed)
    exact_curv = curvature(exact_dense, exact.closed)
    endpoint_drift = 0.0
    if not closed and len(smooth_dense) >= 2:
        endpoint_drift = float(
            max(
                np.linalg.norm(smooth_dense[0] - exact_dense[0]),
                np.linalg.norm(smooth_dense[-1] - exact_dense[-1]),
            )
        )
    metrics: dict[str, float | int] = {
        "mean_disp": move["mean"],
        "p95_disp": move["p95"],
        "max_disp": move["max"],
        "max_curvature": curv["max"],
        "p95_curvature": curv["p95"],
        "exact_max_curvature": exact_curv["max"],
        "length_exact": path_length(exact_dense, exact.closed),
        "length_smooth": path_length(smooth_dense, closed),
        "endpoint_drift": endpoint_drift,
        "self_intersections": self_intersections(smooth_dense, closed),
    }
    if closed:
        metrics["area_exact"] = closed_area(exact_dense)
        metrics["area_smooth"] = closed_area(smooth_dense)
        metrics["area_change"] = metrics["area_smooth"] - metrics["area_exact"]
    return metrics


def recolour_labels(labels: np.ndarray) -> np.ndarray:
    background = np.array([28, 28, 32], dtype=np.uint8)
    palette = [
        (225, 100, 100), (70, 130, 210), (150, 100, 205), (235, 140, 60),
        (80, 190, 205), (200, 90, 170), (110, 170, 230), (170, 110, 75),
        (220, 130, 160), (100, 200, 170), (180, 180, 220), (150, 150, 150),
        (240, 160, 120), (60, 100, 180), (190, 140, 230), (120, 80, 150),
        (245, 180, 140), (140, 80, 50), (60, 160, 190), (210, 170, 200),
    ]
    out = np.empty((*labels.shape, 3), dtype=np.uint8)
    out[:] = background
    for label in np.unique(labels):
        value = int(label)
        if value:
            out[labels == value] = palette[(value - 1) % len(palette)]
    return out


def render_overlay(
    labels: np.ndarray,
    exact_paths: list[BoundaryPath],
    smoothed: list[np.ndarray],
    *,
    title: str = "",
    thickness: int = 3,
) -> Image.Image:
    """Recoloured mask + exact thin white + smoothed thick green."""
    image = Image.fromarray(recolour_labels(labels)).convert("RGB")
    draw = ImageDraw.Draw(image)
    for path in exact_paths:
        points = [tuple(point) for point in path.points]
        if len(points) >= 2:
            if path.closed:
                points.append(points[0])
            draw.line(points, fill=(255, 255, 255), width=1)
    for points in smoothed:
        points = [tuple(point) for point in np.asarray(points)]
        if len(points) < 2:
            continue
        draw.line(points, fill=(8, 8, 10), width=thickness + 3, joint="curve")
        draw.line(points, fill=(0, 255, 90), width=thickness, joint="curve")
    if title:
        font = ImageFont.load_default()
        draw.rectangle([0, 0, min(image.width, 520), 20], fill=(0, 0, 0))
        draw.text((6, 4), title, fill=(255, 230, 80), font=font)
    return image


def contact_sheet(rows: list[tuple[str, Image.Image, str]], out_path: Path, cell=(440, 280)) -> None:
    label_width = 190
    height = 32 + len(rows) * cell[1]
    canvas = Image.new("RGB", (label_width + cell[0], max(1, height)), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (panel_id, image, text) in enumerate(rows):
        y = 32 + index * cell[1]
        draw.text((6, y + 8), panel_id, fill=(235, 235, 235), font=font)
        draw.text((6, y + 28), text, fill=(170, 170, 170), font=font)
        fitted = image.copy()
        fitted.thumbnail(cell, Image.Resampling.LANCZOS)
        canvas.paste(fitted, (label_width, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
