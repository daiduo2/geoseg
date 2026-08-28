#!/usr/bin/env python3
"""Direct discrete-colorbar matching for Figures 6(c) and 7.

The initial label map is computed on the original/rectified pixels without
denoising, inpainting, median filtering, hole filling, or component merging.
Palette-relative residuals are saved before any reviewed local repair.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from geoseg.modules.text_removal import remove_text
from geoseg.modules.visual_audit.color_residual import (
    compute_palette_match_residuals,
    create_color_residual_overlay,
    find_high_deviation_regions,
)
from geoseg.preprocessing.absorption import absorb_artifacts
from geoseg.preprocessing.geometry import rectify_quadrilateral
from geoseg.modules.post_process.merge import filter_small_components
from geoseg.modules.segment_engines.horizon.fitting import _fit_quintic


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output/figures/j_tecto_2019_06_024/source"
OUTPUT = ROOT / "runs/j_tecto_2019_06_024_direct_colorbar"

# Visually reviewed on the full-resolution Fig. 6(c) clean-basemap audit.
# Coordinates are (x0, y0, x1, y1); the true dark surface layer lies above
# these boxes and is therefore protected from the targeted dark-RGB cleanup.
FIG6C_BLACK_VISUAL_ROIS = (
    (0, 220, 1150, 610),
    (0, 610, 1150, 755),
    (2880, 170, 3335, 800),
)
FIG6C_UPPER_SMOOTH_Y = 620


def _polygon_mask(shape: tuple[int, int], points: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 1)
    return mask.astype(bool)


def _body_mask(name: str, shape: tuple[int, int]) -> np.ndarray:
    if name == "fig6c":
        mask = np.zeros(shape, dtype=bool)
        mask[20:980, :3358] = True
        return mask
    mask = np.zeros(shape, dtype=bool)
    mask[3 : shape[0] - 3, 3 : shape[1] - 3] = True
    return mask


def _extract_discrete_palette(colorbar: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    """Extract the 20 colored blocks while excluding unit text to the right."""
    probe_y = min(30, colorbar.shape[0] - 1)
    row = colorbar[probe_y]
    saturation = row.max(axis=1).astype(np.int16) - row.min(axis=1).astype(np.int16)
    colored = (saturation > 18) & (row.mean(axis=1) < 245)

    runs: list[tuple[int, int]] = []
    start: int | None = None
    for x, value in enumerate(colored):
        if value and start is None:
            start = x
        if start is not None and (not value or x == len(colored) - 1):
            stop = x if not value else x + 1
            if stop - start >= 20:
                runs.append((start, stop))
            start = None

    if len(runs) != 20:
        raise ValueError(f"expected 20 discrete colorbar blocks, found {len(runs)}")

    y0, y1 = 25, min(46, colorbar.shape[0])
    palette = []
    for x0, x1 in runs:
        inner_x0 = x0 + min(4, max(0, (x1 - x0) // 4))
        inner_x1 = x1 - min(4, max(0, (x1 - x0) // 4))
        pixels = colorbar[y0:y1, inner_x0:inner_x1]
        palette.append(np.median(pixels.reshape(-1, 3), axis=0))
    return np.asarray(palette, dtype=np.uint8), [list(run) for run in runs]


def _reconstruct(labels: np.ndarray, palette: np.ndarray, body: np.ndarray) -> np.ndarray:
    result = np.full((*labels.shape, 3), 255, dtype=np.uint8)
    result[body] = palette[labels[body]]
    return result


def _hard_match_palette_rgb(
    image_rgb: np.ndarray,
    palette_rgb: np.ndarray,
    *,
    chunk_size: int = 200_000,
) -> dict[str, np.ndarray]:
    """Assign every pixel to its nearest exact colorbar RGB vector."""
    flat = image_rgb.reshape(-1, 3).astype(np.int16)
    palette = palette_rgb.astype(np.int16)
    labels = np.empty(len(flat), dtype=np.int16)
    residual = np.empty(len(flat), dtype=np.float32)
    margin = np.empty(len(flat), dtype=np.float32)
    for start in range(0, len(flat), chunk_size):
        stop = min(start + chunk_size, len(flat))
        distances_sq = (
            (flat[start:stop, None, :] - palette[None, :, :]).astype(np.int32)
            ** 2
        ).sum(axis=2)
        nearest_two = np.partition(distances_sq, kth=1, axis=1)[:, :2]
        nearest_two.sort(axis=1)
        labels[start:stop] = distances_sq.argmin(axis=1)
        residual[start:stop] = np.sqrt(nearest_two[:, 0])
        margin[start:stop] = (
            np.sqrt(nearest_two[:, 1]) - residual[start:stop]
        )
    shape = image_rgb.shape[:2]
    return {
        "labels": labels.reshape(shape),
        "rgb_residual": residual.reshape(shape),
        "margin_rgb": margin.reshape(shape),
    }


def _save_fig7_hard_match_only(
    out: Path,
    panel: np.ndarray,
    colorbar: np.ndarray,
    palette: np.ndarray,
    block_runs: list[list[int]],
) -> dict[str, object]:
    """Save the colorbar RGB assignment stage with no post-processing."""
    raw = out / "raw_match"
    raw.mkdir(parents=True, exist_ok=True)
    body = _body_mask("fig7", panel.shape[:2])
    matched = _hard_match_palette_rgb(panel, palette)
    labels = matched["labels"]
    labels[~body] = -1
    reconstruction = _reconstruct(labels, palette, body)
    vector_residual = np.abs(
        panel.astype(np.int16) - reconstruction.astype(np.int16)
    ).astype(np.uint8)
    vector_residual[~body] = 0

    Image.fromarray(panel).save(out / "01_panel.png")
    Image.fromarray(colorbar).save(out / "02_colorbar.png")
    swatch = np.repeat(palette[np.newaxis, :, :], 40, axis=0)
    swatch = np.repeat(swatch, 48, axis=1)
    Image.fromarray(swatch).save(out / "03_exact_palette.png")
    Image.fromarray(body.astype(np.uint8) * 255).save(out / "04_body_mask.png")
    Image.fromarray(reconstruction).save(raw / "reconstructed.png")
    Image.fromarray(vector_residual).save(raw / "rgb_vector_residual_abs.png")
    Image.fromarray(
        np.concatenate((panel, reconstruction, vector_residual), axis=1)
    ).save(raw / "comparison_panel_hard_match_rgb_residual.png")
    np.savez_compressed(
        raw / "labels_and_residuals.npz",
        labels=labels,
        body_mask=body,
        palette_rgb=palette,
        rgb_residual=matched["rgb_residual"],
        margin_rgb=matched["margin_rgb"],
        rgb_vector_residual_abs=vector_residual,
    )

    values = matched["rgb_residual"][body]
    report = {
        "method": "hard_nearest_colorbar_rgb_no_pre_or_post_processing",
        "assignment_metric": "euclidean_rgb_vector_distance",
        "palette_rgb": palette.tolist(),
        "colorbar_block_runs": block_runs,
        "pre_segmentation_smoothing": False,
        "post_processing": False,
        "rgb_residual": {
            "mean": round(float(values.mean()), 4),
            "p90": round(float(np.percentile(values, 90)), 4),
            "p95": round(float(np.percentile(values, 95)), 4),
            "p99": round(float(np.percentile(values, 99)), 4),
            "max": round(float(values.max()), 4),
        },
    }
    (raw / "residual_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "shape": list(panel.shape),
        "body_fraction": round(float(body.mean()), 5),
        "raw_match": str(raw.relative_to(ROOT)),
        "method": report["method"],
        "post_processing": False,
    }


def _extract_frequency_palette(
    panel: np.ndarray,
    body: np.ndarray,
    *,
    n_colors: int = 14,
    min_rgb_distance: float = 20.0,
    exclude_near_black_neutral: bool = True,
) -> tuple[np.ndarray, list[dict[str, object]], list[dict[str, object]]]:
    """Select distinct, frequent fill-color modes from original panel pixels."""
    colors, counts = np.unique(panel[body], axis=0, return_counts=True)
    order = np.argsort(counts)[::-1]
    total = int(body.sum())
    ranked = [
        {
            "rgb": colors[index].tolist(),
            "count": int(counts[index]),
            "fraction": round(float(counts[index] / total), 8),
        }
        for index in order[:100]
    ]

    selected_indices: list[int] = []
    for index in order:
        color = colors[index].astype(np.float32)
        mean = float(color.mean())
        channel_range = float(color.max() - color.min())
        if mean > 245 or (
            exclude_near_black_neutral and mean < 60 and channel_range < 60
        ):
            continue
        if any(
            np.linalg.norm(color - colors[chosen].astype(np.float32))
            < min_rgb_distance
            for chosen in selected_indices
        ):
            continue
        selected_indices.append(int(index))
        if len(selected_indices) == n_colors:
            break
    if len(selected_indices) != n_colors:
        raise ValueError(
            f"expected {n_colors} distinct frequency colors, "
            f"found {len(selected_indices)}"
        )

    palette = colors[selected_indices].astype(np.uint8)
    selected = [
        {
            "rank": rank,
            "rgb": palette[rank - 1].tolist(),
            "count": int(counts[index]),
            "fraction": round(float(counts[index] / total), 8),
        }
        for rank, index in enumerate(selected_indices, start=1)
    ]
    return palette, selected, ranked


def _save_fig7_frequency_match_only(
    out: Path,
    panel: np.ndarray,
) -> dict[str, object]:
    """Count dominant panel colors, then RGB-match with no post-processing."""
    match_out = out / "frequency_match"
    match_out.mkdir(parents=True, exist_ok=True)
    body = _body_mask("fig7", panel.shape[:2])
    palette, selected, ranked = _extract_frequency_palette(panel, body)
    matched = _hard_match_palette_rgb(panel, palette)
    labels = matched["labels"]
    labels[~body] = -1
    reconstruction = _reconstruct(labels, palette, body)
    vector_residual = np.abs(
        panel.astype(np.int16) - reconstruction.astype(np.int16)
    ).astype(np.uint8)
    vector_residual[~body] = 0

    Image.fromarray(panel).save(out / "01_panel.png")
    swatch = np.repeat(palette[np.newaxis, :, :], 50, axis=0)
    swatch = np.repeat(swatch, 70, axis=1)
    Image.fromarray(swatch).save(out / "02_frequency_palette.png")
    Image.fromarray(body.astype(np.uint8) * 255).save(out / "03_body_mask.png")
    Image.fromarray(reconstruction).save(match_out / "reconstructed.png")
    Image.fromarray(vector_residual).save(
        match_out / "rgb_vector_residual_abs.png"
    )
    Image.fromarray(
        np.concatenate((panel, reconstruction, vector_residual), axis=1)
    ).save(match_out / "comparison_panel_frequency_match_rgb_residual.png")
    np.savez_compressed(
        match_out / "labels_and_residuals.npz",
        labels=labels,
        body_mask=body,
        palette_rgb=palette,
        rgb_residual=matched["rgb_residual"],
        margin_rgb=matched["margin_rgb"],
        rgb_vector_residual_abs=vector_residual,
    )
    report = {
        "method": "distinct_frequent_rgb_modes_then_nearest_rgb_match",
        "pre_segmentation_smoothing": False,
        "post_processing": False,
        "selection": {
            "n_colors": 14,
            "minimum_rgb_distance": 20.0,
            "excluded_near_white": "mean RGB > 245",
            "excluded_near_black_neutral": (
                "mean RGB < 60 and channel range < 60"
            ),
        },
        "selected_palette": selected,
        "top_100_exact_rgb_counts": ranked,
    }
    (match_out / "frequency_palette_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "shape": list(panel.shape),
        "body_fraction": round(float(body.mean()), 5),
        "palette_rgb": palette.tolist(),
        "selected_palette": selected,
        "match_output": str(match_out.relative_to(ROOT)),
        "method": report["method"],
        "post_processing": False,
    }


def _detect_white_label_boxes(
    panel: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    """Detect compact white-backed labels such as C1/C2/C3 and Moho."""
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


def _fit_boundary_endpoint(
    trace: np.ndarray,
    *,
    edge_x: int,
    from_left: bool,
    max_samples: int = 24,
) -> tuple[float, float, int] | None:
    """Estimate boundary position and tangent at one side of an occlusion."""
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


def _bridge_boundary_with_endpoint_tangents(
    trace: np.ndarray,
    *,
    start: int,
    stop: int,
) -> tuple[np.ndarray, dict[str, float | int]] | None:
    """Bridge ``[start, stop)`` with a tangent-constrained cubic Hermite curve."""
    if stop <= start:
        return None
    left = _fit_boundary_endpoint(trace, edge_x=start, from_left=True)
    right = _fit_boundary_endpoint(trace, edge_x=stop - 1, from_left=False)
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


def _repair_white_label_occluded_boundaries(
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
    """Continue label boundaries through white boxes with smooth tangents."""
    result = labels.copy()
    box_mask = np.zeros(body.shape, dtype=bool)
    change_mask = np.zeros(body.shape, dtype=bool)
    curve_mask = np.zeros(body.shape, dtype=bool)
    records: list[dict[str, object]] = []

    for box in _detect_white_label_boxes(panel):
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

        bridge = _bridge_boundary_with_endpoint_tangents(
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


def _save_white_label_boundary_roi_montage(
    path: Path,
    panel: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    records: list[dict[str, object]],
) -> None:
    """Save enlarged source/before/after/difference rows for repaired boxes."""
    rows: list[np.ndarray] = []
    for record in records:
        if record["status"] != "repaired":
            continue
        x0, y0, x1, y1 = record["bbox"]
        pad = 18
        crop_x0 = max(0, x0 - pad)
        crop_y0 = max(0, y0 - pad)
        crop_x1 = min(panel.shape[1], x1 + pad)
        crop_y1 = min(panel.shape[0], y1 + pad)
        crops = [
            image[crop_y0:crop_y1, crop_x0:crop_x1]
            for image in (panel, before, after)
        ]
        difference = np.abs(
            crops[1].astype(np.int16) - crops[2].astype(np.int16)
        ).astype(np.uint8)
        row_image = np.concatenate((*crops, difference), axis=1)
        row_image = cv2.resize(
            row_image,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_NEAREST,
        )
        title_height = 28
        row = np.full(
            (row_image.shape[0] + title_height, row_image.shape[1], 3),
            255,
            dtype=np.uint8,
        )
        row[title_height:] = row_image
        cv2.putText(
            row,
            f"bbox={x0},{y0},{x1},{y1} pair={record['label_pair']}",
            (6, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        rows.append(row)
    if not rows:
        return
    canvas = np.full(
        (sum(row.shape[0] for row in rows), max(row.shape[1] for row in rows), 3),
        255,
        dtype=np.uint8,
    )
    y = 0
    for row in rows:
        canvas[y : y + row.shape[0], : row.shape[1]] = row
        y += row.shape[0]
    Image.fromarray(canvas).save(path)


def _save_fig6c_frequency_annotation_cleanup(
    out: Path,
    panel: np.ndarray,
) -> dict[str, object]:
    """Reprocess Fig. 6(c) from observed frequency colors and clean labels."""
    match_out = out / "frequency_match"
    cleanup_out = out / "annotation_cleanup"
    match_out.mkdir(parents=True, exist_ok=True)
    cleanup_out.mkdir(parents=True, exist_ok=True)
    body = _body_mask("fig6c", panel.shape[:2])
    occlusion = np.zeros(body.shape, dtype=bool)
    occlusion[835:975, 1830:3358] = True
    palette_sample = body & ~occlusion
    palette, selected, ranked = _extract_frequency_palette(
        panel,
        palette_sample,
        n_colors=18,
        min_rgb_distance=15.0,
        exclude_near_black_neutral=False,
    )
    matched = _hard_match_palette_rgb(panel, palette)
    raw_labels = matched["labels"].astype(np.int16)

    # Recover only the region physically hidden by the embedded colorbar. Use
    # the current Fig. 7 clean labels as a calibrated corresponding sample;
    # if unavailable, nearest labels around the occlusion remain the fallback.
    fig7_clean_path = (
        OUTPUT / "fig7" / "annotation_cleanup" / "clean_basemap.png"
    )
    if fig7_clean_path.exists():
        fig7_clean = np.asarray(Image.open(fig7_clean_path).convert("RGB"))
        yy, xx = np.where(occlusion)
        source_x = np.rint(0.625 * xx + 137.5).astype(np.int32)
        source_y = np.rint((yy - 10.0) / 1.5).astype(np.int32)
        inside = (
            (source_x >= 0)
            & (source_x < fig7_clean.shape[1])
            & (source_y >= 0)
            & (source_y < fig7_clean.shape[0])
        )
        projected = np.full((len(xx), 3), 255, dtype=np.uint8)
        projected[inside] = fig7_clean[source_y[inside], source_x[inside]]
        distances = (
            (
                projected[:, None, :].astype(np.int16)
                - palette[None, :, :].astype(np.int16)
            ).astype(np.int32)
            ** 2
        ).sum(axis=2)
        valid_projection = inside
        raw_labels[yy[valid_projection], xx[valid_projection]] = (
            distances[valid_projection].argmin(axis=1).astype(np.int16)
        )
        occlusion_policy = "projected_from_fig7_clean_basemap"
    else:
        # Keep the colorbar from leaking into the output even when the
        # calibrated Fig. 7 source is unavailable (for example, an isolated
        # Fig. 6(c) run).  This is only a conservative fallback; normal runs
        # use the calibrated projection above.
        valid = body & ~occlusion
        _, nearest = ndimage.distance_transform_edt(
            ~valid, return_indices=True
        )
        yy, xx = np.where(occlusion)
        raw_labels[yy, xx] = raw_labels[
            nearest[0][yy, xx], nearest[1][yy, xx]
        ]
        occlusion_policy = "nearest_label_fallback"

    safe = np.clip(raw_labels, 0, len(palette) - 1).astype(np.uint8)
    # Keep the categorical cleanup local to annotation strokes.  A large
    # window crosses the thin orange/pink boundary at the bottom of Fig. 6(c)
    # and expands the pink floor, increasing the RGB residual there.
    median_labels = cv2.medianBlur(safe, 5).astype(np.int32)
    shifted = np.where(body, median_labels + 1, 0)
    shifted = filter_small_components(
        shifted, min_area_ratio=0.0003, fill="nearest"
    )
    cleaned_labels = np.where(body, shifted - 1, -1).astype(np.int16)
    source_residual = np.linalg.norm(
        panel.astype(np.float32) - palette[safe].astype(np.float32), axis=2
    )
    component_mask = np.zeros(body.shape, dtype=bool)
    removed_components: list[dict[str, object]] = []
    for label_id in range(len(palette)):
        components, count = ndimage.label(
            (cleaned_labels == label_id) & body & ~occlusion
        )
        for component_id in range(1, count + 1):
            component = components == component_id
            area = int(component.sum())
            if not 100 <= area < 15_000:
                continue
            mean_residual = float(source_residual[component].mean())
            if mean_residual <= 25.0:
                continue
            component_mask |= component
            yy, xx = np.where(component)
            removed_components.append(
                {
                    "label": label_id,
                    "area": area,
                    "mean_rgb_residual": round(mean_residual, 4),
                    "bbox": [
                        int(xx.min()),
                        int(yy.min()),
                        int(xx.max() + 1),
                        int(yy.max() + 1),
                    ],
                }
            )
    valid_source = (
        body
        & ~component_mask
        & ~occlusion
        & (source_residual <= 25.0)
    )
    _, nearest = ndimage.distance_transform_edt(
        ~valid_source, return_indices=True
    )
    yy, xx = np.where(component_mask)
    cleaned_labels[yy, xx] = safe[
        nearest[0][yy, xx], nearest[1][yy, xx]
    ]

    # Black boundary strokes are neutral in the source RGB, whereas the real
    # darkest velocity layers are chromatic purple/blue.  Remove only those
    # neutral source strokes, including a one-pixel antialiasing fringe, and
    # fill them from the nearest valid categorical labels on either side.
    source_max = panel.max(axis=2).astype(np.int16)
    source_min = panel.min(axis=2).astype(np.int16)
    source_gray = cv2.cvtColor(panel, cv2.COLOR_RGB2GRAY)
    black_boundary_mask = (
        (source_gray < 90)
        & ((source_max - source_min) < 18)
        & body
        & ~occlusion
    )
    black_boundary_mask = ndimage.binary_dilation(
        black_boundary_mask,
        structure=np.ones((3, 3), dtype=bool),
        iterations=1,
    ) & body & ~occlusion
    # Do not seed the fill from adjacent antialiased stroke pixels that were
    # also assigned to the dark-purple class.  Low-residual palette cores are
    # reliable representatives of the actual layers on both sides.
    valid_source = (
        body
        & ~black_boundary_mask
        & ~occlusion
        & (source_residual <= 25.0)
    )
    _, nearest = ndimage.distance_transform_edt(
        ~valid_source, return_indices=True
    )
    yy, xx = np.where(black_boundary_mask)
    cleaned_labels[yy, xx] = safe[
        nearest[0][yy, xx], nearest[1][yy, xx]
    ]

    # The projected Fig. 7 fill can carry isolated annotation remnants into
    # the hidden colorbar footprint.  Regularize categorical IDs only inside
    # that synthetic fill; source RGB matching elsewhere remains untouched.
    projected_fill_labels = cv2.medianBlur(
        np.clip(cleaned_labels, 0, len(palette) - 1).astype(np.uint8), 15
    )
    cleaned_labels[occlusion] = projected_fill_labels[occlusion]

    # Fig. 7 benefits from a 15x15 categorical median.  Reuse that stronger
    # smoothing only above the thin warm/deep layers in Fig. 6(c), so text
    # remnants are suppressed without expanding the bottom pink floor.
    pre_upper_smoothing_labels = cleaned_labels.copy()
    upper_smoothed_labels = cv2.medianBlur(
        np.clip(cleaned_labels, 0, len(palette) - 1).astype(np.uint8), 15
    )
    upper_smoothing_region = np.zeros(body.shape, dtype=bool)
    upper_smoothing_region[:FIG6C_UPPER_SMOOTH_Y] = True
    upper_smoothing_region &= body & ~occlusion
    cleaned_labels[upper_smoothing_region] = upper_smoothed_labels[
        upper_smoothing_region
    ]
    upper_smoothing_change_mask = (
        (cleaned_labels != pre_upper_smoothing_labels)
        & upper_smoothing_region
    )
    pre_visual_roi_labels = cleaned_labels.copy()

    # Iterative visual-ROI cleanup, following the project's prior text-noise
    # workflow: the ROI stabilizes the visual selection, while the actual mask
    # remains RGB/label-specific inside each box.  Class 12 is the frequency
    # palette's near-black purple, which is spurious in these reviewed ROIs.
    visual_roi_mask = np.zeros(body.shape, dtype=bool)
    for x0, y0, x1, y1 in FIG6C_BLACK_VISUAL_ROIS:
        visual_roi_mask[y0:y1, x0:x1] = True
    visual_roi_mask &= body & ~occlusion
    visual_black_mask = visual_roi_mask & (cleaned_labels == 12)
    visual_roi_iterations = 0
    for _ in range(3):
        target = visual_roi_mask & (cleaned_labels == 12)
        if not target.any():
            break
        trusted = (
            body
            & ~occlusion
            & (source_residual <= 25.0)
            & (safe != 12)
        )
        _, nearest = ndimage.distance_transform_edt(
            ~trusted, return_indices=True
        )
        yy, xx = np.where(target)
        cleaned_labels[yy, xx] = safe[
            nearest[0][yy, xx], nearest[1][yy, xx]
        ]
        visual_roi_iterations += 1
    pre_white_box_repair_labels = cleaned_labels.copy()
    (
        cleaned_labels,
        white_label_box_mask,
        white_label_boundary_change_mask,
        white_label_fitted_curve_mask,
        white_label_boundary_records,
    ) = _repair_white_label_occluded_boundaries(
        cleaned_labels, panel, body
    )
    cleaned_labels[~body] = -1

    raw_labels[~body] = -1
    raw_reconstruction = _reconstruct(raw_labels, palette, body)
    pre_upper_smoothing_clean = _reconstruct(
        pre_upper_smoothing_labels, palette, body
    )
    pre_visual_roi_clean = _reconstruct(
        pre_visual_roi_labels, palette, body
    )
    pre_white_box_repair_clean = _reconstruct(
        pre_white_box_repair_labels, palette, body
    )
    clean = _reconstruct(cleaned_labels, palette, body)
    raw_residual = np.abs(
        panel.astype(np.int16) - raw_reconstruction.astype(np.int16)
    ).astype(np.uint8)
    clean_residual = np.abs(
        panel.astype(np.int16) - clean.astype(np.int16)
    ).astype(np.uint8)
    raw_residual[~body] = 0
    clean_residual[~body] = 0
    label_change_mask = (cleaned_labels != raw_labels) & body

    Image.fromarray(panel).save(out / "01_panel.png")
    Image.fromarray(palette_sample.astype(np.uint8) * 255).save(
        out / "04_palette_sample_mask.png"
    )
    Image.fromarray(occlusion.astype(np.uint8) * 255).save(
        out / "05_source_occlusion_mask.png"
    )
    swatch = np.repeat(palette[np.newaxis, :, :], 50, axis=0)
    swatch = np.repeat(swatch, 55, axis=1)
    Image.fromarray(swatch).save(out / "02_frequency_palette.png")
    Image.fromarray(raw_reconstruction).save(match_out / "reconstructed.png")
    Image.fromarray(raw_residual).save(
        match_out / "rgb_vector_residual_abs.png"
    )
    Image.fromarray(
        np.concatenate((panel, raw_reconstruction, raw_residual), axis=1)
    ).save(match_out / "comparison_panel_frequency_match_rgb_residual.png")
    Image.fromarray(clean).save(cleanup_out / "clean_basemap.png")
    Image.fromarray(clean_residual).save(
        cleanup_out / "rgb_vector_residual_abs.png"
    )
    Image.fromarray(
        np.concatenate((panel, clean, clean_residual), axis=1)
    ).save(cleanup_out / "comparison_panel_clean_rgb_residual.png")
    Image.fromarray(label_change_mask.astype(np.uint8) * 255).save(
        cleanup_out / "label_cleanup_change_mask.png"
    )
    Image.fromarray(component_mask.astype(np.uint8) * 255).save(
        cleanup_out / "residual_component_annotation_mask.png"
    )
    Image.fromarray(black_boundary_mask.astype(np.uint8) * 255).save(
        cleanup_out / "black_boundary_annotation_mask.png"
    )
    Image.fromarray(visual_roi_mask.astype(np.uint8) * 255).save(
        cleanup_out / "visual_black_roi_mask.png"
    )
    Image.fromarray(visual_black_mask.astype(np.uint8) * 255).save(
        cleanup_out / "visual_black_absorption_mask.png"
    )
    Image.fromarray(upper_smoothing_change_mask.astype(np.uint8) * 255).save(
        cleanup_out / "upper_text_smoothing_change_mask.png"
    )
    Image.fromarray(white_label_box_mask.astype(np.uint8) * 255).save(
        cleanup_out / "white_label_box_mask.png"
    )
    Image.fromarray(
        white_label_boundary_change_mask.astype(np.uint8) * 255
    ).save(cleanup_out / "white_label_boundary_repair_mask.png")
    white_label_overlay = pre_white_box_repair_clean.copy()
    for record in white_label_boundary_records:
        x0, y0, x1, y1 = record["bbox"]
        color = (255, 0, 255) if record["status"] == "repaired" else (128, 128, 128)
        cv2.rectangle(white_label_overlay, (x0, y0), (x1 - 1, y1 - 1), color, 3)
    white_label_overlay[white_label_fitted_curve_mask] = (0, 255, 255)
    Image.fromarray(white_label_overlay).save(
        cleanup_out / "white_label_boundary_repair_overlay.png"
    )
    Image.fromarray(
        np.concatenate(
            (pre_white_box_repair_clean, clean, white_label_overlay), axis=1
        )
    ).save(cleanup_out / "comparison_white_label_boundary_repair.png")
    _save_white_label_boundary_roi_montage(
        cleanup_out / "white_label_boundary_roi_montage.png",
        panel,
        pre_white_box_repair_clean,
        clean,
        white_label_boundary_records,
    )
    Image.fromarray(
        np.concatenate(
            (pre_upper_smoothing_clean, clean, clean_residual), axis=1
        )
    ).save(cleanup_out / "comparison_upper_text_smoothing.png")
    visual_roi_overlay = pre_visual_roi_clean.copy()
    for x0, y0, x1, y1 in FIG6C_BLACK_VISUAL_ROIS:
        cv2.rectangle(
            visual_roi_overlay,
            (x0, y0),
            (x1 - 1, y1 - 1),
            (255, 0, 255),
            4,
        )
    Image.fromarray(
        np.concatenate(
            (pre_visual_roi_clean, visual_roi_overlay, clean), axis=1
        )
    ).save(cleanup_out / "comparison_visual_black_roi.png")
    np.savez_compressed(
        cleanup_out / "labels.npz",
        labels=cleaned_labels,
        raw_frequency_labels=raw_labels,
        body_mask=body,
        palette_rgb=palette,
        source_occlusion_mask=occlusion,
        residual_component_annotation_mask=component_mask,
        black_boundary_annotation_mask=black_boundary_mask,
        visual_black_roi_mask=visual_roi_mask,
        visual_black_absorption_mask=visual_black_mask,
        upper_text_smoothing_change_mask=upper_smoothing_change_mask,
        white_label_box_mask=white_label_box_mask,
        white_label_boundary_repair_mask=white_label_boundary_change_mask,
        white_label_fitted_curve_mask=white_label_fitted_curve_mask,
        label_cleanup_change_mask=label_change_mask,
        rgb_vector_residual_abs=clean_residual,
    )
    report = {
        "method": [
            "distinct_frequent_rgb_modes_excluding_embedded_colorbar",
            "direct_rgb_nearest_match",
            "projected_fig7_clean_fill_for_colorbar_occlusion",
            "5x5_categorical_median_on_label_ids",
            "remove_components_below_0.03pct_with_nearest_label_fill",
            "remove_components_area_100_to_14999_and_mean_rgb_residual_above_25",
            "remove_neutral_black_source_strokes_with_nearest_label_fill",
            "15x15_categorical_median_inside_projected_occlusion_only",
            "15x15_categorical_median_above_y620_only",
            "iterative_visual_roi_absorption_of_palette_rgb_40_32_73",
            "quintic_denoised_endpoint_tangent_bridge_through_white_label_boxes",
        ],
        "palette_rgb": palette.tolist(),
        "selected_palette": selected,
        "top_100_exact_rgb_counts": ranked,
        "occlusion_policy": occlusion_policy,
        "pre_segmentation_rgb_smoothing": False,
        "manual_rectangular_masks": False,
        "visual_roi_guidance": True,
        "rectangular_roi_pixels_are_not_removed_directly": True,
        "label_change_fraction_of_body": round(
            float(label_change_mask.sum() / body.sum()), 6
        ),
        "residual_component_mask_fraction_of_body": round(
            float(component_mask.sum() / body.sum()), 6
        ),
        "black_boundary_mask_fraction_of_body": round(
            float(black_boundary_mask.sum() / body.sum()), 6
        ),
        "visual_black_rois_xyxy": [
            list(roi) for roi in FIG6C_BLACK_VISUAL_ROIS
        ],
        "visual_black_absorption_fraction_of_body": round(
            float(visual_black_mask.sum() / body.sum()), 6
        ),
        "upper_text_smoothing_limit_y": FIG6C_UPPER_SMOOTH_Y,
        "upper_text_smoothing_change_fraction_of_body": round(
            float(upper_smoothing_change_mask.sum() / body.sum()), 6
        ),
        "visual_roi_iterations": visual_roi_iterations,
        "white_label_boundary_repairs": white_label_boundary_records,
        "removed_components": removed_components,
    }
    (cleanup_out / "cleanup_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "shape": list(panel.shape),
        "body_fraction": round(float(body.mean()), 5),
        "palette_rgb": palette.tolist(),
        "selected_palette": selected,
        "match_output": str(match_out.relative_to(ROOT)),
        "annotation_cleanup": str(cleanup_out.relative_to(ROOT)),
        "cleanup_report": report,
    }


def _save_fig7_frequency_annotation_cleanup(
    out: Path,
    panel: np.ndarray,
) -> dict[str, object]:
    """Remove annotation components after the frozen frequency RGB match."""
    match_report = _save_fig7_frequency_match_only(out, panel)
    match_out = out / "frequency_match"
    cleanup_out = out / "annotation_cleanup"
    cleanup_out.mkdir(parents=True, exist_ok=True)
    data = np.load(match_out / "labels_and_residuals.npz")
    raw_labels = data["labels"].astype(np.int16)
    body = data["body_mask"].astype(bool)
    palette = data["palette_rgb"]
    safe_labels = np.clip(raw_labels, 0, len(palette) - 1)

    # Operate only on hard-matched categorical IDs. The panel RGB is not
    # smoothed, so the frequency palette and first-stage match stay frozen.
    median_labels = cv2.medianBlur(
        safe_labels.astype(np.uint8), 15
    ).astype(np.int32)
    shifted = np.where(body, median_labels + 1, 0)
    shifted = filter_small_components(
        shifted, min_area_ratio=0.001, fill="nearest"
    )
    cleaned_labels = np.where(body, shifted - 1, -1).astype(np.int16)

    source_residual = np.linalg.norm(
        panel.astype(np.float32) - palette[safe_labels].astype(np.float32),
        axis=2,
    )
    component_mask = np.zeros(body.shape, dtype=bool)
    removed_components: list[dict[str, object]] = []
    for label_id in range(len(palette)):
        components, count = ndimage.label(
            (cleaned_labels == label_id) & body
        )
        for component_id in range(1, count + 1):
            component = components == component_id
            area = int(component.sum())
            if not 100 <= area < 15_000:
                continue
            mean_residual = float(source_residual[component].mean())
            if mean_residual <= 25.0:
                continue
            yy, xx = np.where(component)
            component_mask |= component
            removed_components.append(
                {
                    "label": label_id,
                    "area": area,
                    "mean_rgb_residual": round(mean_residual, 4),
                    "bbox": [
                        int(xx.min()),
                        int(yy.min()),
                        int(xx.max() + 1),
                        int(yy.max() + 1),
                    ],
                }
            )

    valid_source = body & ~component_mask
    _, nearest = ndimage.distance_transform_edt(
        ~valid_source, return_indices=True
    )
    yy, xx = np.where(component_mask)
    cleaned_labels[yy, xx] = cleaned_labels[
        nearest[0][yy, xx], nearest[1][yy, xx]
    ]
    pre_white_box_repair_labels = cleaned_labels.copy()
    (
        cleaned_labels,
        white_label_box_mask,
        white_label_boundary_change_mask,
        white_label_fitted_curve_mask,
        white_label_boundary_records,
    ) = _repair_white_label_occluded_boundaries(
        cleaned_labels, panel, body
    )
    cleaned_labels[~body] = -1

    pre_white_box_repair_clean = _reconstruct(
        pre_white_box_repair_labels, palette, body
    )
    clean = _reconstruct(cleaned_labels, palette, body)
    rgb_vector_residual = np.abs(
        panel.astype(np.int16) - clean.astype(np.int16)
    ).astype(np.uint8)
    rgb_vector_residual[~body] = 0
    label_change_mask = (cleaned_labels != raw_labels) & body
    mask_overlay = panel.copy()
    mask_overlay[component_mask] = (
        0.45 * mask_overlay[component_mask]
        + 0.55 * np.array([255, 0, 255])
    ).astype(np.uint8)

    Image.fromarray(label_change_mask.astype(np.uint8) * 255).save(
        cleanup_out / "label_cleanup_change_mask.png"
    )
    Image.fromarray(component_mask.astype(np.uint8) * 255).save(
        cleanup_out / "residual_component_annotation_mask.png"
    )
    Image.fromarray(mask_overlay).save(
        cleanup_out / "residual_component_annotation_mask_overlay.png"
    )
    Image.fromarray(white_label_box_mask.astype(np.uint8) * 255).save(
        cleanup_out / "white_label_box_mask.png"
    )
    Image.fromarray(
        white_label_boundary_change_mask.astype(np.uint8) * 255
    ).save(cleanup_out / "white_label_boundary_repair_mask.png")
    white_label_overlay = pre_white_box_repair_clean.copy()
    for record in white_label_boundary_records:
        x0, y0, x1, y1 = record["bbox"]
        color = (
            (255, 0, 255)
            if record["status"] == "repaired"
            else (128, 128, 128)
        )
        cv2.rectangle(
            white_label_overlay,
            (x0, y0),
            (x1 - 1, y1 - 1),
            color,
            3,
        )
    white_label_overlay[white_label_fitted_curve_mask] = (0, 255, 255)
    Image.fromarray(white_label_overlay).save(
        cleanup_out / "white_label_boundary_repair_overlay.png"
    )
    Image.fromarray(
        np.concatenate(
            (pre_white_box_repair_clean, clean, white_label_overlay), axis=1
        )
    ).save(cleanup_out / "comparison_white_label_boundary_repair.png")
    _save_white_label_boundary_roi_montage(
        cleanup_out / "white_label_boundary_roi_montage.png",
        panel,
        pre_white_box_repair_clean,
        clean,
        white_label_boundary_records,
    )
    Image.fromarray(clean).save(cleanup_out / "clean_basemap.png")
    Image.fromarray(rgb_vector_residual).save(
        cleanup_out / "rgb_vector_residual_abs.png"
    )
    Image.fromarray(
        np.concatenate((panel, clean, rgb_vector_residual), axis=1)
    ).save(cleanup_out / "comparison_panel_clean_rgb_residual.png")
    np.savez_compressed(
        cleanup_out / "labels.npz",
        labels=cleaned_labels,
        raw_frequency_labels=raw_labels,
        body_mask=body,
        palette_rgb=palette,
        label_cleanup_change_mask=label_change_mask,
        residual_component_annotation_mask=component_mask,
        white_label_box_mask=white_label_box_mask,
        white_label_boundary_repair_mask=white_label_boundary_change_mask,
        white_label_fitted_curve_mask=white_label_fitted_curve_mask,
        rgb_vector_residual_abs=rgb_vector_residual,
    )
    cleanup_report = {
        "method": [
            "frozen_frequency_palette_rgb_match",
            "15x15_categorical_median_on_label_ids",
            "remove_components_below_0.1pct_with_nearest_label_fill",
            "remove_components_area_100_to_14999_and_mean_rgb_residual_above_25",
            "quintic_denoised_endpoint_tangent_bridge_through_white_label_boxes",
        ],
        "source_rgb_smoothing": False,
        "manual_rectangular_masks": False,
        "label_change_fraction_of_body": round(
            float(label_change_mask.sum() / body.sum()), 6
        ),
        "residual_component_mask_fraction_of_body": round(
            float(component_mask.sum() / body.sum()), 6
        ),
        "removed_components": removed_components,
        "white_label_boundary_repairs": white_label_boundary_records,
    }
    (cleanup_out / "cleanup_report.json").write_text(
        json.dumps(cleanup_report, indent=2) + "\n", encoding="utf-8"
    )
    return {
        **match_report,
        "annotation_cleanup": str(cleanup_out.relative_to(ROOT)),
        "cleanup_report": cleanup_report,
    }


def _detect_fig6c_orange_dashes(panel: np.ndarray) -> np.ndarray:
    """Detect all four orange guide lines as separate dash components."""
    warm = (
        (panel[:, :, 0] > 180)
        & (panel[:, :, 1] < 195)
        & (panel[:, :, 2] < 115)
    )
    mask = np.zeros(panel.shape[:2], dtype=np.uint8)
    for x0, x1 in (
        (352, 380),
        (738, 770),
        (1764, 1792),
        (2234, 2257),
    ):
        roi = warm[20:735, x0:x1].astype(np.uint8)
        count, components, stats, _ = cv2.connectedComponentsWithStats(
            roi, connectivity=8
        )
        for component in range(1, count):
            x, y, width, height, area = stats[component]
            thin_dash = (
                2 <= width <= 10 and 15 <= height <= 24 and area >= 28
            )
            thick_dash = (
                5 <= width <= 15 and 30 <= height <= 45 and area >= 180
            )
            if thin_dash or thick_dash:
                selected = components == component
                mask[20:735, x0:x1][selected] = 1

    # Cover only anti-aliased rims around each dash; gaps between dashes stay
    # untouched and continue to supply local categorical memberships.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.dilate(mask, kernel, iterations=1).astype(bool)


def _reviewed_repair_masks(
    name: str,
    panel: np.ndarray,
    delta_e: np.ndarray,
    body: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build paper-specific masks from residual evidence and visual review."""
    _, detected = remove_text(
        panel,
        brightness_thresh=0,
        max_area=1200 if name == "fig6c" else 900,
        lap_threshold=28 if name == "fig6c" else 32,
        dilate_iter=1,
        inpaint_radius=3,
        residual_lap_threshold=18 if name == "fig6c" else 20,
        residual_median_ksize=51 if name == "fig6c" else 41,
    )
    detected = detected.astype(bool)
    threshold = 24.0 if name == "fig6c" else 38.0
    mask = ((delta_e > threshold) | (detected & (delta_e > 10.0))) & body

    # One-pixel expansion covers anti-aliased annotation rims without turning
    # the repair into a broad rectangular fill.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    occlusion = np.zeros(body.shape, dtype=bool)
    nearest_fill = np.zeros(body.shape, dtype=bool)
    orange_dashes = np.zeros(body.shape, dtype=bool)

    if name == "fig6c":
        # The inset colorbar hides the original model. Mark it unknown rather
        # than fabricate lower-mantle geometry beneath a large rectangle.
        occlusion[835:975, 1870:3358] = True

        # These exact-palette annotations evade residual thresholding. Detect
        # their actual dash silhouettes instead of masking full-height strips.
        orange_dashes = _detect_fig6c_orange_dashes(panel)
        mask |= orange_dashes
        nearest_fill |= orange_dashes

        # White-backed crustal labels require their complete small backing,
        # not only the glyph pixels found by the residual detector.
        label_boxes = [
            (960, 110, 1070, 170), (925, 175, 1030, 235),
            (920, 235, 1030, 295), (1110, 335, 1215, 395),
            (1080, 405, 1250, 470), (1080, 535, 1215, 610),
            (1030, 640, 1325, 720), (1660, 80, 1780, 145),
            (2730, 125, 2825, 185), (2740, 200, 2845, 260),
            (2500, 255, 2610, 315), (2260, 310, 2375, 370),
            (2510, 375, 2700, 440),
        ]
        for x0, y0, x1, y1 in label_boxes:
            mask[y0:y1, x0:x1] = True
            nearest_fill[y0:y1, x0:x1] = True
    # Fig. 7's previous manually reviewed polygons were calibrated against an
    # obsolete rectification. Its new baseline therefore uses only masks
    # derived from current-panel residual evidence until a fresh visual audit.

    mask &= body & ~occlusion
    nearest_fill &= mask
    orange_dashes &= mask
    return mask, occlusion, nearest_fill, orange_dashes


def _inpaint_label_memberships(
    labels: np.ndarray, mask: np.ndarray, n_classes: int
) -> np.ndarray:
    """Extend categorical memberships through a reviewed local mask."""
    if not mask.any():
        return labels.copy()
    mask_u8 = mask.astype(np.uint8) * 255
    scores = np.empty((n_classes, *labels.shape), dtype=np.uint8)
    for label in range(n_classes):
        membership = (labels == label).astype(np.uint8) * 255
        scores[label] = cv2.inpaint(
            membership, mask_u8, 5, cv2.INPAINT_NS
        )
    result = labels.copy()
    result[mask] = np.argmax(scores[:, mask], axis=0)
    return result


def _regularize_fig7_labels_after_match(
    labels: np.ndarray, body: np.ndarray
) -> np.ndarray:
    """Denoise hard-matched category IDs without filtering source RGB."""
    working = np.where(body, labels, 0).astype(np.uint8)
    working = cv2.medianBlur(working, 15).astype(np.int32)

    # filter_small_components reserves zero for background, while palette
    # class zero is a real velocity class. Shift all valid classes by one so
    # every colorbar class participates in component filtering and filling.
    shifted = np.where(body, working + 1, 0)
    shifted = filter_small_components(
        shifted, min_area_ratio=0.001, fill="nearest"
    )
    result = (shifted - 1).astype(np.int16)
    result[~body] = -1
    return result


def _recover_fig6c_occlusion_from_fig7(
    labels: np.ndarray, occlusion: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project the corresponding Fig. 7 labels into Fig. 6(c)'s colorbar area."""
    source_path = OUTPUT / "fig7" / "final" / "labels.npz"
    if not source_path.exists():
        return labels, occlusion.copy()
    fig7_labels = np.load(source_path)["labels"]
    projection_source = cv2.medianBlur(
        np.clip(fig7_labels, 0, 19).astype(np.uint8), 9
    )
    yy, xx = np.where(occlusion)
    source_x = np.rint(0.625 * xx + 137.5).astype(np.int32)
    source_y = np.rint((yy - 10.0) / 1.5).astype(np.int32)
    inside = (
        (source_x >= 0)
        & (source_x < fig7_labels.shape[1])
        & (source_y >= 0)
        & (source_y < fig7_labels.shape[0])
    )
    sampled = np.full(len(xx), -1, dtype=np.int16)
    sampled[inside] = projection_source[source_y[inside], source_x[inside]]
    recovered = sampled >= 0
    result = labels.copy()
    result[yy[recovered], xx[recovered]] = sampled[recovered]
    remaining = occlusion.copy()
    remaining[yy[recovered], xx[recovered]] = False
    return result, remaining


def _restore_fig6c_deepest_floor(
    labels: np.ndarray, body: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Replace the bottom-connected pale frame artifact with class 19."""
    result = labels.copy()
    repaired = np.zeros(body.shape, dtype=bool)
    for x in range(labels.shape[1]):
        body_y = np.where(body[:, x])[0]
        if not len(body_y):
            continue
        floor = int(body_y[-1])
        lower = np.where((labels[: floor + 1, x] >= 16))[0]
        lower = lower[lower >= int(0.65 * labels.shape[0])]
        if not len(lower):
            continue
        last_warm = int(lower[-1])
        class19 = np.where(labels[: last_warm + 1, x] == 19)[0]
        recent_class19 = class19[class19 >= max(0, last_warm - 24)]
        start = int(recent_class19[0]) if len(recent_class19) else last_warm + 1
        if start <= floor:
            repaired[start : floor + 1, x] = labels[start : floor + 1, x] != 19
            result[start : floor + 1, x] = 19
    return result, repaired


def _save_final(
    name: str,
    out: Path,
    panel: np.ndarray,
    palette: np.ndarray,
    raw_labels: np.ndarray,
    raw_delta_e: np.ndarray,
    body: np.ndarray,
) -> dict[str, object]:
    final = out / "final"
    final.mkdir(parents=True, exist_ok=True)
    matched_labels = raw_labels.copy()
    if name == "fig7":
        matched_labels = _regularize_fig7_labels_after_match(
            matched_labels, body
        )
    repair_mask, occlusion, nearest_fill, orange_dashes = _reviewed_repair_masks(
        name, panel, raw_delta_e, body
    )
    raw_reconstruction = _reconstruct(raw_labels, palette, body)
    denoised_reconstruction = _reconstruct(matched_labels, palette, body)
    if name == "fig7":
        # Keep the first rerun auditable: hard palette assignment followed by
        # categorical denoising only. Artifact repairs will be calibrated from
        # the new residual map instead of inherited from the obsolete baseline.
        labels = matched_labels.copy()
    else:
        label_candidates = _inpaint_label_memberships(
            matched_labels, repair_mask, len(palette)
        )
        nearest_repaired = denoised_reconstruction.copy()
        nearest_repaired[nearest_fill] = palette[label_candidates[nearest_fill]]
        repaired_rgb = absorb_artifacts(
            nearest_repaired,
            repair_mask & ~nearest_fill,
            inpaint_radius=5,
            dilate_iters=0,
            method="NS",
        )
        rematched = compute_palette_match_residuals(repaired_rgb, palette)
        labels = rematched["labels"].astype(np.int16)
    remaining_occlusion = occlusion.copy()
    deepest_floor_repair = np.zeros(body.shape, dtype=bool)
    if name == "fig6c":
        labels, remaining_occlusion = _recover_fig6c_occlusion_from_fig7(
            labels, occlusion
        )
        labels, deepest_floor_repair = _restore_fig6c_deepest_floor(
            labels, body
        )
    valid_body = body & ~remaining_occlusion
    labels[~valid_body] = -1
    clean = _reconstruct(labels, palette, valid_body)

    rgb_vector_residual = np.abs(
        panel.astype(np.int16) - clean.astype(np.int16)
    ).astype(np.uint8)
    rgb_vector_residual[~valid_body] = 0
    original_clean_residual = np.linalg.norm(
        rgb_vector_residual.astype(np.float32), axis=2
    )
    unexplained = original_clean_residual.copy()
    explained = repair_mask | occlusion | deepest_floor_repair | ~body
    unexplained[explained] = 0
    changed = (labels != matched_labels) & valid_body

    Image.fromarray(repair_mask.astype(np.uint8) * 255).save(
        final / "reviewed_artifact_mask.png"
    )
    Image.fromarray(occlusion.astype(np.uint8) * 255).save(
        final / "source_occlusion_mask.png"
    )
    Image.fromarray(remaining_occlusion.astype(np.uint8) * 255).save(
        final / "unknown_occlusion_mask.png"
    )
    Image.fromarray(deepest_floor_repair.astype(np.uint8) * 255).save(
        final / "deepest_floor_repair_mask.png"
    )
    Image.fromarray(orange_dashes.astype(np.uint8) * 255).save(
        final / "orange_dash_repair_mask.png"
    )
    Image.fromarray(clean).save(final / "clean_basemap.png")
    Image.fromarray(denoised_reconstruction).save(
        final / "hard_match_denoised.png"
    )
    Image.fromarray(rgb_vector_residual).save(
        final / "rgb_vector_residual_abs.png"
    )
    Image.fromarray(
        _residual_overlay(
            original_clean_residual, panel, np.maximum(labels, 0), valid_body
        )
    ).save(final / "original_vs_clean_rgb_residual.png")
    Image.fromarray(
        _residual_overlay(unexplained, panel, np.maximum(labels, 0), valid_body)
    ).save(final / "unexplained_residual_audit.png")
    raw_comparison = np.concatenate((raw_reconstruction, clean), axis=1)
    Image.fromarray(raw_comparison).save(final / "raw_match_before_after.png")
    comparison = np.concatenate((panel, clean), axis=1)
    Image.fromarray(comparison).save(final / "before_after.png")
    before_after_residual = np.concatenate(
        (panel, clean, rgb_vector_residual), axis=1
    )
    Image.fromarray(before_after_residual).save(
        final / "before_after_residual.png"
    )
    raw_before_after_residual = np.concatenate(
        (raw_reconstruction, clean, rgb_vector_residual), axis=1
    )
    Image.fromarray(raw_before_after_residual).save(
        final / "raw_match_before_after_residual.png"
    )
    total_comparison = np.concatenate(
        (panel, clean, rgb_vector_residual), axis=1
    )
    Image.fromarray(total_comparison).save(
        final / "comparison_original_clean_rgb_vector_residual.png"
    )
    Image.fromarray(total_comparison).save(final / "comparison.png")
    np.savez_compressed(
        final / "labels.npz",
        labels=labels,
        body_mask=valid_body,
        palette_rgb=palette,
        repair_mask=repair_mask,
        nearest_fill_mask=nearest_fill,
        orange_dash_repair_mask=orange_dashes,
        source_occlusion_mask=occlusion,
        unknown_occlusion_mask=remaining_occlusion,
        deepest_floor_repair_mask=deepest_floor_repair,
        rgb_vector_residual_abs=rgb_vector_residual,
        original_vs_clean_rgb_residual=original_clean_residual,
        unexplained_rgb_residual=unexplained,
    )

    valid_values = unexplained[valid_body]
    report = {
        "repair_order": (
            [
                "hard_nearest_palette_assignment_on_original_pixels",
                "15x15_categorical_median_on_matched_label_ids",
                "remove_components_below_0.1pct_with_nearest_label_fill",
                "rgb_vector_residual_audit",
            ]
            if name == "fig7"
            else [
                "direct_palette_match_on_original_pixels",
                "rgb_lab_residual_audit",
                "reviewed_local_artifact_mask",
                "categorical_membership_inpaint_for_reviewed_large_symbols",
                "local_ns_inpaint_on_palette_reconstruction",
                "rematch_to_same_exact_palette",
            ]
        ),
        "pre_segmentation_smoothing": False,
        "repair_mask_fraction_of_body": round(
            float(repair_mask.sum() / max(body.sum(), 1)), 5
        ),
        "unknown_occlusion_fraction_of_body": round(
            float(remaining_occlusion.sum() / max(body.sum(), 1)), 5
        ),
        "changed_label_fraction_of_valid_body": round(
            float(changed.sum() / max(valid_body.sum(), 1)), 5
        ),
        "unexplained_rgb_residual": {
            "mean": round(float(valid_values.mean()), 4),
            "p95": round(float(np.percentile(valid_values, 95)), 4),
            "p99": round(float(np.percentile(valid_values, 99)), 4),
        },
        "unknown_policy": (
            "embedded colorbar recovered from calibrated corresponding Fig. 7 labels"
            if name == "fig6c"
            else "none"
        ),
        "fig6c_occlusion_projection": (
            {"x7": "0.625*x6+137.5", "y7": "(y6-10)/1.5"}
            if name == "fig6c"
            else None
        ),
        "deepest_floor_repair_pixels": int(deepest_floor_repair.sum()),
        "rgb_vector_residual": {
            "definition": "abs(original_rgb-clean_rgb) within valid body",
            "mean_abs_per_channel": {
                channel: round(float(rgb_vector_residual[valid_body, index].mean()), 4)
                for index, channel in enumerate(("R", "G", "B"))
            },
            "mean_l2": round(float(original_clean_residual[valid_body].mean()), 4),
            "p90_l2": round(
                float(np.percentile(original_clean_residual[valid_body], 90)), 4
            ),
            "max_l2": round(float(original_clean_residual[valid_body].max()), 4),
        },
    }
    (final / "audit_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _residual_overlay(
    residual: np.ndarray,
    panel: np.ndarray,
    labels: np.ndarray,
    body: np.ndarray,
    candidates: list[dict] | None = None,
) -> np.ndarray:
    audit_labels = labels + 1
    audit_labels[~body] = 0
    clipped = residual.copy()
    values = clipped[body]
    scale = float(np.percentile(values, 99)) if values.size else 1.0
    clipped = np.clip(clipped, 0, max(scale, 1e-6))
    clipped[~body] = 0
    return create_color_residual_overlay(
        clipped, panel, audit_labels, candidates=candidates, alpha=0.65
    )


def _save_panel(
    name: str,
    panel: np.ndarray,
    colorbar: np.ndarray,
    palette: np.ndarray,
    block_runs: list[list[int]],
) -> dict[str, object]:
    out = OUTPUT / name
    raw = out / "raw_match"
    raw.mkdir(parents=True, exist_ok=True)
    body = _body_mask(name, panel.shape[:2])

    matched = compute_palette_match_residuals(panel, palette)
    labels = matched["labels"]
    labels[~body] = -1
    reconstruction = _reconstruct(labels, palette, body)

    audit_labels = labels + 1
    audit_labels[~body] = 0
    candidates = find_high_deviation_regions(
        audit_labels,
        matched["delta_e"],
        min_area_frac=0.0002,
        deviation_percentile=95.0,
    )

    Image.fromarray(panel).save(out / "01_panel.png")
    Image.fromarray(colorbar).save(out / "02_colorbar.png")
    swatch = np.repeat(palette[np.newaxis, :, :], 40, axis=0)
    swatch = np.repeat(swatch, 48, axis=1)
    Image.fromarray(swatch).save(out / "03_exact_palette.png")
    Image.fromarray(body.astype(np.uint8) * 255).save(out / "04_body_mask.png")

    np.savez_compressed(
        raw / "labels_and_residuals.npz",
        labels=labels,
        body_mask=body,
        palette_rgb=palette,
        rgb_residual=matched["rgb_residual"],
        delta_e=matched["delta_e"],
        margin_delta_e=matched["margin_delta_e"],
    )
    Image.fromarray(reconstruction).save(raw / "reconstructed.png")
    Image.fromarray(
        _residual_overlay(
            matched["rgb_residual"], panel, labels, body, candidates
        )
    ).save(raw / "rgb_residual_overlay.png")
    Image.fromarray(
        _residual_overlay(matched["delta_e"], panel, labels, body, candidates)
    ).save(raw / "lab_residual_overlay.png")

    ambiguity = np.zeros_like(matched["margin_delta_e"])
    body_margin = matched["margin_delta_e"][body]
    ceiling = float(np.percentile(body_margin, 95)) if body_margin.size else 1.0
    ambiguity[body] = np.clip(ceiling - matched["margin_delta_e"][body], 0, None)
    Image.fromarray(
        _residual_overlay(ambiguity, panel, labels, body)
    ).save(raw / "match_ambiguity_overlay.png")

    stats = {}
    for key in ("rgb_residual", "delta_e", "margin_delta_e"):
        values = matched[key][body]
        stats[key] = {
            "mean": round(float(values.mean()), 4),
            "p90": round(float(np.percentile(values, 90)), 4),
            "p95": round(float(np.percentile(values, 95)), 4),
            "p99": round(float(np.percentile(values, 99)), 4),
            "max": round(float(values.max()), 4),
        }
    report = {
        "method": "direct_nearest_colorbar_lab_no_pre_smoothing",
        "palette_rgb": palette.tolist(),
        "colorbar_block_runs": block_runs,
        "residual_stats": stats,
        "high_deviation_regions": candidates,
    }
    (raw / "residual_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    final_report = _save_final(
        name,
        out,
        panel,
        palette,
        labels,
        matched["delta_e"],
        body,
    )
    return {
        "shape": list(panel.shape),
        "body_fraction": round(float(body.mean()), 5),
        "raw_match": str(raw.relative_to(ROOT)),
        "residual_stats": stats,
        "high_deviation_region_count": len(candidates),
        "final": str((out / "final").relative_to(ROOT)),
        "final_audit": final_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--fig7-hard-match-only",
        action="store_true",
        help="stop after direct RGB-to-colorbar assignment for Fig. 7",
    )
    mode.add_argument(
        "--fig7-frequency-match-only",
        action="store_true",
        help="derive frequent Fig. 7 RGB modes and stop after RGB assignment",
    )
    mode.add_argument(
        "--fig7-frequency-clean-annotations",
        action="store_true",
        help="clean annotation components after the frequency RGB match",
    )
    mode.add_argument(
        "--fig6c-frequency-clean-annotations",
        action="store_true",
        help="reprocess Fig. 6(c) from observed frequency colors",
    )
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig7_only = (
        args.fig7_hard_match_only
        or args.fig7_frequency_match_only
        or args.fig7_frequency_clean_annotations
    )
    fig6c_only = args.fig6c_frequency_clean_annotations
    output_names = (
        ("fig7",)
        if fig7_only
        else (("fig6c",) if fig6c_only else ("fig7", "fig6c"))
    )
    for panel_name in output_names:
        panel_output = OUTPUT / panel_name
        if panel_output.exists():
            shutil.rmtree(panel_output)
    fig6 = np.asarray(Image.open(SOURCE / "fig6_source.png").convert("RGB"))
    fig7 = np.asarray(Image.open(SOURCE / "fig7_original.jpg").convert("RGB"))
    points = np.array(
        [[765, 755], [2908, 627], [2908, 1048], [765, 1350]],
        dtype=np.float32,
    )
    fig7_rectified, homography = rectify_quadrilateral(
        fig7,
        points,
        (2400, 700),
        interpolation=cv2.INTER_NEAREST,
    )
    fig7_out = OUTPUT / "fig7"
    fig7_out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(fig7).save(fig7_out / "00_pdf_embedded_original.png")
    source_quad_audit = fig7.copy()
    cv2.polylines(
        source_quad_audit,
        [points.astype(np.int32)],
        isClosed=True,
        color=(255, 0, 255),
        thickness=8,
    )
    Image.fromarray(source_quad_audit).save(
        fig7_out / "00_rectification_source_quad.png"
    )

    colorbar = fig6[1705:1765, 2010:3490]
    palette, runs = _extract_discrete_palette(colorbar)
    if args.fig6c_frequency_clean_annotations:
        fig6c_panel = fig6[850:1831, 137:3509]
        fig6c_report = _save_fig6c_frequency_annotation_cleanup(
            OUTPUT / "fig6c", fig6c_panel
        )
        summary = {
            "source": "j.tecto.2019.06.024.pdf",
            "scope": ["fig6c"],
            "method": "frequency RGB palette with annotation cleanup",
            "pre_segmentation_smoothing": False,
            "panels": {"fig6c": fig6c_report},
        }
        (OUTPUT / "run_config.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
        return
    if args.fig7_frequency_clean_annotations:
        fig7_report = _save_fig7_frequency_annotation_cleanup(
            fig7_out, fig7_rectified
        )
        summary = {
            "source": "j.tecto.2019.06.024.pdf",
            "scope": ["fig7"],
            "method": "frequency RGB palette with annotation cleanup",
            "pre_segmentation_smoothing": False,
            "fig7_source": str(
                (SOURCE / "fig7_original.jpg").relative_to(ROOT)
            ),
            "fig7_rectification_interpolation": "nearest",
            "fig7_homography": homography.tolist(),
            "panels": {"fig7": fig7_report},
        }
        (OUTPUT / "run_config.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
        return
    if args.fig7_frequency_match_only:
        fig7_report = _save_fig7_frequency_match_only(fig7_out, fig7_rectified)
        summary = {
            "source": "j.tecto.2019.06.024.pdf",
            "scope": ["fig7"],
            "method": "frequent panel RGB modes then nearest RGB match",
            "pre_segmentation_smoothing": False,
            "post_processing": False,
            "fig7_source": str(
                (SOURCE / "fig7_original.jpg").relative_to(ROOT)
            ),
            "fig7_rectification_interpolation": "nearest",
            "fig7_homography": homography.tolist(),
            "panels": {"fig7": fig7_report},
        }
        (OUTPUT / "run_config.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
        return
    if args.fig7_hard_match_only:
        fig7_report = _save_fig7_hard_match_only(
            fig7_out, fig7_rectified, colorbar, palette, runs
        )
        summary = {
            "source": "j.tecto.2019.06.024.pdf",
            "scope": ["fig7"],
            "method": "hard nearest colorbar RGB match only",
            "pre_segmentation_smoothing": False,
            "post_processing": False,
            "palette_rgb": palette.tolist(),
            "fig7_source": str(
                (SOURCE / "fig7_original.jpg").relative_to(ROOT)
            ),
            "fig7_rectification_interpolation": "nearest",
            "fig7_homography": homography.tolist(),
            "panels": {"fig7": fig7_report},
        }
        (OUTPUT / "run_config.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
        return
    panels = {
        "fig7": fig7_rectified,
        # Fig. 6(c) is processed second because its colorbar occlusion is
        # recovered from the calibrated corresponding labels in Fig. 7.
        "fig6c": fig6[850:1831, 137:3509],
    }

    summary: dict[str, object] = {
        "source": "j.tecto.2019.06.024.pdf",
        "scope": ["fig6c", "fig7"],
        "method": "direct exact discrete colorbar match before residual audit",
        "pre_segmentation_smoothing": False,
        "palette_rgb": palette.tolist(),
        "fig7_source": str(
            (SOURCE / "fig7_original.jpg").relative_to(ROOT)
        ),
        "fig7_rectification_interpolation": "nearest",
        "fig7_homography": homography.tolist(),
        "panels": {},
    }
    for name, panel in panels.items():
        summary["panels"][name] = _save_panel(
            name, panel, colorbar, palette, runs
        )

    (OUTPUT / "run_config.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
