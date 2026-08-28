#!/usr/bin/env python3
"""Split coarse color regions again using incomplete fault-line barriers.

The first stage supplies material labels (for example, yellow/blue).  The
second stage extracts long red structural lines, completes an open line to the
edge of its current material region when needed, and relabels the connected
areas on either side.  Short red annotations are retained for visualization
but do not become splitting barriers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize


REGION_COLORS = np.array(
    [
        [65, 181, 103],
        [216, 82, 143],
        [69, 154, 211],
        [220, 207, 62],
        [132, 65, 190],
        [46, 184, 178],
        [238, 132, 55],
        [121, 112, 203],
    ],
    dtype=np.uint8,
)


def detect_red(rgb: np.ndarray) -> np.ndarray:
    """Return red ink while rejecting black text and cyan boundary overlays."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0]
    return (
        ((hue <= 10) | (hue >= 170))
        & (hsv[..., 1] >= 115)
        & (hsv[..., 2] >= 135)
    )


def close_red_gaps(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 3))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0


def count_large_components(mask: np.ndarray, min_area: int) -> int:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return sum(int(stats[i, cv2.CC_STAT_AREA]) >= min_area for i in range(1, n))


def component_endpoints(component: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Estimate endpoints of a long component along its skeleton principal axis."""
    ys, xs = np.nonzero(skeletonize(component))
    if len(xs) < 20:
        return None
    points = np.column_stack([xs, ys]).astype(np.float64)
    centered = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    projection = centered @ axis
    lo = points[projection <= np.percentile(projection, 1.0)].mean(axis=0)
    hi = points[projection >= np.percentile(projection, 99.0)].mean(axis=0)
    return lo, hi


def ray_to_region_edge(
    endpoint: np.ndarray,
    direction: np.ndarray,
    material: np.ndarray,
    max_steps: int,
) -> tuple[int, int] | None:
    """Trace an endpoint tangent until it exits the current material region."""
    height, width = material.shape
    norm = float(np.linalg.norm(direction))
    if norm == 0:
        return None
    direction = direction / norm
    x0, y0 = endpoint
    last = (int(round(x0)), int(round(y0)))
    entered = False
    for step in range(max_steps + 1):
        x = int(round(x0 + direction[0] * step))
        y = int(round(y0 + direction[1] * step))
        if not (0 <= x < width and 0 <= y < height):
            return last if entered else None
        if material[y, x]:
            entered = True
            last = (x, y)
        elif entered:
            return last
    return last if entered else None


def complete_open_barriers(
    labels: np.ndarray,
    red_mask: np.ndarray,
    *,
    min_region_area: int,
    min_fault_span: int,
    barrier_width: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Complete long open faults only in materials not already split."""
    completed = np.zeros_like(red_mask)
    splitting = np.zeros_like(red_mask)
    records: list[dict[str, object]] = []

    n_red, red_cc, red_stats, _ = cv2.connectedComponentsWithStats(
        red_mask.astype(np.uint8), 8
    )
    structure_ids = []
    for component_id in range(1, n_red):
        x, y, w, h, area = map(int, red_stats[component_id])
        if area >= 500 and max(w, h) >= min_fault_span:
            structure_ids.append(component_id)

    for material_id in sorted(int(v) for v in np.unique(labels) if v > 0):
        material = labels == material_id
        observed = red_mask & material
        observed_barrier = cv2.dilate(
            observed.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (barrier_width, barrier_width)),
        ) > 0
        before = count_large_components(material & ~observed_barrier, min_region_area)
        splitting |= observed_barrier
        extensions: list[list[list[int]]] = []

        # A material already split by observed ink needs no speculative extension.
        if before <= 1:
            for component_id in structure_ids:
                component = (red_cc == component_id) & material
                ys, xs = np.nonzero(component)
                if len(xs) < 100 or max(np.ptp(xs), np.ptp(ys)) < min_fault_span:
                    continue
                endpoints = component_endpoints(component)
                if endpoints is None:
                    continue
                lo, hi = endpoints
                axis = hi - lo
                for endpoint, direction in ((lo, -axis), (hi, axis)):
                    end = ray_to_region_edge(
                        endpoint, direction, material, max_steps=max(labels.shape) * 2
                    )
                    if end is None:
                        continue
                    start = tuple(int(round(v)) for v in endpoint)
                    if np.hypot(end[0] - start[0], end[1] - start[1]) < barrier_width * 2:
                        continue
                    line = np.zeros_like(red_mask, dtype=np.uint8)
                    cv2.line(line, start, end, 255, barrier_width, cv2.LINE_AA)
                    line = (line > 0) & material
                    completed |= line
                    splitting |= line
                    extensions.append([list(start), list(end)])

        after = count_large_components(material & ~splitting, min_region_area)
        records.append(
            {
                "material_id": material_id,
                "large_regions_before_completion": before,
                "large_regions_after_completion": after,
                "extensions": extensions,
            }
        )
    return splitting, completed, records


def label_split_regions(
    labels: np.ndarray,
    barrier: np.ndarray,
    min_region_area: int,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    """Connected-component label each material, then fill barrier pixels back in."""
    result = np.zeros_like(labels, dtype=np.int32)
    region_records: list[dict[str, int]] = []
    next_id = 1

    for material_id in sorted(int(v) for v in np.unique(labels) if v > 0):
        material = labels == material_id
        n, cc, stats, _ = cv2.connectedComponentsWithStats(
            (material & ~barrier).astype(np.uint8), 8
        )
        large_ids = [
            i
            for i in range(1, n)
            if int(stats[i, cv2.CC_STAT_AREA]) >= min_region_area
        ]
        if not large_ids:
            large_ids = [int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1]

        material_regions = np.zeros_like(labels, dtype=np.int32)
        local_to_global: dict[int, int] = {}
        for component_id in large_ids:
            global_id = next_id
            next_id += 1
            local_to_global[component_id] = global_id
            material_regions[cc == component_id] = global_id

        missing = material & (material_regions == 0)
        if np.any(missing):
            # Nearest labeled pixel fills the barrier and absorbs tiny fragments.
            _, indices = ndimage.distance_transform_edt(
                material_regions == 0, return_indices=True
            )
            nearest = material_regions[indices[0], indices[1]]
            material_regions[missing] = nearest[missing]

        result[material] = material_regions[material]
        for component_id, global_id in local_to_global.items():
            region_records.append(
                {
                    "region_id": global_id,
                    "material_id": material_id,
                    "seed_area": int(stats[component_id, cv2.CC_STAT_AREA]),
                    "filled_area": int(np.count_nonzero(result == global_id)),
                }
            )
    # Closed annotation symbols can enclose tiny components.  Merge each such
    # island into the neighboring region with which it shares the most border.
    for _ in range(4):
        changed = False
        for material_id in sorted(int(v) for v in np.unique(labels) if v > 0):
            material = labels == material_id
            region_ids = sorted(int(v) for v in np.unique(result[material]) if v > 0)
            for region_id in region_ids:
                n, cc, stats, _ = cv2.connectedComponentsWithStats(
                    (result == region_id).astype(np.uint8), 8
                )
                if n <= 2:
                    continue
                main_id = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
                for component_id in range(1, n):
                    if component_id == main_id:
                        continue
                    island = cc == component_id
                    ring = cv2.dilate(
                        island.astype(np.uint8), np.ones((3, 3), np.uint8)
                    ).astype(bool) & ~island & material
                    neighbors = result[ring]
                    neighbors = neighbors[(neighbors > 0) & (neighbors != region_id)]
                    if neighbors.size == 0:
                        continue
                    target = int(np.bincount(neighbors).argmax())
                    result[island] = target
                    changed = True
        if not changed:
            break

    region_records = [
        {
            **record,
            "filled_area": int(np.count_nonzero(result == record["region_id"])),
        }
        for record in region_records
        if np.any(result == record["region_id"])
    ]
    return result, region_records


def colorize(labels: np.ndarray) -> np.ndarray:
    output = np.full((*labels.shape, 3), 255, dtype=np.uint8)
    for region_id in sorted(int(v) for v in np.unique(labels) if v > 0):
        output[labels == region_id] = REGION_COLORS[(region_id - 1) % len(REGION_COLORS)]
    return output


def load_boundary_hints(
    path: Path | None, shape: tuple[int, int], width: int
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Rasterize reviewed structural boundaries supplied as named polylines."""
    mask = np.zeros(shape, dtype=np.uint8)
    if path is None:
        return mask.astype(bool), []
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("polylines", [])
    for record in records:
        points = np.asarray(record["points"], dtype=np.int32)
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
            raise ValueError(f"invalid boundary hint points: {record!r}")
        cv2.polylines(mask, [points], False, 255, width, cv2.LINE_AA)
    return mask > 0, records


def region_boundaries(labels: np.ndarray) -> np.ndarray:
    """Return internal region boundaries plus the outside edge of the model."""
    active = labels > 0
    boundary = cv2.morphologyEx(
        active.astype(np.uint8),
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), np.uint8),
    ) > 0
    horizontal = (labels[:, 1:] != labels[:, :-1]) & (labels[:, 1:] > 0) & (
        labels[:, :-1] > 0
    )
    vertical = (labels[1:, :] != labels[:-1, :]) & (labels[1:, :] > 0) & (
        labels[:-1, :] > 0
    )
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    return boundary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("first_stage", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--display-image",
        type=Path,
        help="Optional clean original used for overlays and the comparison panel.",
    )
    parser.add_argument(
        "--boundary-hints",
        type=Path,
        help="Optional JSON with reviewed F1/F2/F3-style boundary polylines.",
    )
    parser.add_argument(
        "--domain-npz",
        type=Path,
        help="Optional NPZ whose body_mask or nonzero labels define the valid panel body.",
    )
    parser.add_argument(
        "--include-legend",
        action="store_true",
        help="Treat the legend bbox as covered geology and fill it from first-stage labels.",
    )
    parser.add_argument("--min-region-area", type=int, default=5000)
    parser.add_argument("--min-fault-span", type=int, default=300)
    parser.add_argument("--barrier-width", type=int, default=7)
    args = parser.parse_args()

    rgb = np.asarray(Image.open(args.image).convert("RGB"))
    display_rgb = (
        np.asarray(Image.open(args.display_image).convert("RGB"))
        if args.display_image is not None
        else rgb
    )
    if display_rgb.shape != rgb.shape:
        raise ValueError(
            f"display image shape mismatch: display={display_rgb.shape}, image={rgb.shape}"
        )
    first = np.load(args.first_stage)
    labels = first["labels"].astype(np.int32)
    if labels.shape != rgb.shape[:2]:
        raise ValueError(f"shape mismatch: image={rgb.shape[:2]}, labels={labels.shape}")

    domain_data = None
    legend_fill_material: int | None = None
    legend_mask = None
    if args.domain_npz is not None:
        domain_data = np.load(args.domain_npz)
        if "body_mask" in domain_data.files:
            domain = domain_data["body_mask"].astype(bool)
        else:
            domain = domain_data["labels"] > 0
        if args.include_legend and "legend_mask" in domain_data.files:
            legend_mask = domain_data["legend_mask"].astype(bool)
            domain |= legend_mask
        if domain.shape != labels.shape:
            raise ValueError(
                f"domain shape mismatch: domain={domain.shape}, labels={labels.shape}"
            )
        labels = labels.copy()
        labels[~domain] = 0
        if legend_mask is not None:
            legend_values = labels[legend_mask]
            legend_values = legend_values[legend_values > 0]
            if legend_values.size:
                legend_fill_material = int(np.bincount(legend_values).argmax())
                labels[legend_mask] = legend_fill_material

    removed_material_islands: dict[str, int] = {}
    for material_id in sorted(int(v) for v in np.unique(labels) if v > 0):
        n, cc, stats, _ = cv2.connectedComponentsWithStats(
            (labels == material_id).astype(np.uint8), 8
        )
        if n <= 2:
            removed_material_islands[str(material_id)] = 0
            continue
        main_id = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        islands = (labels == material_id) & (cc != main_id)
        removed_material_islands[str(material_id)] = int(np.count_nonzero(islands))
        labels[islands] = 0

    detected_red = detect_red(rgb)
    if "red_lines" in first.files:
        detected_red |= first["red_lines"].astype(bool)
    elif "red_fault_mask" in first.files:
        detected_red |= first["red_fault_mask"].astype(bool)
    if domain_data is not None and "red_lines" in domain_data.files:
        detected_red |= domain_data["red_lines"].astype(bool)
    observed_red = close_red_gaps(detected_red)

    barrier, inferred, completion_records = complete_open_barriers(
        labels,
        observed_red,
        min_region_area=args.min_region_area,
        min_fault_span=args.min_fault_span,
        barrier_width=args.barrier_width,
    )
    reviewed_boundaries, boundary_hint_records = load_boundary_hints(
        args.boundary_hints, labels.shape, args.barrier_width
    )
    reviewed_boundaries &= labels > 0
    barrier |= reviewed_boundaries
    regions, region_records = label_split_regions(
        labels, barrier, args.min_region_area
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    filled = colorize(regions)
    overlay = cv2.addWeighted(display_rgb, 0.48, filled, 0.52, 0)
    overlay[observed_red] = [238, 36, 42]
    overlay[inferred] = [255, 0, 220]

    boundary = region_boundaries(regions)
    boundary_halo = cv2.dilate(
        boundary.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ) > 0
    overlay_with_boundaries = overlay.copy()
    overlay_with_boundaries[boundary_halo] = [250, 250, 250]
    overlay_with_boundaries[boundary] = [24, 31, 38]

    comparison = np.concatenate(
        [display_rgb, overlay_with_boundaries, filled], axis=1
    )

    barrier_view = np.zeros_like(rgb)
    barrier_view[observed_red] = [238, 36, 42]
    barrier_view[inferred] = [255, 0, 220]

    Image.fromarray(filled).save(args.output_dir / "secondary_regions.png")
    Image.fromarray(overlay).save(args.output_dir / "secondary_overlay.png")
    Image.fromarray(overlay_with_boundaries).save(
        args.output_dir / "secondary_overlay_boundaries.png"
    )
    Image.fromarray(comparison).save(
        args.output_dir / "comparison_original_overlay_mask.png"
    )
    Image.fromarray(barrier_view).save(args.output_dir / "completed_fault_barrier.png")
    Image.fromarray((barrier.astype(np.uint8) * 255)).save(
        args.output_dir / "splitting_barrier_mask.png"
    )
    np.savez_compressed(
        args.output_dir / "secondary_labels.npz",
        labels=regions,
        material_labels=labels,
        observed_red_mask=observed_red,
        inferred_completion_mask=inferred,
        reviewed_boundary_mask=reviewed_boundaries,
        splitting_barrier=barrier,
        region_boundary_mask=boundary,
    )

    report = {
        "image": str(args.image),
        "display_image": str(args.display_image) if args.display_image else None,
        "first_stage": str(args.first_stage),
        "domain_npz": str(args.domain_npz) if args.domain_npz else None,
        "boundary_hints": str(args.boundary_hints) if args.boundary_hints else None,
        "include_legend": args.include_legend,
        "legend_fill_material": legend_fill_material,
        "method": "color labels -> long red barriers -> tangent completion -> connected fill",
        "parameters": {
            "min_region_area": args.min_region_area,
            "min_fault_span": args.min_fault_span,
            "barrier_width": args.barrier_width,
        },
        "n_materials": len(set(labels.ravel()) - {0}),
        "n_secondary_regions": len(set(regions.ravel()) - {0}),
        "removed_first_stage_island_pixels": removed_material_islands,
        "completion": completion_records,
        "reviewed_boundaries": boundary_hint_records,
        "regions": region_records,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
