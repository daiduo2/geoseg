"""Turn interface curves into a label image.

Boundaries are drawn from the spline polylines, then connected components of
the remaining pixels become regions.  Background pixels (explicit mask) and
boundary pixels stay at 0.  Existing regions keep their numeric identity via
``region_seeds``: the seed that falls inside a component supplies its label, so
a local edit does not renumber every other region.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import numpy as np

if TYPE_CHECKING:
    from geoseg.modules.curve_editor.model import Interface


def draw_interfaces(
    interfaces: Iterable[Interface],
    shape: tuple[int, int],
    *,
    thickness: int = 1,
) -> np.ndarray:
    """Rasterize interface polylines into a boolean boundary mask."""
    from scipy import ndimage
    from skimage.draw import line

    from geoseg.modules.curve_editor.spline import curve_from_interface

    h, w = shape
    boundary = np.zeros((h, w), dtype=bool)
    for interface in interfaces:
        curve = curve_from_interface(interface)
        if len(curve) < 2:
            continue
        rounded = np.rint(curve).astype(int)
        for (x0, y0), (x1, y1) in zip(rounded[:-1], rounded[1:]):
            rows, cols = line(y0, x0, y1, x1)
            valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
            boundary[rows[valid], cols[valid]] = True
    if thickness > 1:
        boundary = ndimage.binary_dilation(boundary, iterations=max(0, thickness // 2))
    return boundary


def _seed(mask: np.ndarray) -> tuple[float, float]:
    """Interior point of a component, cropped to its bounding box."""
    from scipy import ndimage

    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return (0.0, 0.0)
    row0, row1 = int(rows.min()), int(rows.max()) + 1
    col0, col1 = int(cols.min()), int(cols.max()) + 1
    sub = mask[row0:row1, col0:col1]
    distance = ndimage.distance_transform_edt(sub)
    rr, cc = np.unravel_index(int(np.argmax(distance)), sub.shape)
    return float(cc + col0), float(rr + row0)


def rasterize_interfaces(
    interfaces: Iterable[Interface],
    *,
    shape: tuple[int, int],
    background_mask: np.ndarray,
    region_seeds: dict[int, tuple[float, float]] | None = None,
    next_region_id: int = 1,
    thickness: int = 2,
    merge_unseeded: bool = False,
) -> tuple[np.ndarray, dict[int, tuple[float, float]], int]:
    """Rasterize curves and return ``(labels, region_seeds, next_region_id)``.

    ``merge_unseeded`` is used for models imported from a label image.  The
    boundary network can leave thin slivers between duplicate/junction curves;
    those components are assigned to the nearest seeded region instead of
    creating dozens of spurious region IDs.
    """
    from scipy import ndimage

    h, w = shape
    boundary = draw_interfaces(interfaces, shape, thickness=thickness)
    fillable = ~boundary & ~np.asarray(background_mask, dtype=bool)
    # Boundary pixels are 8-connected; fill must be 4-connected so a
    # diagonally touching boundary does not leak the outside into a closed
    # lens region.
    components, n_components = ndimage.label(fillable)

    regions = dict(region_seeds or {})
    seeded_components: dict[int, list[int]] = {}
    for region_id, (x, y) in regions.items():
        row, col = int(round(y)), int(round(x))
        if 0 <= row < h and 0 <= col < w:
            component_id = int(components[row, col])
            if component_id > 0:
                seeded_components.setdefault(component_id, []).append(region_id)

    mapping = np.zeros(n_components + 1, dtype=np.int32)
    for component_id, candidates in seeded_components.items():
        mapping[component_id] = min(candidates)

    unseeded = [cid for cid in range(1, n_components + 1) if mapping[cid] == 0]
    next_id = int(next_region_id)
    labels = mapping[components]

    # A boundary gap (or an edit that opens a curve) can leave one component
    # containing several seeds.  Splitting it by nearest seed keeps the other
    # regions alive instead of collapsing the whole component to the smallest
    # region ID.
    slices = ndimage.find_objects(components)
    for component_id, candidates in seeded_components.items():
        if len(candidates) < 2:
            continue
        sl = slices[component_id - 1]
        if sl is None:
            continue
        sub_components = components[sl]
        local = sub_components == component_id
        seeds = []
        for region_id in candidates:
            x, y = regions[region_id]
            row = int(round(y)) - sl[0].start
            col = int(round(x)) - sl[1].start
            if 0 <= row < local.shape[0] and 0 <= col < local.shape[1] and local[row, col]:
                seeds.append((region_id, row, col))
        if seeds:
            from skimage.graph import MCP_Geometric

            # Geodesic distance inside the component: a seed on the far side
            # of a narrow boundary gap cannot steal the whole region the way
            # Euclidean distance would.
            costs = np.where(local, 1.0, np.inf)
            best = np.full(local.shape, np.inf)
            nearest = np.zeros(local.shape, dtype=np.int32)
            for region_id, row, col in seeds:
                cumulative, _ = MCP_Geometric(costs, fully_connected=True).find_costs(
                    [(row, col)]
                )
                better = cumulative < best
                best[better] = cumulative[better]
                nearest[better] = region_id
            view = labels[sl]
            view[local] = nearest[local]
        else:
            view = labels[sl]
            view[local] = min(candidates)

    if unseeded:
        merged = False
        if merge_unseeded and regions:
            seed_map = np.zeros((h, w), dtype=np.int32)
            for region_id, (x, y) in regions.items():
                row, col = int(round(y)), int(round(x))
                if 0 <= row < h and 0 <= col < w:
                    seed_map[row, col] = region_id
            if seed_map.any():
                _, indices = ndimage.distance_transform_edt(
                    seed_map == 0, return_indices=True
                )
                nearest = seed_map[indices[0], indices[1]]
                unseeded_mask = (components > 0) & (labels == 0)
                labels[unseeded_mask] = nearest[unseeded_mask]
                merged = True
        if not merged:
            for component_id in unseeded:
                region_id = next_id
                next_id += 1
                regions[region_id] = _seed(components == component_id)
                mapping[component_id] = region_id
            unseeded_mask = (components > 0) & (labels == 0)
            labels[unseeded_mask] = mapping[components[unseeded_mask]]

    labels[boundary] = 0
    labels[np.asarray(background_mask, dtype=bool)] = 0
    return labels, regions, next_id


__all__ = ["draw_interfaces", "rasterize_interfaces"]
