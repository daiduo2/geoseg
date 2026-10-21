"""Interface/control-point state model for the curve editor.

The model is deliberately independent of napari and Qt so it can be driven by
mouse interactions, tests, or an agent command API.  Coordinates are pixel
coordinates ``(x=column, y=row)``; a world/metric transform can be stored in
``metadata`` later.

Key invariants:

- Control point IDs are stable for the lifetime of the model.  Inserting or
  deleting points never renumbers existing IDs.
- Background is explicit.  A pixel in ``background_mask`` is always label 0 and
  is never turned into a region.
- ``to_labels()`` is exactly idempotent when nothing was edited: the original
  labels are returned, not re-rasterized.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Literal

import numpy as np

PointKind = Literal["anchor", "curvature", "free"]
PointMode = Literal["knots", "control_points"]


@dataclass
class ControlPoint:
    id: str
    x: float
    y: float
    kind: PointKind = "free"
    locked: bool = False


@dataclass
class Interface:
    id: str
    level: int
    points: list[ControlPoint]
    closed: bool = False
    basis: str = "B3"
    point_mode: PointMode = "knots"
    fit_status: Literal["ok", "multi_segment", "rejected"] = "ok"
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)

    def point_by_id(self, point_id: str) -> ControlPoint:
        for point in self.points:
            if point.id == point_id:
                return point
        raise KeyError(f"Control point {point_id!r} not found in interface {self.id!r}")


class InterfaceModel:
    """Headless editor state: interfaces, stable IDs, background, labels."""

    def __init__(
        self,
        shape: tuple[int, int],
        *,
        background_mask: np.ndarray | None = None,
        base_labels: np.ndarray | None = None,
        merge_unseeded: bool = False,
    ) -> None:
        h, w = int(shape[0]), int(shape[1])
        if h <= 0 or w <= 0:
            raise ValueError("shape must be positive")
        self.shape = (h, w)
        self.merge_unseeded = bool(merge_unseeded)
        if background_mask is not None and background_mask.shape != self.shape:
            raise ValueError("background_mask shape does not match model shape")
        if base_labels is not None and base_labels.shape != self.shape:
            raise ValueError("base_labels shape does not match model shape")
        self.background_mask = (
            np.asarray(background_mask, dtype=bool).copy()
            if background_mask is not None
            else np.zeros(self.shape, dtype=bool)
        )
        self.base_labels = (
            np.asarray(base_labels, dtype=np.int32).copy()
            if base_labels is not None
            else None
        )
        self.interfaces: dict[str, Interface] = {}
        self._next_curve = 1
        self._next_point = 1
        self._dirty = False
        self._cached_labels: np.ndarray | None = None
        self._cached_regions: dict[int, tuple[float, float]] | None = None
        self._next_region = 1
        self.regions: dict[int, tuple[float, float]] = {}
        if self.base_labels is not None:
            self.regions = self._seeds_from_labels(self.base_labels)
            self._next_region = max(self.regions, default=0) + 1

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def _new_curve_id(self) -> str:
        curve_id = f"if_{self._next_curve:04d}"
        self._next_curve += 1
        return curve_id

    def _new_point_id(self) -> str:
        point_id = f"p_{self._next_point:04d}"
        self._next_point += 1
        return point_id

    def add_interface(
        self,
        points: list[tuple[float, float]] | list[ControlPoint],
        *,
        level: int | None = None,
        closed: bool = False,
        basis: str = "B3",
        point_mode: PointMode = "knots",
        initial: bool = False,
        interface_id: str | None = None,
    ) -> Interface:
        """Add an interface.

        ``initial=True`` is used when loading curves that already describe the
        base labels (for example extracted horizons).  It preserves the exact
        ``base_labels`` so an untouched load round-trips byte-for-byte.
        """
        if len(points) < 2:
            raise ValueError("an interface needs at least two points")
        if interface_id is None:
            interface_id = self._new_curve_id()
        elif interface_id.startswith("if_"):
            try:
                self._next_curve = max(self._next_curve, int(interface_id[3:]) + 1)
            except ValueError:
                pass
        control_points: list[ControlPoint] = []
        for item in points:
            if isinstance(item, ControlPoint):
                control_points.append(item)
                self._bump_point_counter(item.id)
            else:
                x, y = item
                control_points.append(ControlPoint(self._new_point_id(), float(x), float(y)))
        if level is None:
            level = max((i.level for i in self.interfaces.values()), default=-1) + 1
        interface = Interface(
            id=interface_id,
            level=int(level),
            points=control_points,
            closed=closed,
            basis=basis,
            point_mode=point_mode,
        )
        self.interfaces[interface_id] = interface
        if not initial:
            self._mark_dirty()
        return interface

    def _bump_point_counter(self, point_id: str) -> None:
        if point_id.startswith("p_"):
            try:
                self._next_point = max(self._next_point, int(point_id[2:]) + 1)
            except ValueError:
                pass

    def add_region_interface(
        self,
        points: list[tuple[float, float]] | np.ndarray,
        *,
        interior: tuple[float, float] | None = None,
        level: int | None = None,
        basis: str = "B3",
    ) -> Interface:
        """Add a closed boundary and seed the region the user clicked inside.

        The seed is what distinguishes a newly drawn region from the thin
        unseeded slivers produced by boundary gaps in imported models.
        """
        array = np.asarray(points, dtype=float)
        if array.ndim != 2 or array.shape[1] != 2 or len(array) < 3:
            raise ValueError("a region needs a 2-D boundary with at least three points")
        from geoseg.modules.curve_editor.region import interior_point

        seed = interior_point(array, self.shape, preferred=interior)
        interface = self.add_interface(
            [(float(x), float(y)) for x, y in array],
            level=level,
            closed=True,
            basis=basis,
        )
        region_id = self._next_region
        self._next_region += 1
        self.regions[region_id] = seed
        interface.metadata["region_id"] = region_id
        interface.metadata["interior"] = [seed[0], seed[1]]
        return interface

    def add_split_interface(
        self,
        points: list[tuple[float, float]] | np.ndarray,
        *,
        level: int | None = None,
        basis: str = "B3",
    ) -> Interface:
        """Add an open boundary and seed the side that has no region seed.

        An open interface spanning a region splits it in two, but imported
        models use ``merge_unseeded=True`` to absorb skeleton slivers.  Without
        an explicit seed the two sides would get the same old region ID, so we
        place a seed on the side opposite the region's existing seed.
        """
        array = np.asarray(points, dtype=float)
        if array.ndim != 2 or array.shape[1] != 2 or len(array) < 2:
            raise ValueError("an open interface needs a 2-D path with two points")
        before = self.to_labels()
        interface = self.add_interface(
            [(float(x), float(y)) for x, y in array],
            level=level,
            closed=False,
            basis=basis,
        )

        midpoint = array[len(array) // 2]
        col = int(np.clip(round(midpoint[0]), 0, self.shape[1] - 1))
        row = int(np.clip(round(midpoint[1]), 0, self.shape[0] - 1))
        primary = int(before[row, col])
        if primary == 0:
            return interface
        seeds = [seed for region_id, seed in self.regions.items() if region_id == primary]
        if not seeds:
            return interface
        seed = min(seeds, key=lambda item: float(np.hypot(item[0] - midpoint[0], item[1] - midpoint[1])))

        index = len(array) // 2
        tangent = array[min(index + 1, len(array) - 1)] - array[max(index - 1, 0)]
        length = float(np.hypot(*tangent))
        if length < 1e-9:
            tangent = array[-1] - array[0]
            length = float(np.hypot(*tangent))
        if length < 1e-9:
            return interface
        normal = np.array([-tangent[1], tangent[0]], dtype=float) / length
        projection = float(np.dot(midpoint - np.asarray(seed), normal))
        direction = normal if projection >= 0 else -normal

        from geoseg.modules.curve_editor.boundary_ops import label_boundary_mask

        boundary = label_boundary_mask(before)
        for distance in (6.0, 10.0, 14.0, 18.0, 24.0):
            candidate = midpoint + direction * distance
            candidate_col = int(round(candidate[0]))
            candidate_row = int(round(candidate[1]))
            if not (0 <= candidate_row < self.shape[0] and 0 <= candidate_col < self.shape[1]):
                continue
            if boundary[candidate_row, candidate_col]:
                continue
            if int(before[candidate_row, candidate_col]) != primary:
                continue
            region_id = self._next_region
            self._next_region += 1
            self.regions[region_id] = (float(candidate[0]), float(candidate[1]))
            interface.metadata["split_region_id"] = region_id
            return interface
        return interface

    def remove_interface(self, interface_id: str) -> None:
        self.interfaces.pop(interface_id, None)
        self._mark_dirty()

    def get_interface(self, interface_id: str) -> Interface:
        try:
            return self.interfaces[interface_id]
        except KeyError as exc:
            raise KeyError(f"Interface {interface_id!r} not found") from exc

    def __iter__(self) -> Iterator[Interface]:
        return iter(self.interfaces.values())

    def __len__(self) -> int:
        return len(self.interfaces)

    # ------------------------------------------------------------------
    # editing
    # ------------------------------------------------------------------
    def move_point(self, interface_id: str, point_id: str, dx: float, dy: float) -> ControlPoint:
        point = self.get_interface(interface_id).point_by_id(point_id)
        if point.locked:
            raise ValueError(f"control point {point_id!r} is locked")
        point.x += float(dx)
        point.y += float(dy)
        self._mark_dirty()
        return point

    def set_point(self, interface_id: str, point_id: str, x: float, y: float) -> ControlPoint:
        point = self.get_interface(interface_id).point_by_id(point_id)
        if point.locked:
            raise ValueError(f"control point {point_id!r} is locked")
        point.x = float(x)
        point.y = float(y)
        self._mark_dirty()
        return point

    def insert_point(
        self,
        interface_id: str,
        after_point_id: str,
        x: float,
        y: float,
        *,
        kind: PointKind = "free",
    ) -> ControlPoint:
        interface = self.get_interface(interface_id)
        index = next(
            (i for i, p in enumerate(interface.points) if p.id == after_point_id), None
        )
        if index is None:
            raise KeyError(f"Control point {after_point_id!r} not found")
        point = ControlPoint(self._new_point_id(), float(x), float(y), kind=kind)
        interface.points.insert(index + 1, point)
        self._mark_dirty()
        return point

    def delete_point(self, interface_id: str, point_id: str) -> None:
        interface = self.get_interface(interface_id)
        minimum = 3 if interface.closed else 2
        if len(interface.points) <= minimum:
            raise ValueError(
                f"interface {interface_id!r} needs at least {minimum} points"
            )
        for index, point in enumerate(interface.points):
            if point.id == point_id:
                del interface.points[index]
                self._mark_dirty()
                return
        raise KeyError(f"Control point {point_id!r} not found")

    def set_locked(self, interface_id: str, point_id: str, locked: bool) -> ControlPoint:
        point = self.get_interface(interface_id).point_by_id(point_id)
        point.locked = bool(locked)
        return point

    def move_interface(self, interface_id: str, dx: float, dy: float) -> None:
        interface = self.get_interface(interface_id)
        for point in interface.points:
            if not point.locked:
                point.x += float(dx)
                point.y += float(dy)
        self._mark_dirty()

    # ------------------------------------------------------------------
    # labels / regions
    # ------------------------------------------------------------------
    @staticmethod
    def _seeds_from_labels(labels: np.ndarray) -> dict[int, tuple[float, float]]:
        from scipy import ndimage

        seeds: dict[int, tuple[float, float]] = {}
        for label_id in sorted(int(v) for v in np.unique(labels) if v != 0):
            mask = labels == label_id
            if not mask.any():
                continue
            distance = ndimage.distance_transform_edt(mask)
            row, col = np.unravel_index(int(np.argmax(distance)), mask.shape)
            seeds[label_id] = (float(col), float(row))
        return seeds

    def _mark_dirty(self) -> None:
        self._dirty = True
        self._cached_labels = None
        self._cached_regions = None
        self.base_labels = None

    def mark_clean(self) -> None:
        """Treat the current geometry as the new saved baseline."""
        if self.base_labels is not None:
            self._dirty = False
            self._cached_labels = None
            self._cached_regions = None
            return
        self.rasterize()
        self._dirty = False

    def to_labels(self) -> np.ndarray:
        """Return labels; exact original labels when nothing was edited."""
        if not self._dirty:
            if self._cached_labels is not None:
                return self._cached_labels.copy()
            if self.base_labels is not None:
                return self.base_labels.copy()
        return self.rasterize()

    def rasterize(self) -> np.ndarray:
        from geoseg.modules.curve_editor.rasterize import rasterize_interfaces

        labels, regions, next_region = rasterize_interfaces(
            list(self.interfaces.values()),
            shape=self.shape,
            background_mask=self.background_mask,
            region_seeds=self.regions,
            next_region_id=self._next_region,
            merge_unseeded=self.merge_unseeded,
        )
        self.regions = regions
        self._next_region = next_region
        self._cached_labels = labels
        self._cached_regions = dict(regions)
        self._dirty = False
        return labels.copy()

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "shape": list(self.shape),
            "merge_unseeded": self.merge_unseeded,
            "interfaces": [
                {
                    "id": interface.id,
                    "level": interface.level,
                    "closed": interface.closed,
                    "basis": interface.basis,
                    "point_mode": interface.point_mode,
                    "fit_status": interface.fit_status,
                    "confidence": interface.confidence,
                    "metadata": interface.metadata,
                    "points": [
                        {
                            "id": point.id,
                            "x": point.x,
                            "y": point.y,
                            "kind": point.kind,
                            "locked": point.locked,
                        }
                        for point in interface.points
                    ],
                }
                for interface in self.interfaces.values()
            ],
            "regions": {str(k): list(v) for k, v in self.regions.items()},
        }


__all__ = ["ControlPoint", "Interface", "InterfaceModel", "PointKind", "PointMode"]
