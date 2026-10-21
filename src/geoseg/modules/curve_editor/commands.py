"""Headless commands shared by mouse interactions, agent language, and tests.

Every command mutates the model and returns a small JSON-serializable payload.
This keeps the napari layer thin and lets natural-language or programmatic
editing drive exactly the same operations as dragging a control point.
"""

from __future__ import annotations

from typing import Any

from geoseg.modules.curve_editor.model import ControlPoint, InterfaceModel


class EditorCommands:
    def __init__(self, model: InterfaceModel) -> None:
        self.model = model

    def move_point(
        self, interface_id: str, point_id: str, dx: float = 0.0, dy: float = 0.0
    ) -> dict[str, Any]:
        point = self.model.move_point(interface_id, point_id, dx, dy)
        return self._point_payload(interface_id, point)

    def set_point(
        self, interface_id: str, point_id: str, x: float, y: float
    ) -> dict[str, Any]:
        point = self.model.set_point(interface_id, point_id, x, y)
        return self._point_payload(interface_id, point)

    def insert_point(
        self,
        interface_id: str,
        after_point_id: str,
        x: float,
        y: float,
        *,
        kind: str = "free",
    ) -> dict[str, Any]:
        point = self.model.insert_point(
            interface_id, after_point_id, x, y, kind=kind  # type: ignore[arg-type]
        )
        return self._point_payload(interface_id, point)

    def delete_point(self, interface_id: str, point_id: str) -> dict[str, Any]:
        self.model.delete_point(interface_id, point_id)
        return {"status": "ok", "interface": interface_id, "deleted_point": point_id}

    def move_interface(self, interface_id: str, dx: float = 0.0, dy: float = 0.0) -> dict[str, Any]:
        self.model.move_interface(interface_id, dx, dy)
        return {"status": "ok", "interface": interface_id, "dx": dx, "dy": dy}

    def set_locked(self, interface_id: str, point_id: str, locked: bool) -> dict[str, Any]:
        point = self.model.set_locked(interface_id, point_id, locked)
        return self._point_payload(interface_id, point)

    def add_interface(
        self,
        points: list[tuple[float, float]],
        *,
        level: int | None = None,
        closed: bool = False,
        basis: str = "B3",
        point_mode: str = "knots",
    ) -> dict[str, Any]:
        interface = self.model.add_interface(
            points,
            level=level,
            closed=closed,
            basis=basis,
            point_mode=point_mode,  # type: ignore[arg-type]
        )
        return {
            "status": "ok",
            "interface": interface.id,
            "level": interface.level,
            "points": [p.id for p in interface.points],
        }

    def add_region(
        self,
        points: list[tuple[float, float]],
        *,
        interior: tuple[float, float] | None = None,
        level: int | None = None,
        basis: str = "B3",
    ) -> dict[str, Any]:
        interface = self.model.add_region_interface(
            points, interior=interior, level=level, basis=basis
        )
        return {
            "status": "ok",
            "interface": interface.id,
            "region_id": interface.metadata.get("region_id"),
            "points": [p.id for p in interface.points],
            "interior": interface.metadata.get("interior"),
        }

    def add_split_interface(
        self,
        points: list[tuple[float, float]],
        *,
        level: int | None = None,
        basis: str = "B3",
    ) -> dict[str, Any]:
        interface = self.model.add_split_interface(
            points, level=level, basis=basis
        )
        return {
            "status": "ok",
            "interface": interface.id,
            "split_region_id": interface.metadata.get("split_region_id"),
            "points": [p.id for p in interface.points],
        }

    def remove_interface(self, interface_id: str) -> dict[str, Any]:
        self.model.remove_interface(interface_id)
        return {"status": "ok", "removed_interface": interface_id}

    @staticmethod
    def _point_payload(interface_id: str, point: ControlPoint) -> dict[str, Any]:
        return {
            "status": "ok",
            "interface": interface_id,
            "point": point.id,
            "x": point.x,
            "y": point.y,
        }


__all__ = ["EditorCommands"]
