"""Napari view for the control-point interface editor.

The view is intentionally a thin public-API wrapper:

- ``Image``: the source panel.
- ``Labels``: read-only rasterized result.
- ``Shapes`` (path): the spline interfaces.
- ``Points``: draggable control handles.
- a second ``Points`` layer for drafting a new interface.

No private napari APIs are used.  The layer data is rebuilt from the headless
model after edits, so the model remains the source of truth.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from geoseg.modules.curve_editor.boundary_ops import (
    clip_polygon_to_rect,
    extend_open_curve,
    self_intersects,
)
from geoseg.modules.curve_editor.commands import EditorCommands
from geoseg.modules.curve_editor.io import (
    load_labels,
    load_session,
    model_from_labels,
    save_labels,
    save_session,
    save_shapes,
)
from geoseg.modules.curve_editor.model import InterfaceModel
from geoseg.modules.curve_editor.region import (
    close_and_resample,
    select_control_points,
    snap_region,
)
from geoseg.modules.curve_editor.spline import curve_from_interface

__all__ = ["CurveEditorView", "main", "open_editor"]


def _point_keys(model: InterfaceModel) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    for interface in sorted(model.interfaces.values(), key=lambda item: (item.level, item.id)):
        keys.extend((interface.id, point.id) for point in interface.points)
    return keys


def _points_array(model: InterfaceModel, keys: list[tuple[str, str]]) -> np.ndarray:
    coords = []
    for interface_id, point_id in keys:
        point = model.get_interface(interface_id).point_by_id(point_id)
        coords.append((point.y, point.x))
    return np.asarray(coords, dtype=float) if coords else np.empty((0, 2), dtype=float)


def _add_points_layer(viewer: Any, data: np.ndarray, **kwargs: Any):
    """napari <0.7 spells the point outline ``edge_color``, >=0.7 ``border_color``."""
    name = kwargs.pop("name")
    size = kwargs.pop("size")
    face_color = kwargs.pop("face_color")
    outline = kwargs.pop("outline_color")
    try:
        return viewer.add_points(
            data,
            name=name,
            size=size,
            face_color=face_color,
            border_color=outline,
            border_width=1,
        )
    except TypeError:
        return viewer.add_points(
            data,
            name=name,
            size=size,
            face_color=face_color,
            edge_color=outline,
            edge_width=1,
        )


class CurveEditorView:
    """Wire the headless model to napari layers."""

    def __init__(
        self,
        model: InterfaceModel,
        image: np.ndarray | None = None,
        *,
        viewer: Any | None = None,
        title: str = "geoseg curve editor",
        output_labels: str | Path | None = None,
        output_shapes: str | Path | None = None,
        output_session: str | Path | None = None,
    ) -> None:
        import napari

        self.model = model
        self.commands = EditorCommands(model)
        self.output_labels = Path(output_labels) if output_labels else None
        self.output_shapes = Path(output_shapes) if output_shapes else None
        self.output_session = Path(output_session) if output_session else None
        self.viewer = viewer or napari.Viewer(title=title)
        self._syncing = False
        self._point_order: list[tuple[str, str]] = []
        self._timer: Any | None = None

        if image is not None:
            self.image_layer = self.viewer.add_image(image, name="image")
        else:
            self.image_layer = None
        self.labels_layer = self.viewer.add_labels(
            self.model.to_labels(), name="labels", opacity=0.6
        )
        try:
            self.labels_layer.editable = False
        except AttributeError:
            self.labels_layer.mode = "pan_zoom"
        self.interfaces_layer = self.viewer.add_shapes(
            name="interfaces",
            edge_color="white",
            edge_width=2.0,
            face_color="transparent",
            opacity=0.9,
        )
        try:
            self.interfaces_layer.editable = False
        except AttributeError:
            self.interfaces_layer.mode = "pan_zoom"
        self.points_layer = _add_points_layer(
            self.viewer,
            np.empty((0, 2), dtype=float),
            name="control points",
            size=8,
            face_color="#ffd23f",
            outline_color="black",
        )
        self.points_layer.mode = "select"
        self.draft_layer = _add_points_layer(
            self.viewer,
            np.empty((0, 2), dtype=float),
            name="draft interface",
            size=8,
            face_color="#ff4fd8",
            outline_color="white",
        )
        self.draft_layer.visible = False
        self.draft_layer.mode = "pan_zoom"
        self.region_preview_layer = self.viewer.add_shapes(
            name="region draft",
            edge_color="#35e0ff",
            edge_width=3.0,
            face_color="transparent",
            opacity=1.0,
        )
        try:
            self.region_preview_layer.editable = False
        except AttributeError:
            self.region_preview_layer.mode = "pan_zoom"
        self.region_preview_layer.visible = False

        self.region_mode = False
        self.region_phase = "idle"  # idle | drawing | awaiting_interior
        self._region_stroke: list[tuple[float, float]] = []
        self._region_boundary: np.ndarray | None = None
        self._region_control_points: np.ndarray | None = None

        self._setup_timer()
        self.points_layer.events.data.connect(self._on_points_data)
        self.viewer.mouse_drag_callbacks.append(self._on_viewer_drag)
        self._bind_keys()
        self.viewer.layers.selection.active = self.points_layer
        self.viewer.status = (
            "n new interface | r region (draw circle, click inside) | "
            "Enter commit | Esc cancel | drag points | i insert | Delete remove | Ctrl-S save"
        )
        self.refresh()

    # ------------------------------------------------------------------
    # sync
    # ------------------------------------------------------------------
    def _setup_timer(self) -> None:
        try:
            from qtpy.QtCore import QTimer

            self._timer = QTimer()
            self._timer.setSingleShot(True)
            self._timer.setInterval(30)
            self._timer.timeout.connect(self._flush_refresh)
        except Exception:
            self._timer = None

    def _schedule_refresh(self) -> None:
        if self._timer is not None:
            if not self._timer.isActive():
                self._timer.start()
        else:
            self.refresh(points=False)

    def _flush_refresh(self) -> None:
        self.refresh(points=False)

    def _bind_keys(self) -> None:
        self.viewer.bind_key("n", self._start_draft)
        self.viewer.bind_key("r", self._toggle_region_mode)
        self.viewer.bind_key("Enter", self._commit_draft)
        self.viewer.bind_key("Shift-Enter", self._commit_closed_draft)
        self.viewer.bind_key("Escape", self._on_escape)
        self.viewer.bind_key("i", self._insert_point)
        self.viewer.bind_key("Delete", self._delete_selected)
        self.viewer.bind_key("Backspace", self._delete_selected)
        self.viewer.bind_key("Control-S", self.save)

    @staticmethod
    def _set_paths(layer: Any, paths: list[np.ndarray]) -> None:
        layer.data = paths
        if paths:
            try:
                layer.shape_type = ["path"] * len(paths)
            except Exception:
                pass

    @staticmethod
    def _event_xy(event: Any) -> tuple[float, float]:
        position = getattr(event, "position", None)
        if position is None or len(position) < 2:
            return (0.0, 0.0)
        return (float(position[-1]), float(position[-2]))

    def refresh(self, *, points: bool = True) -> None:
        labels = self.model.to_labels()
        self.labels_layer.data = labels

        paths: list[np.ndarray] = []
        for interface in sorted(
            self.model.interfaces.values(), key=lambda item: (item.level, item.id)
        ):
            curve = curve_from_interface(interface)
            if len(curve) >= 2:
                paths.append(np.column_stack([curve[:, 1], curve[:, 0]]))
        previous_selection = self._selected_point_keys()
        self._syncing = True
        try:
            self._set_paths(self.interfaces_layer, paths)
            if points:
                keys = _point_keys(self.model)
                self._point_order = keys
                self.points_layer.data = _points_array(self.model, keys)
                if previous_selection:
                    selected = {
                        index
                        for index, key in enumerate(keys)
                        if key in previous_selection
                    }
                    self.points_layer.selected_data = selected
        finally:
            self._syncing = False

    def _selected_point_keys(self) -> set[tuple[str, str]]:
        selected = set()
        for index in sorted(getattr(self.points_layer, "selected_data", set())):
            if 0 <= index < len(self._point_order):
                selected.add(self._point_order[index])
        return selected

    def _on_points_data(self, event=None) -> None:
        if self._syncing:
            return
        data = np.asarray(self.points_layer.data)
        if data.ndim != 2 or data.shape[0] != len(self._point_order):
            self.refresh()
            return

        changed = False
        needs_reset = False
        h, w = self.model.shape
        for index, (interface_id, point_id) in enumerate(self._point_order):
            point = self.model.get_interface(interface_id).point_by_id(point_id)
            if point.locked:
                if data[index, 0] != point.y or data[index, 1] != point.x:
                    needs_reset = True
                continue
            y = float(np.clip(data[index, 0], 0, h - 1))
            x = float(np.clip(data[index, 1], 0, w - 1))
            if abs(point.x - x) > 1e-6 or abs(point.y - y) > 1e-6:
                self.model.set_point(interface_id, point_id, x, y)
                changed = True
            if x != data[index, 1] or y != data[index, 0]:
                needs_reset = True
        if changed:
            self._schedule_refresh()
        if needs_reset:
            self.refresh()

    # ------------------------------------------------------------------
    # gestures
    # ------------------------------------------------------------------
    def _update_region_preview(self) -> None:
        if len(self._region_stroke) < 2:
            return
        data = np.asarray([(y, x) for x, y in self._region_stroke], dtype=float)
        self._set_paths(self.region_preview_layer, [data])
        self.region_preview_layer.visible = True

    def _on_viewer_drag(self, viewer, event):
        """Freehand capture for region mode; ignored otherwise."""
        if not self.region_mode:
            return
        if self.region_phase == "awaiting_interior":
            if event.type == "mouse_press":
                self._commit_region_interior(self._event_xy(event))
                try:
                    event.handled = True
                except Exception:
                    pass
            return

        if event.type != "mouse_press":
            return
        self.region_phase = "drawing"
        self._region_stroke = [self._event_xy(event)]
        self._update_region_preview()
        try:
            event.handled = True
        except Exception:
            pass
        yield
        while event.type == "mouse_move":
            if not self.region_mode or self.region_phase != "drawing":
                return
            self._region_stroke.append(self._event_xy(event))
            if len(self._region_stroke) % 2 == 0:
                self._update_region_preview()
            try:
                event.handled = True
            except Exception:
                pass
            yield
        if event.type == "mouse_release" and self.region_mode and self.region_phase == "drawing":
            self._finish_region_stroke()
        elif self.region_phase == "drawing":
            self._cancel_region_state(keep_mode=self.region_mode)

    def _finish_region_stroke(self) -> None:
        self.region_phase = "idle"
        stroke = np.asarray(self._region_stroke, dtype=float)
        if len(stroke) < 8:
            self._cancel_region_state(keep_mode=True)
            self.viewer.status = "Stroke too short; region mode still on"
            return
        try:
            clipped = clip_polygon_to_rect(stroke, self.model.shape)
            if self_intersects(close_and_resample(clipped, 128)):
                self._cancel_region_state(keep_mode=True)
                self.viewer.status = (
                    "Stroke crosses itself; draw one simple closed loop"
                )
                return
            if self.image_layer is not None:
                image = np.asarray(self.image_layer.data)
                radius = max(12.0, min(40.0, 0.06 * min(self.model.shape)))
                snapped = snap_region(
                    image, clipped, samples=192, search_radius=radius
                )
                boundary = snapped.boundary
                control = snapped.control_points
            else:
                boundary = close_and_resample(clipped, 192)
                control = select_control_points(boundary)
        except ValueError as exc:
            self._cancel_region_state(keep_mode=True)
            self.viewer.status = f"Region stroke rejected: {exc}"
            return

        self._region_boundary = boundary
        self._region_control_points = control
        self._set_paths(
            self.region_preview_layer,
            [np.column_stack([boundary[:, 1], boundary[:, 0]])],
        )
        self.region_preview_layer.visible = True
        self.draft_layer.data = np.column_stack([control[:, 1], control[:, 0]])
        self.draft_layer.mode = "pan_zoom"
        self.draft_layer.visible = True
        self.region_phase = "awaiting_interior"
        self.viewer.status = (
            f"Snapped boundary with {len(control)} control points. "
            "Click inside the region to fill it; Esc cancels."
        )

    def _commit_region_interior(self, interior: tuple[float, float]) -> None:
        if self._region_control_points is None:
            return
        points = [(float(x), float(y)) for x, y in self._region_control_points]
        try:
            result = self.commands.add_region(
                points, interior=interior, basis="CatmullRom"
            )
        except ValueError as exc:
            self.viewer.status = f"Region rejected: {exc}"
            return
        self._cancel_region_state(keep_mode=True)
        self.refresh()
        self.viewer.status = (
            f"Filled region {result['region_id']} "
            f"(interface {result['interface']}); region mode still on"
        )

    def _toggle_region_mode(self, viewer=None) -> None:
        if self.region_mode:
            self._cancel_region_state(keep_mode=False)
            self.viewer.status = "Region mode off"
            return
        self._hide_draft()
        self.region_mode = True
        self.region_phase = "idle"
        self.viewer.status = (
            "Region mode: hold and drag a closed loop around the region, "
            "then click inside."
        )

    def _cancel_region_state(self, *, keep_mode: bool) -> None:
        self._region_stroke = []
        self._region_boundary = None
        self._region_control_points = None
        self.region_phase = "idle"
        try:
            self._set_paths(self.region_preview_layer, [])
        except Exception:
            pass
        self.region_preview_layer.visible = False
        self.draft_layer.data = np.empty((0, 2), dtype=float)
        self.draft_layer.mode = "pan_zoom"
        self.draft_layer.visible = False
        if not keep_mode:
            self.region_mode = False

    def _on_escape(self, viewer=None) -> None:
        if self.region_mode or self.region_phase != "idle":
            self._cancel_region_state(keep_mode=False)
            self.viewer.status = "Region cancelled"
        else:
            self._cancel_draft()

    def _start_draft(self, viewer=None) -> None:
        if self.region_mode:
            self._cancel_region_state(keep_mode=False)
        self.draft_layer.visible = True
        self.draft_layer.mode = "add"
        self.draft_layer.data = np.empty((0, 2), dtype=float)
        self.viewer.layers.selection.active = self.draft_layer
        self.viewer.status = "Draft: click points, Enter commit, Shift-Enter closed, Esc cancel"

    def _commit_draft(self, viewer=None) -> None:
        self._commit_draft_impl(closed=False)

    def _commit_closed_draft(self, viewer=None) -> None:
        self._commit_draft_impl(closed=True)

    def _commit_draft_impl(self, *, closed: bool) -> None:
        data = np.asarray(self.draft_layer.data)
        if len(data) < 2:
            self.viewer.status = "Draft needs at least two points"
            return
        points = np.asarray([(float(col), float(row)) for row, col in data], dtype=float)
        try:
            if closed:
                points = clip_polygon_to_rect(points, self.model.shape)
                if len(points) < 3 or self_intersects(points):
                    self.viewer.status = "Closed draft must be a simple polygon"
                    return
            else:
                points = extend_open_curve(points, self.model.to_labels())
        except ValueError as exc:
            self.viewer.status = f"Draft endpoint adjustment failed: {exc}"
            return
        if closed:
            result = self.commands.add_region(
                [(float(x), float(y)) for x, y in points], basis="B1"
            )
            message = f"Closed region {result['region_id']} added"
        else:
            result = self.commands.add_split_interface(
                [(float(x), float(y)) for x, y in points], basis="B3"
            )
            message = f"Open interface {result['interface']} extended to boundaries"
            if result.get("split_region_id") is not None:
                message += f"; new region {result['split_region_id']}"
        self._hide_draft()
        self.viewer.layers.selection.active = self.points_layer
        self.refresh()
        self.viewer.status = message

    def _cancel_draft(self, viewer=None) -> None:
        self._hide_draft()
        self.viewer.layers.selection.active = self.points_layer
        self.viewer.status = "Draft cancelled"

    def _hide_draft(self) -> None:
        self.draft_layer.data = np.empty((0, 2), dtype=float)
        self.draft_layer.mode = "pan_zoom"
        self.draft_layer.visible = False

    def _insert_point(self, viewer=None) -> None:
        selected = sorted(getattr(self.points_layer, "selected_data", set()))
        if not selected or selected[0] >= len(self._point_order):
            self.viewer.status = "Select a control point first"
            return
        interface_id, point_id = self._point_order[selected[0]]
        position = self.viewer.cursor.position
        if position is None or len(position) < 2:
            return
        y, x = float(position[-2]), float(position[-1])
        self.commands.insert_point(interface_id, point_id, x, y)
        self.refresh()
        new_index = min(selected[0] + 1, len(self._point_order) - 1)
        self.points_layer.selected_data = {new_index}

    def _delete_selected(self, viewer=None) -> None:
        selected = sorted(getattr(self.points_layer, "selected_data", set()), reverse=True)
        for index in selected:
            if index >= len(self._point_order):
                continue
            interface_id, point_id = self._point_order[index]
            try:
                self.commands.delete_point(interface_id, point_id)
            except (ValueError, KeyError):
                pass
        self.refresh()

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self, viewer=None) -> None:
        written = []
        if self.output_labels:
            save_labels(self.model, self.output_labels)
            written.append(str(self.output_labels))
        if self.output_shapes:
            save_shapes(self.model, self.output_shapes)
            written.append(str(self.output_shapes))
        if self.output_session:
            save_session(self.model, self.output_session)
            written.append(str(self.output_session))
        if written:
            self.viewer.status = "Saved: " + ", ".join(written)
        else:
            self.viewer.status = "No output path configured"
        if hasattr(self.viewer, "status"):
            print(self.viewer.status)

    def _on_close(self, event=None) -> None:
        self.save()

    def run(self) -> None:
        import napari

        self.viewer.events.close.connect(self._on_close)
        napari.run()


def open_editor(
    model: InterfaceModel,
    image: np.ndarray | None = None,
    *,
    title: str = "geoseg curve editor",
    **kwargs: Any,
) -> CurveEditorView:
    return CurveEditorView(model, image, title=title, **kwargs)


def _load_image(path: str | Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Control-point interface editor")
    parser.add_argument("--image", type=Path, help="source panel image")
    parser.add_argument("--labels", type=Path, help="input labels .npz")
    parser.add_argument("--session", type=Path, help="input curve-editor session .npz")
    parser.add_argument("--output-labels", type=Path, help="save labels .npz")
    parser.add_argument("--output-shapes", type=Path, help="save curves .json")
    parser.add_argument("--output-session", type=Path, help="save full session .npz")
    parser.add_argument("--no-extract", action="store_true", help="do not extract curves from labels")
    parser.add_argument("--title", default="geoseg curve editor")
    args = parser.parse_args(argv)

    image = _load_image(args.image) if args.image else None
    if args.session:
        model = load_session(args.session)
    elif args.labels:
        labels = load_labels(args.labels)
        model = model_from_labels(labels, extract=not args.no_extract)
    elif image is not None:
        model = InterfaceModel(image.shape[:2])
    else:
        parser.error("provide --image, --labels, or --session")
        return 2

    if image is not None and image.shape[:2] != model.shape:
        parser.error(
            f"image shape {image.shape[:2]} does not match model shape {model.shape}"
        )

    view = CurveEditorView(
        model,
        image,
        title=args.title,
        output_labels=args.output_labels,
        output_shapes=args.output_shapes,
        output_session=args.output_session,
    )
    view.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
