"""Napari application for geoseg segmentation editor (Shapes-primary).

Shapes layer is the ONLY interaction layer. Labels are computed from topology.
Native napari tools only — no custom shortcuts.

Usage:
    source .venv/bin/activate
    python -m geoseg.modules.editor.napari_app --labels path.npz --image path.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from napari.layers.shapes.shapes import ActionType
from napari.utils.notifications import show_info
from PIL import Image

from geoseg.modules.editor.editor_core import (
    RegionProperties,
    fill_boundary_gaps,
    labels_to_shapes,
)
from geoseg.modules.editor.editor_model import EditorModel


class GeoSegEditor:
    """Napari-based segmentation editor — Shapes-primary, topology-computed labels."""

    def __init__(
        self,
        labels: np.ndarray,
        image: np.ndarray | None = None,
        properties: dict[str, dict] | None = None,
    ) -> None:
        import napari

        self.viewer = napari.Viewer(title="geoseg editor")
        self._image_shape = labels.shape

        # Reference image layer
        if image is not None:
            self.viewer.add_image(image, name="reference", opacity=0.3)

        # Labels layer — read-only display of computed topology
        self.labels_layer = self.viewer.add_labels(labels, name="regions")
        self.labels_layer.editable = False
        self.labels_layer.mode = "pan_zoom"

        # Shapes layer — primary interaction layer
        initial_shapes = labels_to_shapes(labels)
        shape_types = ["polygon"] * len(initial_shapes)

        self.shapes_layer = self.viewer.add_shapes(
            initial_shapes,
            name="boundaries",
            shape_type=shape_types,
            edge_color="white",
            face_color="transparent",
            edge_width=2,
        )

        # Pure state model; kept in sync with the napari Shapes layer.
        self.model = EditorModel(
            image_shape=self._image_shape,
            properties=RegionProperties.from_dict(properties or {}),
        )
        for vertices, shape_type in zip(initial_shapes, shape_types):
            self.model.add_initial_shape(vertices, shape_type)

        # Prevent recursive updates while we mutate the Shapes layer ourselves.
        self._updating = False

        # Bind events
        self.shapes_layer.events.data.connect(self._on_shapes_changed)

    def _on_shapes_changed(self, event=None) -> None:
        """Recompute labels when shapes are finished changing.

        Napari emits transient events ('adding', 'changing', 'removing') while
        the user is still drawing or dragging. We ignore those and only act on
        the final 'added', 'changed', or 'removed' events.
        """
        if self._updating:
            return

        action = getattr(event, "action", None)
        if action in (ActionType.ADDING, ActionType.CHANGING, ActionType.REMOVING):
            return

        self._updating = True
        try:
            if action == ActionType.ADDED:
                self._handle_shapes_added(event)
            elif action == ActionType.CHANGED:
                self._handle_shapes_changed(event)
            elif action == ActionType.REMOVED:
                self._handle_shapes_removed(event)
            else:
                self._sync_all_from_layer()
                self._refresh_labels()
        finally:
            self._updating = False

    def _handle_shapes_added(self, event) -> None:
        """Snap and recompute labels for newly added shapes."""
        # Labels before the new shapes exist — used for snapping endpoints.
        labels_before = self.model.recompute_labels()

        new_indices = self._resolve_added_indices(event)
        for i in new_indices:
            self.model.add_shape(
                np.asarray(self.shapes_layer.data[i], dtype=float),
                self.shapes_layer.shape_type[i],
            )

        snapped_any = False
        for i in new_indices:
            model_index = self._layer_index_to_model_index(i, new_indices)
            if self.model.snap_shape(model_index, labels_before):
                snapped_any = True

        if snapped_any:
            self._sync_layer_from_model()

        self._refresh_labels()

    def _handle_shapes_changed(self, event) -> None:
        """Update model for modified shapes and recompute labels (no snapping)."""
        for i in event.data_indices:
            if not (0 <= i < len(self.shapes_layer.data)):
                continue
            self.model.update_shape(
                i, np.asarray(self.shapes_layer.data[i], dtype=float)
            )
        self._refresh_labels()

    def _handle_shapes_removed(self, event) -> None:
        """Remove shapes from model and recompute labels."""
        for i in sorted(event.data_indices, reverse=True):
            if not (0 <= i < len(self.shapes_layer.data) + len(event.data_indices)):
                continue
            self.model.remove_shape(i)
        self._refresh_labels()

    def _resolve_added_indices(self, event) -> list[int]:
        """Return the layer indices of shapes that were just added.

        Napari uses (-1,) as a placeholder during interactive drawing; the new
        shape is the last one in the layer.
        """
        data_indices = getattr(event, "data_indices", (-1,))
        if data_indices == (-1,) and len(self.shapes_layer.data) > 0:
            return [len(self.shapes_layer.data) - 1]
        return [i for i in data_indices if 0 <= i < len(self.shapes_layer.data)]

    def _layer_index_to_model_index(
        self, layer_index: int, added_indices: list[int]
    ) -> int:
        """Map a layer index to the corresponding model index.

        Since new shapes are appended, the model index is simply the layer
        index (model and layer stay aligned).
        """
        return layer_index

    def _sync_layer_from_model(self) -> None:
        """Push current model shapes into napari's Shapes layer (public setter).

        This triggers a CHANGED data event, which is ignored because
        `_updating` is True.
        """
        self.shapes_layer.data = [
            np.asarray(vertices, dtype=float) for vertices in self.model.shapes
        ]

    def _sync_all_from_layer(self) -> None:
        """Rebuild the model from the current Shapes layer data."""
        self.model = EditorModel(
            image_shape=self._image_shape,
            properties=self.model.properties,
        )
        for vertices, shape_type in zip(
            self.shapes_layer.data, self.shapes_layer.shape_type
        ):
            self.model.add_initial_shape(
                np.asarray(vertices, dtype=float), shape_type
            )

    def _refresh_labels(self) -> None:
        """Compute labels from the model and refresh the Labels layer."""
        self.labels_layer.data = self.model.recompute_labels()

    def save_shapes(self, path: str) -> None:
        """Export current shapes data to JSON."""
        import json

        shapes_data = []
        for vertices, shape_type in zip(self.model.shapes, self.model.shape_types):
            shapes_data.append(
                {
                    "type": shape_type,
                    "vertices": np.asarray(vertices).tolist(),
                }
            )

        payload = {
            "image_shape": list(self._image_shape),
            "shapes": shapes_data,
            "properties": self.model.properties.to_dict(),
        }

        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def save_labels(self, path: str) -> None:
        """Recompute labels from current shapes and save to NPZ.

        Boundary gaps (label 0 from thin separator lines) are filled by
        nearest-neighbour so downstream post-process treats them as regions.
        """
        labels = self.model.recompute_labels()
        labels = fill_boundary_gaps(labels)
        np.savez(path, labels=labels)

    def run(self) -> None:
        import napari

        napari.run()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_from_session(
    session_path: str, figure_id: str
) -> tuple[np.ndarray, np.ndarray | None, dict | None, str | None, str | None]:
    """Load labels/image/properties from SessionState, return paths for output."""
    from geoseg.session_state import load_session

    state = load_session(session_path)
    entry = next((e for e in state.workset if e.figure_id == figure_id), None)
    if entry is None:
        raise ValueError(f"Figure '{figure_id}' not found in session {session_path}")

    seg = entry.segmentation
    if seg is None:
        raise ValueError(f"Figure '{figure_id}' has no segmentation record")

    labels = np.load(seg.labels_path)["labels"]

    image = None
    if seg.overlay_path and Path(seg.overlay_path).exists():
        try:
            image = np.array(Image.open(seg.overlay_path))
        except Exception:
            pass

    properties = None
    if seg.shapes_path and Path(seg.shapes_path).exists():
        import json
        with open(seg.shapes_path) as f:
            shapes_data = json.load(f)
            properties = shapes_data.get("properties")

    result_dir = Path(seg.result_dir) if seg.result_dir else None
    output_labels = str(result_dir / "labels_edited.npz") if result_dir else None
    output_shapes = str(result_dir / "shapes.json") if result_dir else None

    return labels, image, properties, output_labels, output_shapes


def _resolve_file_inputs(
    labels_path: str,
    image_path: str | None,
    properties_path: str | None,
    shapes_path: str | None,
) -> tuple[np.ndarray, np.ndarray | None, dict | None]:
    """Load labels/image/properties from file paths.

    Priority for properties:
        1. ``--properties`` JSON file
        2. ``--shapes`` JSON file (reads its ``properties`` field)
        3. ``shapes.json`` in the same directory as ``--labels``
    """
    import json

    labels = np.load(labels_path)["labels"]

    image = None
    if image_path:
        image = np.array(Image.open(image_path))

    properties = None
    if properties_path and Path(properties_path).exists():
        with open(properties_path) as f:
            raw = json.load(f)
            properties = {
                str(k) if isinstance(k, int) else k: v for k, v in raw.items()
            }
    elif shapes_path and Path(shapes_path).exists():
        with open(shapes_path) as f:
            shapes_data = json.load(f)
            properties = shapes_data.get("properties")
    elif labels_path:
        auto_shapes = Path(labels_path).parent / "shapes.json"
        if auto_shapes.exists():
            with open(auto_shapes) as f:
                shapes_data = json.load(f)
                properties = shapes_data.get("properties")

    return labels, image, properties


def main() -> None:
    parser = argparse.ArgumentParser(description="geoseg napari editor")
    parser.add_argument("--labels", help="Path to labels .npz file")
    parser.add_argument("--image", help="Optional reference image path")
    parser.add_argument("--properties", help="Optional properties JSON path")
    parser.add_argument(
        "--shapes",
        help="Optional shapes JSON path; properties are read from its 'properties' field",
    )
    parser.add_argument("--session", help="Path to session JSON (alternative to --labels)")
    parser.add_argument("--figure", help="Figure ID within session (required with --session)")
    parser.add_argument(
        "--output-shapes",
        help="Path to save shapes JSON on exit (auto-saved after window closes)",
    )
    parser.add_argument(
        "--output-labels",
        help="Path to save recomputed labels NPZ on exit",
    )
    args = parser.parse_args()

    if args.session:
        if not args.figure:
            parser.error("--figure is required when using --session")
        labels, image, properties, default_labels, default_shapes = _resolve_from_session(
            args.session, args.figure
        )
        output_labels = args.output_labels or default_labels
        output_shapes = args.output_shapes or default_shapes
    elif args.labels:
        labels, image, properties = _resolve_file_inputs(
            args.labels, args.image, args.properties, args.shapes
        )
        output_labels = args.output_labels
        output_shapes = args.output_shapes
    else:
        parser.error("Either --labels or --session/--figure must be provided")

    editor = GeoSegEditor(labels, image=image, properties=properties)
    editor.run()

    # Auto-save on exit
    if output_shapes:
        editor.save_shapes(output_shapes)
        show_info(f"Shapes saved to {output_shapes}")

    if output_labels:
        editor.save_labels(output_labels)
        show_info(f"Labels saved to {output_labels}")

    # Update session state if applicable
    if args.session and output_labels and output_shapes:
        from geoseg.session_state import load_session, save_session, update_figure, FigureStatus, NapariEditRecord
        state = load_session(args.session)
        entry = next((e for e in state.workset if e.figure_id == args.figure), None)
        if entry and entry.segmentation:
            seg = entry.segmentation
            seg.edited_labels_path = output_labels
            seg.shapes_path = output_shapes
            seg.napari_edited = True
            seg.napari_edit = NapariEditRecord(
                edited_labels_path=output_labels,
                shapes_path=output_shapes,
            )
            state = update_figure(
                state,
                args.figure,
                status=FigureStatus.REVIEWED,
                segmentation=seg,
            )
            save_session(state, args.session)
            show_info(f"Session updated: {args.session}")


if __name__ == "__main__":
    main()
