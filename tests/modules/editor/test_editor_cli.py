"""Tests for editor CLI input resolution.

Run: python -m pytest geoseg/modules/editor/test_editor_cli.py -v
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest
from PIL import Image

from geoseg.modules.editor.napari_app import _resolve_file_inputs, _resolve_from_session


class TestResolveFileInputs:
    """Loading labels, image, and properties from file paths."""

    def test_loads_labels_and_image(self, tmp_path) -> None:
        labels = np.zeros((20, 20), dtype=np.int32)
        labels[5:15, 5:15] = 1
        labels_path = str(tmp_path / "labels.npz")
        np.savez(labels_path, labels=labels)

        image = np.full((20, 20, 3), 128, dtype=np.uint8)
        image_path = str(tmp_path / "image.png")
        from PIL import Image
        Image.fromarray(image).save(image_path)

        loaded_labels, loaded_image, loaded_props = _resolve_file_inputs(
            labels_path, image_path, None, None
        )
        assert np.array_equal(loaded_labels, labels)
        assert loaded_image is not None
        assert loaded_image.shape == image.shape
        assert loaded_props is None

    def test_loads_properties_from_explicit_properties_path(self, tmp_path) -> None:
        labels = np.zeros((20, 20), dtype=np.int32)
        labels_path = str(tmp_path / "labels.npz")
        np.savez(labels_path, labels=labels)

        props = {"layer_1": {"Vp": 3000.0}}
        props_path = str(tmp_path / "props.json")
        with open(props_path, "w") as f:
            json.dump(props, f)

        _, _, loaded_props = _resolve_file_inputs(
            labels_path, None, props_path, None
        )
        assert loaded_props == props

    def test_loads_properties_from_shapes_json(self, tmp_path) -> None:
        labels = np.zeros((20, 20), dtype=np.int32)
        labels_path = str(tmp_path / "labels.npz")
        np.savez(labels_path, labels=labels)

        props = {"layer_1": {"Vp": 3000.0}}
        shapes_path = str(tmp_path / "shapes.json")
        with open(shapes_path, "w") as f:
            json.dump(
                {"image_shape": [20, 20], "shapes": [], "properties": props}, f
            )

        _, _, loaded_props = _resolve_file_inputs(
            labels_path, None, None, shapes_path
        )
        assert loaded_props == props

    def test_falls_back_to_same_dir_shapes_json(self, tmp_path) -> None:
        labels = np.zeros((20, 20), dtype=np.int32)
        labels_path = str(tmp_path / "labels_edited.npz")
        np.savez(labels_path, labels=labels)

        props = {"layer_1": {"Vp": 3000.0}}
        shapes_path = str(tmp_path / "shapes.json")
        with open(shapes_path, "w") as f:
            json.dump(
                {"image_shape": [20, 20], "shapes": [], "properties": props}, f
            )

        _, _, loaded_props = _resolve_file_inputs(
            labels_path, None, None, None
        )
        assert loaded_props == props

    def test_explicit_properties_override_shapes_json(self, tmp_path) -> None:
        labels = np.zeros((20, 20), dtype=np.int32)
        labels_path = str(tmp_path / "labels.npz")
        np.savez(labels_path, labels=labels)

        props_explicit = {"layer_1": {"Vp": 4000.0}}
        props_shapes = {"layer_1": {"Vp": 3000.0}}

        props_path = str(tmp_path / "props.json")
        with open(props_path, "w") as f:
            json.dump(props_explicit, f)

        shapes_path = str(tmp_path / "shapes.json")
        with open(shapes_path, "w") as f:
            json.dump(
                {"image_shape": [20, 20], "shapes": [], "properties": props_shapes}, f
            )

        _, _, loaded_props = _resolve_file_inputs(
            labels_path, None, props_path, shapes_path
        )
        assert loaded_props == props_explicit

    def test_missing_files_return_none_properties(self, tmp_path) -> None:
        labels = np.zeros((20, 20), dtype=np.int32)
        labels_path = str(tmp_path / "labels.npz")
        np.savez(labels_path, labels=labels)

        _, _, loaded_props = _resolve_file_inputs(
            labels_path, None, None, None
        )
        assert loaded_props is None


class TestResolveFromSession:
    """Loading labels/image/properties from SessionState."""

    def test_resolve_from_session(self, tmp_path) -> None:
        from geoseg.session_state import (
            FigureEntry,
            SegmentationRecord,
            SessionState,
        )

        result_dir = tmp_path / "fig1"
        result_dir.mkdir()
        labels = np.zeros((20, 20), dtype=np.int32)
        labels[5:15, 5:15] = 1
        labels_path = str(result_dir / "labels.npz")
        np.savez(labels_path, labels=labels)

        overlay = np.full((20, 20, 3), 128, dtype=np.uint8)
        overlay_path = str(result_dir / "overlay.png")
        Image.fromarray(overlay).save(overlay_path)

        props = {"layer_1": {"Vp": 3000.0}}
        shapes_path = str(result_dir / "shapes.json")
        with open(shapes_path, "w") as f:
            json.dump(
                {"image_shape": [20, 20], "shapes": [], "properties": props}, f
            )

        session = SessionState(
            workset=[
                FigureEntry(
                    figure_id="fig1",
                    source_path=str(tmp_path / "fig1.png"),
                    segmentation=SegmentationRecord(
                        result_dir=str(result_dir),
                        engine="v4_kmeans",
                        n_layers=2,
                        quality_score=0.8,
                        overlay_path=overlay_path,
                        labels_path=labels_path,
                        edited_labels_path=labels_path,
                        shapes_path=shapes_path,
                    ),
                )
            ]
        )
        session_path = str(tmp_path / "session.json")
        with open(session_path, "w") as f:
            f.write(session.model_dump_json())

        loaded_labels, loaded_image, loaded_props, default_labels, default_shapes = (
            _resolve_from_session(session_path, "fig1")
        )
        assert np.array_equal(loaded_labels, labels)
        assert loaded_image is not None
        assert loaded_image.shape == overlay.shape
        assert loaded_props == props
        assert default_labels == str(result_dir / "labels_edited.npz")
        assert default_shapes == shapes_path

    def test_resolve_missing_figure_raises(self, tmp_path) -> None:
        from geoseg.session_state import SessionState

        session = SessionState(workset=[])
        session_path = str(tmp_path / "session.json")
        with open(session_path, "w") as f:
            f.write(session.model_dump_json())

        with pytest.raises(ValueError, match="not found"):
            _resolve_from_session(session_path, "missing")


class TestMainCLI:
    """Tests for napari_app.main() entry point."""

    def test_main_with_labels_path(self, tmp_path, monkeypatch) -> None:
        import geoseg.modules.editor.napari_app as napari_app

        labels = np.zeros((20, 20), dtype=np.int32)
        labels[5:15, 5:15] = 1
        labels_path = str(tmp_path / "labels.npz")
        np.savez(labels_path, labels=labels)

        calls = []
        editor_inst = None

        class FakeEditor:
            def __init__(self, labels, image=None, properties=None):
                nonlocal editor_inst
                editor_inst = self
                self.labels = labels
                self.image = image
                self.properties = properties
                self.save_shapes_path = None
                self.save_labels_path = None

            def run(self):
                calls.append("run")

            def save_shapes(self, path):
                self.save_shapes_path = path

            def save_labels(self, path):
                self.save_labels_path = path

        monkeypatch.setattr(napari_app, "GeoSegEditor", FakeEditor)
        monkeypatch.setattr(sys, "argv", [
            "napari_app",
            "--labels", labels_path,
            "--output-shapes", str(tmp_path / "out_shapes.json"),
            "--output-labels", str(tmp_path / "out_labels.npz"),
        ])

        napari_app.main()

        assert len(calls) == 1
        assert editor_inst.save_shapes_path == str(tmp_path / "out_shapes.json")
        assert editor_inst.save_labels_path == str(tmp_path / "out_labels.npz")

    def test_main_with_session(self, tmp_path, monkeypatch) -> None:
        import geoseg.modules.editor.napari_app as napari_app
        from geoseg.session_state import (
            FigureEntry,
            SegmentationRecord,
            SessionState,
        )

        result_dir = tmp_path / "fig1"
        result_dir.mkdir()
        labels = np.zeros((20, 20), dtype=np.int32)
        labels_path = str(result_dir / "labels.npz")
        np.savez(labels_path, labels=labels)

        session = SessionState(
            workset=[
                FigureEntry(
                    figure_id="fig1",
                    source_path=str(tmp_path / "fig1.png"),
                    segmentation=SegmentationRecord(
                        result_dir=str(result_dir),
                        engine="v4_kmeans",
                        n_layers=2,
                        quality_score=0.8,
                        overlay_path="",
                        labels_path=labels_path,
                        edited_labels_path=labels_path,
                    ),
                )
            ]
        )
        session_path = str(tmp_path / "session.json")
        with open(session_path, "w") as f:
            f.write(session.model_dump_json())

        calls = []
        editor_inst = None

        class FakeEditor:
            def __init__(self, labels, image=None, properties=None):
                nonlocal editor_inst
                editor_inst = self
                self.labels = labels
                self.image = image
                self.properties = properties
                calls.append(("init", image, properties))

            def run(self):
                calls.append("run")

            def save_shapes(self, path):
                pass

            def save_labels(self, path):
                pass

        monkeypatch.setattr(napari_app, "GeoSegEditor", FakeEditor)
        monkeypatch.setattr(sys, "argv", [
            "napari_app",
            "--session", session_path,
            "--figure", "fig1",
        ])

        napari_app.main()

        assert any(c[0] == "init" for c in calls)
        assert "run" in calls
        assert editor_inst is not None

    def test_main_errors_without_labels_or_session(self, tmp_path, monkeypatch) -> None:
        import geoseg.modules.editor.napari_app as napari_app

        monkeypatch.setattr(sys, "argv", ["napari_app"])
        with pytest.raises(SystemExit):
            napari_app.main()
