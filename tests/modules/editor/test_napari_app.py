"""Integration tests for the napari editor application.

Run: python -m pytest tests/modules/editor/test_napari_app.py -v -m gui

These tests create real napari Viewers but only exercise programmatic
shape changes, avoiding mouse/keyboard automation. They are skipped when
napari cannot be initialized (e.g., headless CI without Qt).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

# Try to set napari test mode before importing viewer.
os.environ.setdefault("NAPARI_TEST", "1")

try:
    import napari

    NAPARI_AVAILABLE = True
except Exception:
    NAPARI_AVAILABLE = False

from geoseg.modules.editor.napari_app import GeoSegEditor

pytestmark = pytest.mark.gui


@pytest.fixture
def sample_labels() -> np.ndarray:
    labels = np.zeros((100, 100), dtype=np.int32)
    labels[20:80, 20:80] = 1
    return labels


@pytest.mark.skipif(not NAPARI_AVAILABLE, reason="napari not available")
class TestGeoSegEditorLifecycle:
    """End-to-end editor lifecycle with a real napari Viewer."""

    def test_initial_state_has_one_region(self, sample_labels: np.ndarray) -> None:
        editor = GeoSegEditor(sample_labels)
        try:
            unique = set(editor.labels_layer.data.flatten()) - {0}
            assert len(unique) == 1
            assert editor.labels_layer.mode == "pan_zoom"
            assert editor.labels_layer.editable is False
        finally:
            editor.viewer.close()

    def test_add_line_splits_region(self, sample_labels: np.ndarray) -> None:
        editor = GeoSegEditor(sample_labels)
        try:
            # Simulate finished line addition via the public API
            editor.shapes_layer.add(
                np.array([[50.0, 25.0], [50.0, 75.0]]),
                shape_type="line",
            )
            # Programmatic add emits ADDED; the handler should snap and recompute.
            # We expect 3 regions: left interior, right interior, and exterior.
            labels = editor.labels_layer.data
            unique = set(labels.flatten()) - {0}
            assert len(unique) == 3
        finally:
            editor.viewer.close()

    def test_remove_shape_merges_regions(self, sample_labels: np.ndarray) -> None:
        editor = GeoSegEditor(sample_labels)
        try:
            editor.shapes_layer.add(
                np.array([[50.0, 25.0], [50.0, 75.0]]),
                shape_type="line",
            )
            # Square interior split into two + exterior background = 3 regions.
            n_before = len(set(editor.labels_layer.data.flatten()) - {0})
            assert n_before == 3

            editor.shapes_layer.selected_data = {len(editor.shapes_layer.data) - 1}
            editor.shapes_layer.remove_selected()
            n_after = len(set(editor.labels_layer.data.flatten()) - {0})
            assert n_after == 2
        finally:
            editor.viewer.close()

    def test_change_event_updates_labels(self, sample_labels: np.ndarray) -> None:
        editor = GeoSegEditor(sample_labels)
        try:
            editor.shapes_layer.add(
                np.array([[50.0, 25.0], [50.0, 75.0]]),
                shape_type="line",
            )
            labels_before = editor.labels_layer.data.copy()

            # Programmatically edit the first shape
            new_data = list(editor.shapes_layer.data)
            new_data[0] = np.array([[30.0, 25.0], [30.0, 75.0]])
            editor.shapes_layer.data = new_data

            labels_after = editor.labels_layer.data
            assert not np.array_equal(labels_before, labels_after)
        finally:
            editor.viewer.close()

    def test_snap_only_once_per_new_shape(self, sample_labels: np.ndarray) -> None:
        editor = GeoSegEditor(sample_labels)
        try:
            editor.shapes_layer.add(
                np.array([[50.0, 30.0], [50.0, 70.0]]),
                shape_type="line",
            )
            last_idx = len(editor.model.shapes) - 1

            # Manually move the same shape; subsequent CHANGED events should not
            # trigger a re-snap.
            new_data = list(editor.shapes_layer.data)
            new_data[-1] = np.array([[50.0, 35.0], [50.0, 65.0]])
            editor.shapes_layer.data = new_data

            assert np.allclose(
                editor.model.get_shape(last_idx),
                np.array([[50.0, 35.0], [50.0, 65.0]]),
                atol=0.1,
            )
        finally:
            editor.viewer.close()


@pytest.mark.skipif(not NAPARI_AVAILABLE, reason="napari not available")
class TestGeoSegEditorSave:
    """Save/load round-trips."""

    def test_save_shapes_roundtrip(self, sample_labels: np.ndarray, tmp_path) -> None:
        editor = GeoSegEditor(sample_labels)
        try:
            shapes_path = str(tmp_path / "shapes.json")
            editor.save_shapes(shapes_path)

            import json
            with open(shapes_path) as f:
                payload = json.load(f)

            assert payload["image_shape"] == [100, 100]
            assert len(payload["shapes"]) >= 1
            assert "properties" in payload
        finally:
            editor.viewer.close()

    def test_save_labels_roundtrip(self, sample_labels: np.ndarray, tmp_path) -> None:
        editor = GeoSegEditor(sample_labels)
        try:
            labels_path = str(tmp_path / "labels_edited.npz")
            editor.save_labels(labels_path)

            loaded = np.load(labels_path)["labels"]
            assert loaded.shape == (100, 100)
        finally:
            editor.viewer.close()
