"""GUI smoke tests for the napari curve editor (programmatic, no mouse)."""

from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("NAPARI_TEST", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from geoseg.modules.curve_editor import InterfaceModel, model_from_labels  # noqa: E402
from geoseg.modules.curve_editor.view import CurveEditorView  # noqa: E402

pytestmark = pytest.mark.gui


def test_empty_model_draft_creates_interface():
    model = InterfaceModel((80, 120))
    view = CurveEditorView(model)
    try:
        assert set(np.unique(view.labels_layer.data)) == {1}
        view.draft_layer.data = np.array([[20, 30], [30, 60], [40, 90]], dtype=float)
        view._commit_draft()
        assert len(model.interfaces) == 1
        assert view.draft_layer.visible is False
        assert set(np.unique(view.labels_layer.data)) == {0, 1, 2}
    finally:
        view.viewer.close()


def test_imported_labels_move_point_updates_labels():
    labels = np.zeros((100, 120), dtype=np.int32)
    labels[20:45] = 1
    labels[45:75] = 2
    labels[75:100] = 3
    model = model_from_labels(labels)
    assert np.array_equal(model.to_labels(), labels)

    view = CurveEditorView(model)
    try:
        assert len(view.points_layer.data) > 0
        data = np.asarray(view.points_layer.data).copy()
        data[0, 0] += 4
        view.points_layer.data = data
        view._flush_refresh()

        edited = view.labels_layer.data
        assert not np.array_equal(edited, labels)
        assert edited[labels == 0].max() == 0
        assert set(np.unique(edited)) >= {1, 2}
    finally:
        view.viewer.close()


def test_region_freehand_callback_snaps_and_fills():
    labels = np.zeros((120, 160), dtype=np.int32)
    labels[10:110, 10:150] = 1
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[40:80, 50:110] = [220, 220, 220]
    model = model_from_labels(labels)
    view = CurveEditorView(model, image)
    try:
        view.region_mode = True
        event = SimpleNamespace(
            type="mouse_press", position=(60.0, 80.0), handled=False
        )
        callback = view._on_viewer_drag(view.viewer, event)
        next(callback)
        for angle in np.linspace(0, 2 * np.pi, 60, endpoint=False):
            event.type = "mouse_move"
            event.position = (
                60 + 25 * np.sin(angle),
                80 + 36 * np.cos(angle),
            )
            next(callback)
        event.type = "mouse_release"
        with pytest.raises(StopIteration):
            next(callback)

        assert view.region_phase == "awaiting_interior"
        assert view._region_control_points is not None
        view._commit_region_interior((80, 60))

        edited = view.labels_layer.data
        region_ids = set(np.unique(edited).tolist()) - {0, 1}
        assert region_ids
        assert view.region_phase == "idle"
    finally:
        view.viewer.close()
