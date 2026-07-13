"""Unit tests for the editor model (napari-independent state machine).

Run: python -m pytest geoseg/modules/editor/test_editor_model.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from geoseg.modules.editor.editor_model import EditorModel


class TestEditorModelBasics:
    """Core state management without napari."""

    def test_empty_model_yields_single_region(self) -> None:
        """No shapes => the whole canvas is one region."""
        model = EditorModel((100, 100))
        labels = model.recompute_labels()

        assert labels.shape == (100, 100)
        # Borders are boundary (0), interior is one label
        assert labels[0, 0] == 0
        assert labels[50, 50] != 0
        assert len(set(labels.flatten()) - {0}) == 1

    def test_add_line_splits_region(self) -> None:
        """Adding a finished horizontal line splits the canvas."""
        model = EditorModel((100, 100))
        model.add_shape(np.array([[50.0, 0.0], [50.0, 99.0]]), "line")
        labels = model.recompute_labels()

        unique = set(labels.flatten())
        assert unique == {0, 1, 2}
        assert labels[25, 50] in (1, 2)
        assert labels[75, 50] in (1, 2)
        assert labels[25, 50] != labels[75, 50]

    def test_remove_shape_merges_regions(self) -> None:
        """Removing a split line merges the two regions."""
        model = EditorModel((100, 100))
        idx = model.add_shape(np.array([[50.0, 0.0], [50.0, 99.0]]), "line")
        model.remove_shape(idx)
        labels = model.recompute_labels()

        assert len(set(labels.flatten()) - {0}) == 1


class TestSnapping:
    """Endpoint snapping behavior."""

    def test_new_line_is_snapped_once(self) -> None:
        """A newly added line gets snapped exactly once."""
        model = EditorModel((100, 100))
        # Square region in the middle: the line should snap to x=20 and x=80
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[20:80, 20:80] = 1

        idx = model.add_shape(np.array([[50.0, 30.0], [50.0, 70.0]]), "line")
        changed = model.snap_shape(idx, labels)

        assert changed is True
        snapped = model.get_shape(idx)
        assert snapped[0, 1] == pytest.approx(20.0, abs=2.0)
        assert snapped[-1, 1] == pytest.approx(80.0, abs=2.0)

    def test_already_snapped_line_not_resnapped(self) -> None:
        """Calling snap twice on the same shape does nothing the second time."""
        model = EditorModel((100, 100))
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[20:80, 20:80] = 1

        idx = model.add_shape(np.array([[50.0, 30.0], [50.0, 70.0]]), "line")
        model.snap_shape(idx, labels)
        first = model.get_shape(idx).copy()

        changed = model.snap_shape(idx, labels)
        assert changed is False
        assert np.allclose(model.get_shape(idx), first)

    def test_modified_line_is_not_auto_resnapped(self) -> None:
        """After explicit update, the shape stays where we put it."""
        model = EditorModel((100, 100))
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[20:80, 20:80] = 1

        idx = model.add_shape(np.array([[50.0, 30.0], [50.0, 70.0]]), "line")
        model.snap_shape(idx, labels)

        # User drags the endpoint away from boundary
        model.update_shape(idx, np.array([[50.0, 35.0], [50.0, 65.0]]))
        changed = model.snap_shape(idx, labels)

        assert changed is False
        assert model.get_shape(idx)[0, 1] == pytest.approx(35.0, abs=0.1)

    def test_new_path_is_snapped_once(self) -> None:
        """A newly added path gets its endpoints snapped."""
        model = EditorModel((100, 100))
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[20:80, 20:80] = 1

        vertices = np.array([[50.0, 30.0], [50.0, 50.0], [50.0, 70.0]])
        idx = model.add_shape(vertices, "path")
        changed = model.snap_shape(idx, labels)

        assert changed is True
        snapped = model.get_shape(idx)
        assert snapped[0, 1] == pytest.approx(20.0, abs=2.0)
        assert snapped[-1, 1] == pytest.approx(80.0, abs=2.0)
        # Middle vertex preserved
        assert snapped[1, 1] == pytest.approx(50.0, abs=0.1)

    def test_polygon_is_not_snapped(self) -> None:
        """Polygons are left untouched by snapping."""
        model = EditorModel((100, 100))
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[20:80, 20:80] = 1

        vertices = np.array([[30.0, 30.0], [30.0, 70.0], [70.0, 70.0], [70.0, 30.0]])
        idx = model.add_shape(vertices, "polygon")
        changed = model.snap_shape(idx, labels)

        assert changed is False
        assert np.allclose(model.get_shape(idx), vertices)


class TestShapeTracking:
    """Shape lifecycle tracking."""

    def test_add_returns_index(self) -> None:
        model = EditorModel((10, 10))
        idx = model.add_shape(np.array([[0.0, 0.0], [1.0, 1.0]]), "line")
        assert idx == 0

    def test_remove_updates_indices(self) -> None:
        model = EditorModel((10, 10))
        model.add_shape(np.array([[0.0, 0.0], [1.0, 1.0]]), "line")
        model.add_shape(np.array([[2.0, 2.0], [3.0, 3.0]]), "line")
        model.remove_shape(0)

        assert len(model.shapes) == 1
        assert model.get_shape(0) is not None

    def test_recompute_after_update(self) -> None:
        """Updating a shape changes recomputed labels."""
        model = EditorModel((100, 100))
        idx = model.add_shape(np.array([[50.0, 0.0], [50.0, 99.0]]), "line")
        labels_before = model.recompute_labels()

        model.update_shape(idx, np.array([[25.0, 0.0], [25.0, 99.0]]))
        labels_after = model.recompute_labels()

        assert not np.array_equal(labels_before, labels_after)
        assert labels_after[12, 50] in set(labels_after.flatten()) - {0}
        assert labels_after[50, 50] in set(labels_after.flatten()) - {0}


class TestEditorModelEdgeCases:
    """Error paths and boundary conditions."""

    def test_update_out_of_range_raises(self) -> None:
        model = EditorModel((10, 10))
        with pytest.raises(IndexError):
            model.update_shape(0, np.array([[0.0, 0.0], [1.0, 1.0]]))

    def test_remove_out_of_range_raises(self) -> None:
        model = EditorModel((10, 10))
        with pytest.raises(IndexError):
            model.remove_shape(0)

    def test_get_shape_out_of_range_raises(self) -> None:
        model = EditorModel((10, 10))
        with pytest.raises(IndexError):
            model.get_shape(0)

    def test_snap_shape_out_of_range_raises(self) -> None:
        model = EditorModel((10, 10))
        labels = np.zeros((10, 10), dtype=np.int32)
        with pytest.raises(IndexError):
            model.snap_shape(0, labels)

    def test_snap_short_line_marks_snapped(self) -> None:
        """A line with <2 vertices is marked snapped but not changed."""
        model = EditorModel((100, 100))
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[20:80, 20:80] = 1
        idx = model.add_shape(np.array([[50.0, 50.0]]), "line")
        changed = model.snap_shape(idx, labels)
        assert changed is False
        assert idx in model._snapped_indices

    def test_snap_line_target_is_boundary(self) -> None:
        """Line midpoint on label 0 yields no snap."""
        model = EditorModel((100, 100))
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[20:80, 20:80] = 1
        # Midpoint of the line lies outside label 1 (on label 0)
        idx = model.add_shape(np.array([[50.0, 19.0], [50.0, 18.0]]), "line")
        changed = model.snap_shape(idx, labels)
        assert changed is False

    def test_snap_path_both_targets_zero(self) -> None:
        """Path whose endpoints both map to label 0 yields no snap."""
        model = EditorModel((100, 100))
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[20:80, 20:80] = 1
        # First and last segment midpoints lie outside label 1
        vertices = np.array([
            [50.0, 18.0], [50.0, 19.0], [50.0, 50.0],
            [50.0, 81.0], [50.0, 82.0],
        ])
        idx = model.add_shape(vertices, "path")
        changed = model.snap_shape(idx, labels)
        assert changed is False

    def test_polygon_marked_snapped(self) -> None:
        """Polygons are marked snapped and never auto-snapped."""
        model = EditorModel((100, 100))
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[20:80, 20:80] = 1
        vertices = np.array([[30.0, 30.0], [30.0, 70.0], [70.0, 70.0], [70.0, 30.0]])
        idx = model.add_shape(vertices, "polygon")
        changed = model.snap_shape(idx, labels)
        assert changed is False
        assert idx in model._snapped_indices
