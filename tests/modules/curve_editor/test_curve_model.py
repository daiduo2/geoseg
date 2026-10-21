"""Headless tests for the control-point interface editor."""

from __future__ import annotations

import numpy as np
import pytest

from geoseg.modules.curve_editor import (
    EditorCommands,
    InterfaceModel,
    curve_from_interface,
    fit_freehand,
    resample_by_arc_length,
)


def _layered_model():
    """Three regions with a background band at the top, plus matching curves."""
    labels = np.zeros((120, 120), dtype=np.int32)
    labels[20:50] = 1
    labels[50:80] = 2
    labels[80:120] = 3
    background = np.zeros(labels.shape, dtype=bool)
    background[:20] = True

    model = InterfaceModel(labels.shape, background_mask=background, base_labels=labels)
    first = model.add_interface(
        [(0, 50), (30, 50), (60, 50), (90, 50), (119, 50)],
        level=0,
        initial=True,
    )
    second = model.add_interface(
        [(0, 80), (30, 80), (60, 80), (90, 80), (119, 80)],
        level=1,
        initial=True,
    )
    return model, first, second, labels, background


def test_curve_from_interface_interpolates_endpoints():
    model = InterfaceModel((100, 100))
    interface = model.add_interface(
        [(10, 50), (30, 46), (50, 54), (70, 47), (90, 50)]
    )
    curve = curve_from_interface(interface, samples=64)

    assert curve.shape == (64, 2)
    assert np.allclose(curve[0], [10, 50], atol=0.5)
    assert np.allclose(curve[-1], [90, 50], atol=0.5)
    assert curve[:, 1].min() > 40
    assert curve[:, 1].max() < 60


def test_unedited_model_round_trips_base_labels_exactly():
    model, _, _, base, _ = _layered_model()

    labels = model.to_labels()

    assert np.array_equal(labels, base)
    labels[10, 10] = 7
    assert np.array_equal(model.to_labels(), base), "to_labels must return a copy"


def test_move_point_is_local_and_preserves_background():
    model, first, _, base, background = _layered_model()
    point = first.points[2]
    model.move_point(first.id, point.id, 0, 6)

    labels = model.to_labels()

    assert not np.array_equal(labels, base)
    assert (labels[background] == 0).all()
    assert np.array_equal(labels[:20], base[:20])
    assert np.array_equal(labels[85:], base[85:])
    changed_rows = np.flatnonzero((labels != base).any(axis=1))
    assert changed_rows.size > 0
    assert changed_rows.max() <= 81
    assert set(np.unique(labels)) >= {0, 1, 2, 3}


def test_region_ids_stay_stable_after_edit():
    model, first, _, _, _ = _layered_model()
    seeds_before = dict(model.regions)

    model.move_point(first.id, first.points[1].id, 0, 4)
    after = model.to_labels()

    for region_id in (1, 2, 3):
        assert (after == region_id).any()
        seed_x, seed_y = seeds_before[region_id]
        assert after[int(round(seed_y)), int(round(seed_x))] == region_id


def test_insert_and_delete_keep_control_point_ids_stable():
    model = InterfaceModel((60, 100))
    interface = model.add_interface([(10, 20), (30, 20), (50, 20), (70, 20)])
    original = [point.id for point in interface.points]

    inserted = model.insert_point(interface.id, original[1], 40, 25)
    assert [point.id for point in interface.points] == [
        original[0],
        original[1],
        inserted.id,
        original[2],
        original[3],
    ]

    model.delete_point(interface.id, inserted.id)
    assert [point.id for point in interface.points] == original

    model.delete_point(interface.id, original[0])
    assert [point.id for point in interface.points] == original[1:]


def test_minimum_point_count_and_locking():
    model = InterfaceModel((60, 100))
    interface = model.add_interface([(10, 20), (30, 20)])
    with pytest.raises(ValueError):
        model.delete_point(interface.id, interface.points[0].id)

    model.set_locked(interface.id, interface.points[0].id, True)
    with pytest.raises(ValueError):
        model.move_point(interface.id, interface.points[0].id, 1, 1)

    model.move_interface(interface.id, 0, 5)
    assert interface.points[0].y == 20
    assert interface.points[1].y == 25


def test_closed_interface_creates_a_lens_region():
    model = InterfaceModel((100, 100))
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    points = [(50 + 30 * np.cos(a), 50 + 20 * np.sin(a)) for a in angles]
    interface = model.add_interface(points, closed=True)

    labels = model.to_labels()
    inside = int(labels[50, 50])
    outside = int(labels[2, 2])

    assert inside != 0
    assert outside != 0
    assert inside != outside

    model.move_point(interface.id, interface.points[0].id, 2, 0)
    moved = model.to_labels()
    assert moved[2, 2] == outside
    assert moved[50, 50] != 0


def test_mark_clean_after_initial_curves():
    model = InterfaceModel((40, 40))
    model.add_interface([(0, 20), (20, 18), (39, 20)], initial=True)
    model.mark_clean()

    first = model.to_labels()
    second = model.to_labels()
    assert np.array_equal(first, second)
    assert first[0, 0] != 0


def test_fit_freehand_and_arc_length_resampling():
    trace = np.array([[0, 0], [10, 0], [20, 0], [30, 0]], dtype=float)
    fitted = fit_freehand(trace, n_points=4)
    assert len(fitted) == 4
    assert fitted[0] == pytest.approx((0.0, 0.0))
    assert fitted[-1] == pytest.approx((30.0, 0.0))

    model = InterfaceModel((100, 100))
    interface = model.add_interface([(10, 40), (40, 20), (70, 40), (90, 60)])
    dense = resample_by_arc_length(interface, 25)
    assert dense.shape == (25, 2)
    assert np.allclose(dense[0], [10, 40], atol=1.0)
    assert np.allclose(dense[-1], [90, 60], atol=1.0)


def test_commands_return_stable_payloads():
    model = InterfaceModel((60, 80))
    commands = EditorCommands(model)
    added = commands.add_interface([(10, 20), (30, 20), (50, 20)])
    point_id = added["points"][1]

    moved = commands.move_point(added["interface"], point_id, 0, 5)

    assert moved["status"] == "ok"
    assert moved["point"] == point_id
    assert moved["y"] == 25
    assert model.to_dict()["interfaces"][0]["id"] == added["interface"]
