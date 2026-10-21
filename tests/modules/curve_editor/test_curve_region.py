"""Headless tests for freehand region snapping and interior filling."""

from __future__ import annotations

import numpy as np
from skimage.draw import polygon2mask

from geoseg.modules.curve_editor import (
    EditorCommands,
    InterfaceModel,
    interior_point,
    model_from_labels,
    select_control_points,
    snap_closed_curve,
    snap_region,
)


def _rectangle_image():
    h, w = 200, 220
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[50:150, 60:170] = [220, 220, 220]
    truth = np.zeros((h, w), dtype=bool)
    truth[50:150, 60:170] = True
    return image, truth


def _rough_stroke():
    rng = np.random.default_rng(0)
    angle = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    return np.column_stack(
        [
            115 + 64 * np.cos(angle) + rng.normal(0, 4, len(angle)),
            100 + 57 * np.sin(angle) + rng.normal(0, 4, len(angle)),
        ]
    )


def _mask(boundary, shape):
    return polygon2mask(shape, np.column_stack([boundary[:, 1], boundary[:, 0]]))


def test_snap_moves_rough_stroke_to_image_boundary():
    image, truth = _rectangle_image()
    stroke = _rough_stroke()
    raw = _mask(stroke, truth.shape)
    snapped = _mask(snap_closed_curve(image, stroke, search_radius=16), truth.shape)

    raw_iou = (raw & truth).sum() / (raw | truth).sum()
    snapped_iou = (snapped & truth).sum() / (snapped | truth).sum()

    assert snapped_iou > raw_iou + 0.10
    assert snapped_iou > 0.90


def test_snap_region_returns_curvature_control_points_and_interior():
    image, truth = _rectangle_image()
    result = snap_region(
        image, _rough_stroke(), interior=(115, 100), search_radius=16
    )

    assert len(result.control_points) >= 4
    assert result.mean_gradient > 0.0
    assert _mask(result.boundary, truth.shape)[100, 115]
    assert 60 <= result.interior[0] <= 170
    assert 50 <= result.interior[1] <= 150


def test_control_points_include_rectangle_corners():
    boundary = np.array([[60, 50], [170, 50], [170, 150], [60, 150]], dtype=float)
    control = select_control_points(boundary, samples=256, min_spacing=10)

    corners = np.array([[60, 50], [170, 50], [170, 150], [60, 150]], dtype=float)
    for corner in corners:
        distance = np.hypot(
            control[:, 0] - corner[0], control[:, 1] - corner[1]
        ).min()
        assert distance < 10, f"no control point near {corner}"


def test_interior_point_corrects_click_outside():
    boundary = np.array([[40, 40], [120, 40], [120, 110], [40, 110]], dtype=float)
    shape = (160, 180)

    interior = interior_point(boundary, shape, preferred=(170, 150))
    mask = polygon2mask(shape, np.column_stack([boundary[:, 1], boundary[:, 0]]))

    assert mask[int(round(interior[1])), int(round(interior[0]))]


def test_add_region_interface_creates_a_new_stable_region():
    labels = np.zeros((120, 140), dtype=np.int32)
    labels[20:100, 20:120] = 1
    model = model_from_labels(labels)
    boundary = np.array([[55, 45], [100, 45], [100, 75], [55, 75]], dtype=float)

    interface = model.add_region_interface(boundary, interior=(70, 60), basis="B1")
    output = model.to_labels()

    region_id = interface.metadata["region_id"]
    assert region_id > 1
    assert (output == region_id).sum() > 0
    assert (output == 1).sum() > 0
    assert model.regions[region_id] == (70.0, 60.0)


def test_commands_add_region_payload():
    model = InterfaceModel((80, 100))
    commands = EditorCommands(model)
    result = commands.add_region(
        [(20, 20), (70, 20), (70, 60), (20, 60)], interior=(40, 40)
    )

    assert result["status"] == "ok"
    assert result["region_id"] == 1
    assert result["interior"] == [40.0, 40.0]
    assert result["points"]
