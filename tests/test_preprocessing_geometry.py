import cv2
import numpy as np
import pytest

from geoseg.preprocessing.geometry import rectify_quadrilateral
from geoseg.preprocessing.absorption import fill_mask_nearest_along_axis


def test_rectify_quadrilateral_maps_corners() -> None:
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    points = np.array([[10, 10], [90, 5], [85, 70], [15, 75]], dtype=np.float32)
    for point, color in zip(
        points.astype(int),
        ([255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]),
        strict=True,
    ):
        cv2.circle(image, tuple(point), 4, color, -1)

    result, transform = rectify_quadrilateral(
        image, points, (120, 60), interpolation=cv2.INTER_NEAREST
    )

    assert result.shape == (60, 120, 3)
    assert transform.shape == (3, 3)
    assert result[0:5, 0:5, 0].max() == 255
    assert result[0:5, -5:, 1].max() == 255
    assert result[-5:, -5:, 2].max() == 255


def test_rectify_quadrilateral_validates_inputs() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape"):
        rectify_quadrilateral(image, np.zeros((3, 2)))
    with pytest.raises(ValueError, match="positive"):
        rectify_quadrilateral(image, np.zeros((4, 2)), (0, 10))


def test_fill_mask_nearest_along_axis_preserves_rows() -> None:
    image = np.zeros((2, 5, 3), dtype=np.uint8)
    image[0] = np.array([10, 10, 90, 10, 10])[:, None]
    image[1] = np.array([200, 200, 90, 200, 200])[:, None]
    mask = np.zeros((2, 5), dtype=bool)
    mask[:, 2] = True

    result = fill_mask_nearest_along_axis(image, mask, axis="horizontal")

    assert result[0, 2].tolist() == [10, 10, 10]
    assert result[1, 2].tolist() == [200, 200, 200]
