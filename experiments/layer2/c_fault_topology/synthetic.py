"""Synthetic fault / pinch-out cases for Layer 2 experiment C.

Each case is a small integer label array plus an *explicit* fault mask (or
None for the pinch-out).  The fault mask is ground truth supplied by the case
generator; it is never inferred from the curated panels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.ndimage import binary_dilation


@dataclass
class OffsetProbe:
    """Compare the vertical level of a boundary on the two sides of a fault."""

    name: str
    labels_pair: tuple[int, int]
    x_left: float
    x_right: float
    expected_throw: float | None = None


@dataclass
class SyntheticCase:
    case_id: str
    labels: np.ndarray
    fault_mask: np.ndarray | None
    fault_explicit: bool
    probes: list[OffsetProbe] = field(default_factory=list)
    fault_polyline: list[tuple[float, float]] | None = None
    description: str = ""
    layer_note: str = ""


def _shape(height: int, width: int) -> tuple[int, int]:
    return int(height), int(width)


def _fault_x_at_y(polyline: list[tuple[float, float]], y: np.ndarray) -> np.ndarray:
    ys = np.array([point[1] for point in polyline], dtype=float)
    xs = np.array([point[0] for point in polyline], dtype=float)
    order = np.argsort(ys)
    ys, xs = ys[order], xs[order]
    # np.interp clamps outside, which is what we want for full-height faults.
    return np.interp(y, ys, xs)


def _raster_fault(shape: tuple[int, int], polyline: list[tuple[float, float]], width: int = 1) -> np.ndarray:
    height, width_px = shape
    canvas = Image.new("L", (width_px, height), 0)
    draw = ImageDraw.Draw(canvas)
    draw.line([tuple(map(float, point)) for point in polyline], fill=1, width=width)
    mask = np.asarray(canvas, dtype=bool)
    if width <= 1:
        mask = binary_dilation(mask, iterations=1)
    return mask


def _offset_layers(
    shape: tuple[int, int],
    polyline: list[tuple[float, float]],
    interface,
    *,
    top_label: int = 1,
    bottom_label: int = 2,
    throw: float = 0.0,
) -> np.ndarray:
    """Two layers offset vertically across a (possibly listric) fault plane."""
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width]
    fault_x = _fault_x_at_y(polyline, yy.astype(float))
    left = xx < fault_x
    base = interface(xx.astype(float))
    offset = np.where(left, base, base + throw)
    labels = np.full(shape, top_label, dtype=np.int32)
    labels[yy >= offset] = bottom_label
    return labels


def _three_layers_offset(
    shape: tuple[int, int],
    fault_x: float,
    top_interface,
    thickness: float,
    throw: float,
) -> np.ndarray:
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width]
    left = xx < fault_x
    y1 = top_interface(xx.astype(float)) + np.where(left, 0.0, throw)
    y2 = y1 + thickness
    labels = np.ones(shape, dtype=np.int32)
    labels[yy >= y1] = 2
    labels[yy >= y2] = 3
    return labels


def normal_fault() -> SyntheticCase:
    height, width = _shape(220, 360)
    fault_x = 200.0
    polyline = [(fault_x, 0.0), (fault_x, float(height - 1))]
    throw = 26.0

    def interface(x: np.ndarray) -> np.ndarray:
        return 100.0 + 3.5 * np.sin(x / 34.0) + 0.02 * (x - 180.0)

    labels = _offset_layers((height, width), polyline, interface, throw=throw)
    fault_mask = _raster_fault((height, width), polyline, width=1)
    return SyntheticCase(
        case_id="syn_normal_fault",
        labels=labels,
        fault_mask=fault_mask,
        fault_explicit=True,
        probes=[
            OffsetProbe("normal_throw", (1, 2), fault_x - 16.0, fault_x + 16.0, throw),
        ],
        fault_polyline=polyline,
        description=(
            "Two layers separated by a gently undulating interface. A vertical normal fault at "
            f"x={fault_x:.0f} offsets the interface by {throw:.0f} px (hanging wall down). "
            "The exact 1-2 boundary is one continuous path with a crisp vertical jog."
        ),
        layer_note="labels: 1 = hanging-wall/top unit, 2 = footwall/bottom unit",
    )


def listric_kink() -> SyntheticCase:
    height, width = _shape(250, 380)
    polyline = [
        (304.0, 0.0),
        (300.0, 60.0),
        (296.0, 105.0),
        (270.0, 150.0),
        (233.0, 190.0),
        (185.0, 222.0),
        (145.0, 246.0),
    ]
    throw = 22.0

    def interface(x: np.ndarray) -> np.ndarray:
        return 88.0 + 0.11 * (x - 150.0) + 4.0 * np.sin(x / 33.0)

    labels = _offset_layers((height, width), polyline, interface, throw=throw)
    fault_mask = _raster_fault((height, width), polyline, width=1)
    return SyntheticCase(
        case_id="syn_listric_kink",
        labels=labels,
        fault_mask=fault_mask,
        fault_explicit=True,
        probes=[
            OffsetProbe("listric_upper", (1, 2), 270.0, 322.0, throw),
            OffsetProbe("listric_lower", (1, 2), 215.0, 252.0, throw),
        ],
        fault_polyline=polyline,
        description=(
            "Two layers cut by a listric fault. The fault is steep at the top, kinks at "
            "(296,105), then flattens into a detachment. The 1-2 interface is offset by "
            f"{throw:.0f} px across the fault; the fault plane itself is not a label boundary."
        ),
        layer_note="labels: 1 = upper plate, 2 = lower plate",
    )


def three_layer_fault() -> SyntheticCase:
    height, width = _shape(230, 380)
    fault_x = 219.0
    throw = 18.0
    thickness = 12.0
    polyline = [(fault_x, 0.0), (fault_x, float(height - 1))]

    def top_interface(x: np.ndarray) -> np.ndarray:
        return 68.0 + 5.0 * np.sin(x / 31.0) + 0.015 * (x - 190.0)

    labels = _three_layers_offset((height, width), fault_x, top_interface, thickness, throw)
    fault_mask = _raster_fault((height, width), polyline, width=1)
    return SyntheticCase(
        case_id="syn_three_layer_fault",
        labels=labels,
        fault_mask=fault_mask,
        fault_explicit=True,
        probes=[
            OffsetProbe("thin_top", (1, 2), fault_x - 14.0, fault_x + 14.0, throw),
            OffsetProbe("thin_bottom", (2, 3), fault_x - 14.0, fault_x + 14.0, throw),
        ],
        fault_polyline=polyline,
        description=(
            "Three layers: a 12 px thin middle layer between two thicker units, cut by a "
            f"vertical fault at x={fault_x:.0f} with {throw:.0f} px throw. Both interfaces "
            "jog at the fault; the thin layer must survive both offsets."
        ),
        layer_note="labels: 1 = top, 2 = thin middle layer, 3 = bottom",
    )


def fault_zone() -> SyntheticCase:
    height, width = _shape(270, 380)
    polyline = [
        (290.0, 0.0),
        (284.0, 80.0),
        (272.0, 130.0),
        (245.0, 180.0),
        (205.0, 220.0),
        (155.0, 250.0),
        (100.0, 268.0),
    ]
    throw = 20.0

    def interface(x: np.ndarray) -> np.ndarray:
        return 98.0 + 0.08 * (x - 150.0) + 4.0 * np.sin(x / 40.0)

    labels = _offset_layers((height, width), polyline, interface, throw=throw)
    band_radius = 4.5
    fault_line = _raster_fault((height, width), polyline, width=1)
    distance = ndimage.distance_transform_edt(~fault_line)
    band = distance <= band_radius
    labels = labels.copy()
    labels[band] = 3
    return SyntheticCase(
        case_id="syn_fault_zone",
        labels=labels,
        fault_mask=band,
        fault_explicit=True,
        probes=[
            OffsetProbe("zone_throw", (1, 2), 252.0, 290.0, None),
        ],
        fault_polyline=polyline,
        description=(
            "Two layers plus an explicit 9 px fault-zone label (3) following a listric fault "
            "with a kink. The 1-2 interface terminates on both sides of the zone with a throw; "
            "1-3 and 2-3 run along the zone. Junctions at the zone are explicit."
        ),
        layer_note="labels: 1 = upper unit, 2 = lower unit, 3 = fault zone",
    )


def pinchout() -> SyntheticCase:
    height, width = _shape(220, 340)

    def y1(x: np.ndarray) -> np.ndarray:
        return 72.0 + 3.0 * np.sin(x / 40.0)

    def y2(x: np.ndarray) -> np.ndarray:
        thickness = np.where(x <= 292.0, 0.55 * (292.0 - x), 0.0)
        return y1(x) + thickness

    xx = np.arange(width, dtype=float)
    yy = np.arange(height, dtype=float)[:, None]
    top = y1(xx)[None, :]
    bottom = y2(xx)[None, :]
    labels = np.ones((height, width), dtype=np.int32)
    labels[yy >= top] = 2
    labels[yy >= bottom] = 3
    return SyntheticCase(
        case_id="syn_pinchout",
        labels=labels,
        fault_mask=None,
        fault_explicit=False,
        probes=[],
        description=(
            "Three layers where the middle layer pinches out at x~292: its thickness goes to "
            "2.5 px before the top and bottom layers meet. A cusp/triple junction forms at the "
            "pinch-out; this is the topology stress case (no fault)."
        ),
        layer_note="labels: 1 = top, 2 = wedge pinching out, 3 = bottom",
    )


def all_cases() -> list[SyntheticCase]:
    return [normal_fault(), listric_kink(), three_layer_fault(), fault_zone(), pinchout()]


__all__ = ["OffsetProbe", "SyntheticCase", "all_cases"]
