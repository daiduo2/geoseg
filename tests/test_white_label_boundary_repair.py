import importlib.util
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/process_j_tecto_2019_06_024_direct.py"
)
SPEC = importlib.util.spec_from_file_location("j_tecto_direct", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
_bridge_boundary_with_endpoint_tangents = (
    MODULE._bridge_boundary_with_endpoint_tangents
)
_repair_white_label_occluded_boundaries = (
    MODULE._repair_white_label_occluded_boundaries
)


def test_endpoint_tangent_bridge_preserves_linear_boundary() -> None:
    x = np.arange(100, dtype=np.float64)
    trace = 20.0 + 0.2 * x
    trace[35:65] = np.nan

    bridged = _bridge_boundary_with_endpoint_tangents(
        trace, start=35, stop=65
    )

    assert bridged is not None
    segment, metrics = bridged
    np.testing.assert_allclose(segment, 20.0 + 0.2 * x[35:65], atol=0.25)
    assert abs(float(metrics["left_endpoint_slope"]) - 0.2) < 0.02
    assert abs(float(metrics["right_endpoint_slope"]) - 0.2) < 0.02
    assert float(metrics["max_curvature_change"]) < 0.01


def test_endpoint_tangent_bridge_requires_support_on_both_sides() -> None:
    trace = np.full(80, np.nan, dtype=np.float64)
    trace[10:35] = 25.0
    trace[60:64] = 28.0

    assert (
        _bridge_boundary_with_endpoint_tangents(trace, start=35, stop=60)
        is None
    )


def test_white_label_repair_replaces_block_with_smooth_boundary() -> None:
    height, width = 100, 150
    panel = np.full((height, width, 3), (80, 120, 160), dtype=np.uint8)
    panel[38:67, 55:95] = 255
    body = np.ones((height, width), dtype=bool)
    x = np.arange(width)
    boundary = np.rint(45.0 + 0.1 * x).astype(int)
    labels = np.ones((height, width), dtype=np.int16)
    for column, boundary_y in enumerate(boundary):
        labels[: boundary_y + 1, column] = 0

    # Simulate the old nearest-region fill leaving a rectangular step.
    labels[38:67, 55:95] = 0
    repaired, _, changed, curve, records = (
        _repair_white_label_occluded_boundaries(labels, panel, body)
    )

    successful = [record for record in records if record["status"] == "repaired"]
    assert len(successful) == 1
    assert changed[38:67, 55:95].any()
    assert curve[38:67, 55:95].sum() == 40
    repaired_boundary = np.array(
        [np.max(np.flatnonzero(repaired[:, column] == 0)) for column in range(55, 95)]
    )
    assert np.max(np.abs(np.diff(repaired_boundary, n=2))) <= 1
    np.testing.assert_allclose(repaired_boundary, boundary[55:95], atol=1)
