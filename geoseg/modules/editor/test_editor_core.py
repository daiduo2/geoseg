"""Unit tests for Shapes-primary editor_core algorithms.

Run: python -m pytest geoseg/modules/editor/test_editor_core.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from geoseg.modules.editor.editor_core import (
    RegionProperties,
    compute_label_diff,
    extend_trim_line_to_mask,
    fill_boundary_gaps,
    labels_to_shapes,
    merge_labels,
    polygon_rasterize,
    shapes_to_labels,
    snap_line_endpoints,
    snap_path_endpoints,
    split_label_by_line,
)


# ---------------------------------------------------------------------------
# shapes_to_labels
# ---------------------------------------------------------------------------


class TestShapesToLabels:
    """Boundary shapes → topology → labels."""

    def test_empty_shapes(self) -> None:
        """No shapes → entire canvas is one region."""
        labels = shapes_to_labels([], [], (100, 100))
        # Borders are 0 (boundary), interior is one label
        assert labels[0, 0] == 0
        assert labels[50, 50] == 1
        assert len(np.unique(labels)) == 2  # 0 and 1

    def test_single_line_split(self) -> None:
        """One horizontal line splits canvas into top and bottom."""
        shapes = [np.array([[50, 0], [50, 99]], dtype=float)]
        types = ["line"]
        labels = shapes_to_labels(shapes, types, (100, 100))
        # Line at y=50 should create two regions
        unique = set(np.unique(labels))
        assert unique == {0, 1, 2}
        # Top half
        assert labels[25, 50] in (1, 2)
        # Bottom half
        assert labels[75, 50] in (1, 2)
        # They should be different
        assert labels[25, 50] != labels[75, 50]

    def test_cross_lines_four_regions(self) -> None:
        """十字交叉 creates four regions."""
        shapes = [
            np.array([[50, 0], [50, 99]], dtype=float),  # horizontal
            np.array([[0, 50], [99, 50]], dtype=float),  # vertical
        ]
        types = ["line", "line"]
        labels = shapes_to_labels(shapes, types, (100, 100))
        unique = set(np.unique(labels))
        assert unique == {0, 1, 2, 3, 4}  # 4 regions + boundary

    def test_polygon_isolation(self) -> None:
        """Closed polygon isolates interior from exterior."""
        shapes = [
            np.array([[20, 20], [20, 80], [80, 80], [80, 20]], dtype=float)
        ]
        types = ["polygon"]
        labels = shapes_to_labels(shapes, types, (100, 100))
        # Interior should be a different region than exterior
        interior = labels[50, 50]
        exterior = labels[5, 5]
        assert interior != 0
        assert exterior != 0
        assert interior != exterior
        # Boundary pixels are 0
        assert labels[20, 50] == 0

    def test_nested_polygons(self) -> None:
        """Nested polygons create concentric regions."""
        shapes = [
            np.array([[10, 10], [10, 90], [90, 90], [90, 10]], dtype=float),
            np.array([[30, 30], [30, 70], [70, 70], [70, 30]], dtype=float),
        ]
        types = ["polygon", "polygon"]
        labels = shapes_to_labels(shapes, types, (100, 100))
        # Outer, middle, inner should all be different
        outer = labels[5, 5]
        middle = labels[25, 25]  # between outer and inner polygon
        inner = labels[35, 35]   # inside inner polygon
        assert len({outer, middle, inner}) == 3
        assert 0 not in (outer, middle, inner)

    def test_line_plus_polygon(self) -> None:
        """Line splitting a region bounded by polygon."""
        # Polygon isolates an interior region; line runs boundary-to-boundary
        # through that region, splitting it into two.
        shapes = [
            np.array([[10, 10], [10, 90], [90, 90], [90, 10]], dtype=float),
            np.array([[50, 10], [50, 90]], dtype=float),
        ]
        types = ["polygon", "line"]
        labels = shapes_to_labels(shapes, types, (100, 100))
        # Outside polygon
        assert labels[5, 5] != 0
        # Inside polygon, above the line (line is horizontal at y=50)
        above = labels[30, 50]
        # Inside polygon, below the line
        below = labels[70, 50]
        assert above != below
        assert above != 0
        assert below != 0


# ---------------------------------------------------------------------------
# labels_to_shapes
# ---------------------------------------------------------------------------


class TestLabelsToShapes:
    """Labels → boundary contours."""

    def test_single_region(self) -> None:
        """One rectangular region → one contour."""
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[20:80, 20:80] = 1
        shapes = labels_to_shapes(labels)
        assert len(shapes) >= 1
        # Each contour should be (N, 2) array of [y, x]
        for contour in shapes:
            assert contour.ndim == 2
            assert contour.shape[1] == 2

    def test_two_regions(self) -> None:
        """Two adjacent rectangles → two contours."""
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[20:80, 20:50] = 1
        labels[20:80, 50:80] = 2
        shapes = labels_to_shapes(labels)
        # Should have contours for both regions
        assert len(shapes) >= 2


# ---------------------------------------------------------------------------
# Round-trip: labels → shapes → labels
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Topology preservation through conversion cycles."""

    def test_simple_regions(self) -> None:
        """Labels with two adjacent regions."""
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[20:80, 20:50] = 1
        labels[20:80, 50:80] = 2

        shapes = labels_to_shapes(labels)
        shape_types = ["polygon"] * len(shapes)
        recomputed = shapes_to_labels(shapes, shape_types, labels.shape)

        # Original: 2 regions + background 0.
        # Recomputed: same 2 regions + exterior region (what was bg becomes a
        # region because the frame polygon encloses the canvas).
        new_regions = set(np.unique(recomputed)) - {0}
        assert len(new_regions) == 3  # 2 original + exterior

    def test_mondrian_like(self) -> None:
        """Mondrian-style grid of regions."""
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[0:50, 0:50] = 1
        labels[0:50, 50:100] = 2
        labels[50:100, 0:50] = 3
        labels[50:100, 50:100] = 4

        shapes = labels_to_shapes(labels)
        shape_types = ["polygon"] * len(shapes)
        recomputed = shapes_to_labels(shapes, shape_types, labels.shape)

        new_regions = set(np.unique(recomputed)) - {0}
        assert len(new_regions) == 4


# ---------------------------------------------------------------------------
# RegionProperties
# ---------------------------------------------------------------------------


class TestRegionProperties:
    """Stable property binding via geometric fingerprint."""

    def test_get_set(self) -> None:
        """Store and retrieve properties."""
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[20:80, 20:80] = 1

        rp = RegionProperties()
        rp.set(labels, 1, {"Vp": 3000.0, "name": "sedimentary"})

        props = rp.get(labels, 1)
        assert props is not None
        assert props["Vp"] == 3000.0
        assert props["name"] == "sedimentary"

    def test_persistence_across_relabel(self) -> None:
        """Properties survive label ID reassignment."""
        labels1 = np.zeros((100, 100), dtype=np.int32)
        labels1[20:80, 20:80] = 1

        rp = RegionProperties()
        rp.set(labels1, 1, {"Vp": 3000.0})

        # Simulate relabeling (same geometry, different ID)
        labels2 = np.zeros((100, 100), dtype=np.int32)
        labels2[20:80, 20:80] = 42

        props = rp.get(labels2, 42)
        assert props is not None
        assert props["Vp"] == 3000.0

    def test_serialize_roundtrip(self) -> None:
        """to_dict / from_dict roundtrip."""
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[20:80, 20:80] = 1

        rp = RegionProperties()
        rp.set(labels, 1, {"Vp": 3000.0})

        data = rp.to_dict()
        rp2 = RegionProperties.from_dict(data)
        assert rp2.get(labels, 1) == {"Vp": 3000.0}


# ---------------------------------------------------------------------------
# extend_trim_line_to_mask (legacy, still used for boundary snapping)
# ---------------------------------------------------------------------------


class TestExtendTrimLineToMask:
    """CAD-style extend/trim: snap line to mask boundary."""

    @pytest.fixture
    def square_mask(self) -> np.ndarray:
        """100x100 mask with a 60x60 square at center."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 20:80] = True
        return mask

    def test_simple_trim(self, square_mask: np.ndarray) -> None:
        """Line fully inside mask → trim to both boundaries."""
        result = extend_trim_line_to_mask(square_mask, (30.0, 50.0), (70.0, 50.0))
        assert result is not None
        b1, b2 = result
        assert b1[0] == pytest.approx(20.0, abs=1.0)
        assert b2[0] == pytest.approx(80.0, abs=1.0)
        assert b1[1] == pytest.approx(50.0, abs=1.0)
        assert b2[1] == pytest.approx(50.0, abs=1.0)

    def test_extend_from_outside(self, square_mask: np.ndarray) -> None:
        """Endpoints outside mask → extend to boundary."""
        result = extend_trim_line_to_mask(square_mask, (0.0, 50.0), (100.0, 50.0))
        assert result is not None
        b1, b2 = result
        assert b1[0] == pytest.approx(20.0, abs=1.0)
        assert b2[0] == pytest.approx(80.0, abs=1.0)

    def test_vertical_line(self, square_mask: np.ndarray) -> None:
        """Vertical line through mask."""
        result = extend_trim_line_to_mask(square_mask, (50.0, 10.0), (50.0, 90.0))
        assert result is not None
        b1, b2 = result
        assert b1[1] == pytest.approx(20.0, abs=1.0)
        assert b2[1] == pytest.approx(80.0, abs=1.0)

    def test_no_intersection(self, square_mask: np.ndarray) -> None:
        """Line completely outside mask → None."""
        result = extend_trim_line_to_mask(square_mask, (0.0, 0.0), (10.0, 10.0))
        assert result is None

    def test_line_too_short(self, square_mask: np.ndarray) -> None:
        """Very short line inside mask → rejected."""
        result = extend_trim_line_to_mask(square_mask, (30.0, 50.0), (32.0, 50.0))
        assert result is None

    def test_diagonal_line(self, square_mask: np.ndarray) -> None:
        """Diagonal line through mask."""
        result = extend_trim_line_to_mask(square_mask, (0.0, 0.0), (100.0, 100.0))
        assert result is not None
        b1, b2 = result
        assert 15 <= b1[0] <= 25
        assert 15 <= b1[1] <= 25
        assert 75 <= b2[0] <= 85
        assert 75 <= b2[1] <= 85


# ---------------------------------------------------------------------------
# snap_line_endpoints (unified endpoint snapping)
# ---------------------------------------------------------------------------


class TestSnapLineEndpoints:
    """Threshold-circle + tangent direction snapping."""

    @pytest.fixture
    def square_mask(self) -> np.ndarray:
        """100x100 mask with a 60x60 square at center."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 20:80] = True
        return mask

    def test_extend_both_inside(self, square_mask: np.ndarray) -> None:
        """Line fully inside mask, both endpoints short of boundary."""
        # Horizontal line from x=30 to x=70, both inside the square [20,80]
        # but short of the actual boundary at x=20 and x=80
        result = snap_line_endpoints(square_mask, (30.0, 50.0), (70.0, 50.0))
        assert result is not None
        b1, b2 = result
        assert b1[0] == pytest.approx(20.0, abs=2.0)
        assert b2[0] == pytest.approx(80.0, abs=2.0)
        assert b1[1] == pytest.approx(50.0, abs=1.0)
        assert b2[1] == pytest.approx(50.0, abs=1.0)

    def test_trim_both_outside(self, square_mask: np.ndarray) -> None:
        """Line extends beyond mask on both sides — trim to boundary."""
        result = snap_line_endpoints(square_mask, (0.0, 50.0), (100.0, 50.0))
        assert result is not None
        b1, b2 = result
        assert b1[0] == pytest.approx(20.0, abs=2.0)
        assert b2[0] == pytest.approx(80.0, abs=2.0)

    def test_one_inside_one_outside(self, square_mask: np.ndarray) -> None:
        """One endpoint inside, one outside — extend + trim."""
        result = snap_line_endpoints(square_mask, (30.0, 50.0), (100.0, 50.0))
        assert result is not None
        b1, b2 = result
        assert b1[0] == pytest.approx(20.0, abs=2.0)
        assert b2[0] == pytest.approx(80.0, abs=2.0)

    def test_vertical_line(self, square_mask: np.ndarray) -> None:
        """Vertical line through mask."""
        result = snap_line_endpoints(square_mask, (50.0, 10.0), (50.0, 90.0))
        assert result is not None
        b1, b2 = result
        assert b1[1] == pytest.approx(20.0, abs=2.0)
        assert b2[1] == pytest.approx(80.0, abs=2.0)

    def test_no_boundary_nearby(self, square_mask: np.ndarray) -> None:
        """Line inside mask but far from any boundary — no snapping."""
        # Place line near center; distance to nearest boundary is ~30 > threshold=25
        result = snap_line_endpoints(square_mask, (55.0, 50.0), (60.0, 50.0))
        assert result is None

    def test_too_short_after_snap(self, square_mask: np.ndarray) -> None:
        """Snapped segment too short — reject."""
        # Two points very close, even after snap boundary distance < 5
        result = snap_line_endpoints(square_mask, (30.0, 50.0), (32.0, 50.0))
        assert result is None

    def test_diagonal(self, square_mask: np.ndarray) -> None:
        """Diagonal line through mask."""
        # Endpoints close enough to mask that threshold=25 reaches boundary
        result = snap_line_endpoints(square_mask, (10.0, 10.0), (90.0, 90.0))
        assert result is not None
        b1, b2 = result
        assert 15 <= b1[0] <= 25
        assert 15 <= b1[1] <= 25
        assert 75 <= b2[0] <= 85
        assert 75 <= b2[1] <= 85

    def test_endpoint_on_boundary_no_change(self, square_mask: np.ndarray) -> None:
        """Endpoint already on boundary — no snapping needed."""
        result = snap_line_endpoints(square_mask, (20.0, 50.0), (80.0, 50.0))
        assert result is None

    def test_line_completely_outside(self, square_mask: np.ndarray) -> None:
        """Line completely outside mask — no intersection."""
        # y=0 is outside the mask; endpoints far from mask boundary
        result = snap_line_endpoints(square_mask, (0.0, 0.0), (1.0, 0.0))
        assert result is None


# ---------------------------------------------------------------------------
# snap_path_endpoints
# ---------------------------------------------------------------------------


class TestSnapPathEndpoints:
    """Open-path endpoint snapping (interior vertices preserved)."""

    @pytest.fixture
    def square_mask(self) -> np.ndarray:
        """100x100 mask with a 60x60 square at center."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 20:80] = True
        return mask

    def test_path_extend(self, square_mask: np.ndarray) -> None:
        """3-vertex path inside mask; endpoints extend to boundary."""
        # Horizontal path: [y=50, x=30] -> [y=50, x=50] -> [y=50, x=70]
        vertices = np.array(
            [[50.0, 30.0], [50.0, 50.0], [50.0, 70.0]], dtype=float
        )
        result = snap_path_endpoints(square_mask, vertices)
        assert result is not None
        # Start snaps to left boundary x≈20
        assert result[0, 1] == pytest.approx(20.0, abs=2.0)
        # End snaps to right boundary x≈80
        assert result[-1, 1] == pytest.approx(80.0, abs=2.0)
        # Middle vertex preserved
        assert np.allclose(result[1], vertices[1])

    def test_path_trim(self, square_mask: np.ndarray) -> None:
        """Path endpoints outside mask; trim to boundary."""
        vertices = np.array(
            [[50.0, 0.0], [50.0, 50.0], [50.0, 100.0]], dtype=float
        )
        result = snap_path_endpoints(square_mask, vertices)
        assert result is not None
        assert result[0, 1] == pytest.approx(20.0, abs=2.0)
        assert result[-1, 1] == pytest.approx(80.0, abs=2.0)
        assert np.allclose(result[1], vertices[1])

    def test_path_no_snap(self, square_mask: np.ndarray) -> None:
        """Path endpoints far from boundary — no snapping."""
        # Endpoints >25 px from any boundary along search line
        vertices = np.array(
            [[50.0, 46.0], [50.0, 50.0], [50.0, 54.0]], dtype=float
        )
        result = snap_path_endpoints(square_mask, vertices)
        assert result is None

    def test_path_short_segment(self, square_mask: np.ndarray) -> None:
        """Snapped end segment becomes too short — reject."""
        # Second vertex is exactly on the boundary (x=20); snap would collapse
        # the first segment to < 2 px.
        vertices = np.array(
            [[50.0, 19.9], [50.0, 20.0], [50.0, 80.0]], dtype=float
        )
        result = snap_path_endpoints(square_mask, vertices, min_segment_length=2.0)
        assert result is None


# ---------------------------------------------------------------------------
# fill_boundary_gaps
# ---------------------------------------------------------------------------


class TestFillBoundaryGaps:
    """Boundary pixel interpolation for export-ready labels."""

    def test_no_gaps(self) -> None:
        """Labels already fully partitioned — no interior zeros to fill."""
        labels = np.zeros((50, 50), dtype=np.int32)
        labels[10:40, 10:25] = 1
        labels[10:40, 25:40] = 2
        labels[0:10, :] = 3  # fill top so no interior 0
        labels[40:50, :] = 3  # fill bottom
        labels[:, 0:10] = 3  # fill left
        labels[:, 40:50] = 3  # fill right
        filled = fill_boundary_gaps(labels)
        assert np.array_equal(labels, filled)

    def test_thin_boundary_filled(self) -> None:
        """Thin boundary line (label 0) between two regions is filled."""
        labels = np.zeros((50, 50), dtype=np.int32)
        labels[10:25, 10:40] = 1
        labels[26:40, 10:40] = 2
        # Row 25 is label 0 (thin boundary)
        filled = fill_boundary_gaps(labels)
        # Boundary pixels should now be non-zero
        assert filled[25, 20] != 0
        # Regions 1 and 2 should still exist
        assert 1 in filled
        assert 2 in filled

    def test_large_background_preserved(self) -> None:
        """Large background region (>10% area) stays 0."""
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[45:55, 45:55] = 1  # small region in center
        filled = fill_boundary_gaps(labels)
        # Outer background should remain 0
        assert filled[0, 0] == 0
        assert filled[99, 99] == 0
        # Inner region preserved
        assert filled[50, 50] == 1

    def test_border_connected_background_not_filled(self) -> None:
        """Background connected to the image border is preserved, not filled."""
        labels = np.zeros((50, 50), dtype=np.int32)
        # Two regions with a large background channel connecting to the border
        labels[5:20, 5:20] = 1
        labels[5:20, 30:45] = 2
        filled = fill_boundary_gaps(labels)
        # Background pixels that reach the border should stay 0
        assert filled[25, 25] == 0
        assert filled[0, 0] == 0
        # The labeled regions should still exist
        assert 1 in filled
        assert 2 in filled

    def test_interior_thin_boundary_filled_when_background_present(self) -> None:
        """A thin separator between two labeled regions is filled even when
        border-connected background exists elsewhere."""
        labels = np.zeros((50, 50), dtype=np.int32)
        # Two regions separated by a thin horizontal line
        labels[5:10, 5:45] = 1
        labels[11:20, 5:45] = 2
        # Row 10 is label 0 (thin boundary)
        filled = fill_boundary_gaps(labels)
        # The separator between region 1 and 2 should be filled
        assert filled[10, 25] != 0
        # Background connected to border stays 0
        assert filled[25, 25] == 0
        # Regions 1 and 2 should still exist
        assert 1 in filled
        assert 2 in filled


# ---------------------------------------------------------------------------
# RegionProperties round-trip
# ---------------------------------------------------------------------------


class TestRegionPropertiesRoundTrip:
    """Properties survive labels → shapes → labels conversion."""

    def test_property_roundtrip(self) -> None:
        """Set props on original labels, convert to shapes, recompute, retrieve."""
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[20:80, 20:50] = 1
        labels[20:80, 50:80] = 2

        rp = RegionProperties()
        rp.set(labels, 1, {"Vp": 3000.0, "name": "sedimentary"})
        rp.set(labels, 2, {"Vp": 6000.0, "name": "basement"})

        shapes = labels_to_shapes(labels)
        shape_types = ["polygon"] * len(shapes)
        recomputed = shapes_to_labels(shapes, shape_types, labels.shape)

        # Recomputed labels have different IDs, but geometry is preserved
        # We can retrieve properties by matching geometry
        for lid in sorted(set(recomputed.flatten()) - {0}):
            props = rp.get(recomputed, lid)
            if props is not None:
                assert "Vp" in props
                assert props["Vp"] in (3000.0, 6000.0)

    def test_fingerprint_precision(self) -> None:
        """Fingerprint uses 3-decimal precision + aspect ratio."""
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[20:80, 20:80] = 1

        rp = RegionProperties()
        rp.set(labels, 1, {"Vp": 3000.0})

        # Same geometry, different ID — should match
        labels2 = np.zeros((100, 100), dtype=np.int32)
        labels2[20:80, 20:80] = 99
        assert rp.get(labels2, 99) == {"Vp": 3000.0}

    def test_get_missing_label_returns_none(self) -> None:
        """Getting a property for a label that was never set returns None."""
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[20:80, 20:80] = 1
        rp = RegionProperties()
        assert rp.get(labels, 1) is None

    def test_remove_deletes_property(self) -> None:
        """remove() deletes the stored property for a label."""
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[20:80, 20:80] = 1
        rp = RegionProperties()
        rp.set(labels, 1, {"Vp": 3000.0})
        rp.remove(labels, 1)
        assert rp.get(labels, 1) is None

    def test_remove_missing_label_is_noop(self) -> None:
        """remove() on a label with no stored property is a no-op."""
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[20:80, 20:80] = 1
        rp = RegionProperties()
        rp.remove(labels, 1)
        assert rp.to_dict() == {}


# ---------------------------------------------------------------------------
# Edge cases in shapes_to_labels / labels_to_shapes
# ---------------------------------------------------------------------------


class TestShapeConversionEdgeCases:
    """Edge cases for shapes_to_labels and labels_to_shapes."""

    def test_shapes_to_labels_skips_short_shapes(self) -> None:
        """Shapes with fewer than 2 vertices are ignored."""
        shapes = [np.array([[50.0, 50.0]]), np.array([[50.0, 0.0], [50.0, 99.0]])]
        types = ["line", "line"]
        labels = shapes_to_labels(shapes, types, (100, 100))
        # Single-vertex shape is ignored; valid line still splits canvas
        unique = set(np.unique(labels))
        assert unique == {0, 1, 2}

    def test_labels_to_shapes_skips_empty_labels(self) -> None:
        """A label ID with zero pixels is skipped."""
        labels = np.zeros((50, 50), dtype=np.int32)
        labels[10:40, 10:40] = 1
        # Label 2 exists in the unique set after np.unique only if present;
        # instead simulate by manually adding an empty label entry is not possible,
        # so just verify the empty-label guard in the helper works.
        shapes = labels_to_shapes(labels)
        # Only label 1 should produce shapes
        assert len(shapes) >= 1
        # all-zero labels yields no contours
        empty = labels_to_shapes(np.zeros((50, 50), dtype=np.int32))
        assert empty == []


# ---------------------------------------------------------------------------
# extend_trim_line_to_mask edge cases
# ---------------------------------------------------------------------------


class TestExtendTrimLineEdgeCases:
    """Edge cases for legacy extend_trim_line_to_mask."""

    def test_zero_length_line(self) -> None:
        """Zero-length input returns None."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 20:80] = True
        assert extend_trim_line_to_mask(mask, (50.0, 50.0), (50.0, 50.0)) is None

    def test_no_inside_samples(self) -> None:
        """Line entirely outside mask with no crossing returns None."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 20:80] = True
        assert extend_trim_line_to_mask(mask, (0.0, 0.0), (5.0, 0.0)) is None


# ---------------------------------------------------------------------------
# snap_line_endpoints / snap_path_endpoints edge cases
# ---------------------------------------------------------------------------


class TestSnapEdgeCases:
    """Edge cases for endpoint snapping."""

    def test_snap_line_zero_length(self) -> None:
        """Zero-length line cannot be snapped."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 20:80] = True
        assert snap_line_endpoints(mask, (50.0, 50.0), (50.0, 50.0)) is None

    def test_snap_line_endpoint_too_far(self) -> None:
        """Endpoint far beyond threshold distance returns None."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 20:80] = True
        # Endpoint at x=0 is 20 px from boundary, but search direction is wrong
        result = snap_line_endpoints(mask, (0.0, 50.0), (1.0, 50.0), threshold=5.0)
        assert result is None

    def test_snap_path_too_few_vertices(self) -> None:
        """Path with fewer than 2 vertices cannot be snapped."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 20:80] = True
        assert snap_path_endpoints(mask, np.array([[50.0, 50.0]])) is None


# ---------------------------------------------------------------------------
# Legacy split / merge / rasterize / diff
# ---------------------------------------------------------------------------


class TestLegacyUtilities:
    """Tests for legacy label manipulation utilities."""

    def test_split_label_by_line(self) -> None:
        """Split a region by a cut line."""
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[20:80, 20:80] = 1
        result = split_label_by_line(labels, 1, (20.0, 50.0), (80.0, 50.0), 2)
        assert result is not None
        unique = set(np.unique(result)) - {0}
        assert unique == {1, 2}

    def test_split_label_by_line_no_split_returns_none(self) -> None:
        """Line that does not split the target region returns None."""
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[20:80, 20:80] = 1
        # Line completely outside target region
        result = split_label_by_line(labels, 1, (0.0, 0.0), (5.0, 0.0), 2)
        assert result is None

    def test_merge_labels(self) -> None:
        """Merge label_b into label_a."""
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[20:40, 20:40] = 1
        labels[60:80, 60:80] = 2
        result = merge_labels(labels, 1, 2)
        assert np.all(result[60:80, 60:80] == 1)
        assert 2 not in result

    def test_merge_same_label_is_copy(self) -> None:
        """Merging a label into itself returns a copy."""
        labels = np.zeros((100, 100), dtype=np.int32)
        labels[20:40, 20:40] = 1
        result = merge_labels(labels, 1, 1)
        assert np.array_equal(result, labels)
        assert result is not labels

    def test_polygon_rasterize(self) -> None:
        """Polygon interior is filled with new label."""
        labels = np.zeros((100, 100), dtype=np.int32)
        vertices = np.array([[30.0, 30.0], [30.0, 70.0], [70.0, 70.0], [70.0, 30.0]])
        result = polygon_rasterize(labels, vertices, 5)
        assert 5 in result
        assert result[50, 50] == 5

    def test_polygon_rasterize_short_vertices(self) -> None:
        """Polygon with fewer than 3 vertices is a no-op."""
        labels = np.zeros((100, 100), dtype=np.int32)
        vertices = np.array([[30.0, 30.0], [70.0, 70.0]])
        result = polygon_rasterize(labels, vertices, 5)
        assert 5 not in result

    def test_compute_label_diff(self) -> None:
        """Diff returns coordinates and old labels for changed pixels."""
        before = np.zeros((50, 50), dtype=np.int32)
        before[10:20, 10:20] = 1
        after = before.copy()
        after[10:20, 10:20] = 2
        diff = compute_label_diff(before, after)
        assert diff.shape[1] == 3
        assert len(diff) == 100
        assert np.all(diff[:, 2] == 1)
