"""Tests for regional fusion: label-level freeze + multi-engine segmentation.

Experimental feature coverage:
- fuse_with_freeze basic behavior and seam smoothing
- generate_overlay_with_legend creates legend in bottom-right corner
- regional_segment fallback when no audit provided
- per_label_metrics in metrics.py
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from geoseg.modules.segment_engines.regional_fusion import (
    FusionConfig,
    RegionalAudit,
    fuse_with_freeze,
    generate_overlay_with_legend,
    regional_segment,
)
from geoseg.modules.segment_engines.metrics import per_label_metrics


# ---------------------------------------------------------------------------
# fuse_with_freeze
# ---------------------------------------------------------------------------


class TestFuseWithFreeze:
    def test_basic_freeze(self) -> None:
        base = np.array([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4],
        ], dtype=np.int32)
        patch = np.array([
            [10, 10, 20, 20],
            [10, 10, 20, 20],
            [30, 30, 40, 40],
            [30, 30, 40, 40],
        ], dtype=np.int32)
        freeze = np.zeros((4, 4), dtype=bool)
        freeze[:2, :2] = True  # freeze top-left

        result = fuse_with_freeze(base, patch, freeze, seam_width=0)

        assert result[0, 0] == 1
        assert result[1, 1] == 1
        assert result[0, 2] == 20
        assert result[2, 0] == 30

    def test_seam_fills_zero_gaps(self) -> None:
        base = np.array([
            [1, 1, 0, 2],
            [1, 1, 0, 2],
            [1, 1, 3, 3],
            [1, 1, 3, 3],
        ], dtype=np.int32)
        patch = np.array([
            [1, 1, 0, 2],
            [1, 1, 0, 2],
            [4, 4, 3, 3],
            [4, 4, 3, 3],
        ], dtype=np.int32)
        freeze = np.zeros((4, 4), dtype=bool)
        freeze[:, :2] = True  # freeze left half

        result = fuse_with_freeze(base, patch, freeze, seam_width=2)

        # Left half should be base
        assert np.array_equal(result[:, :2], base[:, :2])
        # Right half should be patch (or seam-filled, but not 0)
        assert result[0, 3] == 2
        assert result[2, 2] == 3
        # Gap pixels (value 0) at boundary should be filled by seam smoothing
        assert result[0, 2] != 0
        assert result[1, 2] != 0

    def test_no_boundary_no_transition(self) -> None:
        base = np.ones((4, 4), dtype=np.int32)
        patch = np.full((4, 4), 2, dtype=np.int32)
        freeze = np.ones((4, 4), dtype=bool)  # freeze all

        result = fuse_with_freeze(base, patch, freeze, seam_width=3)
        assert np.array_equal(result, base)

    def test_shape_mismatch_raises(self) -> None:
        base = np.ones((4, 4), dtype=np.int32)
        patch = np.ones((3, 4), dtype=np.int32)
        freeze = np.ones((4, 4), dtype=bool)

        with pytest.raises(ValueError, match="Shape mismatch"):
            fuse_with_freeze(base, patch, freeze)

    def test_seam_width_zero_skips_smoothing(self) -> None:
        base = np.array([[1, 0], [0, 2]], dtype=np.int32)
        patch = np.array([[10, 0], [0, 20]], dtype=np.int32)
        freeze = np.array([[True, False], [False, False]], dtype=bool)

        result = fuse_with_freeze(base, patch, freeze, seam_width=0)
        assert result[0, 0] == 1
        assert result[1, 1] == 20


# ---------------------------------------------------------------------------
# generate_overlay_with_legend
# ---------------------------------------------------------------------------


class TestGenerateOverlayWithLegend:
    def test_output_shape_matches_input(self) -> None:
        panel = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        labels = np.zeros((100, 80), dtype=np.int32)
        labels[:50, :40] = 1
        labels[:50, 40:] = 2
        labels[50:, :40] = 3
        labels[50:, 40:] = 0

        overlay = generate_overlay_with_legend(panel, labels)

        assert overlay.shape == panel.shape
        assert overlay.dtype == np.uint8

    def test_legend_region_is_darker(self) -> None:
        panel = np.full((60, 60, 3), 200, dtype=np.uint8)
        labels = np.zeros((60, 60), dtype=np.int32)
        labels[10:50, 10:50] = 1

        overlay = generate_overlay_with_legend(panel, labels)

        # Bottom-right corner should have the semi-transparent black legend bg
        br = overlay[-20:, -20:]
        # Legend background is darker than original white panel
        assert br.mean() < 180

    def test_empty_labels_returns_panel_like(self) -> None:
        panel = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        labels = np.zeros((50, 50), dtype=np.int32)

        overlay = generate_overlay_with_legend(panel, labels)

        assert overlay.shape == panel.shape


# ---------------------------------------------------------------------------
# regional_segment fallback
# ---------------------------------------------------------------------------


class TestRegionalSegmentFallback:
    def test_no_audit_runs_primary_engine(self) -> None:
        panel = np.random.randint(0, 255, (40, 40, 3), dtype=np.uint8)

        result = regional_segment(panel, n_layers=3)

        assert "labels" in result
        assert result["labels"].shape == (40, 40)
        assert result["meta"]["fusion_applied"] is False
        assert result["meta"]["path"] == "primary_only"

    def test_empty_retry_labels_runs_primary(self) -> None:
        panel = np.random.randint(0, 255, (40, 40, 3), dtype=np.uint8)
        audit = RegionalAudit(frozen_labels=[1], retry_labels=[], notes="all good")
        labels_a = np.ones((40, 40), dtype=np.int32)

        result = regional_segment(
            panel,
            n_layers=3,
            primary_result={"labels": labels_a},
            audit=audit,
        )

        assert result["meta"]["fusion_applied"] is False


# ---------------------------------------------------------------------------
# regional_segment local fixes
# ---------------------------------------------------------------------------


class TestRegionalSegmentLocalFixes:
    def test_split_label_by_color_components(self) -> None:
        """A split local fix should divide an over-merged label before fusion."""
        panel = np.zeros((20, 40, 3), dtype=np.uint8)
        # Top half is one label (red); bottom half is a single label that
        # actually contains two colours (blue left, green right).
        panel[:10, :] = [200, 0, 0]
        panel[10:, :20] = [0, 0, 200]
        panel[10:, 20:] = [0, 200, 0]

        labels_a = np.zeros((20, 40), dtype=np.int32)
        labels_a[:10, :] = 5
        labels_a[10:, :] = 6

        # _reorder_labels_by_median_y maps the bottom label to 1.  Freeze the
        # two components that the split will produce so they survive fusion.
        audit = RegionalAudit(
            frozen_labels=[1, 2],
            retry_labels=[3],
            secondary_engine="grayscale",
            local_fixes=[
                {
                    "action": "split_label_by_color_components",
                    "label_id": 1,
                    "color_space": "LAB",
                    "k": 2,
                    "min_component_area": 1,
                }
            ],
        )

        result = regional_segment(
            panel,
            n_layers=2,
            primary_result={"labels": labels_a},
            audit=audit,
        )

        out_labels = result["labels"]
        unique = set(np.unique(out_labels))
        assert 2 in unique, "split should create a second component label"
        assert result["meta"]["fusion_applied"] is True


# ---------------------------------------------------------------------------
# per_label_metrics
# ---------------------------------------------------------------------------


class TestPerLabelMetrics:
    def test_per_label_boundary_alignment(self) -> None:
        # Create synthetic image with horizontal stripes
        img = np.zeros((40, 40, 3), dtype=np.uint8)
        img[:10, :] = [255, 0, 0]   # red stripe
        img[10:20, :] = [0, 255, 0]  # green stripe
        img[20:30, :] = [0, 0, 255]  # blue stripe
        img[30:, :] = [255, 255, 0]  # yellow stripe

        labels = np.zeros((40, 40), dtype=np.int32)
        labels[:10, :] = 1
        labels[10:20, :] = 2
        labels[20:30, :] = 3
        labels[30:, :] = 4

        result = per_label_metrics(labels, img)

        assert len(result) == 4
        for lbl, metrics in result.items():
            assert "boundary_alignment" in metrics
            assert "area_fraction" in metrics
            assert "has_tiny_fragments" in metrics
            assert 0.0 <= metrics["boundary_alignment"] <= 1.0

    def test_missing_label_returns_zeros(self) -> None:
        img = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
        labels = np.ones((20, 20), dtype=np.int32)

        result = per_label_metrics(labels, img, label_ids=[1, 99])

        assert result[1]["area_fraction"] > 0
        assert result[99]["boundary_alignment"] == 0.0
        assert result[99]["area_fraction"] == 0.0


# ---------------------------------------------------------------------------
# RegionalAudit / FusionConfig dataclasses
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_regional_audit_defaults(self) -> None:
        audit = RegionalAudit()
        assert audit.frozen_labels == []
        assert audit.retry_labels == []
        assert audit.notes == ""
        assert audit.iteration == 0

    def test_fusion_config_defaults(self) -> None:
        cfg = FusionConfig()
        assert cfg.primary_engine == "v4_kmeans"
        assert cfg.secondary_engines == ["edge_guided", "kmeans_full"]
        assert cfg.max_iterations == 3
        assert cfg.seam_smooth_width == 3
        assert cfg.enable_legend is True
