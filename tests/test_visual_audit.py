"""Tests for geoseg.modules.visual_audit."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from geoseg.modules.visual_audit import (
    create_audit_report,
    create_audit_views,
)


def _make_image(shape: tuple[int, int, int], color: tuple[int, int, int]) -> np.ndarray:
    img = np.zeros(shape, dtype=np.uint8)
    img[:] = color
    return img


def test_create_audit_views_returns_expected_keys(tmp_path: Path):
    h, w = 200, 200
    panel = _make_image((h, w, 3), (180, 160, 140))
    labels = np.zeros((h, w), dtype=np.int32)
    labels[: h // 2] = 1
    labels[h // 2 :] = 2

    views = create_audit_views(labels, panel)
    expected = {
        "boundary_on_original",
        "pure_mask",
        "fragment_highlight",
        "text_residual_map",
        "topology_map",
        "difference_heatmap",
        "color_residual",
        "side_by_side",
        "plume_comparison",
    }
    assert expected.issubset(views.keys())
    for name, arr in views.items():
        if name in {"side_by_side", "plume_comparison"}:
            assert arr.shape[0] == h
            assert arr.shape[1] >= w
        else:
            assert arr.shape[:2] == (h, w)


def test_create_audit_report_generates_files(tmp_path: Path):
    h, w = 200, 200
    panel = _make_image((h, w, 3), (180, 160, 140))
    labels = np.zeros((h, w), dtype=np.int32)
    labels[: h // 2] = 1
    labels[h // 2 :] = 2

    output_dir = tmp_path / "audit"
    report = create_audit_report(labels, panel, str(output_dir))

    assert Path(report["summary_image_path"]).exists()
    assert Path(report["report_path"]).exists()
    assert (output_dir / "views").exists()
    assert (output_dir / "crops").exists()
    assert "diagnostic_signals" in report
    assert "label_color_map" in report
    assert "boundary_alignment" in report["diagnostic_signals"]
    assert "color_residual" in report["diagnostic_signals"]
    assert "high_deviation_regions" in report["diagnostic_signals"]["color_residual"]
    assert "color_residual" in report["view_paths"]
    assert Path(report["view_paths"]["color_residual"]).exists()
    assert "plume_fidelity" in report["diagnostic_signals"] or "iou" in report["diagnostic_signals"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
