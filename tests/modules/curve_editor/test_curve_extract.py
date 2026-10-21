"""Extraction, persistence, and curated no-edit round-trip tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from geoseg.modules.curve_editor import (
    curves_from_labels,
    load_session,
    model_from_labels,
    save_session,
)

CURATED = Path(__file__).resolve().parents[3] / "validation" / "curated"


def _layered_labels() -> np.ndarray:
    labels = np.zeros((120, 120), dtype=np.int32)
    labels[20:50] = 1
    labels[50:80] = 2
    labels[80:120] = 3
    return labels


def test_extract_layered_horizons_and_rerasterize():
    labels = _layered_labels()
    curves = curves_from_labels(labels)
    assert len(curves) >= 2
    assert any(not curve.closed for curve in curves)

    model = model_from_labels(labels)
    assert np.array_equal(model.to_labels(), labels)
    rasterized = model.rasterize()

    assert (rasterized == labels).mean() > 0.93


def test_extract_closed_lens_and_rerasterize():
    labels = np.zeros((100, 100), dtype=np.int32)
    labels[20:80, 20:80] = 1
    labels[40:60, 40:60] = 2

    curves = curves_from_labels(labels)
    assert any(curve.closed for curve in curves)

    model = model_from_labels(labels)
    rasterized = model.rasterize()
    assert (rasterized == labels).mean() > 0.90
    assert set(np.unique(rasterized)) >= {0, 1, 2}


def test_session_round_trip_preserves_curves_and_labels(tmp_path):
    model = model_from_labels(_layered_labels())
    interface = next(iter(model.interfaces.values()))
    model.move_point(interface.id, interface.points[1].id, 0, 3)
    expected_labels = model.to_labels()

    path = save_session(model, tmp_path / "session.npz")
    loaded = load_session(path)

    assert loaded.to_dict()["interfaces"] == model.to_dict()["interfaces"]
    assert np.array_equal(loaded.to_labels(), expected_labels)


def test_untouched_session_round_trip_is_exact(tmp_path):
    labels = _layered_labels()
    model = model_from_labels(labels)
    path = save_session(model, tmp_path / "session.npz")

    loaded = load_session(path)

    assert np.array_equal(loaded.to_labels(), labels)
    assert loaded.base_labels is not None


def _curated_records():
    if not (CURATED / "manifest.jsonl").exists():
        return []
    records = []
    for line in (CURATED / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        records.append(record)
    return records


@pytest.mark.parametrize("record", _curated_records(), ids=lambda r: r["panel_id"])
def test_curated_panel_no_edit_round_trip(record):
    labels = np.load(CURATED / record["files"]["labels"])["labels"]
    model = model_from_labels(labels)

    assert np.array_equal(model.to_labels(), labels)
    assert len(model.interfaces) > 0


@pytest.mark.parametrize("record", _curated_records(), ids=lambda r: r["panel_id"])
def test_curated_panel_rasterization_keeps_region_ids(record):
    labels = np.load(CURATED / record["files"]["labels"])["labels"]
    model = model_from_labels(labels)
    rasterized = model.rasterize()

    base_ids = set(np.unique(labels).tolist()) - {0}
    out_ids = set(np.unique(rasterized).tolist()) - {0}
    assert out_ids <= base_ids
    assert (rasterized == labels).mean() > 0.65
