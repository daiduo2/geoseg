"""Tests for panel preflight, audits, and the agent-driven sandbox session."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from geoseg.modules.iteration import (
    AuditConfig,
    IterationSession,
    audit_labels,
    compact_labels,
    crop_panel,
    detect_panel_bbox,
)


def _panel_with_axes() -> np.ndarray:
    h, w = 120, 160
    image = np.full((h, w, 3), 255, dtype=np.uint8)
    image[20:100, 30:150] = [210, 40, 40]
    image[20:100, 90:150] = [40, 170, 70]
    image[20:100, 120:150] = [40, 60, 210]
    # Black tick labels / axis outside the coloured panel.
    image[10:110, 0:25] = 0
    image[0:15, 40:150] = 0
    return image


def test_detect_panel_bbox_excludes_axes_and_margins():
    image = _panel_with_axes()
    bbox = detect_panel_bbox(image)

    assert bbox == (30, 20, 150, 100)


def test_crop_panel_returns_crop_and_offset():
    image = _panel_with_axes()
    crop = crop_panel(image)

    assert crop.bbox == (30, 20, 150, 100)
    assert crop.image.shape[:2] == (80, 120)
    assert crop.offset == (30, 20)
    assert crop.method == "saturation_bbox"


def test_manual_crop_validation():
    image = _panel_with_axes()
    crop = crop_panel(image, bbox=(10, 5, 100, 90))
    assert crop.bbox == (10, 5, 100, 90)
    assert crop.method == "manual"
    with pytest.raises(ValueError):
        crop_panel(image, bbox=(0, 0, 999, 10))


def test_audit_detects_under_seg_and_text():
    h, w = 80, 120
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[:30] = [220, 40, 40]
    image[30:55] = [40, 180, 60]
    image[55:] = [40, 60, 220]
    image[35:45, 50:70] = 255
    labels = np.zeros((h, w), dtype=np.int32)
    labels[:30] = 1
    labels[30:55] = 1  # merged red/green
    labels[55:] = 2
    labels[35:45, 50:70] = 3  # white text label

    report = audit_labels(
        image, labels, config=AuditConfig(expected_min_layers=3)
    )

    assert 1 in report.under_segmented_labels
    assert 3 in report.text_labels
    assert not report.accepted


def test_ph01_curated_labels_are_under_segmented():
    root = Path(__file__).resolve().parents[3] / "validation" / "curated"
    record = next(
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if "ph01_page8_panel3" in line
    )
    image = np.asarray(Image.open(root / record["files"]["image"]).convert("RGB"))
    labels = np.load(root / record["files"]["labels"])["labels"]

    report = audit_labels(image, labels, config=AuditConfig(expected_min_layers=8))

    assert {2, 4}.issubset(set(report.under_segmented_labels))
    assert 3 in report.text_labels
    assert report.layer_count_issue == "too_few_layers"


def test_ph01_auto_crop_removes_axis(tmp_path):
    root = Path(__file__).resolve().parents[3] / "validation" / "curated"
    record = next(
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if "ph01_page8_panel3" in line
    )
    image = np.asarray(Image.open(root / record["files"]["image"]).convert("RGB"))
    session = IterationSession(image, output_dir=tmp_path)

    assert session.crop.bbox[0] >= 25
    assert session.crop.image.shape[1] < image.shape[1]
    assert session.crop.image.shape[0] == image.shape[0]
    assert (tmp_path / "crop.json").exists()


def test_session_records_attempts_without_choosing_actions(tmp_path):
    image = _panel_with_axes()
    # Fake runner: deliberately returns two very different attempts.
    def runner(cropped, engine, n_layers, params):
        h, w = cropped.shape[:2]
        labels = np.zeros((h, w), dtype=np.int32)
        labels[:, : w // 2] = 1
        labels[:, w // 2 :] = 2 if n_layers > 2 else 1
        return labels

    session = IterationSession(
        image,
        output_dir=tmp_path,
        audit_config=AuditConfig(expected_min_layers=2),
        runner=runner,
    )
    first = session.run_attempt("v4_kmeans", 1, note="initial")
    second = session.run_attempt("v4_kmeans", 3, note="more layers")

    assert first.attempt == 1
    assert second.attempt == 2
    assert len(session.attempts) == 2
    assert session.accepted_index is None
    session.accept(2)
    session.save_best()

    assert session.accepted_index == 1
    assert (tmp_path / "labels.npz").exists()
    assert (tmp_path / "overlay.jpg").exists()
    summary = json.loads((tmp_path / "session.json").read_text())
    assert summary["accepted_attempt"] == 2
    assert len(summary["attempts"]) == 2
    assert summary["attempts"][1]["n_layers"] == 3
    assert summary["crop"]["bbox"] == [30, 20, 150, 100]


def test_compact_labels_remaps_foreground_only():
    labels = np.array([[0, 3, 3], [7, 7, 9]], dtype=np.int32)
    compacted, mapping = compact_labels(labels)

    assert mapping == {3: 1, 7: 2, 9: 3}
    assert compacted.tolist() == [[0, 1, 1], [2, 2, 3]]
