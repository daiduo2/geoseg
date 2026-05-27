"""Integration test for the full geoseg backend pipeline.

Uses ph01 fixture (observational_data — should be skipped) and synthetic
figures to cover multiple paths: conceptual, grayscale, multi-panel, batch.

Run:
    pytest tests/test_integration_ph01.py -v
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from geoseg.batch_processor import process_directory
from geoseg.controller import run_pipeline


def test_ph01_observational_data_is_skipped() -> None:
    """ph01 fixture is a velocity cross-section map → should be skipped."""
    fixture_path = Path(__file__).parent / "fixtures" / "ph01" / "ph01_page8_300dpi.png"
    if not fixture_path.exists():
        pytest.skip(f"Fixture not found: {fixture_path}")

    img = np.array(Image.open(fixture_path).convert("RGB"))
    result = run_pipeline(
        img,
        n_layers=4,
        quality_preference="fast",
        skip_non_velocity_model=True,
        use_vlm=False,
    )
    assert result["status"] == "skipped"
    assert "figure_type=" in result.get("reason", "")


def test_synthetic_conceptual_runs_end_to_end() -> None:
    """Synthetic conceptual figure should process through all stages."""
    rng = np.random.default_rng(42)
    img = np.ones((320, 560, 3), dtype=np.uint8) * 245

    # Distinct colored regions (conceptual-model-like)
    img[20:140, 20:260] = [210, 60, 60]
    img[20:140, 300:540] = [60, 210, 60]
    img[180:300, 20:540] = [60, 60, 210]

    # Scribbled edges / annotations
    for _ in range(80):
        x = rng.integers(20, 540)
        y = rng.integers(20, 300)
        angle = rng.random() * 3.14159
        length = rng.integers(8, 30)
        for l in range(length):
            px = int(x + l * np.cos(angle))
            py = int(y + l * np.sin(angle))
            if 0 <= px < 560 and 0 <= py < 320:
                img[py, px] = [25, 25, 25]

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "out"
        result = run_pipeline(
            img,
            n_layers=3,
            quality_preference="fast",
            skip_non_velocity_model=False,
            use_vlm=False,
            output_dir=out_dir,
            save_intermediates=True,
        )

        assert result["status"] == "ok"
        assert len(result["panels"]) >= 1

        ok_panels = [p for p in result["panels"] if p["status"] == "ok"]
        assert len(ok_panels) >= 1

        for p in ok_panels:
            assert p["n_components"] >= 1
            assert p["n_polygons"] >= 1
            assert "properties" in p

        # Check artifacts exported
        files = list(out_dir.iterdir()) if out_dir.exists() else []
        names = [f.name for f in files]
        assert any(n.endswith("_tomo.xyz") for n in names)
        assert any(n.endswith("_polygons.geojson") for n in names)
        assert any(n.endswith("_properties.json") for n in names)


def test_unknown_colors_get_auto_properties() -> None:
    """Layers with unknown color names should auto-generate Vp/Vs/rho."""
    rng = np.random.default_rng(7)
    img = np.ones((200, 360, 3), dtype=np.uint8) * 250
    img[20:180, 20:170] = [200, 70, 70]
    img[20:180, 190:340] = [70, 70, 200]

    # Add noise to trigger non-v4 engine
    img = np.clip(img.astype(np.int16) + rng.integers(-10, 10, img.shape), 0, 255).astype(np.uint8)

    result = run_pipeline(
        img,
        n_layers=2,
        quality_preference="fast",
        skip_non_velocity_model=False,
        use_vlm=False,
        save_intermediates=False,
    )
    assert result["status"] == "ok"
    ok_panels = [p for p in result["panels"] if p["status"] == "ok"]
    assert len(ok_panels) >= 1
    for p in ok_panels:
        props = p["properties"]
        for name, vals in props.items():
            assert "Vp" in vals
            assert "Vs" in vals
            assert "rho" in vals
            assert vals["Vp"] > vals["Vs"]


def test_grayscale_image_is_processed() -> None:
    """Grayscale conceptual figure should route to grayscale engine."""
    rng = np.random.default_rng(99)
    # True grayscale with distinct regions
    img = np.ones((240, 400, 3), dtype=np.uint8) * 200
    img[20:120, 20:190] = 120
    img[20:120, 210:380] = 80
    img[140:220, 20:380] = 160

    # Add edges
    for _ in range(40):
        x = rng.integers(20, 380)
        y = rng.integers(20, 220)
        angle = rng.random() * 3.14159
        for l in range(rng.integers(8, 25)):
            px = int(x + l * np.cos(angle))
            py = int(y + l * np.sin(angle))
            if 0 <= px < 400 and 0 <= py < 240:
                img[py, px] = 30

    result = run_pipeline(
        img,
        n_layers=3,
        quality_preference="fast",
        skip_non_velocity_model=False,
        use_vlm=False,
        save_intermediates=False,
    )
    assert result["status"] == "ok"
    ok_panels = [p for p in result["panels"] if p["status"] == "ok"]
    assert len(ok_panels) >= 1


def test_multi_panel_figure() -> None:
    """Multi-panel figure should detect and process each panel."""
    img = np.ones((200, 400, 3), dtype=np.uint8) * 245

    # Panel 1: top-left
    img[10:90, 10:190] = [210, 60, 60]
    # Panel 2: top-right
    img[10:90, 210:390] = [60, 210, 60]
    # Panel 3: bottom
    img[110:190, 10:390] = [60, 60, 210]

    # White gaps between panels (to help detection)
    img[95:105, :] = 255
    img[:, 195:205] = 255

    result = run_pipeline(
        img,
        n_layers=3,
        quality_preference="fast",
        skip_non_velocity_model=False,
        use_vlm=False,
        save_intermediates=False,
    )
    assert result["status"] == "ok"
    # Should have at least 2 panels (detection may find 2-3)
    assert len(result["panels"]) >= 2


def test_batch_processor_resume() -> None:
    """Batch processor should skip already-processed images on resume."""
    rng = np.random.default_rng(123)
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        images_dir = td_path / "images"
        output_dir = td_path / "output"
        images_dir.mkdir()

        # Create 2 images (must be >= 300x200 for size gate)
        img1 = np.ones((200, 320, 3), dtype=np.uint8) * 240
        img1[20:100, 20:150] = [200, 60, 60]
        img1[100:180, 20:150] = [60, 60, 200]
        Image.fromarray(img1).save(images_dir / "img1.jpg")

        img2 = np.ones((200, 320, 3), dtype=np.uint8) * 240
        img2[20:100, 170:300] = [60, 200, 60]
        img2[100:180, 170:300] = [200, 200, 60]
        Image.fromarray(img2).save(images_dir / "img2.jpg")

        # First run
        summary1 = process_directory(
            images_dir=images_dir,
            output_dir=output_dir,
            n_layers=2,
            quality_preference="fast",
            use_vlm=False,
            skip_non_velocity_model=False,
            resume=False,
        )
        assert summary1["total"] == 2
        assert summary1["errors"] == 0
        assert summary1["processed"] == 2

        # Second run with resume
        summary2 = process_directory(
            images_dir=images_dir,
            output_dir=output_dir,
            n_layers=2,
            quality_preference="fast",
            use_vlm=False,
            skip_non_velocity_model=False,
            resume=True,
        )
        assert summary2["total"] == 2
        # Should not process any new images
        assert summary2["processed"] == 0


def test_batch_processor_error_isolation() -> None:
    """One bad image should not stop the batch."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        images_dir = td_path / "images"
        output_dir = td_path / "output"
        images_dir.mkdir()

        # Valid image (must be >= 300x200 for size gate)
        img1 = np.ones((200, 320, 3), dtype=np.uint8) * 240
        img1[20:100, 20:150] = [200, 60, 60]
        img1[100:180, 20:150] = [60, 60, 200]
        Image.fromarray(img1).save(images_dir / "img1.jpg")

        # Corrupted image (empty file — will cause PIL to fail)
        (images_dir / "img2_bad.jpg").write_bytes(b"not an image")

        summary = process_directory(
            images_dir=images_dir,
            output_dir=output_dir,
            n_layers=2,
            quality_preference="fast",
            use_vlm=False,
            skip_non_velocity_model=False,
            resume=False,
        )
        assert summary["total"] == 2
        assert summary["errors"] == 1
        assert summary["processed"] == 1
        assert summary["results"]["img2_bad.jpg"]["status"] == "error"
