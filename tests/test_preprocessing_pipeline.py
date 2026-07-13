"""Tests for geoseg.preprocessing.pipeline."""
from __future__ import annotations

import numpy as np
import pytest

from geoseg.preprocessing.pipeline import (
    ArtifactAbsorptionConfig,
    process_image,
)


def _synthetic_image() -> np.ndarray:
    """Create a small two-panel figure with a red line and a black cross."""
    img = np.full((220, 260, 3), 255, dtype=np.uint8)
    # Panel 0: blue region
    img[20:100, 50:230] = [80, 120, 200]
    # Red fault line inside panel 0
    img[30:90, 100:103] = [200, 40, 40]
    # Panel 1: green region
    img[120:200, 50:230] = [80, 180, 100]
    # Black cross inside panel 1
    img[155:160, 135:155] = [20, 20, 20]
    img[150:165, 142:148] = [20, 20, 20]
    return img


@pytest.fixture
def synthetic_config(tmp_path):
    img_path = tmp_path / "input.jpg"
    from PIL import Image

    Image.fromarray(_synthetic_image()).save(img_path)
    return ArtifactAbsorptionConfig(
        image_path=img_path,
        output_dir=tmp_path / "out",
        panel_bboxes=[(50, 20, 180, 80), (50, 120, 180, 80)],
        n_layers=3,
        red_params={
            "frangi_threshold": 0.01,
            "min_area": 10,
            "angle_ranges": [[70, 110]],
            "dilation_kernel_size": 5,
        },
        cross_params={
            "max_gray": 100,
            "min_diff": 5,
            "cross_area_range": [10, 200],
        },
        inpaint_dilate_iters=2,
        per_panel=True,
        artifact_labels={0: [2], 1: [2]},
    )


def test_process_image_per_panel_outputs_exist(synthetic_config):
    result = process_image(synthetic_config)

    assert result["red_pixels"] > 0
    assert result["cross_pixels"] > 0
    assert (synthetic_config.output_dir / "03_cleaned.jpg").exists()
    assert (synthetic_config.output_dir / "09_per_panel_overlay.jpg").exists()
    assert (synthetic_config.output_dir / "09_per_panel_labels.npz").exists()

    for panel_id in range(2):
        panel_dir = synthetic_config.output_dir / "panels" / f"panel_{panel_id}"
        assert (panel_dir / "labels.npz").exists()
        assert (panel_dir / "overlay_legend.jpg").exists()


def test_process_image_per_panel_label_offsets_avoid_collision(synthetic_config):
    result = process_image(synthetic_config)

    full = np.load(synthetic_config.output_dir / "09_per_panel_labels.npz")["labels"]
    # Extract per-panel regions and verify their label sets do not overlap.
    panel0 = full[20:100, 50:230]
    panel1 = full[120:200, 50:230]
    labels0 = set(np.unique(panel0[panel0 >= 0]))
    labels1 = set(np.unique(panel1[panel1 >= 0]))
    assert labels0.isdisjoint(labels1)


def test_artifact_absorption_config_from_dict_roundtrip():
    data = {
        "image_path": "/tmp/img.jpg",
        "output_dir": "/tmp/out",
        "n_layers": 4,
        "per_panel": True,
        "artifact_labels": {"0": [4], "1": [3]},
    }
    cfg = ArtifactAbsorptionConfig.from_dict(data)
    assert cfg.n_layers == 4
    assert cfg.per_panel is True
    assert cfg.artifact_labels == {"0": [4], "1": [3]}
