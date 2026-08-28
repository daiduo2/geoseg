import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts/export_segmentation_txt.py"
)
SPEC = importlib.util.spec_from_file_location("export_segmentation_txt", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_normalize_direct_colorbar_labels_preserves_class_zero() -> None:
    labels = np.array([[-1, 0], [1, 0]], dtype=np.int16)
    palette_rgb = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)

    normalized, palette, mapping = MODULE._normalize_direct_colorbar_labels(
        labels, palette_rgb
    )

    np.testing.assert_array_equal(normalized, [[0, 1], [2, 1]])
    assert palette == {0: (255, 255, 255), 1: (10, 20, 30), 2: (40, 50, 60)}
    assert mapping == {-1: 0, 0: 1, 1: 2}


def test_export_direct_colorbar_run_writes_round_trip_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    figure_dir = run_dir / "fig7"
    cleanup_dir = figure_dir / "annotation_cleanup"
    cleanup_dir.mkdir(parents=True)
    labels = np.array([[-1, 0, 0], [1, 1, 0]], dtype=np.int16)
    palette = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)
    np.savez_compressed(cleanup_dir / "labels.npz", labels=labels, palette_rgb=palette)
    Image.fromarray(np.full((2, 3, 3), 128, dtype=np.uint8)).save(
        figure_dir / "01_panel.png"
    )
    output_dir = run_dir / "txt_export"
    output_dir.mkdir()

    assert MODULE._export_direct_colorbar_run(run_dir, output_dir, ["fig7"]) == 0

    exported = np.loadtxt(
        output_dir / "fig7_labels.txt", skiprows=1, dtype=np.int32
    )
    assert exported[:, 2].tolist() == [0, 1, 1, 2, 2, 1]
    assert (output_dir / "fig7_palette.txt").exists()
    assert (output_dir / "fig7_reconstructed.png").exists()
    assert (output_dir / "fig7_comparison.png").exists()
    assert (output_dir / "export_manifest.json").exists()
