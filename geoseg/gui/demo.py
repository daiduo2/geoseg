"""Demo / test scenario for GUI segmentation view.

Creates a synthetic figure with known ground-truth labels, and opens the
interactive view.  This avoids relying on the segmentation pipeline to
successfully split a toy synthetic image.

Run:
    python -m geoseg.gui.demo
"""

from __future__ import annotations

import sys

import numpy as np
from PySide6.QtWidgets import QApplication

from geoseg.gui.main_window import MainWindow


def _make_test_image() -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic conceptual figure with small draggable fragments."""
    rng = np.random.default_rng(42)
    h, w = 200, 350
    img = np.ones((h, w, 3), dtype=np.uint8) * 245
    labels = np.zeros((h, w), dtype=np.int32)

    # Layer 1 — red
    img[20:90, 20:160] = [210, 60, 60]
    labels[20:90, 20:160] = 1
    # Small red fragment (disconnected from big red)
    img[45:55, 165:185] = [210, 60, 60]
    labels[45:55, 165:185] = 1

    # Layer 2 — green (big region first)
    img[20:90, 190:330] = [60, 210, 60]
    labels[20:90, 190:330] = 2

    # Layer 3 — blue
    img[110:180, 20:330] = [60, 60, 210]
    labels[110:180, 20:330] = 3
    # Small blue fragment above main blue
    img[95:105, 80:95] = [60, 60, 210]
    labels[95:105, 80:95] = 3

    # Small green fragment inside blue region (set AFTER layer 3 so it overwrites)
    img[130:140, 160:175] = [60, 210, 60]
    labels[130:140, 160:175] = 2

    # Noise/edges to make it look more real
    for _ in range(60):
        x, y = rng.integers(10, w - 10), rng.integers(10, h - 10)
        angle = rng.random() * 3.14159
        for l in range(rng.integers(5, 20)):
            px = int(x + l * np.cos(angle))
            py = int(y + l * np.sin(angle))
            if 0 <= px < w and 0 <= py < h:
                img[py, px] = [30, 30, 30]

    return img, labels


def main() -> int:
    app = QApplication(sys.argv)

    print("Generating synthetic test figure ...")
    img, labels = _make_test_image()
    n_layers = len(np.unique(labels)) - 1  # exclude background 0
    print(f"Image: {img.shape}, Labels: {labels.shape}, Layers: {n_layers}")

    window = MainWindow()
    window._img_rgb = img
    window._labels = labels
    window._labels_original = labels.copy()
    window._try_load_view()
    window.show()

    print("GUI opened. Try:")
    print("  - Drag small white-bordered regions onto big black-bordered regions")
    print("  - Right-click small region for context menu assignment")
    print("  - Adjust area threshold slider to change what counts as 'small'")
    print("  - Export SPECFEM button writes tomo.xyz + parfile_snippet")

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
