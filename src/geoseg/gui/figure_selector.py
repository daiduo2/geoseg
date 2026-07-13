"""Figure selection dialog: grid of thumbnails from extracted PDF figures.

User picks one figure to load into the segmentation editor.
Each thumbnail is pre-classified (fast CV heuristic) to hint processability.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
)

from geoseg.modules.cv_detect.figure_classifier import classify


THUMB_SIZE = 128


def _classify_hint(path: str) -> dict:
    """Fast CV-only hint; returns dict with keys figure_type, sat."""
    try:
        import numpy as np

        from geoseg.modules.segment_engines._shared import saturation_ratio

        img = np.array(Image.open(path).convert("RGB"))
        result = classify(img)
        ft = result.get("figure_type", "unknown")
        sat = saturation_ratio(img)
        return {
            "figure_type": ft,
            "saturation_ratio": sat,
        }
    except Exception:
        return {"figure_type": "unknown", "saturation_ratio": 0.0}


class FigureSelectorDialog(QDialog):
    """Dialog showing extracted figure thumbnails with processability hints."""

    figure_selected = Signal(str)  # emitted with chosen image path

    def __init__(self, image_paths: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Figure")
        self.resize(800, 600)
        self._image_paths = image_paths
        self._selected_path: str | None = None

        layout = QVBoxLayout(self)

        # Header
        header = QLabel(
            f"Found {len(image_paths)} figures. "
            "Labels show fast CV heuristic type; VLM confirms after selection:"
        )
        layout.addWidget(header)

        # Thumbnail list
        self._list = QListWidget()
        self._list.setViewMode(QListWidget.ViewMode.IconMode)
        self._list.setIconSize(THUMB_SIZE)
        self._list.setGridSize(THUMB_SIZE + 50)
        self._list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._list.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self._list)

        self._populate_thumbnails()

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _populate_thumbnails(self) -> None:
        for path in self._image_paths:
            try:
                thumb = self._make_thumbnail(path)
            except Exception:
                thumb = QPixmap(THUMB_SIZE, THUMB_SIZE)
                thumb.fill(Qt.GlobalColor.gray)

            hint = _classify_hint(path)
            ft = hint["figure_type"]
            sat = hint["saturation_ratio"]

            # Label text: figure type on first line, saturation on second
            label_text = f"{ft}\nsat:{sat:.2f}"
            item = QListWidgetItem(QIcon(thumb), label_text)
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setToolTip(
                f"File: {Path(path).name}\n"
                f"CV type: {ft}\n"
                f"Saturation: {sat:.3f}\n"
                f"(VLM will confirm after selection)"
            )

            self._list.addItem(item)

    def _make_thumbnail(self, path: str) -> QPixmap:
        img = Image.open(path)
        img.thumbnail((THUMB_SIZE, THUMB_SIZE))
        if img.mode != "RGB":
            img = img.convert("RGB")
        data = img.tobytes("raw", "RGB")
        from PySide6.QtGui import QImage

        qimg = QImage(data, img.width, img.height, img.width * 3, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def _on_double_click(self, item: QListWidgetItem) -> None:
        self._selected_path = item.data(Qt.ItemDataRole.UserRole)
        self.figure_selected.emit(self._selected_path)
        self.accept()

    def _on_accept(self) -> None:
        selected = self._list.selectedItems()
        if not selected:
            return
        self._selected_path = selected[0].data(Qt.ItemDataRole.UserRole)
        self.figure_selected.emit(self._selected_path)
        self.accept()
