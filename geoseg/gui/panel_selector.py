"""Panel selection dialog: choose which detected panel to edit.

Shown after figure classification when detect_panels returns multiple panels,
or when the user wants to override automatic detection and use the whole image.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)


THUMB_SIZE = 160


def _array_to_pixmap(arr: np.ndarray) -> QPixmap:
    """Convert RGB uint8 array to QPixmap."""
    h, w = arr.shape[:2]
    if arr.shape[2] == 4:
        fmt = QImage.Format.Format_RGBA8888
        stride = w * 4
        data = arr.tobytes()
    else:
        fmt = QImage.Format.Format_RGB888
        stride = w * 3
        data = arr.tobytes()
    qimg = QImage(data, w, h, stride, fmt)
    return QPixmap.fromImage(qimg)


def _make_thumbnail(arr: np.ndarray, size: int = THUMB_SIZE) -> QPixmap:
    """Create a scaled thumbnail pixmap from an RGB array."""
    img = Image.fromarray(arr)
    img.thumbnail((size, size))
    data = img.tobytes("raw", "RGB")
    qimg = QImage(data, img.width, img.height, img.width * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


class PanelSelectorDialog(QDialog):
    """Dialog for choosing a panel from auto-detected candidates.

    Signals:
        panel_selected(int): panel index chosen, or -1 for whole image
    """

    panel_selected = Signal(int)

    def __init__(
        self,
        img_rgb: np.ndarray,
        panels: list[dict],
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Select Panel")
        self.resize(700, 500)
        self._img_rgb = img_rgb
        self._panels = panels
        self._selected_idx: int | None = None

        layout = QVBoxLayout(self)

        # Header
        n = len(panels)
        header = QLabel(
            f"Detected {n} panel{'s' if n > 1 else ''}. "
            "Pick one to edit, or use the whole image:"
        )
        layout.addWidget(header)

        # Thumbnail list
        self._list = QListWidget()
        self._list.setViewMode(QListWidget.ViewMode.IconMode)
        self._list.setIconSize(THUMB_SIZE)
        self._list.setGridSize(THUMB_SIZE + 60)
        self._list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._list.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self._list)

        self._populate_items()

        # Buttons
        btn_layout = QHBoxLayout()

        self._whole_btn = QPushButton("Use Whole Image")
        self._whole_btn.setToolTip("Bypass panel detection and segment the entire image")
        self._whole_btn.clicked.connect(self._on_whole_image)
        btn_layout.addWidget(self._whole_btn)

        btn_layout.addStretch()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        btn_layout.addWidget(buttons)

        layout.addLayout(btn_layout)

    def _populate_items(self) -> None:
        # "Whole image" option at the top
        whole_thumb = _make_thumbnail(self._img_rgb)
        h, w = self._img_rgb.shape[:2]
        whole_item = QListWidgetItem(
            whole_thumb,
            f"Whole image\n{w}x{h}"
        )
        whole_item.setData(Qt.ItemDataRole.UserRole, -1)
        whole_item.setToolTip("Use the entire image as a single panel")
        self._list.addItem(whole_item)

        # Individual panels
        for i, pb in enumerate(self._panels):
            x, y, pw, ph = pb["bbox"]
            h_img, w_img = self._img_rgb.shape[:2]
            x = max(0, min(x, w_img - 1))
            y = max(0, min(y, h_img - 1))
            pw = min(pw, w_img - x)
            ph = min(ph, h_img - y)
            panel_img = self._img_rgb[y : y + ph, x : x + pw]

            thumb = _make_thumbnail(panel_img)
            conf = pb.get("confidence", 0.0)
            label_text = f"Panel {i}\n{pw}x{ph}\nconf:{conf:.2f}"
            item = QListWidgetItem(thumb, label_text)
            item.setData(Qt.ItemDataRole.UserRole, i)
            item.setToolTip(f"bbox: [{x}, {y}, {pw}, {ph}]\nconfidence: {conf:.2f}")
            self._list.addItem(item)

        self._list.setCurrentRow(0)

    def _on_double_click(self, item: QListWidgetItem) -> None:
        self._selected_idx = item.data(Qt.ItemDataRole.UserRole)
        self.panel_selected.emit(self._selected_idx)
        self.accept()

    def _on_whole_image(self) -> None:
        self._selected_idx = -1
        self.panel_selected.emit(-1)
        self.accept()

    def _on_accept(self) -> None:
        selected = self._list.selectedItems()
        if not selected:
            return
        self._selected_idx = selected[0].data(Qt.ItemDataRole.UserRole)
        self.panel_selected.emit(self._selected_idx)
        self.accept()

    def selected_index(self) -> int | None:
        return self._selected_idx
