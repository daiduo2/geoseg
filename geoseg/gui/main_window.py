"""Main window for geoseg GUI.

Minimal first version:
- Central: SegmentationView
- Bottom: Export button + area threshold slider
- Menu: Open image, Open labels
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSlider,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from geoseg.gui.figure_selector import FigureSelectorDialog
from geoseg.gui.panel_selector import PanelSelectorDialog
from geoseg.gui.pdf_import_worker import PdfImportWorker
from geoseg.gui.pdf_page_review_worker import PdfPageReviewWorker
from geoseg.gui.segmentation_view import SegmentationView
from geoseg.modules.exporter.specfem import (
    labels_to_grids,
    write_parfile_snippet,
    write_tomography_file,
)
from geoseg.modules.post_process.properties import generate_properties_for_layers
from geoseg.modules.segment_engines.full_pipeline import process_figure
from geoseg.modules.segment_engines.router import route_and_segment


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("geoseg — Region Adjustment")
        self.resize(1200, 900)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        # Segmentation view
        self.view = SegmentationView()
        layout.addWidget(self.view, stretch=1)

        # Bottom controls
        controls = QHBoxLayout()
        self._threshold_label = QLabel("Area threshold: 2%")
        controls.addWidget(self._threshold_label)

        self._threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self._threshold_slider.setRange(1, 20)
        self._threshold_slider.setValue(2)
        self._threshold_slider.valueChanged.connect(self._on_threshold_changed)
        controls.addWidget(self._threshold_slider)

        self._export_btn = QPushButton("Export SPECFEM")
        self._export_btn.clicked.connect(self._on_export)
        controls.addWidget(self._export_btn)

        self._reload_btn = QPushButton("Reload")
        self._reload_btn.clicked.connect(self._on_reload)
        controls.addWidget(self._reload_btn)

        controls_widget = QWidget()
        controls_widget.setLayout(controls)
        layout.addWidget(controls_widget)

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)

        # Menu
        self._build_menu()

        self._img_rgb: np.ndarray | None = None
        self._labels: np.ndarray | None = None
        self._labels_original: np.ndarray | None = None

        # Retain PDF context so users can pick another figure after a skip.
        self._current_pdf_images: list[str] | None = None
        self._current_extracted_dir: str | None = None
        self._current_caption_map: dict[str, str] | None = None
        self._current_text_blocks_map: dict[int, list[dict]] | None = None
        self._current_page_map: dict[str, int] | None = None

    def _build_menu(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("File")

        open_pdf = QAction("Open PDF...", self)
        open_pdf.triggered.connect(self._open_pdf)
        file_menu.addAction(open_pdf)

        file_menu.addSeparator()

        open_img = QAction("Open Image...", self)
        open_img.setShortcut(QKeySequence.StandardKey.Open)
        open_img.triggered.connect(self._open_image)
        file_menu.addAction(open_img)

        open_labels = QAction("Open Labels (.npz)...", self)
        open_labels.triggered.connect(self._open_labels)
        file_menu.addAction(open_labels)

        export_act = QAction("Export SPECFEM...", self)
        export_act.setShortcut(QKeySequence("Ctrl+S"))
        export_act.triggered.connect(self._on_export)
        file_menu.addAction(export_act)

    def _open_pdf(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open PDF", "", "PDF (*.pdf)")
        if path:
            self._start_pdf_import(path)

    def _start_pdf_import(self, pdf_path: str) -> None:
        self._status.showMessage(f"Importing {Path(pdf_path).name} ...")

        # Reset parallel result holders
        self._mineru_result: tuple[str, list[str], dict, dict, dict] | None = None
        self._page_review_result: tuple[list[int], list[dict]] | None = None

        # Progress dialog covers both workers
        self._progress_dialog = QProgressDialog(
            f"Importing {Path(pdf_path).name}...\nMinerU extraction + local page review",
            "Cancel", 0, 0, self,
        )
        self._progress_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        self._progress_dialog.setAutoClose(True)
        self._progress_dialog.show()

        # Worker 1: MinerU extraction
        self._import_worker = PdfImportWorker(pdf_path)
        self._import_worker.progress.connect(self._status.showMessage)
        self._import_worker.finished_success.connect(self._on_mineru_done)
        self._import_worker.finished_error.connect(self._on_pdf_import_error)
        self._import_worker.finished.connect(self._check_parallel_done)
        self._progress_dialog.canceled.connect(self._import_worker.requestInterruption)
        self._import_worker.start()

        # Worker 2: Local page rendering + VLM review
        self._review_worker = PdfPageReviewWorker(pdf_path, use_vlm=True)
        self._review_worker.progress.connect(self._status.showMessage)
        self._review_worker.review_done.connect(self._on_page_review_done)
        self._review_worker.review_error.connect(self._on_page_review_error)
        self._review_worker.finished.connect(self._check_parallel_done)
        self._progress_dialog.canceled.connect(self._review_worker.requestInterruption)
        self._review_worker.start()

    def _on_mineru_done(
        self, extracted_dir: str, image_paths: list[str], page_map: dict, caption_map: dict,
        text_blocks_map: dict,
    ) -> None:
        self._mineru_result = (extracted_dir, image_paths, page_map, caption_map, text_blocks_map)

    def _on_page_review_done(self, target_pages: list[int], page_details: list[dict]) -> None:
        self._page_review_result = (target_pages, page_details)

    def _on_page_review_error(self, message: str) -> None:
        # Page review failure is non-fatal: we will show all figures
        self._status.showMessage(f"Page review failed: {message}")
        self._page_review_result = ([], [])

    def _check_parallel_done(self) -> None:
        """Called when either worker finishes. Show selector only when both are done."""
        if self._mineru_result is None or self._page_review_result is None:
            return

        if self._progress_dialog is not None:
            self._progress_dialog.close()

        extracted_dir, image_paths, page_map, caption_map, text_blocks_map = self._mineru_result
        target_pages, page_details = self._page_review_result

        # Filter figures to only those from target pages
        if target_pages:
            filtered = [
                path for path in image_paths
                if page_map.get(path) in target_pages
            ]
        else:
            filtered = []

        # Fallback: if filtering yields nothing, show all extracted figures
        if not filtered and image_paths:
            filtered = image_paths

        self._status.showMessage(
            f"MinerU: {len(image_paths)} figures, "
            f"VLM target pages: {len(target_pages)}, "
            f"showing {len(filtered)} figures."
        )
        self._current_extracted_dir = extracted_dir
        self._current_pdf_images = filtered
        self._current_caption_map = caption_map
        self._current_text_blocks_map = text_blocks_map
        self._current_page_map = page_map
        self._open_figure_selector(filtered)

    def _open_figure_selector(self, image_paths: list[str]) -> None:
        if not image_paths:
            QMessageBox.information(self, "No Figures", "No figures found in PDF.")
            return
        dialog = FigureSelectorDialog(image_paths, self)
        dialog.figure_selected.connect(self._load_figure_for_edit)
        dialog.exec()

    def _offer_reselect(self) -> None:
        """Let user pick another figure from the same PDF after a skip."""
        if not self._current_pdf_images:
            return
        reply = QMessageBox.question(
            self,
            "Figure Skipped",
            "This figure was skipped. Select another figure from the same PDF?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._open_figure_selector(self._current_pdf_images)

    def _on_pdf_import_error(self, message: str) -> None:
        self._status.showMessage(f"PDF import failed: {message}")
        if self._progress_dialog is not None:
            self._progress_dialog.close()

    def _load_figure_for_edit(self, image_path: str) -> None:
        self._status.showMessage(f"Loading {Path(image_path).name} ...")
        img = np.array(Image.open(image_path).convert("RGB"))

        caption = ""
        if self._current_caption_map:
            caption = self._current_caption_map.get(image_path, "")

        # Gather text blocks from the figure's page for spatial context
        text_blocks: list[dict] = []
        if self._current_text_blocks_map and self._current_page_map:
            page_idx = self._current_page_map.get(image_path)
            if page_idx is not None:
                text_blocks = self._current_text_blocks_map.get(page_idx, [])

        result = process_figure(
            img,
            caption=caption,
            text_blocks=text_blocks,
            n_layers=5,
            quality_preference="balanced",
            skip_non_velocity_model=True,
            use_vlm=True,
        )

        if result["summary"]["status"] == "skipped":
            reason = result["summary"].get("reason", "unknown")
            self._status.showMessage(f"Figure skipped: {reason}")
            self._offer_reselect()
            return

        panels = result.get("panels", [])
        if not panels:
            self._status.showMessage("No panels found in figure.")
            self._offer_reselect()
            return

        # Show figure-level review warnings
        review_warnings = result.get("summary", {}).get("review_warnings", [])
        if review_warnings:
            warn_text = "; ".join(review_warnings)
            self._status.showMessage(f"Review warnings: {warn_text}")

        if len(panels) == 1:
            self._load_panel(img, panels[0], image_path, panel_idx=0, total_panels=1)
        else:
            # Multiple panels detected — let user choose
            dialog = PanelSelectorDialog(img, panels, self)
            dialog.panel_selected.connect(
                lambda idx: self._on_panel_selected(img, panels, image_path, idx)
            )
            code = dialog.exec()
            if code != QDialog.DialogCode.Accepted:
                self._offer_reselect()

    def _on_panel_selected(
        self,
        img: np.ndarray,
        panels: list[dict],
        image_path: str,
        selected_idx: int,
    ) -> None:
        """Handle panel selection from dialog."""
        caption = ""
        if self._current_caption_map:
            caption = self._current_caption_map.get(image_path, "")

        if selected_idx == -1:
            # Whole image — bypass panel detection and segment entire image
            self._status.showMessage("Segmenting whole image ...")
            seg = route_and_segment(
                img,
                n_layers=5,
                quality_preference="balanced",
                is_velocity_model=True,
            )
            self._img_rgb = img
            self._labels = seg["labels"]
            self._labels_original = seg["labels"].copy()
            self._try_load_view()
            self._status.showMessage(
                f"Loaded whole image from {Path(image_path).name}"
            )
        elif 0 <= selected_idx < len(panels):
            self._load_panel(img, panels[selected_idx], image_path, selected_idx, len(panels), caption)
        else:
            self._status.showMessage("No panel selected.")
            self._offer_reselect()

    def _load_panel(
        self,
        img: np.ndarray,
        panel: dict,
        image_path: str,
        panel_idx: int,
        total_panels: int,
        caption: str = "",
    ) -> None:
        """Load a single panel into the editor."""
        seg = panel.get("segmentation")
        if seg is None:
            self._status.showMessage("No segmentation result for this panel.")
            self._offer_reselect()
            return

        labels = seg["labels"]
        x, y, pw, ph = panel["bbox"]
        h_img, w_img = img.shape[:2]
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        pw = min(pw, w_img - x)
        ph = min(ph, h_img - y)
        panel_img = img[y : y + ph, x : x + pw]

        self._img_rgb = panel_img
        self._labels = labels
        self._labels_original = labels.copy()
        self._try_load_view()

        # Show review warnings if any
        review = panel.get("review", {})
        n_layers_found = review.get("n_layers_found", 0)
        msg = f"Loaded panel {panel_idx + 1}/{total_panels} from {Path(image_path).name}"
        if n_layers_found < 2:
            msg += f" (WARNING: only {n_layers_found} layer found)"
        self._status.showMessage(msg)

    def _open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self._img_rgb = np.array(Image.open(path).convert("RGB"))
            self._status.showMessage(f"Loaded image: {Path(path).name} {self._img_rgb.shape}")
            self._try_load_view()

    def _open_labels(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open Labels", "", "NPZ (*.npz)")
        if path:
            data = np.load(path)
            self._labels = data["labels"].astype(np.int32)
            self._labels_original = self._labels.copy()
            self._status.showMessage(f"Loaded labels: {Path(path).name} {self._labels.shape}")
            self._try_load_view()

    def _try_load_view(self) -> None:
        if self._img_rgb is not None and self._labels is not None:
            threshold = self._threshold_slider.value() / 100.0
            self.view.load_from_arrays(self._img_rgb, self._labels, threshold)
            n_components = len(self.view._region_items)
            n_small = sum(1 for ri in self.view._region_items if ri.is_small)
            self._status.showMessage(
                f"Regions: {n_components} total, {n_small} small, {n_components - n_small} locked"
            )

    def _on_threshold_changed(self, value: int) -> None:
        self._threshold_label.setText(f"Area threshold: {value}%")
        if self._img_rgb is not None and self._labels is not None:
            self._try_load_view()

    def _on_reload(self) -> None:
        if self._labels_original is not None:
            self._labels = self._labels_original.copy()
            self._try_load_view()

    def _on_export(self) -> None:
        labels = self.view.get_labels()
        if labels is None:
            self._status.showMessage("No labels to export")
            return

        out_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if not out_dir:
            return

        out_path = Path(out_dir)
        color_names = [f"layer_{i}" for i in sorted(set(labels.flatten()) - {0})]
        props = generate_properties_for_layers(color_names)
        vp, vs, rho = labels_to_grids(labels, props, color_names=color_names)

        h, w = labels.shape
        x_coords = np.linspace(0, w - 1, w)
        z_coords = np.linspace(0, h - 1, h)

        write_tomography_file(vp, vs, rho, x_coords, z_coords, out_path / "tomo.xyz")
        write_parfile_snippet(color_names, props, out_path / "parfile_snippet.txt", nx=w, nz=h)

        self._status.showMessage(f"Exported to {out_path}")
