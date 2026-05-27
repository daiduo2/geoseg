"""Background worker: render PDF pages locally and VLM-classify each page.

Used in parallel with PdfImportWorker (MinerU) to pre-filter which pages
contain conceptual model figures before showing the figure selector.
"""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from PySide6.QtCore import QThread, Signal


class PdfPageReviewWorker(QThread):
    """Render PDF pages and run VLM figure classification on each.

    Emits per-page results so the GUI can filter figures by page number.
    """

    progress = Signal(str)
    review_done = Signal(list, list)  # (target_page_indices, page_details)
    review_error = Signal(str)

    def __init__(
        self,
        pdf_path: str,
        dpi: int = 150,
        page_indices: list[int] | None = None,
        use_vlm: bool = True,
    ):
        super().__init__()
        self._pdf_path = Path(pdf_path)
        self._dpi = dpi
        self._page_indices = page_indices
        self._use_vlm = use_vlm

    def run(self) -> None:
        try:
            self._do_review()
        except Exception as exc:
            self.review_error.emit(str(exc))

    def _do_review(self) -> None:
        doc = fitz.open(self._pdf_path)
        total = len(doc)
        pages_to_review = self._page_indices if self._page_indices else list(range(total))

        target_pages: list[int] = []
        page_details: list[dict] = []

        for idx in pages_to_review:
            if idx >= total:
                continue
            self.progress.emit(f"Reviewing page {idx + 1}/{total} ...")
            page = doc[idx]
            img = self._render_page(page)

            # Fast CV heuristic first (cheap filter)
            from geoseg.modules.cv_detect.figure_classifier import classify as cv_classify
            cv_result = cv_classify(img)
            cv_type = cv_result["figure_type"]

            # Skip clearly non-target pages quickly
            if cv_type in ("observational_data", "other") and not self._use_vlm:
                page_details.append({
                    "page_idx": idx,
                    "cv_type": cv_type,
                    "vlm_type": None,
                    "vlm_confidence": 0.0,
                    "target": False,
                })
                continue

            # VLM semantic classification
            vlm_type = None
            vlm_confidence = 0.0
            target = False
            if self._use_vlm:
                try:
                    from geoseg.modules.vlm_client import classify_figure
                    vlm_result = classify_figure(
                        img,
                        mode="auto",
                        min_confidence=0.0,  # Don't raise on low confidence
                    )
                    if hasattr(vlm_result, "figure_type"):
                        vlm_type = vlm_result.figure_type
                        vlm_confidence = vlm_result.confidence
                    else:
                        vlm_type = vlm_result.get("figure_type")
                        vlm_confidence = vlm_result.get("confidence", 0.0)

                    if vlm_type in ("velocity_model", "geological_cross_section"):
                        target = True
                        target_pages.append(idx)
                except Exception as exc:
                    # VLM failed — fall back to CV decision
                    if cv_type not in ("observational_data", "other"):
                        target = True
                        target_pages.append(idx)
                    vlm_type = f"error: {exc}"
            else:
                # No VLM — use CV only
                if cv_type not in ("observational_data", "other"):
                    target = True
                    target_pages.append(idx)

            page_details.append({
                "page_idx": idx,
                "cv_type": cv_type,
                "vlm_type": vlm_type,
                "vlm_confidence": vlm_confidence,
                "target": target,
            })

        doc.close()
        self.progress.emit(
            f"Page review complete: {len(target_pages)}/{len(pages_to_review)} target pages."
        )
        self.review_done.emit(target_pages, page_details)

    def _render_page(self, page: fitz.Page) -> np.ndarray:
        """Render a fitz page to RGB uint8 array at self._dpi."""
        zoom = self._dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n > 3:
            img = img[:, :, :3]
        return img
