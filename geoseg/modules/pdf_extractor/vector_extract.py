"""M0.5v: rasterize a PDF page (vector content) to a high-resolution bitmap.

Complements `extract.py` which only pulls embedded XObject images. When the
target figure is drawn as PDF vector primitives (paths, shadings, text) there
is no XObject to extract; rasterizing the rendered page is the simplest way
to obtain a bitmap that the existing CV/VLM pipeline can consume.

This module is independent: it does not import from `extract.py` and is not
imported by it. Callers decide which path to use.

See `docs/PDF_VECTOR_EXTRACTION_SPEC.md` for the contract.
"""

from pathlib import Path

import fitz  # PyMuPDF
import numpy as np


def extract_page_figure_as_bitmap(
    pdf_path: Path,
    page_idx: int,
    dpi: int = 600,
    crop_bbox: tuple[float, float, float, float] | None = None,
) -> dict:
    """Rasterize a PDF page (or sub-rectangle) to a high-resolution RGB bitmap.

    Used to recover figures stored as PDF vector primitives rather than
    embedded XObject images.

    Args:
        pdf_path: Path to the PDF file.
        page_idx: Zero-based page index.
        dpi: Render resolution. 600 dpi is print quality; 300 is the safe
             default for memory-constrained callers.
        crop_bbox: Optional clip rectangle (x0, y0, x1, y1) in page-space
                   points. When omitted the whole page is rasterized.

    Returns:
        Dict with keys: page_idx, dpi, width, height,
        image (np.ndarray (H, W, 3) uint8 RGB), source ("pdf_rasterize").

    Raises:
        FileNotFoundError: pdf_path does not exist.
        ValueError: page_idx out of range, dpi non-positive, or crop_bbox
                    malformed.

    Test scenario:
        >>> r = extract_page_figure_as_bitmap(
        ...     Path("tests/fixtures/ph01/gxae11701.pdf"), page_idx=6, dpi=600)
        >>> assert r["width"] >= 1343 and r["height"] >= 691
        >>> assert r["image"].shape[2] == 3
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if dpi <= 0:
        raise ValueError(f"dpi must be positive, got {dpi}")

    doc = fitz.open(str(pdf_path))
    try:
        if not (0 <= page_idx < doc.page_count):
            raise ValueError(f"page_idx {page_idx} out of range [0, {doc.page_count})")
        page = doc[page_idx]

        clip = None
        if crop_bbox is not None:
            x0, y0, x1, y1 = crop_bbox
            if x0 >= x1 or y0 >= y1:
                raise ValueError(f"invalid crop_bbox: {crop_bbox}")
            clip = fitz.Rect(x0, y0, x1, y1)

        # PDF default is 72 dpi; scale matrix maps user-space units to pixels.
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False, clip=clip)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, 3
        ).copy()  # detach from PyMuPDF's internal buffer

        return {
            "page_idx": page_idx,
            "dpi": dpi,
            "width": int(pix.width),
            "height": int(pix.height),
            "image": img,
            "source": "pdf_rasterize",
        }
    finally:
        doc.close()
