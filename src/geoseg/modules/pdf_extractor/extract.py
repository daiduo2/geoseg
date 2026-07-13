"""PDF extractor: extract embedded images (XObject) and text blocks per page.

Uses PyMuPDF (fitz) to avoid rasterization. Outputs structured data for downstream
VLM and CV modules.
"""

import io
from pathlib import Path
from typing import List, Dict, Any

import fitz  # PyMuPDF
from PIL import Image


def extract_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """Extract embedded images and text blocks from each page of a PDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of page dicts, one per page:
        {
            "page_idx": int,
            "page_width": float,
            "page_height": float,
            "images": [
                {
                    "xref": int,
                    "width": int,
                    "height": int,
                    "bbox_on_page": [x0, y0, x1, y1],
                    "ext": str,  # "png", "jpeg", etc.
                    "data": bytes,  # raw image bytes
                }
            ],
            "text_blocks": [
                {
                    "bbox": [x0, y0, x1, y1],
                    "text": str,
                }
            ],
        }
    """
    doc = fitz.open(str(pdf_path))
    results = []

    for page_idx in range(doc.page_count):
        page = doc[page_idx]
        rect = page.rect

        page_data = {
            "page_idx": page_idx,
            "page_width": rect.width,
            "page_height": rect.height,
            "images": [],
            "text_blocks": [],
        }

        # Extract embedded images (XObject)
        img_list = page.get_images(full=True)
        for img_index, img in enumerate(img_list):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)

            # Convert pixmap to PIL Image for robust colorspace handling
            if pix.n == 1:
                mode = "L"
            elif pix.n == 2:
                mode = "LA"
            elif pix.n == 3:
                mode = "RGB"
            elif pix.n == 4:
                mode = "RGBA"
            else:
                # CMYK or other: convert via RGB
                pix = fitz.Pixmap(fitz.csRGB, pix)
                mode = "RGB"

            img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            data = buf.getvalue()

            # Get image bbox on page
            img_rects = page.get_image_rects(xref)
            if img_rects:
                r = img_rects[0]
                bbox = [r.x0, r.y0, r.x1, r.y1]
            else:
                bbox = [0.0, 0.0, float(pix.width), float(pix.height)]

            page_data["images"].append({
                "xref": xref,
                "width": pix.width,
                "height": pix.height,
                "bbox_on_page": bbox,
                "ext": "png",
                "data": data,
            })
            pix = None

        # Extract text blocks with bounding boxes
        blocks = page.get_text("blocks")
        for b in blocks:
            # b = (x0, y0, x1, y1, text, block_no, block_type)
            x0, y0, x1, y1, text, *_ = b
            text = text.strip()
            if text:
                page_data["text_blocks"].append({
                    "bbox": [x0, y0, x1, y1],
                    "text": text,
                })

        results.append(page_data)

    doc.close()
    return results


def save_extracted_images(
    extracted: List[Dict[str, Any]],
    out_dir: Path,
    prefix: str = "page",
) -> List[Path]:
    """Save extracted image bytes to files.

    Args:
        extracted: Output from extract_pdf().
        out_dir: Directory to save images.
        prefix: Filename prefix.

    Returns:
        List of saved image paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for page in extracted:
        page_idx = page["page_idx"]
        for i, img in enumerate(page["images"]):
            fname = f"{prefix}_{page_idx:03d}_img_{i}.{img['ext']}"
            fpath = out_dir / fname
            fpath.write_bytes(img["data"])
            saved.append(fpath)

    return saved


def rasterize_page(
    pdf_path: Path,
    page_idx: int,
    dpi: int = 300,
    crop_bbox: tuple[float, float, float, float] | None = None,
) -> dict:
    """Rasterize a PDF page (or region) to a high-resolution bitmap.

    This complements XObject extraction for figures that are drawn as
    vector graphics on the page rather than embedded as images.

    Args:
        pdf_path: Path to the PDF file.
        page_idx: Page number (0-based).
        dpi: Rendering resolution. 300 for screen, 600 for print-quality.
        crop_bbox: Optional page-coordinate crop (x0, y0, x1, y1) in PDF points.

    Returns:
        {
            "page_idx": int,
            "dpi": int,
            "width": int,
            "height": int,
            "image": np.ndarray,  # RGB, shape (H, W, 3)
            "source": "pdf_rasterize",
        }

    Raises:
        FileNotFoundError: pdf_path does not exist.
        ValueError: page_idx out of range.

    Test scenario:
        >>> result = rasterize_page(
        ...     Path("tests/fixtures/ph01/gxae11701.pdf"), page_idx=7, dpi=300
        ... )
        >>> assert result["width"] >= 1000
        >>> assert result["height"] >= 600
        >>> assert result["image"].ndim == 3
    """
    import numpy as np

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    if page_idx < 0 or page_idx >= doc.page_count:
        doc.close()
        raise ValueError(
            f"page_idx {page_idx} out of range (0-{doc.page_count - 1})"
        )

    page = doc[page_idx]
    mat = fitz.Matrix(dpi / 72, dpi / 72)

    if crop_bbox:
        x0, y0, x1, y1 = crop_bbox
        clip = fitz.Rect(x0, y0, x1, y1)
        pix = page.get_pixmap(matrix=mat, clip=clip)
    else:
        pix = page.get_pixmap(matrix=mat)

    # Convert pixmap to numpy RGB array
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )
    if pix.n == 4:
        img = img[:, :, :3]  # Drop alpha
    elif pix.n == 1:
        img = np.stack([img[:, :, 0]] * 3, axis=-1)  # Grayscale -> RGB

    doc.close()

    return {
        "page_idx": page_idx,
        "dpi": dpi,
        "width": pix.width,
        "height": pix.height,
        "image": img,
        "source": "pdf_rasterize",
    }
