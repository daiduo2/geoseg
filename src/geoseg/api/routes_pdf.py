"""PDF API routes."""

from __future__ import annotations

from fastapi import APIRouter, File, UploadFile

from geoseg.api.schemas import PdfImportResponse, PdfStatusResponse

router = APIRouter(prefix="/api/pdf")


@router.post("/import", response_model=PdfImportResponse)
async def import_pdf(pdf: UploadFile = File(...)) -> PdfImportResponse:
    """Upload a PDF and start MinerU extraction."""
    raise NotImplementedError("import_pdf stub - implement in Week 5")


@router.get("/status/{job_id}", response_model=PdfStatusResponse)
async def pdf_status(job_id: str) -> PdfStatusResponse:
    """Poll extraction status and get figure list."""
    raise NotImplementedError("pdf_status stub - implement in Week 5")


__all__ = ["import_pdf", "pdf_status", "router"]
