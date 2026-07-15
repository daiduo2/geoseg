"""FastAPI app assembly for geoseg."""

from __future__ import annotations

from fastapi import FastAPI

from geoseg.api.routes_agent import router as agent_router
from geoseg.api.routes_export import router as export_router
from geoseg.api.routes_manual import router as manual_router
from geoseg.api.routes_pdf import router as pdf_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="geoseg", version="2.0.0")
    app.include_router(pdf_router)
    app.include_router(agent_router)
    app.include_router(manual_router)
    app.include_router(export_router)
    return app


app = create_app()


__all__ = ["app", "create_app"]
