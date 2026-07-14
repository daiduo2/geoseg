"""FastAPI HTTP server entry point."""

from __future__ import annotations

from geoseg.api.app import app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


__all__ = ["app"]
