"""Batch session initialization helpers."""

from __future__ import annotations

from pathlib import Path

from geoseg.session_state import SessionState, create_session, load_session


def init_session(images_dir: Path, output_dir: Path) -> SessionState:
    """Create or load an existing session for the batch."""
    session_path = output_dir / "session.json"
    if session_path.exists():
        try:
            return load_session(session_path)
        except Exception:
            pass
    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    return create_session([str(p) for p in image_files])


__all__ = ["init_session"]
