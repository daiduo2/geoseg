"""Persistent session state for CLI human-in-the-loop workflow.

Tracks the full lifecycle of each figure in a workset, enabling:
- Batch processing with集中 review
- User backtracking to upstream stages via natural language
- Resume across agent restarts

Usage:
    state = create_session(["fig1.png", "fig2.png"])
    state = update_figure(state, "fig1", status=FigureStatus.CLASSIFIED, ...)
    state = backtrack(state, "fig1", to_stage="classify")  # immutable
    save_session(state, "runs/session_001.json")
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FigureStatus(str, Enum):
    """Lifecycle stages of a single figure."""

    PENDING = "pending"
    CLASSIFIED = "classified"
    SEGMENTED = "segmented"
    REVIEWED = "reviewed"
    EXPORTED = "exported"
    SKIPPED = "skipped"
    ERROR = "error"


class BacktrackStage(str, Enum):
    """Stages a user can backtrack to during review."""

    CLASSIFY = "classify"  # Re-run figure classification
    PANEL = "panel"  # Re-run panel detection / selection
    SEGMENT = "segment"  # Re-run segmentation only


# ---------------------------------------------------------------------------
# Sub-models (immutable via frozen=False, but we rebuild parent)
# ---------------------------------------------------------------------------

class ClassificationRecord(BaseModel):
    """Result of figure classification (M1)."""

    figure_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""
    user_override: str | None = None

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class PanelSelection(BaseModel):
    """Detected panels + chosen target."""

    detected: list[dict[str, Any]] = Field(default_factory=list)
    target_panel_id: int = 0
    user_override: int | None = None


class SegmentationAttempt(BaseModel):
    """One engine attempt within a segmentation session."""

    engine: str
    n_layers: int
    quality_score: float = Field(ge=0.0, le=1.0)
    notes: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SegmentationRecord(BaseModel):
    """Best segmentation result + full attempt history."""

    result_dir: str
    engine: str
    n_layers: int
    quality_score: float = Field(ge=0.0, le=1.0)
    overlay_path: str
    labels_path: str
    attempts: list[SegmentationAttempt] = Field(default_factory=list)
    user_feedback: list[str] = Field(default_factory=list)


class ExportRecord(BaseModel):
    """SPECFEM export artifacts."""

    tomo_xyz: str | None = None
    parfile_snippet: str | None = None


class FigureEntry(BaseModel):
    """Mutable state of one figure in the workset.

    We keep this unfrozen so BaseModel.copy() works cleanly; immutability is
    enforced at the SessionState level (every update returns a new SessionState).
    """

    figure_id: str
    source_path: str
    status: FigureStatus = FigureStatus.PENDING
    classification: ClassificationRecord | None = None
    panels: PanelSelection | None = None
    segmentation: SegmentationRecord | None = None
    export: ExportRecord | None = None
    skip_reason: str | None = None
    error_message: str | None = None


class SessionState(BaseModel):
    """Root state object — persisted as JSON."""

    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    workset: list[FigureEntry] = Field(default_factory=list)

    def model_dump_json(self, **kwargs: Any) -> str:
        kwargs.setdefault("indent", 2)
        return super().model_dump_json(**kwargs)


# ---------------------------------------------------------------------------
# CRUD operations (immutable — always return new SessionState)
# ---------------------------------------------------------------------------

def create_session(figure_paths: list[str]) -> SessionState:
    """Create a fresh session from a list of image paths."""
    entries = [
        FigureEntry(
            figure_id=_figure_id_from_path(p),
            source_path=p,
        )
        for p in figure_paths
    ]
    return SessionState(workset=entries)


def update_figure(
    state: SessionState,
    figure_id: str,
    **kwargs: Any,
) -> SessionState:
    """Return a NEW session with one figure entry updated."""
    new_workset: list[FigureEntry] = []
    found = False
    for entry in state.workset:
        if entry.figure_id == figure_id:
            found = True
            updated_data = entry.model_dump()
            updated_data.update(kwargs)
            new_workset.append(FigureEntry(**updated_data))
        else:
            new_workset.append(entry)
    if not found:
        raise ValueError(f"Figure '{figure_id}' not found in session workset")
    return state.model_copy(update={"workset": new_workset})


def backtrack(
    state: SessionState,
    figure_id: str,
    to_stage: BacktrackStage,
) -> SessionState:
    """Backtrack a figure to an earlier stage, clearing downstream data.

    Backtrack targets:
        classify -> clear classification, panels, segmentation, export
        panel    -> clear panels, segmentation, export
        segment  -> clear segmentation, export
    """
    entry = _find_entry(state, figure_id)
    data = entry.model_dump()
    data.pop("figure_id", None)  # avoid duplicate kwarg in update_figure

    if to_stage == BacktrackStage.CLASSIFY:
        data["status"] = FigureStatus.PENDING
        data["classification"] = None
        data["panels"] = None
        data["segmentation"] = None
        data["export"] = None
    elif to_stage == BacktrackStage.PANEL:
        if data.get("classification") is None:
            raise ValueError("Cannot backtrack to 'panel' before classification exists")
        data["status"] = FigureStatus.CLASSIFIED
        data["panels"] = None
        data["segmentation"] = None
        data["export"] = None
    elif to_stage == BacktrackStage.SEGMENT:
        if data.get("panels") is None:
            raise ValueError("Cannot backtrack to 'segment' before panel selection exists")
        data["status"] = FigureStatus.CLASSIFIED  # or keep CLASSIFIED, re-detect panels
        data["segmentation"] = None
        data["export"] = None
    else:
        raise ValueError(f"Unknown backtrack stage: {to_stage}")

    return update_figure(state, figure_id, **data)


def get_summary(state: SessionState) -> dict[str, Any]:
    """Return a human-readable summary of session progress."""
    total = len(state.workset)
    by_status: dict[str, int] = {}
    for entry in state.workset:
        by_status[entry.status.value] = by_status.get(entry.status.value, 0) + 1

    return {
        "session_id": state.session_id,
        "total_figures": total,
        "by_status": by_status,
        "pending": by_status.get(FigureStatus.PENDING.value, 0),
        "classified": by_status.get(FigureStatus.CLASSIFIED.value, 0),
        "segmented": by_status.get(FigureStatus.SEGMENTED.value, 0),
        "reviewed": by_status.get(FigureStatus.REVIEWED.value, 0),
        "exported": by_status.get(FigureStatus.EXPORTED.value, 0),
        "skipped": by_status.get(FigureStatus.SKIPPED.value, 0),
        "errors": by_status.get(FigureStatus.ERROR.value, 0),
    }


def list_ready_for_review(state: SessionState) -> list[FigureEntry]:
    """Figures that have been segmented and await user review."""
    return [e for e in state.workset if e.status == FigureStatus.SEGMENTED]


def list_ready_for_export(state: SessionState) -> list[FigureEntry]:
    """Figures that have been reviewed and await export."""
    return [e for e in state.workset if e.status == FigureStatus.REVIEWED]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_session(state: SessionState, path: str | Path) -> Path:
    """Serialize session state to JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(state.model_dump_json(), encoding="utf-8")
    return p


def load_session(path: str | Path) -> SessionState:
    """Deserialize session state from JSON."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Session file not found: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    return SessionState.model_validate(raw)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _figure_id_from_path(path: str) -> str:
    """Derive a short figure ID from file path."""
    stem = Path(path).stem
    # Sanitize: replace spaces/special chars
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    return safe


def _find_entry(state: SessionState, figure_id: str) -> FigureEntry:
    for entry in state.workset:
        if entry.figure_id == figure_id:
            return entry
    raise ValueError(f"Figure '{figure_id}' not found in session workset")


__all__ = [
    "FigureStatus",
    "BacktrackStage",
    "ClassificationRecord",
    "PanelSelection",
    "SegmentationAttempt",
    "SegmentationRecord",
    "ExportRecord",
    "FigureEntry",
    "SessionState",
    "create_session",
    "update_figure",
    "backtrack",
    "get_summary",
    "list_ready_for_review",
    "list_ready_for_export",
    "save_session",
    "load_session",
]