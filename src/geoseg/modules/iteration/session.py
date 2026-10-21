"""Agent-driven sandbox session for iterative segmentation.

The session fixes the infrastructure, not the decisions:

- crop the panel once, before any engine;
- run an engine attempt and immediately audit it;
- record every attempt, repair, and acceptance;
- save labels/overlay/audit per attempt.

The agent decides which engine, layer count, crop, merge, text removal, or
refinement to try next.  There is no hard-coded repair policy.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from geoseg.core.image_ops import create_overlay
from geoseg.modules.iteration.audit import AuditConfig, AuditReport, audit_labels
from geoseg.modules.iteration.preprocess import CropResult, crop_panel


Runner = Callable[[np.ndarray, str, int, dict | None], np.ndarray]


@dataclass
class AttemptRecord:
    attempt: int
    engine: str
    n_layers: int
    labels: np.ndarray
    audit: AuditReport
    note: str = ""
    params: dict = field(default_factory=dict)
    parent: int | None = None
    label_hash: str = ""
    artifacts: dict = field(default_factory=dict)


@dataclass
class IterationSession:
    image: np.ndarray
    output_dir: Path | None = None
    crop_bbox: tuple[int, int, int, int] | None = None
    auto_crop: bool = True
    audit_config: AuditConfig = field(default_factory=AuditConfig)
    runner: Runner | None = None
    save_artifacts: bool = True
    crop: CropResult = field(init=False)
    attempts: list[AttemptRecord] = field(default_factory=list, init=False)
    accepted_index: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        crop = crop_panel(self.image, self.crop_bbox, auto=self.auto_crop)
        self.crop = crop
        self.image = crop.image
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "crop.json").write_text(
                json.dumps(
                    {
                        "bbox": list(crop.bbox),
                        "offset": list(crop.offset),
                        "method": crop.method,
                        "saturation_fraction": crop.saturation_fraction,
                        "shape": list(crop.image.shape[:2]),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    @property
    def labels(self) -> np.ndarray:
        return self.best.labels

    @property
    def best(self) -> AttemptRecord:
        if not self.attempts:
            raise RuntimeError("no attempts recorded")
        if self.accepted_index is not None:
            return self.attempts[self.accepted_index]
        return max(self.attempts, key=lambda item: item.audit.score)

    def audit(self, labels: np.ndarray) -> AuditReport:
        return audit_labels(self.image, labels, config=self.audit_config)

    def run_attempt(
        self,
        engine: str,
        n_layers: int,
        *,
        params: dict | None = None,
        note: str = "",
        parent: int | None = None,
    ) -> AttemptRecord:
        params = dict(params or {})
        if self.runner is None:
            from geoseg.modules.segment_engines.runner import run_engine

            result = run_engine(
                engine,
                self.image,
                params.get("reps"),
                params.get("colorbar"),
                n_layers,
                n_color_zones=int(params.get("n_color_zones", 0)),
            )
            labels = np.asarray(result["labels"], dtype=np.int32)
        else:
            labels = np.asarray(
                self.runner(self.image, engine, n_layers, params), dtype=np.int32
            )
        return self.record_attempt(
            labels,
            engine=engine,
            n_layers=n_layers,
            params=params,
            note=note,
            parent=parent,
        )

    def record_attempt(
        self,
        labels: np.ndarray,
        *,
        engine: str = "manual",
        n_layers: int = 0,
        params: dict | None = None,
        note: str = "",
        parent: int | None = None,
    ) -> AttemptRecord:
        labels = np.asarray(labels, dtype=np.int32)
        if labels.shape != self.image.shape[:2]:
            raise ValueError(
                f"labels shape {labels.shape} does not match cropped image {self.image.shape[:2]}"
            )
        report = self.audit(labels)
        record = AttemptRecord(
            attempt=len(self.attempts) + 1,
            engine=engine,
            n_layers=int(n_layers),
            labels=labels,
            audit=report,
            note=note,
            params=dict(params or {}),
            parent=parent,
            label_hash=hashlib.sha1(labels.tobytes()).hexdigest()[:12],
        )
        self.attempts.append(record)
        if self.output_dir is not None and self.save_artifacts:
            record.artifacts = self._save_attempt(record)
            self._save_summary()
        return record

    def accept(self, attempt: int | None = None) -> AttemptRecord:
        index = len(self.attempts) - 1 if attempt is None else int(attempt) - 1
        if not (0 <= index < len(self.attempts)):
            raise IndexError("attempt index out of range")
        self.accepted_index = index
        if self.output_dir is not None:
            self._save_summary()
        return self.attempts[index]

    def _save_attempt(self, record: AttemptRecord) -> dict:
        assert self.output_dir is not None
        attempt_dir = self.output_dir / "attempts" / f"{record.attempt:02d}_{record.engine}_n{record.n_layers}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(attempt_dir / "labels.npz", labels=record.labels)
        overlay = create_overlay(self.image, record.labels, None)
        from PIL import Image

        Image.fromarray(overlay).save(attempt_dir / "overlay.jpg", quality=92)
        (attempt_dir / "audit.json").write_text(
            json.dumps(record.audit.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return {
            "labels": str(attempt_dir / "labels.npz"),
            "overlay": str(attempt_dir / "overlay.jpg"),
            "audit": str(attempt_dir / "audit.json"),
        }

    def _save_summary(self) -> None:
        if self.output_dir is None:
            return
        payload = {
            "crop": {
                "bbox": list(self.crop.bbox),
                "offset": list(self.crop.offset),
                "method": self.crop.method,
                "shape": list(self.crop.image.shape[:2]),
            },
            "accepted_attempt": None
            if self.accepted_index is None
            else self.accepted_index + 1,
            "attempts": [
                {
                    "attempt": record.attempt,
                    "engine": record.engine,
                    "n_layers": record.n_layers,
                    "params": record.params,
                    "note": record.note,
                    "parent": record.parent,
                    "label_hash": record.label_hash,
                    "n_labels": record.audit.n_labels,
                    "score": round(record.audit.score, 3),
                    "accepted": record.audit.accepted,
                    "suggested_action": record.audit.suggested_action,
                    "under_segmented": record.audit.under_segmented_labels,
                    "text_labels": record.audit.text_labels,
                    "fragmented": record.audit.fragmented_labels,
                    "text_pixel_fraction": round(record.audit.text_pixel_fraction, 4),
                    "artifacts": record.artifacts,
                }
                for record in self.attempts
            ],
        }
        (self.output_dir / "session.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def save_best(self) -> None:
        if self.output_dir is None:
            return
        best = self.best
        np.savez_compressed(self.output_dir / "labels.npz", labels=best.labels)
        overlay = create_overlay(self.image, best.labels, None)
        from PIL import Image

        Image.fromarray(overlay).save(self.output_dir / "overlay.jpg", quality=92)
        self._save_summary()


def compact_labels(labels: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    """Renumber foreground labels to 1..N while preserving 0 background."""
    labels = np.asarray(labels, dtype=np.int32)
    present = sorted(int(value) for value in np.unique(labels) if value != 0)
    mapping = {old: new for new, old in enumerate(present, start=1)}
    out = np.zeros_like(labels, dtype=np.int32)
    for old, new in mapping.items():
        out[labels == old] = new
    return out, mapping


__all__ = ["AttemptRecord", "IterationSession", "compact_labels"]
