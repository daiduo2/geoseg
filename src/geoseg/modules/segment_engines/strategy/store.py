"""Strategy template persistence."""

from __future__ import annotations

import json
from pathlib import Path

from geoseg.modules.segment_engines.strategy.records import ensure_dir

DEFAULT_TEMPLATE_PATH = Path("runs/sandbox/strategy_templates.json")


def save_templates(
    analysis: dict,
    output_path: Path | str | None = None,
) -> Path:
    """Save extracted patterns as strategy templates for agent reference."""
    if output_path is None:
        output_path = DEFAULT_TEMPLATE_PATH
    path = Path(output_path)
    ensure_dir(path)

    path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2))
    return path


def load_templates(
    path: Path | str | None = None,
) -> dict:
    """Load strategy templates for agent pre-flight reference."""
    if path is None:
        path = DEFAULT_TEMPLATE_PATH
    p = Path(path)
    if not p.exists():
        return {"patterns": [], "engine_success_rates": {}, "summary": {}}
    return json.loads(p.read_text(encoding="utf-8"))


__all__ = ["DEFAULT_TEMPLATE_PATH", "load_templates", "save_templates"]
