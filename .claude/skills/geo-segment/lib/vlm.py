"""VLM prompt templates + JSON loader.

Design constraint (reaffirmed by user): ALL LLM vision work happens inside the
running Claude Code session.  There is NO external API client — no Moonshot
SDK, no Anthropic SDK, no REST call from Python.

Workflow:
    1. User drops the figure into the current Claude Code chat.
    2. Claude (this session) answers T1/T2/T3 using PROMPTS below.
    3. User (or a thin shell helper) saves the JSON reply to disk.
    4. geo-segment consumes the JSON via `--vlm-json <path>`.

Public API:
    PROMPTS: dict[str, str]    # task_id → prompt template
    load_vlm_json(path) -> dict
"""

from __future__ import annotations

import json
from pathlib import Path

PROMPTS = {
    "T1": (
        "How many sub-panels are in this figure (vertically stacked or "
        "horizontally arranged)? Return the precise bounding box of each "
        "panel. Output ONLY JSON matching this schema, no explanation, no "
        "markdown:\n"
        '{"image_size": {"width": int, "height": int}, '
        '"panels": [{"id": int, "x1": int, "y1": int, "x2": int, '
        '"y2": int, "description": str}]}\n'
        "Coordinates are absolute pixels, (x1, y1) at top-left and "
        "(x2, y2) at bottom-right. The image is {width} x {height}."
    ),
    "T2": (
        "In sub-panel #{panel_id}, identify all non-data overlay elements: "
        "numbers, text, circles/stars, white wedges, contour lines, "
        "arrows, scalebars, axis labels. Output ONLY JSON, no markdown:\n"
        '{"panel_id": int, '
        '"noise_elements": [{"kind": str, "x": int, "y": int, '
        '"size_px": int, "content": str}]}\n'
        "x, y is the element center; size_px is its approximate size."
    ),
    "T3": (
        "In sub-panel #{panel_id}, based on the colorbar, identify the "
        "main color zones (typically red, orange, yellow, green, blue). "
        "Ignore overlay noise (numbers, circles, lines). For each color, "
        "pick one representative point INSIDE that zone, far from boundaries.\n"
        "Output ONLY JSON, no markdown:\n"
        '{"panel_id": int, "zones": ['
        '{"color_name": str, "colorbar_value": int, '
        '"representative_point": {"x": int, "y": int}}]}'
    ),
}


def load_vlm_json(path: str | Path) -> dict:
    """Load a saved VLM-output JSON (the session mode's data source)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = ["PROMPTS", "load_vlm_json"]
