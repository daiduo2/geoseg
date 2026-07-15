"""Strategy memory records and feature extraction."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DEFAULT_MEMORY_PATH = Path("runs/sandbox/strategy_memory.jsonl")


def ensure_dir(path: Path) -> None:
    """Ensure a file path's parent directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def extract_features(panel_rgb: np.ndarray) -> dict:
    """Extract lightweight image features for similarity matching."""
    from geoseg.modules.segment_engines.internal.color import saturation_ratio
    from skimage.filters import sobel

    h, w = panel_rgb.shape[:2]
    gray = panel_rgb.mean(axis=2)
    edges = sobel(gray)
    edge_dens = float((np.abs(edges) > 0.05).mean())
    sat = saturation_ratio(panel_rgb)

    pixels = panel_rgb.reshape(-1, 3)
    if len(pixels) > 5000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pixels), 5000, replace=False)
        sample = pixels[idx]
    else:
        sample = pixels

    groups = []
    tol_sq = 60 * 60
    for px in sample.astype(np.float32):
        matched = False
        for i, (mean, count) in enumerate(groups):
            diff = px - mean
            if np.dot(diff, diff) <= tol_sq:
                groups[i] = ((mean * count + px) / (count + 1), count + 1)
                matched = True
                break
        if not matched:
            groups.append((px.copy(), 1))

    return {
        "h": h,
        "w": w,
        "aspect_ratio": round(max(h, w) / max(min(h, w), 1), 2),
        "saturation": round(sat, 4),
        "edge_density": round(edge_dens, 4),
        "n_color_groups": len(groups),
    }


def read_records(memory_path: Path | str | None = None) -> list[dict]:
    """Read valid JSONL strategy memory records."""
    if memory_path is None:
        memory_path = DEFAULT_MEMORY_PATH
    path = Path(memory_path)

    if not path.exists():
        return []

    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def record_attempt(
    panel_rgb: np.ndarray,
    engine: str,
    params: dict,
    scores: dict,
    outcome: str,
    notes: str = "",
    memory_path: Path | str | None = None,
) -> Path:
    """Record a segmentation attempt to strategy memory."""
    if memory_path is None:
        memory_path = DEFAULT_MEMORY_PATH
    path = Path(memory_path)
    ensure_dir(path)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image_features": extract_features(panel_rgb),
        "engine": engine,
        "params": params,
        "scores": scores,
        "outcome": outcome,
        "notes": notes,
    }

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return path


__all__ = [
    "DEFAULT_MEMORY_PATH",
    "ensure_dir",
    "extract_features",
    "read_records",
    "record_attempt",
]
