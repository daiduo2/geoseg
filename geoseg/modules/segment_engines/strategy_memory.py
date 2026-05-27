"""Strategy memory for agent self-learning.

Records every segmentation attempt with image features, engine choice,
quantitative scores, and final outcome. Agent reads this before processing
new panels and writes to it after each attempt.

File format: newline-delimited JSON (jsonl) at runs/sandbox/strategy_memory.jsonl
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_MEMORY_PATH = Path("runs/sandbox/strategy_memory.jsonl")


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _extract_features(panel_rgb: np.ndarray) -> dict:
    """Extract lightweight image features for similarity matching."""
    from geoseg.modules.segment_engines._shared import saturation_ratio
    from skimage.filters import sobel

    h, w = panel_rgb.shape[:2]
    gray = panel_rgb.mean(axis=2)
    edges = sobel(gray)
    edge_dens = float((np.abs(edges) > 0.05).mean())
    sat = saturation_ratio(panel_rgb)

    # Color diversity: number of distinct color clusters
    pixels = panel_rgb.reshape(-1, 3)
    if len(pixels) > 5000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pixels), 5000, replace=False)
        sample = pixels[idx]
    else:
        sample = pixels

    # Simple online grouping for diversity estimate
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

    n_colors = len(groups)

    return {
        "h": h,
        "w": w,
        "aspect_ratio": round(max(h, w) / max(min(h, w), 1), 2),
        "saturation": round(sat, 4),
        "edge_density": round(edge_dens, 4),
        "n_color_groups": n_colors,
    }


def record_attempt(
    panel_rgb: np.ndarray,
    engine: str,
    params: dict,
    scores: dict,
    outcome: str,
    notes: str = "",
    memory_path: Path | str | None = None,
) -> Path:
    """Record a segmentation attempt to strategy memory.

    Args:
        panel_rgb: The panel image (for feature extraction).
        engine: Engine name used.
        params: Engine parameters (n_layers, etc.).
        scores: Quantitative scores from metrics.py.
        outcome: "success", "retry", "abandoned", "fallback".
        notes: Free-text notes about the attempt.
        memory_path: Where to append the record. Default: runs/sandbox/strategy_memory.jsonl.

    Returns:
        Path to the memory file.
    """
    if memory_path is None:
        memory_path = DEFAULT_MEMORY_PATH
    path = Path(memory_path)
    _ensure_dir(path)

    features = _extract_features(panel_rgb)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image_features": features,
        "engine": engine,
        "params": params,
        "scores": scores,
        "outcome": outcome,
        "notes": notes,
    }

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return path


def _feature_distance(a: dict, b: dict) -> float:
    """Compute weighted distance between two feature dicts."""
    keys = ["saturation", "edge_density", "n_color_groups", "aspect_ratio"]
    weights = {"saturation": 2.0, "edge_density": 1.0, "n_color_groups": 1.0, "aspect_ratio": 0.5}
    total = 0.0
    wsum = 0.0
    for k in keys:
        if k in a and k in b:
            diff = abs(a[k] - b[k])
            w = weights.get(k, 1.0)
            total += diff * w
            wsum += w
    return total / max(wsum, 1e-9)


def query_similar(
    panel_rgb: np.ndarray,
    top_k: int = 3,
    memory_path: Path | str | None = None,
) -> list[dict]:
    """Find similar historical attempts and their outcomes.

    Returns:
        List of historical records sorted by similarity (most similar first),
        filtered to only include successful attempts (outcome == "success").
    """
    if memory_path is None:
        memory_path = DEFAULT_MEMORY_PATH
    path = Path(memory_path)

    if not path.exists():
        return []

    target_features = _extract_features(panel_rgb)

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Filter: only consider successful attempts (outcome judged by VLM)
            if rec.get("outcome", "") != "success":
                continue
            feat = rec.get("image_features", {})
            dist = _feature_distance(target_features, feat)
            records.append((dist, rec))

    records.sort(key=lambda x: x[0])
    return [r[1] for r in records[:top_k]]


def analyze_batch(
    memory_path: Path | str | None = None,
    min_samples: int = 5,
) -> dict:
    """Analyze all records in memory to extract successful strategy patterns.

    This is the meta-learning layer. Call after each batch completes.

    Returns:
        Dict with:
            - patterns: list of {feature_pattern, recommended_engines, confidence}
            - engine_success_rates: {engine: success_rate}
            - summary: overall statistics
    """
    if memory_path is None:
        memory_path = DEFAULT_MEMORY_PATH
    path = Path(memory_path)

    if not path.exists():
        return {
            "patterns": [],
            "engine_success_rates": {},
            "summary": {"total_records": 0, "message": "No memory yet"},
        }

    all_records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                all_records.append(rec)
            except json.JSONDecodeError:
                continue

    if len(all_records) < min_samples:
        return {
            "patterns": [],
            "engine_success_rates": {},
            "summary": {
                "total_records": len(all_records),
                "message": f"Need at least {min_samples} records for pattern extraction",
            },
        }

    # Engine success rates
    engine_outcomes = defaultdict(lambda: {"success": 0, "total": 0})
    for rec in all_records:
        engine = rec.get("engine", "unknown")
        outcome = rec.get("outcome", "")
        engine_outcomes[engine]["total"] += 1
        if outcome == "success":
            engine_outcomes[engine]["success"] += 1

    engine_rates = {
        e: round(v["success"] / max(v["total"], 1), 3)
        for e, v in engine_outcomes.items()
        if v["total"] >= 2  # Require at least 2 samples
    }

    # Feature → engine success patterns
    # Discretize features into bins
    def _bin_sat(sat: float) -> str:
        if sat < 0.1:
            return "pastel"
        if sat < 0.5:
            return "mixed"
        return "vivid"

    def _bin_edge(ed: float) -> str:
        if ed < 0.01:
            return "low"
        if ed < 0.03:
            return "medium"
        return "high"

    pattern_groups = defaultdict(list)
    for rec in all_records:
        feat = rec.get("image_features", {})
        key = (_bin_sat(feat.get("saturation", 0)), _bin_edge(feat.get("edge_density", 0)))
        pattern_groups[key].append(rec)

    patterns = []
    for (sat_bin, edge_bin), records in pattern_groups.items():
        if len(records) < 3:
            continue
        # Find best engine for this pattern
        best_engine = None
        best_rate = 0.0
        engine_counts = Counter(r.get("engine", "unknown") for r in records)
        engine_success = Counter(
            r.get("engine", "unknown")
            for r in records
            if r.get("outcome") == "success"
        )
        for engine, count in engine_counts.most_common():
            rate = engine_success.get(engine, 0) / count
            if rate > best_rate and count >= 2:
                best_rate = rate
                best_engine = engine

        if best_engine:
            # Use boundary_alignment as objective reference (not a "score")
            alignments = [
                r.get("scores", {}).get("boundary_alignment", 0.0)
                for r in records
                if r.get("engine") == best_engine
            ]
            avg_alignment = float(np.mean(alignments)) if alignments else 0.0

            patterns.append({
                "feature_pattern": {
                    "saturation": sat_bin,
                    "edge_density": edge_bin,
                },
                "recommended_engine": best_engine,
                "success_rate": round(best_rate, 3),
                "avg_boundary_alignment": round(avg_alignment, 3),
                "sample_size": len(records),
                "confidence": round(min(1.0, len(records) / 20.0), 3),  # More samples = higher confidence
            })

    patterns.sort(key=lambda p: p["success_rate"], reverse=True)

    return {
        "patterns": patterns,
        "engine_success_rates": engine_rates,
        "summary": {
            "total_records": len(all_records),
            "n_patterns_found": len(patterns),
            "engines_evaluated": list(engine_rates.keys()),
        },
    }


def save_templates(
    analysis: dict,
    output_path: Path | str | None = None,
) -> Path:
    """Save extracted patterns as strategy templates for agent reference.

    Call after analyze_batch() to persist templates.
    """
    if output_path is None:
        output_path = Path("runs/sandbox/strategy_templates.json")
    path = Path(output_path)
    _ensure_dir(path)

    path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2))
    return path


def load_templates(
    path: Path | str | None = None,
) -> dict:
    """Load strategy templates for agent pre-flight reference."""
    if path is None:
        path = Path("runs/sandbox/strategy_templates.json")
    p = Path(path)
    if not p.exists():
        return {"patterns": [], "engine_success_rates": {}, "summary": {}}
    return json.loads(p.read_text(encoding="utf-8"))
