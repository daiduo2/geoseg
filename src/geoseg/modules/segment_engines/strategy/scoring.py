"""Strategy memory scoring and analysis."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from geoseg.modules.segment_engines.strategy.records import (
    DEFAULT_MEMORY_PATH,
    extract_features,
    read_records,
)


def feature_distance(a: dict, b: dict) -> float:
    """Compute weighted distance between two feature dicts."""
    keys = ["saturation", "edge_density", "n_color_groups", "aspect_ratio"]
    weights = {
        "saturation": 2.0,
        "edge_density": 1.0,
        "n_color_groups": 1.0,
        "aspect_ratio": 0.5,
    }
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
    """Find similar successful historical attempts."""
    if memory_path is None:
        memory_path = DEFAULT_MEMORY_PATH

    target_features = extract_features(panel_rgb)
    records = []
    for rec in read_records(memory_path):
        if rec.get("outcome", "") != "success":
            continue
        feat = rec.get("image_features", {})
        dist = feature_distance(target_features, feat)
        records.append((dist, rec))

    records.sort(key=lambda x: x[0])
    return [r[1] for r in records[:top_k]]


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


def _engine_success_rates(records: list[dict]) -> dict[str, float]:
    engine_outcomes = defaultdict(lambda: {"success": 0, "total": 0})
    for rec in records:
        engine = rec.get("engine", "unknown")
        outcome = rec.get("outcome", "")
        engine_outcomes[engine]["total"] += 1
        if outcome == "success":
            engine_outcomes[engine]["success"] += 1

    return {
        e: round(v["success"] / max(v["total"], 1), 3)
        for e, v in engine_outcomes.items()
        if v["total"] >= 2
    }


def _extract_patterns(records: list[dict]) -> list[dict]:
    pattern_groups = defaultdict(list)
    for rec in records:
        feat = rec.get("image_features", {})
        key = (
            _bin_sat(feat.get("saturation", 0)),
            _bin_edge(feat.get("edge_density", 0)),
        )
        pattern_groups[key].append(rec)

    patterns = []
    for (sat_bin, edge_bin), grouped_records in pattern_groups.items():
        if len(grouped_records) < 3:
            continue

        best_engine = None
        best_rate = 0.0
        engine_counts = Counter(r.get("engine", "unknown") for r in grouped_records)
        engine_success = Counter(
            r.get("engine", "unknown")
            for r in grouped_records
            if r.get("outcome") == "success"
        )
        for engine, count in engine_counts.most_common():
            rate = engine_success.get(engine, 0) / count
            if rate > best_rate and count >= 2:
                best_rate = rate
                best_engine = engine

        if best_engine:
            alignments = [
                r.get("scores", {}).get("boundary_alignment", 0.0)
                for r in grouped_records
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
                "sample_size": len(grouped_records),
                "confidence": round(min(1.0, len(grouped_records) / 20.0), 3),
            })

    patterns.sort(key=lambda p: p["success_rate"], reverse=True)
    return patterns


def analyze_batch(
    memory_path: Path | str | None = None,
    min_samples: int = 5,
) -> dict:
    """Analyze records in memory to extract successful strategy patterns."""
    all_records = read_records(memory_path)

    if not all_records:
        return {
            "patterns": [],
            "engine_success_rates": {},
            "summary": {"total_records": 0, "message": "No memory yet"},
        }

    if len(all_records) < min_samples:
        return {
            "patterns": [],
            "engine_success_rates": {},
            "summary": {
                "total_records": len(all_records),
                "message": f"Need at least {min_samples} records for pattern extraction",
            },
        }

    engine_rates = _engine_success_rates(all_records)
    patterns = _extract_patterns(all_records)

    return {
        "patterns": patterns,
        "engine_success_rates": engine_rates,
        "summary": {
            "total_records": len(all_records),
            "n_patterns_found": len(patterns),
            "engines_evaluated": list(engine_rates.keys()),
        },
    }


__all__ = ["analyze_batch", "feature_distance", "query_similar"]
