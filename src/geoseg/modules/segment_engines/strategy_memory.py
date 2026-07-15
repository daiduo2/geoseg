"""Compatibility facade for strategy memory helpers."""

from geoseg.modules.segment_engines.strategy import (
    DEFAULT_MEMORY_PATH,
    analyze_batch,
    extract_features as _extract_features,
    feature_distance as _feature_distance,
    load_templates,
    query_similar,
    record_attempt,
    save_templates,
)
from geoseg.modules.segment_engines.strategy.records import ensure_dir as _ensure_dir

__all__ = [
    "DEFAULT_MEMORY_PATH",
    "_ensure_dir",
    "_extract_features",
    "_feature_distance",
    "analyze_batch",
    "load_templates",
    "query_similar",
    "record_attempt",
    "save_templates",
]
