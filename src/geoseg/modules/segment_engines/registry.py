"""Segmentation engine registry.

This module describes the engines the router can dispatch to. It keeps
engine metadata separate from routing policy and execution mechanics so new
engines do not require editing a large conditional router.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class EngineSpec:
    """Static metadata for a segmentation engine."""

    name: str
    callable_path: str
    adapter: str = "reps_positional"
    requires_reps: bool = False
    fallback_engine: str | None = "v4_kmeans"
    is_post_processor: bool = False


ENGINE_REGISTRY: dict[str, EngineSpec] = {
    "grayscale_agglomerative": EngineSpec(
        "grayscale_agglomerative",
        "geoseg.modules.segment_engines.grayscale:segment",
        adapter="grayscale",
    ),
    "v4_kmeans_colorbar": EngineSpec(
        "v4_kmeans_colorbar",
        "geoseg.modules.segment_engines.v4_kmeans:segment_colorbar_guided",
        adapter="colorbar_guided",
    ),
    "v4_kmeans_pastel": EngineSpec(
        "v4_kmeans_pastel",
        "geoseg.modules.segment_engines.v4_kmeans:segment_pastel_faded",
        adapter="colorbar_guided",
    ),
    "v4_kmeans": EngineSpec(
        "v4_kmeans",
        "geoseg.modules.segment_engines.v4_kmeans:segment",
        adapter="v4",
        fallback_engine=None,
    ),
    "kmeans_full": EngineSpec(
        "kmeans_full",
        "geoseg.modules.segment_engines.kmeans_full:segment",
        requires_reps=True,
    ),
    "edge_guided": EngineSpec(
        "edge_guided",
        "geoseg.modules.segment_engines.edge_guided:segment",
        requires_reps=True,
    ),
    "edge_grow": EngineSpec(
        "edge_grow",
        "geoseg.modules.segment_engines.edge_grow:segment",
        requires_reps=True,
    ),
    "ensemble": EngineSpec(
        "ensemble",
        "geoseg.modules.segment_engines.ensemble:segment",
        requires_reps=True,
    ),
    "tubular": EngineSpec(
        "tubular",
        "geoseg.modules.segment_engines.tubular_structure:segment",
        adapter="reps_keyword",
        fallback_engine=None,
    ),
    "horizon_refinement": EngineSpec(
        "horizon_refinement",
        "geoseg.modules.segment_engines.horizon_refinement:segment",
        adapter="horizon_refinement",
        fallback_engine="v4_kmeans",
        is_post_processor=True,
    ),
}


def get_engine_spec(name: str) -> EngineSpec | None:
    """Return metadata for an engine name."""
    return ENGINE_REGISTRY.get(name)


def list_engines() -> list[str]:
    """Return registered engine names in registry order."""
    return list(ENGINE_REGISTRY)


def load_engine_callable(spec: EngineSpec) -> Callable[..., Any]:
    """Load the callable registered for an engine spec."""
    module_name, _, attr_name = spec.callable_path.partition(":")
    if not module_name or not attr_name:
        raise ValueError(f"Invalid engine callable path: {spec.callable_path}")
    module = import_module(module_name)
    return getattr(module, attr_name)


__all__ = [
    "ENGINE_REGISTRY",
    "EngineSpec",
    "get_engine_spec",
    "list_engines",
    "load_engine_callable",
]
