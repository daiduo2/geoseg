"""Segmentation engine registry.

This module describes the engines the router can dispatch to. It keeps
engine metadata separate from routing policy and execution mechanics so new
engines do not require editing a large conditional router.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EngineSpec:
    """Static metadata for a segmentation engine."""

    name: str
    requires_reps: bool = False
    fallback_engine: str | None = "v4_kmeans"
    is_post_processor: bool = False


ENGINE_REGISTRY: dict[str, EngineSpec] = {
    "grayscale_agglomerative": EngineSpec("grayscale_agglomerative"),
    "v4_kmeans_colorbar": EngineSpec("v4_kmeans_colorbar"),
    "v4_kmeans_pastel": EngineSpec("v4_kmeans_pastel"),
    "v4_kmeans": EngineSpec("v4_kmeans", fallback_engine=None),
    "kmeans_full": EngineSpec("kmeans_full", requires_reps=True),
    "edge_guided": EngineSpec("edge_guided", requires_reps=True),
    "edge_grow": EngineSpec("edge_grow", requires_reps=True),
    "ensemble": EngineSpec("ensemble", requires_reps=True),
    "tubular": EngineSpec("tubular", fallback_engine=None),
    "horizon_refinement": EngineSpec(
        "horizon_refinement",
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


__all__ = ["ENGINE_REGISTRY", "EngineSpec", "get_engine_spec", "list_engines"]
