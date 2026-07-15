"""Execution helpers for segmentation engines."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from geoseg.core.models import SegmentationResult
from geoseg.modules.segment_engines.registry import (
    EngineSpec,
    get_engine_spec,
    load_engine_callable,
)


def _normalize_result(raw: dict, engine_name: str, n_layers: int) -> SegmentationResult:
    """Normalize engine output to SegmentationResult."""
    meta = raw.get("meta", {})
    result: SegmentationResult = {
        "labels": raw["labels"],
        "overlay": raw.get("overlay"),
        "meta": {
            "engine": meta.get("engine", engine_name),
            "color_names": meta.get("color_names", []),
            "n_layers": n_layers,
            "quality_score": meta.get("quality_score"),
        },
    }
    if "seeds" in raw:
        result["seeds"] = raw["seeds"]
    return result


def _run_with_fallback(
    primary_fn: Callable[[], dict],
    fallback_fn: Callable[[], dict],
    engine_name: str,
    panel_rgb: np.ndarray,
    n_layers: int,
) -> SegmentationResult:
    try:
        return _normalize_result(primary_fn(), engine_name, n_layers)
    except Exception as exc:
        try:
            result = fallback_fn()
            result["meta"]["fallback_reason"] = str(exc)
            return _normalize_result(result, engine_name, n_layers)
        except Exception:
            return _normalize_result(
                {
                    "labels": np.zeros(panel_rgb.shape[:2], dtype=np.int32),
                    "seeds": [],
                    "overlay": panel_rgb.copy(),
                    "meta": {
                        "engine": engine_name,
                        "error": f"primary: {exc}; fallback also failed",
                    },
                },
                engine_name,
                n_layers,
            )


def _call_engine(
    spec: EngineSpec,
    panel_rgb: np.ndarray,
    reps: list[dict] | None,
    colorbar_rgb: np.ndarray | None,
    n_layers: int,
    n_color_zones: int,
) -> dict:
    segment = load_engine_callable(spec)

    if spec.adapter == "v4":
        return segment(
            panel_rgb,
            reps=reps,
            colorbar_rgb=colorbar_rgb,
            n_layers=n_layers,
            n_color_zones=n_color_zones,
        )
    if spec.adapter == "colorbar_guided":
        return segment(
            panel_rgb,
            colorbar_rgb,
            n_layers=n_layers,
            n_color_zones=n_color_zones,
        )
    if spec.adapter == "grayscale":
        return segment(panel_rgb, n_layers=n_layers, reps=reps)
    if spec.adapter == "reps_keyword":
        return segment(panel_rgb, reps=reps, n_layers=n_layers)
    if spec.adapter == "reps_positional":
        return segment(panel_rgb, reps, n_layers=n_layers)

    raise ValueError(f"Unsupported engine adapter: {spec.adapter}")


def run_engine(
    engine: str,
    panel_rgb: np.ndarray,
    reps: list[dict] | None,
    colorbar_rgb: np.ndarray | None,
    n_layers: int,
    n_color_zones: int = 0,
) -> SegmentationResult:
    """Run a registered engine by name."""

    def _v4_fallback() -> dict:
        fallback_spec = get_engine_spec("v4_kmeans")
        if fallback_spec is None:
            raise ValueError("Missing v4_kmeans fallback engine")
        return _call_engine(
            fallback_spec,
            panel_rgb,
            reps,
            colorbar_rgb,
            n_layers,
            n_color_zones,
        )

    spec = get_engine_spec(engine)
    if spec is None:
        return _normalize_result(_v4_fallback(), "v4_kmeans", n_layers)

    if spec.requires_reps and not reps:
        return _normalize_result(_v4_fallback(), "v4_kmeans", n_layers)

    if spec.adapter == "horizon_refinement":
        coarse = _normalize_result(_v4_fallback(), "v4_kmeans", n_layers)
        if (coarse["labels"] != 0).sum() == 0:
            return coarse
        return _run_with_fallback(
            lambda: load_engine_callable(spec)(
                panel_rgb,
                n_layers=n_layers,
                coarse_labels=coarse["labels"],
            ),
            _v4_fallback,
            engine,
            panel_rgb,
            n_layers,
        )

    if spec.fallback_engine is None:
        return _normalize_result(
            _call_engine(spec, panel_rgb, reps, colorbar_rgb, n_layers, n_color_zones),
            engine,
            n_layers,
        )

    return _run_with_fallback(
        lambda: _call_engine(spec, panel_rgb, reps, colorbar_rgb, n_layers, n_color_zones),
        _v4_fallback,
        engine,
        panel_rgb,
        n_layers,
    )


__all__ = ["_normalize_result", "run_engine"]
