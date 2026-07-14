"""Execution helpers for segmentation engines."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from geoseg.core.models import SegmentationResult
from geoseg.modules.segment_engines.registry import get_engine_spec


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
        from geoseg.modules.segment_engines.v4_kmeans import segment

        return segment(
            panel_rgb,
            reps=reps,
            colorbar_rgb=colorbar_rgb,
            n_layers=n_layers,
            n_color_zones=n_color_zones,
        )

    spec = get_engine_spec(engine)

    if spec and spec.requires_reps and not reps:
        return _normalize_result(_v4_fallback(), "v4_kmeans", n_layers)

    if engine == "grayscale_agglomerative":
        from geoseg.modules.segment_engines.grayscale import segment

        return _run_with_fallback(
            lambda: segment(panel_rgb, n_layers=n_layers, reps=reps),
            _v4_fallback,
            engine,
            panel_rgb,
            n_layers,
        )

    if engine in ("v4_kmeans_colorbar", "v4_kmeans_pastel"):
        from geoseg.modules.segment_engines.v4_kmeans import (
            segment_colorbar_guided,
            segment_pastel_faded,
        )

        if engine == "v4_kmeans_colorbar":
            return _run_with_fallback(
                lambda: segment_colorbar_guided(
                    panel_rgb,
                    colorbar_rgb,
                    n_layers=n_layers,
                    n_color_zones=n_color_zones,
                ),
                _v4_fallback,
                engine,
                panel_rgb,
                n_layers,
            )
        return _run_with_fallback(
            lambda: segment_pastel_faded(
                panel_rgb,
                colorbar_rgb,
                n_layers=n_layers,
                n_color_zones=n_color_zones,
            ),
            _v4_fallback,
            engine,
            panel_rgb,
            n_layers,
        )

    if engine == "v4_kmeans":
        return _normalize_result(_v4_fallback(), "v4_kmeans", n_layers)

    if engine == "kmeans_full":
        from geoseg.modules.segment_engines.kmeans_full import segment

        return _run_with_fallback(
            lambda: segment(panel_rgb, reps, n_layers=n_layers),
            _v4_fallback,
            engine,
            panel_rgb,
            n_layers,
        )

    if engine == "edge_guided":
        from geoseg.modules.segment_engines.edge_guided import segment

        return _run_with_fallback(
            lambda: segment(panel_rgb, reps, n_layers=n_layers),
            _v4_fallback,
            engine,
            panel_rgb,
            n_layers,
        )

    if engine == "edge_grow":
        from geoseg.modules.segment_engines.edge_grow import segment

        return _run_with_fallback(
            lambda: segment(panel_rgb, reps, n_layers=n_layers),
            _v4_fallback,
            engine,
            panel_rgb,
            n_layers,
        )

    if engine == "ensemble":
        from geoseg.modules.segment_engines.ensemble import segment

        return _run_with_fallback(
            lambda: segment(panel_rgb, reps, n_layers=n_layers),
            _v4_fallback,
            engine,
            panel_rgb,
            n_layers,
        )

    if engine == "tubular":
        from geoseg.modules.segment_engines.tubular_structure import segment

        return _normalize_result(
            segment(panel_rgb, reps=reps, n_layers=n_layers),
            "tubular",
            n_layers,
        )

    if engine == "horizon_refinement":
        from geoseg.modules.segment_engines.horizon_refinement import segment as hr_segment

        coarse = _normalize_result(_v4_fallback(), "v4_kmeans", n_layers)
        if (coarse["labels"] != 0).sum() == 0:
            return coarse
        return _run_with_fallback(
            lambda: hr_segment(
                panel_rgb,
                n_layers=n_layers,
                coarse_labels=coarse["labels"],
            ),
            _v4_fallback,
            engine,
            panel_rgb,
            n_layers,
        )

    return _normalize_result(_v4_fallback(), "v4_kmeans", n_layers)


__all__ = ["_normalize_result", "run_engine"]
