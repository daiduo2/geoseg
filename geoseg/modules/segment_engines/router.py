"""Algorithm router: select segmentation engine based on image features.

Two-layer routing:
1. Pixel features (saturation, edge density) -> vivid / mixed / pastel / grayscale
2. Quality/speed preference -> specific engine
"""

from __future__ import annotations

import numpy as np

from geoseg.modules.segment_engines._shared import saturation_ratio
from geoseg.pipeline_interfaces import SegmentationResult, SegmentationMeta


def _is_grayscale(panel_rgb: np.ndarray, threshold: float = 15.0) -> bool:
    """Check if panel is grayscale (low per-pixel saturation)."""
    diff = panel_rgb.max(axis=2).astype(np.float32) - panel_rgb.min(axis=2).astype(np.float32)
    return float(diff.mean()) < threshold


def _edge_density(panel_rgb: np.ndarray) -> float:
    """Estimate edge density via grayscale Sobel magnitude."""
    from skimage.filters import sobel
    gray = panel_rgb.mean(axis=2)
    edges = sobel(gray)
    return float((edges > 0.05).mean())


def _stable_panel_hash(panel_rgb: np.ndarray, mod: int) -> int:
    """Deterministic hash from scattered pixels for engine rotation."""
    h, w = panel_rgb.shape[:2]
    # Sample grid pixels to avoid edge artifacts
    ys = np.linspace(h // 4, 3 * h // 4, 4).astype(int)
    xs = np.linspace(w // 4, 3 * w // 4, 4).astype(int)
    samples = panel_rgb[np.ix_(ys, xs)].flatten().astype(np.uint64)
    return int((np.sum(samples) + h * 7 + w * 13) % mod)


# DEPRECATED: Hard-coded retry chain replaced by `.claude/skills/sandbox-segment`.
# The agent now autonomously selects engines, evaluates results, and decides
# retry strategy. Kept for backward compatibility only.
_RETRY_CHAIN: dict[str, str] = {
    "grayscale_agglomerative": "v4_kmeans",
    "v4_kmeans_pastel": "v4_kmeans_colorbar",
    "v4_kmeans_colorbar": "kmeans_full",
    "v4_kmeans": "edge_guided",
    "edge_guided": "edge_grow",
    "edge_grow": "kmeans_full",
    "kmeans_full": "edge_guided",
}


def select_engine(
    panel_rgb: np.ndarray,
    quality_preference: str = "balanced",  # "fast" | "balanced" | "best"
    has_colorbar: bool = False,
    is_velocity_model: bool = True,
) -> str:
    """DEPRECATED: Hard-coded engine selection replaced by `.claude/skills/sandbox-segment`.
    The agent now autonomously decides which engine(s) to run based on visual
    analysis. Kept for backward compatibility only.

    Select segmentation engine based on pixel features.

    Args:
        panel_rgb: RGB uint8 array.
        quality_preference: "fast" (~0.1s), "balanced" (~0.2s), or "best" (~1s).
        has_colorbar: Whether a colorbar is available for seeding.
        is_velocity_model: If False, skip segmentation (not a velocity model).

    Returns:
        Engine name string.
    """
    if not is_velocity_model:
        return "skip"

    sat = saturation_ratio(panel_rgb)

    # Truly grayscale (near-zero saturation) -> grayscale engine
    if sat < 0.005:
        return "grayscale_agglomerative"

    if sat < 0.1:
        # Pastel / faded -> colorbar-guided if available, else fallback
        return "v4_kmeans_colorbar" if has_colorbar else "v4_kmeans_pastel"

    if 0.1 <= sat < 0.5:
        # Mixed saturation -> v4_kmeans is safest
        return "v4_kmeans"

    # Vivid (sat >= 0.5) — rotate among rep-based engines deterministically
    if quality_preference == "fast":
        return "v4_kmeans"
    if quality_preference == "balanced":
        engines = ["kmeans_full", "edge_guided", "edge_grow"]
        return engines[_stable_panel_hash(panel_rgb, len(engines))]
    if quality_preference == "best":
        engines = ["ensemble", "kmeans_full", "edge_guided", "edge_grow"]
        return engines[_stable_panel_hash(panel_rgb, len(engines))]

    # Default for vivid with smooth boundary preference
    return "edge_guided"


def _normalize_result(raw: dict, engine_name: str, n_layers: int) -> SegmentationResult:
    """Normalize engine output to SegmentationResult."""
    meta = raw.get("meta", {})
    return {
        "labels": raw["labels"],
        "overlay": raw.get("overlay"),
        "meta": {
            "engine": meta.get("engine", engine_name),
            "color_names": meta.get("color_names", []),
            "n_layers": n_layers,
            "quality_score": meta.get("quality_score"),
        },
    }


def _run_engine(
    engine: str,
    panel_rgb: np.ndarray,
    reps: list[dict] | None,
    colorbar_rgb: np.ndarray | None,
    n_layers: int,
    n_color_zones: int = 0,
) -> SegmentationResult:
    """Run a specific engine by name (internal helper)."""

    def _v4_fallback():
        from geoseg.modules.segment_engines.v4_kmeans import segment
        return segment(panel_rgb, reps=reps, colorbar_rgb=colorbar_rgb, n_layers=n_layers, n_color_zones=n_color_zones)

    def _normalize(engine_name: str, raw: dict) -> SegmentationResult:
        meta = raw.get("meta", {})
        return {
            "labels": raw["labels"],
            "overlay": raw.get("overlay"),
            "meta": {
                "engine": meta.get("engine", engine_name),
                "color_names": meta.get("color_names", []),
                "n_layers": n_layers,
                "quality_score": meta.get("quality_score"),
            },
        }

    def _run_with_fallback(primary_fn, fallback_fn, engine_name: str):
        try:
            return _normalize(engine_name, primary_fn())
        except Exception as exc:
            try:
                result = fallback_fn()
                result["meta"]["fallback_reason"] = str(exc)
                return _normalize(engine_name, result)
            except Exception:
                return _normalize(
                    engine_name,
                    {
                        "labels": np.zeros(panel_rgb.shape[:2], dtype=np.int32),
                        "seeds": [],
                        "overlay": panel_rgb.copy(),
                        "meta": {
                            "engine": engine_name,
                            "error": f"primary: {exc}; fallback also failed",
                        },
                    },
                )

    if engine == "grayscale_agglomerative":
        from geoseg.modules.segment_engines.grayscale import segment
        return _run_with_fallback(
            lambda: segment(panel_rgb, n_layers=n_layers, reps=reps),
            _v4_fallback,
            engine,
        )

    if engine in ("v4_kmeans_colorbar", "v4_kmeans_pastel"):
        from geoseg.modules.segment_engines.v4_kmeans import segment_colorbar_guided, segment_pastel_faded
        if engine == "v4_kmeans_colorbar":
            return _run_with_fallback(
                lambda: segment_colorbar_guided(panel_rgb, colorbar_rgb, n_layers=n_layers, n_color_zones=n_color_zones),
                _v4_fallback,
                engine,
            )
        return _run_with_fallback(
            lambda: segment_pastel_faded(panel_rgb, colorbar_rgb, n_layers=n_layers, n_color_zones=n_color_zones),
            _v4_fallback,
            engine,
        )

    if engine == "v4_kmeans":
        from geoseg.modules.segment_engines.v4_kmeans import segment
        return _normalize(
            "v4_kmeans",
            segment(panel_rgb, reps=reps, colorbar_rgb=colorbar_rgb, n_layers=n_layers, n_color_zones=n_color_zones),
        )

    if engine == "kmeans_full":
        from geoseg.modules.segment_engines.kmeans_full import segment
        if not reps:
            return _normalize("v4_kmeans", _v4_fallback())
        return _run_with_fallback(
            lambda: segment(panel_rgb, reps, n_layers=n_layers),
            _v4_fallback,
            engine,
        )

    if engine == "edge_guided":
        from geoseg.modules.segment_engines.edge_guided import segment
        if not reps:
            return _normalize("v4_kmeans", _v4_fallback())
        return _run_with_fallback(
            lambda: segment(panel_rgb, reps, n_layers=n_layers),
            _v4_fallback,
            engine,
        )

    if engine == "edge_grow":
        from geoseg.modules.segment_engines.edge_grow import segment
        if not reps:
            return _normalize("v4_kmeans", _v4_fallback())
        return _run_with_fallback(
            lambda: segment(panel_rgb, reps, n_layers=n_layers),
            _v4_fallback,
            engine,
        )

    if engine == "ensemble":
        from geoseg.modules.segment_engines.ensemble import segment
        if not reps:
            return _normalize("v4_kmeans", _v4_fallback())
        return _run_with_fallback(
            lambda: segment(panel_rgb, reps, n_layers=n_layers),
            _v4_fallback,
            engine,
        )

    return _normalize("v4_kmeans", _v4_fallback())


def route_and_segment(
    panel_rgb: np.ndarray,
    reps: list[dict] | None = None,
    colorbar_rgb: np.ndarray | None = None,
    n_layers: int = 5,
    quality_preference: str = "balanced",
    is_velocity_model: bool = True,
    retry_on_underseg: bool = True,
    n_color_zones: int = 0,
) -> SegmentationResult:
    """Route to engine and run segmentation.

    Implements the `Segmenter` Protocol.

    Args:
        panel_rgb: RGB uint8 array (H, W, 3).
        reps: VLM representative points (for vivid paths).
        colorbar_rgb: Optional colorbar strip.
        n_layers: Number of layers.
        quality_preference: "fast", "balanced", or "best".
        is_velocity_model: If False, returns skip result.
        retry_on_underseg: If True, retry with fallback engine when n_layers < 2.

    Returns:
        SegmentationResult with keys: labels, overlay, meta.
    """
    engine = select_engine(
        panel_rgb,
        quality_preference=quality_preference,
        has_colorbar=colorbar_rgb is not None and colorbar_rgb.size > 0,
        is_velocity_model=is_velocity_model,
    )

    if engine == "skip":
        return {
            "labels": np.zeros(panel_rgb.shape[:2], dtype=np.int32),
            "overlay": panel_rgb.copy(),
            "meta": {
                "engine": "skip",
                "color_names": [],
                "n_layers": n_layers,
                "reason": "not_velocity_model",
            },
        }

    seg = _run_engine(engine, panel_rgb, reps, colorbar_rgb, n_layers, n_color_zones=n_color_zones)

    # ── Auto-retry on under-segmentation ─────────────────────────────
    if retry_on_underseg:
        labels = seg["labels"]
        n_found = len(set(labels.flatten()) - {0})
        if n_found < 2:
            retry_engine = _RETRY_CHAIN.get(engine)
            if retry_engine:
                seg_retry = _run_engine(retry_engine, panel_rgb, reps, colorbar_rgb, n_layers, n_color_zones=n_color_zones)
                labels_retry = seg_retry["labels"]
                n_found_retry = len(set(labels_retry.flatten()) - {0})
                if n_found_retry > n_found:
                    seg = seg_retry
                    seg["meta"]["retry_from"] = engine
                    seg["meta"]["retry_reason"] = f"under_segmented_{n_found}_layers"

    return seg
