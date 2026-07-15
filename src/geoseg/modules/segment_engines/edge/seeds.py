"""Seed preparation shared by edge-based engines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.color import rgb2lab

from geoseg.modules.segment_engines.internal.color import _estimate_background_color
from geoseg.modules.segment_engines.internal.seeds import (
    _auto_k,
    _cv_seeds,
    _refine_vlm_seeds,
)


@dataclass(frozen=True)
class EdgeSeeds:
    """Prepared seed data for edge-based engines."""

    refined_seeds_rgb: np.ndarray
    refined_reps: list[dict]
    cv_seeds_rgb: np.ndarray
    bg_rgb: np.ndarray
    color_names: list[str]
    initial_rep_count: int

    @property
    def auto_k_added(self) -> int:
        return len(self.refined_reps) - self.initial_rep_count


def prepare_edge_seeds(
    panel_rgb: np.ndarray,
    panel_lab: np.ndarray,
    reps: list[dict] | None,
    n_layers: int,
    max_auto_k: int,
) -> EdgeSeeds:
    """Prepare CV/VLM/auto-k seeds for edge-based engines."""
    h, w = panel_rgb.shape[:2]
    bg_rgb = _estimate_background_color(panel_rgb)
    min_auto_count = max(50, h * w // 2000)

    if reps:
        cv_seeds_rgb, cv_tags = _cv_seeds(panel_rgb, k=len(reps))
        used_cv_indices: set[int] = set()
        refined_seeds, refined_reps = _refine_vlm_seeds(
            panel_rgb, reps, bg_rgb, cv_seeds_rgb, cv_tags, used_cv_indices
        )
        color_names = [r.get("color_name", f"layer_{i + 1}") for i, r in enumerate(reps)]
    else:
        cv_seeds_rgb, cv_tags = _cv_seeds(panel_rgb, k=n_layers)
        used_cv_indices = set()
        refined_seeds = []
        refined_reps = []
        color_names = [f"layer_{i + 1}" for i in range(n_layers)]

    refined_seeds, refined_reps = _auto_k(
        panel_rgb,
        panel_lab,
        bg_rgb,
        refined_seeds,
        refined_reps,
        cv_seeds_rgb,
        cv_tags,
        used_cv_indices,
        max_auto_k,
        min_auto_count,
    )
    if len(refined_reps) > len(color_names):
        color_names = color_names + [r["name"] for r in refined_reps[len(color_names):]]

    if not refined_seeds:
        refined_seeds = [cv_seeds_rgb[i] for i in range(min(n_layers, len(cv_seeds_rgb)))]

    return EdgeSeeds(
        refined_seeds_rgb=np.array(refined_seeds, dtype=np.uint8),
        refined_reps=refined_reps,
        cv_seeds_rgb=cv_seeds_rgb,
        bg_rgb=bg_rgb,
        color_names=color_names,
        initial_rep_count=len(reps) if reps else 0,
    )


def seeds_lab(seeds_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB seeds to LAB."""
    return rgb2lab(seeds_rgb[np.newaxis, ...])[0]


__all__ = ["EdgeSeeds", "prepare_edge_seeds", "seeds_lab"]
