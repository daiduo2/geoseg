"""Automatic missing seed selection."""

from __future__ import annotations

import numpy as np
from skimage.color import rgb2lab

from geoseg.modules.segment_engines.internal.seeds.parse import _parse_count_from_tag
from geoseg.modules.segment_engines.internal.seeds.scan import _scan_for_missing_colors
from geoseg.modules.segment_engines.internal.seeds.search import _find_pixel_for_color


def _auto_k(
    panel_rgb: np.ndarray,
    panel_lab: np.ndarray,
    bg_rgb: np.ndarray,
    refined_seeds: list[np.ndarray],
    refined_reps: list[dict],
    cv_seeds_rgb: np.ndarray,
    cv_tags: list[str],
    used_cv_indices: set[int],
    max_auto_k: int,
    min_auto_count: int,
) -> tuple[list[np.ndarray], list[dict]]:
    """Detect missing colors from unused CV seeds + full-image scan."""
    h, w = panel_rgb.shape[:2]
    if not refined_seeds:
        refined_seeds_arr = np.empty((0, 3), dtype=np.uint8)
        refined_lab = np.empty((0, 3), dtype=np.float64)
    else:
        refined_seeds_arr = np.array(refined_seeds, dtype=np.uint8)
        refined_lab = rgb2lab(refined_seeds_arr[np.newaxis, ...])[0]

    auto_seeds: list[np.ndarray] = []
    auto_reps: list[dict] = []
    auto_rgb_list: list[np.ndarray] = []

    if max_auto_k > 0 and len(cv_seeds_rgb) > len(used_cv_indices):
        candidates = []
        has_refined = refined_lab.shape[0] > 0
        for ci, (cseed, tag) in enumerate(zip(cv_seeds_rgb, cv_tags)):
            if ci in used_cv_indices:
                continue
            count = _parse_count_from_tag(tag)
            if count < min_auto_count:
                continue
            cseed_lab = rgb2lab(cseed[np.newaxis, ...])[0]
            if has_refined:
                d = float(np.linalg.norm(refined_lab - cseed_lab, axis=1).min())
            else:
                d = float('inf')
            candidates.append((ci, cseed, tag, count, d))

        candidates.sort(key=lambda t: (t[3], t[4]), reverse=True)

        for ci, cseed, tag, count, d in candidates:
            if len(auto_seeds) >= max_auto_k:
                break
            # Require substantial LAB distance from existing seeds
            if d < 30:
                continue
            if auto_rgb_list:
                auto_arr = np.array(auto_rgb_list, dtype=np.float32)
                if np.linalg.norm(auto_arr - cseed.astype(np.float32), axis=1).min() <= 30:
                    continue
            # Require the color to cover a meaningful area
            found_px = _find_pixel_for_color(panel_rgb, cseed, bg_rgb, color_tol=40, bg_tol=50)
            if found_px is None:
                continue
            cx, cy = found_px
            # Additional area check: color must occupy at least 0.5% of image
            color_mask = np.linalg.norm(
                panel_rgb.astype(np.float32) - cseed.astype(np.float32), axis=2
            ) < 50
            bg_mask = np.linalg.norm(
                panel_rgb.astype(np.float32) - bg_rgb.astype(np.float32), axis=2
            ) < 50
            color_mask &= ~bg_mask
            if color_mask.sum() < h * w * 0.005:
                continue
            auto_seeds.append(cseed)
            auto_rgb_list.append(cseed)
            auto_reps.append({
                "name": f"auto_{len(auto_seeds)}",
                "vlm_x": None,
                "vlm_y": None,
                "rgb": cseed.tolist(),
                "internal_x": cx,
                "internal_y": cy,
                "on_background": False,
                "source": f"auto_cv_{tag}",
            })

    if len(auto_seeds) < max_auto_k:
        remaining = max_auto_k - len(auto_seeds)
        scan_results = _scan_for_missing_colors(
            panel_rgb,
            refined_lab,
            bg_rgb,
            max_auto_k=remaining,
            min_auto_count=min_auto_count,
            existing_auto_rgb=auto_rgb_list if auto_rgb_list else None,
        )
        for cseed, cx, cy, count in scan_results:
            auto_seeds.append(cseed)
            auto_rgb_list.append(cseed)
            auto_reps.append({
                "name": f"auto_{len(auto_seeds)}",
                "vlm_x": None,
                "vlm_y": None,
                "rgb": cseed.tolist(),
                "internal_x": cx,
                "internal_y": cy,
                "on_background": False,
                "source": f"auto_scan_count={count}",
            })

    if auto_seeds:
        refined_seeds = refined_seeds + auto_seeds
        refined_reps = refined_reps + auto_reps

    return refined_seeds, refined_reps

__all__ = ["_auto_k"]
