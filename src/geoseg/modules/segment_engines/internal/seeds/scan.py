"""Full-image missing-colour seed scanning."""

from __future__ import annotations

import numpy as np
from skimage.color import rgb2lab

from geoseg.modules.segment_engines.internal.color import _is_background_v2
from geoseg.modules.segment_engines.internal.seeds.cv import _online_color_groups
from geoseg.modules.segment_engines.internal.seeds.search import _find_pixel_for_color


def _scan_for_missing_colors(
    panel_rgb: np.ndarray,
    existing_seeds_lab: np.ndarray,
    bg_rgb: np.ndarray,
    max_auto_k: int,
    min_auto_count: int,
    existing_auto_rgb: list[np.ndarray] | None = None,
) -> list[tuple[np.ndarray, int, int, int]]:
    """Scan the full image for dominant colors not already covered by existing seeds."""
    centers, counts = _online_color_groups(
        panel_rgb,
        tolerance=40.0,
        max_groups=30,
        max_samples=20000,
        seed=42,
    )

    auto_selected: list[tuple[np.ndarray, int, int, int]] = []
    auto_rgb_list: list[np.ndarray] = list(existing_auto_rgb) if existing_auto_rgb else []

    sorted_groups = sorted(zip(centers, counts), key=lambda t: t[1], reverse=True)

    for cseed, count in sorted_groups:
        if len(auto_selected) >= max_auto_k:
            break

        if _is_background_v2(cseed, bg_rgb, threshold=60.0):
            continue

        if count < min_auto_count:
            continue

        cseed_lab = rgb2lab(cseed[np.newaxis, ...])[0]
        d = float(np.linalg.norm(existing_seeds_lab - cseed_lab, axis=1).min())
        if d < 20.0:
            continue

        if auto_rgb_list:
            auto_arr = np.array(auto_rgb_list, dtype=np.float32)
            if np.linalg.norm(auto_arr - cseed.astype(np.float32), axis=1).min() <= 30.0:
                continue

        found_px = _find_pixel_for_color(panel_rgb, cseed, bg_rgb, color_tol=40.0, bg_tol=50.0)
        if found_px is None:
            continue

        cx, cy = found_px
        auto_selected.append((cseed, cx, cy, int(count)))
        auto_rgb_list.append(cseed)

    return auto_selected

__all__ = ["_scan_for_missing_colors"]
