"""VLM representative point seed refinement."""

from __future__ import annotations

import numpy as np
from skimage.color import rgb2lab
from skimage.morphology import disk, erosion

from geoseg.modules.segment_engines.internal.color import (
    _is_background_v2,
    _label_by_nearest,
)
from geoseg.modules.segment_engines.internal.regions import _erode_internal_point
from geoseg.modules.segment_engines.internal.seeds.search import (
    _find_pixel_for_color,
    _spiral_search,
)


def _refine_vlm_seeds(
    panel_rgb: np.ndarray,
    reps: list[dict] | None,
    bg_rgb: np.ndarray,
    cv_seeds_rgb: np.ndarray,
    cv_tags: list[str],
    used_cv_indices: set[int],
) -> tuple[list[np.ndarray], list[dict]]:
    """Refine VLM representative points into robust seed colors + locations.

    Returns (refined_seeds, refined_reps_metadata).
    """
    if not reps:
        return [], []

    h, w = panel_rgb.shape[:2]
    panel_lab = rgb2lab(panel_rgb)

    raw_rgb = []
    for r in reps:
        rp = r.get("representative_point", {}) if isinstance(r, dict) else {}
        x = int(rp.get("x", w // 2))
        y = int(rp.get("y", h // 2))
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        x0, x1 = max(0, x - 1), min(w, x + 2)
        y0, y1 = max(0, y - 1), min(h, y + 2)
        patch = panel_rgb[y0:y1, x0:x1]
        rgb = patch.reshape(-1, 3).mean(axis=0).astype(np.uint8)
        raw_rgb.append(rgb)

    raw_rgb = np.array(raw_rgb, dtype=np.uint8)
    raw_lab = rgb2lab(raw_rgb[np.newaxis, ...])[0]
    rough_labels = _label_by_nearest(panel_lab, raw_lab)

    refined_seeds = []
    refined_reps = []

    for idx, r in enumerate(reps):
        rp = r.get("representative_point", {}) if isinstance(r, dict) else {}
        ox = int(rp.get("x", w // 2))
        oy = int(rp.get("y", h // 2))
        ox = max(0, min(w - 1, ox))
        oy = max(0, min(h - 1, oy))
        raw_vlm_rgb = panel_rgb[oy, ox]

        cx, cy, rgb, source = ox, oy, raw_vlm_rgb, "raw_vlm"

        if _is_background_v2(raw_vlm_rgb, bg_rgb):
            found = _spiral_search(
                panel_rgb, ox, oy, radius=min(h, w) // 3,
                is_bg_func=lambda c: _is_background_v2(c, bg_rgb)
            )
            if found:
                cx, cy = found
                rgb = panel_rgb[cy, cx]
                source = "spiral_search"
            else:
                if len(cv_seeds_rgb) > 0:
                    best_idx = None
                    best_score = -1.0
                    for ci, cseed in enumerate(cv_seeds_rgb):
                        if ci in used_cv_indices:
                            continue
                        bg_dist = float(
                            np.linalg.norm(cseed.astype(np.float32) - bg_rgb.astype(np.float32))
                        )
                        if bg_dist > best_score:
                            best_score = bg_dist
                            best_idx = ci
                    if best_idx is not None:
                        used_cv_indices.add(best_idx)
                        cseed = cv_seeds_rgb[best_idx]
                        found_px = _find_pixel_for_color(panel_rgb, cseed, bg_rgb)
                        if found_px:
                            cx, cy = found_px
                            rgb = panel_rgb[cy, cx]
                        else:
                            rgb = cseed
                        source = f"cv_{cv_tags[best_idx]}"
                    else:
                        source = "failed_all_cv_used"
                else:
                    source = "failed_no_cv"
        else:
            best_cx, best_cy = ox, oy
            best_rgb = raw_vlm_rgb
            source = "raw_vlm"

            y0, y1 = max(0, oy - 30), min(h, oy + 31)
            x0, x1 = max(0, ox - 30), min(w, ox + 31)
            local_mask = (rough_labels[y0:y1, x0:x1] == idx)
            if local_mask.any():
                m = local_mask.copy()
                for rad in (5, 3, 1):
                    eroded = erosion(m, footprint=disk(rad))
                    if eroded.any():
                        m = eroded
                        break
                ys, xs = np.where(m)
                if len(xs) > 0:
                    lcx = int(np.median(xs)) + x0
                    lcy = int(np.median(ys)) + y0
                    if abs(lcx - ox) <= 20 and abs(lcy - oy) <= 20:
                        best_cx, best_cy = lcx, lcy
                        best_rgb = panel_rgb[lcy, lcx]
                        source = "local_erode"

            if _is_background_v2(best_rgb, bg_rgb):
                found = _spiral_search(
                    panel_rgb, ox, oy, radius=20,
                    is_bg_func=lambda c: _is_background_v2(c, bg_rgb)
                )
                if found:
                    best_cx, best_cy = found
                    best_rgb = panel_rgb[best_cy, best_cx]
                    source = "spiral_search_nearby"
                else:
                    mask = rough_labels == idx
                    ip = _erode_internal_point(mask)
                    if ip is not None:
                        best_cx, best_cy = ip
                        best_rgb = panel_rgb[best_cy, best_cx]
                        source = "global_erode_fallback"

            cx, cy = best_cx, best_cy
            cx = max(0, min(w - 1, cx))
            cy = max(0, min(h - 1, cy))
            rgb = panel_rgb[cy, cx]

        refined_seeds.append(rgb)
        refined_reps.append({
            "name": r.get("color_name", f"layer_{idx + 1}") if isinstance(r, dict) else f"layer_{idx + 1}",
            "vlm_x": ox,
            "vlm_y": oy,
            "rgb": raw_rgb[idx].tolist(),
            "internal_x": cx,
            "internal_y": cy,
            "on_background": bool(_is_background_v2(raw_vlm_rgb, bg_rgb)),
            "source": source,
        })

    return refined_seeds, refined_reps

__all__ = ["_refine_vlm_seeds"]
