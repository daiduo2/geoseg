"""Edge-enhanced multi-source region growing for vivid jet-colormap panels.

Uses LAB gradient magnitude as a barrier during Dijkstra expansion so that
region growing stops at actual geological boundaries rather than bleeding
across gradual colour transitions.
"""

from __future__ import annotations

import heapq

import numpy as np
from skimage.color import rgb2lab
from skimage.filters import sobel

from lib.segment import (
    SegmentResult,
    _cv_seeds,
    _estimate_background_color,
    _find_pixel_for_color,
    _is_background_v2,
    _label_by_nearest,
    _parse_count_from_tag,
    _scan_for_missing_colors,
    _shape_filter,
    _spiral_search,
    saturation_ratio,
)


def _refine_seeds(
    panel_rgb: np.ndarray,
    panel_lab: np.ndarray,
    reps: list[dict],
    bg_rgb: np.ndarray,
    cv_seeds_rgb: np.ndarray,
    cv_tags: list[str],
    used_cv_indices: set[int],
) -> tuple[list[np.ndarray], list[dict]]:
    """Refine VLM seeds with multi-source fallback (same logic as segment.py)."""
    h, w = panel_rgb.shape[:2]
    raw_rgb = []
    for r in reps:
        x = int(r["representative_point"]["x"])
        y = int(r["representative_point"]["y"])
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
        ox = int(r["representative_point"]["x"])
        oy = int(r["representative_point"]["y"])
        ox = max(0, min(w - 1, ox))
        oy = max(0, min(h - 1, oy))
        raw_vlm_rgb = panel_rgb[oy, ox]

        cx, cy, rgb, source = ox, oy, raw_vlm_rgb, "raw_vlm"

        if _is_background_v2(raw_vlm_rgb, bg_rgb):
            found = _spiral_search(
                panel_rgb, ox, oy, radius=min(h, w) // 3, is_bg_func=lambda c: _is_background_v2(c, bg_rgb)
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
                from skimage.morphology import disk, erosion
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
                    panel_rgb, ox, oy, radius=20, is_bg_func=lambda c: _is_background_v2(c, bg_rgb)
                )
                if found:
                    best_cx, best_cy = found
                    best_rgb = panel_rgb[best_cy, best_cx]
                    source = "spiral_search_nearby"
                else:
                    mask = rough_labels == idx
                    from lib.segment import _erode_internal_point
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
            "name": r["color_name"],
            "vlm_x": ox,
            "vlm_y": oy,
            "rgb": raw_rgb[idx].tolist(),
            "internal_x": cx,
            "internal_y": cy,
            "on_background": bool(_is_background_v2(raw_vlm_rgb, bg_rgb)),
            "source": source,
        })

    return refined_seeds, refined_reps


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
    refined_seeds_arr = np.array(refined_seeds, dtype=np.uint8)
    refined_lab = rgb2lab(refined_seeds_arr[np.newaxis, ...])[0]

    auto_seeds: list[np.ndarray] = []
    auto_reps: list[dict] = []
    auto_rgb_list: list[np.ndarray] = []

    # CV path
    if max_auto_k > 0 and len(cv_seeds_rgb) > len(used_cv_indices):
        candidates = []
        for ci, (cseed, tag) in enumerate(zip(cv_seeds_rgb, cv_tags)):
            if ci in used_cv_indices:
                continue
            count = _parse_count_from_tag(tag)
            if count < min_auto_count:
                continue
            cseed_lab = rgb2lab(cseed[np.newaxis, ...])[0]
            d = float(np.linalg.norm(refined_lab - cseed_lab, axis=1).min())
            candidates.append((ci, cseed, tag, count, d))

        candidates.sort(key=lambda t: (t[3], t[4]), reverse=True)

        for ci, cseed, tag, count, d in candidates:
            if len(auto_seeds) >= max_auto_k:
                break
            if d < 20:
                continue
            if auto_rgb_list:
                auto_arr = np.array(auto_rgb_list, dtype=np.float32)
                if np.linalg.norm(auto_arr - cseed.astype(np.float32), axis=1).min() <= 30:
                    continue
            found_px = _find_pixel_for_color(panel_rgb, cseed, bg_rgb, color_tol=40, bg_tol=50)
            if found_px:
                cx, cy = found_px
            else:
                cx, cy = w // 2, h // 2
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

    # Full-image scan path
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


def _region_grow_dijkstra_edge(
    panel_lab: np.ndarray,
    seeds_xy: list[tuple[int, int]],
    seeds_lab: np.ndarray,
    edge_map: np.ndarray,
    edge_penalty: float = 100.0,
) -> np.ndarray:
    """Multi-source Dijkstra in LAB space with edge barrier penalty.

    Each seed grows outward. Crossing a strong edge (high gradient) incurs an
    additional cost proportional to ``edge_penalty * edge_map[ny, nx]``.
    Pixels are assigned to the seed that reaches them with the smallest total
    cost. Unassigned pixels are filled with the nearest seed label.
    """
    h, w = panel_lab.shape[:2]
    k = len(seeds_xy)

    # Precompute LAB distances from every pixel to every seed: (H, W, k)
    diff = panel_lab[:, :, None, :] - seeds_lab[None, None, :, :]
    dists = np.linalg.norm(diff, axis=3)

    best_cost = np.full((h, w), np.inf, dtype=np.float32)
    best_label = np.full((h, w), -1, dtype=np.int32)
    heap = []

    for i, (x, y) in enumerate(seeds_xy):
        d = float(dists[y, x, i])
        best_cost[y, x] = d
        best_label[y, x] = i
        heapq.heappush(heap, (d, i, x, y))

    while heap:
        cost, i, x, y = heapq.heappop(heap)
        if cost > best_cost[y, x] + 1e-6:
            continue
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                edge_cost = edge_penalty * float(edge_map[ny, nx])
                color_cost = float(dists[ny, nx, i])
                new_cost = cost + color_cost + edge_cost
                if new_cost < best_cost[ny, nx] - 1e-6:
                    best_cost[ny, nx] = new_cost
                    best_label[ny, nx] = i
                    heapq.heappush(heap, (new_cost, i, nx, ny))

    # Fill any unassigned pixels with nearest seed
    unassigned = best_label == -1
    if unassigned.any():
        nearest = dists.argmin(axis=2)
        best_label[unassigned] = nearest[unassigned]

    return best_label


def segment_jet_vivid_edge_grow(
    panel_rgb: np.ndarray,
    reps: list[dict],
    max_auto_k: int = 3,
    edge_penalty: float = 100.0,
) -> SegmentResult:
    """Edge-enhanced multi-source region growing for vivid jet-colormap panels.

    1. Compute LAB gradient magnitude and normalize to [0, 1] edge map.
    2. Refine VLM seeds (same multi-source fallback as baseline).
    3. Auto-k for missing colors.
    4. Multi-source Dijkstra with edge barrier penalty.
    5. Post-process with perimeter^2 / area shape filter.
    """
    if not reps:
        raise ValueError("jet_vivid_edge_grow path requires at least one rep")

    h, w, _ = panel_rgb.shape
    panel_lab = rgb2lab(panel_rgb)
    bg_rgb = _estimate_background_color(panel_rgb)
    min_auto_count = max(10, h * w // 3000)

    # --- Edge map ---
    gradient = np.zeros((h, w), dtype=np.float32)
    for c in range(3):
        gradient += sobel(panel_lab[..., c]) ** 2
    gradient = np.sqrt(gradient)
    edge_map = gradient / (gradient.max() + 1e-9)

    # --- CV fallback seeds ---
    cv_seeds_rgb, cv_tags = _cv_seeds(panel_rgb, k=len(reps))
    used_cv_indices: set[int] = set()

    # --- Seed refinement ---
    refined_seeds, refined_reps = _refine_seeds(
        panel_rgb, panel_lab, reps, bg_rgb, cv_seeds_rgb, cv_tags, used_cv_indices
    )
    color_names = [r["color_name"] for r in reps]

    # --- Auto-k ---
    refined_seeds, refined_reps = _auto_k(
        panel_rgb, panel_lab, bg_rgb, refined_seeds, refined_reps,
        cv_seeds_rgb, cv_tags, used_cv_indices, max_auto_k, min_auto_count
    )
    if len(refined_reps) > len(color_names):
        color_names = color_names + [r["name"] for r in refined_reps[len(color_names):]]

    refined_seeds_arr = np.array(refined_seeds, dtype=np.uint8)
    seeds_lab = rgb2lab(refined_seeds_arr[np.newaxis, ...])[0]
    seeds_xy = [(rep["internal_x"], rep["internal_y"]) for rep in refined_reps]

    # --- Edge-enhanced Dijkstra region growing ---
    labels = _region_grow_dijkstra_edge(
        panel_lab, seeds_xy, seeds_lab, edge_map, edge_penalty=edge_penalty
    )

    # --- Shape filter ---
    labels = _shape_filter(labels)

    return SegmentResult(
        labels=labels,
        palette=refined_seeds_arr,
        color_names=color_names,
        path="jet_vivid_edge_grow",
        saturation_ratio=saturation_ratio(panel_rgb),
        notes={
            "reps_refined": refined_reps,
            "cv_seeds": cv_seeds_rgb.tolist() if len(cv_seeds_rgb) else [],
            "bg_rgb": bg_rgb.tolist(),
            "auto_k_added": len(refined_reps) - len(reps),
            "edge_penalty": edge_penalty,
            "edge_map_stats": {
                "min": float(edge_map.min()),
                "max": float(edge_map.max()),
                "mean": float(edge_map.mean()),
                "median": float(np.median(edge_map)),
            },
        },
    )
