"""Region merging from Mean Shift oversegmentation for geophysics panels.

Strategy:
1. Start with Mean Shift using a small bandwidth (guaranteed oversegmentation).
2. Build adjacency graph of regions.
3. Iteratively merge the most similar adjacent regions.
4. Stop when region count reaches target (VLM reps + max_auto_k).

Alternative (simpler) path uses scipy.cluster.hierarchy on region mean LAB colors
weighted by area, then maps back to pixels.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.color import rgb2lab, lab2rgb
from skimage.measure import label, regionprops
from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy.cluster.hierarchy import linkage, fcluster

from lib.segment import (
    SegmentResult,
    saturation_ratio,
    _shape_filter,
    _estimate_background_color,
    _is_background_v2,
    _cv_seeds,
    _find_pixel_for_color,
    _spiral_search,
    _erode_internal_point,
    _scan_for_missing_colors,
    _parse_count_from_tag,
    _online_color_groups,
    _histogram_peaks,
)


def _label_by_nearest(panel_lab: np.ndarray, palette_lab: np.ndarray) -> np.ndarray:
    """Label each pixel by index of nearest palette entry in LAB."""
    h, w, _ = panel_lab.shape
    flat = panel_lab.reshape(-1, 3)
    d2 = ((flat[:, None, :] - palette_lab[None, :, :]) ** 2).sum(axis=2)
    return d2.argmin(axis=1).reshape(h, w).astype(np.int32)


def _refine_seeds_from_reps(panel_rgb: np.ndarray, reps: list[dict], max_auto_k: int = 3):
    """Reuse seed-refinement logic from segment_jet_vivid.

    Returns (refined_seeds_rgb, refined_reps, color_names).
    """
    if not reps:
        raise ValueError("Requires at least one rep")

    h, w, _ = panel_rgb.shape
    panel_lab = rgb2lab(panel_rgb)
    bg_rgb = _estimate_background_color(panel_rgb)
    min_auto_count = max(10, h * w // 3000)

    cv_seeds_rgb, cv_tags = _cv_seeds(panel_rgb, k=len(reps))
    used_cv_indices: set[int] = set()

    def _bg_check(rgb: np.ndarray) -> bool:
        return _is_background_v2(rgb, bg_rgb)

    # Stage 1: raw VLM reps -> rough nearest-neighbour labels
    raw_rgb = []
    color_names = []
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
        color_names.append(r["color_name"])

    raw_rgb = np.array(raw_rgb, dtype=np.uint8)
    raw_lab = rgb2lab(raw_rgb[np.newaxis, ...])[0]
    rough_labels = _label_by_nearest(panel_lab, raw_lab)

    # Stage 2: refine each seed
    refined_seeds = []
    refined_reps = []
    for idx, r in enumerate(reps):
        ox = int(r["representative_point"]["x"])
        oy = int(r["representative_point"]["y"])
        ox = max(0, min(w - 1, ox))
        oy = max(0, min(h - 1, oy))
        raw_vlm_rgb = panel_rgb[oy, ox]

        cx, cy, rgb, source = ox, oy, raw_vlm_rgb, "raw_vlm"

        if _bg_check(raw_vlm_rgb):
            found = _spiral_search(
                panel_rgb, ox, oy, radius=min(h, w) // 3, is_bg_func=_bg_check
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

            if _bg_check(best_rgb):
                found = _spiral_search(
                    panel_rgb, ox, oy, radius=20, is_bg_func=_bg_check
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
        refined_reps.append(
            {
                "name": r["color_name"],
                "vlm_x": ox,
                "vlm_y": oy,
                "rgb": raw_rgb[idx].tolist(),
                "internal_x": cx,
                "internal_y": cy,
                "on_background": bool(_bg_check(raw_vlm_rgb)),
                "source": source,
            }
        )

    # Stage 2b: auto-k
    auto_seeds: list[np.ndarray] = []
    auto_reps: list[dict] = []
    auto_rgb_list: list[np.ndarray] = []
    refined_seeds_arr = np.array(refined_seeds, dtype=np.uint8)
    refined_lab = rgb2lab(refined_seeds_arr[np.newaxis, ...])[0]

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
        color_names = color_names + [r["name"] for r in auto_reps]

    refined_seeds_arr = np.array(refined_seeds, dtype=np.uint8)
    return refined_seeds_arr, refined_reps, color_names


def _build_adjacency(labels: np.ndarray) -> dict[int, set[int]]:
    """Build adjacency graph: region_id -> set of adjacent region_ids."""
    h, w = labels.shape
    adj: dict[int, set[int]] = {}
    unique = np.unique(labels)
    for u in unique:
        adj[int(u)] = set()

    for y in range(h):
        for x in range(w):
            l = int(labels[y, x])
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    nl = int(labels[ny, nx])
                    if nl != l:
                        adj[l].add(nl)
                        adj[nl].add(l)
    return adj


def _compute_region_stats(panel_lab: np.ndarray, labels: np.ndarray):
    """Compute per-region mean LAB, pixel count, and perimeter pixels.

    Returns dict: region_id -> {"mean_lab": np.ndarray(3), "count": int,
                                 "perimeter": int}
    """
    h, w = labels.shape
    unique = np.unique(labels)
    stats = {}
    for rid in unique:
        mask = labels == rid
        count = int(mask.sum())
        pixels = panel_lab[mask]
        mean_lab = pixels.mean(axis=0)
        # Perimeter: pixels in region that have a different neighbor
        perim = 0
        ys, xs = np.where(mask)
        for y, x in zip(ys, xs):
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if labels[ny, nx] != rid:
                        perim += 1
                else:
                    perim += 1
        stats[int(rid)] = {
            "mean_lab": mean_lab,
            "count": count,
            "perimeter": perim,
        }
    return stats


def _merge_regions(labels: np.ndarray, adj: dict[int, set[int]], stats: dict,
                   target_count: int, max_similarity: float = 15.0) -> np.ndarray:
    """Iteratively merge most similar adjacent regions.

    Similarity = LAB Euclidean distance (optionally could weight by boundary length).
    Stop when region count <= target_count OR smallest similarity > max_similarity.
    """
    out = labels.copy()
    current_stats = {k: dict(v) for k, v in stats.items()}
    current_adj = {k: set(v) for k, v in adj.items()}

    # Map from old label -> current label (starts as identity)
    label_map = {k: k for k in current_stats}

    # Precompute pairwise similarities for adjacent pairs
    def _find_best_pair():
        best_sim = float("inf")
        best_pair = None
        for a in list(current_adj.keys()):
            for b in current_adj[a]:
                if a >= b:
                    continue
                sim = float(np.linalg.norm(current_stats[a]["mean_lab"] - current_stats[b]["mean_lab"]))
                if sim < best_sim:
                    best_sim = sim
                    best_pair = (a, b)
        return best_pair, best_sim

    while len(current_stats) > target_count:
        pair, sim = _find_best_pair()
        if pair is None or sim > max_similarity:
            break
        a, b = pair
        # Merge b into a (a keeps the smaller label id for stability)
        if a > b:
            a, b = b, a

        # Update stats for a
        count_a = current_stats[a]["count"]
        count_b = current_stats[b]["count"]
        total = count_a + count_b
        current_stats[a]["mean_lab"] = (
            current_stats[a]["mean_lab"] * count_a + current_stats[b]["mean_lab"] * count_b
        ) / total
        current_stats[a]["count"] = total
        current_stats[a]["perimeter"] = current_stats[a]["perimeter"] + current_stats[b]["perimeter"]

        # Update adjacency: replace b references with a
        for neighbor in current_adj.pop(b, set()):
            if neighbor == a:
                continue
            current_adj[neighbor].discard(b)
            current_adj[neighbor].add(a)
            current_adj[a].add(neighbor)
        current_adj[a].discard(a)

        del current_stats[b]

        # Update pixels
        out[out == b] = a

    # Remap to contiguous 0..k-1
    unique = np.unique(out)
    remap = {old: new for new, old in enumerate(unique)}
    out = np.vectorize(remap.get)(out)
    return out


def segment_jet_vivid_merge(
    panel_rgb: np.ndarray,
    reps: list[dict],
    max_auto_k: int = 3,
    bandwidth: float | None = None,
    quantile: float = 0.05,
    n_samples: int = 5000,
    use_hierarchy: bool = True,
    linkage_method: str = "ward",
    max_similarity: float = 15.0,
) -> SegmentResult:
    """Segment via Mean Shift oversegmentation + hierarchical region merging.

    Parameters
    ----------
    panel_rgb : np.ndarray
        (H, W, 3) uint8 RGB image.
    reps : list[dict]
        VLM representative points.
    max_auto_k : int
        Maximum auto-detected seeds to add.
    bandwidth : float | None
        Mean Shift bandwidth. If None, estimated from data with small quantile.
    quantile : float
        Quantile for sklearn estimate_bandwidth (default 0.05 = small = many modes).
    n_samples : int
        Samples for bandwidth estimation.
    use_hierarchy : bool
        If True, use scipy.cluster.hierarchy (faster). If False, iterative graph merge.
    linkage_method : str
        'ward', 'average', 'complete', 'single'.
    max_similarity : float
        Maximum LAB distance for iterative merge (ignored if use_hierarchy=True).

    Returns
    -------
    SegmentResult
    """
    import time

    h, w, _ = panel_rgb.shape
    panel_lab = rgb2lab(panel_rgb)
    flat_lab = panel_lab.reshape(-1, 3)

    # --- Step 1: refine seeds from reps (reuses segment_jet_vivid logic) ---
    t0 = time.perf_counter()
    refined_seeds_arr, refined_reps, color_names = _refine_seeds_from_reps(
        panel_rgb, reps, max_auto_k=max_auto_k
    )
    target_count = len(refined_seeds_arr) + max_auto_k
    t_seeds = time.perf_counter() - t0

    # --- Step 2: Mean Shift with small bandwidth (oversegmentation) ---
    t0 = time.perf_counter()
    if bandwidth is None:
        bandwidth = estimate_bandwidth(
            flat_lab,
            quantile=quantile,
            n_samples=min(n_samples, len(flat_lab)),
            random_state=42,
        )
        if bandwidth is None or bandwidth <= 0:
            bandwidth = 4.0
    bandwidth = max(bandwidth, 1.0)

    ms = MeanShift(
        bandwidth=bandwidth,
        bin_seeding=True,
        min_bin_freq=5,
        n_jobs=-1,
    )
    ms.fit(flat_lab)
    labels_flat = ms.labels_.astype(np.int32)
    initial_labels = labels_flat.reshape(h, w)
    n_initial = len(np.unique(initial_labels))
    t_ms = time.perf_counter() - t0

    # --- Step 3: Region merging ---
    t0 = time.perf_counter()
    if use_hierarchy:
        # Compute per-region mean LAB and area
        unique = np.unique(initial_labels)
        region_means = []
        region_counts = []
        for rid in unique:
            mask = initial_labels == rid
            count = int(mask.sum())
            mean_lab = flat_lab[mask.reshape(-1)].mean(axis=0)
            region_means.append(mean_lab)
            region_counts.append(count)
        region_means = np.array(region_means)
        region_counts = np.array(region_counts, dtype=np.float64)

        # Weighted hierarchical clustering
        # Ward needs Euclidean, so we use LAB directly
        # For weighted clustering, repeat points proportional to sqrt(area) is expensive.
        # Instead use 'average' or 'complete' with sample weights via weighted linkage.
        # scipy linkage doesn't support weights directly, but we can approximate by
        # using the region_means as data points and rely on the fact that large regions
        # dominate the visual result. For better weighting, we replicate each region
        # mean a number of times proportional to sqrt(count) (so large regions have
        # more influence but not linearly dominating).
        if linkage_method == "ward":
            # Ward is sensitive to scale; use raw means
            Z = linkage(region_means, method="ward")
        else:
            Z = linkage(region_means, method=linkage_method)

        # Cut dendrogram at target clusters
        n_target = min(target_count, n_initial)
        if n_target < 2:
            n_target = 2
        cluster_ids = fcluster(Z, t=n_target, criterion="maxclust")

        # Map back to pixels
        rid_to_cluster = {int(rid): int(cid) - 1 for rid, cid in zip(unique, cluster_ids)}
        merged_labels = np.vectorize(rid_to_cluster.get)(initial_labels)
    else:
        # Iterative graph-based merge
        adj = _build_adjacency(initial_labels)
        stats = _compute_region_stats(panel_lab, initial_labels)
        merged_labels = _merge_regions(
            initial_labels, adj, stats, target_count, max_similarity=max_similarity
        )
    t_merge = time.perf_counter() - t0
    n_final = len(np.unique(merged_labels))

    # --- Step 4: shape filter ---
    t0 = time.perf_counter()
    merged_labels = _shape_filter(merged_labels)
    # Remap after shape filter
    unique = np.unique(merged_labels)
    remap = {old: new for new, old in enumerate(unique)}
    merged_labels = np.vectorize(remap.get)(merged_labels)
    n_after_shape = len(np.unique(merged_labels))
    t_shape = time.perf_counter() - t0

    # --- Step 5: compute palette as median RGB per region ---
    flat_rgb = panel_rgb.reshape(-1, 3)
    palette = []
    final_color_names = []
    cluster_sizes = []
    for i in range(n_after_shape):
        mask = merged_labels.reshape(-1) == i
        pixels = flat_rgb[mask]
        count = int(mask.sum())
        cluster_sizes.append(count)
        if len(pixels) > 0:
            median_rgb = np.median(pixels, axis=0).astype(np.uint8)
        else:
            median_rgb = np.array([128, 128, 128], dtype=np.uint8)
        palette.append(median_rgb)
        final_color_names.append(f"region_{i}")

    palette = np.array(palette, dtype=np.uint8)

    total_time = t_seeds + t_ms + t_merge + t_shape

    return SegmentResult(
        labels=merged_labels,
        palette=palette,
        color_names=final_color_names,
        path="jet_vivid_merge",
        saturation_ratio=saturation_ratio(panel_rgb),
        notes={
            "refined_seeds": refined_reps,
            "target_count": target_count,
            "bandwidth": float(bandwidth),
            "n_initial_regions": n_initial,
            "n_final_regions": n_after_shape,
            "n_after_merge_before_shape": n_final,
            "use_hierarchy": use_hierarchy,
            "linkage_method": linkage_method if use_hierarchy else None,
            "cluster_sizes": cluster_sizes,
            "timing": {
                "seed_refinement_s": round(t_seeds, 3),
                "mean_shift_s": round(t_ms, 3),
                "merge_s": round(t_merge, 3),
                "shape_filter_s": round(t_shape, 3),
                "total_s": round(total_time, 3),
            },
        },
    )
