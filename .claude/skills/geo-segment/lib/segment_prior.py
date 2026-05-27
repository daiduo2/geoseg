"""Prior-constrained segmentation for geophysics interpretation panels (e019).

Key geological prior: "bands have regular boundaries and regions are large and continuous".
Instead of post-processing a noisy segmentation, this prior is embedded into the
segmentation itself via hierarchical region merging with a geological cost function.

Pipeline:
    1. Initial oversegmentation: Mean Shift with small bandwidth (quantile=0.05)
       to get ~20-50 color-homogeneous super-regions.
    2. Build region adjacency graph with node/edge attributes.
    3. Hierarchical merging with geological cost function:
       merge_cost(i,j) = w1 * color_distance(LAB)
                         - w2 * shared_boundary_length / perimeter(i)
                         - w3 * log(min(area_i, area_j))
                         + w4 * shape_irregularity(i∪j)
    4. Fast path: scipy.cluster.hierarchy on region features weighted by area.
    5. Map final labels back to pixels, apply shape filter, return SegmentResult.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.cluster.hierarchy import linkage, fcluster
from skimage.color import rgb2lab, lab2rgb
from skimage.measure import label, regionprops
from sklearn.cluster import MeanShift, estimate_bandwidth

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


# ---------------------------------------------------------------------------
# Geological prior cost weights (tunable)
# ---------------------------------------------------------------------------
W_COLOR = 1.0          # similar colors should merge
W_BOUNDARY = 0.5       # regions with long shared boundaries prefer merging
W_AREA = 0.3           # penalize keeping tiny regions
W_SHAPE = 0.2          # penalize irregular shapes (thin/elongated)


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


# ---------------------------------------------------------------------------
# Region adjacency graph with geological attributes
# ---------------------------------------------------------------------------

def _build_region_graph(labels: np.ndarray) -> dict[int, set[int]]:
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


def _compute_boundary_length(labels: np.ndarray, rid_a: int, rid_b: int) -> int:
    """Count 4-connected pixel pairs where one is in rid_a and the other in rid_b."""
    h, w = labels.shape
    count = 0
    for y in range(h):
        for x in range(w):
            if labels[y, x] != rid_a:
                continue
            for dx, dy in [(1, 0), (0, 1)]:
                nx, ny = x + dx, y + dy
                if nx < w and ny < h and labels[ny, nx] == rid_b:
                    count += 1
    return count


def _compute_region_stats(panel_lab: np.ndarray, labels: np.ndarray):
    """Compute per-region statistics needed for geological cost.

    Returns dict: region_id -> {
        "mean_lab": np.ndarray(3),
        "area": int,
        "centroid": (cx, cy),
        "perimeter": int,
        "bbox": (min_y, min_x, max_y, max_x),
    }
    """
    h, w = labels.shape
    unique = np.unique(labels)
    stats = {}
    for rid in unique:
        mask = labels == rid
        area = int(mask.sum())
        pixels = panel_lab[mask]
        mean_lab = pixels.mean(axis=0) if area > 0 else np.zeros(3)
        ys, xs = np.where(mask)
        centroid = (float(xs.mean()) if len(xs) else 0.0, float(ys.mean()) if len(ys) else 0.0)

        # Perimeter: pixels in region that have a different neighbor
        perim = 0
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
            "area": area,
            "centroid": centroid,
            "perimeter": perim,
            "bbox": (int(ys.min()) if len(ys) else 0, int(xs.min()) if len(xs) else 0,
                     int(ys.max()) if len(ys) else 0, int(xs.max()) if len(xs) else 0),
        }
    return stats


def _shape_irregularity(perimeter: float, area: float) -> float:
    """perimeter^2 / area.  Circle ~ 12.6, square ~ 16, thin line -> inf."""
    area = max(area, 1e-9)
    return (perimeter ** 2) / area


# ---------------------------------------------------------------------------
# Iterative graph-based merge with geological cost
# ---------------------------------------------------------------------------

def _iterative_geological_merge(
    labels: np.ndarray,
    panel_lab: np.ndarray,
    target_count: int,
    w_color: float = W_COLOR,
    w_boundary: float = W_BOUNDARY,
    w_area: float = W_AREA,
    w_shape: float = W_SHAPE,
    max_cost_percentile: float = 95.0,
) -> np.ndarray:
    """Iteratively merge regions using the geological prior cost function.

    Returns merged labels remapped to contiguous 0..k-1.

    The stopping criterion uses a dynamic percentile threshold on the cost
    distribution rather than a hard absolute value, because the cost scale
    varies with image size and region count.
    """
    out = labels.copy()
    current_stats = _compute_region_stats(panel_lab, out)
    current_adj = _build_region_graph(out)

    # Precompute boundary lengths for all adjacent pairs
    boundary_lengths: dict[tuple[int, int], int] = {}
    for a in current_adj:
        for b in current_adj[a]:
            if a < b:
                boundary_lengths[(a, b)] = _compute_boundary_length(out, a, b)

    def _compute_all_costs():
        costs = []
        for (a, b), bl in boundary_lengths.items():
            if a not in current_stats or b not in current_stats:
                continue
            sa = current_stats[a]
            sb = current_stats[b]

            color_dist = float(np.linalg.norm(sa["mean_lab"] - sb["mean_lab"]))
            perim_a = max(sa["perimeter"], 1e-9)
            boundary_bonus = bl / perim_a

            min_area = min(sa["area"], sb["area"])
            area_bonus = np.log(max(min_area, 1.0))

            # Shape irregularity of merged region (approximate via union)
            merged_perim = sa["perimeter"] + sb["perimeter"] - 2 * bl
            merged_area = sa["area"] + sb["area"]
            shape_irreg = _shape_irregularity(merged_perim, merged_area)

            cost = (
                w_color * color_dist
                - w_boundary * boundary_bonus
                - w_area * area_bonus
                + w_shape * shape_irreg
            )
            costs.append((cost, a, b))
        return costs

    while len(current_stats) > target_count:
        costs = _compute_all_costs()
        if not costs:
            break

        # Dynamic threshold: only merge if cost is below the given percentile
        all_cost_values = [c[0] for c in costs]
        threshold = np.percentile(all_cost_values, max_cost_percentile)

        # Find best pair below threshold
        valid = [(c, a, b) for c, a, b in costs if c <= threshold]
        if not valid:
            break
        valid.sort(key=lambda t: t[0])
        _, a, b = valid[0]
        if a > b:
            a, b = b, a

        # Merge b into a
        count_a = current_stats[a]["area"]
        count_b = current_stats[b]["area"]
        total = count_a + count_b
        current_stats[a]["mean_lab"] = (
            current_stats[a]["mean_lab"] * count_a + current_stats[b]["mean_lab"] * count_b
        ) / total
        current_stats[a]["area"] = total

        # Recompute perimeter of merged region
        shared = boundary_lengths.get((a, b), 0)
        current_stats[a]["perimeter"] = (
            current_stats[a]["perimeter"] + current_stats[b]["perimeter"] - 2 * shared
        )

        # Update adjacency
        for neighbor in list(current_adj[b]):
            if neighbor == a:
                continue
            current_adj[neighbor].discard(b)
            current_adj[neighbor].add(a)
            current_adj[a].add(neighbor)
            # Update boundary length
            bl_an = boundary_lengths.get((min(a, neighbor), max(a, neighbor)), 0)
            bl_bn = boundary_lengths.get((min(b, neighbor), max(b, neighbor)), 0)
            new_bl = bl_an + bl_bn
            boundary_lengths[(min(a, neighbor), max(a, neighbor))] = new_bl
        current_adj[a].discard(a)
        current_adj.pop(b, None)

        # Remove old boundary entries involving b
        keys_to_remove = [k for k in boundary_lengths if b in k]
        for k in keys_to_remove:
            boundary_lengths.pop(k, None)

        del current_stats[b]

        # Update pixels
        out[out == b] = a

    # Remap to contiguous 0..k-1
    unique = np.unique(out)
    remap = {old: new for new, old in enumerate(unique)}
    out = np.vectorize(remap.get)(out)
    return out


# ---------------------------------------------------------------------------
# Fast hierarchical path via scipy.cluster.hierarchy
# ---------------------------------------------------------------------------

def _hierarchical_merge(
    initial_labels: np.ndarray,
    panel_lab: np.ndarray,
    target_count: int,
    linkage_method: str = "ward",
) -> np.ndarray:
    """Use scipy.cluster.hierarchy on region features weighted by area.

    Features per region: [mean_L, mean_A, mean_B, centroid_x, centroid_y, area].
    """
    h, w = initial_labels.shape
    flat_lab = panel_lab.reshape(-1, 3)
    unique = np.unique(initial_labels)

    region_features = []
    region_counts = []
    for rid in unique:
        mask = initial_labels == rid
        count = int(mask.sum())
        ys, xs = np.where(mask)
        mean_lab = flat_lab[mask.reshape(-1)].mean(axis=0)
        centroid_x = float(xs.mean()) if len(xs) else 0.0
        centroid_y = float(ys.mean()) if len(ys) else 0.0
        # Normalize spatial coords to LAB scale for balanced clustering
        features = [
            mean_lab[0],
            mean_lab[1],
            mean_lab[2],
            centroid_x / max(w, 1) * 50.0,  # scale to ~LAB magnitude
            centroid_y / max(h, 1) * 50.0,
            np.log(max(count, 1.0)) * 5.0,   # area as weak feature
        ]
        region_features.append(features)
        region_counts.append(count)

    region_features = np.array(region_features, dtype=np.float64)
    region_counts = np.array(region_counts, dtype=np.float64)

    # Weighted linkage: replicate each region mean proportional to sqrt(area)
    # This gives large regions more influence without linear domination.
    weighted_data = []
    for feat, count in zip(region_features, region_counts):
        n_reps = max(1, int(np.sqrt(count) / 10))
        for _ in range(n_reps):
            weighted_data.append(feat)
    weighted_data = np.array(weighted_data, dtype=np.float64)

    Z = linkage(weighted_data, method=linkage_method)
    n_target = min(target_count, len(unique))
    if n_target < 2:
        n_target = 2
    cluster_ids = fcluster(Z, t=n_target, criterion="maxclust")

    # Map back: since we replicated points, take the cluster of the first replica
    rid_to_cluster = {}
    idx = 0
    for rid, count in zip(unique, region_counts):
        n_reps = max(1, int(np.sqrt(count) / 10))
        rid_to_cluster[int(rid)] = int(cluster_ids[idx]) - 1
        idx += n_reps

    merged_labels = np.vectorize(rid_to_cluster.get)(initial_labels)
    return merged_labels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def segment_jet_vivid_prior(
    panel_rgb: np.ndarray,
    reps: list[dict],
    max_auto_k: int = 3,
    bandwidth: float | None = None,
    quantile: float = 0.05,
    n_samples: int = 5000,
    use_iterative: bool = False,
    linkage_method: str = "ward",
    w_color: float = W_COLOR,
    w_boundary: float = W_BOUNDARY,
    w_area: float = W_AREA,
    w_shape: float = W_SHAPE,
) -> SegmentResult:
    """Prior-constrained segmentation for jet-vivid panels.

    Embeds the geological prior ("bands have regular boundaries and regions are
    large and continuous") directly into the segmentation via hierarchical
    region merging with a geological cost function.

    Parameters
    ----------
    panel_rgb : np.ndarray
        (H, W, 3) uint8 RGB image.
    reps : list[dict]
        VLM representative points.
    max_auto_k : int
        Maximum auto-detected seeds to add.
    bandwidth : float | None
        Mean Shift bandwidth. If None, estimated with small quantile (0.05).
    quantile : float
        Quantile for sklearn estimate_bandwidth (default 0.05 = many modes).
    n_samples : int
        Samples for bandwidth estimation.
    use_iterative : bool
        If True, use iterative graph-based merge with geological cost.
        If False (default), use fast scipy.cluster.hierarchy.
    linkage_method : str
        'ward', 'average', 'complete', 'single'.
    w_color, w_boundary, w_area, w_shape : float
        Weights for the geological cost function.

    Returns
    -------
    SegmentResult
    """
    h, w, _ = panel_rgb.shape
    panel_lab = rgb2lab(panel_rgb)
    flat_lab = panel_lab.reshape(-1, 3)

    # --- Step 1: refine seeds from reps ---
    t0 = time.perf_counter()
    refined_seeds_arr, refined_reps, color_names = _refine_seeds_from_reps(
        panel_rgb, reps, max_auto_k=max_auto_k
    )
    target_count = len(refined_seeds_arr) + max_auto_k
    t_seeds = time.perf_counter() - t0

    # --- Step 2: Mean Shift oversegmentation with small bandwidth ---
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

    # --- Step 3: Hierarchical merging with geological prior ---
    t0 = time.perf_counter()
    if use_iterative:
        merged_labels = _iterative_geological_merge(
            initial_labels,
            panel_lab,
            target_count=target_count,
            w_color=w_color,
            w_boundary=w_boundary,
            w_area=w_area,
            w_shape=w_shape,
        )
    else:
        merged_labels = _hierarchical_merge(
            initial_labels,
            panel_lab,
            target_count=target_count,
            linkage_method=linkage_method,
        )
    n_after_merge = len(np.unique(merged_labels))
    t_merge = time.perf_counter() - t0

    # --- Step 4: shape filter ---
    t0 = time.perf_counter()
    merged_labels = _shape_filter(merged_labels)
    unique = np.unique(merged_labels)
    remap = {old: new for new, old in enumerate(unique)}
    merged_labels = np.vectorize(remap.get)(merged_labels)
    n_after_shape = len(np.unique(merged_labels))
    t_shape = time.perf_counter() - t0

    # --- Step 5: compute palette as median RGB per region ---
    flat_rgb = panel_rgb.reshape(-1, 3)
    palette = []
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

    palette = np.array(palette, dtype=np.uint8)

    total_time = t_seeds + t_ms + t_merge + t_shape

    return SegmentResult(
        labels=merged_labels,
        palette=palette,
        color_names=color_names[:n_after_shape] if len(color_names) >= n_after_shape else color_names + [f"region_{i}" for i in range(len(color_names), n_after_shape)],
        path="jet_vivid_prior",
        saturation_ratio=saturation_ratio(panel_rgb),
        notes={
            "refined_seeds": refined_reps,
            "target_count": target_count,
            "bandwidth": float(bandwidth),
            "n_initial_regions": n_initial,
            "n_after_merge": n_after_merge,
            "n_final_regions": n_after_shape,
            "use_iterative": use_iterative,
            "linkage_method": linkage_method if not use_iterative else None,
            "weights": {
                "w_color": w_color,
                "w_boundary": w_boundary,
                "w_area": w_area,
                "w_shape": w_shape,
            },
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
