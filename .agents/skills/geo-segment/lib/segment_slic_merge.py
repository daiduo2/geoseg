"""SLIC superpixel + graph merging segmentation for jet-vivid panels (e013).

Pipeline:
    1. Run SLIC to get spatially coherent superpixels (respecting color boundaries)
    2. Compute mean RGB and mean LAB for each superpixel
    3. Assign each superpixel to nearest seed in LAB space (preserves geological meaning)
    4. Build superpixel adjacency graph
    5. Merge small/isolated superpixel clusters into neighbours by color similarity
    6. All pixels in a superpixel get the cluster label
    7. Apply shape filter post-processing

Key advantage over pixel-level methods: spatial coherence from SLIC boundaries
reduces salt-and-pepper noise while nearest-seed assignment preserves layer accuracy.
"""

from __future__ import annotations

import numpy as np
from skimage.color import rgb2lab
from skimage.segmentation import slic
from scipy import ndimage

from lib.segment import (
    SegmentResult,
    saturation_ratio,
    _estimate_background_color,
    _is_background_v2,
    _cv_seeds,
    _find_pixel_for_color,
    _spiral_search,
    _erode_internal_point,
    _scan_for_missing_colors,
    _parse_count_from_tag,
    _shape_filter,
    _online_color_groups,
    _histogram_peaks,
)


def segment_jet_vivid_slic_merge(
    panel_rgb: np.ndarray,
    reps: list[dict],
    max_auto_k: int = 3,
    compactness: int = 10,
) -> SegmentResult:
    """SLIC superpixel + graph merging segmentation for vivid jet-colormap panels.

    Parameters
    ----------
    panel_rgb : np.ndarray
        (H, W, 3) uint8 cropped panel.
    reps : list[dict]
        VLM representative points, each with ``color_name`` and
        ``representative_point`` {"x", "y"}.
    max_auto_k : int
        Maximum extra seeds to auto-detect from CV / scan.
    compactness : int
        SLIC compactness (5 = colour-sensitive, 10 = balanced, 20 = spatially regular).

    Returns
    -------
    SegmentResult
    """
    if not reps:
        raise ValueError("jet_vivid_slic_merge path requires at least one rep")

    h, w, _ = panel_rgb.shape
    panel_lab = rgb2lab(panel_rgb)
    bg_rgb = _estimate_background_color(panel_rgb)
    min_auto_count = max(10, h * w // 3000)

    # --- Step 1: SLIC superpixels ---
    n_segments = max(200, h * w // 500)
    sp_labels = slic(
        panel_rgb,
        n_segments=n_segments,
        compactness=compactness,
        start_label=0,
        channel_axis=2,
        convert2lab=False,  # we handle LAB ourselves
    )
    n_superpixels = int(sp_labels.max()) + 1

    # --- Step 2: Compute mean RGB and mean LAB per superpixel ---
    sp_mean_rgb = np.zeros((n_superpixels, 3), dtype=np.float32)
    sp_mean_lab = np.zeros((n_superpixels, 3), dtype=np.float32)
    sp_counts = np.zeros(n_superpixels, dtype=np.int64)

    flat_sp = sp_labels.ravel()
    flat_rgb = panel_rgb.reshape(-1, 3).astype(np.float32)
    flat_lab = panel_lab.reshape(-1, 3).astype(np.float32)

    for i in range(n_superpixels):
        mask = flat_sp == i
        count = int(mask.sum())
        sp_counts[i] = count
        if count > 0:
            sp_mean_rgb[i] = flat_rgb[mask].mean(axis=0)
            sp_mean_lab[i] = flat_lab[mask].mean(axis=0)

    # --- CV fallback seeds (computed once) ---
    cv_seeds_rgb, cv_tags = _cv_seeds(panel_rgb, k=len(reps))
    used_cv_indices: set[int] = set()

    def _bg_check(rgb: np.ndarray) -> bool:
        return _is_background_v2(rgb, bg_rgb)

    # --- Stage 1: raw VLM reps -> rough nearest-neighbour labels ---
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

    # --- Stage 2: refine each seed with multi-source fallback ---
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

            # Try local erosion around the VLM point first
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

            # If local erosion failed or gave background, spiral search nearby
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

    # --- Stage 2b: auto-k --- detect missing colors from unused CV seeds ---
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

    # --- Stage 2c: supplement with full-image scan if still under max_auto_k ---
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

    # Merge VLM + auto seeds
    if auto_seeds:
        refined_seeds = refined_seeds + auto_seeds
        refined_reps = refined_reps + auto_reps
        color_names = color_names + [r["name"] for r in auto_reps]

    refined_seeds_arr = np.array(refined_seeds, dtype=np.uint8)
    seeds_lab = rgb2lab(refined_seeds_arr[np.newaxis, ...])[0]

    # --- Step 3: Assign each superpixel to nearest seed in LAB space ---
    # This preserves the geological meaning of seeds (unlike K-means which shifts centroids)
    sp_to_cluster = {}
    for sp_idx in range(n_superpixels):
        if sp_counts[sp_idx] == 0:
            continue
        dists = np.linalg.norm(seeds_lab - sp_mean_lab[sp_idx], axis=1)
        sp_to_cluster[sp_idx] = int(dists.argmin())

    # --- Step 4: Build superpixel adjacency graph and merge small clusters ---
    # Find adjacent superpixels
    adjacency = {i: set() for i in range(n_superpixels)}
    for y in range(h):
        for x in range(w):
            sp = sp_labels[y, x]
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    nsp = sp_labels[ny, nx]
                    if nsp != sp:
                        adjacency[sp].add(nsp)
                        adjacency[nsp].add(sp)

    # Compute cluster sizes (in pixels)
    cluster_sizes = {}
    for sp_idx, cl in sp_to_cluster.items():
        cluster_sizes[cl] = cluster_sizes.get(cl, 0) + sp_counts[sp_idx]

    # Merge small clusters: if a cluster is tiny (< 1% of image), merge its superpixels
    # into the adjacent cluster with most similar color
    min_cluster_size = max(50, h * w // 100)
    merged = True
    merge_round = 0
    while merged and merge_round < 5:
        merged = False
        merge_round += 1
        # Recompute cluster sizes
        cluster_sizes = {}
        for sp_idx, cl in sp_to_cluster.items():
            cluster_sizes[cl] = cluster_sizes.get(cl, 0) + sp_counts[sp_idx]

        for sp_idx, cl in list(sp_to_cluster.items()):
            # Check if this superpixel's cluster is too small
            if cluster_sizes.get(cl, 0) >= min_cluster_size:
                continue

            # Find adjacent clusters
            neigh_clusters = {}
            for nsp in adjacency.get(sp_idx, set()):
                ncl = sp_to_cluster.get(nsp)
                if ncl is not None and ncl != cl:
                    neigh_clusters[ncl] = neigh_clusters.get(ncl, 0) + sp_counts.get(nsp, 0)

            if not neigh_clusters:
                continue

            # Pick adjacent cluster with most similar mean LAB color
            best_cl = None
            best_dist = float("inf")
            for ncl in neigh_clusters:
                # Compute mean LAB of the adjacent cluster
                ncl_sp = [s for s, c in sp_to_cluster.items() if c == ncl]
                if not ncl_sp:
                    continue
                ncl_mean = np.mean([sp_mean_lab[s] for s in ncl_sp], axis=0)
                dist = float(np.linalg.norm(sp_mean_lab[sp_idx] - ncl_mean))
                if dist < best_dist:
                    best_dist = dist
                    best_cl = ncl

            if best_cl is not None:
                sp_to_cluster[sp_idx] = best_cl
                merged = True

    # --- Step 5: Assign pixel labels from superpixel clusters ---
    labels = np.zeros((h, w), dtype=np.int32)
    for sp_idx in range(n_superpixels):
        mask = sp_labels == sp_idx
        if mask.any():
            labels[mask] = sp_to_cluster.get(sp_idx, 0)

    # --- Step 6: shape filter post-processing ---
    labels = _shape_filter(labels)

    return SegmentResult(
        labels=labels,
        palette=refined_seeds_arr,
        color_names=color_names,
        path="jet_vivid_slic_merge",
        saturation_ratio=saturation_ratio(panel_rgb),
        notes={
            "reps_refined": refined_reps,
            "cv_seeds": cv_seeds_rgb.tolist() if len(cv_seeds_rgb) else [],
            "bg_rgb": bg_rgb.tolist(),
            "auto_k_added": len(auto_seeds),
            "slic_params": {
                "n_segments": n_segments,
                "compactness": compactness,
                "n_superpixels": n_superpixels,
            },
            "merge_rounds": merge_round,
        },
    )


def _label_by_nearest(panel_lab: np.ndarray, palette_lab: np.ndarray) -> np.ndarray:
    """Label each pixel by index of nearest palette entry in LAB."""
    h, w, _ = panel_lab.shape
    flat = panel_lab.reshape(-1, 3)
    d2 = ((flat[:, None, :] - palette_lab[None, :, :]) ** 2).sum(axis=2)
    return d2.argmin(axis=1).reshape(h, w).astype(np.int32)
