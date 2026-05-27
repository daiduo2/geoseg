"""Edge-guided K-means segmentation for jet-vivid panels (e014).

K-means alone struggles because geological layer boundaries are gradual color
transitions.  This module first detects edges/boundaries via multi-channel Sobel
in LAB space, then uses the edge map as a spatial constraint during clustering.
Boundary pixels are penalised for crossing edges, which snaps boundaries to
actual layer transitions.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy import ndimage
from skimage.color import rgb2lab
from skimage.filters import sobel
from skimage.measure import label, regionprops

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
)


def _label_by_nearest(panel_lab: np.ndarray, palette_lab: np.ndarray) -> np.ndarray:
    """Label each pixel by index of nearest palette entry in LAB."""
    h, w, _ = panel_lab.shape
    flat = panel_lab.reshape(-1, 3)
    d2 = ((flat[:, None, :] - palette_lab[None, :, :]) ** 2).sum(axis=2)
    return d2.argmin(axis=1).reshape(h, w).astype(np.int32)


def _compute_edge_map(
    panel_lab: np.ndarray,
    method: str = "canny",
    canny_sigma: float = 2.0,
    canny_low: float = 0.05,
    canny_high: float = 0.15,
    sobel_percentile: float = 85.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect geological layer boundaries.

    Two methods are supported:
    * ``canny`` (default): Canny edge detector on the L channel. Produces thin,
      connected boundaries that align well with gradual layer transitions.
    * ``sobel``: Multi-channel Sobel gradient magnitude in LAB space. More
      sensitive to faint edges but produces thicker, noisier responses.

    Returns
    -------
    gradient : (H, W) float32
        Edge strength map (Canny returns binary 0/1, Sobel returns gradient mag).
    edge_mask : (H, W) bool
        True at detected boundary pixels.
    """
    h, w = panel_lab.shape[:2]

    if method == "canny":
        from skimage.feature import canny
        from skimage.morphology import closing, disk

        l_norm = panel_lab[..., 0] / 100.0  # L ranges ~0-100
        edge_mask = canny(l_norm, sigma=canny_sigma, low_threshold=canny_low, high_threshold=canny_high)
        # Close small gaps in layer boundaries while preserving thin edges
        edge_mask = closing(edge_mask, footprint=disk(1))
        gradient = edge_mask.astype(np.float32)
    else:
        gradient = np.zeros((h, w), dtype=np.float32)
        for c in range(3):
            gradient += sobel(panel_lab[..., c]) ** 2
        gradient = np.sqrt(gradient)
        threshold = np.percentile(gradient, sobel_percentile)
        edge_mask = gradient > threshold

    return gradient, edge_mask


def _edge_guided_kmeans(
    panel_lab: np.ndarray,
    seeds_lab: np.ndarray,
    edge_mask: np.ndarray,
    edge_weight: float = 0.3,
    sigma: float = 4.0,
    max_iter: int = 30,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Edge-guided K-means: standard K-means + selective boundary-pixel snapping.

    Algorithm:
        1. Run standard K-means in LAB space with ``seeds_lab`` as initial
           centroids.
        2. Label each connected component of the non-edge regions.
        3. Map each region to its dominant cluster.
        4. For pixels near an edge that are *ambiguous* (LAB distance to their
           current centroid is within ``edge_weight`` fraction of the distance
           to the second-best centroid), snap them to the dominant cluster of
           their region.  This only affects uncertain boundary pixels, leaving
           confident interior pixels untouched.

    Parameters
    ----------
    panel_lab : (H, W, 3) float64
        Image in LAB space.
    seeds_lab : (k, 3) float64
        Initial centroids.
    edge_mask : (H, W) bool
        True at detected boundary pixels.
    edge_weight : float
        Ambiguity threshold.  A pixel is snapped if:
        ``d_best / d_second < (1 - edge_weight)``.
        ``edge_weight=0`` disables snapping entirely.
    sigma : float
        Distance from edge (pixels) within which snapping is considered.
    max_iter : int
    tol : float
        Relative change in centroids for convergence.

    Returns
    -------
    centroids : (k, 3) float64
    labels : (H, W) int32
    """
    h, w = panel_lab.shape[:2]
    flat_lab = panel_lab.reshape(-1, 3)
    k = seeds_lab.shape[0]

    # --- Step 1: Standard K-means ---
    centroids, labels_flat = kmeans2(flat_lab, seeds_lab, minit="matrix", iter=max_iter, thresh=tol)
    labels = labels_flat.reshape(h, w).astype(np.int32)
    centroids = centroids.astype(np.float64)

    if edge_weight <= 0 or not edge_mask.any():
        return centroids, labels

    # --- Step 2: Build region-to-cluster map from non-edge connected components ---
    dist_to_edge = ndimage.distance_transform_edt(~edge_mask).astype(np.float32)
    snap_zone = dist_to_edge <= sigma

    regions = label(~edge_mask, connectivity=2)
    region_to_cluster: dict[int, int] = {}
    region_props = regionprops(regions)
    for rp in region_props:
        rid = rp.label
        mask = regions == rid
        vals, counts = np.unique(labels[mask], return_counts=True)
        region_to_cluster[rid] = int(vals[counts.argmax()])

    # --- Step 3: Compute ambiguity for every pixel ---
    # Distance from each pixel to each centroid
    d_all = np.linalg.norm(flat_lab[:, None, :] - centroids[None, :, :], axis=2)
    d_sorted = np.partition(d_all, kth=1, axis=1)  # two smallest distances
    d_best = d_sorted[:, 0]
    d_second = d_sorted[:, 1]

    # Ambiguity ratio: close to 1 means pixel is on the fence between two clusters
    ambiguity = d_best / (d_second + 1e-9)
    # Pixels with ambiguity > (1 - edge_weight) are snapped
    # (e.g. edge_weight=0.3 -> snap if d_best/d_second > 0.7)
    ambiguous = ambiguity > (1.0 - edge_weight)
    ambiguous = ambiguous.reshape(h, w)

    # --- Step 4: Snap ambiguous pixels in the snap zone ---
    labels_snapped = labels.copy()
    candidates = snap_zone & ambiguous
    snap_y, snap_x = np.where(candidates)
    for y, x in zip(snap_y, snap_x):
        rid = regions[y, x]
        if rid in region_to_cluster:
            labels_snapped[y, x] = region_to_cluster[rid]

    return centroids, labels_snapped


def segment_jet_vivid_edge_guided(
    panel_rgb: np.ndarray,
    reps: list[dict],
    max_auto_k: int = 3,
    edge_weight: float = 0.3,
    edge_percentile: float = 90.0,
    sigma: float = 4.0,
) -> SegmentResult:
    """Edge-guided K-means segmentation for vivid jet-colormap panels.

    Reuses the seed-refinement + auto-k logic from ``segment_jet_vivid``
    (Stages 1-2) but replaces the nearest-median classifier with an edge-guided
    K-means in LAB space.

    Parameters
    ----------
    panel_rgb : np.ndarray
        (H, W, 3) uint8 cropped panel.
    reps : list[dict]
        VLM representative points.
    max_auto_k : int
        Maximum extra seeds to auto-detect.
    edge_weight : float
        Spatial penalty strength (0 = standard K-means).
    edge_percentile : float
        Percentile threshold for edge detection (85-95 typical).
    sigma : float
        Gaussian fall-off width for the edge penalty (3-5 pixels typical).

    Returns
    -------
    SegmentResult
    """
    if not reps:
        raise ValueError("jet_vivid_edge_guided path requires at least one rep")

    h, w, _ = panel_rgb.shape
    panel_lab = rgb2lab(panel_rgb)
    bg_rgb = _estimate_background_color(panel_rgb)
    min_auto_count = max(10, h * w // 3000)

    # --- Edge detection ---
    gradient, edge_mask = _compute_edge_map(
        panel_lab,
        method="canny",
        canny_sigma=1.0,
        canny_low=0.02,
        canny_high=0.1,
    )

    # --- CV fallback seeds ---
    cv_seeds_rgb, cv_tags = _cv_seeds(panel_rgb, k=len(reps))
    used_cv_indices: set[int] = set()

    def _bg_check(rgb: np.ndarray) -> bool:
        return _is_background_v2(rgb, bg_rgb)

    # --- Stage 1: raw VLM reps → rough nearest-neighbour labels ---
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

    # --- Stage 2b: auto-k ---
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

    # --- Stage 2c: supplement with full-image scan ---
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

    # --- Stage 3: Edge-guided K-means ---
    centroids, labels = _edge_guided_kmeans(
        panel_lab,
        seeds_lab,
        edge_mask,
        edge_weight=edge_weight,
        sigma=sigma,
    )

    # --- Stage 4: shape filter ---
    labels = _shape_filter(labels)

    return SegmentResult(
        labels=labels,
        palette=refined_seeds_arr,
        color_names=color_names,
        path="jet_vivid_edge_guided",
        saturation_ratio=saturation_ratio(panel_rgb),
        notes={
            "reps_refined": refined_reps,
            "cv_seeds": cv_seeds_rgb.tolist() if len(cv_seeds_rgb) else [],
            "bg_rgb": bg_rgb.tolist(),
            "auto_k_added": len(auto_seeds),
            "edge_weight": edge_weight,
            "edge_percentile": edge_percentile,
            "sigma": sigma,
            "edge_pixels_pct": float(edge_mask.mean() * 100),
            "centroids_lab": centroids.tolist(),
        },
    )
