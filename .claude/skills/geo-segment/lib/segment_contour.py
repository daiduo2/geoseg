"""Active contour (snakes / level set) segmentation for geophysics panels.

Builds on the jet_vivid seed-refinement pipeline but replaces the final
nearest-median labeling with active contour evolution so boundaries snap
to actual colour-gradient edges.

Public API:
    segment_jet_vivid_contour(panel_rgb, reps, max_auto_k=3) -> SegmentResult
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.color import rgb2lab
from skimage.filters import sobel, gaussian
from skimage.morphology import disk, erosion, dilation, opening, closing
from skimage.segmentation import (
    morphological_geodesic_active_contour as mgac,
    chan_vese,
    find_boundaries,
)

from lib.segment import (
    SegmentResult,
    saturation_ratio,
    _estimate_background_color,
    _is_background_v2,
    _spiral_search,
    _find_pixel_for_color,
    _erode_internal_point,
    _label_by_nearest,
    _shape_filter,
    _cv_seeds,
    _parse_count_from_tag,
    _scan_for_missing_colors,
    _nearest_median,
)


def _edge_map_from_lab(panel_lab: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Compute a normalised edge-strength map from LAB gradient magnitude."""
    h, w = panel_lab.shape[:2]
    gradient = np.zeros((h, w), dtype=np.float32)
    for c in range(3):
        gradient += sobel(panel_lab[..., c]) ** 2
    gradient = np.sqrt(gradient)
    # Gaussian smoothing to suppress high-freq noise
    if sigma > 0:
        gradient = gaussian(gradient, sigma=sigma)
    # Normalise to [0, 1]
    gmax = gradient.max()
    if gmax > 0:
        gradient /= gmax
    return gradient


def _refine_label_with_chan_vese(
    binary_mask: np.ndarray,
    panel_gray: np.ndarray,
    iterations: int = 100,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
) -> np.ndarray:
    """Run Chan-Vese on a single binary mask to shrink/grow to fit edges.

    Returns a refined binary mask (bool).
    """
    # chan_vese expects float image in [0, 1]
    img = panel_gray.astype(np.float32) / 255.0
    init = binary_mask.astype(np.float32)
    cv = chan_vese(
        img,
        mu=0.25,
        lambda1=lambda1,
        lambda2=lambda2,
        tol=1e-3,
        max_num_iter=iterations,
        dt=0.5,
        init_level_set=init,
    )
    return cv > 0.5


def _refine_label_with_mgac(
    binary_mask: np.ndarray,
    edge_map: np.ndarray,
    iterations: int = 25,
    balloon: float = -0.5,
    threshold: float = 0.5,
    smoothing: int = 1,
) -> np.ndarray:
    """Run morphological geodesic active contour on a single label mask.

    The snake evolves along the edge map to snap boundaries to nearby edges.
    balloon < 0 shrinks, > 0 grows; we use slight shrink to let edges pull
    the boundary inward.
    """
    init = binary_mask.astype(np.float32)
    # Invert edge map: mgac expects high values = strong edges (stopping)
    gimg = 1.0 - edge_map
    # Pre-smooth the level set
    init = gaussian(init, sigma=1.0)
    out = mgac(
        gimg,
        num_iter=iterations,
        init_level_set=init,
        threshold=threshold,
        balloon=balloon,
        smoothing=smoothing,
    )
    return out > 0.5


def _balloon_expand_from_mask(
    panel_lab: np.ndarray,
    seed_mask: np.ndarray,
    edge_map: np.ndarray,
    max_iter: int = 50,
    color_tol: float = 25.0,
) -> np.ndarray:
    """Simple balloon expansion from a seed mask with edge + color stopping.

    At each iteration, dilate the mask by 1 pixel.  Newly added pixels are
    kept only if:
      - edge_map < 0.4  (not on a strong edge)
      - LAB distance to seed mean < color_tol * 1.5
    Pixels on strong edges are rejected, letting the boundary "stick" to edges.
    """
    h, w = panel_lab.shape[:2]
    mask = seed_mask.copy()
    seed_lab_mean = panel_lab[seed_mask].mean(axis=0)

    for _ in range(max_iter):
        dilated = dilation(mask, footprint=disk(1))
        frontier = dilated & ~mask
        if not frontier.any():
            break

        ys, xs = np.where(frontier)
        frontier_lab = panel_lab[ys, xs]
        color_dists = np.linalg.norm(frontier_lab - seed_lab_mean, axis=1)
        edge_vals = edge_map[ys, xs]

        keep = (edge_vals < 0.4) & (color_dists < color_tol * 1.5)
        if not keep.any():
            break

        mask[ys[keep], xs[keep]] = True

    return mask


def _active_contour_labels(
    panel_rgb: np.ndarray,
    panel_lab: np.ndarray,
    seeds_lab: np.ndarray,
    seeds_xy: list[tuple[int, int]],
    method: str = "mgac",
    median_size: int = 3,
) -> np.ndarray:
    """Multi-label segmentation via active contour refinement.

    1. Start with nearest-neighbor labels (like _nearest_median without median).
    2. For each label, extract binary mask, run chosen contour method.
    3. Recombine refined masks (pixel goes to mask with largest original overlap).
    4. Apply light median filter + shape filter for cleanup.

    Parameters
    ----------
    method : str
        "mgac"  – morphological geodesic active contour (default)
        "chan_vese" – Chan-Vese level set per label
        "balloon" – simple dilation with edge stopping
    """
    h, w = panel_rgb.shape[:2]
    k = len(seeds_xy)

    # Initial labels via nearest seed in LAB
    labels = _label_by_nearest(panel_lab, seeds_lab)

    # Edge map shared by all methods
    edge_map = _edge_map_from_lab(panel_lab, sigma=1.0)

    # Gray image for chan_vese
    gray = panel_rgb.mean(axis=2)

    refined_masks = []
    for i in range(k):
        mask = labels == i
        if not mask.any():
            refined_masks.append(mask)
            continue

        # Erode slightly to get a robust interior seed
        seed = erosion(mask, footprint=disk(2))
        if not seed.any():
            seed = mask

        if method == "mgac":
            refined = _refine_label_with_mgac(
                seed, edge_map, iterations=30, balloon=-0.3, threshold=0.5, smoothing=1
            )
        elif method == "chan_vese":
            # Use the full mask (not just eroded seed) as init for Chan-Vese
            # so it can both shrink and grow
            refined = _refine_label_with_chan_vese(
                mask, gray, iterations=80, lambda1=1.0, lambda2=1.0
            )
        elif method == "balloon":
            refined = _balloon_expand_from_mask(
                panel_lab, seed, edge_map, max_iter=40, color_tol=20.0
            )
        else:
            raise ValueError(f"Unknown contour method: {method}")

        refined_masks.append(refined)

    # Recombine: assign each pixel to the refined mask that originally had
    # the strongest claim (largest overlap with initial label region).
    # Simple approach: for pixels claimed by multiple masks, use nearest seed.
    combined = np.full((h, w), -1, dtype=np.int32)
    for y in range(h):
        for x in range(w):
            claims = [i for i, m in enumerate(refined_masks) if m[y, x]]
            if len(claims) == 1:
                combined[y, x] = claims[0]
            elif len(claims) > 1:
                # Tie-break by nearest seed in LAB
                d2 = ((panel_lab[y, x] - seeds_lab) ** 2).sum(axis=1)
                combined[y, x] = int(d2.argmin())
            else:
                # Unclaimed: nearest seed fallback
                d2 = ((panel_lab[y, x] - seeds_lab) ** 2).sum(axis=1)
                combined[y, x] = int(d2.argmin())

    # Light median filter for spatial coherence
    if median_size > 1:
        combined = ndimage.median_filter(combined, size=median_size)

    return combined


def segment_jet_vivid_contour(
    panel_rgb: np.ndarray,
    reps: list[dict],
    max_auto_k: int = 3,
    contour_method: str = "mgac",
) -> SegmentResult:
    """Multi-source seeded segmentation with active contour boundary refinement.

    Reuses the full seed-refinement pipeline from ``segment_jet_vivid``
    (raw VLM reps → spiral search / erosion / auto-k) but replaces the
    final ``_nearest_median`` labeling with active contour evolution so
    boundaries align with actual colour-gradient edges.

    Parameters
    ----------
    reps : list[dict]
        VLM color zones, each with ``color_name`` and ``representative_point``.
    max_auto_k : int
        Maximum auto-detected seeds to add (same as segment_jet_vivid).
    contour_method : str
        ``"mgac"`` (default), ``"chan_vese"``, or ``"balloon"``.
    """
    if not reps:
        raise ValueError("jet_vivid_contour path requires at least one rep")

    h, w, _ = panel_rgb.shape
    panel_lab = rgb2lab(panel_rgb)
    bg_rgb = _estimate_background_color(panel_rgb)
    min_auto_count = max(10, h * w // 3000)

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

    # --- Stage 2: refine each seed (identical to segment_jet_vivid) ---
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

    # --- Stage 2c: full-image scan supplement ---
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
    seeds_lab = rgb2lab(refined_seeds_arr[np.newaxis, ...])[0]
    seeds_xy = [(rep["internal_x"], rep["internal_y"]) for rep in refined_reps]

    # --- Stage 3: Active contour refinement ---
    labels = _active_contour_labels(
        panel_rgb,
        panel_lab,
        seeds_lab,
        seeds_xy,
        method=contour_method,
        median_size=3,
    )

    # --- Stage 4: shape filter ---
    labels = _shape_filter(labels)

    return SegmentResult(
        labels=labels,
        palette=refined_seeds_arr,
        color_names=color_names,
        path="jet_vivid_contour",
        saturation_ratio=saturation_ratio(panel_rgb),
        notes={
            "reps_refined": refined_reps,
            "cv_seeds": cv_seeds_rgb.tolist() if len(cv_seeds_rgb) else [],
            "bg_rgb": bg_rgb.tolist(),
            "auto_k_added": len(auto_seeds),
            "contour_method": contour_method,
        },
    )
