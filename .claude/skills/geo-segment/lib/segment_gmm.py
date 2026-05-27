"""Gaussian Mixture Model (GMM) probabilistic segmentation for jet-vivid panels (e011).

Unlike K-means (hard assignment), GMM models each layer as a Gaussian distribution
in LAB space. This naturally handles color gradations within layers and provides
probabilistic assignments.

Pipeline:
    1. Reuse Stages 1-2 (seed refinement) and Stages 2b-2c (auto-k) from segment_jet_vivid.
    2. Convert panel to LAB space.
    3. Fit sklearn.mixture.GaussianMixture with refined seeds as initial means.
    4. Predict labels for all pixels.
    5. Compute palette as median RGB of each component.
    6. Apply _shape_filter() post-processing.
    7. Return SegmentResult.
"""

from __future__ import annotations

import time
import numpy as np
from skimage.color import rgb2lab
from sklearn.mixture import GaussianMixture

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


def _label_by_nearest(panel_lab: np.ndarray, palette_lab: np.ndarray) -> np.ndarray:
    """Label each pixel by index of nearest palette entry in LAB."""
    h, w, _ = panel_lab.shape
    flat = panel_lab.reshape(-1, 3)
    d2 = ((flat[:, None, :] - palette_lab[None, :, :]) ** 2).sum(axis=2)
    return d2.argmin(axis=1).reshape(h, w).astype(np.int32)


def segment_jet_vivid_gmm(
    panel_rgb: np.ndarray,
    reps: list[dict],
    max_auto_k: int = 3,
    covariance_type: str = "full",
    max_samples: int = 50000,
    seed: int = 42,
) -> SegmentResult:
    """GMM segmentation for vivid jet-colormap panels.

    Reuses the seed-refinement logic from ``segment_jet_vivid`` (Stages 1-2)
    but replaces the nearest-median / K-means classifier with a global
    Gaussian Mixture Model in LAB space.

    Parameters
    ----------
    panel_rgb : np.ndarray
        (H, W, 3) uint8 cropped panel.
    reps : list[dict]
        VLM representative points, each with ``color_name`` and
        ``representative_point`` {"x", "y"}.
    max_auto_k : int
        Maximum extra seeds to auto-detect from CV / scan.
    covariance_type : str
        GMM covariance type: "full", "tied", "diag", or "spherical".
    max_samples : int
        Maximum number of pixels to sample for fitting GMM (for speed).
        All pixels are still predicted after fitting.
    seed : int
        Random seed for GMM and pixel sampling.

    Returns
    -------
    SegmentResult
    """
    if not reps:
        raise ValueError("jet_vivid_gmm path requires at least one rep")

    h, w, _ = panel_rgb.shape
    panel_lab = rgb2lab(panel_rgb)
    bg_rgb = _estimate_background_color(panel_rgb)
    min_auto_count = max(10, h * w // 3000)

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

    # --- Stage 2b: auto-k — detect missing colors from unused CV seeds ---
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

    # --- Stage 3: GMM in LAB space ---
    flat_lab = panel_lab.reshape(-1, 3)
    n_components = len(refined_seeds)

    # Subsample for speed if panel is large
    n_pixels = flat_lab.shape[0]
    if n_pixels > max_samples:
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(n_pixels, max_samples, replace=False)
        sample_lab = flat_lab[sample_idx]
    else:
        sample_lab = flat_lab

    # Initialize GMM means with refined seeds
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        means_init=seeds_lab,
        random_state=seed,
        max_iter=200,
        tol=1e-3,
        n_init=1,
    )

    t0 = time.perf_counter()
    gmm.fit(sample_lab)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    labels_flat = gmm.predict(flat_lab)
    predict_time = time.perf_counter() - t0

    labels = labels_flat.reshape(h, w).astype(np.int32)

    # --- Stage 4: Compute palette as median RGB of each component ---
    palette_rgb = []
    for i in range(n_components):
        mask = labels_flat == i
        if mask.sum() > 0:
            median_rgb = np.median(flat_lab[mask], axis=0)
            # Convert LAB median back to RGB for palette
            from skimage.color import lab2rgb
            median_rgb = (lab2rgb(median_rgb[np.newaxis, np.newaxis, :])[0, 0] * 255).clip(0, 255).astype(np.uint8)
        else:
            median_rgb = refined_seeds_arr[i]
        palette_rgb.append(median_rgb)
    palette_rgb = np.array(palette_rgb, dtype=np.uint8)

    # --- Stage 5: shape filter merges thin 1-D noise ---
    labels = _shape_filter(labels)

    return SegmentResult(
        labels=labels,
        palette=palette_rgb,
        color_names=color_names,
        path="jet_vivid_gmm",
        saturation_ratio=saturation_ratio(panel_rgb),
        notes={
            "reps_refined": refined_reps,
            "cv_seeds": cv_seeds_rgb.tolist() if len(cv_seeds_rgb) else [],
            "bg_rgb": bg_rgb.tolist(),
            "auto_k_added": len(auto_seeds),
            "covariance_type": covariance_type,
            "gmm_converged": bool(gmm.converged_),
            "gmm_n_iter": int(gmm.n_iter_),
            "gmm_log_likelihood": float(gmm.lower_bound_),
            "fit_time_sec": round(fit_time, 4),
            "predict_time_sec": round(predict_time, 4),
            "n_pixels": n_pixels,
            "n_samples": len(sample_lab),
        },
    )
