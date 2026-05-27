"""Shared utilities for segmentation engines.

Extracted from skill lib/segment.py to avoid duplication across engines.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.color import rgb2lab
from skimage.measure import label, regionprops
from skimage.morphology import disk, erosion
from skimage import segmentation


SATURATION_THRESHOLD = 80
SHAPE_RATIO_THRESHOLD = 35.0


def _detect_background_label(labels: np.ndarray) -> int | None:
    """Detect the label most likely to be background.

    Heuristic: the label that covers the largest fraction of the image edge
    AND occupies a substantial total area is treated as background.
    """
    h, w = labels.shape
    edge_margin = max(3, min(h, w) // 50)
    edge_mask = np.zeros((h, w), dtype=bool)
    edge_mask[:edge_margin, :] = True
    edge_mask[-edge_margin:, :] = True
    edge_mask[:, :edge_margin] = True
    edge_mask[:, -edge_margin:] = True

    unique = np.unique(labels)
    best_label = None
    best_score = 0.0
    for lbl in unique:
        mask = labels == lbl
        edge_count = int(mask[edge_mask].sum())
        total_count = int(mask.sum())
        if total_count == 0:
            continue
        edge_ratio = edge_count / edge_mask.sum()
        area_ratio = total_count / (h * w)
        score = edge_ratio * area_ratio
        if score > best_score and edge_ratio > 0.25 and area_ratio > 0.08:
            best_score = score
            best_label = int(lbl)
    return best_label


def _create_overlay(
    panel_rgb: np.ndarray,
    labels: np.ndarray,
    seeds_rgb: np.ndarray,
    alpha: float = 0.35,
    boundary_mode: str = "thin",
    skip_background: bool = True,
    min_area_frac: float = 0.002,
) -> np.ndarray:
    """Blend seed colors onto panel and draw white boundaries.

    Args:
        panel_rgb: Original RGB image.
        labels: Label map (int array).
        seeds_rgb: Color palette, one per label.
        alpha: Blending strength [0, 1].
        boundary_mode: "thin" | "thick" | "inner".  Prefer "thin" to avoid
            noisy boundary clutter from small fragments.
        skip_background: If True, auto-detect and skip the background label.
        min_area_frac: Merge connected components smaller than this fraction
            before drawing the overlay, to suppress noise boundaries.
    """
    # Pre-clean labels: merge tiny fragments so they don't spawn spurious boundaries
    cleaned_labels = _merge_small_regions(labels, min_area_frac=min_area_frac)

    # Optionally skip the background label
    bg_label = None
    if skip_background:
        bg_label = _detect_background_label(cleaned_labels)

    overlay = panel_rgb.copy()
    colors = seeds_rgb.astype(np.uint8)

    for l in range(len(colors)):
        if bg_label is not None and l == bg_label:
            continue
        mask = cleaned_labels == l
        if mask.any():
            overlay[mask] = (overlay[mask] * (1 - alpha) + colors[l] * alpha).astype(np.uint8)

    boundaries = segmentation.find_boundaries(cleaned_labels, mode=boundary_mode)
    if bg_label is not None:
        boundaries &= cleaned_labels != bg_label
    overlay[boundaries] = [255, 255, 255]
    return overlay


def _saturation(rgb: np.ndarray) -> np.ndarray:
    """Per-pixel max-min over RGB. Input (H,W,3) uint8 -> (H,W) int."""
    return rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)


def saturation_ratio(panel_rgb: np.ndarray, threshold: int = SATURATION_THRESHOLD) -> float:
    """Fraction of pixels with saturation > threshold."""
    s = _saturation(panel_rgb)
    return float((s > threshold).mean())


def _label_by_nearest(panel_lab: np.ndarray, palette_lab: np.ndarray) -> np.ndarray:
    """Label each pixel by index of nearest palette entry in LAB."""
    h, w, _ = panel_lab.shape
    flat = panel_lab.reshape(-1, 3)
    d2 = ((flat[:, None, :] - palette_lab[None, :, :]) ** 2).sum(axis=2)
    return d2.argmin(axis=1).reshape(h, w).astype(np.int32)


def _erode_internal_point(mask: np.ndarray) -> tuple[int, int] | None:
    """Return (x, y) of a robustly-internal pixel of a binary mask via erosion."""
    m = mask.copy()
    for r in (5, 3, 1):
        eroded = erosion(m, footprint=disk(r))
        if eroded.any():
            m = eroded
            break
    if not m.any():
        return None
    ys, xs = np.where(m)
    cx, cy = int(xs.mean()), int(ys.mean())
    if not m[cy, cx]:
        cx, cy = int(np.median(xs)), int(np.median(ys))
    return cx, cy


def _shape_filter(labels: np.ndarray, ratio_threshold: float = SHAPE_RATIO_THRESHOLD) -> np.ndarray:
    """Post-process labels: merge thin 1-D components into adjacent 2-D zones."""
    h, w = labels.shape
    out = labels.copy()
    cc = label(labels > -1, connectivity=2)
    regions = regionprops(cc)
    if not regions:
        return out

    thin_mask = np.zeros((h, w), dtype=bool)
    thin_labels = set()
    for r in regions:
        area = max(r.area, 1e-9)
        perim = r.perimeter
        ratio = float("inf") if perim == 0 else (perim ** 2) / area
        if ratio > ratio_threshold:
            thin_mask[cc == r.label] = True
            thin_labels.add(r.label)

    if not thin_labels:
        return out

    for r in regions:
        if r.label not in thin_labels:
            continue
        comp_mask = cc == r.label
        neigh = ndimage.binary_dilation(comp_mask, structure=np.ones((3, 3), dtype=bool))
        neigh_pixels = out[neigh & ~thin_mask]
        if neigh_pixels.size == 0:
            continue
        vals, counts = np.unique(neigh_pixels, return_counts=True)
        best = vals[counts.argmax()]
        out[comp_mask] = best

    return out


def _merge_small_regions(labels: np.ndarray, min_area_frac: float = 0.003) -> np.ndarray:
    """Merge tiny connected components (< min_area_frac of image) into largest neighbor."""
    h, w = labels.shape
    out = labels.copy()
    min_area = max(30, int(h * w * min_area_frac))

    cc = label(out >= 0, connectivity=2)
    regions = regionprops(cc)

    for r in regions:
        if r.area >= min_area:
            continue
        comp_mask = cc == r.label
        dilated = ndimage.binary_dilation(comp_mask, structure=np.ones((3, 3), dtype=bool))
        neighbors = out[dilated & ~comp_mask]
        if len(neighbors) == 0:
            continue
        vals, counts = np.unique(neighbors, return_counts=True)
        best = vals[counts.argmax()]
        out[comp_mask] = best
    return out


def _estimate_background_color(panel_rgb: np.ndarray) -> np.ndarray:
    """Estimate background colour from image corners and centre (median)."""
    h, w = panel_rgb.shape[:2]
    corners = np.array(
        [
            panel_rgb[0, 0],
            panel_rgb[0, w - 1],
            panel_rgb[h - 1, 0],
            panel_rgb[h - 1, w - 1],
            panel_rgb[h // 2, w // 2],
        ],
        dtype=np.float32,
    )
    return np.median(corners, axis=0).astype(np.uint8)


def _is_background_v2(
    rgb: np.ndarray, bg_rgb: np.ndarray, threshold: float = 60.0
) -> bool:
    """RGB Euclidean distance to the estimated background colour."""
    dist = float(np.linalg.norm(rgb.astype(np.float32) - bg_rgb.astype(np.float32)))
    return dist < threshold


def _spiral_search(
    panel_rgb: np.ndarray,
    start_x: int,
    start_y: int,
    radius: int = 100,
    is_bg_func=None,
) -> tuple[int, int] | None:
    """Search outward in a square spiral for a non-background pixel."""
    h, w, _ = panel_rgb.shape
    _bg = is_bg_func if is_bg_func is not None else lambda c: _is_background_v2(c, _estimate_background_color(panel_rgb))
    if 0 <= start_x < w and 0 <= start_y < h:
        if not _bg(panel_rgb[start_y, start_x]):
            return start_x, start_y

    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    x, y = start_x, start_y
    step_len = 1
    dir_idx = 0

    while abs(x - start_x) <= radius and abs(y - start_y) <= radius:
        dx, dy = dirs[dir_idx]
        for _ in range(step_len):
            x += dx
            y += dy
            if 0 <= x < w and 0 <= y < h:
                if not _bg(panel_rgb[y, x]):
                    return x, y
            if abs(x - start_x) > radius or abs(y - start_y) > radius:
                return None
        dir_idx = (dir_idx + 1) % 4
        if dir_idx % 2 == 0:
            step_len += 1
    return None


def _find_pixel_for_color(
    panel_rgb: np.ndarray,
    target_rgb: np.ndarray,
    bg_rgb: np.ndarray,
    color_tol: float = 35.0,
    bg_tol: float = 40.0,
) -> tuple[int, int] | None:
    """Find the largest connected component of pixels matching target_rgb and not background."""
    diff = np.linalg.norm(
        panel_rgb.astype(np.float32) - target_rgb.astype(np.float32), axis=2
    )
    mask = diff <= color_tol
    bg_diff = np.linalg.norm(
        panel_rgb.astype(np.float32) - bg_rgb.astype(np.float32), axis=2
    )
    mask &= bg_diff > bg_tol

    if not mask.any():
        return None

    cc = label(mask, connectivity=2)
    regions = regionprops(cc)
    if not regions:
        return None
    largest = max(regions, key=lambda r: r.area)
    cy, cx = int(largest.centroid[0]), int(largest.centroid[1])
    return cx, cy


def _online_color_groups(
    panel_rgb: np.ndarray,
    tolerance: float = 120.0,
    max_groups: int = 15,
    max_samples: int = 5000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Online tolerance-based colour grouping."""
    pixels = panel_rgb.reshape(-1, 3)
    n = len(pixels)
    if n > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, max_samples, replace=False)
        sample = pixels[idx].astype(np.float32)
    else:
        sample = pixels.astype(np.float32)

    groups: list[tuple[np.ndarray, int]] = []
    tol_sq = tolerance * tolerance

    for px in sample:
        matched = False
        for i, (mean, count) in enumerate(groups):
            diff = px - mean
            if np.dot(diff, diff) <= tol_sq:
                new_mean = (mean * count + px) / (count + 1)
                groups[i] = (new_mean, count + 1)
                matched = True
                break
        if not matched:
            groups.append((px.copy(), 1))
            if len(groups) > max_groups * 2:
                groups.sort(key=lambda g: g[1], reverse=True)
                groups = groups[:max_groups]

    groups.sort(key=lambda g: g[1], reverse=True)
    groups = groups[:max_groups]

    centers = np.array([g[0] for g in groups], dtype=np.uint8)
    counts = np.array([g[1] for g in groups], dtype=np.int64)
    return centers, counts


def _histogram_peaks(
    panel_rgb: np.ndarray,
    n_bins: int = 25,
    min_peak_ratio: float = 0.02,
) -> np.ndarray:
    """Find foreground colour peaks via grayscale histogram."""
    gray = panel_rgb.mean(axis=2).astype(np.uint8)
    hist, bin_edges = np.histogram(gray.flatten(), bins=n_bins, range=(0, 256))

    bg_idx = int(np.argmax(hist))
    bg_val = (bin_edges[bg_idx] + bin_edges[bg_idx + 1]) / 2.0
    total = gray.size

    peaks = []
    for i, count in enumerate(hist):
        if count / total < min_peak_ratio:
            continue
        val = (bin_edges[i] + bin_edges[i + 1]) / 2.0
        if abs(val - bg_val) < 30:
            continue
        peaks.append((i, val, count))

    if not peaks:
        return np.empty((0, 3), dtype=np.uint8)

    peaks.sort(key=lambda p: p[2], reverse=True)

    centers = []
    for i, _, _ in peaks[:8]:
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (gray >= low) & (gray < high)
        if mask.sum() == 0:
            continue
        rep = np.median(panel_rgb[mask], axis=0).astype(np.uint8)
        centers.append(rep)

    return np.array(centers, dtype=np.uint8)


def _cv_seeds(
    panel_rgb: np.ndarray, k: int
) -> tuple[np.ndarray, list[str]]:
    """Compute multi-source CV seeds by combining online groups and histogram peaks."""
    bg = _estimate_background_color(panel_rgb)

    og_centers, og_counts = _online_color_groups(
        panel_rgb, tolerance=60, max_groups=20
    )
    hp_centers = _histogram_peaks(panel_rgb, n_bins=25)

    candidates: list[np.ndarray] = []
    tags: list[str] = []

    for c, count in zip(og_centers, og_counts):
        if _is_background_v2(c, bg, threshold=80):
            continue
        candidates.append(c)
        tags.append(f"online(count={count})")

    for c in hp_centers:
        if _is_background_v2(c, bg, threshold=80):
            continue
        if not candidates:
            candidates.append(c)
            tags.append("histogram")
            continue
        existing = np.array(candidates, dtype=np.float32)
        dists = np.linalg.norm(existing - c.astype(np.float32), axis=1)
        if dists.min() > 40:
            candidates.append(c)
            tags.append("histogram")

    if not candidates:
        return np.empty((0, 3), dtype=np.uint8), []

    seeds = np.array(candidates, dtype=np.uint8)
    if len(seeds) > k + 2:
        seeds = seeds[: k + 2]
        tags = tags[: k + 2]
    return seeds, tags


def _parse_count_from_tag(tag: str) -> int:
    """Extract count from tags like 'online(count=123)' -> 123."""
    if "count=" in tag:
        try:
            return int(tag.split("count=")[1].rstrip(")"))
        except ValueError:
            return 0
    return 0


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


def _refine_vlm_seeds(
    panel_rgb: np.ndarray,
    reps: list[dict],
    bg_rgb: np.ndarray,
    cv_seeds_rgb: np.ndarray,
    cv_tags: list[str],
    used_cv_indices: set[int],
) -> tuple[list[np.ndarray], list[dict]]:
    """Refine VLM representative points into robust seed colors + locations.

    Returns (refined_seeds, refined_reps_metadata).
    """
    h, w = panel_rgb.shape[:2]
    panel_lab = rgb2lab(panel_rgb)

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


def adaptive_blur(panel_rgb: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """Apply Gaussian blur to suppress high-frequency noise before segmentation.

    Blur is applied only to the spatial axes; the color channel axis is left
    untouched so that RGB relationships are preserved.

    Args:
        panel_rgb: uint8 array (H, W, 3).
        sigma: Gaussian sigma in pixels. If None, computed from image diagonal
            as max(1.0, diag / 1000.0) so that larger images get slightly more
            blur. Capped at 3.0 to avoid erasing legitimate fault boundaries.

    Returns:
        Blurred uint8 array of the same shape.
    """
    from scipy.ndimage import gaussian_filter

    h, w = panel_rgb.shape[:2]
    if sigma is None:
        diag = (h * h + w * w) ** 0.5
        sigma = min(2.0, max(0.5, diag / 2000.0))

    blurred = gaussian_filter(panel_rgb, sigma=(sigma, sigma, 0))
    return np.clip(blurred, 0, 255).astype(np.uint8)


def estimate_noise_level(panel_rgb: np.ndarray) -> float:
    """Estimate perceptual noise level [0.0, 1.0] from edge density.

    Noisy images (text, grid lines, annotation markers) have high edge density.
    Smooth velocity-model layers have lower edge density.  This is a fast proxy
    that correlates well with the number of noise warnings observed in batch
    tests.
    """
    from skimage.filters import sobel

    gray = panel_rgb.mean(axis=2).astype(np.float32)
    edges = sobel(gray)
    edge_dens = float((np.abs(edges) > 0.05).mean())
    noise_score = float(np.clip(edge_dens * 1.5, 0.0, 1.0))
    return round(noise_score, 4)
