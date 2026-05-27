"""Dual-path color segmentation for geophysics interpretation panels.

Advisor guidance (2026-05-14):

- **Primary engine**: K-means clustering in LAB space on *all* panel pixels.
  VLM representative points are used only as initial seeds (not direct RGB
  sampling + nearest-neighbour).
- **Noise handling**: Do NOT aggressively mask-out text / lines before
  segmentation.  Instead, let K-means label them, then post-process connected
  components with a **perimeter² / area** shape filter.  Thin 1-D elements
  (text, contour lines, tick marks) get high ratios and are merged into the
  neighbouring 2-D colour zone.
- **Dual paths kept**:
  * jet-vivid  : VLM reps → K-means seeds → shape filter
  * pastel-faded : colorbar → K-means seeds → shape filter

Public API:
    segment(panel_rgb, kimi_reps=None, colorbar_rgb=None, k=5) -> SegmentResult
    segment_jet_vivid(panel_rgb, reps) -> SegmentResult
    segment_pastel_faded(panel_rgb, colorbar_rgb, k=5) -> SegmentResult
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy import ndimage
from skimage.color import lab2rgb, rgb2lab
from skimage.measure import label, regionprops
from skimage.filters import sobel
from skimage.morphology import disk, erosion
from skimage.segmentation import watershed


SATURATION_THRESHOLD = 80
JET_VIVID_RATIO = 0.05
SHAPE_RATIO_THRESHOLD = 35.0  # perimeter² / area; circles≈12.6, squares≈16


@dataclass
class SegmentResult:
    """Output of one segmentation pass."""
    labels: np.ndarray            # (H, W) int array, values 0..k-1
    palette: np.ndarray           # (k, 3) reference RGBs used
    color_names: list[str]        # length k
    path: str                     # "jet_vivid" | "pastel_faded"
    saturation_ratio: float
    notes: dict


def _saturation(rgb: np.ndarray) -> np.ndarray:
    """Per-pixel max-min over RGB. Input (H,W,3) uint8 → (H,W) int."""
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
    """Post-process labels: merge thin 1-D components into adjacent 2-D zones.

    For every connected component compute ``perimeter² / area``.  Values above
    ``ratio_threshold`` are treated as noise (text, contour lines, ticks) and
    reassigned to the most common neighbouring non-noise label.
    """
    h, w = labels.shape
    out = labels.copy()

    # Label connected components independently of their cluster id
    cc = label(labels > -1, connectivity=2)  # all non-background pixels
    # regionprops needs a background label of 0; our labels are 0..k-1,
    # so cc will start at 1.

    regions = regionprops(cc)
    if not regions:
        return out

    # Determine which components are "thin"
    thin_mask = np.zeros((h, w), dtype=bool)
    thin_labels = set()
    for r in regions:
        area = max(r.area, 1e-9)
        perim = r.perimeter
        # Single pixel: perimeter == 0, treat as infinitely thin
        ratio = float("inf") if perim == 0 else (perim ** 2) / area
        if ratio > ratio_threshold:
            thin_mask[cc == r.label] = True
            thin_labels.add(r.label)

    if not thin_labels:
        return out

    # Map each thin component to the dominant neighbour label
    for r in regions:
        if r.label not in thin_labels:
            continue
        comp_mask = cc == r.label
        # Neighbourhood: dilate once and intersect with non-thin pixels
        neigh = ndimage.binary_dilation(comp_mask, structure=np.ones((3, 3), dtype=bool))
        neigh_pixels = out[neigh & ~thin_mask]
        if neigh_pixels.size == 0:
            # isolated thin speck — keep original label (or could drop to -1)
            continue
        vals, counts = np.unique(neigh_pixels, return_counts=True)
        best = vals[counts.argmax()]
        out[comp_mask] = best

    return out


def _is_background(rgb: np.ndarray) -> bool:
    """Legacy heuristic: bright/grey and low saturation → background / text.

    Prefer _is_background_v2() which uses an estimated background colour.
    """
    mx, mn = int(rgb.max()), int(rgb.min())
    return mx > 200 and (mx - mn) < 45


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


def _online_color_groups(
    panel_rgb: np.ndarray,
    tolerance: float = 120.0,
    max_groups: int = 15,
    max_samples: int = 5000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Online tolerance-based colour grouping (WebPlotDigitizer-style).

    Scans pixels sequentially (randomly sampled for speed) and groups them
    by RGB Euclidean distance <= ``tolerance``.  Groups are sorted by pixel
    count descending.  The largest group is usually the background.
    """
    pixels = panel_rgb.reshape(-1, 3)
    n = len(pixels)
    if n > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, max_samples, replace=False)
        sample = pixels[idx].astype(np.float32)
    else:
        sample = pixels.astype(np.float32)

    groups: list[tuple[np.ndarray, int]] = []  # (mean_rgb, count)
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
    """Find foreground colour peaks via grayscale histogram.

    Returns representative RGBs for each significant non-background peak.
    """
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
    """Compute multi-source CV seeds by combining online groups and histogram peaks.

    Returns ``(seeds_rgb, tags)`` where ``tags`` describes the origin of each seed.
    """
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


def _find_pixel_for_color(
    panel_rgb: np.ndarray,
    target_rgb: np.ndarray,
    bg_rgb: np.ndarray,
    color_tol: float = 35.0,
    bg_tol: float = 40.0,
) -> tuple[int, int] | None:
    """Find the largest connected component of pixels matching ``target_rgb`` and not background."""
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


def _spiral_search(
    panel_rgb: np.ndarray,
    start_x: int,
    start_y: int,
    radius: int = 100,
    is_bg_func=None,
) -> tuple[int, int] | None:
    """Search outward in a square spiral for a non-background pixel.

    Spiral pattern: right 1, down 1, left 2, up 2, right 3, down 3, ...
    ``is_bg_func`` overrides the default _is_background heuristic.
    """
    h, w, _ = panel_rgb.shape
    _bg = is_bg_func if is_bg_func is not None else _is_background
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


def _kmeans_from_seeds(
    panel_lab: np.ndarray,
    seeds_lab: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run K-means in LAB space initialized with ``seeds_lab``.

    Returns ``(centroids, labels_flat)`` where labels_flat ∈ [0, k-1].
    """
    flat = panel_lab.reshape(-1, 3)
    centroids, labels_flat = kmeans2(flat, seeds_lab, minit="matrix")
    return centroids, labels_flat.astype(np.int32)


def _region_grow_dijkstra(
    panel_lab: np.ndarray,
    seeds_xy: list[tuple[int, int]],
    seeds_lab: np.ndarray,
    color_scale: float = 2.5,
    min_tol: float = 12.0,
) -> np.ndarray:
    """Multi-source region growing in LAB space using Dijkstra-like competition.

    Each seed grows outward to pixels whose LAB distance to the seed is
    below an adaptive threshold (std around seed * color_scale).
    Pixels are assigned to the seed that reaches them with the smallest
    distance.  Unassigned pixels are filled with the nearest seed label.
    """
    import heapq

    h, w = panel_lab.shape[:2]
    k = len(seeds_xy)

    # Adaptive thresholds per seed
    thresholds = []
    for x, y in seeds_xy:
        y0, y1 = max(0, y - 3), min(h, y + 4)
        x0, x1 = max(0, x - 3), min(w, x + 4)
        patch = panel_lab[y0:y1, x0:x1]
        std = float(patch.reshape(-1, 3).std(axis=0).mean())
        thresholds.append(max(std * color_scale, min_tol))
    thresholds = np.array(thresholds, dtype=np.float32)

    # Precompute distances from every pixel to every seed: (H, W, k)
    diff = panel_lab[:, :, None, :] - seeds_lab[None, None, :, :]
    dists = np.linalg.norm(diff, axis=3)

    # Dijkstra expansion from all seeds simultaneously
    best_dist = np.full((h, w), np.inf, dtype=np.float32)
    best_label = np.full((h, w), -1, dtype=np.int32)
    heap = []

    for i, (x, y) in enumerate(seeds_xy):
        d = float(dists[y, x, i])
        best_dist[y, x] = d
        best_label[y, x] = i
        heapq.heappush(heap, (d, i, x, y))

    while heap:
        d, i, x, y = heapq.heappop(heap)
        if d > best_dist[y, x] + 1e-6:
            continue
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                nd = float(dists[ny, nx, i])
                if nd < thresholds[i] and nd < best_dist[ny, nx] - 1e-6:
                    best_dist[ny, nx] = nd
                    best_label[ny, nx] = i
                    heapq.heappush(heap, (nd, i, nx, ny))

    # Fill any unassigned pixels with nearest seed
    unassigned = best_label == -1
    if unassigned.any():
        nearest = dists.argmin(axis=2)
        best_label[unassigned] = nearest[unassigned]

    return best_label


def _watershed_from_seeds(
    panel_lab: np.ndarray,
    seeds_xy: list[tuple[int, int]],
    seeds_lab: np.ndarray,
    bg_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Watershed segmentation using LAB gradient magnitude + seed markers.

    Boundaries form at ridges in the color-gradient landscape, so they
    align much better with actual geological layer edges than pure
    color-distance region growing.
    """
    h, w = panel_lab.shape[:2]

    # Multi-channel gradient: sum of Sobel magnitudes over L, A, B
    gradient = np.zeros((h, w), dtype=np.float32)
    for c in range(3):
        gradient += sobel(panel_lab[..., c]) ** 2
    gradient = np.sqrt(gradient)

    # Markers: 1-indexed (0 = background / unmarked for watershed)
    markers = np.zeros((h, w), dtype=np.int32)
    for i, (x, y) in enumerate(seeds_xy):
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        markers[y, x] = i + 1

    # Optional background mask to exclude gutters / axes
    mask = ~bg_mask if bg_mask is not None else None
    labels = watershed(gradient, markers, mask=mask)
    return labels - 1  # convert to 0-indexed


def _nearest_median(
    panel_lab: np.ndarray,
    seeds_lab: np.ndarray,
    median_size: int = 5,
) -> np.ndarray:
    """Per-pixel nearest seed in LAB, followed by median-filter smoothing.

    For smooth colormap panels (jet, rainbow) the true boundaries are
    colour contours, not sharp edges, so watershed / edge-detection often
    fails.  Nearest-neighbour in LAB space respects the actual colour
    distribution, and a modest median filter removes star / text noise
    while preserving layer geometry.
    """
    labels = _label_by_nearest(panel_lab, seeds_lab)
    if median_size > 1:
        labels = ndimage.median_filter(labels, size=median_size)
    return labels


def _parse_count_from_tag(tag: str) -> int:
    """Extract count from tags like 'online(count=123)' → 123."""
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
    """Scan the full image for dominant colors not already covered by existing seeds.

    Runs a fresh online color grouping on the full image with tighter tolerance
    to catch smaller distinct layers, then filters out background colors and
    colors too close to existing seeds.

    ``existing_auto_rgb`` is used to deduplicate against auto seeds already
    selected from the CV path (RGB distance <= 30 skips).

    Returns a list of (rgb_array, x, y, count) tuples, up to max_auto_k.
    """
    centers, counts = _online_color_groups(
        panel_rgb,
        tolerance=40.0,
        max_groups=30,
        max_samples=20000,
        seed=42,
    )

    auto_selected: list[tuple[np.ndarray, int, int, int]] = []
    auto_rgb_list: list[np.ndarray] = list(existing_auto_rgb) if existing_auto_rgb else []

    # Sort by count descending
    sorted_groups = sorted(zip(centers, counts), key=lambda t: t[1], reverse=True)

    for cseed, count in sorted_groups:
        if len(auto_selected) >= max_auto_k:
            break

        # Skip background
        if _is_background_v2(cseed, bg_rgb, threshold=60.0):
            continue

        # Skip if count too low
        if count < min_auto_count:
            continue

        # Compute LAB distance to nearest existing seed
        cseed_lab = rgb2lab(cseed[np.newaxis, ...])[0]
        d = float(np.linalg.norm(existing_seeds_lab - cseed_lab, axis=1).min())
        if d < 20.0:
            continue

        # Deduplicate against other already-selected auto seeds (RGB distance > 30)
        if auto_rgb_list:
            auto_arr = np.array(auto_rgb_list, dtype=np.float32)
            if np.linalg.norm(auto_arr - cseed.astype(np.float32), axis=1).min() <= 30.0:
                continue

        # Find actual pixel coordinates for this color
        found_px = _find_pixel_for_color(panel_rgb, cseed, bg_rgb, color_tol=40.0, bg_tol=50.0)
        if found_px is None:
            continue

        cx, cy = found_px
        auto_selected.append((cseed, cx, cy, int(count)))
        auto_rgb_list.append(cseed)

    return auto_selected


def segment_jet_vivid(
    panel_rgb: np.ndarray,
    reps: list[dict],
    max_auto_k: int = 3,
) -> SegmentResult:
    """Multi-source seeded region-growing segmentation for vivid jet-colormap panels.

    `reps` is a list of {"color_name": str, "representative_point": {"x", "y"}}.

    Strategy (multi-source fallback when VLM reps land on background):
        1. Compute CV fallback seeds (online colour groups + histogram peaks).
        2. Raw VLM reps → rough nearest-neighbour labels.
        3. For each rep:
           a. If raw point is background → spiral search.
           b. If spiral fails → fallback to best unused CV seed.
           c. If raw point is OK → erode zone for robust internal point.
        4. Auto-k: after VLM seeds are refined, detect unused CV seeds that are
           far from all VLM seeds and represent significant pixel groups.
           These catch missing color layers.
        5. Region growing (Dijkstra competition) in LAB space from all seeds.
        6. Post-process with perimeter²/area filter to swallow text/lines.
        7. Record source of each seed for review.json.
    """
    if not reps:
        raise ValueError("jet_vivid path requires at least one rep")

    h, w, _ = panel_rgb.shape
    panel_lab = rgb2lab(panel_rgb)
    bg_rgb = _estimate_background_color(panel_rgb)
    min_auto_count = max(10, h * w // 3000)  # noise floor for auto-detected groups

    # --- CV fallback seeds (computed once) ---
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
            # For region growing the spatial seed matters more than for K-means.
            # Try to find a robust internal point NEAR the original VLM point
            # instead of the global centroid of the rough label mask (which can
            # be completely wrong for thin layers).
            best_cx, best_cy = ox, oy
            best_rgb = raw_vlm_rgb
            source = "raw_vlm"

            # Try local erosion around the VLM point first
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
                    # Last resort: global erosion (K-means legacy)
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
    # Build refined seeds LAB for distance checks (used by both CV and scan paths)
    refined_seeds_arr = np.array(refined_seeds, dtype=np.uint8)
    refined_lab = rgb2lab(refined_seeds_arr[np.newaxis, ...])[0]

    if max_auto_k > 0 and len(cv_seeds_rgb) > len(used_cv_indices):
        # Candidate CV seeds sorted by count descending
        candidates = []
        for ci, (cseed, tag) in enumerate(zip(cv_seeds_rgb, cv_tags)):
            if ci in used_cv_indices:
                continue
            count = _parse_count_from_tag(tag)
            if count < min_auto_count:
                continue
            # Distance to nearest refined seed in LAB
            cseed_lab = rgb2lab(cseed[np.newaxis, ...])[0]
            d = float(np.linalg.norm(refined_lab - cseed_lab, axis=1).min())
            candidates.append((ci, cseed, tag, count, d))

        # Sort by count descending, then by distance descending
        candidates.sort(key=lambda t: (t[3], t[4]), reverse=True)

        # Deduplicate among auto-candidates themselves (RGB dist > 30)
        for ci, cseed, tag, count, d in candidates:
            if len(auto_seeds) >= max_auto_k:
                break
            # Must be far from all refined seeds (LAB distance > 20 ≈ RGB 25)
            if d < 20:
                continue
            # Deduplicate against already-selected auto seeds
            if auto_rgb_list:
                auto_arr = np.array(auto_rgb_list, dtype=np.float32)
                if np.linalg.norm(auto_arr - cseed.astype(np.float32), axis=1).min() <= 30:
                    continue
            # Find a pixel location for this color
            found_px = _find_pixel_for_color(panel_rgb, cseed, bg_rgb, color_tol=40, bg_tol=50)
            if found_px:
                cx, cy = found_px
            else:
                # Fallback: use the color itself but place at panel centre
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
    seeds_xy = [(rep["internal_x"], rep["internal_y"]) for rep in refined_reps]

    # --- Stage 3: Nearest seed in LAB + median smoothing ---
    # For smooth colormap panels edge-based methods (watershed) fail because
    # layer transitions are gradual, not sharp ridges.  Color-based nearest
    # neighbour respects the actual colour distribution; median filter adds
    # spatial coherence and removes text / star noise.
    labels = _nearest_median(panel_lab, seeds_lab, median_size=5)

    # --- Stage 4: shape filter merges thin 1-D noise ---
    labels = _shape_filter(labels)

    return SegmentResult(
        labels=labels,
        palette=refined_seeds_arr,
        color_names=color_names,
        path="jet_vivid",
        saturation_ratio=saturation_ratio(panel_rgb),
        notes={
            "reps_refined": refined_reps,
            "cv_seeds": cv_seeds_rgb.tolist() if len(cv_seeds_rgb) else [],
            "bg_rgb": bg_rgb.tolist(),
            "auto_k_added": len(auto_seeds),
        },
    )


def _sample_colorbar_seeds(colorbar_rgb: np.ndarray, k: int) -> tuple[np.ndarray, list[str]]:
    """Sample k evenly-spaced RGBs along a colorbar strip."""
    h, w, _ = colorbar_rgb.shape
    if h >= w:  # vertical colorbar
        ys = np.linspace(int(0.05 * h), int(0.95 * h) - 1, k).astype(int)
        cx = w // 2
        seeds = np.array([colorbar_rgb[y, cx] for y in ys])
    else:  # horizontal
        xs = np.linspace(int(0.05 * w), int(0.95 * w) - 1, k).astype(int)
        cy = h // 2
        seeds = np.array([colorbar_rgb[cy, x] for x in xs])
    names = _name_palette(seeds, k)
    return seeds.astype(np.uint8), names


def extract_colorbar_seeds(
    colorbar_rgb: np.ndarray,
    k: int = 5,
    velocity_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, list[str], list[float] | None]:
    """Extract k seed RGBs from a colorbar strip with optional velocity mapping.

    This is the e026 workflow entry-point: given a colorbar image, sample
    k evenly-spaced RGB seeds and optionally map each to a velocity value
    via linear interpolation over the colorbar's velocity range.

    Parameters
    ----------
    colorbar_rgb :
        (H, W, 3) uint8 colorbar strip (vertical or horizontal).
    k :
        Number of seeds to extract.
    velocity_range :
        Optional (min_v, max_v) tuple.  For vertical colorbars, top=high_v
        (cool/white) and bottom=low_v (warm/dark).  For horizontal, left=low_v
        and right=high_v (standard convention).

    Returns
    -------
    seeds_rgb : (k, 3) uint8
        Sampled RGB colors.
    names : list[str]
        Human-readable color names.
    velocities : list[float] | None
        Velocity values for each seed, or None if velocity_range not given.
    """
    seeds_rgb, names = _sample_colorbar_seeds(colorbar_rgb, k)

    velocities = None
    if velocity_range is not None:
        min_v, max_v = velocity_range
        h, w, _ = colorbar_rgb.shape
        # For vertical: top (index 0) = high_v, bottom = low_v
        # For horizontal: left = low_v, right = high_v
        if h >= w:
            # vertical: seed 0 is near top → high_v
            velocities = [max_v - i * (max_v - min_v) / max(1, k - 1) for i in range(k)]
        else:
            # horizontal: seed 0 is near left → low_v
            velocities = [min_v + i * (max_v - min_v) / max(1, k - 1) for i in range(k)]

    return seeds_rgb, names, velocities


def _name_palette(seeds_rgb: np.ndarray, k: int) -> list[str]:
    """Label k seed colors with conventional names."""
    standard = ["red", "orange", "yellow", "green", "blue", "purple"]
    if k > len(standard):
        standard = standard + [f"c{i}" for i in range(len(standard), k)]
    pool = standard[:k]
    rgb = seeds_rgb.astype(np.float32) / 255.0
    mx = rgb.max(axis=1)
    mn = rgb.min(axis=1)
    diff = mx - mn + 1e-9
    h = np.zeros(k)
    for i in range(k):
        r, g, b = rgb[i]
        if mx[i] == r:
            h[i] = (60 * ((g - b) / diff[i]) + 360) % 360
        elif mx[i] == g:
            h[i] = 60 * ((b - r) / diff[i]) + 120
        else:
            h[i] = 60 * ((r - g) / diff[i]) + 240
    order = np.argsort(h)
    names = [""] * k
    for rank, original_idx in enumerate(order):
        names[original_idx] = pool[rank]
    return names


def _reorder_labels_by_median_y(labels: np.ndarray) -> np.ndarray:
    """Reorder labels so that top=lowest index, bottom=highest index.

    Computes median y-coordinate for each label and reassigns so that
    the layer with the smallest median y (top of panel) gets label 0.
    """
    h, w = labels.shape
    unique = np.unique(labels[labels >= 0])
    if len(unique) == 0:
        return labels.copy()

    median_y = {}
    for lbl in unique:
        ys = np.where(labels == lbl)[0]
        median_y[lbl] = np.median(ys) if len(ys) > 0 else h

    sorted_by_y = sorted(median_y.items(), key=lambda x: x[1])
    old_to_new = {old: new for new, (old, _) in enumerate(sorted_by_y)}

    out = np.full_like(labels, -1)
    for old, new in old_to_new.items():
        out[labels == old] = new
    return out


def _fill_holes(labels: np.ndarray) -> np.ndarray:
    """Fill holes inside each labeled region. Physically reasonable — no voids in rock."""
    out = labels.copy()
    for lbl in range(int(labels.max()) + 1):
        mask = labels == lbl
        if not mask.any():
            continue
        filled = ndimage.binary_fill_holes(mask)
        out[filled & (labels != lbl)] = lbl
    return out


def _remove_small_components(labels: np.ndarray, min_area_frac: float = 0.001) -> np.ndarray:
    """Merge tiny connected components (< min_area_frac of panel area) into neighbors.

    Preserves valid fracture/fault regions that would be deleted by
    keep_largest_component. Only removes specks that are clearly noise.
    """
    h, w = labels.shape
    out = labels.copy()
    min_area = max(50, int(h * w * min_area_frac))

    for lbl in range(int(labels.max()) + 1):
        mask = out == lbl
        if not mask.any():
            continue
        labeled, num = ndimage.label(mask)
        if num <= 1:
            continue
        sizes = ndimage.sum(mask, labeled, range(1, num + 1))
        for comp_id in range(1, num + 1):
            if sizes[comp_id - 1] < min_area:
                comp_mask = labeled == comp_id
                dilated = ndimage.binary_dilation(comp_mask, structure=np.ones((3, 3), dtype=bool))
                neighbors = out[dilated & ~comp_mask & (out >= 0)]
                if len(neighbors) > 0:
                    new_lbl = int(np.bincount(neighbors).argmax())
                    out[comp_mask] = new_lbl
    return out


def _enhance_close_boundaries(
    panel_rgb: np.ndarray,
    labels: np.ndarray,
    palette_rgb: np.ndarray,
    color_dist_threshold: float = 55.0,
) -> np.ndarray:
    """Re-classify boundary pixels between adjacent layers with similar seed colors.

    For adjacent layer pairs whose seed RGB distance < threshold,
    re-classify boundary pixels based on direct distance to the two seed colors.
    This sharpens boundaries that K-means may have blurred.
    """
    out = labels.copy()
    k = len(palette_rgb)
    if k < 2:
        return out

    for i in range(k - 1):
        d = float(np.linalg.norm(palette_rgb[i].astype(np.float32) - palette_rgb[i + 1].astype(np.float32)))
        if d >= color_dist_threshold:
            continue

        mask1 = out == i
        mask2 = out == (i + 1)
        if not mask1.any() or not mask2.any():
            continue

        dilated1 = ndimage.binary_dilation(mask1, structure=np.ones((3, 3), dtype=bool))
        dilated2 = ndimage.binary_dilation(mask2, structure=np.ones((3, 3), dtype=bool))
        boundary = dilated1 & dilated2
        if not boundary.any():
            continue

        coords = np.where(boundary)
        boundary_pixels = panel_rgb[coords].astype(np.float32)
        d1 = np.linalg.norm(boundary_pixels - palette_rgb[i].astype(np.float32), axis=1)
        d2 = np.linalg.norm(boundary_pixels - palette_rgb[i + 1].astype(np.float32), axis=1)
        reclass = d2 < d1
        out[coords[0][reclass], coords[1][reclass]] = i + 1
        out[coords[0][~reclass], coords[1][~reclass]] = i

    return out


def segment_colorbar_guided(
    panel_rgb: np.ndarray,
    colorbar_rgb: np.ndarray,
    k: int = 5,
    color_dist_threshold: float = 55.0,
    explicit_seeds: list[dict] | None = None,
) -> SegmentResult:
    """Clean colorbar-guided segmentation — default for pastel-faded panels.

    Developed in experiment e026 and proved superior to prior approaches:
    1. K-means in RGB space with colorbar-extracted seeds as init= parameter
    2. NO bilateral denoising (creates boundary artifacts)
    3. NO keep_largest_component (deletes valid fracture/fault regions)
    4. Reorder labels by median_y (top=high velocity/low index, bottom=low velocity)
    5. Only fill_holes (physically reasonable — no voids in rock)
    6. Remove small connected components (< 0.1% panel area) by merging to neighbors
    7. Enhance boundaries for adjacent layers with seed color distance < threshold

    Parameters
    ----------
    panel_rgb : np.ndarray
        (H, W, 3) uint8 cropped panel.
    colorbar_rgb : np.ndarray
        (H, W, 3) uint8 colorbar strip (vertical or horizontal).
    k : int
        Number of color zones to extract from the colorbar.
    color_dist_threshold : float
        RGB distance threshold for boundary enhancement between adjacent layers.
    explicit_seeds : list[dict] | None
        Optional pre-computed colorbar seeds. Each dict has keys:
        - "rgb": [r, g, b] list of uint8
        - "name": str (optional)
        - "velocity": float (optional)
        If provided, these seeds are used directly instead of auto-sampling
        from the colorbar image. This is the e026 physical-prior workflow.

    Returns
    -------
    SegmentResult
    """
    h, w, _ = panel_rgb.shape
    pixels = panel_rgb.reshape(-1, 3).astype(np.float64)

    # Step 1: Get seeds — explicit (e026 workflow) or auto-sampled
    if explicit_seeds is not None and len(explicit_seeds) > 0:
        seeds_rgb = np.array([s["rgb"] for s in explicit_seeds], dtype=np.uint8)
        names = [s.get("name", f"layer_{i+1}") for i, s in enumerate(explicit_seeds)]
        k = len(explicit_seeds)
    else:
        seeds_rgb, names = _sample_colorbar_seeds(colorbar_rgb, k)
    seeds_arr = seeds_rgb.astype(np.float64)

    # Step 2: K-means with colorbar seeds as initial centroids
    centroids, labels_flat = kmeans2(pixels, seeds_arr, minit="matrix")
    labels = labels_flat.reshape(h, w).astype(np.int32)

    # Step 3: Reorder by median y (top to bottom)
    labels = _reorder_labels_by_median_y(labels)

    # Step 4: Fill holes
    labels = _fill_holes(labels)

    # Step 5: Remove small components
    labels = _remove_small_components(labels, min_area_frac=0.001)

    # Step 6: Enhance boundaries for close-color adjacent layers
    # Rebuild palette from centroids, ordered to match reordered labels
    ordered_palette = np.zeros((k, 3), dtype=np.uint8)
    for lbl in range(k):
        mask = labels == lbl
        if mask.any():
            ordered_palette[lbl] = panel_rgb[mask].mean(axis=0).astype(np.uint8)
        else:
            ordered_palette[lbl] = (lab2rgb(centroids[lbl][np.newaxis, ...])[0] * 255).clip(0, 255).astype(np.uint8)

    labels = _enhance_close_boundaries(panel_rgb, labels, ordered_palette, color_dist_threshold)

    # Rebuild final palette from actual pixel means
    final_palette = np.zeros((k, 3), dtype=np.uint8)
    for lbl in range(k):
        mask = labels == lbl
        if mask.any():
            final_palette[lbl] = panel_rgb[mask].mean(axis=0).astype(np.uint8)
        else:
            final_palette[lbl] = ordered_palette[lbl]

    return SegmentResult(
        labels=labels,
        palette=final_palette,
        color_names=names,
        path="colorbar_guided",
        saturation_ratio=saturation_ratio(panel_rgb),
        notes={
            "seed_origin": "explicit_seeds" if explicit_seeds is not None else "colorbar",
            "centroids_lab": rgb2lab(final_palette[np.newaxis, ...])[0].tolist(),
            "color_dist_threshold": color_dist_threshold,
            "explicit_seeds_count": len(explicit_seeds) if explicit_seeds is not None else 0,
        },
    )


def segment_pastel_faded(
    panel_rgb: np.ndarray,
    colorbar_rgb: np.ndarray | None,
    k: int = 5,
) -> SegmentResult:
    """K-means in LAB space, optionally seeded from the panel's colorbar.

    Same post-processing shape filter is applied so text / thin contours are
    merged into neighbouring zones.

    NOTE: When a colorbar is available, ``segment_colorbar_guided()`` is
    preferred (it is the new default). This function is kept for backward
    compatibility and for the no-colorbar fallback path.
    """
    h, w, _ = panel_rgb.shape
    panel_lab = rgb2lab(panel_rgb).reshape(-1, 3)

    if colorbar_rgb is not None and colorbar_rgb.size > 0:
        seeds_rgb, names = _sample_colorbar_seeds(colorbar_rgb, k)
        seeds_lab = rgb2lab(seeds_rgb[np.newaxis, ...])[0]
        centroids, labels_flat = kmeans2(panel_lab, seeds_lab, minit="matrix")
        seed_origin = "colorbar"
    else:
        # Try CV multi-source seeds before falling back to random kmeans++
        cv_seeds_rgb, cv_tags = _cv_seeds(panel_rgb, k=k)
        if len(cv_seeds_rgb) >= k:
            seeds_rgb = cv_seeds_rgb[:k]
            seeds_lab = rgb2lab(seeds_rgb[np.newaxis, ...])[0]
            centroids, labels_flat = kmeans2(panel_lab, seeds_lab, minit="matrix")
            seed_origin = "cv_multi_source"
            names = [f"cv_{i}" for i in range(k)]
        else:
            centroids, labels_flat = kmeans2(panel_lab, k, minit="++", seed=42)
            approx = (lab2rgb(centroids[np.newaxis, ...])[0] * 255).clip(0, 255).astype(np.uint8)
            seeds_rgb = approx
            names = _name_palette(seeds_rgb, k)
            seed_origin = "kmeans++_random"

    labels = labels_flat.reshape(h, w).astype(np.int32)
    labels = _shape_filter(labels)

    return SegmentResult(
        labels=labels,
        palette=seeds_rgb,
        color_names=names,
        path="pastel_faded",
        saturation_ratio=saturation_ratio(panel_rgb),
        notes={"seed_origin": seed_origin, "centroids_lab": centroids.tolist()},
    )


def segment(
    panel_rgb: np.ndarray,
    kimi_reps: list[dict] | None = None,
    colorbar_rgb: np.ndarray | None = None,
    k: int = 5,
    max_auto_k: int = 3,
) -> SegmentResult:
    """Dispatcher: pick jet_vivid or colorbar-guided by saturation ratio.

    Routing logic:
    - sat >= JET_VIVID_RATIO AND kimi_reps present → jet_vivid (VLM rep points)
    - colorbar_rgb present → colorbar_guided (clean method, e026-proven)
    - fallback → pastel_faded (legacy K-means + shape filter)
    """
    ratio = saturation_ratio(panel_rgb)
    if ratio >= JET_VIVID_RATIO and kimi_reps:
        return segment_jet_vivid(panel_rgb, kimi_reps, max_auto_k=max_auto_k)
    if colorbar_rgb is not None and colorbar_rgb.size > 0:
        return segment_colorbar_guided(panel_rgb, colorbar_rgb, k=k)
    return segment_pastel_faded(panel_rgb, colorbar_rgb, k)


__all__ = [
    "SegmentResult",
    "saturation_ratio",
    "segment",
    "segment_jet_vivid",
    "segment_colorbar_guided",
    "segment_pastel_faded",
    "extract_colorbar_seeds",
    "SATURATION_THRESHOLD",
    "JET_VIVID_RATIO",
    "SHAPE_RATIO_THRESHOLD",
]
