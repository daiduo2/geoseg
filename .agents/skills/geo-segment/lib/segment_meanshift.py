"""Mean Shift color clustering segmentation for geophysics interpretation panels.

Mean Shift discovers color modes automatically in LAB space, eliminating the need
to pre-specify K.  The number of discovered modes = number of layers.
"""

from __future__ import annotations

import numpy as np
from skimage.color import rgb2lab, lab2rgb
from sklearn.cluster import MeanShift, estimate_bandwidth

from lib.segment import SegmentResult, saturation_ratio, _shape_filter


def _estimate_bandwidth_from_reps(panel_lab: np.ndarray, reps: list[dict] | None) -> float | None:
    """Estimate Mean Shift bandwidth from VLM seed colors in LAB space.

    Uses the **minimum** pairwise distance between seed points as a heuristic.
    The rationale: bandwidth should be smaller than the closest pair of distinct
    seeds, otherwise Mean Shift will merge them into a single mode.
    """
    if not reps:
        return None

    # Sample a small patch around each rep point to get local LAB mean
    h, w = panel_lab.shape[:2]
    seed_labs = []
    for r in reps:
        x = int(r["representative_point"]["x"])
        y = int(r["representative_point"]["y"])
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        # 3x3 patch around seed
        x0, x1 = max(0, x - 1), min(w, x + 2)
        y0, y1 = max(0, y - 1), min(h, y + 2)
        patch = panel_lab[y0:y1, x0:x1]
        seed_labs.append(patch.reshape(-1, 3).mean(axis=0))

    seed_labs = np.array(seed_labs)
    if len(seed_labs) < 2:
        return None

    # Minimum pairwise distance in LAB — bandwidth must be smaller than this
    # or distinct seeds will collapse into one mode.
    dists = []
    for i in range(len(seed_labs)):
        for j in range(i + 1, len(seed_labs)):
            dists.append(np.linalg.norm(seed_labs[i] - seed_labs[j]))
    min_dist = float(np.min(dists))
    # Use 0.7 * min_dist as bandwidth — empirically tuned on 181218:
    # 0.6 -> 12 modes (over-segmented), 0.7 -> 10 modes (matches ~10 layers),
    # 0.8 -> 8 modes (under-segmented).  Factor can be exposed as parameter.
    return min_dist * 0.7


def segment_jet_vivid_meanshift(
    panel_rgb: np.ndarray,
    reps: list[dict] | None = None,
    bandwidth: float | None = None,
    quantile: float = 0.2,
    n_samples: int = 5000,
    bin_seeding: bool = True,
    min_bin_freq: int = 5,
) -> SegmentResult:
    """Segment a panel using Mean Shift clustering in LAB space.

    Parameters
    ----------
    panel_rgb : np.ndarray
        (H, W, 3) uint8 RGB image.
    reps : list[dict] | None
        Optional VLM representative points.  If ``bandwidth`` is None and
        ``reps`` is provided, bandwidth is estimated from median pairwise
        distance between rep points in LAB space.
    bandwidth : float | None
        Mean Shift bandwidth in LAB space.  If None, estimated from data
        via ``sklearn.cluster.estimate_bandwidth`` or from ``reps``.
    quantile : float
        Quantile for ``estimate_bandwidth`` when bandwidth is not provided
        and reps are not available.
    n_samples : int
        Number of samples for ``estimate_bandwidth``.
    bin_seeding : bool
        Use bin seeding for Mean Shift (faster, more stable).
    min_bin_freq : int
        Minimum number of points in a bin to be considered a seed.

    Returns
    -------
    SegmentResult
    """
    h, w, _ = panel_rgb.shape
    panel_lab = rgb2lab(panel_rgb)
    flat_lab = panel_lab.reshape(-1, 3)

    # --- Bandwidth estimation ---
    if bandwidth is None:
        if reps is not None:
            bw = _estimate_bandwidth_from_reps(panel_lab, reps)
            if bw is not None and bw > 0:
                bandwidth = bw
                bw_source = "reps_median_pairwise"
            else:
                bw_source = "estimate_bandwidth_fallback"
        else:
            bw_source = "estimate_bandwidth"

        if bandwidth is None:
            # Use sklearn's estimate_bandwidth on a subsample for speed
            bandwidth = estimate_bandwidth(
                flat_lab,
                quantile=quantile,
                n_samples=min(n_samples, len(flat_lab)),
                random_state=42,
            )
            if bandwidth is None or bandwidth <= 0:
                # Fallback: heuristic based on LAB space scale
                bandwidth = 8.0
    else:
        bw_source = "user_provided"

    # Ensure bandwidth is positive
    bandwidth = max(bandwidth, 1.0)

    # --- Mean Shift clustering ---
    ms = MeanShift(
        bandwidth=bandwidth,
        bin_seeding=bin_seeding,
        min_bin_freq=min_bin_freq,
        n_jobs=-1,
    )
    ms.fit(flat_lab)

    labels_flat = ms.labels_.astype(np.int32)
    cluster_centers_lab = ms.cluster_centers_
    n_modes = len(cluster_centers_lab)

    # Map labels back to image
    labels = labels_flat.reshape(h, w)

    # --- Post-processing: shape filter ---
    labels = _shape_filter(labels)

    # Recompute labels after shape filter (some may have merged)
    unique_labels = np.unique(labels)
    # Remap to contiguous 0..k-1
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels = np.vectorize(label_map.get)(labels)
    n_modes = len(unique_labels)

    # --- Compute palette as median RGB of each cluster ---
    flat_rgb = panel_rgb.reshape(-1, 3)
    palette = []
    color_names = []
    for i in range(n_modes):
        mask = labels_flat.reshape(-1) == unique_labels[i] if i < len(unique_labels) else (labels_flat.reshape(-1) == i)
        # After remapping, use the remapped labels
        mask = labels.reshape(-1) == i
        pixels = flat_rgb[mask]
        if len(pixels) > 0:
            median_rgb = np.median(pixels, axis=0).astype(np.uint8)
        else:
            # Fallback to cluster center
            center_lab = cluster_centers_lab[i] if i < len(cluster_centers_lab) else np.zeros(3)
            median_rgb = (lab2rgb(center_lab[np.newaxis, ...])[0] * 255).clip(0, 255).astype(np.uint8)
        palette.append(median_rgb)
        color_names.append(f"mode_{i}")

    palette = np.array(palette, dtype=np.uint8)

    # Compute cluster sizes
    cluster_sizes = []
    for i in range(n_modes):
        count = int((labels == i).sum())
        cluster_sizes.append(count)

    return SegmentResult(
        labels=labels,
        palette=palette,
        color_names=color_names,
        path="jet_vivid_meanshift",
        saturation_ratio=saturation_ratio(panel_rgb),
        notes={
            "bandwidth": float(bandwidth),
            "bandwidth_source": bw_source,
            "n_modes_discovered": n_modes,
            "cluster_sizes": cluster_sizes,
            "cluster_centers_lab": cluster_centers_lab.tolist() if len(cluster_centers_lab) == n_modes else [],
            "reps": reps,
        },
    )
