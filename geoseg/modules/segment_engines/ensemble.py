"""Adaptive ensemble segmentation with saturation gating + consistency-weighted voting (e017v2).

Key improvements over v1:
1. Saturation-adaptive algorithm selection.
2. Consistency-weighted voting.
3. Fully vectorised voting.
"""

from __future__ import annotations

import time

import numpy as np
from scipy import ndimage
from skimage.color import rgb2lab
from skimage.measure import label, regionprops

from geoseg.modules.segment_engines._shared import (
    _create_overlay,
    _estimate_background_color,
    saturation_ratio,
)
from geoseg.modules.segment_engines import v4_kmeans
from geoseg.modules.segment_engines import edge_guided as edge_guided_mod
from geoseg.modules.segment_engines import kmeans_full


SAT_VIVID = 0.5
SAT_MIXED = 0.1


def _compute_consistency(labels: np.ndarray) -> float:
    """Spatial coherence: fraction of 4-neighbors sharing the same label."""
    same = np.ones_like(labels, dtype=bool)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted = np.roll(labels, shift=(dy, dx), axis=(0, 1))
        same &= (labels == shifted)
    return float(same.mean())


def _weighted_vote(labels_stack: np.ndarray, weights: list[float]) -> np.ndarray:
    """Vectorised consistency-weighted voting."""
    h, w, n = labels_stack.shape
    h, w = int(h), int(w)
    assert len(weights) == n

    flat_labels = labels_stack.reshape(-1, n)
    vote_weights = np.array(weights, dtype=np.float32)

    max_label = int(labels_stack.max()) + 1
    if max_label <= 0:
        return np.zeros((h, w), dtype=np.int32)

    vote_matrix = np.zeros((flat_labels.shape[0], max_label), dtype=np.float32)

    for algo_idx in range(n):
        algo_labs = flat_labels[:, algo_idx]
        algo_w = vote_weights[algo_idx]
        np.add.at(vote_matrix, (np.arange(vote_matrix.shape[0]), algo_labs), algo_w)

    voted_flat = vote_matrix.argmax(axis=1)
    return voted_flat.reshape(h, w).astype(np.int32)


def _map_to_common(
    src_labels: np.ndarray,
    src_palette: np.ndarray,
    common_palette_lab: np.ndarray,
) -> np.ndarray:
    """Map src algorithm labels to common label space via nearest palette color."""
    src_palette_lab = rgb2lab(src_palette[np.newaxis, ...])[0]
    mapping = {}
    for i, sp in enumerate(src_palette_lab):
        d = np.linalg.norm(common_palette_lab - sp, axis=1)
        mapping[i] = int(d.argmin())
    return np.vectorize(mapping.get)(src_labels)


def _merge_small_components(
    labels: np.ndarray, min_area_ratio: float = 0.01
) -> np.ndarray:
    """Merge connected components smaller than min_area_ratio * total pixels."""
    h, w = labels.shape
    total = h * w
    min_area = max(1, int(min_area_ratio * total))

    out = labels.copy()
    cc = label(out, connectivity=2)
    regions = regionprops(cc)

    adj: dict[int, set[int]] = {int(r.label): set() for r in regions}
    for y in range(h):
        for x in range(w):
            cid = int(cc[y, x])
            if cid not in adj:
                continue
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    nid = int(cc[ny, nx])
                    if nid != cid and nid in adj:
                        adj[cid].add(nid)
                        adj[nid].add(cid)

    for r in regions:
        if r.area >= min_area:
            continue
        cid = r.label
        neigh_labels = []
        for nid in adj.get(cid, set()):
            neigh_labels.extend(out[cc == nid].tolist())
        if not neigh_labels:
            continue
        vals, counts = np.unique(neigh_labels, return_counts=True)
        best = vals[counts.argmax()]
        out[cc == cid] = best

    unique = np.unique(out)
    remap = {old: new for new, old in enumerate(unique)}
    return np.vectorize(remap.get)(out)


def _compute_palette(
    panel_rgb: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    """Compute median RGB per label."""
    unique = np.unique(labels)
    palette = []
    color_names = []
    flat_rgb = panel_rgb.reshape(-1, 3)
    flat_labels = labels.reshape(-1)
    for i, lid in enumerate(unique):
        mask = flat_labels == lid
        pixels = flat_rgb[mask]
        median_rgb = (
            np.median(pixels, axis=0).astype(np.uint8) if len(pixels) > 0
            else np.array([128, 128, 128], dtype=np.uint8)
        )
        palette.append(median_rgb)
        color_names.append(f"region_{i}")
    return np.array(palette, dtype=np.uint8), color_names


def segment(
    panel_rgb: np.ndarray,
    reps: list[dict] | None = None,
    n_layers: int = 5,
    max_auto_k: int = 3,
) -> dict:
    """Adaptive ensemble segmentation.

    Algorithm selection by saturation:
      - saturation >= 0.5 (vivid) : baseline + edge_guided + kmeans
      - 0.1 <= saturation < 0.5   : baseline + kmeans
      - saturation < 0.1 (pastel) : baseline only (direct return)

    Voting: consistency-weighted rather than naive majority.

    Args:
        panel_rgb: RGB uint8 array (H, W, 3).
        reps: Optional VLM representative points. If None, each sub-engine uses CV seeds.
        n_layers: Not used directly, kept for interface consistency.
        max_auto_k: Maximum extra seeds to auto-detect.

    Returns:
        dict with keys: labels, seeds, overlay, meta.
    """

    h, w, _ = panel_rgb.shape
    sat = saturation_ratio(panel_rgb)
    bg_rgb = _estimate_background_color(panel_rgb)

    timings: dict[str, float] = {}
    t_overall = time.perf_counter()

    # --- Algorithm 1: baseline (always run) ---
    t0 = time.perf_counter()
    result_base = v4_kmeans.segment_jet_vivid(panel_rgb, reps, max_auto_k=max_auto_k)
    timings["baseline"] = time.perf_counter() - t0
    base_consistency = _compute_consistency(result_base["labels"])

    algo_results = [(result_base, base_consistency)]
    algo_names = ["baseline"]

    if sat >= SAT_MIXED:
        t0 = time.perf_counter()
        result_km = kmeans_full.segment(panel_rgb, reps, max_auto_k=max_auto_k)
        timings["kmeans"] = time.perf_counter() - t0
        km_consistency = _compute_consistency(result_km["labels"])
        algo_results.append((result_km, km_consistency))
        algo_names.append("kmeans")

    if sat >= SAT_VIVID:
        t0 = time.perf_counter()
        result_eg = edge_guided_mod.segment(panel_rgb, reps, max_auto_k=max_auto_k, edge_weight=0.3)
        timings["edge_guided"] = time.perf_counter() - t0
        eg_consistency = _compute_consistency(result_eg["labels"])
        algo_results.append((result_eg, eg_consistency))
        algo_names.append("edge_guided")

    # --- Fast path: only 1 algorithm -> direct return ---
    if len(algo_results) == 1:
        result = algo_results[0][0]
        timings["total"] = time.perf_counter() - t_overall
        return {
            "labels": result["labels"].copy(),
            "seeds": result["seeds"][:],
            "overlay": result["overlay"].copy(),
            "meta": {
                "engine": "ensemble",
                "path": "ensemble_fallback",
                "reps_refined": result["meta"].get("reps_refined", []),
                "bg_rgb": bg_rgb.tolist(),
                "auto_k_added": result["meta"].get("auto_k_added", 0),
                "timings_sec": {k: round(v, 3) for k, v in timings.items()},
                "saturation": round(sat, 4),
                "algo_selection": algo_names,
                "fallback_reason": f"saturation={round(sat,4)} < {SAT_MIXED}",
            },
        }

    # --- Align all algorithms to common label space (baseline palette) ---
    common_palette = np.array(result_base["seeds"], dtype=np.uint8)
    common_palette_lab = rgb2lab(common_palette[np.newaxis, ...])[0]

    label_maps = []
    consistencies = []
    for res, cons in algo_results:
        src_palette = np.array(res["seeds"], dtype=np.uint8)
        mapped = _map_to_common(res["labels"], src_palette, common_palette_lab)
        label_maps.append(mapped)
        consistencies.append(cons)

    labels_stack = np.stack(label_maps, axis=2)

    # --- Consistency-weighted voting ---
    voted_labels = _weighted_vote(labels_stack, consistencies)

    # --- Post-process: merge tiny connected components ---
    voted_labels = _merge_small_components(voted_labels, min_area_ratio=0.005)

    # --- Compute final palette ---
    final_palette, final_color_names = _compute_palette(panel_rgb, voted_labels)

    # --- Disagreement diagnostics ---
    disagreements = np.zeros((h, w), dtype=bool)
    for i in range(labels_stack.shape[2]):
        for j in range(i + 1, labels_stack.shape[2]):
            disagreements |= (labels_stack[:, :, i] != labels_stack[:, :, j])
    disagreement_pct = float(disagreements.mean() * 100)

    timings["ensemble_overhead"] = time.perf_counter() - t_overall - sum(
        v for k, v in timings.items() if k != "ensemble_overhead"
    )
    timings["total"] = time.perf_counter() - t_overall

    overlay = _create_overlay(panel_rgb, voted_labels, final_palette)

    return {
        "labels": voted_labels,
        "seeds": final_palette.tolist(),
        "overlay": overlay,
        "meta": {
            "engine": "ensemble",
            "path": "jet_vivid_ensemble",
            "reps_refined": result_base["meta"].get("reps_refined", []),
            "bg_rgb": bg_rgb.tolist(),
            "auto_k_added": result_base["meta"].get("auto_k_added", 0),
            "timings_sec": {k: round(v, 3) for k, v in timings.items()},
            "saturation": round(sat, 4),
            "algo_selection": algo_names,
            "consistency_per_algo": [round(c, 4) for c in consistencies],
            "disagreement_pct": round(disagreement_pct, 2),
            "num_labels_per_algo": [
                int(len(np.unique(res["labels"]))) for res, _ in algo_results
            ],
            "final_num_labels": int(len(np.unique(voted_labels))),
        },
    }
