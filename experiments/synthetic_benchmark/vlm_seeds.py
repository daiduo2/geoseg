"""Synthetic VLM-style representative-point generator.

The `kmeans_full`, `edge_guided`, `edge_grow` engines all expect
`reps: list[dict]` where each dict carries a `color_name` and a
`representative_point: {x, y}`. In a real geoseg run the VLM (vision
language model) gives us these; in the synthetic benchmark we have
ground-truth labels, so we can synthesize representative points by
picking a pixel per label and naming its color from a fixed palette.

Why this matters: those three engines are tagged `requires_reps=True`
in the engine registry, so `run_engine` falls back to `v4_kmeans` if
`reps` is None — silently masking any difference between them and
v4_kmeans in our previous default_benchmark results. Synthesizing
plausible reps lets us exercise the primary paths.

Why this is honest: the synthesized reps are *better* than what a real
VLM produces — exact label-center pixels, correct color names — so
the benchmark upper-bounds what these engines can do given good seed
information. Differences in metrics between "synthesized reps" and
"no reps (fallback to v4_kmeans)" measure the engine's capacity to
*use* seed information; differences between engines measure their
algorithmic choices given the same seeds.
"""

from __future__ import annotations

import numpy as np

# Conventional color names in order of scalar field value [0, 1].
# Matches the convention used in geoseg's palette module.
_COLOR_NAMES = [
    "dark_blue", "blue", "cyan", "green", "yellow",
    "orange", "red", "dark_red", "purple", "pink",
    "brown", "white",
]


def _name_for_value(v: float) -> str:
    """Map a scalar field value [0, 1] to a conventional color name."""
    n = len(_COLOR_NAMES)
    idx = min(n - 1, max(0, int(round(v * (n - 1)))))
    return _COLOR_NAMES[idx]


def synthesize_reps(
    gt_labels: np.ndarray,
    label_to_value: dict[int, float],
    panel_rgb: np.ndarray | None = None,
    *,
    strategy: str = "label_centroid",
    seed: int = 0,
) -> list[dict]:
    """Build VLM-style reps for each non-zero ground-truth label.

    Args:
        gt_labels: (H, W) int32 ground-truth label image.
        label_to_value: mapping {label_id: scalar field value in [0, 1]}
            from the generator's gen_meta.
        panel_rgb: optional RGB image for color-name fallback; not used
            by the centroid strategy (we trust label_to_value).
        strategy: "label_centroid" picks one pixel near the centroid of
            each label's pixels. Reserved for future strategies.
        seed: RNG seed (centroid jitter for ties).

    Returns:
        list of dicts, each shaped like a VLM rep:
          {"color_name": str, "representative_point": {"x": int, "y": int}}

    Order is by label id ascending (1, 2, ..., N), matching the
    generator's label assignment.
    """
    h, w = gt_labels.shape
    reps: list[dict] = []
    for lbl in sorted(label_to_value.keys()):
        if lbl == 0:
            continue
        mask = gt_labels == lbl
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        if strategy == "label_centroid":
            # Pick the pixel closest to the centroid (deterministic given
            # mask); small jitter if we want to simulate VLM noise
            cy = float(ys.mean())
            cx = float(xs.mean())
            # Nearest actual mask pixel to centroid
            d2 = (ys - cy) ** 2 + (xs - cx) ** 2
            pick = int(d2.argmin())
            y, x = int(ys[pick]), int(xs[pick])
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        value = float(label_to_value[lbl])
        reps.append({
            "color_name": _name_for_value(value),
            "representative_point": {"x": int(x), "y": int(y)},
        })
    return reps


def synthesize_noisy_reps(
    gt_labels: np.ndarray,
    label_to_value: dict[int, float],
    *,
    offset_max: int = 6,
    color_noise: float = 0.15,
    seed: int = 0,
) -> list[dict]:
    """Same as `synthesize_reps` but with VLM-realistic noise.

    - `offset_max`: pixel position perturbed by up to ±offset_max
    - `color_noise`: fraction of labels whose color_name is perturbed
      (cycled to a neighbouring name in the palette).

    Useful for stress-testing: the engine's `_refine_vlm_seeds` should
    correct for these errors via spiral search and CV seed fallback.
    """
    base = synthesize_reps(gt_labels, label_to_value, seed=seed)
    rng = np.random.default_rng(seed + 1)
    h, w = gt_labels.shape
    for i, rep in enumerate(base):
        rp = rep["representative_point"]
        rp["x"] = max(0, min(w - 1, rp["x"] + int(rng.integers(-offset_max, offset_max + 1))))
        rp["y"] = max(0, min(h - 1, rp["y"] + int(rng.integers(-offset_max, offset_max + 1))))
        if rng.random() < color_noise:
            # Cycle to next name in palette to simulate color-name error
            try:
                cur = _COLOR_NAMES.index(rep["color_name"])
                rep["color_name"] = _COLOR_NAMES[(cur + 1) % len(_COLOR_NAMES)]
            except ValueError:
                pass
    return base


__all__ = [
    "synthesize_reps",
    "synthesize_noisy_reps",
]
