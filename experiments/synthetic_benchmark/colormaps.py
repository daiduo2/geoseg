"""Tiny colormap registry used by the synthetic benchmark.

Geoseg's real-world panels come in a handful of canonical colormaps.
We deliberately do not depend on matplotlib (which isn't always
available in the headless geoseg env). The lookups below are pixel-
level copies of the canonical matplotlib colormaps, hard-coded so the
benchmark is self-contained.
"""

from __future__ import annotations

import numpy as np


def _lerp(c0: tuple[int, int, int], c1: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return (
        int(round(c0[0] + (c1[0] - c0[0]) * t)),
        int(round(c0[1] + (c1[1] - c0[1]) * t)),
        int(round(c0[2] + (c1[2] - c0[2]) * t)),
    )


# Eight-stop colour tables approximating matplotlib's named colormaps.
# Each entry is (stop_position, RGB). Position in [0, 1].
_TABLES: dict[str, list[tuple[float, tuple[int, int, int]]]] = {
    "jet": [
        (0.0, (0, 0, 127)),
        (0.125, (0, 0, 255)),
        (0.375, (0, 255, 255)),
        (0.625, (255, 255, 0)),
        (0.875, (255, 0, 0)),
        (1.0, (127, 0, 0)),
    ],
    "viridis": [
        (0.0, (68, 1, 84)),
        (0.25, (59, 82, 139)),
        (0.5, (33, 144, 141)),
        (0.75, (94, 201, 98)),
        (1.0, (253, 231, 37)),
    ],
    "plasma": [
        (0.0, (13, 8, 135)),
        (0.25, (126, 3, 168)),
        (0.5, (203, 70, 121)),
        (0.75, (248, 149, 64)),
        (1.0, (240, 249, 33)),
    ],
    "gray": [
        (0.0, (0, 0, 0)),
        (1.0, (255, 255, 255)),
    ],
}


def available() -> list[str]:
    return sorted(_TABLES.keys())


def map_value(value: float, name: str) -> tuple[int, int, int]:
    """Map a scalar in [0, 1] to an RGB triple via piecewise-linear interp."""
    if name not in _TABLES:
        raise KeyError(
            f"Unknown colormap {name!r}. Available: {available()}"
        )
    table = _TABLES[name]
    v = max(0.0, min(1.0, float(value)))
    # Locate segment
    for i in range(len(table) - 1):
        p0, c0 = table[i]
        p1, c1 = table[i + 1]
        if v <= p1:
            if p1 == p0:
                return c0
            t = (v - p0) / (p1 - p0)
            return _lerp(c0, c1, t)
    return table[-1][1]


def map_array(values: np.ndarray, name: str) -> np.ndarray:
    """Vectorised mapping. values in [0, 1]; output is uint8 (H, W, 3)."""
    if name not in _TABLES:
        raise KeyError(f"Unknown colormap {name!r}")
    table = np.array([c for _, c in _TABLES[name]], dtype=np.float32)
    stops = np.array([p for p, _ in _TABLES[name]], dtype=np.float32)
    v = np.clip(values.astype(np.float32), 0.0, 1.0)

    # Find segment index per pixel
    idx = np.searchsorted(stops, v, side="right") - 1
    idx = np.clip(idx, 0, len(stops) - 2)

    p0, p1 = stops[idx], stops[idx + 1]
    c0, c1 = table[idx], table[idx + 1]
    span = np.maximum(p1 - p0, 1e-9)
    t = (v - p0) / span
    rgb = c0 + (c1 - c0) * t[..., None]
    return np.clip(rgb, 0, 255).astype(np.uint8)
