"""Realistic-noise simulation for synthetic geoseg panels.

The whole point of the synthetic benchmark is to expose weaknesses in
segmentation engines that only show up on real-world panels. So we
intentionally simulate four classes of artifact seen in published
geoscience figures:

1. JPEG-style compression noise (low-amplitude additive Gaussian).
2. Black rectangular text blocks at random positions (caption strips,
   sub-panel labels, axis ticks).
3. Contour lines (white-on-colour overlays at regular value intervals).
4. A vertical colourbar on the right edge.

All four are no-ops by default; the generator builds the clean image
and the noise step is opt-in per benchmark config.
"""

from __future__ import annotations

import numpy as np


def _add_jitter(rgb: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    if std <= 0:
        return rgb
    noise = rng.normal(0, std, size=rgb.shape).astype(np.float32)
    out = rgb.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def _add_text_blocks(
    rgb: np.ndarray,
    n_blocks: int,
    rng: np.random.Generator,
    min_size: int = 14,
    max_size: int = 28,
) -> tuple[np.ndarray, np.ndarray]:
    """Stamp black rectangular blocks; return (rgb, mask).

    The mask is also returned so the evaluator can mark those pixels as
    "ignore" when computing mIoU (text regions are not part of the
    segmentation target).
    """
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    if n_blocks <= 0:
        return rgb, mask

    out = rgb.copy()
    for _ in range(n_blocks):
        bh = int(rng.integers(min_size, max_size + 1))
        bw = int(rng.integers(min_size * 2, max_size * 4 + 1))
        y0 = int(rng.integers(0, max(1, h - bh)))
        x0 = int(rng.integers(0, max(1, w - bw)))
        out[y0 : y0 + bh, x0 : x0 + bw] = (0, 0, 0)
        mask[y0 : y0 + bh, x0 : x0 + bw] = True
    return out, mask


def _add_contour_lines(
    rgb: np.ndarray,
    n_levels: int,
    rng: np.random.Generator,
    value_field: np.ndarray | None = None,
) -> np.ndarray:
    """Draw white contour lines at regular scalar-field intervals.

    If `value_field` is None, we draw evenly spaced rows.
    """
    h, w = rgb.shape[:2]
    out = rgb.copy()
    if n_levels <= 0:
        return out

    if value_field is not None and value_field.shape == (h, w):
        levels = np.linspace(0.05, 0.95, n_levels + 2)[1:-1]
        for lvl in levels:
            edge = np.abs(value_field - lvl) < 0.01
            out[edge] = (255, 255, 255)
    else:
        ys = np.linspace(0, h - 1, n_levels + 2)[1:-1].astype(int)
        for y in ys:
            out[y] = (255, 255, 255)
    return out


def _add_colorbar(
    rgb: np.ndarray,
    width_px: int = 24,
    colormap: str = "jet",
) -> tuple[np.ndarray, np.ndarray]:
    """Append a vertical colormap gradient on the right edge.

    `colormap` should match the panel's own colormap so the
    colorbar_guided engine samples the correct palette (rather than
    always seeing jet, which would mismatch viridis/plasma/gray
    panels). Returns (rgb_with_colorbar, colorbar_rgb).
    """
    h, w = rgb.shape[:2]
    out = rgb.copy()

    gradient = np.linspace(0, 1, h).reshape(-1, 1).repeat(width_px, axis=1)
    from synthetic_benchmark.colormaps import map_array

    cb_rgb = map_array(gradient, colormap)
    out[:, w - width_px :] = cb_rgb
    return out, cb_rgb


def apply_noise(
    panel_rgb: np.ndarray,
    gt_labels: np.ndarray,
    *,
    jpeg_std: float = 4.0,
    n_text_blocks: int = 4,
    n_contours: int = 6,
    add_colorbar: bool = True,
    colormap: str = "jet",
    value_field: np.ndarray | None = None,
    seed: int = 0,
) -> dict:
    """Run the full noise pipeline. Returns a dict of artefacts.

    Returns dict with keys:
      - rgb       : final RGB after noise (uint8, H', W')
      - gt_labels : unchanged ground truth, but cropped to match the
                    original panel (we do NOT crop here — colorbar is
                    added in-place on the right, so gt_labels stays the
                    same width; engines that only look at the left part
                    of the panel are responsible for cropping).
      - text_mask : boolean mask over the original panel where text
                    blocks were stamped; evaluator should exclude these.
      - colorbar_rgb : the appended colorbar (or None if disabled).

    The pipeline is intentionally simple so each step can be re-run
    independently during debugging.
    """
    rng = np.random.default_rng(seed)

    rgb = _add_jitter(panel_rgb, jpeg_std, rng)
    rgb, text_mask = _add_text_blocks(rgb, n_text_blocks, rng)
    rgb = _add_contour_lines(rgb, n_contours, rng, value_field=value_field)

    colorbar_rgb = None
    if add_colorbar:
        rgb, colorbar_rgb = _add_colorbar(rgb, colormap=colormap)

    return {
        "rgb": rgb,
        "gt_labels": gt_labels,
        "text_mask": text_mask,
        "colorbar_rgb": colorbar_rgb,
    }
