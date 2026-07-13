#!/usr/bin/env python3
"""Analyze vertical_scan_reps behavior on ml_velocity and tomography_review panels."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoseg.modules.segment_engines.vlm_reps import vertical_scan_reps
from geoseg.modules.segment_engines._shared import _estimate_background_color

TEST_IMAGES = [
    ("ml_velocity page22_img2", "papers_new/to_process/ml_velocity_2024/ml_velocity_2024_page22_img2.png"),
    ("ml_velocity page22_img3", "papers_new/to_process/ml_velocity_2024/ml_velocity_2024_page22_img3.png"),
    ("tomography_review page5_img2", "papers_new/to_process/tomography_review_2024/tomography_review_2024_page5_img2.png"),
    ("tomography_review page8_img3", "papers_new/to_process/tomography_review_2024/tomography_review_2024_page8_img3.png"),
]


def analyze_panel(img_rgb: np.ndarray, name: str) -> None:
    from skimage.color import rgb2lab
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    h, w = img_rgb.shape[:2]
    cx = w // 2
    cols = [max(0, min(cx + i, w - 1)) for i in range(-1, 2)]
    column_rgb = img_rgb[:, cols, :].mean(axis=1).astype(np.uint8)
    column_lab = rgb2lab(column_rgb)
    column_lab_smooth = gaussian_filter1d(column_lab, sigma=max(1.5, h / 300), axis=0)
    diffs = np.linalg.norm(np.diff(column_lab_smooth, axis=0), axis=1)

    print(f"\n{name}: {w}x{h}")
    print(f"  diffs stats: min={diffs.min():.2f} max={diffs.max():.2f} mean={diffs.mean():.2f} median={np.median(diffs):.2f}")
    for p in [50, 60, 70, 80, 85, 90, 95]:
        print(f"  p{p}: {np.percentile(diffs, p):.2f}")

    for n_hint in [3, 4, 5, 6, 7]:
        min_dist = max(8, h // (n_hint * 2))
        for abs_min in [2.0, 3.0, 5.0, 8.0]:
            for rel_p in [60, 70, 80, 85, 90]:
                threshold = max(abs_min, np.percentile(diffs, rel_p))
                peaks, _ = find_peaks(diffs, height=threshold, distance=min_dist)
                n_peaks = len(peaks)
                n_layers = n_peaks + 1
                print(f"  hint={n_hint} abs_min={abs_min} rel_p={rel_p} -> threshold={threshold:.2f} min_dist={min_dist} peaks={n_peaks} layers={n_layers}")

    # Default behavior
    reps = vertical_scan_reps(img_rgb, n_layers_hint=5)
    print(f"  DEFAULT vertical_scan: {len(reps)} reps")


def main() -> None:
    for name, path_str in TEST_IMAGES:
        img_path = Path(path_str)
        if not img_path.exists():
            continue
        img_rgb = np.array(Image.open(img_path).convert("RGB"))
        analyze_panel(img_rgb, name)


if __name__ == "__main__":
    main()
