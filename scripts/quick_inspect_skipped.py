#!/usr/bin/env python3
"""Quick inspect skipped figures to see if any can be salvaged as velocity_model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoseg.modules.cv_detect.figure_classifier import classify

SKIPPED = [
    ("seismic_inv_2024/seismic_inv_2024_page9_img1", "papers_new/to_process/seismic_inv_2024/seismic_inv_2024_page9_img1.png"),
    ("seismic_inv_2024/seismic_inv_2024_page7_img1", "papers_new/to_process/seismic_inv_2024/seismic_inv_2024_page7_img1.png"),
    ("uncertainty_fwi_2024/uncertainty_fwi_2024_page5_img6", "papers_new/to_process/uncertainty_fwi_2024/uncertainty_fwi_2024_page5_img6.png"),
    ("wise_fwi_2024/wise_fwi_2024_page7_img2", "papers_new/to_process/wise_fwi_2024/wise_fwi_2024_page7_img2.png"),
]


def main() -> None:
    for fig_key, img_path_str in SKIPPED:
        img_path = Path(img_path_str)
        if not img_path.exists():
            print(f"SKIP {fig_key}: not found")
            continue

        img_rgb = np.array(Image.open(img_path).convert("RGB"))
        h, w = img_rgb.shape[:2]
        cls = classify(img_rgb)

        print(f"\n{fig_key}")
        print(f"  Size: {w}x{h}")
        print(f"  CV type: {cls['figure_type']}")
        print(f"  Saturation: {cls.get('saturation_ratio', 0):.4f}")
        print(f"  Features: { {k: round(float(v), 3) for k, v in cls.get('features', {}).items()} }")


if __name__ == "__main__":
    main()
