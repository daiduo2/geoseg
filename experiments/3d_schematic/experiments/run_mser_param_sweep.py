"""
MSER 参数扫描实验脚本。
扫描 max_area, max_aspect, min_area，配合 brightness_thresh=180。
先跑 panel_3，再推广到 panel_1, panel_2。
"""
import cv2
import numpy as np
from pathlib import Path
import itertools

from mser_v2_framework import run_experiment

BASE = Path("/Users/daiduo2/geoseg/src/3d_schematic")
PANEL_3 = [BASE / "figures" / "panels" / "panel_3.png"]
PANEL_1 = [BASE / "figures" / "panels" / "panel_1.png"]
PANEL_2 = [BASE / "figures" / "panels" / "panel_2.png"]
OUT = BASE / "experiments" / "text_removal_v2" / "mser_params"

# 参数扫描空间
MAX_AREAS = [800, 1200, 1500, 2000]
MAX_ASPECTS = [10, 15, 20]
MIN_AREAS = [10, 30, 50]

def sweep(panels, out_subdir, prefix):
    out = OUT / out_subdir
    out.mkdir(parents=True, exist_ok=True)
    for max_area, max_aspect, min_area in itertools.product(MAX_AREAS, MAX_ASPECTS, MIN_AREAS):
        suffix = f"{prefix}_ma{max_area}_asp{max_aspect}_mi{min_area}"
        print(f"\n=== {suffix} ===")
        run_experiment(
            panels, out, suffix,
            brightness_thresh=180,
            dilate_iter=1,
            inpaint_radius=3,
            min_area=min_area,
            max_area=max_area,
            max_aspect=max_aspect,
            lap_threshold=15,
        )

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1: Panel 3 sweep")
    print("=" * 60)
    sweep(PANEL_3, "panel3_sweep", "p3")

    print("\n" + "=" * 60)
    print("Phase 2: Panel 1 + 2 validation (best params)")
    print("=" * 60)
    # 先跑全扫描，后续根据最佳参数再跑 panel_1/2
    sweep(PANEL_1, "panel1_sweep", "p1")
    sweep(PANEL_2, "panel2_sweep", "p2")
