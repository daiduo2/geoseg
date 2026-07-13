"""
MSER 参数扫描第二轮：聚焦 max_area 和 brightness_thresh。
基于第一轮发现：
- max_area 是主要杠杆，aspect/min_area 影响微弱
- brightness_thresh=180 不足以过滤树状结构（因为文字和树状结构都是高亮）
- 需要探索 lower max_area + 配合其他过滤
"""
import cv2
import numpy as np
from pathlib import Path

from mser_v2_framework import run_experiment

BASE = Path("/Users/daiduo2/geoseg/src/3d_schematic")
PANEL_3 = [BASE / "figures" / "panels" / "panel_3.png"]
PANEL_1 = [BASE / "figures" / "panels" / "panel_1.png"]
PANEL_2 = [BASE / "figures" / "panels" / "panel_2.png"]
OUT = BASE / "experiments" / "text_removal_v2" / "mser_params"

# 第二轮参数：更激进的 max_area 降低 + 不同 brightness_thresh
MAX_AREAS = [400, 600, 800]
BRIGHTNESS_THRESHES = [180, 200, 220]
MAX_ASPECTS = [10, 15]

def sweep_round2(panels, out_subdir, prefix):
    out = OUT / out_subdir
    out.mkdir(parents=True, exist_ok=True)
    for max_area, brightness_thresh, max_aspect in [(a, b, c) for a in MAX_AREAS for b in BRIGHTNESS_THRESHES for c in MAX_ASPECTS]:
        suffix = f"{prefix}_ma{max_area}_bt{brightness_thresh}_asp{max_aspect}"
        print(f"\n=== {suffix} ===")
        run_experiment(
            panels, out, suffix,
            brightness_thresh=brightness_thresh,
            dilate_iter=1,
            inpaint_radius=3,
            min_area=30,
            max_area=max_area,
            max_aspect=max_aspect,
            lap_threshold=15,
        )

if __name__ == "__main__":
    print("=" * 60)
    print("Round 2: Aggressive max_area + brightness sweep")
    print("=" * 60)
    sweep_round2(PANEL_3, "panel3_round2", "p3")
    sweep_round2(PANEL_1, "panel1_round2", "p1")
    sweep_round2(PANEL_2, "panel2_round2", "p2")
