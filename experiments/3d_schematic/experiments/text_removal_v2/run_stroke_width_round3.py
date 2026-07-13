"""
Round 3: Best parameter validation on panel_1 and panel_2.
Best config from R1+R2: max_stroke_width=6, dilate_iter=1, inpaint_radius=3, brightness_thresh=180.
Compare against baseline (no stroke-width constraint).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from mser_v2_framework import run_experiment

BASE = Path("/Users/daiduo2/geoseg/src/3d_schematic")
PANELS_12 = [BASE / "figures" / "panels" / f"panel_{i}.png" for i in (1, 2)]
PANELS_ALL = [BASE / "figures" / "panels" / f"panel_{i}.png" for i in (1, 2, 3)]
OUT = BASE / "experiments" / "text_removal_v2" / "stroke_width"

print("=" * 60)
print("ROUND 3: Validate best config on panel_1 & panel_2")
print("=" * 60)

# Baseline (no stroke-width constraint)
print("\n--- Baseline (no stroke-width constraint) ---")
run_experiment(
    PANELS_12, OUT / "round3", "b180_sw0_d1_i3",
    brightness_thresh=180, max_stroke_width=0, dilate_iter=1, inpaint_radius=3,
)

# Best config
print("\n--- Best: max_stroke_width=6, dilate_iter=1, inpaint_radius=3 ---")
run_experiment(
    PANELS_12, OUT / "round3", "b180_sw6_d1_i3",
    brightness_thresh=180, max_stroke_width=6, dilate_iter=1, inpaint_radius=3,
)

# Also run best config on all 3 panels for final comparison
print("\n--- Best on all panels ---")
run_experiment(
    PANELS_ALL, OUT / "round3", "best_b180_sw6_d1_i3",
    brightness_thresh=180, max_stroke_width=6, dilate_iter=1, inpaint_radius=3,
)

print("\nRound 3 complete. Results in:", OUT / "round3")
