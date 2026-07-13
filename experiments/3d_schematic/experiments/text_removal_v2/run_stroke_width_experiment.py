"""
Stroke-width-constrained text removal experiment.
Parameter sweep: max_stroke_width, with/without dilation, on panel_3 first.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from mser_v2_framework import run_experiment

BASE = Path("/Users/daiduo2/geoseg/src/3d_schematic")
PANEL_3 = [BASE / "figures" / "panels" / "panel_3.png"]
PANELS_ALL = [BASE / "figures" / "panels" / f"panel_{i}.png" for i in (1, 2, 3)]
OUT = BASE / "experiments" / "text_removal_v2" / "stroke_width"

# --- Round 1: panel_3 parameter sweep ---
print("=" * 60)
print("ROUND 1: Panel 3 sweep")
print("=" * 60)

# Baseline: no stroke-width constraint (max_stroke_width=0)
print("\n--- Baseline (no stroke-width constraint) ---")
run_experiment(
    PANEL_3, OUT / "round1", "b180_sw0_d1_i3",
    brightness_thresh=180, max_stroke_width=0, dilate_iter=1, inpaint_radius=3,
)

# Stroke-width sweep with dilation
for sw in [4, 6, 8, 10, 12]:
    print(f"\n--- max_stroke_width={sw}, dilate_iter=1 ---")
    run_experiment(
        PANEL_3, OUT / "round1", f"b180_sw{sw}_d1_i3",
        brightness_thresh=180, max_stroke_width=sw, dilate_iter=1, inpaint_radius=3,
    )

# Stroke-width sweep WITHOUT dilation
for sw in [4, 6, 8, 10, 12]:
    print(f"\n--- max_stroke_width={sw}, dilate_iter=0 ---")
    run_experiment(
        PANEL_3, OUT / "round1", f"b180_sw{sw}_d0_i3",
        brightness_thresh=180, max_stroke_width=sw, dilate_iter=0, inpaint_radius=3,
    )

print("\nRound 1 complete. Results in:", OUT / "round1")
