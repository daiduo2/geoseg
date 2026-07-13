"""
Round 2: Fine-tuned sweep based on Round 1 observations.
Best from R1: sw=6, dilate=1. Now explore sw=5,7,8 with dilate=1,
and sw=6,8,10 with dilate=0 (no-dilate sweet spot).
Also test sw=6 with dilate=2 to see if over-blur occurs.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from mser_v2_framework import run_experiment

BASE = Path("/Users/daiduo2/geoseg/src/3d_schematic")
PANEL_3 = [BASE / "figures" / "panels" / "panel_3.png"]
OUT = BASE / "experiments" / "text_removal_v2" / "stroke_width"

print("=" * 60)
print("ROUND 2: Fine-tuned sweep")
print("=" * 60)

# Fine-tune around sw=6 with dilate=1
for sw in [5, 7, 8]:
    print(f"\n--- max_stroke_width={sw}, dilate_iter=1 ---")
    run_experiment(
        PANEL_3, OUT / "round2", f"b180_sw{sw}_d1_i3",
        brightness_thresh=180, max_stroke_width=sw, dilate_iter=1, inpaint_radius=3,
    )

# No-dilate sweet spot exploration
for sw in [6, 8, 10]:
    print(f"\n--- max_stroke_width={sw}, dilate_iter=0 ---")
    run_experiment(
        PANEL_3, OUT / "round2", f"b180_sw{sw}_d0_i3",
        brightness_thresh=180, max_stroke_width=sw, dilate_iter=0, inpaint_radius=3,
    )

# Over-dilate test
print("\n--- max_stroke_width=6, dilate_iter=2 ---")
run_experiment(
    PANEL_3, OUT / "round2", "b180_sw6_d2_i3",
    brightness_thresh=180, max_stroke_width=6, dilate_iter=2, inpaint_radius=3,
)

# Larger inpaint radius with best config
print("\n--- max_stroke_width=6, dilate_iter=1, inpaint_radius=5 ---")
run_experiment(
    PANEL_3, OUT / "round2", "b180_sw6_d1_i5",
    brightness_thresh=180, max_stroke_width=6, dilate_iter=1, inpaint_radius=5,
)

print("\nRound 2 complete. Results in:", OUT / "round2")
