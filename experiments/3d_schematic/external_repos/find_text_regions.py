"""Find text-heavy regions per panel for PatchMatch testing."""
import numpy as np
import cv2
from pathlib import Path

BASE = Path("/Users/daiduo2/geoseg/src/3d_schematic")

for idx in [1, 2, 3]:
    stem = f"panel_{idx}"
    image = cv2.imread(str(BASE / f"figures/panels/{stem}.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(BASE / f"experiments/text_removal_v2/final_pipeline/{stem}_mask.png"), cv2.IMREAD_GRAYSCALE)

    h, w = mask.shape
    print(f"\n=== {stem} ({h}x{w}) ===")

    # Slide 200x200 windows, find top-5 by hole density
    step = 100
    scores = []
    for y in range(0, h - 200, step):
        for x in range(0, w - 200, step):
            crop = mask[y:y+200, x:x+200]
            hole = np.sum(crop > 127)
            scores.append((hole, y, x))

    scores.sort(reverse=True)
    for hole, y, x in scores[:5]:
        print(f"  hole={hole:4d} @ ({y:4d}, {x:4d}) - ({y+200}, {x+200})  pct={hole/40000*100:.1f}%")

    # Also check larger 300x300 regions
    scores_large = []
    for y in range(0, h - 300, 150):
        for x in range(0, w - 300, 150):
            crop = mask[y:y+300, x:x+300]
            hole = np.sum(crop > 127)
            scores_large.append((hole, y, x))
    scores_large.sort(reverse=True)
    print("  Top 300x300 regions:")
    for hole, y, x in scores_large[:5]:
        print(f"    hole={hole:4d} @ ({y:4d}, {x:4d})  pct={hole/90000*100:.1f}%")
