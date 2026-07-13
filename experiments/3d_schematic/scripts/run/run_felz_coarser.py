import sys; sys.path.insert(0, "/Users/daiduo2/geoseg/src/3d_schematic")
from process_final_v3 import *
from pathlib import Path

base = Path(__file__).parent.parent.parent
for i in range(1, 4):
    p = base / f"panel_{i}_front.png"
    img = np.array(Image.open(p).convert("RGB"))
    cleaned = remove_text(img)
    enhanced = enhance_v(cleaned)
    labels = felzenszwalb(enhanced, scale=500, sigma=1.0, min_size=50)
    labels = post_merge(labels, enhanced, small_ratio=0.01, max_score=0.6)
    fill = render_label_fill(labels)
    boundaries = draw_boundaries(fill, labels)
    Image.fromarray(boundaries).save(base / f"result_felz_coarser_{i}.png")
    print(f"Panel {i}: {len(np.unique(labels))} labels")
