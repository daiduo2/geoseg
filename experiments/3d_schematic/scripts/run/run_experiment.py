import sys; sys.path.insert(0, "/Users/daiduo2/geoseg/src/3d_schematic")
from process_final_v3 import *
from pathlib import Path

base = Path(__file__).parent.parent.parent
for i in range(1, 4):
    p = base / f"panel_{i}_front.png"
    img = np.array(Image.open(p).convert("RGB"))
    cleaned = remove_text(img)
    enhanced = enhance_v(cleaned)
    labels = felzenszwalb(enhanced, scale=200, sigma=0.3, min_size=20)
    labels = post_merge(labels, enhanced)
    fill = render_label_fill(labels)
    boundaries = draw_boundaries(fill, labels)
    Image.fromarray(boundaries).save(base / f"result_felz_finer_{i}.png")
    print(f"Panel {i}: {len(np.unique(labels))} labels")
