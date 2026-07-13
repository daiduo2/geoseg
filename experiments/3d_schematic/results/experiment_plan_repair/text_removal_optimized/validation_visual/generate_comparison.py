import cv2
import numpy as np
from pathlib import Path

def load(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to load {path}")
    return img

def hstack_with_padding(images, pad=10, bg=(255,255,255)):
    h = max(im.shape[0] for im in images)
    out = []
    for im in images:
        if im.shape[0] < h:
            top = (h - im.shape[0]) // 2
            bot = h - im.shape[0] - top
            im = cv2.copyMakeBorder(im, top, bot, 0, 0, cv2.BORDER_CONSTANT, value=bg)
        out.append(im)
        out.append(np.full((h, pad, 3), bg, dtype=np.uint8))
    return np.concatenate(out[:-1], axis=1)

def add_label(img, text, color=(0,0,0)):
    h, w = img.shape[:2]
    label_h = 40
    canvas = np.full((h + label_h, w, 3), (255,255,255), dtype=np.uint8)
    canvas[label_h:, :] = img
    cv2.putText(canvas, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return canvas

def crop(img, x, y, w, h):
    H, W = img.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)
    return img[y1:y2, x1:x2]

base = Path("/Users/daiduo2/geoseg/src/3d_schematic")
out_dir = base / "results/experiment_plan_repair/text_removal_optimized/validation_visual"
out_dir.mkdir(parents=True, exist_ok=True)

panels = [
    {
        "name": "panel1",
        "orig": base / "figures/panels/panel_1.png",
        "base": base / "experiments/text_removal_v2/final_pipeline/panel_1_final.png",
        "prop": base / "results/experiment_plan_repair/text_removal_optimized/panel_1_proposed.png",
        "crops": [
            ("high_density_text",   20,  20, 180, 220),   # upper-left text block
            ("boundary_zone",      130, 180, 220, 200),   # central boundary area
            ("texture_region",      40, 320, 160, 200),   # lower-left texture
            ("colorbar_legend",    270,  10, 120, 120),   # upper-right corner
            ("top_left_annotation", 10,  10, 150, 120),   # top-left label area
            ("bottom_right_edge",  250, 420, 140, 140),   # bottom-right edge
        ]
    },
    {
        "name": "panel2",
        "orig": base / "figures/panels/panel_2.png",
        "base": base / "experiments/text_removal_v2/final_pipeline/panel_2_final.png",
        "prop": base / "results/experiment_plan_repair/text_removal_optimized/panel_2_proposed.png",
        "crops": [
            ("high_density_text",   20,  20, 180, 220),   # upper-left text
            ("boundary_zone",      130, 180, 220, 200),   # central boundary
            ("texture_region",      40, 320, 160, 200),   # lower-left texture
            ("colorbar_legend",    270,  10, 120, 120),   # upper-right
            ("top_left_annotation", 10,  10, 150, 120),   # top-left
            ("bottom_right_edge",  250, 420, 140, 140),   # bottom-right
        ]
    },
    {
        "name": "panel3",
        "orig": base / "figures/panels/panel_3.png",
        "base": base / "experiments/text_removal_v2/final_pipeline/panel_3_final.png",
        "prop": base / "results/experiment_plan_repair/text_removal_optimized/panel_3_proposed.png",
        "crops": [
            ("high_density_text",   20,  20, 180, 220),   # upper-left text
            ("boundary_zone",      130, 180, 220, 200),   # central boundary
            ("texture_region",      40, 320, 160, 200),   # lower-left texture
            ("colorbar_legend",    270,  10, 120, 120),   # upper-right
            ("top_left_annotation", 10,  10, 150, 120),   # top-left
            ("bottom_right_edge",  250, 420, 140, 140),   # bottom-right
        ]
    },
]

for p in panels:
    orig = load(p["orig"])
    base_img = load(p["base"])
    prop_img = load(p["prop"])
    for region_name, x, y, w, h in p["crops"]:
        c_orig = crop(orig, x, y, w, h)
        c_base = crop(base_img, x, y, w, h)
        c_prop = crop(prop_img, x, y, w, h)
        c_orig = add_label(c_orig, "Original")
        c_base = add_label(c_base, "Baseline")
        c_prop = add_label(c_prop, "Proposed")
        comp = hstack_with_padding([c_orig, c_base, c_prop], pad=10)
        out_path = out_dir / f"{p['name']}_{region_name}.png"
        cv2.imwrite(str(out_path), comp)
        print(f"Saved {out_path}")

print("Done.")
