#!/usr/bin/env python3
"""生成关键区域裁剪对比图 —— 聚焦已知文字残留位置。"""
from pathlib import Path
import cv2
import numpy as np

BASE = Path("/Users/daiduo2/geoseg/src/3d_schematic")
RESULTS = BASE / "results" / "experiment_plan_repair"
OUT_DIR = RESULTS / "audit_crops"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FONT = cv2.FONT_HERSHEY_SIMPLEX
BG = (30, 30, 30)

# Crop regions for each panel: (y1, y2, x1, x2) in original coordinates
CROPS = {
    1: {
        "top_left": (0, 600, 0, 900),      # Continental crust area
        "mid_left": (1200, 2000, 0, 900),  # partial melting area
        "bottom_right": (2800, 3480, 900, 1740),  # Mantle / bottom
    },
    2: {
        "top_right": (0, 600, 900, 1740),     # uplift area
        "mid_right": (1000, 2200, 900, 1740), # removed mantle lithosphere
        "bottom_right": (2600, 3480, 900, 1740), # Mantle area
    },
    3: {
        "top_right": (0, 600, 900, 1740),     # weak zone area
        "mid_left": (800, 1800, 0, 900),      # refractory / peridotite
        "bottom_left": (2200, 3480, 0, 900),  # Fragments area
    },
}


def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def crop(img: np.ndarray, box: tuple) -> np.ndarray:
    y1, y2, x1, x2 = box
    return img[y1:y2, x1:x2]


def add_label(img: np.ndarray, label: str) -> np.ndarray:
    h, w = img.shape[:2]
    lh = 28
    canvas = np.full((h + lh, w, 3), BG, dtype=np.uint8)
    canvas[lh:, :] = img
    cv2.putText(canvas, label, (4, 20), FONT, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


def build_crop_comparison(panel_idx: int, crop_name: str, box: tuple) -> np.ndarray:
    stem = f"panel_{panel_idx}"
    a_dir = RESULTS / "group_a_replacement"
    b_dir = RESULTS / "group_b_detect_repair"
    c_dir = RESULTS / "group_c_post_smooth"
    d_dir = RESULTS / "group_d_full_pipeline"

    images = []
    labels = []

    # References
    for path, label in [
        (BASE / f"figures/panels/{stem}.png", "original"),
        (BASE / f"experiments/text_removal_v2/final_pipeline/{stem}_final.png", "v2_final"),
        (BASE / f"experiments/text_removal_v2/final_pipeline/{stem}_single.png", "v2_single"),
    ]:
        img = load_rgb(path)
        images.append(crop(img, box))
        labels.append(label)

    # Group A selections
    for path, label in [
        (a_dir / f"{stem}_telea_r3.png", "A_telea_r3"),
        (a_dir / f"{stem}_telea_r7.png", "A_telea_r7"),
        (a_dir / f"{stem}_median_k71.png", "A_median_k71"),
        (a_dir / f"{stem}_biharmonic.png", "A_biharmonic"),
    ]:
        img = load_rgb(path)
        images.append(crop(img, box))
        labels.append(label)

    # Group B selections
    for path, label in [
        (b_dir / f"{stem}_b1_brightness_t15_r3.png", "B1_bright"),
        (b_dir / f"{stem}_b3_grow_t20_k71.png", "B3_grow"),
    ]:
        img = load_rgb(path)
        images.append(crop(img, box))
        labels.append(label)

    # Group C selections
    for path, label in [
        (c_dir / f"{stem}_bilateral_d9_sc75_ss75.png", "C_bilateral"),
    ]:
        img = load_rgb(path)
        images.append(crop(img, box))
        labels.append(label)

    # Group D selections
    for path, label in [
        (d_dir / f"{stem}_d1_telea_full.png", "D1_telea_full"),
    ]:
        img = load_rgb(path)
        images.append(crop(img, box))
        labels.append(label)

    # Resize all to same height
    target_h = 350
    resized = []
    for img in images:
        h, w = img.shape[:2]
        scale = target_h / h
        new_w = int(w * scale)
        resized.append(cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA))

    labeled = [add_label(im, lbl) for im, lbl in zip(resized, labels)]
    return np.hstack(labeled)


def main():
    for panel_idx, crops in CROPS.items():
        for crop_name, box in crops.items():
            print(f"Panel {panel_idx} — {crop_name}...")
            comp = build_crop_comparison(panel_idx, crop_name, box)
            out_path = OUT_DIR / f"panel_{panel_idx}_{crop_name}_crop.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
            print(f"  Saved: {out_path} ({comp.shape[1]}x{comp.shape[0]})")


if __name__ == "__main__":
    main()
