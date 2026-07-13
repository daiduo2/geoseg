#!/usr/bin/env python3
"""生成视觉审计网格图 —— 每个 panel 一张大图，包含所有关键实验结果。"""
from pathlib import Path
import cv2
import numpy as np

BASE = Path("/Users/daiduo2/geoseg/src/3d_schematic")
RESULTS = BASE / "results" / "experiment_plan_repair"
OUT_DIR = RESULTS / "audit_grids"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Each panel's grid layout: rows = experiment groups, cols = selected variants
# Images are resized to a fixed height for the grid
GRID_HEIGHT = 400
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_COLOR = (255, 255, 255)
BG_COLOR = (40, 40, 40)


def load_and_resize(path: Path, target_h: int) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = target_h / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def add_label(img: np.ndarray, label: str) -> np.ndarray:
    h, w = img.shape[:2]
    label_h = 24
    canvas = np.full((h + label_h, w, 3), BG_COLOR, dtype=np.uint8)
    canvas[label_h:, :] = img
    cv2.putText(canvas, label, (4, 18), FONT, FONT_SCALE, FONT_COLOR, 1, cv2.LINE_AA)
    return canvas


def build_row(images: list[np.ndarray | None], labels: list[str]) -> np.ndarray:
    labeled = []
    for img, label in zip(images, labels):
        if img is not None:
            labeled.append(add_label(img, label))
        else:
            # placeholder
            ph = np.full((GRID_HEIGHT + 24, GRID_HEIGHT, 3), BG_COLOR, dtype=np.uint8)
            labeled.append(add_label(ph, f"{label} (missing)"))
    # Pad to same height
    max_h = max(im.shape[0] for im in labeled)
    padded = []
    for im in labeled:
        h, w = im.shape[:2]
        if h < max_h:
            pad = np.full((max_h, w, 3), BG_COLOR, dtype=np.uint8)
            pad[:h, :] = im
            padded.append(pad)
        else:
            padded.append(im)
    return np.hstack(padded)


def build_panel_grid(panel_idx: int) -> np.ndarray:
    stem = f"panel_{panel_idx}"
    rows = []
    row_labels = []

    # Row 0: References
    refs = [
        (BASE / f"figures/panels/{stem}.png", "original"),
        (BASE / f"experiments/text_removal_v2/final_pipeline/{stem}_final.png", "v2_final"),
        (BASE / f"experiments/text_removal_v2/final_pipeline/{stem}_single.png", "v2_single"),
    ]
    row_imgs = [load_and_resize(p, GRID_HEIGHT) for p, _ in refs]
    rows.append(build_row(row_imgs, [l for _, l in refs]))
    row_labels.append("REFERENCES")

    # Row 1: Group A — replacement algorithms
    a_dir = RESULTS / "group_a_replacement"
    a_items = [
        (a_dir / f"{stem}_telea_r3.png", "telea_r3"),
        (a_dir / f"{stem}_telea_r7.png", "telea_r7"),
        (a_dir / f"{stem}_ns_r3.png", "ns_r3"),
        (a_dir / f"{stem}_ns_r7.png", "ns_r7"),
        (a_dir / f"{stem}_median_k71.png", "median_k71"),
        (a_dir / f"{stem}_biharmonic.png", "biharmonic"),
        (a_dir / f"{stem}_baseline_inpaint3_median71.png", "baseline"),
    ]
    row_imgs = [load_and_resize(p, GRID_HEIGHT) for p, _ in a_items]
    rows.append(build_row(row_imgs, [l for _, l in a_items]))
    row_labels.append("GROUP A: repair replacement")

    # Row 2: Group B — detect + repair
    b_dir = RESULTS / "group_b_detect_repair"
    b_items = [
        (b_dir / f"{stem}_b1_brightness_t15_r3.png", "b1_bright_t15"),
        (b_dir / f"{stem}_b2_dog_1.5_3.0_t25.png", "b2_dog"),
        (b_dir / f"{stem}_b3_grow_t20_k71.png", "b3_grow_t20"),
        (b_dir / f"{stem}_b4_diff_t20.png", "b4_diff_t20"),
    ]
    row_imgs = [load_and_resize(p, GRID_HEIGHT) for p, _ in b_items]
    rows.append(build_row(row_imgs, [l for _, l in b_items]))
    row_labels.append("GROUP B: detect+repair")

    # Row 3: Group C — post smoothing
    c_dir = RESULTS / "group_c_post_smooth"
    c_items = [
        (c_dir / f"{stem}_bilateral_d9_sc75_ss75.png", "bilateral_d9"),
        (c_dir / f"{stem}_gaussian_k11_s2.0.png", "gaussian_k11"),
    ]
    row_imgs = [load_and_resize(p, GRID_HEIGHT) for p, _ in c_items]
    rows.append(build_row(row_imgs, [l for _, l in c_items]))
    row_labels.append("GROUP C: post smooth")

    # Row 4: Group D — full pipeline
    d_dir = RESULTS / "group_d_full_pipeline"
    d_items = [
        (d_dir / f"{stem}_d1_telea_full.png", "d1_telea_full"),
        (d_dir / f"{stem}_d2_biharmonic_full.png", "d2_biharmonic_full"),
        (d_dir / f"{stem}_d4_telea_biharmonic.png", "d4_telea+biharm"),
    ]
    row_imgs = [load_and_resize(p, GRID_HEIGHT) for p, _ in d_items]
    rows.append(build_row(row_imgs, [l for _, l in d_items]))
    row_labels.append("GROUP D: full pipeline")

    # Stack rows with labels on the left
    max_w = max(r.shape[1] for r in rows)
    label_w = 220
    full_rows = []
    for row, label in zip(rows, row_labels):
        h = row.shape[0]
        label_canvas = np.full((h, label_w, 3), BG_COLOR, dtype=np.uint8)
        cv2.putText(label_canvas, label, (4, h // 2), FONT, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        # Pad row to max_w
        if row.shape[1] < max_w:
            pad = np.full((h, max_w - row.shape[1], 3), BG_COLOR, dtype=np.uint8)
            row = np.hstack([row, pad])
        full_rows.append(np.hstack([label_canvas, row]))

    return np.vstack(full_rows)


def main():
    for i in (1, 2, 3):
        print(f"Building audit grid for panel_{i}...")
        grid = build_panel_grid(i)
        out_path = OUT_DIR / f"panel_{i}_audit_grid.png"
        cv2.imwrite(str(out_path), grid)
        print(f"  Saved: {out_path} ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    main()
