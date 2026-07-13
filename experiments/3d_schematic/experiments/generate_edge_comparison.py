#!/usr/bin/env python3
"""生成边界保持度对比图 —— 聚焦地质层位交界线。"""
from pathlib import Path
import cv2
import numpy as np

BASE = Path("/Users/daiduo2/geoseg/src/3d_schematic")
RESULTS = BASE / "results" / "experiment_plan_repair"
OUT_DIR = RESULTS / "audit_edges"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FONT = cv2.FONT_HERSHEY_SIMPLEX
BG = (30, 30, 30)

# Edge regions: (y1, y2, x1, x2)
EDGES = {
    1: {
        "gray_blue_boundary": (300, 700, 400, 1300),
        "blue_orange_boundary": (700, 1100, 400, 1300),
        "orange_yellow_boundary": (1600, 2200, 400, 1300),
    },
    2: {
        "gray_blue_boundary": (300, 700, 400, 1300),
        "blue_orange_boundary": (700, 1100, 400, 1300),
        "orange_yellow_boundary": (1600, 2200, 400, 1300),
    },
    3: {
        "gray_blue_boundary": (300, 700, 400, 1300),
        "green_orange_boundary": (700, 1200, 400, 1300),
        "orange_yellow_boundary": (1800, 2600, 400, 1300),
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


def build_edge_comparison(panel_idx: int, edge_name: str, box: tuple) -> np.ndarray:
    stem = f"panel_{panel_idx}"
    a_dir = RESULTS / "group_a_replacement"
    c_dir = RESULTS / "group_c_post_smooth"

    images = []
    labels = []

    # Original as reference for edge sharpness
    orig = load_rgb(BASE / f"figures/panels/{stem}.png")
    images.append(crop(orig, box))
    labels.append("original")

    # Final as baseline
    final = load_rgb(BASE / f"experiments/text_removal_v2/final_pipeline/{stem}_final.png")
    images.append(crop(final, box))
    labels.append("v2_final")

    # Key algorithms to compare
    for path, label in [
        (a_dir / f"{stem}_telea_r3.png", "telea_r3"),
        (a_dir / f"{stem}_telea_r7.png", "telea_r7"),
        (a_dir / f"{stem}_ns_r3.png", "ns_r3"),
        (a_dir / f"{stem}_median_k71.png", "median_k71"),
        (c_dir / f"{stem}_bilateral_d9_sc75_ss75.png", "bilateral"),
        (c_dir / f"{stem}_gaussian_k11_s2.0.png", "gaussian"),
    ]:
        img = load_rgb(path)
        images.append(crop(img, box))
        labels.append(label)

    target_h = 280
    resized = []
    for img in images:
        h, w = img.shape[:2]
        scale = target_h / h
        new_w = int(w * scale)
        resized.append(cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA))

    labeled = [add_label(im, lbl) for im, lbl in zip(resized, labels)]
    return np.hstack(labeled)


def main():
    for panel_idx, edges in EDGES.items():
        for edge_name, box in edges.items():
            print(f"Panel {panel_idx} — {edge_name}...")
            comp = build_edge_comparison(panel_idx, edge_name, box)
            out_path = OUT_DIR / f"panel_{panel_idx}_{edge_name}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
            print(f"  Saved: {out_path} ({comp.shape[1]}x{comp.shape[0]})")


if __name__ == "__main__":
    main()
