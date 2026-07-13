#!/usr/bin/env python3
"""
Agent-2: 阈值化 + 扩展参数扫描实验

扫描 diff_thresh 和 expand_radius，观察叠层掩码对文字的覆盖完整度。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# 把 design_diff_overlay.py 加入路径
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from design_diff_overlay import extract_detail_layer, create_overlay_mask


def generate_overlay_visualizations(
    image: np.ndarray,
    detail: np.ndarray,
    overlay_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """生成三种可视化图."""
    # 差分图（归一化到 0-255）
    detail_norm = (detail / detail.max() * 255).astype(np.uint8)

    # 叠层掩码图（白色=叠层，黑色=非叠层）
    overlay_uint8 = overlay_mask.astype(np.uint8) * 255

    # 叠层叠加图（原图上用洋红色标记叠层区域）
    overlay_vis = image.copy()
    overlay_vis[overlay_mask] = [255, 0, 255]  # 洋红色 BGR

    return {
        "detail": detail_norm,
        "overlay": overlay_uint8,
        "overlay_vis": overlay_vis,
    }


def run_parameter_scan(
    image_path: Path,
    out_dir: Path,
    blur_ksize: int,
    blur_sigma: float,
    diff_thresh_values: list[float],
    expand_radius_values: list[int],
) -> None:
    """对单张图跑两组参数扫描."""
    img = np.array(Image.open(image_path).convert("RGB"))
    panel_name = image_path.stem  # e.g. "panel_1_front"

    # Step 1: 固定差分参数提取 detail（两组扫描共用）
    detail = extract_detail_layer(img, blur_ksize, blur_sigma)

    # ---- 扫描 A: diff_thresh ----
    print(f"\n[{panel_name}] Scanning diff_thresh = {diff_thresh_values} (expand_radius=15)")
    for diff_thresh in diff_thresh_values:
        overlay_mask = create_overlay_mask(detail, diff_thresh, expand_radius=15)
        vis = generate_overlay_visualizations(img, detail, overlay_mask)

        suffix = f"_{panel_name}_t{diff_thresh}_e15"
        Image.fromarray(vis["detail"]).save(out_dir / f"detail{suffix}.png")
        Image.fromarray(vis["overlay"]).save(out_dir / f"overlay{suffix}.png")
        Image.fromarray(vis["overlay_vis"]).save(out_dir / f"overlay_vis{suffix}.png")

        coverage = overlay_mask.mean() * 100
        print(f"  diff_thresh={diff_thresh:2.0f} -> overlay coverage: {coverage:.2f}%")

    # ---- 扫描 B: expand_radius ----
    print(f"\n[{panel_name}] Scanning expand_radius = {expand_radius_values} (diff_thresh=20)")
    for expand_radius in expand_radius_values:
        overlay_mask = create_overlay_mask(detail, diff_thresh=20.0, expand_radius=expand_radius)
        vis = generate_overlay_visualizations(img, detail, overlay_mask)

        suffix = f"_{panel_name}_t20_e{expand_radius}"
        Image.fromarray(vis["detail"]).save(out_dir / f"detail{suffix}.png")
        Image.fromarray(vis["overlay"]).save(out_dir / f"overlay{suffix}.png")
        Image.fromarray(vis["overlay_vis"]).save(out_dir / f"overlay_vis{suffix}.png")

        coverage = overlay_mask.mean() * 100
        print(f"  expand_radius={expand_radius:2d} -> overlay coverage: {coverage:.2f}%")


def main() -> None:
    base = Path("/Users/daiduo2/geoseg/src/3d_schematic")
    out_dir = base / "diff_overlay_experiments" / "agent2_overlay_masks"
    out_dir.mkdir(parents=True, exist_ok=True)

    panels = [
        base / "panel_1_front.png",
        base / "panel_2_front.png",
        base / "panel_3_front.png",
    ]

    # 固定差分参数
    blur_ksize = 15
    blur_sigma = 3.0

    # 扫描参数
    diff_thresh_values = [10, 15, 20, 30, 50]
    expand_radius_values = [5, 10, 15, 20, 30]

    for panel_path in panels:
        if not panel_path.exists():
            print(f"Warning: {panel_path} not found, skipping.")
            continue
        run_parameter_scan(
            panel_path,
            out_dir,
            blur_ksize,
            blur_sigma,
            diff_thresh_values,
            expand_radius_values,
        )

    print(f"\nAll results saved to: {out_dir}")


if __name__ == "__main__":
    main()
