#!/usr/bin/env python3
"""
Agent-1: 差分提取参数扫描实验
扫描高斯模糊参数，观察差分图对文字和地质结构的分离效果。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 添加项目根目录到路径，以便导入 design_diff_overlay
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from design_diff_overlay import extract_detail_layer


def normalize_detail(detail: np.ndarray) -> np.ndarray:
    """将差分图归一化到 0-255 uint8."""
    dmax = detail.max()
    if dmax > 0:
        normalized = (detail / dmax * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(detail, dtype=np.uint8)
    return normalized


def save_detail_image(detail: np.ndarray, path: Path) -> None:
    """保存归一化差分图."""
    normalized = normalize_detail(detail)
    Image.fromarray(normalized).save(path)


def run_parameter_scan(
    image_path: Path,
    out_dir: Path,
    ksize_list: list[int] | None = None,
    sigma_list: list[float] | None = None,
    fixed_sigma: float | None = None,
    fixed_ksize: int | None = None,
) -> dict:
    """
    对单张图运行参数扫描。

    Returns:
        dict: {param_str: detail_array}
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    results = {}

    if ksize_list is not None and fixed_sigma is not None:
        # 固定 sigma，扫描 ksize
        for ksize in ksize_list:
            detail = extract_detail_layer(img, blur_ksize=ksize, blur_sigma=fixed_sigma)
            param_str = f"k{ksize}_s{fixed_sigma}"
            results[param_str] = detail
            save_detail_image(detail, out_dir / f"{image_path.stem}_{param_str}.png")

    if sigma_list is not None and fixed_ksize is not None:
        # 固定 ksize，扫描 sigma
        for sigma in sigma_list:
            detail = extract_detail_layer(img, blur_ksize=fixed_ksize, blur_sigma=sigma)
            param_str = f"k{fixed_ksize}_s{sigma}"
            results[param_str] = detail
            save_detail_image(detail, out_dir / f"{image_path.stem}_{param_str}.png")

    return results


def build_comparison_grid(
    all_results: dict[str, dict[str, np.ndarray]],
    param_labels: list[str],
    out_path: Path,
) -> None:
    """
    生成对比总图。

    Args:
        all_results: {panel_name: {param_str: detail_array}}
        param_labels: 参数标签列表，决定列顺序
        out_path: 输出图片路径
    """
    panel_names = sorted(all_results.keys())
    n_rows = len(panel_names)
    n_cols = len(param_labels)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.5))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for row_idx, panel_name in enumerate(panel_names):
        panel_results = all_results[panel_name]
        for col_idx, param_label in enumerate(param_labels):
            ax = axes[row_idx, col_idx]
            detail = panel_results[param_label]
            normalized = normalize_detail(detail)
            ax.imshow(normalized, cmap="gray")
            ax.set_title(f"{panel_name}\n{param_label}", fontsize=10)
            ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison grid saved to {out_path}")


def main() -> None:
    base_dir = Path("/Users/daiduo2/geoseg/src/3d_schematic")
    out_dir = base_dir / "diff_overlay_experiments" / "agent1_detail_extraction"
    out_dir.mkdir(parents=True, exist_ok=True)

    panels = [
        base_dir / "panel_1_front.png",
        base_dir / "panel_2_front.png",
        base_dir / "panel_3_front.png",
    ]

    # 参数扫描配置
    ksize_scan = [7, 11, 15, 21, 31]
    fixed_sigma_for_ksize = 3.0

    sigma_scan = [1.0, 2.0, 3.0, 5.0]
    fixed_ksize_for_sigma = 15

    all_results = {}

    for panel_path in panels:
        panel_name = panel_path.stem
        print(f"\nProcessing {panel_name}...")
        all_results[panel_name] = {}

        # Scan 1: 固定 sigma=3.0，扫描 ksize
        print(f"  Scanning ksize with fixed sigma={fixed_sigma_for_ksize}")
        ksize_results = run_parameter_scan(
            panel_path,
            out_dir,
            ksize_list=ksize_scan,
            fixed_sigma=fixed_sigma_for_ksize,
        )
        all_results[panel_name].update(ksize_results)

        # Scan 2: 固定 ksize=15，扫描 sigma
        print(f"  Scanning sigma with fixed ksize={fixed_ksize_for_sigma}")
        sigma_results = run_parameter_scan(
            panel_path,
            out_dir,
            sigma_list=sigma_scan,
            fixed_ksize=fixed_ksize_for_sigma,
        )
        all_results[panel_name].update(sigma_results)

    # 生成对比总图 1: ksize 扫描 (3 rows x 5 cols)
    ksize_labels = [f"k{ksize}_s{fixed_sigma_for_ksize}" for ksize in ksize_scan]
    build_comparison_grid(
        all_results,
        ksize_labels,
        out_dir / "comparison_ksize_scan.png",
    )

    # 生成对比总图 2: sigma 扫描 (3 rows x 4 cols)
    sigma_labels = [f"k{fixed_ksize_for_sigma}_s{sigma}" for sigma in sigma_scan]
    build_comparison_grid(
        all_results,
        sigma_labels,
        out_dir / "comparison_sigma_scan.png",
    )

    # 生成综合对比图：ksize + sigma 合并 (3 rows x 9 cols)
    all_labels = ksize_labels + sigma_labels
    build_comparison_grid(
        all_results,
        all_labels,
        out_dir / "comparison_all_params.png",
    )

    print(f"\nAll results saved to {out_dir}")


if __name__ == "__main__":
    main()
