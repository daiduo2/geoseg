"""
局部二次修复 - 最终版本。

最佳策略（经 Panel 1/2/3 验证）：
1. 第一次修复：remove_text_mser_v2(brightness_thresh=170, dilate_iter=1, inpaint_radius=3)
2. 残留检测：在修复后图像上重新运行 MSER+Laplacian 文字检测，与第一次 mask 取交集得到种子
3. 区域生长：从种子出发，在 mask 区域内向亮度相似的像素生长（阈值=20）
4. 轻微膨胀：3x3 kernel, 1 iteration
5. 二次修复：对 mask 区域用 71x71 median blur 的值直接替换

关键洞察：
- 简单膨胀 mask 会波及地质结构线 → 区域生长更精确
- inpaint 对模糊斑块效果有限 → 直接用大核 median 替换更有效
- 71x71 median 在清除残留和保护背景之间达到最佳平衡
"""
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from mser_v2_framework import remove_text_mser_v2, detect_text_mser, detect_text_laplacian


def detect_residual_region_growing(
    first_result,
    first_mask,
    mser_min_area=5,
    mser_max_area=3000,
    mser_max_aspect=30,
    lap_threshold=10,
    lap_max_area=3000,
    grow_threshold=20,
    dilate_kernel_size=3,
    dilate_iterations=1,
):
    """
    检测第一次修复后的文字残留。

    策略：在修复后图像上重新运行文字检测 → 与第一次 mask 取交集得种子
    → 区域生长扩展 → 轻微膨胀覆盖文字主体。
    """
    mask_bool = first_mask.astype(bool)
    gray_repaired = cv2.cvtColor(first_result, cv2.COLOR_RGB2GRAY)

    # Step 1: Re-run text detection on repaired image
    mser_mask = detect_text_mser(
        gray_repaired, min_area=mser_min_area, max_area=mser_max_area, max_aspect=mser_max_aspect
    )
    lap_mask = detect_text_laplacian(gray_repaired, threshold=lap_threshold, max_area=lap_max_area)
    combined = cv2.bitwise_or(mser_mask, lap_mask)
    seeds = ((combined > 0) & mask_bool)

    if not np.any(seeds):
        return np.zeros_like(first_mask)

    # Step 2: Region growing from seeds within masked regions
    residual_grown = seeds.copy()
    changed = True
    while changed:
        changed = False
        dilated = (
            cv2.dilate(
                residual_grown.astype(np.uint8) * 255,
                np.ones((3, 3), np.uint8),
                iterations=1,
            ).astype(bool)
        )
        candidates = dilated & mask_bool & (~residual_grown)
        if np.any(candidates):
            mean_bright = gray_repaired[residual_grown].mean()
            new_pixels = candidates & (
                np.abs(gray_repaired.astype(float) - mean_bright) < grow_threshold
            )
            if np.any(new_pixels):
                residual_grown = residual_grown | new_pixels
                changed = True

    # Step 3: Slight dilation to cover full text bodies
    residual_mask = cv2.dilate(
        residual_grown.astype(np.uint8) * 255,
        np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8),
        iterations=dilate_iterations,
    )

    return residual_mask


def repair_median_replace(image_rgb, residual_mask, ksize=71):
    """
    二次修复：对 mask 区域用大核 median blur 的值直接替换。

    比 cv2.inpaint 更有效地清除模糊斑块状残留。
    """
    mask_bool = residual_mask.astype(bool)
    result = image_rgb.copy().astype(np.float32)
    for ch in range(3):
        chan = result[:, :, ch]
        med = cv2.medianBlur(chan.astype(np.uint8), ksize).astype(np.float32)
        result[:, :, ch] = np.where(mask_bool, med, chan)
    return result.astype(np.uint8)


def remove_text_two_pass(
    image_rgb,
    first_pass_params=None,
    detect_params=None,
    repair_ksize=71,
):
    """
    两阶段文字移除：第一次修复 + 残留检测 + 二次修复。

    Returns:
        (final_result, first_result, first_mask, residual_mask)
    """
    if first_pass_params is None:
        first_pass_params = dict(
            brightness_thresh=170, dilate_iter=1, inpaint_radius=3
        )
    if detect_params is None:
        detect_params = {}

    # First pass
    first_result, first_mask = remove_text_mser_v2(image_rgb, **first_pass_params)

    # Detect residual
    residual_mask = detect_residual_region_growing(first_result, first_mask, **detect_params)

    # Second pass repair
    if np.any(residual_mask):
        final_result = repair_median_replace(first_result, residual_mask, ksize=repair_ksize)
    else:
        final_result = first_result

    return final_result, first_result, first_mask, residual_mask


if __name__ == "__main__":
    BASE = Path("/Users/daiduo2/geoseg/src/3d_schematic")
    PANELS = [BASE / "figures" / "panels" / f"panel_{i}.png" for i in (1, 2, 3)]
    OUT = BASE / "experiments" / "text_removal_v2" / "second_pass"
    OUT.mkdir(parents=True, exist_ok=True)

    for p in PANELS:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        final, first, mask, residual = remove_text_two_pass(img_rgb)

        stem = p.stem
        cv2.imwrite(
            str(OUT / f"{stem}_two_pass_final.png"),
            cv2.cvtColor(final, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            str(OUT / f"{stem}_two_pass_residual.png"),
            residual,
        )

        cov_first = np.count_nonzero(mask) / mask.size * 100
        cov_residual = np.count_nonzero(residual) / residual.size * 100
        print(
            f"[{stem}] first mask: {cov_first:.2f}%, residual: {cov_residual:.3f}%"
        )

    print(f"\nOutput: {OUT}")
