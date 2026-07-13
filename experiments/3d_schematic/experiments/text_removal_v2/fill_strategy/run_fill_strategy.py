"""
填充策略对比实验脚本。
固定 mask 生成参数，对比不同修复/填充方法在相同精炼 mask 上的效果。
"""
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from mser_v2_framework import detect_text_mser, detect_text_laplacian


def build_refined_mask(image_rgb, brightness_thresh=180, max_stroke_width=6, dilate_iter=1):
    """固定参数生成精炼 mask（所有策略共享同一 mask）。"""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    mask_orig = detect_text_mser(gray, min_area=10, max_area=2000, max_aspect=20)
    mask_inv = detect_text_mser(255 - gray, min_area=10, max_area=2000, max_aspect=20)
    mask_lap = detect_text_laplacian(gray, threshold=15, max_area=2000)

    combined = cv2.bitwise_or(mask_orig, mask_inv)
    combined = cv2.bitwise_or(combined, mask_lap)

    # 1) 亮度过滤
    brightness_mask = (gray > brightness_thresh).astype(np.uint8) * 255
    combined = cv2.bitwise_and(combined, brightness_mask)

    # 2) 笔画宽度约束
    dist = cv2.distanceTransform(combined, cv2.DIST_L2, 5)
    half = max(1, max_stroke_width // 2)
    stroke_mask = ((dist > 0) & (dist <= half)).astype(np.uint8) * 255
    combined = stroke_mask

    # 3) 膨胀
    if dilate_iter > 0:
        kernel = np.ones((3, 3), np.uint8)
        combined = cv2.dilate(combined, kernel, iterations=dilate_iter)

    return combined


def fill_inpaint_telea(image_rgb, mask, radius):
    """cv2.inpaint TELEA 策略。"""
    return cv2.inpaint(image_rgb, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)


def fill_inpaint_ns(image_rgb, mask, radius):
    """cv2.inpaint NS 策略。"""
    return cv2.inpaint(image_rgb, mask, inpaintRadius=radius, flags=cv2.INPAINT_NS)


def fill_median_blur(image_rgb, mask, ksize=7):
    """Median Blur 填充策略。"""
    mask_bool = mask.astype(bool)
    result = image_rgb.copy().astype(np.float32)
    for ch in range(3):
        chan = result[:, :, ch]
        med = cv2.medianBlur(chan.astype(np.uint8), ksize).astype(np.float32)
        result[:, :, ch] = np.where(mask_bool, med, chan)
    return result.astype(np.uint8)


def fill_neighbor_mean(image_rgb, mask):
    """周围非 mask 像素均值填充策略。
    对 mask 内每个像素，取周围 5x5 邻域中非 mask 像素的均值填充。
    """
    mask_bool = mask.astype(bool)
    result = image_rgb.copy().astype(np.float32)

    # 创建反 mask：非 mask 区域为 1，mask 区域为 0
    inv_mask = (~mask_bool).astype(np.float32)

    for ch in range(3):
        chan = result[:, :, ch]
        # 仅保留非 mask 像素值
        masked_chan = chan * inv_mask

        # 计算邻域和与邻域内非 mask 像素计数
        kernel = np.ones((5, 5), np.float32)
        sum_neighbors = cv2.filter2D(masked_chan, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        count_neighbors = cv2.filter2D(inv_mask, -1, kernel, borderType=cv2.BORDER_REPLICATE)

        # 避免除零
        count_neighbors = np.maximum(count_neighbors, 1e-6)
        mean_neighbors = sum_neighbors / count_neighbors

        result[:, :, ch] = np.where(mask_bool, mean_neighbors, chan)

    return result.astype(np.uint8)


def fill_overlay_mask(image_rgb, mask):
    """不填充，直接标记为叠层（红色半透明覆盖）。"""
    result = image_rgb.copy()
    overlay = np.zeros_like(result)
    overlay[mask.astype(bool)] = [255, 0, 0]  # 红色标记
    result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)
    return result


def run_strategy(image_rgb, mask, strategy_name, **kwargs):
    """根据策略名称调用对应填充方法。"""
    if strategy_name == "inpaint_telea":
        return fill_inpaint_telea(image_rgb, mask, kwargs["radius"])
    elif strategy_name == "inpaint_ns":
        return fill_inpaint_ns(image_rgb, mask, kwargs["radius"])
    elif strategy_name == "median_blur":
        return fill_median_blur(image_rgb, mask, kwargs.get("ksize", 7))
    elif strategy_name == "neighbor_mean":
        return fill_neighbor_mean(image_rgb, mask)
    elif strategy_name == "overlay_mask":
        return fill_overlay_mask(image_rgb, mask)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


def main():
    BASE = Path("/Users/daiduo2/geoseg/src/3d_schematic")
    PANELS_DIR = BASE / "figures" / "panels"
    OUT_DIR = BASE / "experiments" / "text_removal_v2" / "fill_strategy"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 定义所有策略
    strategies = [
        ("inpaint_telea_r3", {"strategy": "inpaint_telea", "radius": 3}),
        ("inpaint_telea_r5", {"strategy": "inpaint_telea", "radius": 5}),
        ("inpaint_telea_r7", {"strategy": "inpaint_telea", "radius": 7}),
        ("inpaint_ns_r3", {"strategy": "inpaint_ns", "radius": 3}),
        ("inpaint_ns_r5", {"strategy": "inpaint_ns", "radius": 5}),
        ("inpaint_ns_r7", {"strategy": "inpaint_ns", "radius": 7}),
        ("median_blur_7", {"strategy": "median_blur", "ksize": 7}),
        ("neighbor_mean", {"strategy": "neighbor_mean"}),
        ("overlay_mask", {"strategy": "overlay_mask"}),
    ]

    # 先跑 Panel 3，再推广到 panel_1 和 panel_2
    panels_order = ["panel_3", "panel_1", "panel_2"]

    for panel_name in panels_order:
        panel_path = PANELS_DIR / f"{panel_name}.png"
        if not panel_path.exists():
            print(f"Skip {panel_name}: not found")
            continue

        print(f"\n=== {panel_name} ===")
        img = cv2.imread(str(panel_path), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 生成共享 mask
        mask = build_refined_mask(
            img_rgb,
            brightness_thresh=180,
            max_stroke_width=6,
            dilate_iter=1,
        )
        mask_coverage = np.count_nonzero(mask) / mask.size * 100
        print(f"  Mask coverage: {mask_coverage:.2f}%")

        # 保存共享 mask
        cv2.imwrite(
            str(OUT_DIR / f"{panel_name}_shared_mask.png"),
            mask,
        )

        # 跑每种策略
        for suffix, params in strategies:
            params_copy = dict(params)
            strategy_name = params_copy.pop("strategy")
            result = run_strategy(img_rgb, mask, strategy_name, **params_copy)

            out_path = OUT_DIR / f"{panel_name}_{suffix}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            print(f"  -> {suffix} saved")

    print(f"\nAll results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
