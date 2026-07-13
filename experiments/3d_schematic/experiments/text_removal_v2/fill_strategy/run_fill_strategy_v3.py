"""
填充策略对比实验脚本 v3。
v2 改进：detect_dark_text 正确方向。
v3 改进：mask 内部填充（闭运算 + 孔洞填充），使文字区域被完整覆盖。
"""
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from mser_v2_framework import detect_text_mser, detect_text_laplacian


def fill_holes(mask):
    """填充 mask 中的孔洞。"""
    # 复制 mask
    flood_fill = mask.copy()
    # 创建比原图大 2 像素的图像，用于 floodFill
    h, w = mask.shape[:2]
    flood_fill_mask = np.zeros((h + 2, w + 2), np.uint8)
    # 从 (0,0) 开始 flood fill，背景变为 128
    cv2.floodFill(flood_fill, flood_fill_mask, (0, 0), 128)
    # 反转：被 flood fill 的区域是背景，保留原 mask + 孔洞
    flood_fill_inv = cv2.bitwise_not(flood_fill)
    # 孔洞 = 背景中被反转后仍为 255 的区域（原 mask 为 0 且不是外部背景）
    holes = ((flood_fill == 128) & (mask == 0)).astype(np.uint8) * 255
    # 合并原 mask 和孔洞
    return cv2.bitwise_or(mask, holes)


def build_refined_mask_v3(image_rgb, brightness_thresh=180, max_stroke_width=6, dilate_iter=1,
                          close_iter=2):
    """
    v3: 在 v2 基础上增加闭运算和孔洞填充，使 mask 覆盖完整文字区域。
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    mask_orig = detect_text_mser(gray, min_area=10, max_area=2000, max_aspect=20)
    mask_inv = detect_text_mser(255 - gray, min_area=10, max_area=2000, max_aspect=20)
    mask_lap = detect_text_laplacian(gray, threshold=15, max_area=2000)

    combined = cv2.bitwise_or(mask_orig, mask_inv)
    combined = cv2.bitwise_or(combined, mask_lap)

    # 1) 亮度过滤：文字比背景暗
    brightness_mask = (gray < brightness_thresh).astype(np.uint8) * 255
    combined = cv2.bitwise_and(combined, brightness_mask)

    # 2) 笔画宽度约束
    dist = cv2.distanceTransform(combined, cv2.DIST_L2, 5)
    half = max(1, max_stroke_width // 2)
    stroke_mask = ((dist > 0) & (dist <= half)).astype(np.uint8) * 255
    combined = stroke_mask

    # 3) 闭运算填充文字内部孔洞
    if close_iter > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=close_iter)

    # 4) 孔洞填充
    combined = fill_holes(combined)

    # 5) 膨胀
    if dilate_iter > 0:
        kernel = np.ones((3, 3), np.uint8)
        combined = cv2.dilate(combined, kernel, iterations=dilate_iter)

    return combined


def fill_inpaint_telea(image_rgb, mask, radius):
    return cv2.inpaint(image_rgb, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)


def fill_inpaint_ns(image_rgb, mask, radius):
    return cv2.inpaint(image_rgb, mask, inpaintRadius=radius, flags=cv2.INPAINT_NS)


def fill_median_blur(image_rgb, mask, ksize=7):
    mask_bool = mask.astype(bool)
    result = image_rgb.copy().astype(np.float32)
    for ch in range(3):
        chan = result[:, :, ch]
        med = cv2.medianBlur(chan.astype(np.uint8), ksize).astype(np.float32)
        result[:, :, ch] = np.where(mask_bool, med, chan)
    return result.astype(np.uint8)


def fill_neighbor_mean(image_rgb, mask):
    mask_bool = mask.astype(bool)
    result = image_rgb.copy().astype(np.float32)
    inv_mask = (~mask_bool).astype(np.float32)

    for ch in range(3):
        chan = result[:, :, ch]
        masked_chan = chan * inv_mask
        kernel = np.ones((5, 5), np.float32)
        sum_neighbors = cv2.filter2D(masked_chan, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        count_neighbors = cv2.filter2D(inv_mask, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        count_neighbors = np.maximum(count_neighbors, 1e-6)
        mean_neighbors = sum_neighbors / count_neighbors
        result[:, :, ch] = np.where(mask_bool, mean_neighbors, chan)

    return result.astype(np.uint8)


def fill_overlay_mask(image_rgb, mask):
    result = image_rgb.copy()
    overlay = np.zeros_like(result)
    overlay[mask.astype(bool)] = [255, 0, 0]
    result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)
    return result


def run_strategy(image_rgb, mask, strategy_name, **kwargs):
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

    strategies = [
        ("inpaint_telea_r3", {"strategy": "inpaint_telea", "radius": 3}),
        ("inpaint_telea_r5", {"strategy": "inpaint_telea", "radius": 5}),
        ("inpaint_telea_r7", {"strategy": "inpaint_telea", "radius": 7}),
        ("inpaint_ns_r3", {"strategy": "inpaint_ns", "radius": 3}),
        ("inpaint_ns_r5", {"strategy": "inpaint_ns", "radius": 5}),
        ("inpaint_ns_r7", {"strategy": "inpaint_ns", "radius": 7}),
        ("median_blur_5", {"strategy": "median_blur", "ksize": 5}),
        ("median_blur_7", {"strategy": "median_blur", "ksize": 7}),
        ("median_blur_9", {"strategy": "median_blur", "ksize": 9}),
        ("neighbor_mean", {"strategy": "neighbor_mean"}),
        ("overlay_mask", {"strategy": "overlay_mask"}),
    ]

    panels = ["panel_3", "panel_1", "panel_2"]

    for panel_name in panels:
        panel_path = PANELS_DIR / f"{panel_name}.png"
        if not panel_path.exists():
            print(f"Skip {panel_name}: not found")
            continue

        print(f"\n=== {panel_name} ===")
        img = cv2.imread(str(panel_path), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = build_refined_mask_v3(
            img_rgb,
            brightness_thresh=180,
            max_stroke_width=6,
            dilate_iter=1,
            close_iter=2,
        )
        mask_coverage = np.count_nonzero(mask) / mask.size * 100
        print(f"  Mask coverage: {mask_coverage:.2f}%")

        cv2.imwrite(str(OUT_DIR / f"{panel_name}_shared_mask_v3.png"), mask)

        for suffix, params in strategies:
            params_copy = dict(params)
            strategy_name = params_copy.pop("strategy")
            result = run_strategy(img_rgb, mask, strategy_name, **params_copy)

            out_path = OUT_DIR / f"{panel_name}_{suffix}_v3.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            print(f"  -> {suffix}_v3 saved")

    print(f"\nAll v3 results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
