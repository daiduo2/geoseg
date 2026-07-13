"""
局部二次修复实验框架。
Round 1: 残留检测策略对比 (A/B/C)
Round 2: 二次修复方法对比 (1-4)
"""
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from mser_v2_framework import remove_text_mser_v2


def first_pass(image_rgb, **kwargs):
    """第一次修复。"""
    return remove_text_mser_v2(image_rgb, **kwargs)


def detect_residual_strategy_a(first_result, mask_bool, brightness_thresh=200):
    """策略 A: 亮度残留检测。
    mask 区域内亮度仍 > thresh 的像素 = 残留文字。
    """
    gray_repaired = cv2.cvtColor(first_result, cv2.COLOR_RGB2GRAY)
    residual = (gray_repaired > brightness_thresh) & mask_bool
    return residual.astype(np.uint8) * 255


def detect_residual_strategy_b(image_rgb, first_result, mask_bool, diff_thresh=30):
    """策略 B: 与原图差异检测。
    mask 区域内与原图亮度差异小的像素 = 修复效果不好。
    """
    gray_orig = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray_repaired = cv2.cvtColor(first_result, cv2.COLOR_RGB2GRAY)
    diff = np.abs(gray_repaired.astype(float) - gray_orig.astype(float))
    residual = (diff < diff_thresh) & mask_bool
    return residual.astype(np.uint8) * 255


def detect_residual_strategy_c(first_result, mask_bool, lap_thresh=15):
    """策略 C: 局部对比度检测。
    修复后 mask 区域内仍有高对比度边缘 = 残留文字。
    """
    gray_repaired = cv2.cvtColor(first_result, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray_repaired, cv2.CV_64F)
    residual = (np.abs(lap) > lap_thresh) & mask_bool
    return residual.astype(np.uint8) * 255


def detect_residual_combined_ab(image_rgb, first_result, mask_bool,
                                 brightness_thresh=200, diff_thresh=30):
    """组合策略 A AND B: 亮度高 AND 差异小。"""
    gray_orig = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray_repaired = cv2.cvtColor(first_result, cv2.COLOR_RGB2GRAY)
    diff = np.abs(gray_repaired.astype(float) - gray_orig.astype(float))
    residual = (gray_repaired > brightness_thresh) & (diff < diff_thresh) & mask_bool
    return residual.astype(np.uint8) * 255


def repair_method_1_inpaint(image_rgb, residual_mask, radius=7):
    """方法 1: inpaint(radius=7, TELEA)。"""
    return cv2.inpaint(image_rgb, residual_mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)


def repair_method_2_median(image_rgb, residual_mask, ksize=11):
    """方法 2: median blur 11x11。"""
    mask_bool = residual_mask.astype(bool)
    result = image_rgb.copy().astype(np.float32)
    for ch in range(3):
        chan = result[:, :, ch]
        med = cv2.medianBlur(chan.astype(np.uint8), ksize).astype(np.float32)
        result[:, :, ch] = np.where(mask_bool, med, chan)
    return result.astype(np.uint8)


def repair_method_3_neighborhood_mean(image_rgb, residual_mask):
    """方法 3: 周围非 mask 像素均值。
    使用更大的膨胀后的 mask 的补集来采样周围像素。
    """
    mask_bool = residual_mask.astype(bool)
    # 膨胀 mask 以获取邻域
    kernel = np.ones((21, 21), np.uint8)
    dilated = cv2.dilate(residual_mask, kernel, iterations=1).astype(bool)
    # 邻域但不包括 mask 自身
    neighborhood = dilated & (~mask_bool)

    result = image_rgb.copy().astype(np.float32)
    for ch in range(3):
        chan = result[:, :, ch].copy()
        # 计算邻域均值（使用大核模糊近似）
        blurred = cv2.blur(chan, (21, 21)).astype(np.float32)
        result[:, :, ch] = np.where(mask_bool, blurred, chan)
    return result.astype(np.uint8)


def repair_method_4_large_median(image_rgb, residual_mask, ksize=21):
    """方法 4: 更大半径的邻域中位数（21x21）。"""
    mask_bool = residual_mask.astype(bool)
    result = image_rgb.copy().astype(np.float32)
    for ch in range(3):
        chan = result[:, :, ch]
        med = cv2.medianBlur(chan.astype(np.uint8), ksize).astype(np.float32)
        result[:, :, ch] = np.where(mask_bool, med, chan)
    return result.astype(np.uint8)


def run_second_pass_experiment(
    panel_path,
    out_dir,
    first_pass_params=None,
    detection_strategies=None,
    repair_methods=None,
):
    """
    运行完整的二次修复实验。

    detection_strategies: list of tuples (name, func, kwargs)
    repair_methods: list of tuples (name, func, kwargs)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if first_pass_params is None:
        first_pass_params = dict(brightness_thresh=170, dilate_iter=1, inpaint_radius=3)

    img = cv2.imread(str(panel_path), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # First pass
    first_result, first_mask = first_pass(img_rgb, **first_pass_params)
    mask_bool = first_mask.astype(bool)

    # Save first pass
    stem = panel_path.stem
    cv2.imwrite(
        str(out_dir / f"{stem}_first_pass.png"),
        cv2.cvtColor(first_result, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(out_dir / f"{stem}_first_mask.png"),
        first_mask,
    )
    print(f"[{stem}] First pass mask coverage: {np.count_nonzero(first_mask)/first_mask.size*100:.2f}%")

    if detection_strategies is None:
        detection_strategies = [
            ("A_brightness200", detect_residual_strategy_a, {"brightness_thresh": 200}),
            ("A_brightness180", detect_residual_strategy_a, {"brightness_thresh": 180}),
            ("B_diff30", detect_residual_strategy_b, {"diff_thresh": 30}),
            ("B_diff50", detect_residual_strategy_b, {"diff_thresh": 50}),
            ("C_lap15", detect_residual_strategy_c, {"lap_thresh": 15}),
            ("C_lap25", detect_residual_strategy_c, {"lap_thresh": 25}),
            ("AB_combo", detect_residual_combined_ab, {"brightness_thresh": 180, "diff_thresh": 40}),
        ]

    if repair_methods is None:
        repair_methods = [
            ("m1_inpaint7", repair_method_1_inpaint, {"radius": 7}),
            ("m2_median11", repair_method_2_median, {"ksize": 11}),
            ("m3_nbmean", repair_method_3_neighborhood_mean, {}),
            ("m4_median21", repair_method_4_large_median, {"ksize": 21}),
        ]

    results_summary = []

    for det_name, det_func, det_kwargs in detection_strategies:
        # Compute residual mask
        if "image_rgb" in det_func.__code__.co_varnames:
            residual_mask = det_func(img_rgb, first_result, mask_bool, **det_kwargs)
        else:
            residual_mask = det_func(first_result, mask_bool, **det_kwargs)

        cov = np.count_nonzero(residual_mask) / residual_mask.size * 100
        print(f"  Detection {det_name}: residual coverage {cov:.3f}%")

        # Save residual mask
        cv2.imwrite(str(out_dir / f"{stem}_residual_{det_name}.png"), residual_mask)

        for rep_name, rep_func, rep_kwargs in repair_methods:
            second_result = rep_func(first_result, residual_mask, **rep_kwargs)

            out_name = f"{stem}_{det_name}_{rep_name}.png"
            cv2.imwrite(
                str(out_dir / out_name),
                cv2.cvtColor(second_result, cv2.COLOR_RGB2BGR),
            )
            results_summary.append({
                "panel": stem,
                "detection": det_name,
                "repair": rep_name,
                "residual_coverage_pct": cov,
                "out_name": out_name,
            })

    return results_summary


if __name__ == "__main__":
    BASE = Path("/Users/daiduo2/geoseg/src/3d_schematic")
    PANELS = [BASE / "figures" / "panels" / f"panel_{i}.png" for i in (1, 2, 3)]
    OUT = BASE / "experiments" / "text_removal_v2" / "second_pass"

    all_results = []
    for p in PANELS:
        print(f"\n{'='*60}")
        print(f"Processing {p.name}")
        print(f"{'='*60}")
        results = run_second_pass_experiment(p, OUT)
        all_results.extend(results)

    print(f"\n{'='*60}")
    print(f"All experiments complete. {len(all_results)} combinations tested.")
    print(f"Output directory: {OUT}")
