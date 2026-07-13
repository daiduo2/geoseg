"""
MSER v2 优化实验框架。
支持参数化探索：亮度过滤、笔画宽度约束、膨胀次数、inpaint 半径。
"""
import cv2
import numpy as np
from pathlib import Path


def detect_text_mser(gray, min_area=10, max_area=2000, max_aspect=20):
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    for region in regions:
        region = region.reshape(-1, 1, 2)
        area = cv2.contourArea(region)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(region)
        if w == 0 or h == 0:
            continue
        aspect = max(w, h) / min(w, h)
        if aspect > max_aspect:
            continue
        cv2.fillPoly(mask, [region], 255)
    return mask


def detect_text_laplacian(gray, threshold=15, max_area=2000):
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    _, mask = cv2.threshold(laplacian, threshold, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > max_area:
            mask[labels == i] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def remove_text_mser_v2(
    image_rgb,
    brightness_thresh=0,      # 0 = 禁用; >0 则只保留亮度 > thresh 的 mask 像素
    max_stroke_width=0,       # 0 = 禁用; >0 则距离变换收缩到笔画宽度一半
    dilate_iter=1,
    inpaint_radius=3,
    use_median_fill=False,    # True = 7x7 median blur 填充; False = cv2.inpaint
    min_area=10,
    max_area=2000,
    max_aspect=20,
    lap_threshold=15,
):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    mask_orig = detect_text_mser(gray, min_area, max_area, max_aspect)
    mask_inv = detect_text_mser(255 - gray, min_area, max_area, max_aspect)
    mask_lap = detect_text_laplacian(gray, lap_threshold, max_area)

    combined = cv2.bitwise_or(mask_orig, mask_inv)
    combined = cv2.bitwise_or(combined, mask_lap)

    # 1) 亮度过滤
    if brightness_thresh > 0:
        brightness_mask = (gray > brightness_thresh).astype(np.uint8) * 255
        combined = cv2.bitwise_and(combined, brightness_mask)

    # 2) 笔画宽度约束（距离变换收缩）
    if max_stroke_width > 0:
        dist = cv2.distanceTransform(combined, cv2.DIST_L2, 5)
        half = max(1, max_stroke_width // 2)
        stroke_mask = ((dist > 0) & (dist <= half)).astype(np.uint8) * 255
        combined = stroke_mask

    # 3) 膨胀
    if dilate_iter > 0:
        kernel = np.ones((3, 3), np.uint8)
        combined = cv2.dilate(combined, kernel, iterations=dilate_iter)

    # 4) 修复
    if use_median_fill:
        mask_bool = combined.astype(bool)
        result = image_rgb.copy().astype(np.float32)
        for ch in range(3):
            chan = result[:, :, ch]
            med = cv2.medianBlur(chan.astype(np.uint8), 7).astype(np.float32)
            result[:, :, ch] = np.where(mask_bool, med, chan)
        result = result.astype(np.uint8)
    else:
        result = cv2.inpaint(
            image_rgb, combined, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA
        )

    return result, combined


def run_experiment(
    panel_paths,
    out_dir,
    name_suffix,
    **kwargs,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in panel_paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result, mask = remove_text_mser_v2(img_rgb, **kwargs)
        cv2.imwrite(
            str(out_dir / f"{p.stem}_{name_suffix}.png"),
            cv2.cvtColor(result, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            str(out_dir / f"{p.stem}_{name_suffix}_mask.png"),
            mask,
        )
        print(f"  {p.stem} -> mask coverage {np.count_nonzero(mask)/mask.size*100:.2f}%")


if __name__ == "__main__":
    BASE = Path("/Users/daiduo2/geoseg/src/3d_schematic")
    PANELS = [BASE / "figures" / "panels" / f"panel_{i}.png" for i in (1, 2, 3)]
    OUT = BASE / "experiments" / "text_removal_v2"

    # Example: baseline v2 with brightness filter
    print("Running b180_d1_i3...")
    run_experiment(PANELS, OUT / "b180_d1_i3", "v2",
                   brightness_thresh=180, dilate_iter=1, inpaint_radius=3)
