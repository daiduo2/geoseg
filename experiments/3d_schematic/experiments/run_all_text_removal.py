#!/usr/bin/env python3
"""
Parallel text-removal experiment runner.
Runs 5 algorithms on 3 panels, saves results for visual audit.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path("/Users/daiduo2/geoseg/src/3d_schematic")
PANEL_DIR = BASE / "figures" / "panels"
OUT_DIR = BASE / "experiments" / "text_removal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PANELS = [PANEL_DIR / f"panel_{i}.png" for i in (1, 2, 3)]

# ---------------------------------------------------------------------------
# Helper: load / save
# ---------------------------------------------------------------------------
def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_rgb(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------------
# Algorithm 1: baseline (adaptive + Laplacian + inpaint + Gaussian blend)
# ---------------------------------------------------------------------------
def run_baseline(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=25, C=-5
    )
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    laplacian = (laplacian > np.percentile(laplacian, 80)).astype(np.uint8) * 255
    text_mask = ((adaptive > 0) & (laplacian > 0))

    import scipy.ndimage as ndi
    labeled, num = ndi.label(text_mask)
    text_mask_clean = np.zeros_like(text_mask)
    for i in range(1, num + 1):
        comp = labeled == i
        s = comp.sum()
        if 8 < s < 1200:
            text_mask_clean[comp] = True

    kernel = np.ones((5, 5), np.uint8)
    text_dilated = cv2.dilate(text_mask_clean.astype(np.uint8), kernel, iterations=2)
    inpainted = cv2.inpaint(image, text_dilated, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

    # Fast median refill via scipy (vectorised neighbourhood median)
    mask_bool = text_dilated.astype(bool)
    cleaned = inpainted.copy()
    for ch in range(3):
        chan = cleaned[:, :, ch].astype(np.float32)
        # masked pixels → median of 7×7 neighbourhood
        med = ndi.median_filter(chan, size=7, mode="nearest")
        cleaned[:, :, ch] = np.where(mask_bool, med, chan).astype(np.uint8)

    # Gaussian blend at mask edges
    blurred = cv2.GaussianBlur(cleaned, (7, 7), sigmaX=1.5)
    mask_f = text_dilated.astype(np.float32) / 255.0
    mask_f = cv2.GaussianBlur(mask_f, (5, 5), sigmaX=2)
    mask_3ch = np.stack([mask_f] * 3, axis=-1)
    result = (blurred * mask_3ch + cleaned * (1 - mask_3ch)).astype(np.uint8)
    return result


# ---------------------------------------------------------------------------
# Algorithm 2: multiscale (7-channel)
# ---------------------------------------------------------------------------
def run_multiscale(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    laplacian_abs = np.abs(laplacian).astype(np.uint8)
    _, mask_laplacian = cv2.threshold(laplacian_abs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, blockSize=51, C=5)

    blurred31 = cv2.GaussianBlur(gray, (31, 31), 0)
    local_contrast = cv2.absdiff(gray, blurred31)
    _, mask_contrast = cv2.threshold(local_contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    sat_blurred = cv2.GaussianBlur(saturation, (15, 15), 0)
    _, mask_low_sat = cv2.threshold(sat_blurred.astype(np.uint8), 0, 255,
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    blur_small = cv2.GaussianBlur(gray, (3, 3), 0.5)
    blur_large = cv2.GaussianBlur(gray, (11, 11), 2.0)
    dog = cv2.absdiff(blur_small, blur_large)
    _, mask_dog = cv2.threshold(dog, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    canny_edges = cv2.Canny(gray, 50, 150)

    local_mean = cv2.boxFilter(gray.astype(np.float32), -1, (21, 21), normalize=True)
    brightness_deviation = np.abs(gray.astype(np.float32) - local_mean).astype(np.uint8)
    _, mask_local_brightness = cv2.threshold(brightness_deviation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    combined = np.zeros_like(gray)
    for m in (mask_laplacian, mask_adaptive, mask_contrast, mask_low_sat,
              mask_dog, canny_edges, mask_local_brightness):
        combined = cv2.bitwise_or(combined, m)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    filtered_mask = np.zeros_like(combined)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        if area > 8000 or area < 15:
            continue
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        is_text = (aspect > 1.5 and area < 4000) or (aspect <= 1.5 and area < 500)
        if is_text:
            filtered_mask[labels == i] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_mask = cv2.dilate(filtered_mask, kernel, iterations=2)

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    result_bgr = cv2.inpaint(image_bgr, dilated_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Algorithm 3: MSER
# ---------------------------------------------------------------------------
def run_mser(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def detect_mser(g):
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(g)
        mask = np.zeros(g.shape, dtype=np.uint8)
        for region in regions:
            region = region.reshape(-1, 1, 2)
            area = cv2.contourArea(region)
            if area < 10 or area > 2000:
                continue
            x, y, w, h = cv2.boundingRect(region)
            if w == 0 or h == 0:
                continue
            if max(w, h) / min(w, h) > 20:
                continue
            cv2.fillPoly(mask, [region], 255)
        return mask

    mask_orig = detect_mser(gray)
    mask_inv = detect_mser(255 - gray)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    _, mask_lap = cv2.threshold(laplacian, 15, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_lap, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 2000:
            mask_lap[labels == i] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_lap = cv2.morphologyEx(mask_lap, cv2.MORPH_CLOSE, kernel, iterations=2)

    combined = cv2.bitwise_or(mask_orig, mask_inv)
    combined = cv2.bitwise_or(combined, mask_lap)

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(combined, kernel, iterations=4)
    inpainted = cv2.inpaint(image, dilated, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    return inpainted


# ---------------------------------------------------------------------------
# Algorithm 4: SWT (resize to 50% for speed, then upscale)
# ---------------------------------------------------------------------------
def run_swt(image: np.ndarray) -> np.ndarray:
    # Downscale for speed – SWT ray-tracing is O(pixels * max_stroke)
    h, w = image.shape[:2]
    scale = 0.5
    small = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, 30, 100)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    theta = np.arctan2(sobely, sobelx)

    sh, sw = edges.shape
    swt_arr = np.full((sh, sw), np.inf, dtype=np.float32)
    edge_pixels = np.argwhere(edges > 0)
    max_sw = 60

    for y, x in edge_pixels:
        angle = theta[y, x]
        for sign in (1, -1):
            dx = np.cos(angle) * sign
            dy = np.sin(angle) * sign
            for step in range(1, max_sw + 1):
                nx = int(round(x + dx * step))
                ny = int(round(y + dy * step))
                if not (0 <= nx < sw and 0 <= ny < sh):
                    break
                if edges[ny, nx] > 0:
                    dist = np.hypot(nx - x, ny - y)
                    swt_arr[y, x] = min(swt_arr[y, x], dist)
                    break

    swt_arr[np.isinf(swt_arr)] = max_sw

    # Consistency mask
    ws = 5
    half = ws // 2
    padded = np.pad(swt_arr, half, mode="edge")
    mask = np.zeros((sh, sw), dtype=np.uint8)
    for y in range(sh):
        for x in range(sw):
            window = padded[y:y + ws, x:x + ws]
            med = np.median(window)
            if 0 < med <= 30:
                mad = np.median(np.abs(window - med))
                if mad < 4.0:
                    mask[y, x] = 255

    # Laplacian backup
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_64F, ksize=3))
    mx = laplacian.max()
    if mx > 0:
        laplacian = (laplacian / mx * 255).astype(np.uint8)
    _, high_freq = cv2.threshold(laplacian, 12, 255, cv2.THRESH_BINARY)

    _, bright1 = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    _, bright2 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    brightness_mask = cv2.bitwise_or(bright1, bright2)
    kernel = np.ones((5, 5), np.uint8)
    brightness_mask = cv2.dilate(brightness_mask, kernel, iterations=1)
    high_freq = cv2.bitwise_and(high_freq, brightness_mask)

    # Filter laplacian CC
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(high_freq, connectivity=8)
    lap_mask = np.zeros_like(high_freq)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if area < 5 or area > 800 or w == 0 or h == 0:
            continue
        aspect = max(w, h) / max(min(w, h), 1)
        compactness = (w * h) / max(area, 1)
        if aspect < 15 and compactness < 12:
            lap_mask[labels == i] = 255

    # MSER backup
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    mser_mask = np.zeros_like(gray)
    for region in regions:
        if 10 < len(region) < 800:
            for pt in region:
                mser_mask[pt[1], pt[0]] = 255
    mser_mask = cv2.bitwise_and(mser_mask, brightness_mask)
    kernel = np.ones((3, 3), np.uint8)
    mser_mask = cv2.dilate(mser_mask, kernel, iterations=1)

    combined = cv2.bitwise_or(mask, lap_mask)
    combined = cv2.bitwise_or(combined, mser_mask)

    # Filter geological lines
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    filtered = np.zeros_like(combined)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if area < 5 or w == 0 or h == 0:
            continue
        aspect = max(w, h) / max(min(w, h), 1)
        compactness = (w * h) / max(area, 1)
        if aspect < 10 and compactness < 15:
            filtered[labels == i] = 255

    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(filtered, kernel, iterations=2)

    # Inpaint on downscaled image, then upscale
    inpainted_small = cv2.inpaint(small, dilated, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    result = cv2.resize(inpainted_small, (w, h), interpolation=cv2.INTER_CUBIC)
    return result


# ---------------------------------------------------------------------------
# Algorithm 5: diff-overlay
# ---------------------------------------------------------------------------
def run_diff_overlay(image: np.ndarray) -> dict:
    from skimage.segmentation import felzenszwalb

    blur_ksize, blur_sigma, diff_thresh = 15, 3.0, 20.0
    expand_radius, felz_scale, felz_sigma = 15, 300.0, 0.5

    if blur_ksize % 2 == 0:
        blur_ksize += 1
    blurred = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), sigmaX=blur_sigma)
    diff = np.abs(image.astype(np.float32) - blurred.astype(np.float32))
    detail = diff.max(axis=2)

    binary = (detail > diff_thresh).astype(np.uint8) * 255
    if expand_radius > 0:
        ksize = expand_radius * 2 + 1
        blurred_mask = cv2.GaussianBlur(binary, (ksize, ksize), sigmaX=expand_radius)
        overlay_mask = blurred_mask > 64
    else:
        overlay_mask = binary > 0

    inpaint_mask = overlay_mask.astype(np.uint8) * 255
    inpainted = cv2.inpaint(image, inpaint_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    geo_labels = felzenszwalb(inpainted, scale=felz_scale, sigma=felz_sigma, min_size=30)

    final_labels = geo_labels.copy()
    final_labels[overlay_mask] = -1

    # Visualise overlay in magenta
    overlay_vis = image.copy()
    overlay_vis[overlay_mask] = [255, 0, 255]

    return {
        "overlay_vis": overlay_vis,
        "final_labels": final_labels,
        "geo_labels": geo_labels,
        "overlay_mask": overlay_mask,
    }


# ---------------------------------------------------------------------------
# Render labels to colour fill
# ---------------------------------------------------------------------------
def render_label_fill(labels: np.ndarray, overlay_label: int = -1) -> np.ndarray:
    import colorsys
    unique = sorted(np.unique(labels))
    h, w = labels.shape
    result = np.zeros((h, w, 3), dtype=np.uint8)
    colors = []
    for i, lbl in enumerate(unique):
        if lbl == overlay_label:
            colors.append([128, 128, 128])
        else:
            hue = (i * 0.618033988749895) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.75, 0.92)
            colors.append([int(c * 255) for c in rgb])
    for i, lbl in enumerate(unique):
        result[labels == lbl] = colors[i]
    return result


# ---------------------------------------------------------------------------
# Worker dispatch
# ---------------------------------------------------------------------------
ALGORITHMS = {
    "baseline": run_baseline,
    "multiscale": run_multiscale,
    "mser": run_mser,
    "swt": run_swt,
    "diff_overlay": run_diff_overlay,
}


def process_algo(algo_name: str, panel_path: Path) -> tuple[str, str, float]:
    """Run one algorithm on one panel. Returns (algo, panel_name, elapsed_s)."""
    t0 = time.time()
    image = load_rgb(panel_path)
    algo_fn = ALGORITHMS[algo_name]

    out_algo_dir = OUT_DIR / algo_name
    out_algo_dir.mkdir(parents=True, exist_ok=True)

    if algo_name == "diff_overlay":
        result = algo_fn(image)
        save_rgb(result["overlay_vis"], out_algo_dir / f"{panel_path.stem}_overlay.png")
        fill = render_label_fill(result["final_labels"], overlay_label=-1)
        save_rgb(fill, out_algo_dir / f"{panel_path.stem}_labels.png")
        # Also save inpainted (non-overlay region) for fair comparison
        mask = result["overlay_mask"]
        inpainted = cv2.inpaint(image, mask.astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA)
        save_rgb(inpainted, out_algo_dir / f"{panel_path.stem}_inpainted.png")
    else:
        result = algo_fn(image)
        save_rgb(result, out_algo_dir / f"{panel_path.stem}_inpainted.png")

    elapsed = time.time() - t0
    return algo_name, panel_path.stem, elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    tasks = [(algo, panel) for algo in ALGORITHMS for panel in PANELS]
    print(f"Running {len(tasks)} experiments ({len(ALGORITHMS)} algos × {len(PANELS)} panels)...")

    # Run in parallel processes (CPU-bound)
    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_algo, algo, panel): (algo, panel)
                   for algo, panel in tasks}
        for future in as_completed(futures):
            algo, panel_name, elapsed = future.result()
            print(f"  ✓ {algo:12s} / {panel_name:8s}  ({elapsed:.1f}s)")

    print(f"\nAll results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
