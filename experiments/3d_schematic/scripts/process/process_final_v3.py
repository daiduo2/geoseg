"""Final pipeline v3: V-enhance + fine felzenszwalb + two-phase post-merge.

Key improvements:
1. V-channel CLAHE enhancement amplifies subtle brightness differences (helps Panel 3 plume)
2. Fine-grained felzenszwalb (scale=300, sigma=0.5) captures more detail
3. Two-phase post-merge: small fragments first, then conservative gradient-aware merge
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.segmentation import felzenszwalb


def remove_text(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=25, C=-5
    )
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    laplacian = (laplacian > np.percentile(laplacian, 80)).astype(np.uint8) * 255
    text_mask = ((adaptive > 0) & (laplacian > 0))
    labeled, num = ndimage.label(text_mask)
    text_mask_clean = np.zeros_like(text_mask)
    for i in range(1, num + 1):
        comp = labeled == i
        if 8 < comp.sum() < 1200:
            text_mask_clean[comp] = True
    kernel = np.ones((5, 5), np.uint8)
    text_dilated = cv2.dilate(text_mask_clean.astype(np.uint8), kernel, iterations=2)
    inpainted = cv2.inpaint(image, text_dilated, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    blurred = cv2.GaussianBlur(inpainted, (7, 7), sigmaX=1.5)
    mask_f = text_dilated.astype(np.float32) / 255.0
    mask_f = cv2.GaussianBlur(mask_f, (5, 5), sigmaX=2)
    mask_3ch = np.stack([mask_f] * 3, axis=-1)
    cleaned = (blurred * mask_3ch + inpainted * (1 - mask_3ch)).astype(np.uint8)
    return cleaned


def enhance_v(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def compute_color_gradient(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(sobel_x**2 + sobel_y**2)


def post_merge(label_img: np.ndarray, image: np.ndarray,
               small_ratio: float = 0.015,
               max_score: float = 0.8,
               max_color: float = 45.0) -> np.ndarray:
    h, w = label_img.shape
    gradient = compute_color_gradient(image)
    result = label_img.copy()
    total = h * w

    def renumber():
        nonlocal result
        remap = {}
        next_lbl = 0
        new_result = np.zeros_like(result)
        for lbl in sorted(np.unique(result)):
            remap[lbl] = next_lbl
            next_lbl += 1
        for old, new in remap.items():
            new_result[result == old] = new
        result = new_result

    # Phase 1: small fragments
    for _ in range(30):
        unique, counts = np.unique(result, return_counts=True)
        mean_colors = {lbl: image[result == lbl].mean(axis=0) for lbl in unique}
        small = unique[counts < total * small_ratio]
        if len(small) == 0:
            break
        for small_lbl in small:
            mask = result == small_lbl
            adjacent = set()
            ys, xs = np.where(mask)
            for y, x in zip(ys, xs):
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and result[ny, nx] != small_lbl:
                        adjacent.add(result[ny, nx])
            min_dist = float('inf')
            nearest = None
            for adj_lbl in adjacent:
                dist = np.linalg.norm(mean_colors[small_lbl] - mean_colors[adj_lbl])
                if dist < min_dist:
                    min_dist = dist
                    nearest = adj_lbl
            if nearest is not None:
                result[mask] = nearest
        renumber()

    # Phase 2: conservative merge
    for _ in range(30):
        unique = sorted(np.unique(result))
        if len(unique) <= 4:
            break
        mean_colors = {lbl: image[result == lbl].mean(axis=0) for lbl in unique}
        pair_data = {}
        for y in range(h):
            for x in range(w - 1):
                a, b = result[y, x], result[y, x + 1]
                if a != b:
                    pair = (min(a, b), max(a, b))
                    pair_data.setdefault(pair, []).append(gradient[y, x])
            if y < h - 1:
                for x in range(w):
                    a, b = result[y, x], result[y + 1, x]
                    if a != b:
                        pair = (min(a, b), max(a, b))
                        pair_data.setdefault(pair, []).append(gradient[y, x])
        if not pair_data:
            break
        scores = []
        for pair, grads in pair_data.items():
            i, j = pair
            cd = np.linalg.norm(mean_colors[i] - mean_colors[j])
            mg = np.mean(grads)
            score = cd / (mg + 1e-3)
            scores.append((score, pair, cd, mg))
        scores.sort(key=lambda x: x[0])
        best_score, best_pair, cd, mg = scores[0]
        if best_score >= max_score or cd >= max_color:
            break
        print(f"  merge {best_pair}: score={best_score:.2f} cd={cd:.1f} grad={mg:.2f}")
        i, j = best_pair
        result[result == j] = i
        renumber()

    return result


def draw_boundaries(image: np.ndarray, labels: np.ndarray,
                    color: tuple = (0, 0, 0), thickness: int = 2) -> np.ndarray:
    result = image.copy()
    h, w = labels.shape
    for y in range(h - 1):
        for x in range(w):
            if labels[y, x] != labels[y + 1, x]:
                cv2.line(result, (x, y), (x, y + 1), color, thickness)
    for y in range(h):
        for x in range(w - 1):
            if labels[y, x] != labels[y, x + 1]:
                cv2.line(result, (x, y), (x + 1, y), color, thickness)
    return result


def render_label_fill(labels: np.ndarray) -> np.ndarray:
    unique = sorted(np.unique(labels))
    import colorsys
    colors = []
    for i in range(len(unique)):
        hue = (i * 0.618033988749895) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.75, 0.92)
        colors.append([int(c * 255) for c in rgb])
    h, w = labels.shape
    result = np.zeros((h, w, 3), dtype=np.uint8)
    for i, lbl in enumerate(unique):
        mask = labels == lbl
        color = np.array(colors[i % len(colors)], dtype=np.uint8)
        result[mask] = color
    return result


def detect_fragments(image: np.ndarray, labels: np.ndarray,
                     min_area: int = 20, max_area: int = 1500) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    log = cv2.Laplacian(cv2.GaussianBlur(gray, (5, 5), 1.5), cv2.CV_64F)
    log_norm = cv2.normalize(np.abs(log), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, log_mask = cv2.threshold(log_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_blue = np.array([90, 20, 20])
    upper_blue = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    label_coverage = np.zeros_like(gray)
    for lbl in np.unique(labels):
        if lbl == 0:
            continue
        label_coverage[labels == lbl] = 255
    combined = cv2.bitwise_or(log_mask, blue_mask)
    combined = cv2.bitwise_and(combined, cv2.bitwise_not(label_coverage))
    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    num_labels, cc_labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    frag_mask = np.zeros_like(gray)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            frag_mask[cc_labels == i] = 255
    return frag_mask


def overlay_fragments(image: np.ndarray, frag_mask: np.ndarray) -> np.ndarray:
    result = image.copy()
    result[frag_mask > 0] = [0, 100, 255]
    return result


def process_panel(image_path: Path) -> dict:
    img = np.array(Image.open(image_path).convert("RGB"))
    cleaned = remove_text(img)
    enhanced = enhance_v(cleaned)

    labels = felzenszwalb(enhanced, scale=300, sigma=0.5, min_size=30)
    n_init = len(np.unique(labels))
    print(f"  initial: {n_init} labels")

    labels = post_merge(labels, enhanced)
    print(f"  after merge: {len(np.unique(labels))} labels")

    frag_mask = detect_fragments(img, labels)

    fill = render_label_fill(labels)
    boundaries = draw_boundaries(fill, labels)
    frag_overlay = overlay_fragments(boundaries, frag_mask)

    return {
        "original": img,
        "cleaned": cleaned,
        "labels": labels,
        "fill": fill,
        "boundaries": boundaries,
        "fragments": frag_overlay,
    }


def create_figure(results: list[dict]) -> np.ndarray:
    n = len(results)
    h, w = results[0]["original"].shape[:2]
    cols = ["Original", "Text Removed", "Label Fill", "Boundaries+Frags"]
    header_h = 35
    label_w = 100
    cell_h, cell_w = h, w
    canvas = np.ones((n * cell_h + header_h, label_w + len(cols) * cell_w, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    for c, title in enumerate(cols):
        x = label_w + c * cell_w + cell_w // 2 - len(title) * 5
        cv2.putText(canvas, title, (x, 25), font, 0.6, (0, 0, 0), 2)
    row_titles = ["Panel 1", "Panel 2", "Panel 3"]
    keys = ["original", "cleaned", "fill", "fragments"]
    for r, res in enumerate(results):
        y = header_h + r * cell_h
        cv2.putText(canvas, row_titles[r], (10, y + 30), font, 0.6, (0, 0, 0), 2)
        for c, key in enumerate(keys):
            x = label_w + c * cell_w
            img = res[key]
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            canvas[y:y + cell_h, x:x + cell_w] = img
    return canvas


def main():
    base = Path(__file__).parent.parent.parent
    panel_paths = [base / f"panel_{i}_front.png" for i in range(1, 4)]

    results = []
    for p in panel_paths:
        print(f"Processing {p.name}...")
        res = process_panel(p)
        results.append(res)

    fig = create_figure(results)
    out = base / "result_final_v3.png"
    Image.fromarray(fig).save(out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
