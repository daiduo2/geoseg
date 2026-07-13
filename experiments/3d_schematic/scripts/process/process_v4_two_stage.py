"""V4 two-stage pipeline: enhanced text removal (hole backfill) + per-panel best algorithm.

Stage 1: Remove text with hole backfill — after inpainting, identify remaining
text-masked pixels and fill them with the median color of their neighborhood
(ignoring other text-masked pixels). This prevents holes from becoming
independent regions during segmentation.

Stage 2: Per-panel best algorithm based on visual audit results:
- Panel 1: coarse felzenszwalb (scale=500, sigma=1.0, min_size=50)
- Panel 2: coarse felzenszwalb (scale=500, sigma=1.0, min_size=50)
- Panel 3: conservative post_merge on fine felzenszwalb (scale=300, sigma=0.5)
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.segmentation import felzenszwalb


def remove_text_v2(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Enhanced text removal with hole backfill.

    Returns (cleaned_image, text_mask_dilated) so downstream steps can
    use the mask if needed.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=25, C=-5
    )
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    laplacian = (laplacian > np.percentile(laplacian, 80)).astype(np.uint8) * 255
    text_mask = ((adaptive > 0) & (laplacian > 0))

    # Clean text mask: keep only appropriately-sized components
    labeled, num = ndimage.label(text_mask)
    text_mask_clean = np.zeros_like(text_mask)
    for i in range(1, num + 1):
        comp = labeled == i
        if 8 < comp.sum() < 1200:
            text_mask_clean[comp] = True

    # Dilate to cover text edges
    kernel = np.ones((5, 5), np.uint8)
    text_dilated = cv2.dilate(text_mask_clean.astype(np.uint8), kernel, iterations=2)

    # Inpaint
    inpainted = cv2.inpaint(image, text_dilated, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

    # --- NEW: hole backfill ---
    # For pixels still inside the dilated text mask, fill with median of
    # non-masked neighborhood pixels (5x5 window). This prevents inpainting
    # artifacts from becoming spurious independent regions.
    cleaned = inpainted.copy()
    mask_bool = text_dilated.astype(bool)

    # Compute neighborhood median for each channel, ignoring masked pixels
    for ch in range(3):
        channel = inpainted[:, :, ch].astype(np.float32)
        # Use a masked median approach: for each masked pixel, look at 7x7 neighborhood
        ys, xs = np.where(mask_bool)
        for y, x in zip(ys, xs):
            y0, y1 = max(0, y - 3), min(image.shape[0], y + 4)
            x0, x1 = max(0, x - 3), min(image.shape[1], x + 4)
            patch = channel[y0:y1, x0:x1]
            patch_mask = mask_bool[y0:y1, x0:x1]
            valid = patch[~patch_mask]
            if len(valid) > 0:
                cleaned[y, x, ch] = int(np.median(valid))
            else:
                cleaned[y, x, ch] = int(channel[y, x])

    # Additional smoothing: Gaussian blend on the mask boundary to reduce
    # any remaining hard edges between inpainted and backfilled regions
    blurred = cv2.GaussianBlur(cleaned, (7, 7), sigmaX=1.5)
    mask_f = text_dilated.astype(np.float32) / 255.0
    mask_f = cv2.GaussianBlur(mask_f, (5, 5), sigmaX=2)
    mask_3ch = np.stack([mask_f] * 3, axis=-1)
    cleaned = (blurred * mask_3ch + cleaned * (1 - mask_3ch)).astype(np.uint8)

    return cleaned, text_dilated


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


# --- Per-panel best algorithm configs ---

def segment_panel_1(enhanced: np.ndarray) -> np.ndarray:
    """Panel 1 best: coarse felzenszwalb."""
    labels = felzenszwalb(enhanced, scale=500, sigma=1.0, min_size=50)
    labels = post_merge(labels, enhanced, small_ratio=0.01, max_score=0.6)
    return labels


def segment_panel_2(enhanced: np.ndarray) -> np.ndarray:
    """Panel 2 best: coarse felzenszwalb."""
    labels = felzenszwalb(enhanced, scale=500, sigma=1.0, min_size=50)
    labels = post_merge(labels, enhanced, small_ratio=0.01, max_score=0.6)
    return labels


def segment_panel_3(enhanced: np.ndarray) -> np.ndarray:
    """Panel 3 best: fine felzenszwalb + conservative merge."""
    labels = felzenszwalb(enhanced, scale=300, sigma=0.5, min_size=30)
    labels = post_merge(labels, enhanced, small_ratio=0.008, max_score=0.5, max_color=35.0)
    return labels


def create_figure(results: list[dict]) -> np.ndarray:
    n = len(results)
    h, w = results[0]["original"].shape[:2]
    cols = ["Original", "Text Removed v2", "Label Fill", "Boundaries"]
    header_h = 35
    label_w = 100
    cell_h, cell_w = h, w
    canvas = np.ones((n * cell_h + header_h, label_w + len(cols) * cell_w, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    for c, title in enumerate(cols):
        x = label_w + c * cell_w + cell_w // 2 - len(title) * 5
        cv2.putText(canvas, title, (x, 25), font, 0.6, (0, 0, 0), 2)
    row_titles = ["Panel 1", "Panel 2", "Panel 3"]
    keys = ["original", "cleaned", "fill", "boundaries"]
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
    segment_funcs = [segment_panel_1, segment_panel_2, segment_panel_3]

    results = []
    for idx, p in enumerate(panel_paths):
        print(f"Processing {p.name}...")
        img = np.array(Image.open(p).convert("RGB"))

        # Stage 1: Enhanced text removal
        cleaned, _ = remove_text_v2(img)

        # Stage 2: Enhance + per-panel best segmentation
        enhanced = enhance_v(cleaned)
        labels = segment_funcs[idx](enhanced)
        n_labels = len(np.unique(labels))
        print(f"  -> {n_labels} labels")

        fill = render_label_fill(labels)
        boundaries = draw_boundaries(fill, labels)

        results.append({
            "original": img,
            "cleaned": cleaned,
            "labels": labels,
            "fill": fill,
            "boundaries": boundaries,
        })

    fig = create_figure(results)
    out = base / "result_v4_two_stage.png"
    Image.fromarray(fig).save(out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
