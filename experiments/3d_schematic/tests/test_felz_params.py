"""Quick param sweep for felzenszwalb + post-merge."""
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


def compute_color_gradient(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(sobel_x**2 + sobel_y**2)


def post_merge(label_img: np.ndarray, image: np.ndarray,
               small_ratio: float = 0.02,
               max_score: float = 1.0,
               max_color: float = 35.0) -> np.ndarray:
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
    for _ in range(20):
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
    for _ in range(20):
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
        i, j = best_pair
        result[result == j] = i
        renumber()

    return result


def render_fill(labels: np.ndarray) -> np.ndarray:
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
        result[labels == lbl] = colors[i % len(colors)]
    return result


def draw_bounds(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    result = image.copy()
    h, w = labels.shape
    for y in range(h - 1):
        for x in range(w):
            if labels[y, x] != labels[y + 1, x]:
                cv2.line(result, (x, y), (x, y + 1), (0, 0, 0), 2)
    for y in range(h):
        for x in range(w - 1):
            if labels[y, x] != labels[y, x + 1]:
                cv2.line(result, (x, y), (x + 1, y), (0, 0, 0), 2)
    return result


def main():
    base = Path(__file__).parent.parent
    panel_paths = [base / f"panel_{i}_front.png" for i in range(1, 4)]

    configs = [
        (300, 0.5, 50),
        (400, 0.8, 80),
        (500, 1.0, 100),
    ]

    for scale, sigma, min_size in configs:
        rows = []
        for p in panel_paths:
            img = np.array(Image.open(p).convert("RGB"))
            cleaned = remove_text(img)
            labels = felzenszwalb(cleaned, scale=scale, sigma=sigma, min_size=min_size)
            n_init = len(np.unique(labels))
            labels = post_merge(labels, cleaned, small_ratio=0.02, max_score=1.0, max_color=50.0)
            n_final = len(np.unique(labels))
            fill = render_fill(labels)
            bounded = draw_bounds(fill, labels)
            rows.append((img, cleaned, fill, bounded))

        h, w = rows[0][0].shape[:2]
        canvas = np.ones((3 * h, 4 * w, 3), dtype=np.uint8) * 255
        for r, row in enumerate(rows):
            for c, img in enumerate(row):
                canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = img

        out = base / f"felz_s{scale}_sig{sigma}_ms{min_size}.png"
        Image.fromarray(canvas).save(out)
        print(f"scale={scale} sigma={sigma} min={min_size}: {n_init} -> {n_final} labels -> {out}")


if __name__ == "__main__":
    main()
