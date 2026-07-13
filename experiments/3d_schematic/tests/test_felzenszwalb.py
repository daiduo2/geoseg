"""Test felzenszwalb with improved text removal and post-processing."""
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


def merge_small_regions(label_img: np.ndarray, image: np.ndarray, min_ratio: float = 0.02) -> np.ndarray:
    h, w = label_img.shape
    total = h * w
    unique, counts = np.unique(label_img, return_counts=True)
    mean_colors = {lbl: image[label_img == lbl].mean(axis=0) for lbl in unique}
    small_labels = unique[counts < total * min_ratio]
    for small_lbl in small_labels:
        mask = label_img == small_lbl
        min_dist = float('inf')
        nearest = None
        for other_lbl in unique:
            if other_lbl == small_lbl:
                continue
            dist = np.linalg.norm(mean_colors[small_lbl] - mean_colors[other_lbl])
            if dist < min_dist:
                min_dist = dist
                nearest = other_lbl
        if nearest is not None:
            label_img[mask] = nearest
    return label_img


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


def draw_boundaries(image: np.ndarray, labels: np.ndarray, color: tuple = (0, 0, 0), thickness: int = 2) -> np.ndarray:
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


def main():
    base = Path(__file__).parent.parent
    panel_paths = [base / f"panel_{i}_front.png" for i in range(1, 4)]

    configs = [
        (500, 1.0, 100),
        (800, 1.0, 100),
        (500, 1.5, 100),
    ]

    for scale, sigma, min_size in configs:
        rows = []
        for p in panel_paths:
            img = np.array(Image.open(p).convert("RGB"))
            cleaned = remove_text(img)
            labels = felzenszwalb(cleaned, scale=scale, sigma=sigma, min_size=min_size)
            labels = merge_small_regions(labels, cleaned, min_ratio=0.02)
            n_labels = len(np.unique(labels))
            fill = render_label_fill(labels)
            bounded = draw_boundaries(fill, labels)
            rows.append((img, cleaned, fill, bounded))

        h, w = rows[0][0].shape[:2]
        canvas = np.ones((3 * h, 4 * w, 3), dtype=np.uint8) * 255
        for r, row in enumerate(rows):
            for c, img in enumerate(row):
                canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = img

        out = base / f"felz_s{scale}_sig{sigma}.png"
        Image.fromarray(canvas).save(out)
        print(f"scale={scale} sigma={sigma}: {n_labels} labels -> {out}")


if __name__ == "__main__":
    main()
