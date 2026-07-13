"""Test V-channel enhancement for Panel 3 plume detection."""
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
    """CLAHE enhancement on V channel to amplify subtle brightness differences."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


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


def main():
    base = Path(__file__).parent.parent
    p3 = base / "panel_3_front.png"
    img = np.array(Image.open(p3).convert("RGB"))
    cleaned = remove_text(img)
    enhanced = enhance_v(cleaned)

    configs = [
        ("cleaned", cleaned, 300, 0.5, 30),
        ("cleaned", cleaned, 300, 0.3, 20),
        ("enhanced", enhanced, 300, 0.5, 30),
        ("enhanced", enhanced, 300, 0.3, 20),
        ("enhanced", enhanced, 200, 0.3, 20),
    ]

    rows = []
    for name, src, scale, sigma, min_size in configs:
        labels = felzenszwalb(src, scale=scale, sigma=sigma, min_size=min_size)
        n = len(np.unique(labels))
        fill = render_fill(labels)
        rows.append((src, fill, name, n))

    h, w = img.shape[:2]
    canvas = np.ones((len(rows) * h, 3 * w, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    for r, (src, fill, name, n) in enumerate(rows):
        canvas[r * h:(r + 1) * h, 0:w] = src
        canvas[r * h:(r + 1) * h, w:2 * w] = fill
        label = f"{name} s={scale} sig={sigma} n={n}"
        cv2.putText(canvas, label, (2 * w + 10, r * h + 30), font, 0.5, (0, 0, 0), 1)

    out = base / "panel3_enhance_test.png"
    Image.fromarray(canvas).save(out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
