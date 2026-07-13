import sys; sys.path.insert(0, "/Users/daiduo2/geoseg")
from geoseg.modules.segment_engines.slic_kmeans import segment as slic_segment
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

base = Path(__file__).parent.parent.parent

def remove_text_cv(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=25, C=-5)
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    laplacian = (laplacian > np.percentile(laplacian, 80)).astype(np.uint8) * 255
    text_mask = ((adaptive > 0) & (laplacian > 0))
    from scipy import ndimage
    labeled, num = ndimage.label(text_mask)
    text_mask_clean = np.zeros_like(text_mask)
    for j in range(1, num + 1):
        comp = labeled == j
        if 8 < comp.sum() < 1200:
            text_mask_clean[comp] = True
    kernel = np.ones((5, 5), np.uint8)
    text_dilated = cv2.dilate(text_mask_clean.astype(np.uint8), kernel, iterations=2)
    inpainted = cv2.inpaint(image, text_dilated, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    blurred = cv2.GaussianBlur(inpainted, (7, 7), sigmaX=1.5)
    mask_f = text_dilated.astype(np.float32) / 255.0
    mask_f = cv2.GaussianBlur(mask_f, (5, 5), sigmaX=2)
    mask_3ch = np.stack([mask_f] * 3, axis=-1)
    return (blurred * mask_3ch + inpainted * (1 - mask_3ch)).astype(np.uint8)

for i in range(1, 4):
    p = base / f"panel_{i}_front.png"
    img = np.array(Image.open(p).convert("RGB"))
    cleaned = remove_text_cv(img)
    result = slic_segment(cleaned, n_layers=6)
    labels = result["labels"]
    # Render overlay from labels
    from skimage.segmentation import find_boundaries
    boundaries = find_boundaries(labels, mode='thick')
    overlay = cleaned.copy()
    # Color labels
    unique = sorted(np.unique(labels))
    import colorsys
    colors = []
    for j in range(len(unique)):
        hue = (j * 0.618033988749895) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.75, 0.92)
        colors.append([int(c * 255) for c in rgb])
    fill = np.zeros_like(overlay)
    for j, lbl in enumerate(unique):
        fill[labels == lbl] = colors[j % len(colors)]
    fill[boundaries] = [0, 0, 0]
    Image.fromarray(fill).save(base / f"result_slic_kmeans_{i}.png")
    print(f"Panel {i}: {len(unique)} labels")
