import sys; sys.path.insert(0, "/Users/daiduo2/geoseg/src/3d_schematic")
from process_final_v3 import *
from pathlib import Path
import cv2
from sklearn.cluster import KMeans

base = Path(__file__).parent.parent.parent

def remove_text_kmeans(image, k=5):
    h, w = image.shape[:2]
    pixels = image.reshape(-1, 3).astype(np.float32)
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(pixels)
    centers = np.array([pixels[labels == i].mean(axis=0) for i in range(k)])
    brightness = np.linalg.norm(centers, axis=1)
    saturation = np.std(centers, axis=1)
    counts = np.bincount(labels, minlength=k)
    areas = counts / len(labels)
    brightness_norm = brightness / np.max(brightness)
    saturation_norm = saturation / (np.max(saturation) + 1e-6)
    area_norm = 1.0 - (areas / (np.max(areas) + 1e-6))
    scores = 0.5 * brightness_norm + 0.3 * (1.0 - saturation_norm) + 0.2 * area_norm
    text_cluster = int(np.argmax(scores))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[labels.reshape(h, w) == text_cluster] = 255
    kernel = np.ones((3, 3), np.uint8)
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)
    return cv2.inpaint(image, mask_dilated, 5, cv2.INPAINT_TELEA)

for i in range(1, 4):
    p = base / f"panel_{i}_front.png"
    img = np.array(Image.open(p).convert("RGB"))
    cleaned = remove_text_kmeans(img)
    enhanced = enhance_v(cleaned)
    labels = felzenszwalb(enhanced, scale=300, sigma=0.5, min_size=30)
    labels = post_merge(labels, enhanced)
    fill = render_label_fill(labels)
    boundaries = draw_boundaries(fill, labels)
    Image.fromarray(boundaries).save(base / f"result_kmeans_text_{i}.png")
    print(f"Panel {i}: {len(np.unique(labels))} labels")
