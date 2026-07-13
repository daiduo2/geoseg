"""Panel 3 v4: direct tube detection via center-axis tracing + dark blob fragments.

Panel 3 structure from visual inspection:
- Tube: central vertical reddish-orange streak, slightly darker/redder than background
- Fragments: small grey-ish square-ish dark spots scattered in orange mantle
- Background: uniform yellow-orange mantle
- Plume head: bulbous top region above tube
- Crust: grey top layer

Detection strategy:
1. Tube: center-axis vertical tracing (R-dominant narrow streak)
2. Fragments: locally darker compact blobs via adaptive thresholding
3. Head: large warm bulbous region above center
4. Crust: grey top layer by position + desaturation
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.segmentation import felzenszwalb
from skimage.measure import label, regionprops
from skimage.morphology import dilation, disk, closing


def remove_text_v2(image: np.ndarray) -> np.ndarray:
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
    cleaned = inpainted.copy()
    mask_bool = text_dilated.astype(bool)
    for ch in range(3):
        channel = inpainted[:, :, ch].astype(np.float32)
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
    blurred = cv2.GaussianBlur(cleaned, (7, 7), sigmaX=1.5)
    mask_f = text_dilated.astype(np.float32) / 255.0
    mask_f = cv2.GaussianBlur(mask_f, (5, 5), sigmaX=2)
    mask_3ch = np.stack([mask_f] * 3, axis=-1)
    return (blurred * mask_3ch + cleaned * (1 - mask_3ch)).astype(np.uint8)


def enhance_v(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def detect_tube_v4(image: np.ndarray) -> np.ndarray:
    """Detect central vertical tube by R-dominance in center strip."""
    h, w = image.shape[:2]
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # Tube is reddish (R > G and R > B)
    red_dominant = (r.astype(np.int16) > g.astype(np.int16) + 5) & \
                   (r.astype(np.int16) > b.astype(np.int16) + 5)

    # Tube is in center strip
    cx = w // 2
    center_strip = np.zeros_like(red_dominant)
    margin = min(45, w // 4)
    center_strip[:, max(0, cx - margin):min(w, cx + margin)] = True

    tube_cand = red_dominant & center_strip

    # Find connected components, keep longest vertical one
    labels = label(tube_cand)
    best_lbl = None
    best_score = 0
    for rprop in regionprops(labels):
        ys = np.where(labels == rprop.label)[0]
        if len(ys) < 20:
            continue
        height = ys.max() - ys.min() + 1
        width = max(1, rprop.bbox[3] - rprop.bbox[1])
        aspect = height / width
        # Score: long, thin, vertical
        score = height * aspect
        if score > best_score:
            best_score = score
            best_lbl = rprop.label

    tube_mask = np.zeros_like(tube_cand)
    if best_lbl is not None:
        tube_mask[labels == best_lbl] = True
        # Moderate dilation to cover tube width
        tube_mask = dilation(tube_mask, disk(8))
        tube_mask = closing(tube_mask, disk(4))

    return tube_mask


def detect_fragments_v4(image: np.ndarray) -> np.ndarray:
    """Detect fragments as locally darker compact blobs."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Local mean to find darker spots
    local_mean = cv2.boxFilter(gray, -1, (21, 21))
    darker = gray < (local_mean - 12)

    # Keep only small compact components
    labels = label(darker)
    frag_mask = np.zeros_like(darker)
    for r in regionprops(labels):
        if 20 <= r.area <= 600:
            # Compactness check
            bbox_h = r.bbox[2] - r.bbox[0]
            bbox_w = r.bbox[3] - r.bbox[1]
            if bbox_h > 0 and bbox_w > 0:
                aspect = max(bbox_h, bbox_w) / min(bbox_h, bbox_w)
                if aspect < 3.0 and r.solidity > 0.4:
                    frag_mask[labels == r.label] = True

    return frag_mask.astype(np.uint8) * 255


def detect_plume_head_v4(image: np.ndarray) -> np.ndarray:
    """Detect plume head: large warm bulbous region in upper half, above tube."""
    h, w = image.shape[:2]
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # Warm = high R and G, moderate overall brightness
    warm = (r > 120) & (g > 80) & (b < 180)
    warm = closing(warm, disk(5))

    # Upper half
    upper = np.zeros_like(warm)
    upper[:h * 2 // 3, :] = warm[:h * 2 // 3, :]

    # Keep large connected components
    labels = label(upper)
    head_mask = np.zeros_like(upper)
    for r in regionprops(labels):
        if r.area > 800:
            head_mask[labels == r.label] = True

    return head_mask


def detect_crust_v4(image: np.ndarray) -> np.ndarray:
    """Detect top crust: desaturated upper region."""
    h, w = image.shape[:2]
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # Desaturated = channels are similar
    maxc = np.maximum(np.maximum(r, g), b).astype(np.float32)
    minc = np.minimum(np.minimum(r, g), b).astype(np.float32)
    sat = (maxc - minc) / (maxc + 1e-6)

    desat = sat < 0.25
    desat = closing(desat, disk(5))

    # Top portion
    crust_mask = np.zeros_like(desat)
    crust_mask[:h // 3 + 25, :] = desat[:h // 3 + 25, :]

    # Keep large components
    labels = label(crust_mask)
    result = np.zeros_like(crust_mask)
    for r in regionprops(labels):
        if r.area > 300:
            result[labels == r.label] = True

    return result


def segment_panel3_v4(image: np.ndarray) -> np.ndarray:
    """Tube-aware segmentation v4."""
    enhanced = enhance_v(image)
    h, w = image.shape[:2]

    print("  Detecting tube...")
    tube_mask = detect_tube_v4(enhanced)
    print(f"    tube pixels: {tube_mask.sum()}")

    print("  Detecting fragments...")
    frag_mask = detect_fragments_v4(enhanced).astype(bool)
    print(f"    fragment pixels: {frag_mask.sum()}")

    print("  Detecting plume head...")
    head_mask = detect_plume_head_v4(enhanced)
    print(f"    head pixels: {head_mask.sum()}")

    print("  Detecting crust...")
    crust_mask = detect_crust_v4(enhanced)
    print(f"    crust pixels: {crust_mask.sum()}")

    # Run coarse felzenszwalb
    labels = felzenszwalb(enhanced, scale=500, sigma=1.0, min_size=50)
    print(f"  Initial felz: {len(np.unique(labels))} labels")

    # Identify protected labels by overlap
    protected = set()
    for lbl in np.unique(labels):
        mask = labels == lbl
        if mask.sum() == 0:
            continue
        overlap_tube = (mask & tube_mask).sum() / mask.sum()
        overlap_frag = (mask & frag_mask).sum() / mask.sum()
        overlap_head = (mask & head_mask).sum() / mask.sum()
        overlap_crust = (mask & crust_mask).sum() / mask.sum()

        if overlap_tube > 0.20:
            protected.add(lbl)
        if overlap_frag > 0.25:
            protected.add(lbl)
        if overlap_head > 0.25:
            protected.add(lbl)
        if overlap_crust > 0.25:
            protected.add(lbl)

    print(f"  Protected: {len(protected)} labels")

    # Merge small non-protected regions
    result = labels.copy()
    total = h * w

    for _ in range(50):
        unique, counts = np.unique(result, return_counts=True)
        mean_colors = {lbl: enhanced[result == lbl].mean(axis=0) for lbl in unique}

        small = unique[counts < total * 0.015]
        small = [s for s in small if s not in protected]
        if len(small) == 0:
            break

        for small_lbl in small:
            mask = result == small_lbl
            dilated = ndimage.binary_dilation(mask, structure=np.ones((3, 3)))
            adj = set(result[dilated & ~mask]) - {-1} - protected
            if not adj:
                continue
            best_dist = float('inf')
            best_lbl = None
            for a in adj:
                d = np.linalg.norm(mean_colors[small_lbl] - mean_colors[a])
                if d < best_dist:
                    best_dist = d
                    best_lbl = a
            if best_lbl is not None and best_dist < 45:
                result[mask] = best_lbl

    # Renumber
    remap = {}
    next_lbl = 0
    final = np.zeros_like(result)
    for lbl in sorted(np.unique(result)):
        remap[lbl] = next_lbl
        next_lbl += 1
    for old, new in remap.items():
        final[result == old] = new

    print(f"  Final: {len(np.unique(final))} labels")
    return final


def render_labels(labels: np.ndarray) -> np.ndarray:
    import colorsys
    unique = sorted(np.unique(labels))
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


def main():
    base = Path(__file__).parent.parent.parent
    img = np.array(Image.open(base / "panel_3_front.png").convert("RGB"))

    print("Stage 1: Text removal v2...")
    cleaned = remove_text_v2(img)

    print("\nStage 2: Tube-aware segmentation v4...")
    labels = segment_panel3_v4(cleaned)

    fill = render_labels(labels)
    boundaries = draw_boundaries(fill, labels)

    # Comparison figure
    h, w = img.shape[:2]
    canvas = np.ones((h, 4 * w, 3), dtype=np.uint8) * 255
    imgs = [img, cleaned, fill, boundaries]
    titles = ["Original", "Text Removed v2", "Label Fill", "Boundaries"]
    for i, (im, t) in enumerate(zip(imgs, titles)):
        x = i * w
        canvas[:h, x:x+w] = im[:h, :w]
        cv2.putText(canvas, t, (x + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    out = base / "result_panel3_v4_tube_aware.png"
    Image.fromarray(canvas).save(out)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
