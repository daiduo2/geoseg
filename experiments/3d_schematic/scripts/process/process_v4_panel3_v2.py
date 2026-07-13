"""Panel 3 v2: tube-aware protected segmentation.

Key insight from experiments:
- tube_detect finds the plume tube but is too thick
- multiscale_felz finds thin tube but fragments background
- Solution: detect tube skeleton, protect it during segmentation,
  then felz on remaining regions with fragment preservation.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.segmentation import felzenszwalb
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize, dilation, erosion, disk, closing
from skimage.color import rgb2hsv


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


def detect_tube_mask(image: np.ndarray) -> np.ndarray:
    """Detect the plume tube via HSV thresholding + skeleton + shape filtering.
    Returns a mask for the tube region (bool)."""
    hsv = rgb2hsv(image)
    # Warm colored regions: orange-red hue, moderate saturation
    warm = (
        (hsv[:, :, 0] > 0.02) & (hsv[:, :, 0] < 0.12) &
        (hsv[:, :, 1] > 0.08) & (hsv[:, :, 2] > 0.25)
    )
    warm = closing(warm, disk(4))
    warm = erosion(warm, disk(1))

    # Skeletonize to find backbone
    skel = skeletonize(warm)

    # Find long vertical skeleton components
    skel_labels = label(skel)
    tube_mask = np.zeros_like(skel)
    for r in regionprops(skel_labels):
        if r.area < 20:
            continue
        ys = np.where(skel_labels == r.label)[0]
        xs = np.where(skel_labels == r.label)[1]
        if len(ys) < 15:
            continue
        height = ys.max() - ys.min() + 1
        width = max(1, xs.max() - xs.min() + 1)
        aspect = height / width
        # Long thin structure = tube
        if height > 50 and aspect > 2.0:
            tube_mask[skel_labels == r.label] = True

    # Dilate from skeleton to get tube region, but not too much
    tube_region = dilation(tube_mask, disk(6))
    tube_region = closing(tube_region, disk(4))
    return tube_region


def detect_fragments(image: np.ndarray) -> np.ndarray:
    """Detect grey/blue fragments that must be preserved."""
    hsv = rgb2hsv(image)
    # Low saturation + medium value = grey-ish
    grey_like = (
        (hsv[:, :, 1] < 0.30) &
        (hsv[:, :, 2] > 0.20) &
        (hsv[:, :, 2] < 0.80)
    )
    grey_like = closing(grey_like, disk(2))
    labeled = label(grey_like)
    frag_mask = np.zeros_like(grey_like)
    for r in regionprops(labeled):
        if 15 <= r.area <= 1000:
            frag_mask[labeled == r.label] = True
    return frag_mask.astype(np.uint8) * 255


def detect_plume_head(image: np.ndarray) -> np.ndarray:
    """Detect plume head: warm-colored bulbous region near top."""
    hsv = rgb2hsv(image)
    h, w = image.shape[:2]
    # Plume head is warm-colored but more yellow-green, near top
    head = (
        (hsv[:, :, 0] > 0.10) & (hsv[:, :, 0] < 0.25) &
        (hsv[:, :, 1] > 0.08) &
        (hsv[:, :, 2] > 0.30)
    )
    head = closing(head, disk(5))
    # Restrict to upper half
    head_mask = np.zeros_like(head)
    head_mask[:h//2 + 40, :] = head[:h//2 + 40, :]
    return head_mask


def detect_crust(image: np.ndarray) -> np.ndarray:
    """Detect top crust layer."""
    hsv = rgb2hsv(image)
    h, w = image.shape[:2]
    # Crust: grey/blue upper region
    crust = (
        (hsv[:, :, 1] < 0.35) &
        (hsv[:, :, 2] > 0.25) &
        (hsv[:, :, 2] < 0.85)
    )
    crust = closing(crust, disk(4))
    # Restrict to top portion
    crust_mask = np.zeros_like(crust)
    crust_mask[:h//3 + 20, :] = crust[:h//3 + 20, :]
    return crust_mask


def segment_panel3_v2(image: np.ndarray) -> np.ndarray:
    """Tube-aware segmentation for Panel 3.

    Strategy:
    1. Detect and protect: tube, fragments, plume_head, crust
    2. Run felzenszwalb on the image
    3. Post-merge: merge felz regions EXCEPT protected ones
    4. Combine protected regions with merged felz regions
    """
    enhanced = enhance_v(image)

    # Detect protected regions
    tube_mask = detect_tube_mask(enhanced)
    frag_mask = detect_fragments(enhanced).astype(bool)
    head_mask = detect_plume_head(enhanced)
    crust_mask = detect_crust(enhanced)

    # Run felzenszwalb
    labels = felzenszwalb(enhanced, scale=400, sigma=0.8, min_size=40)

    # Renumber to contiguous
    remap = {}
    next_lbl = 0
    result = np.zeros_like(labels)
    for lbl in sorted(np.unique(labels)):
        remap[lbl] = next_lbl
        next_lbl += 1
    for old, new in remap.items():
        result[labels == old] = new

    # Phase 1: merge small non-protected regions
    h, w = image.shape[:2]
    total = h * w
    mean_colors = {lbl: enhanced[result == lbl].mean(axis=0) for lbl in np.unique(result)}

    for _ in range(50):
        unique, counts = np.unique(result, return_counts=True)
        mean_colors = {lbl: enhanced[result == lbl].mean(axis=0) for lbl in unique}

        # Find regions that are NOT protected
        protected_labels = set()
        for lbl in unique:
            mask = result == lbl
            if (mask & tube_mask).sum() > mask.sum() * 0.3:
                protected_labels.add(lbl)
            if (mask & frag_mask).sum() > mask.sum() * 0.3:
                protected_labels.add(lbl)
            if (mask & head_mask).sum() > mask.sum() * 0.3:
                protected_labels.add(lbl)
            if (mask & crust_mask).sum() > mask.sum() * 0.3:
                protected_labels.add(lbl)

        small = unique[counts < total * 0.008]
        small = [s for s in small if s not in protected_labels]
        if len(small) == 0:
            break

        for small_lbl in small:
            mask = result == small_lbl
            # Find adjacent labels
            dilated = ndimage.binary_dilation(mask, structure=np.ones((3, 3)))
            adj_labels = set(result[dilated & ~mask]) - {-1}
            adj_labels = adj_labels - protected_labels
            if not adj_labels:
                continue
            best_dist = float('inf')
            best_lbl = None
            for adj in adj_labels:
                dist = np.linalg.norm(mean_colors[small_lbl] - mean_colors[adj])
                if dist < best_dist:
                    best_dist = dist
                    best_lbl = adj
            if best_lbl is not None and best_dist < 45:
                result[mask] = best_lbl

    # Phase 2: merge similar adjacent non-protected pairs
    for _ in range(30):
        unique = sorted(np.unique(result))
        mean_colors = {lbl: enhanced[result == lbl].mean(axis=0) for lbl in unique}

        # Recompute protected
        protected_labels = set()
        for lbl in unique:
            mask = result == lbl
            if (mask & tube_mask).sum() > mask.sum() * 0.3:
                protected_labels.add(lbl)
            if (mask & frag_mask).sum() > mask.sum() * 0.3:
                protected_labels.add(lbl)
            if (mask & head_mask).sum() > mask.sum() * 0.3:
                protected_labels.add(lbl)
            if (mask & crust_mask).sum() > mask.sum() * 0.3:
                protected_labels.add(lbl)

        if len(unique) <= 6:
            break

        pair_data = {}
        for y in range(h):
            for x in range(w - 1):
                a, b = result[y, x], result[y, x + 1]
                if a != b and a not in protected_labels and b not in protected_labels:
                    pair = (min(a, b), max(a, b))
                    pair_data.setdefault(pair, []).append(1)
            if y < h - 1:
                for x in range(w):
                    a, b = result[y, x], result[y + 1, x]
                    if a != b and a not in protected_labels and b not in protected_labels:
                        pair = (min(a, b), max(a, b))
                        pair_data.setdefault(pair, []).append(1)
        if not pair_data:
            break

        scores = []
        for pair, _ in pair_data.items():
            i, j = pair
            cd = np.linalg.norm(mean_colors[i] - mean_colors[j])
            scores.append((cd, pair))
        scores.sort()
        best_cd, best_pair = scores[0]
        if best_cd >= 40:
            break
        result[result == best_pair[1]] = best_pair[0]

    # Final renumber
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

    print("\nStage 2: Tube-aware segmentation...")
    labels = segment_panel3_v2(cleaned)

    fill = render_labels(labels)
    boundaries = draw_boundaries(fill, labels)

    # Create comparison: original | cleaned | fill | boundaries
    h, w = img.shape[:2]
    canvas = np.ones((h, 4 * w, 3), dtype=np.uint8) * 255
    imgs = [img, cleaned, fill, boundaries]
    titles = ["Original", "Text Removed v2", "Label Fill", "Boundaries"]
    for i, (im, t) in enumerate(zip(imgs, titles)):
        x = i * w
        canvas[:h, x:x+w] = im[:h, :w]
        cv2.putText(canvas, t, (x + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    out = base / "result_panel3_v2_tube_aware.png"
    Image.fromarray(canvas).save(out)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
