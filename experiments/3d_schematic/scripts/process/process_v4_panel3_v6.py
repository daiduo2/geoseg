"""Panel 3 v6: unified fragments label + aggressive merge.

Key fix from v5: each fragment was its own protected label, preventing all
merging. v6 strategy:
1. Detect tube, head, crust, fragments as before
2. Run felz + aggressive merge (only tube/head/crust protected)
3. After merge, collect all fragment regions into ONE unified label
4. This gives: crust | plume_head | tube | mantle | fragments = ~5 labels
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


def detect_plume_head_v5(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = image.shape[:2]
    hsv_img = rgb2hsv(image)
    warm = (
        (hsv_img[:, :, 0] > 0.08) & (hsv_img[:, :, 0] < 0.22) &
        (hsv_img[:, :, 1] > 0.08) &
        (hsv_img[:, :, 2] > 0.30)
    )
    warm = closing(warm, disk(5))
    upper = np.zeros_like(warm)
    upper[:h * 2 // 3, :] = warm[:h * 2 // 3, :]
    labels = label(upper)
    best_lbl = None
    best_area = 0
    for r in regionprops(labels):
        if r.area > best_area and r.area < h * w * 0.15:
            best_area = r.area
            best_lbl = r.label
    head_mask = np.zeros_like(warm)
    if best_lbl is not None:
        head_mask[labels == best_lbl] = True
    ys, xs = np.where(head_mask)
    neck_y, neck_x = h // 2, w // 2
    if len(ys) > 0:
        min_y, max_y = ys.min(), ys.max()
        min_width = float('inf')
        for y in range(min_y + (max_y - min_y) // 2, max_y + 1):
            row_xs = xs[ys == y]
            if len(row_xs) > 0:
                width = row_xs.max() - row_xs.min() + 1
                if width < min_width:
                    min_width = width
                    neck_y = y
                    neck_x = int(row_xs.mean())
    return head_mask, (neck_y, neck_x)


def trace_tube_from_neck(image: np.ndarray, neck_y: int, neck_x: int) -> np.ndarray:
    h, w = image.shape[:2]
    r = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)
    redness = r + (r - g) * 2
    centerline = np.zeros((h, w), dtype=bool)
    cx = neck_x
    for y in range(neck_y, min(h, neck_y + 300)):
        x0 = max(0, cx - 15)
        x1 = min(w, cx + 16)
        row = redness[y, x0:x1]
        if row.max() < 100:
            break
        cx = x0 + int(np.argmax(row))
        centerline[y, cx] = True
    cx = neck_x
    for y in range(neck_y, max(-1, neck_y - 80), -1):
        x0 = max(0, cx - 15)
        x1 = min(w, cx + 16)
        row = redness[y, x0:x1]
        if row.max() < 100:
            break
        cx = x0 + int(np.argmax(row))
        centerline[y, cx] = True
    tube_mask = dilation(centerline, disk(10))
    tube_mask = closing(tube_mask, disk(4))
    return tube_mask


def detect_fragments_v5(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    local_mean = cv2.boxFilter(gray, -1, (25, 25))
    darker = gray < (local_mean - 10)
    labels = label(darker)
    frag_mask = np.zeros_like(darker)
    for r in regionprops(labels):
        if 20 <= r.area <= 500:
            bbox_h = r.bbox[2] - r.bbox[0]
            bbox_w = r.bbox[3] - r.bbox[1]
            if bbox_h > 0 and bbox_w > 0:
                aspect = max(bbox_h, bbox_w) / min(bbox_h, bbox_w)
                if aspect < 2.5:
                    frag_mask[labels == r.label] = True
    return frag_mask.astype(np.uint8) * 255


def detect_crust_v5(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    hsv_img = rgb2hsv(image)
    desat = (
        (hsv_img[:, :, 1] < 0.30) &
        (hsv_img[:, :, 2] > 0.25) &
        (hsv_img[:, :, 2] < 0.80)
    )
    desat = closing(desat, disk(4))
    top = np.zeros_like(desat)
    top[:h // 3 + 15, :] = desat[:h // 3 + 15, :]
    labels = label(top)
    crust_mask = np.zeros_like(top)
    for r in regionprops(labels):
        if 200 <= r.area <= h * w * 0.15:
            crust_mask[labels == r.label] = True
    return crust_mask


def segment_panel3_v6(image: np.ndarray) -> np.ndarray:
    enhanced = enhance_v(image)
    h, w = image.shape[:2]

    head_mask, (neck_y, neck_x) = detect_plume_head_v5(enhanced)
    tube_mask = trace_tube_from_neck(enhanced, neck_y, neck_x)
    frag_mask = detect_fragments_v5(enhanced).astype(bool)
    crust_mask = detect_crust_v5(enhanced)

    plume_mask = head_mask | tube_mask

    print(f"  head={head_mask.sum()}, tube={tube_mask.sum()}, "
          f"frag={frag_mask.sum()}, crust={crust_mask.sum()}")

    # Step 1: felz + aggressive merge (ONLY plume and crust protected)
    labels = felzenszwalb(enhanced, scale=500, sigma=1.0, min_size=50)

    # Protected: plume regions and crust
    protected = set()
    for lbl in np.unique(labels):
        mask = labels == lbl
        if mask.sum() == 0:
            continue
        if (mask & plume_mask).sum() / mask.sum() > 0.10:
            protected.add(lbl)
        if (mask & crust_mask).sum() / mask.sum() > 0.10:
            protected.add(lbl)

    print(f"  Protected (plume+crust): {len(protected)} labels")

    result = labels.copy()
    total = h * w

    # Aggressive merge of non-protected regions
    for _ in range(80):
        unique, counts = np.unique(result, return_counts=True)
        mean_colors = {lbl: enhanced[result == lbl].mean(axis=0) for lbl in unique}
        small = unique[counts < total * 0.025]
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
            if best_lbl is not None and best_dist < 65:
                result[mask] = best_lbl

    # Merge similar non-protected pairs
    for _ in range(40):
        unique = sorted(np.unique(result))
        mean_colors = {lbl: enhanced[result == lbl].mean(axis=0) for lbl in unique}
        if len(unique) <= 5:
            break
        pair_data = {}
        for y in range(h):
            for x in range(w - 1):
                a, b = result[y, x], result[y, x + 1]
                if a != b and a not in protected and b not in protected:
                    pair = (min(a, b), max(a, b))
                    pair_data.setdefault(pair, []).append(1)
            if y < h - 1:
                for x in range(w):
                    a, b = result[y, x], result[y + 1, x]
                    if a != b and a not in protected and b not in protected:
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
        if best_cd >= 60:
            break
        result[result == best_pair[1]] = best_pair[0]

    print(f"  After merge: {len(np.unique(result))} labels")

    # Step 2: UNIFY all fragment regions into ONE label
    # Find all labels that overlap with fragment mask
    frag_labels = set()
    for lbl in np.unique(result):
        mask = result == lbl
        if mask.sum() == 0:
            continue
        if (mask & frag_mask).sum() > 0:
            frag_labels.add(lbl)

    # Exclude labels that are primarily plume or crust
    final_frag_labels = set()
    for lbl in frag_labels:
        mask = result == lbl
        if (mask & plume_mask).sum() / mask.sum() > 0.30:
            continue  # This is plume, not fragment
        if (mask & crust_mask).sum() / mask.sum() > 0.30:
            continue  # This is crust, not fragment
        final_frag_labels.add(lbl)

    print(f"  Fragment labels to unify: {len(final_frag_labels)}")

    # Assign all fragment regions to a new unified label
    max_lbl = int(np.max(result)) + 1
    frag_unified_lbl = max_lbl
    for lbl in final_frag_labels:
        result[result == lbl] = frag_unified_lbl

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

    print("\nStage 2: Tube-aware segmentation v6...")
    labels = segment_panel3_v6(cleaned)

    fill = render_labels(labels)
    boundaries = draw_boundaries(fill, labels)

    h, w = img.shape[:2]
    canvas = np.ones((h, 4 * w, 3), dtype=np.uint8) * 255
    imgs = [img, cleaned, fill, boundaries]
    titles = ["Original", "Text Removed v2", "Label Fill", "Boundaries"]
    for i, (im, t) in enumerate(zip(imgs, titles)):
        x = i * w
        canvas[:h, x:x+w] = im[:h, :w]
        cv2.putText(canvas, t, (x + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    out = base / "result_panel3_v6_tube_aware.png"
    Image.fromarray(canvas).save(out)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
