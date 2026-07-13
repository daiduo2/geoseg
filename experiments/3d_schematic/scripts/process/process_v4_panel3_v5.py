"""Panel 3 v5: trace tube from plume head, precise protected regions.

Key insight from v4: tube was partially detected (lower half only) and
protected regions were too broad (46/50 labels), preventing any merge.

v5 strategy:
1. Detect plume head precisely (bounded area, top position)
2. Find plume head "neck" (narrowest point at bottom)
3. Trace centerline downward from neck → this is the tube
4. Detect fragments by local darkness + compactness
5. Detect crust by top position + desaturation
6. Protected regions are SMALL and PRECISE
7. Run felz, then merge non-protected regions aggressively
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


def detect_plume_head_v5(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Detect plume head and return its mask + neck position.

    Strategy: find large warm region in upper half, then find its
    narrowest point (neck) where tube connects.
    """
    h, w = image.shape[:2]
    from skimage.color import rgb2hsv
    hsv = rgb2hsv(image)

    # Warm region in upper half
    warm = (
        (hsv[:, :, 0] > 0.08) & (hsv[:, :, 0] < 0.22) &
        (hsv[:, :, 1] > 0.08) &
        (hsv[:, :, 2] > 0.30)
    )
    warm = closing(warm, disk(5))

    # Keep only upper portion
    upper = np.zeros_like(warm)
    upper[:h * 2 // 3, :] = warm[:h * 2 // 3, :]

    # Find connected components, keep the largest warm one
    labels = label(upper)
    best_lbl = None
    best_area = 0
    for r in regionprops(labels):
        if r.area > best_area and r.area < h * w * 0.4:
            best_area = r.area
            best_lbl = r.label

    if best_lbl is None:
        return np.zeros_like(warm), (h // 2, w // 2)

    head_mask = np.zeros_like(warm)
    head_mask[labels == best_lbl] = True

    # Find neck: narrowest point at bottom of head
    ys, xs = np.where(head_mask)
    if len(ys) == 0:
        return head_mask, (h // 2, w // 2)

    # For each row in the bottom half of the head, compute width
    min_y, max_y = ys.min(), ys.max()
    neck_y = max_y
    neck_x = int(xs.mean())
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
    """Trace tube downward from neck position.

    Strategy:
    1. Start at neck (y, x)
    2. At each row downward, find the local color minimum (darkest/redder)
       within a small horizontal window
    3. Build centerline
    4. Dilate centerline to get tube mask
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    r = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)

    # Redness score: higher R and higher R-G difference
    redness = r + (r - g) * 2

    centerline = np.zeros((h, w), dtype=bool)
    cx = neck_x
    cy = neck_y

    # Trace downward
    search_w = 15
    for y in range(cy, min(h, cy + 300)):
        x0 = max(0, cx - search_w)
        x1 = min(w, cx + search_w + 1)
        row = redness[y, x0:x1]
        if row.max() < 100:  # No warm region
            break
        local_max_x = x0 + int(np.argmax(row))
        cx = local_max_x
        centerline[y, cx] = True

    # Trace upward from neck (to connect to head)
    cx = neck_x
    for y in range(cy, max(-1, cy - 100), -1):
        x0 = max(0, cx - search_w)
        x1 = min(w, cx + search_w + 1)
        row = redness[y, x0:x1]
        if row.max() < 100:
            break
        local_max_x = x0 + int(np.argmax(row))
        cx = local_max_x
        centerline[y, cx] = True

    # Dilate to tube width
    tube_mask = dilation(centerline, disk(12))
    tube_mask = closing(tube_mask, disk(5))

    return tube_mask


def detect_fragments_v5(image: np.ndarray) -> np.ndarray:
    """Detect fragments: small locally-dark compact blobs."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Local mean
    local_mean = cv2.boxFilter(gray, -1, (25, 25))
    darker = gray < (local_mean - 10)

    # Keep compact components of fragment size
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
    """Detect top crust: grey desaturated layer."""
    h, w = image.shape[:2]
    from skimage.color import rgb2hsv
    hsv = rgb2hsv(image)

    # Desaturated, medium brightness
    desat = (
        (hsv[:, :, 1] < 0.30) &
        (hsv[:, :, 2] > 0.25) &
        (hsv[:, :, 2] < 0.80)
    )
    desat = closing(desat, disk(4))

    # Top portion only
    top = np.zeros_like(desat)
    top[:h // 3 + 20, :] = desat[:h // 3 + 20, :]

    # Keep large component
    labels = label(top)
    result = np.zeros_like(top)
    for r in regionprops(labels):
        if r.area > 200 and r.area < h * w * 0.3:
            result[labels == r.label] = True

    return result


def segment_panel3_v5(image: np.ndarray) -> np.ndarray:
    """Tube-aware segmentation v5."""
    enhanced = enhance_v(image)
    h, w = image.shape[:2]

    print("  Detecting plume head + neck...")
    head_mask, (neck_y, neck_x) = detect_plume_head_v5(enhanced)
    print(f"    head pixels: {head_mask.sum()}, neck at ({neck_y}, {neck_x})")

    print("  Tracing tube from neck...")
    tube_mask = trace_tube_from_neck(enhanced, neck_y, neck_x)
    print(f"    tube pixels: {tube_mask.sum()}")

    print("  Detecting fragments...")
    frag_mask = detect_fragments_v5(enhanced).astype(bool)
    print(f"    fragment pixels: {frag_mask.sum()}")

    print("  Detecting crust...")
    crust_mask = detect_crust_v5(enhanced)
    print(f"    crust pixels: {crust_mask.sum()}")

    # Combine head and tube into one "plume" mask for protection
    plume_mask = head_mask | tube_mask

    # Run felzenszwalb
    labels = felzenszwalb(enhanced, scale=500, sigma=1.0, min_size=50)
    print(f"  Initial felz: {len(np.unique(labels))} labels")

    # Identify protected labels
    protected = set()
    for lbl in np.unique(labels):
        mask = labels == lbl
        if mask.sum() == 0:
            continue
        overlap_plume = (mask & plume_mask).sum() / mask.sum()
        overlap_frag = (mask & frag_mask).sum() / mask.sum()
        overlap_crust = (mask & crust_mask).sum() / mask.sum()

        if overlap_plume > 0.25:
            protected.add(lbl)
        if overlap_frag > 0.25:
            protected.add(lbl)
        if overlap_crust > 0.25:
            protected.add(lbl)

    print(f"  Protected: {len(protected)} labels")

    # Aggressive merge of non-protected regions
    result = labels.copy()
    total = h * w

    for _ in range(60):
        unique, counts = np.unique(result, return_counts=True)
        mean_colors = {lbl: enhanced[result == lbl].mean(axis=0) for lbl in unique}

        small = unique[counts < total * 0.02]
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
            if best_lbl is not None and best_dist < 50:
                result[mask] = best_lbl

    # Merge similar non-protected pairs
    for _ in range(30):
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
        if best_cd >= 45:
            break
        result[result == best_pair[1]] = best_pair[0]

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

    print("\nStage 2: Tube-aware segmentation v5...")
    labels = segment_panel3_v5(cleaned)

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

    out = base / "result_panel3_v5_tube_aware.png"
    Image.fromarray(canvas).save(out)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
