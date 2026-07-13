"""V4 unified pipeline: text removal + per-panel optimal segmentation.

- P1/P2: unified-fragment preservation (no tube separation needed)
- P3: tube-aware + unified-fragment preservation

Key strategy (from v6):
1. Detect fragments by local darkness + compactness
2. Run felzenszwalb coarse segmentation
3. Aggressively merge non-fragment regions
4. After merge, unify ALL fragment regions into ONE label
5. P3 additionally: detect plume head + trace tube downward from neck
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


# ========== Stage 1: Text removal ==========

def remove_text_v2(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
               skip_labels: set | None = None,
               small_ratio: float = 0.015,
               max_score: float = 0.8,
               max_color: float = 45.0) -> np.ndarray:
    h, w = label_img.shape
    gradient = compute_color_gradient(image)
    result = label_img.copy()
    total = h * w
    skip = skip_labels or set()

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
            if small_lbl in skip:
                continue
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
                if adj_lbl in skip:
                    continue
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
                if a != b and a not in skip and b not in skip:
                    pair = (min(a, b), max(a, b))
                    pair_data.setdefault(pair, []).append(gradient[y, x])
            if y < h - 1:
                for x in range(w):
                    a, b = result[y, x], result[y + 1, x]
                    if a != b and a not in skip and b not in skip:
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


# ========== Shared detection utilities ==========

def detect_fragments(image: np.ndarray) -> np.ndarray:
    """Detect fragments: small locally-dark compact blobs.
    Works across all panels (grey in P1/P3, blue in P2).
    """
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


def detect_crust(image: np.ndarray) -> np.ndarray:
    """Detect top crust: grey desaturated layer."""
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


def detect_plume(image: np.ndarray) -> np.ndarray:
    """Detect plume: large warm-colored region (red/orange/yellow)."""
    h, w = image.shape[:2]
    hsv_img = rgb2hsv(image)
    # Wider HSV range to catch red/orange plume across all panels
    warm = (
        (hsv_img[:, :, 0] > 0.00) & (hsv_img[:, :, 0] < 0.16) &
        (hsv_img[:, :, 1] > 0.10) &
        (hsv_img[:, :, 2] > 0.20)
    )
    warm = closing(warm, disk(5))
    labels = label(warm)
    best_lbl = None
    best_area = 0
    for r in regionprops(labels):
        if r.area > best_area and r.area < h * w * 0.7:
            best_area = r.area
            best_lbl = r.label
    plume_mask = np.zeros_like(warm)
    if best_lbl is not None:
        plume_mask[labels == best_lbl] = True
    return plume_mask


# ========== P1/P2: Unified fragment segmentation ==========

def segment_with_fragments(image: np.ndarray, enhanced: np.ndarray) -> np.ndarray:
    """P1/P2 segmentation: baseline felz + gradient-aware merge, preserve fragments.

    Strategy:
    1. Run felzenszwalb (baseline params)
    2. Detect fragments
    3. Identify fragment labels
    4. Run post_merge with fragment labels skipped
    5. Unify all fragment regions into ONE label
    """
    h, w = image.shape[:2]
    frag_mask = detect_fragments(enhanced).astype(bool)
    print(f"  frag={frag_mask.sum()}")

    labels = felzenszwalb(enhanced, scale=400, sigma=0.8, min_size=30)
    print(f"  Initial felz: {len(np.unique(labels))} labels")

    # Identify fragment labels (protected from merge)
    frag_labels = set()
    for lbl in np.unique(labels):
        mask = labels == lbl
        if mask.sum() == 0:
            continue
        if (mask & frag_mask).sum() / mask.sum() > 0.15:
            frag_labels.add(lbl)

    print(f"  Fragment labels: {len(frag_labels)}")

    # Run post_merge, skipping fragment labels
    result = post_merge(labels, enhanced, skip_labels=frag_labels,
                        small_ratio=0.008, max_score=0.7, max_color=55.0)
    print(f"  After merge: {len(np.unique(result))} labels")

    # Unify all fragment regions into ONE label
    final_frag_labels = set()
    for lbl in np.unique(result):
        mask = result == lbl
        if mask.sum() == 0:
            continue
        if (mask & frag_mask).sum() / mask.sum() > 0.05:
            final_frag_labels.add(lbl)

    print(f"  Fragment labels to unify: {len(final_frag_labels)}")

    if final_frag_labels:
        max_lbl = int(np.max(result)) + 1
        for lbl in final_frag_labels:
            result[result == lbl] = max_lbl

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


# ========== P3: Tube-aware + unified fragments ==========

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
        x0 = max(0, cx - 20)
        x1 = min(w, cx + 21)
        row = redness[y, x0:x1]
        if row.max() < 80:
            break
        cx = x0 + int(np.argmax(row))
        centerline[y, cx] = True
    tube_mask = dilation(centerline, disk(14))
    tube_mask = closing(tube_mask, disk(4))
    return tube_mask


def segment_panel3(image: np.ndarray, enhanced: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]

    head_mask, (neck_y, neck_x) = detect_plume_head_v5(enhanced)
    tube_mask = trace_tube_from_neck(enhanced, neck_y, neck_x)
    # Mask tube to NOT include head
    tube_mask = tube_mask & (~head_mask)
    frag_mask = detect_fragments(enhanced).astype(bool)
    crust_mask = detect_crust(enhanced)

    plume_mask = head_mask | tube_mask
    print(f"  head={head_mask.sum()}, tube={tube_mask.sum()}, "
          f"frag={frag_mask.sum()}, crust={crust_mask.sum()}")

    labels = felzenszwalb(enhanced, scale=500, sigma=1.0, min_size=100)

    # Protected: head, tube, and crust (one label each, the one with max overlap)
    head_overlaps = {}
    tube_overlaps = {}
    crust_overlaps = {}
    for lbl in np.unique(labels):
        mask = labels == lbl
        if mask.sum() == 0:
            continue
        h_ov = (mask & head_mask).sum()
        t_ov = (mask & tube_mask).sum()
        c_ov = (mask & crust_mask).sum()
        if h_ov > 0:
            head_overlaps[lbl] = h_ov
        if t_ov > 0:
            tube_overlaps[lbl] = t_ov
        if c_ov > 0:
            crust_overlaps[lbl] = c_ov

    protected = set()
    if head_overlaps:
        protected.add(max(head_overlaps, key=head_overlaps.get))
    if tube_overlaps:
        protected.add(max(tube_overlaps, key=tube_overlaps.get))
    if crust_overlaps:
        protected.add(max(crust_overlaps, key=crust_overlaps.get))

    print(f"  Protected (head+tube+crust): {len(protected)} labels")

    result = labels.copy()
    total = h * w

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

    # Unify fragment regions
    frag_labels = set()
    for lbl in np.unique(result):
        mask = result == lbl
        if mask.sum() == 0:
            continue
        if (mask & frag_mask).sum() > 0:
            frag_labels.add(lbl)

    final_frag_labels = set()
    for lbl in frag_labels:
        mask = result == lbl
        if (mask & head_mask).sum() / mask.sum() > 0.30:
            continue
        if (mask & tube_mask).sum() / mask.sum() > 0.30:
            continue
        if (mask & crust_mask).sum() / mask.sum() > 0.30:
            continue
        final_frag_labels.add(lbl)

    print(f"  Fragment labels to unify: {len(final_frag_labels)}")

    if final_frag_labels:
        max_lbl = int(np.max(result)) + 1
        for lbl in final_frag_labels:
            result[result == lbl] = max_lbl

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


# ========== Visualization ==========

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


# ========== Main ==========

def main():
    base = Path(__file__).parent.parent.parent
    panel_paths = [base / f"panel_{i}_front.png" for i in range(1, 4)]

    results = []
    for idx, p in enumerate(panel_paths):
        print(f"\n{'='*50}")
        print(f"Processing {p.name}...")
        print(f"{'='*50}")
        img = np.array(Image.open(p).convert("RGB"))

        # Stage 1: Text removal
        cleaned, _ = remove_text_v2(img)

        # Stage 2: Enhance
        enhanced = enhance_v(cleaned)

        # Stage 3: Per-panel segmentation
        if idx == 0:  # Panel 1
            labels = segment_with_fragments(img, enhanced)
        elif idx == 1:  # Panel 2
            labels = segment_with_fragments(img, enhanced)
        else:  # Panel 3
            labels = segment_panel3(img, enhanced)

        fill = render_labels(labels)
        boundaries = draw_boundaries(fill, labels)

        results.append({
            "original": img,
            "cleaned": cleaned,
            "labels": labels,
            "fill": fill,
            "boundaries": boundaries,
        })

    fig = create_figure(results)
    out = base / "result_v4_unified.png"
    Image.fromarray(fig).save(out)
    print(f"\n{'='*50}")
    print(f"Saved: {out}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
