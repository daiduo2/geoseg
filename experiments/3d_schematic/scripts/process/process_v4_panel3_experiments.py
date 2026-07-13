"""Panel 3 targeted experiments: preserve fragments + separate plume tube.

Runs on text_removed_v2 output. Tests multiple CV algorithms against
the core challenge: plume tube (orange-red vertical) vs gradient
background (orange-yellow) while keeping grey fragments.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.segmentation import felzenszwalb, quickshift, watershed
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize, dilation, erosion, disk, closing
from skimage.color import rgb2hsv, rgb2gray
from skimage.filters import sobel, threshold_multiotsu


def remove_text_v2(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Same as process_v4_two_stage."""
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


def protect_fragments(image: np.ndarray) -> np.ndarray:
    """Create a mask for grey/blue fragments that must be preserved.
    Fragments are: grey-ish (low saturation, medium brightness) small blobs
    that differ from the orange/yellow mantle background.
    """
    hsv = rgb2hsv(image)
    # Grey fragments: low saturation, medium value, neutral hue
    grey_like = (
        (hsv[:, :, 1] < 0.25) &
        (hsv[:, :, 2] > 0.25) &
        (hsv[:, :, 2] < 0.75)
    )
    # Clean up
    grey_like = closing(grey_like, disk(2))
    # Keep only connected components of appropriate fragment size
    labeled = label(grey_like)
    frag_mask = np.zeros_like(grey_like)
    for r in regionprops(labeled):
        if 20 <= r.area <= 800:
            frag_mask[labeled == r.label] = True
    return frag_mask.astype(np.uint8) * 255


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


# ---------------------------------------------------------------------------
# Algorithm 1: Quick Shift (gradient-robust mode-seeking)
# ---------------------------------------------------------------------------
def algo_quickshift(image: np.ndarray) -> np.ndarray:
    """Quick shift: mode-seeking on color+spatial features. Good for gradients."""
    labels = quickshift(image, kernel_size=5, max_dist=10, ratio=0.5, rng=42)
    return labels


# ---------------------------------------------------------------------------
# Algorithm 2: HSV-plume-seed + Watershed on color gradient
# ---------------------------------------------------------------------------
def algo_watershed_plume(image: np.ndarray) -> np.ndarray:
    """Watershed guided by color gradient with plume tube seed detection."""
    hsv = rgb2hsv(image)

    # Detect plume tube: higher saturation + warm hue (orange-red)
    # Panel 3 plume tube is reddish-orange
    plume_mask = (
        (hsv[:, :, 1] > 0.15) &
        (hsv[:, :, 0] > 0.03) & (hsv[:, :, 0] < 0.12) &  # orange-red hue range
        (hsv[:, :, 2] > 0.35)
    )
    # Clean and connect
    plume_mask = closing(plume_mask, disk(5))
    plume_mask = ndimage.binary_fill_holes(plume_mask)

    # Detect plume head: greenish-yellow region above tube
    plume_head_mask = (
        (hsv[:, :, 1] > 0.1) &
        (hsv[:, :, 0] > 0.12) & (hsv[:, :, 0] < 0.25) &  # yellow-green
        (hsv[:, :, 2] > 0.35)
    )
    plume_head_mask = closing(plume_head_mask, disk(5))

    # Top crust: grey/blue upper region
    crust_mask = (
        (hsv[:, :, 1] < 0.3) &
        (hsv[:, :, 2] > 0.3) &
        (hsv[:, :, 2] < 0.8)
    )
    crust_mask = closing(crust_mask, disk(5))

    # Build seeds: each detected region gets a unique seed value
    seeds = np.zeros(image.shape[:2], dtype=np.int32)
    seeds[plume_head_mask] = 1
    seeds[plume_mask] = 2
    seeds[crust_mask] = 3

    # Remaining area = mantle, but we need more seeds for background
    # Color gradient as elevation map for watershed
    grad = sobel(rgb2gray(image))
    grad = ndimage.gaussian_filter(grad, sigma=2)

    labels = watershed(grad, markers=seeds)
    return labels.astype(np.int32)


# ---------------------------------------------------------------------------
# Algorithm 3: Multi-scale Felzenszwalb fusion
# ---------------------------------------------------------------------------
def algo_multiscale_felz(image: np.ndarray) -> np.ndarray:
    """Run felz at multiple scales, then fuse: prefer fine detail for
    tube-like structures, prefer coarse for background."""
    labels_fine = felzenszwalb(image, scale=200, sigma=0.5, min_size=20)
    labels_coarse = felzenszwalb(image, scale=600, sigma=1.2, min_size=60)

    # Fusion: start from coarse, but where fine detects a long thin
    # component not present in coarse, keep it
    h, w = image.shape[:2]
    result = labels_coarse.copy()

    # For each coarse region, check if fine sub-divides it into
    # elongated (tube-like) components
    for coarse_lbl in np.unique(labels_coarse):
        mask = labels_coarse == coarse_lbl
        if mask.sum() < 500:
            continue

        fine_in_region = labels_fine[mask]
        unique_fine = np.unique(fine_in_region)

        for fl in unique_fine:
            fine_mask = np.zeros((h, w), dtype=bool)
            fine_mask[mask] = (fine_in_region == fl)
            if fine_mask.sum() < 100:
                continue
            # Check elongation
            ys, xs = np.where(fine_mask)
            if len(ys) < 10:
                continue
            height = ys.max() - ys.min() + 1
            width = xs.max() - xs.min() + 1
            if height > width * 2.5 and height > 60:  # tube-like
                # Keep this fine label separate
                result[fine_mask] = labels_fine.max() + fl + 1

    # Renumber
    remap = {}
    next_lbl = 0
    new_result = np.zeros_like(result)
    for lbl in sorted(np.unique(result)):
        remap[lbl] = next_lbl
        next_lbl += 1
    for old, new in remap.items():
        new_result[result == old] = new
    return new_result


# ---------------------------------------------------------------------------
# Algorithm 4: HSV hue-channel multi-Otsu + region props
# ---------------------------------------------------------------------------
def algo_hue_otsu(image: np.ndarray) -> np.ndarray:
    """Multi-Otsu on hue channel separates orange-red (plume) from
    yellow-orange (mantle) from grey/blue (crust/fragments)."""
    hsv = rgb2hsv(image)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Multi-Otsu on hue for color-based classes
    thresholds = threshold_multiotsu(hue, classes=4)
    hue_labels = np.digitize(hue, bins=thresholds)

    # Refine with saturation: high-sat regions in warm hues = plume
    warm_high_sat = ((hue > 0.03) & (hue < 0.15) & (sat > 0.15))
    plume_refined = (hue_labels == 2) | ((hue_labels == 1) & warm_high_sat)

    # Fragments: low sat + medium val
    fragment_mask = ((sat < 0.25) & (val > 0.25) & (val < 0.75))
    fragment_mask = closing(fragment_mask, disk(2))

    # Combine
    result = np.zeros_like(hue_labels)
    result[plume_refined] = 1
    result[fragment_mask] = 2
    result[~plume_refined & ~fragment_mask] = 3

    # Split crust (top) from mantle (bottom) by position
    h = image.shape[0]
    for lbl in np.unique(result):
        mask = result == lbl
        ys = np.where(mask)[0]
        if len(ys) == 0:
            continue
        mean_y = ys.mean()
        if mean_y < h * 0.25 and lbl == 3:
            # Top region is crust
            result[mask] = 4

    return result.astype(np.int32)


# ---------------------------------------------------------------------------
# Algorithm 5: Morphological tube detection + region growing
# ---------------------------------------------------------------------------
def algo_tube_detect(image: np.ndarray) -> np.ndarray:
    """Detect tube-like structures via morphology + skeleton, then
    grow regions from detected tubes."""
    hsv = rgb2hsv(image)
    # Warm colored regions (potential plume)
    warm = (
        (hsv[:, :, 0] > 0.02) & (hsv[:, :, 0] < 0.15) &
        (hsv[:, :, 1] > 0.1) & (hsv[:, :, 2] > 0.3)
    )
    # Remove small noise
    warm = closing(warm, disk(3))
    warm = erosion(warm, disk(1))

    # Skeletonize warm region to find tube backbone
    skel = skeletonize(warm)

    # Detect long vertical skeleton segments
    labels_skel = label(skel)
    tube_seeds = np.zeros_like(skel)
    for r in regionprops(labels_skel):
        if r.area < 30:
            continue
        ys = np.where(labels_skel == r.label)[0]
        xs = np.where(labels_skel == r.label)[1]
        if len(ys) < 10:
            continue
        height = ys.max() - ys.min() + 1
        width = xs.max() - xs.max() + 1
        if height > 40:  # Long enough to be a tube
            tube_seeds[labels_skel == r.label] = True

    # Dilate seeds to get plume region
    plume_mask = dilation(tube_seeds, disk(8))
    plume_mask = closing(plume_mask, disk(5))

    # Also detect head region (larger warm area above tube)
    head_mask = warm & (~plume_mask)
    head_mask = closing(head_mask, disk(5))

    # Detect fragments
    frag_mask = protect_fragments(image).astype(bool)

    # Assign labels
    result = np.zeros(image.shape[:2], dtype=np.int32)
    result[plume_mask] = 1
    result[head_mask] = 2
    result[frag_mask] = 3

    # Remaining = background mantle/crust, split by vertical position
    remaining = (result == 0)
    ys_rem = np.where(remaining)[0]
    if len(ys_rem) > 0:
        median_y = np.median(ys_rem)
        h = image.shape[0]
        for y in range(h):
            row_mask = remaining[y, :]
            if y < median_y:
                result[y, row_mask] = 4  # crust/upper mantle
            else:
                result[y, row_mask] = 5  # lower mantle

    return result


# ---------------------------------------------------------------------------
# Algorithm 6: SLIC with high n_segments (text-robust, low fragments)
# ---------------------------------------------------------------------------
def algo_slic_textured(image: np.ndarray, n_segments: int = 600) -> np.ndarray:
    """SLIC with many segments, then merge based on color+position."""
    from skimage.segmentation import slic
    labels = slic(image, n_segments=n_segments, compactness=10,
                  start_label=0, channel_axis=2)

    # Merge superpixels: keep fragments separate, merge similar mantle regions
    h, w = image.shape[:2]
    result = labels.copy()
    frag_mask = protect_fragments(image).astype(bool)

    unique = np.unique(labels)
    # Compute mean color and position per superpixel
    sp_data = {}
    for lbl in unique:
        mask = labels == lbl
        if mask.sum() == 0:
            continue
        mean_color = image[mask].mean(axis=0)
        ys, xs = np.where(mask)
        mean_y, mean_x = ys.mean(), xs.mean()
        is_frag = (frag_mask & mask).any()
        sp_data[lbl] = {
            'color': mean_color,
            'y': mean_y,
            'x': mean_x,
            'area': mask.sum(),
            'is_frag': is_frag,
        }

    # Merge small non-fragment superpixels with most similar neighbor
    changed = True
    while changed:
        changed = False
        for lbl in list(sp_data.keys()):
            if lbl not in sp_data or sp_data[lbl]['is_frag']:
                continue
            if sp_data[lbl]['area'] > h * w * 0.01:
                continue
            # Find most similar adjacent superpixel
            mask = result == lbl
            dilated = ndimage.binary_dilation(mask, structure=np.ones((3, 3)))
            adj_labels = set(result[dilated & ~mask]) - {0}
            best_dist = float('inf')
            best_lbl = None
            for adj in adj_labels:
                if adj not in sp_data:
                    continue
                dist = np.linalg.norm(sp_data[lbl]['color'] - sp_data[adj]['color'])
                if dist < best_dist:
                    best_dist = dist
                    best_lbl = adj
            if best_lbl is not None and best_dist < 40:
                result[mask] = best_lbl
                # Update merged superpixel
                if best_lbl in sp_data:
                    sp_data[best_lbl]['area'] += sp_data[lbl]['area']
                del sp_data[lbl]
                changed = True

    return result


def run_all(base: Path, cleaned: np.ndarray, image: np.ndarray):
    """Run all algorithms on Panel 3 cleaned image."""
    enhanced = enhance_v(cleaned)

    algorithms = [
        ("quickshift", algo_quickshift),
        ("watershed_plume", algo_watershed_plume),
        ("multiscale_felz", algo_multiscale_felz),
        ("hue_otsu", algo_hue_otsu),
        ("tube_detect", algo_tube_detect),
        ("slic_textured", algo_slic_textured),
    ]

    results = []
    for name, fn in algorithms:
        print(f"Running {name}...")
        try:
            labels = fn(enhanced)
            n = len(np.unique(labels))
            fill = render_labels(labels)
            boundaries = draw_boundaries(fill, labels)
            print(f"  -> {n} labels")

            out_fill = base / f"p3_{name}_fill.png"
            out_bound = base / f"p3_{name}_bound.png"
            Image.fromarray(fill).save(out_fill)
            Image.fromarray(boundaries).save(out_bound)
            results.append((name, n, out_fill, out_bound))
        except Exception as e:
            print(f"  ERROR: {e}")

    return results


def create_comparison_figure(base: Path, original: np.ndarray, cleaned: np.ndarray,
                             results: list) -> np.ndarray:
    """Create comparison figure: original | cleaned | each algo result."""
    n_algos = len(results)
    h, w = original.shape[:2]
    # Layout: top row = labels, bottom rows = each algo
    cols = 2 + n_algos  # Original, Cleaned, + each algo
    header_h = 40
    canvas = np.ones((h + header_h, cols * w, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX

    titles = ["Original", "Text Removed v2"] + [r[0] for r in results]
    images = [original, cleaned]
    for _, _, fill_path, _ in results:
        images.append(np.array(Image.open(fill_path)))

    for c, (title, img) in enumerate(zip(titles, images)):
        x = c * w
        cv2.putText(canvas, title, (x + 5, 28), font, 0.5, (0, 0, 0), 1)
        canvas[header_h:header_h + h, x:x + w] = img[:h, :w]

    return canvas


def main():
    base = Path(__file__).parent.parent.parent
    img = np.array(Image.open(base / "panel_3_front.png").convert("RGB"))

    print("Stage 1: Text removal v2...")
    cleaned, _ = remove_text_v2(img)
    print("  done.")

    print("\nStage 2: Running segmentation algorithms...")
    results = run_all(base, cleaned, img)

    print("\nCreating comparison figure...")
    fig = create_comparison_figure(base, img, cleaned, results)
    out = base / "result_panel3_all_algos.png"
    Image.fromarray(fig).save(out)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
