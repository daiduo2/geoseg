"""
Median-fill text removal experiment.
Compares four median-based fill strategies against cv2.inpaint control.
"""
import cv2
import numpy as np
from pathlib import Path
from collections import deque


def detect_text_mser(gray, min_area=10, max_area=2000, max_aspect=20):
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    for region in regions:
        region = region.reshape(-1, 1, 2)
        area = cv2.contourArea(region)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(region)
        if w == 0 or h == 0:
            continue
        aspect = max(w, h) / min(w, h)
        if aspect > max_aspect:
            continue
        cv2.fillPoly(mask, [region], 255)
    return mask


def detect_text_laplacian(gray, threshold=15, max_area=2000):
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    _, mask = cv2.threshold(laplacian, threshold, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > max_area:
            mask[labels == i] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def build_mask(image_rgb, brightness_thresh=170, dilate_iter=1):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    mask_orig = detect_text_mser(gray)
    mask_inv = detect_text_mser(255 - gray)
    mask_lap = detect_text_laplacian(gray)
    combined = cv2.bitwise_or(mask_orig, mask_inv)
    combined = cv2.bitwise_or(combined, mask_lap)
    if brightness_thresh > 0:
        brightness_mask = (gray > brightness_thresh).astype(np.uint8) * 255
        combined = cv2.bitwise_and(combined, brightness_mask)
    if dilate_iter > 0:
        kernel = np.ones((3, 3), np.uint8)
        combined = cv2.dilate(combined, kernel, iterations=dilate_iter)
    return combined


def method_a_whole_median(image_rgb, mask, ksize):
    """Whole-image median blur then mask-replace."""
    mask_bool = mask.astype(bool)
    result = image_rgb.copy().astype(np.float32)
    for ch in range(3):
        chan = image_rgb[:, :, ch]
        med = cv2.medianBlur(chan, ksize).astype(np.float32)
        result[:, :, ch] = np.where(mask_bool, med, chan)
    return result.astype(np.uint8)


def method_b_annular_median(image_rgb, mask, outer, inner=1):
    """
    For each masked pixel, compute median of pixels in annular neighborhood
    [inner, outer] using Manhattan distance (diamond shape).
    inner=1 ensures we don't sample the pixel itself or immediate 4-neighbors
    that might be part of the text stroke.
    """
    mask_bool = mask.astype(bool)
    result = image_rgb.copy().astype(np.float32)
    h, w = image_rgb.shape[:2]

    ys, xs = np.mgrid[0:h, 0:w]
    my, mx = np.where(mask_bool)
    if len(my) == 0:
        return result.astype(np.uint8)

    for ch in range(3):
        chan = image_rgb[:, :, ch].astype(np.float32)
        out = chan.copy()
        for yy, xx in zip(my, mx):
            y0, y1 = max(0, yy - outer), min(h, yy + outer + 1)
            x0, x1 = max(0, xx - outer), min(w, xx + outer + 1)
            patch = chan[y0:y1, x0:x1]
            pmy = ys[y0:y1, x0:x1] - yy
            pmx = xs[y0:y1, x0:x1] - xx
            dist = np.abs(pmy) + np.abs(pmx)
            ring = (dist >= inner) & (dist <= outer)
            m_patch = mask_bool[y0:y1, x0:x1]
            valid = ring & (~m_patch)
            vals = patch[valid]
            if len(vals) > 0:
                out[yy, xx] = np.median(vals)
            else:
                nm = ~m_patch
                if np.any(nm):
                    out[yy, xx] = np.median(patch[nm])
        result[:, :, ch] = out
    return result.astype(np.uint8)


def method_c_block_median(image_rgb, mask, block_size=20):
    """
    Divide mask into blocks. For each block, compute median of non-mask pixels
    in that block and fill all masked pixels in the block with that median.
    """
    mask_bool = mask.astype(bool)
    result = image_rgb.copy().astype(np.float32)
    h, w = image_rgb.shape[:2]

    for ch in range(3):
        chan = image_rgb[:, :, ch].astype(np.float32)
        out = chan.copy()
        for y0 in range(0, h, block_size):
            for x0 in range(0, w, block_size):
                y1 = min(h, y0 + block_size)
                x1 = min(w, x0 + block_size)
                block_mask = mask_bool[y0:y1, x0:x1]
                if not np.any(block_mask):
                    continue
                block = chan[y0:y1, x0:x1]
                vals = block[~block_mask]
                if len(vals) > 0:
                    med = np.median(vals)
                    out[y0:y1, x0:x1][block_mask] = med
        result[:, :, ch] = out
    return result.astype(np.uint8)


def method_d_gaussian_blend(image_rgb, mask, ksize=15, sigma=5.0):
    """
    Whole-image median blur + Gaussian-weighted blend at mask edges.
    Creates a soft transition band around mask boundary.
    """
    mask_bool = mask.astype(bool)
    dist_out = cv2.distanceTransform((~mask_bool).astype(np.uint8), cv2.DIST_L2, 5)
    dist_in = cv2.distanceTransform(mask_bool.astype(np.uint8), cv2.DIST_L2, 5)

    signed = dist_in.astype(np.float32) - dist_out.astype(np.float32)
    weight = np.exp(-(signed ** 2) / (2 * sigma ** 2))
    weight = np.clip(weight, 0.0, 1.0)

    result = image_rgb.copy().astype(np.float32)
    for ch in range(3):
        chan = image_rgb[:, :, ch].astype(np.float32)
        med = cv2.medianBlur(image_rgb[:, :, ch], ksize).astype(np.float32)
        blended = weight * med + (1.0 - weight) * chan
        result[:, :, ch] = blended
    return np.clip(result, 0, 255).astype(np.uint8)


def method_e_distance_weighted_median(image_rgb, mask, max_radius=15):
    """
    Extra method: for each masked pixel, collect non-mask neighbors within
    max_radius and compute weighted median where closer pixels have higher weight.
    Simpler than full annular but more principled than whole-image median.
    """
    mask_bool = mask.astype(bool)
    result = image_rgb.copy().astype(np.float32)
    h, w = image_rgb.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]

    my, mx = np.where(mask_bool)
    if len(my) == 0:
        return result.astype(np.uint8)

    for ch in range(3):
        chan = image_rgb[:, :, ch].astype(np.float32)
        out = chan.copy()
        for yy, xx in zip(my, mx):
            y0, y1 = max(0, yy - max_radius), min(h, yy + max_radius + 1)
            x0, x1 = max(0, xx - max_radius), min(w, xx + max_radius + 1)
            patch = chan[y0:y1, x0:x1]
            pmy = ys[y0:y1, x0:x1] - yy
            pmx = xs[y0:y1, x0:x1] - xx
            dists = np.sqrt(pmy**2 + pmx**2)
            m_patch = mask_bool[y0:y1, x0:x1]
            valid = (dists <= max_radius) & (~m_patch)
            vals = patch[valid]
            dvals = dists[valid]
            if len(vals) > 0:
                weights = np.exp(-dvals / (max_radius / 3 + 1e-6))
                order = np.argsort(vals)
                svals = vals[order]
                sweights = weights[order]
                cumw = np.cumsum(sweights)
                half = cumw[-1] / 2.0
                idx = np.searchsorted(cumw, half)
                out[yy, xx] = svals[min(idx, len(svals)-1)]
            else:
                nm = ~m_patch
                if np.any(nm):
                    out[yy, xx] = np.median(patch[nm])
        result[:, :, ch] = out
    return result.astype(np.uint8)


def method_f_median_of_nonmask(image_rgb, mask, ksize=15):
    """
    Method F: For each channel, compute medianBlur on image where mask pixels
    are replaced by NaN (or ignored), then fill mask with that result.
    Implemented by: medianBlur on image with mask pixels set to a sentinel,
    but that's not clean. Instead: for each pixel, take ksize x ksize window
    and compute median of non-mask pixels only.
    For efficiency, use integral images or just accept it's slow.
    """
    mask_bool = mask.astype(bool)
    result = image_rgb.copy().astype(np.float32)
    h, w = image_rgb.shape[:2]
    half = ksize // 2

    for ch in range(3):
        chan = image_rgb[:, :, ch].astype(np.float32)
        out = chan.copy()
        my, mx = np.where(mask_bool)
        for yy, xx in zip(my, mx):
            y0, y1 = max(0, yy - half), min(h, yy + half + 1)
            x0, x1 = max(0, xx - half), min(w, xx + half + 1)
            patch = chan[y0:y1, x0:x1]
            m_patch = mask_bool[y0:y1, x0:x1]
            vals = patch[~m_patch]
            if len(vals) > 0:
                out[yy, xx] = np.median(vals)
        result[:, :, ch] = out
    return result.astype(np.uint8)


def run_experiment():
    BASE = Path("/Users/daiduo2/geoseg/src/3d_schematic")
    PANELS = [BASE / "figures" / "panels" / f"panel_{i}.png" for i in (1, 2, 3)]
    OUT = BASE / "experiments" / "text_removal_v2" / "median_fill"
    OUT.mkdir(parents=True, exist_ok=True)

    a_ksizes = [7, 11, 15, 21]
    b_params = [(5, 1), (7, 2), (10, 3)]
    c_blocks = [10, 20, 30]
    d_sigmas = [3.0, 5.0, 8.0]
    e_radii = [7, 10, 15]
    f_ksizes = [7, 11, 15, 21]

    for p in PANELS:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = build_mask(img_rgb, brightness_thresh=170, dilate_iter=1)
        coverage = np.count_nonzero(mask) / mask.size * 100
        print(f"\n{p.stem}: mask coverage {coverage:.2f}%")

        cv2.imwrite(str(OUT / f"{p.stem}_mask.png"), mask)

        ctrl = cv2.inpaint(img_rgb, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        cv2.imwrite(str(OUT / f"{p.stem}_control_inpaint.png"), cv2.cvtColor(ctrl, cv2.COLOR_RGB2BGR))

        for k in a_ksizes:
            res = method_a_whole_median(img_rgb, mask, k)
            cv2.imwrite(str(OUT / f"{p.stem}_A_median_ksize{k}.png"), cv2.cvtColor(res, cv2.COLOR_RGB2BGR))
            print(f"  A ksize={k} done")

        for outer, inner in b_params:
            res = method_b_annular_median(img_rgb, mask, outer, inner)
            cv2.imwrite(str(OUT / f"{p.stem}_B_annular_outer{outer}_inner{inner}.png"), cv2.cvtColor(res, cv2.COLOR_RGB2BGR))
            print(f"  B outer={outer} inner={inner} done")

        for bs in c_blocks:
            res = method_c_block_median(img_rgb, mask, bs)
            cv2.imwrite(str(OUT / f"{p.stem}_C_block_bs{bs}.png"), cv2.cvtColor(res, cv2.COLOR_RGB2BGR))
            print(f"  C block={bs} done")

        for sig in d_sigmas:
            res = method_d_gaussian_blend(img_rgb, mask, ksize=15, sigma=sig)
            cv2.imwrite(str(OUT / f"{p.stem}_D_gauss_sigma{sig:.1f}.png"), cv2.cvtColor(res, cv2.COLOR_RGB2BGR))
            print(f"  D sigma={sig} done")

        for r in e_radii:
            res = method_e_distance_weighted_median(img_rgb, mask, r)
            cv2.imwrite(str(OUT / f"{p.stem}_E_distweight_r{r}.png"), cv2.cvtColor(res, cv2.COLOR_RGB2BGR))
            print(f"  E radius={r} done")

        for k in f_ksizes:
            res = method_f_median_of_nonmask(img_rgb, mask, k)
            cv2.imwrite(str(OUT / f"{p.stem}_F_nonmask_median_k{k}.png"), cv2.cvtColor(res, cv2.COLOR_RGB2BGR))
            print(f"  F ksize={k} done")

    print(f"\nAll results saved to {OUT}")


if __name__ == "__main__":
    run_experiment()
