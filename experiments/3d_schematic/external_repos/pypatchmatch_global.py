"""PatchMatch with global_mask for boundary protection test."""
import numpy as np
import cv2
from pathlib import Path
import time


def patchmatch_global(image: np.ndarray, mask: np.ndarray, global_mask: np.ndarray,
                      patch_size: int = 9, iterations: int = 3) -> np.ndarray:
    """PatchMatch where source patches are restricted to global_mask region."""
    assert patch_size % 2 == 1
    h, w = image.shape[:2]
    half = patch_size // 2

    img = image.astype(np.float32)
    hole = (mask > 127)
    known = ~hole

    if hole.sum() == 0:
        return image.copy()

    img_pad = np.pad(img, ((half, half), (half, half), (0, 0)), mode='reflect')
    known_pad = np.pad(known, ((half, half), (half, half)), mode='constant', constant_values=False)
    global_pad = np.pad(global_mask, ((half, half), (half, half)), mode='constant', constant_values=False)

    hy, hx = np.where(hole)
    n_hole = len(hy)

    hole_idx = np.full((h, w), -1, dtype=np.int32)
    hole_idx[hy, hx] = np.arange(n_hole)

    valid_source = known & global_mask
    valid_source[:half, :] = False
    valid_source[-half:, :] = False
    valid_source[:, :half] = False
    valid_source[:, -half:] = False
    sy_all, sx_all = np.where(valid_source)
    n_src = len(sy_all)

    if n_src == 0:
        return image.copy()

    hole_patches = np.stack([img_pad[y:y + patch_size, x:x + patch_size] for y, x in zip(hy, hx)])
    hole_valid = np.stack([known_pad[y:y + patch_size, x:x + patch_size] for y, x in zip(hy, hx)])

    def patch_dist(i: int, sy: int, sx: int) -> float:
        sp = img_pad[sy:sy + patch_size, sx:sx + patch_size]
        sv = known_pad[sy:sy + patch_size, sx:sx + patch_size]
        valid = hole_valid[i] & sv
        vc = valid.sum()
        if vc == 0:
            return np.inf
        diff = hole_patches[i] - sp
        diff[~valid] = 0
        return np.sum(diff ** 2) / vc

    rng = np.random.default_rng(42)
    rand_idx = rng.integers(0, n_src, size=n_hole)
    nnf_y = sy_all[rand_idx].astype(np.int32)
    nnf_x = sx_all[rand_idx].astype(np.int32)

    dist = np.empty(n_hole, dtype=np.float32)
    for i in range(n_hole):
        dist[i] = patch_dist(i, nnf_y[i], nnf_x[i])

    for it in range(iterations):
        order = np.arange(n_hole) if it % 2 == 0 else np.arange(n_hole - 1, -1, -1)
        improved = 0
        for idx in order:
            cy, cx = hy[idx], hx[idx]
            best_d = dist[idx]
            best_y, best_x = nnf_y[idx], nnf_x[idx]

            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w:
                    ni = hole_idx[ny, nx]
                    if ni >= 0:
                        ty, tx = nnf_y[ni], nnf_x[ni]
                        if half <= ty < h - half and half <= tx < w - half and global_mask[ty, tx]:
                            d = patch_dist(idx, ty, tx)
                            if d < best_d:
                                best_d = d
                                best_y, best_x = ty, tx

            radius = max(h, w)
            while radius >= 1:
                r = radius
                ry_lo = max(half, best_y - r)
                ry_hi = min(h - half - 1, best_y + r)
                rx_lo = max(half, best_x - r)
                rx_hi = min(w - half - 1, best_x + r)
                if ry_hi > ry_lo and rx_hi > rx_lo:
                    n_samples = min(5, (ry_hi - ry_lo) * (rx_hi - rx_lo))
                    ry_s = rng.integers(ry_lo, ry_hi + 1, size=n_samples)
                    rx_s = rng.integers(rx_lo, rx_hi + 1, size=n_samples)
                    for ty, tx in zip(ry_s, rx_s):
                        if known[ty, tx] and global_mask[ty, tx]:
                            d = patch_dist(idx, ty, tx)
                            if d < best_d:
                                best_d = d
                                best_y, best_x = ty, tx
                radius //= 2

            if best_d < dist[idx]:
                dist[idx] = best_d
                nnf_y[idx] = best_y
                nnf_x[idx] = best_x
                improved += 1

        print(f"  Iteration {it + 1}/{iterations}: {improved}/{n_hole} improved")

    vote_sum = np.zeros_like(img)
    vote_cnt = np.zeros_like(img)
    for i, (cy, cx) in enumerate(zip(hy, hx)):
        sy_i, sx_i = nnf_y[i], nnf_x[i]
        y0 = max(0, cy - half)
        x0 = max(0, cx - half)
        y1 = min(h, cy + half + 1)
        x1 = min(w, cx + half + 1)
        dy = y1 - y0
        dx = x1 - x0
        src_y0 = sy_i - half
        src_x0 = sx_i - half
        vote_sum[y0:y1, x0:x1] += img[src_y0:src_y0 + dy, src_x0:src_x0 + dx]
        vote_cnt[y0:y1, x0:x1] += 1

    result = img.copy()
    ok = vote_cnt > 0
    result[ok] = vote_sum[ok] / vote_cnt[ok]
    return np.clip(result, 0, 255).astype(np.uint8)


def run_boundary_test(panel_idx: int, box: tuple, name: str):
    BASE = Path("/Users/daiduo2/geoseg/src/3d_schematic")
    OUT = BASE / "results" / "experiment_plan_repair" / "pypatchmatch_test"
    OUT.mkdir(parents=True, exist_ok=True)

    stem = f"panel_{panel_idx}"
    image = cv2.imread(str(BASE / f"figures/panels/{stem}.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(BASE / f"experiments/text_removal_v2/final_pipeline/{stem}_mask.png"), cv2.IMREAD_GRAYSCALE)

    y1, y2, x1, x2 = box
    crop_img = image[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]

    hole_pct = np.sum(crop_mask > 127) / crop_mask.size * 100
    print(f"[{stem} {name}] {crop_img.shape} hole={hole_pct:.2f}%")

    # Telea baseline
    t0 = time.time()
    telea = cv2.inpaint(crop_img, crop_mask, 3, cv2.INPAINT_TELEA)
    t_telea = time.time() - t0

    # PatchMatch without global_mask
    t0 = time.time()
    pm_plain = patchmatch_global(crop_img, crop_mask, np.ones(crop_img.shape[:2], dtype=bool),
                                 patch_size=9, iterations=3)
    t_pm_plain = time.time() - t0

    # PatchMatch with horizontal global_mask (upper half for upper text, lower half for lower text)
    h, w = crop_img.shape[:2]
    # Detect if text is in upper or lower half of crop
    hy, hx = np.where(crop_mask > 127)
    centroid_y = int(np.mean(hy)) if len(hy) > 0 else h // 2
    global_mask = np.zeros((h, w), dtype=bool)
    if centroid_y < h // 2:
        global_mask[:h//2 + 20, :] = True
    else:
        global_mask[h//2 - 20:, :] = True

    t0 = time.time()
    pm_global = patchmatch_global(crop_img, crop_mask, global_mask, patch_size=9, iterations=3)
    t_pm_global = time.time() - t0

    print(f"  Telea {t_telea:.2f}s | PM plain {t_pm_plain:.2f}s | PM global {t_pm_global:.2f}s")

    m3 = np.stack([crop_mask] * 3, axis=-1)
    comp = np.hstack([crop_img, m3.astype(np.uint8), telea, pm_plain, pm_global])
    fname = OUT / f"{stem}_{name}_global_compare.png"
    cv2.imwrite(str(fname), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
    print(f"  Saved: {fname}")


if __name__ == "__main__":
    # Boundary-adjacent text regions
    TESTS = [
        (1, (1800, 2100, 150, 450), "boundary"),
        (2, (2400, 2700, 150, 450), "boundary"),
    ]
    for panel_idx, box, name in TESTS:
        run_boundary_test(panel_idx, box, name)
    print("\nDone.")
