"""Vectorized pure-Python PatchMatch for small crop validation.
Optimized with numpy sliding_window_view + batched distance computation.
"""
import numpy as np
import cv2
from pathlib import Path
import time


def patchmatch_vectorized(image: np.ndarray, mask: np.ndarray,
                          patch_size: int = 7, source_stride: int = 3) -> np.ndarray:
    """
    Fast vectorized PatchMatch approximation.

    Args:
        image: (H, W, 3) uint8
        mask:  (H, W) uint8, 255 = hole
        patch_size: odd
        source_stride: sample every N-th pixel as source candidate

    Returns:
        (H, W, 3) uint8 inpainted image
    """
    assert patch_size % 2 == 1
    h, w = image.shape[:2]
    half = patch_size // 2

    img = image.astype(np.float32)
    hole = (mask > 127)
    known = ~hole

    if hole.sum() == 0:
        return image.copy()

    # Pad image and mask
    img_pad = np.pad(img, ((half, half), (half, half), (0, 0)), mode='reflect')
    known_pad = np.pad(known, ((half, half), (half, half)), mode='constant', constant_values=False)

    # Extract all patches via sliding window: shape (H, W, ps, ps, 3)
    patches = np.lib.stride_tricks.sliding_window_view(img_pad, (patch_size, patch_size, 3))
    patches = patches.reshape(h, w, patch_size, patch_size, 3)

    # Valid mask for each patch: center must be known AND we only consider valid source centers
    valid_source = known.copy()
    valid_source[:half, :] = False
    valid_source[-half:, :] = False
    valid_source[:, :half] = False
    valid_source[:, -half:] = False

    # Sample source centers
    sy, sx = np.where(valid_source)
    sample_idx = np.arange(0, len(sy), source_stride)
    sy, sx = sy[sample_idx], sx[sample_idx]
    n_src = len(sy)

    # Hole centers
    hy, hx = np.where(hole)
    n_hole = len(hy)

    if n_src == 0 or n_hole == 0:
        return image.copy()

    # Source patches: (n_src, ps, ps, 3)
    src_patches = patches[sy, sx]
    # Source validity: the known_pad region under each source patch
    src_valid = np.stack([known_pad[y:y + patch_size, x:x + patch_size] for y, x in zip(sy, sx)])

    # Hole patches and their validity
    hole_patches = patches[hy, hx]  # (n_hole, ps, ps, 3)
    hole_valid = np.stack([known_pad[y:y + patch_size, x:x + patch_size] for y, x in zip(hy, hx)])

    # For each hole, find best source via batched distance
    best_src = np.zeros((n_hole, 2), dtype=np.int32)
    best_dist = np.full(n_hole, np.inf)

    batch_src = 256
    for i in range(0, n_src, batch_src):
        sp = src_patches[i:i + batch_src]          # (B, ps, ps, 3)
        sv = src_valid[i:i + batch_src]            # (B, ps, ps)

        # Expand: (1, B, ps, ps, 3) vs (n_hole, 1, ps, ps, 3)
        hp_exp = hole_patches[:, None, ...]        # (n_hole, 1, ps, ps, 3)
        sp_exp = sp[None, ...]                     # (1, B, ps, ps, 3)
        hv_exp = hole_valid[:, None, ...]          # (n_hole, 1, ps, ps)
        sv_exp = sv[None, ...]                     # (1, B, ps, ps)

        valid = hv_exp & sv_exp                    # (n_hole, B, ps, ps)
        diff = hp_exp - sp_exp                     # (n_hole, B, ps, ps, 3)
        sqdiff = np.sum(diff ** 2, axis=-1)        # (n_hole, B, ps, ps)

        valid_count = np.sum(valid, axis=(2, 3))   # (n_hole, B)
        dist = np.full((n_hole, batch_src), np.inf)
        ok = valid_count > 0
        # Sum only where valid: use valid as mask
        dist[ok] = np.sum(sqdiff[ok] * valid[ok], axis=(1, 2)) / valid_count[ok]

        # Truncate dist to actual B size
        actual_b = sp.shape[0]
        dist = dist[:, :actual_b]

        batch_best = np.argmin(dist, axis=1)
        batch_best_dist = dist[np.arange(n_hole), batch_best]
        improved = batch_best_dist < best_dist
        best_dist[improved] = batch_best_dist[improved]
        best_src[improved] = np.stack([sy[i + batch_best[improved]],
                                       sx[i + batch_best[improved]]], axis=1)

    # Vote reconstruction
    vote_sum = np.zeros_like(img)
    vote_cnt = np.zeros_like(img)

    for hi, (cy, cx) in enumerate(zip(hy, hx)):
        sy_i, sx_i = best_src[hi]
        # Copy patch from source center to hole center
        y0 = cy - half
        x0 = cx - half
        src_y0 = sy_i - half
        src_x0 = sx_i - half

        y1 = min(h, y0 + patch_size)
        x1 = min(w, x0 + patch_size)
        dy = y1 - y0
        dx = x1 - x0

        vote_sum[y0:y1, x0:x1] += img[src_y0:src_y0 + dy, src_x0:src_x0 + dx]
        vote_cnt[y0:y1, x0:x1] += 1

    result = img.copy()
    ok = vote_cnt > 0
    result[ok] = vote_sum[ok] / vote_cnt[ok]
    return np.clip(result, 0, 255).astype(np.uint8)


def run_crop_test(panel_idx: int, crop_name: str, box: tuple,
                  patch_size: int = 7, source_stride: int = 3):
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
    print(f"[{stem} {crop_name}] {crop_img.shape} hole={hole_pct:.2f}% ps={patch_size} stride={source_stride}")

    # Baselines
    t0 = time.time()
    telea = cv2.inpaint(crop_img, crop_mask, 3, cv2.INPAINT_TELEA)
    t_telea = time.time() - t0

    t0 = time.time()
    ns = cv2.inpaint(crop_img, crop_mask, 3, cv2.INPAINT_NS)
    t_ns = time.time() - t0

    # PatchMatch
    t0 = time.time()
    pm = patchmatch_vectorized(crop_img, crop_mask, patch_size=patch_size, source_stride=source_stride)
    t_pm = time.time() - t0

    print(f"  Telea {t_telea:.2f}s | NS {t_ns:.2f}s | PM {t_pm:.2f}s")

    # Save comparison
    m3 = np.stack([crop_mask] * 3, axis=-1)
    comp = np.hstack([
        crop_img,
        m3.astype(np.uint8),
        telea,
        ns,
        pm,
    ])
    fname = OUT / f"{stem}_{crop_name}_ps{patch_size}_st{source_stride}.png"
    cv2.imwrite(str(fname), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
    return pm, telea, ns


if __name__ == "__main__":
    # Focus on residues visible in prior audit
    CROPS = {
        1: [
            ("top_left", (0, 400, 0, 600)),
            ("mid_left", (1200, 1600, 0, 600)),
            ("bottom_right", (3000, 3480, 1200, 1740)),
        ],
        2: [
            ("top_right", (0, 400, 1200, 1740)),
            ("mid_right", (1400, 1800, 1200, 1740)),
            ("bottom_right", (3000, 3480, 1200, 1740)),
        ],
        3: [
            ("top_right", (0, 400, 1200, 1740)),
            ("mid_left", (1200, 1600, 0, 600)),
            ("bottom_left", (2800, 3480, 0, 600)),
        ],
    }

    for panel_idx, crops in CROPS.items():
        for name, box in crops:
            for ps, stride in [(7, 3), (11, 4), (15, 5)]:
                run_crop_test(panel_idx, name, box, patch_size=ps, source_stride=stride)

    print("\nDone.")
