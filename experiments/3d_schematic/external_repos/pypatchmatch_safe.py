"""Memory-safe pure-Python PatchMatch.
Processes hole pixels in small batches; vectorized within each batch.
"""
import numpy as np
import cv2
from pathlib import Path
import time


def patchmatch_safe(image: np.ndarray, mask: np.ndarray,
                    patch_size: int = 7, source_stride: int = 5,
                    hole_batch: int = 64) -> np.ndarray:
    """
    Memory-safe PatchMatch approximation.

    Args:
        image: (H, W, 3) uint8
        mask:  (H, W) uint8, 255 = hole
        patch_size: odd int
        source_stride: sample every N-th known pixel as source candidate
        hole_batch: number of hole pixels processed together (controls memory)

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

    # Pad
    img_pad = np.pad(img, ((half, half), (half, half), (0, 0)), mode='reflect')
    known_pad = np.pad(known, ((half, half), (half, half)), mode='constant', constant_values=False)

    # Valid source centers (must have full patch inside image)
    valid_source = known.copy()
    valid_source[:half, :] = False
    valid_source[-half:, :] = False
    valid_source[:, :half] = False
    valid_source[:, -half:] = False

    sy, sx = np.where(valid_source)
    sample_idx = np.arange(0, len(sy), source_stride)
    sy, sx = sy[sample_idx], sx[sample_idx]
    n_src = len(sy)

    hy, hx = np.where(hole)
    n_hole = len(hy)

    if n_src == 0 or n_hole == 0:
        return image.copy()

    # Pre-extract source patches and validity masks
    src_patches = np.stack([img_pad[y:y + patch_size, x:x + patch_size] for y, x in zip(sy, sx)])
    src_valid = np.stack([known_pad[y:y + patch_size, x:x + patch_size] for y, x in zip(sy, sx)])

    # For each hole pixel, find best source
    best_src = np.zeros((n_hole, 2), dtype=np.int32)
    best_dist = np.full(n_hole, np.inf, dtype=np.float32)

    src_batch = 256  # also process sources in batches to cap memory

    for h_start in range(0, n_hole, hole_batch):
        h_end = min(n_hole, h_start + hole_batch)
        h_y = hy[h_start:h_end]
        h_x = hx[h_start:h_end]
        b_h = h_end - h_start

        # Extract hole patches for this batch
        hp = np.stack([img_pad[y:y + patch_size, x:x + patch_size] for y, x in zip(h_y, h_x)])
        hv = np.stack([known_pad[y:y + patch_size, x:x + patch_size] for y, x in zip(h_y, h_x)])

        # (b_h, 1, ps, ps, 3)
        hp = hp[:, None, ...]
        hv = hv[:, None, ...]

        local_best_dist = np.full(b_h, np.inf, dtype=np.float32)
        local_best_idx = np.zeros(b_h, dtype=np.int32)

        for s_start in range(0, n_src, src_batch):
            s_end = min(n_src, s_start + src_batch)
            b_s = s_end - s_start

            sp = src_patches[s_start:s_end][None, ...]   # (1, b_s, ps, ps, 3)
            sv = src_valid[s_start:s_end][None, ...]     # (1, b_s, ps, ps)

            valid = hv & sv                              # (b_h, b_s, ps, ps)
            diff = hp - sp                               # (b_h, b_s, ps, ps, 3)
            sqdiff = np.sum(diff ** 2, axis=-1)          # (b_h, b_s, ps, ps)

            valid_count = np.sum(valid, axis=(2, 3))     # (b_h, b_s)
            dist = np.full((b_h, b_s), np.inf, dtype=np.float32)
            ok = valid_count > 0
            # Safe sum avoiding overflow of boolean indexing
            tmp = sqdiff.copy()
            tmp[~valid] = 0
            s = np.sum(tmp, axis=(2, 3))
            dist[ok] = s[ok] / valid_count[ok]

            batch_min = np.argmin(dist, axis=1)
            batch_min_dist = dist[np.arange(b_h), batch_min]
            improved = batch_min_dist < local_best_dist
            local_best_dist[improved] = batch_min_dist[improved]
            local_best_idx[improved] = s_start + batch_min[improved]

        best_src[h_start:h_end, 0] = sy[local_best_idx]
        best_src[h_start:h_end, 1] = sx[local_best_idx]
        best_dist[h_start:h_end] = local_best_dist

    # Vote reconstruction
    vote_sum = np.zeros_like(img)
    vote_cnt = np.zeros_like(img)

    for hi, (cy, cx) in enumerate(zip(hy, hx)):
        sy_i, sx_i = best_src[hi]
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


def run_test(panel_idx: int, crop_name: str, box: tuple,
             patch_size: int, source_stride: int):
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

    t0 = time.time()
    telea = cv2.inpaint(crop_img, crop_mask, 3, cv2.INPAINT_TELEA)
    t_telea = time.time() - t0

    t0 = time.time()
    ns = cv2.inpaint(crop_img, crop_mask, 3, cv2.INPAINT_NS)
    t_ns = time.time() - t0

    t0 = time.time()
    pm = patchmatch_safe(crop_img, crop_mask, patch_size=patch_size, source_stride=source_stride, hole_batch=64)
    t_pm = time.time() - t0

    print(f"  Telea {t_telea:.2f}s | NS {t_ns:.2f}s | PM {t_pm:.2f}s")

    m3 = np.stack([crop_mask] * 3, axis=-1)
    comp = np.hstack([crop_img, m3.astype(np.uint8), telea, ns, pm])
    fname = OUT / f"{stem}_{crop_name}_ps{patch_size}_st{source_stride}.png"
    cv2.imwrite(str(fname), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
    print(f"  Saved: {fname}")
    return pm, telea, ns


if __name__ == "__main__":
    # Smaller focused crops to keep runtime reasonable
    CROPS = {
        1: [
            ("top_left", (50, 350, 50, 500)),
            ("mid_left", (1300, 1600, 50, 500)),
            ("bottom_right", (3100, 3400, 1300, 1700)),
        ],
        2: [
            ("top_right", (50, 350, 1300, 1700)),
            ("mid_right", (1400, 1700, 1300, 1700)),
            ("bottom_right", (3100, 3400, 1300, 1700)),
        ],
        3: [
            ("top_right", (50, 350, 1300, 1700)),
            ("mid_left", (1300, 1600, 50, 500)),
            ("bottom_left", (3000, 3400, 50, 500)),
        ],
    }

    for panel_idx, crops in CROPS.items():
        for name, box in crops:
            for ps, stride in [(7, 5), (11, 6), (15, 8)]:
                run_test(panel_idx, name, box, patch_size=ps, source_stride=stride)

    print("\nAll experiments complete.")
