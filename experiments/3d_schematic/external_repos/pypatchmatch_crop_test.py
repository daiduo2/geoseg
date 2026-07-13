"""PatchMatch 小区域快速验证 —— 只处理关键文字残留区域。"""
import numpy as np
import cv2
from pathlib import Path
import time


def patchmatch_crop(image_crop: np.ndarray, mask_crop: np.ndarray,
                    patch_size: int = 7, source_stride: int = 2) -> np.ndarray:
    """
    简化的 PatchMatch：针对小裁剪区域，枚举所有已知 patch 找最佳匹配。
    """
    assert patch_size % 2 == 1
    h, w = image_crop.shape[:2]
    half = patch_size // 2

    img = image_crop.astype(np.float32)
    hole = (mask_crop > 127)
    known = ~hole

    if hole.sum() == 0:
        return image_crop.copy()

    # Pad
    img_pad = np.pad(img, ((half, half), (half, half), (0, 0)), mode='reflect')
    mask_pad = np.pad(known, ((half, half), (half, half)), mode='constant', constant_values=False)

    hole_y, hole_x = np.where(hole)
    known_y, known_x = np.where(known)

    # Sample source centers with stride
    sample_idx = np.arange(0, len(known_y), source_stride)
    sample_y, sample_x = known_y[sample_idx], known_x[sample_idx]

    # Pre-extract hole patches
    hole_patches = []
    hole_valid = []
    for hy, hx in zip(hole_y, hole_x):
        hole_patches.append(img_pad[hy:hy + patch_size, hx:hx + patch_size])
        hole_valid.append(mask_pad[hy:hy + patch_size, hx:hx + patch_size])
    hole_patches = np.array(hole_patches)
    hole_valid = np.array(hole_valid)

    best_src = np.zeros((len(hole_y), 2), dtype=np.int32)
    best_dist = np.full(len(hole_y), float('inf'))

    batch = 512
    for i in range(0, len(sample_y), batch):
        sy_b = sample_y[i:i + batch]
        sx_b = sample_x[i:i + batch]

        # Extract source patches
        sp = np.stack([img_pad[y:y + patch_size, x:x + patch_size] for y, x in zip(sy_b, sx_b)])
        sv = np.stack([mask_pad[y:y + patch_size, x:x + patch_size] for y, x in zip(sy_b, sx_b)])

        for hi in range(len(hole_y)):
            hp = hole_patches[hi][None, ...]  # (1, ps, ps, 3)
            hv = hole_valid[hi][None, ...]    # (1, ps, ps)
            valid = hv & sv                   # (B, ps, ps)
            diff = hp - sp
            sqdiff = np.sum(diff ** 2, axis=-1)
            vc = np.sum(valid, axis=(1, 2))
            dist = np.full(len(sy_b), np.inf)
            ok = vc > 0
            dist[ok] = np.sum(sqdiff[ok] * valid[ok], axis=(1, 2)) / vc[ok]
            bj = np.argmin(dist)
            if dist[bj] < best_dist[hi]:
                best_dist[hi] = dist[bj]
                best_src[hi] = [sy_b[bj], sx_b[bj]]

    # Vote with overlapping patches
    vote_sum = np.zeros_like(img)
    vote_cnt = np.zeros_like(img)
    for hi, (hy, hx) in enumerate(zip(hole_y, hole_x)):
        sy, sx = best_src[hi]
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                ty, tx = hy + dy, hx + dx
                if 0 <= ty < h and 0 <= tx < w:
                    src_y, src_x = sy + dy, sx + dx
                    if 0 <= src_y < h and 0 <= src_x < w:
                        vote_sum[ty, tx] += img[src_y, src_x]
                        vote_cnt[ty, tx] += 1

    result = img.copy()
    ok = vote_cnt > 0
    result[ok] = vote_sum[ok] / vote_cnt[ok]
    return np.clip(result, 0, 255).astype(np.uint8)


def run_test(panel_idx: int, crop_name: str, box: tuple):
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

    print(f"[{stem} {crop_name}] crop={crop_img.shape}, hole%={np.sum(crop_mask>127)/crop_mask.size*100:.2f}")

    # Baseline Telea
    t0 = time.time()
    telea = cv2.inpaint(crop_img, crop_mask, 3, cv2.INPAINT_TELEA)
    t_telea = time.time() - t0

    # PatchMatch
    t0 = time.time()
    pm = patchmatch_crop(crop_img, crop_mask, patch_size=7, source_stride=2)
    t_pm = time.time() - t0

    print(f"  Telea: {t_telea:.2f}s, PatchMatch: {t_pm:.2f}s")

    # Save comparison
    comp = np.hstack([
        cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR),
        cv2.cvtColor(np.stack([crop_mask]*3, axis=-1).astype(np.uint8), cv2.COLOR_RGB2BGR),
        cv2.cvtColor(telea, cv2.COLOR_RGB2BGR),
        cv2.cvtColor(pm, cv2.COLOR_RGB2BGR),
    ])
    cv2.imwrite(str(OUT / f"{stem}_{crop_name}_compare.png"), comp)

    return pm, telea


if __name__ == "__main__":
    # Focused crops where text residue is visible
    CROPS = {
        1: {"top_left": (0, 600, 0, 900)},
        2: {"mid_right": (1000, 2200, 900, 1740)},
        3: {"bottom_left": (2200, 3480, 0, 900)},
    }

    for panel_idx, crops in CROPS.items():
        for name, box in crops.items():
            run_test(panel_idx, name, box)

    print("\nDone. Results in results/experiment_plan_repair/pypatchmatch_test/")
