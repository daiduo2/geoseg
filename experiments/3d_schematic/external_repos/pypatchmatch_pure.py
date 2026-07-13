"""
Pure Python PatchMatch Inpainting — Simplified Implementation
Based on Barnes et al., SIGGRAPH 2009.

Simplifications for experimental use:
- Single-scale (no image pyramid)
- Fixed patch size
- SSD distance on RGB
- No gradient term
"""
import numpy as np
import cv2
from typing import Optional


def patchmatch_inpaint(image: np.ndarray, mask: np.ndarray,
                       patch_size: int = 11, iterations: int = 5) -> np.ndarray:
    """
    PatchMatch-based inpainting.

    Args:
        image: (H, W, 3) uint8 RGB image
        mask:  (H, W) uint8, 255 = hole (region to inpaint)
        patch_size: odd int, patch diameter
        iterations: number of PatchMatch iterations

    Returns:
        (H, W, 3) uint8 inpainted image
    """
    assert patch_size % 2 == 1, "patch_size must be odd"
    h, w = image.shape[:2]
    half = patch_size // 2

    # Convert to float32 for computation
    img = image.astype(np.float32)
    hole = (mask > 127)
    known = ~hole

    # Pad image with reflection for boundary patches
    img_pad = np.pad(img, ((half, half), (half, half), (0, 0)), mode='reflect')

    # NNF: for each hole pixel, store (y, x) of source pixel
    # Initialize randomly from known region
    nnf = np.zeros((h, w, 2), dtype=np.int32)
    known_y, known_x = np.where(known)
    hole_y, hole_x = np.where(hole)

    if len(known_y) == 0:
        return image.copy()

    for idx in range(len(hole_y)):
        hy, hx = hole_y[idx], hole_x[idx]
        rand_idx = np.random.randint(0, len(known_y))
        nnf[hy, hx] = [known_y[rand_idx], known_x[rand_idx]]

    def get_patch(img_padded, y, x):
        """Extract patch centered at (y, x) in original coords."""
        return img_padded[y:y + patch_size, x:x + patch_size]

    def patch_distance(py, px, qy, qx):
        """SSD between patch at (py, px) and patch at (qy, qx)."""
        p_patch = get_patch(img_pad, py, px)
        q_patch = get_patch(img_pad, qy, qx)
        # Mask out hole pixels in the patch
        # Check which pixels in the patches are known
        p_mask = np.ones((patch_size, patch_size), dtype=bool)
        q_mask = np.ones((patch_size, patch_size), dtype=bool)

        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                yy, xx = py + dy, px + dx
                if 0 <= yy < h and 0 <= xx < w:
                    p_mask[dy + half, dx + half] = known[yy, xx]
                else:
                    p_mask[dy + half, dx + half] = False

                yy, xx = qy + dy, qx + dx
                if 0 <= yy < h and 0 <= xx < w:
                    q_mask[dy + half, dx + half] = known[yy, xx]
                else:
                    q_mask[dy + half, dx + half] = False

        valid = p_mask & q_mask
        if valid.sum() == 0:
            return float('inf')

        diff = p_patch - q_patch
        dist = np.sum(diff[valid] ** 2) / valid.sum()
        return dist

    # Evaluate initial NNF
    dist_field = np.full((h, w), float('inf'))
    for idx in range(len(hole_y)):
        hy, hx = hole_y[idx], hole_x[idx]
        sy, sx = nnf[hy, hx]
        dist_field[hy, hx] = patch_distance(hy, hx, sy, sx)

    # PatchMatch iterations
    for it in range(iterations):
        # Alternating scan direction
        y_range = range(h) if it % 2 == 0 else range(h - 1, -1, -1)
        x_range = range(w) if it % 2 == 0 else range(w - 1, -1, -1)

        for y in y_range:
            for x in x_range:
                if not hole[y, x]:
                    continue

                # Propagation: check neighbors
                # Up/Down neighbor
                for dy in [-1, 1]:
                    ny, nx = y + dy, x
                    if 0 <= ny < h and hole[ny, nx]:
                        sy, sx = nnf[ny, nx]
                        sy2, sx2 = sy - dy, sx
                        if 0 <= sy2 < h and 0 <= sx2 < w and known[sy2, sx2]:
                            d = patch_distance(y, x, sy2, sx2)
                            if d < dist_field[y, x]:
                                dist_field[y, x] = d
                                nnf[y, x] = [sy2, sx2]

                # Left/Right neighbor
                for dx in [-1, 1]:
                    ny, nx = y, x + dx
                    if 0 <= nx < w and hole[ny, nx]:
                        sy, sx = nnf[ny, nx]
                        sy2, sx2 = sy, sx - dx
                        if 0 <= sy2 < h and 0 <= sx2 < w and known[sy2, sx2]:
                            d = patch_distance(y, x, sy2, sx2)
                            if d < dist_field[y, x]:
                                dist_field[y, x] = d
                                nnf[y, x] = [sy2, sx2]

                # Random search
                sy, sx = nnf[y, x]
                radius = max(h, w)
                while radius >= 1:
                    ry = np.random.randint(max(0, sy - radius), min(h, sy + radius + 1))
                    rx = np.random.randint(max(0, sx - radius), min(w, sx + radius + 1))
                    if known[ry, rx]:
                        d = patch_distance(y, x, ry, rx)
                        if d < dist_field[y, x]:
                            dist_field[y, x] = d
                            nnf[y, x] = [ry, rx]
                    radius //= 2

    # Vote: reconstruct hole pixels from NNF
    result = img.copy()
    vote_count = np.zeros((h, w, 3), dtype=np.float32)
    vote_sum = np.zeros((h, w, 3), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            if not hole[y, x]:
                continue
            sy, sx = nnf[y, x]
            # Copy patch from source
            for dy in range(-half, half + 1):
                for dx in range(-half, half + 1):
                    ty, tx = y + dy, x + dx
                    if 0 <= ty < h and 0 <= tx < w:
                        src_y, src_x = sy + dy, sx + dx
                        if 0 <= src_y < h and 0 <= src_x < w:
                            vote_sum[ty, tx] += img[src_y, src_x]
                            vote_count[ty, tx] += 1

    # Average votes
    valid_votes = vote_count > 0
    result[valid_votes] = vote_sum[valid_votes] / vote_count[valid_votes]

    # Fallback: for any remaining holes, copy directly from NNF
    for idx in range(len(hole_y)):
        hy, hx = hole_y[idx], hole_x[idx]
        if vote_count[hy, hx, 0] == 0:
            sy, sx = nnf[hy, hx]
            result[hy, hx] = img[sy, sx]

    return np.clip(result, 0, 255).astype(np.uint8)


def patchmatch_inpaint_fast(image: np.ndarray, mask: np.ndarray,
                            patch_size: int = 11, iterations: int = 3) -> np.ndarray:
    """
    Faster approximation using grid sampling instead of full NNF.
    Good enough for experimental validation.
    """
    assert patch_size % 2 == 1
    h, w = image.shape[:2]
    half = patch_size // 2
    img = image.astype(np.float32)
    hole = (mask > 127)
    known = ~hole

    if hole.sum() == 0:
        return image.copy()

    # Build integral-like known mask for fast valid pixel counting
    result = img.copy()

    # For each hole pixel, find best matching known patch
    known_y, known_x = np.where(known)
    if len(known_y) == 0:
        return image.copy()

    # Sample source candidates (subset of known pixels for speed)
    sample_stride = max(1, int(np.sqrt(len(known_y) / 5000)))
    sample_idx = np.arange(0, len(known_y), sample_stride)
    sample_y = known_y[sample_idx]
    sample_x = known_x[sample_idx]

    hole_y, hole_x = np.where(hole)

    # Pad for patch extraction
    img_pad = np.pad(img, ((half, half), (half, half), (0, 0)), mode='reflect')
    mask_pad = np.pad(known, ((half, half), (half, half)), mode='constant', constant_values=False)

    def get_patch_data(y, x):
        return img_pad[y:y + patch_size, x:x + patch_size], mask_pad[y:y + patch_size, x:x + patch_size]

    # Pre-compute hole patches
    hole_patches = []
    hole_masks = []
    for hy, hx in zip(hole_y, hole_x):
        p, m = get_patch_data(hy, hx)
        hole_patches.append(p)
        hole_masks.append(m)
    hole_patches = np.array(hole_patches)  # (N_hole, ps, ps, 3)
    hole_masks = np.array(hole_masks)  # (N_hole, ps, ps)

    # For each source candidate, compute distance to all hole patches
    best_src = np.zeros((len(hole_y), 2), dtype=np.int32)
    best_dist = np.full(len(hole_y), float('inf'))

    batch_size = 256
    for i in range(0, len(sample_y), batch_size):
        sy_batch = sample_y[i:i + batch_size]
        sx_batch = sample_x[i:i + batch_size]

        src_patches = []
        src_masks = []
        for sy, sx in zip(sy_batch, sx_batch):
            p, m = get_patch_data(sy, sx)
            src_patches.append(p)
            src_masks.append(m)
        src_patches = np.array(src_patches)  # (B, ps, ps, 3)
        src_masks = np.array(src_masks)  # (B, ps, ps)

        # Compute distances: for each hole patch, for each source patch
        for hi in range(len(hole_y)):
            hp = hole_patches[hi]  # (ps, ps, 3)
            hm = hole_masks[hi]  # (ps, ps)

            # Expand for broadcasting
            valid = hm[None, ...] & src_masks  # (B, ps, ps)
            diff = hp[None, ...] - src_patches  # (B, ps, ps, 3)
            sqdiff = np.sum(diff ** 2, axis=-1)  # (B, ps, ps)

            valid_count = np.sum(valid, axis=(1, 2))
            dist = np.full(len(sy_batch), float('inf'))
            valid_mask = valid_count > 0
            dist[valid_mask] = np.sum(sqdiff[valid_mask] * valid[valid_mask], axis=(1, 2)) / valid_count[valid_mask]

            min_idx = np.argmin(dist)
            if dist[min_idx] < best_dist[hi]:
                best_dist[hi] = dist[min_idx]
                best_src[hi] = [sy_batch[min_idx], sx_batch[min_idx]]

    # Vote
    vote_sum = np.zeros((h, w, 3), dtype=np.float32)
    vote_count = np.zeros((h, w, 3), dtype=np.float32)

    for hi, (hy, hx) in enumerate(zip(hole_y, hole_x)):
        sy, sx = best_src[hi]
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                ty, tx = hy + dy, hx + dx
                if 0 <= ty < h and 0 <= tx < w:
                    src_y, src_x = sy + dy, sx + dx
                    if 0 <= src_y < h and 0 <= src_x < w:
                        vote_sum[ty, tx] += img[src_y, src_x]
                        vote_count[ty, tx] += 1

    valid_votes = vote_count > 0
    result[valid_votes] = vote_sum[valid_votes] / vote_count[valid_votes]

    # Fallback
    for hi, (hy, hx) in enumerate(zip(hole_y, hole_x)):
        if vote_count[hy, hx, 0] == 0:
            result[hy, hx] = img[best_src[hi][0], best_src[hi][1]]

    return np.clip(result, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    import time
    from pathlib import Path

    BASE = Path("/Users/daiduo2/geoseg/src/3d_schematic")
    RESULTS = BASE / "results" / "experiment_plan_repair"
    OUT_DIR = RESULTS / "pypatchmatch_test"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load panel and mask
    panel = cv2.imread(str(BASE / "figures/panels/panel_1.png"))
    panel = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)

    # Load v2 mask
    mask = cv2.imread(str(BASE / "experiments/text_removal_v2/final_pipeline/panel_1_mask.png"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print("Mask not found, using median_fill mask")
        mask = cv2.imread(str(BASE / "experiments/text_removal_v2/median_fill/panel_1_mask.png"), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print("No mask found, skipping")
        exit(1)

    print(f"Image: {panel.shape}, Mask coverage: {np.sum(mask > 127) / mask.size * 100:.2f}%")

    # Test fast version
    print("Running patchmatch_inpaint_fast (patch_size=11, iterations=3)...")
    t0 = time.time()
    result = patchmatch_inpaint_fast(panel, mask, patch_size=11, iterations=3)
    print(f"Done in {time.time() - t0:.2f}s")

    cv2.imwrite(str(OUT_DIR / "panel_1_patchmatch_fast.png"), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"Saved: {OUT_DIR / 'panel_1_patchmatch_fast.png'}")

    # Compare with Telea
    print("Running cv2.inpaint Telea r=3 for comparison...")
    t0 = time.time()
    telea = cv2.inpaint(panel, mask, 3, cv2.INPAINT_TELEA)
    print(f"Done in {time.time() - t0:.2f}s")
    cv2.imwrite(str(OUT_DIR / "panel_1_telea_ref.png"), cv2.cvtColor(telea, cv2.COLOR_RGB2BGR))

    # Compare with NS
    print("Running cv2.inpaint NS r=3 for comparison...")
    t0 = time.time()
    ns = cv2.inpaint(panel, mask, 3, cv2.INPAINT_NS)
    print(f"Done in {time.time() - t0:.2f}s")
    cv2.imwrite(str(OUT_DIR / "panel_1_ns_ref.png"), cv2.cvtColor(ns, cv2.COLOR_RGB2BGR))

    print("\nAll results saved to:", OUT_DIR)
