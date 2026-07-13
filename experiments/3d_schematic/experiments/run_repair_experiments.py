#!/usr/bin/env python3
"""
文字残留修复算法对比实验 —— 完整执行脚本。
按 experiment_plan_repair/plan.md 设计，分 4 组实验运行。

资源控制：Mac mini M4 16GB，max_workers=2（ProcessPoolExecutor）。
跳过：LaMa（simple-lama 未安装）、Guided Filter / DTF（cv2.ximgproc 不可用）。
"""
from __future__ import annotations

import sys
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path("/Users/daiduo2/geoseg/src/3d_schematic")
PANEL_DIR = BASE / "figures" / "panels"
FINAL_DIR = BASE / "experiments" / "text_removal_v2" / "final_pipeline"
OUT_DIR = BASE / "results" / "experiment_plan_repair"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PANELS = [PANEL_DIR / f"panel_{i}.png" for i in (1, 2, 3)]

sys.path.insert(0, str(BASE / "experiments"))
from mser_v2_framework import (
    remove_text_mser_v2,
    detect_text_mser,
    detect_text_laplacian,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_rgb(img: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def load_mask(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


# ---------------------------------------------------------------------------
# Repair primitives
# ---------------------------------------------------------------------------
def repair_telea(image: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
    return cv2.inpaint(image, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)


def repair_ns(image: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
    return cv2.inpaint(image, mask, inpaintRadius=radius, flags=cv2.INPAINT_NS)


def repair_biharmonic(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    from skimage.restoration import inpaint_biharmonic
    result = inpaint_biharmonic(image, mask.astype(bool), channel_axis=-1)
    return (np.clip(result, 0, 1) * 255).astype(np.uint8)


def repair_median_replace(image: np.ndarray, mask: np.ndarray, ksize: int) -> np.ndarray:
    mask_bool = mask.astype(bool)
    result = image.copy().astype(np.float32)
    for ch in range(3):
        chan = result[:, :, ch]
        med = cv2.medianBlur(chan.astype(np.uint8), ksize).astype(np.float32)
        result[:, :, ch] = np.where(mask_bool, med, chan)
    return result.astype(np.uint8)


def repair_bilateral(image: np.ndarray, mask: np.ndarray | None, d: int, sigma_color: float, sigma_space: float) -> np.ndarray:
    result = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    if mask is not None:
        mask_f = mask.astype(np.float32) / 255.0
        mask_f = cv2.GaussianBlur(mask_f, (5, 5), sigmaX=2)
        mask_3ch = np.stack([mask_f] * 3, axis=-1)
        result = (result * mask_3ch + image * (1 - mask_3ch)).astype(np.uint8)
    return result


def repair_gaussian(image: np.ndarray, mask: np.ndarray | None, ksize: int, sigma: float) -> np.ndarray:
    if ksize % 2 == 0:
        ksize += 1
    result = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma)
    if mask is not None:
        mask_f = mask.astype(np.float32) / 255.0
        mask_f = cv2.GaussianBlur(mask_f, (5, 5), sigmaX=2)
        mask_3ch = np.stack([mask_f] * 3, axis=-1)
        result = (result * mask_3ch + image * (1 - mask_3ch)).astype(np.uint8)
    return result


# ---------------------------------------------------------------------------
# Detect residual primitives
# ---------------------------------------------------------------------------
def detect_brightness_anomaly(image: np.ndarray, blur_ksize: int, threshold: int) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    bg = cv2.medianBlur(gray, blur_ksize)
    diff = cv2.subtract(gray, bg)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    return mask


def detect_dog_anomaly(image: np.ndarray, sigma1: float, sigma2: float, threshold: int) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur1 = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma1)
    blur2 = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma2)
    dog = cv2.absdiff(blur1, blur2)
    _, mask = cv2.threshold(dog, threshold, 255, cv2.THRESH_BINARY)
    return mask


def detect_diff_residual(image: np.ndarray, original: np.ndarray, threshold: int) -> np.ndarray:
    diff = np.abs(image.astype(np.float32) - original.astype(np.float32)).max(axis=2)
    mask = (diff > threshold).astype(np.uint8) * 255
    return mask


def detect_residual_region_growing(repaired: np.ndarray, first_mask: np.ndarray, grow_threshold: int = 20) -> np.ndarray:
    mask_bool = first_mask.astype(bool)
    gray = cv2.cvtColor(repaired, cv2.COLOR_RGB2GRAY)
    mser_mask = detect_text_mser(gray, min_area=5, max_area=3000, max_aspect=30)
    lap_mask = detect_text_laplacian(gray, threshold=10, max_area=3000)
    combined = cv2.bitwise_or(mser_mask, lap_mask)
    seeds = ((combined > 0) & mask_bool)
    if not np.any(seeds):
        return np.zeros_like(first_mask)
    residual_grown = seeds.copy()
    changed = True
    while changed:
        changed = False
        dilated = cv2.dilate(residual_grown.astype(np.uint8) * 255, np.ones((3, 3), np.uint8), iterations=1).astype(bool)
        candidates = dilated & mask_bool & (~residual_grown)
        if np.any(candidates):
            mean_bright = gray[residual_grown].mean()
            new_pixels = candidates & (np.abs(gray.astype(float) - mean_bright) < grow_threshold)
            if np.any(new_pixels):
                residual_grown = residual_grown | new_pixels
                changed = True
    residual_mask = cv2.dilate(residual_grown.astype(np.uint8) * 255, np.ones((3, 3), np.uint8), iterations=1)
    return residual_mask


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------
@dataclass
class ExperimentTask:
    group: str
    name: str
    panel_idx: int
    panel_path: Path
    func: Callable
    kwargs: dict


def run_single_experiment(task: ExperimentTask) -> tuple[str, float]:
    t0 = time.time()
    out_dir = OUT_DIR / task.group
    out_dir.mkdir(parents=True, exist_ok=True)

    image = load_rgb(task.panel_path)
    result = task.func(image=image, **task.kwargs)

    out_name = f"{task.panel_path.stem}_{task.name}.png"
    save_rgb(result, out_dir / out_name)

    elapsed = time.time() - t0
    return f"{task.group}/{task.name}/{task.panel_path.stem}", elapsed


# ---------------------------------------------------------------------------
# Group A: Repair algorithm replacement (fixed mask)
# ---------------------------------------------------------------------------
def build_group_a() -> list[ExperimentTask]:
    tasks = []
    masks = {1: load_mask(FINAL_DIR / "panel_1_mask.png"),
             2: load_mask(FINAL_DIR / "panel_2_mask.png"),
             3: load_mask(FINAL_DIR / "panel_3_mask.png")}

    for idx, panel_path in enumerate(PANELS, 1):
        mask = masks[idx]
        # A1: Telea
        for r in (3, 5, 7, 9):
            tasks.append(ExperimentTask(
                "group_a_replacement", f"telea_r{r}", idx, panel_path,
                lambda image, mask, r: repair_telea(image, mask, r),
                {"mask": mask, "r": r}
            ))
        # A2: NS
        for r in (3, 5, 7, 9):
            tasks.append(ExperimentTask(
                "group_a_replacement", f"ns_r{r}", idx, panel_path,
                lambda image, mask, r: repair_ns(image, mask, r),
                {"mask": mask, "r": r}
            ))
        # A3: Biharmonic
        tasks.append(ExperimentTask(
            "group_a_replacement", "biharmonic", idx, panel_path,
            lambda image, mask: repair_biharmonic(image, mask),
            {"mask": mask}
        ))
        # A4: Median blur
        for k in (21, 41, 51, 71, 91):
            k = k if k % 2 == 1 else k + 1
            tasks.append(ExperimentTask(
                "group_a_replacement", f"median_k{k}", idx, panel_path,
                lambda image, mask, k: repair_median_replace(image, mask, k),
                {"mask": mask, "k": k}
            ))
        # A6: Baseline (current v2 two-pass equivalent on original)
        tasks.append(ExperimentTask(
            "group_a_replacement", "baseline_inpaint3_median71", idx, panel_path,
            lambda image, mask: repair_median_replace(repair_telea(image, mask, 3), mask, 71),
            {"mask": mask}
        ))
    return tasks


# ---------------------------------------------------------------------------
# Group B: Detect + repair on v2 final output
# ---------------------------------------------------------------------------
def build_group_b() -> list[ExperimentTask]:
    tasks = []
    originals = {1: load_rgb(PANELS[0]), 2: load_rgb(PANELS[1]), 3: load_rgb(PANELS[2])}
    finals = {1: load_rgb(FINAL_DIR / "panel_1_final.png"),
              2: load_rgb(FINAL_DIR / "panel_2_final.png"),
              3: load_rgb(FINAL_DIR / "panel_3_final.png")}
    first_masks = {1: load_mask(FINAL_DIR / "panel_1_mask.png"),
                   2: load_mask(FINAL_DIR / "panel_2_mask.png"),
                   3: load_mask(FINAL_DIR / "panel_3_mask.png")}

    for idx in (1, 2, 3):
        final = finals[idx]
        orig = originals[idx]
        first_mask = first_masks[idx]

        # B1: brightness anomaly + Telea (run on final output)
        for thresh in (10, 15, 20):
            for r in (3, 5):
                tasks.append(ExperimentTask(
                    "group_b_detect_repair", f"b1_brightness_t{thresh}_r{r}", idx, PANELS[idx - 1],
                    lambda image, final, orig, thresh, r:
                        repair_telea(final, detect_brightness_anomaly(final, 15, thresh), r),
                    {"final": final, "orig": orig, "thresh": thresh, "r": r}
                ))

        # B2: DoG anomaly + Telea (run on final output)
        for s1, s2 in ((1.0, 2.0), (1.5, 3.0)):
            for thresh in (15, 25):
                tasks.append(ExperimentTask(
                    "group_b_detect_repair", f"b2_dog_{s1}_{s2}_t{thresh}", idx, PANELS[idx - 1],
                    lambda image, final, s1, s2, thresh:
                        repair_telea(final, detect_dog_anomaly(final, s1, s2, thresh), 3),
                    {"final": final, "s1": s1, "s2": s2, "thresh": thresh}
                ))

        # B3: region growing + median (run on final output)
        for gt in (15, 20, 25):
            for k in (51, 71):
                kk = k if k % 2 == 1 else k + 1
                tasks.append(ExperimentTask(
                    "group_b_detect_repair", f"b3_grow_t{gt}_k{k}", idx, PANELS[idx - 1],
                    lambda image, final, first_mask, gt, kk:
                        repair_median_replace(final, detect_residual_region_growing(final, first_mask, gt), kk),
                    {"final": final, "first_mask": first_mask, "gt": gt, "kk": kk}
                ))

        # B4: diff residual + Telea (run on final output)
        for thresh in (10, 20, 30):
            tasks.append(ExperimentTask(
                "group_b_detect_repair", f"b4_diff_t{thresh}", idx, PANELS[idx - 1],
                lambda image, final, orig, thresh:
                    repair_telea(final, detect_diff_residual(final, orig, thresh), 3),
                {"final": final, "orig": orig, "thresh": thresh}
            ))
    return tasks


# ---------------------------------------------------------------------------
# Group C: Post-smoothing on v2 final output
# ---------------------------------------------------------------------------
def build_group_c() -> list[ExperimentTask]:
    tasks = []
    finals = {1: load_rgb(FINAL_DIR / "panel_1_final.png"),
              2: load_rgb(FINAL_DIR / "panel_2_final.png"),
              3: load_rgb(FINAL_DIR / "panel_3_final.png")}

    for idx in (1, 2, 3):
        final = finals[idx]
        # C3: Bilateral Filter (run on final output)
        for d in (5, 9, 15):
            for sc in (30, 75):
                for ss in (30, 75):
                    tasks.append(ExperimentTask(
                        "group_c_post_smooth", f"bilateral_d{d}_sc{sc}_ss{ss}", idx, PANELS[idx - 1],
                        lambda image, final, d, sc, ss: repair_bilateral(final, None, d, sc, ss),
                        {"final": final, "d": d, "sc": sc, "ss": ss}
                    ))
        # C4: Gaussian blur control (run on final output)
        for k in (5, 11, 21):
            for s in (1.0, 2.0):
                tasks.append(ExperimentTask(
                    "group_c_post_smooth", f"gaussian_k{k}_s{s}", idx, PANELS[idx - 1],
                    lambda image, final, k, s: repair_gaussian(final, None, k, s),
                    {"final": final, "k": k, "s": s}
                ))
    return tasks


# ---------------------------------------------------------------------------
# Group D: Full pipeline replacement
# ---------------------------------------------------------------------------
def build_group_d() -> list[ExperimentTask]:
    tasks = []
    for idx, panel_path in enumerate(PANELS, 1):
        # D1: Telea full pipeline (MSER+Laplacian + Telea r=3)
        tasks.append(ExperimentTask(
            "group_d_full_pipeline", "d1_telea_full", idx, panel_path,
            lambda image:
                repair_telea(image,
                             remove_text_mser_v2(image, brightness_thresh=170, dilate_iter=1, inpaint_radius=3)[1],
                             3),
            {}
        ))
        # D2: Biharmonic full pipeline
        tasks.append(ExperimentTask(
            "group_d_full_pipeline", "d2_biharmonic_full", idx, panel_path,
            lambda image:
                repair_biharmonic(image,
                                  remove_text_mser_v2(image, brightness_thresh=170, dilate_iter=1, inpaint_radius=3)[1]),
            {}
        ))
        # D4: Telea + 2nd pass Biharmonic
        tasks.append(ExperimentTask(
            "group_d_full_pipeline", "d4_telea_biharmonic", idx, panel_path,
            lambda image:
                repair_biharmonic(
                    repair_telea(image,
                                 remove_text_mser_v2(image, brightness_thresh=170, dilate_iter=1, inpaint_radius=3)[1],
                                 3),
                    detect_residual_region_growing(
                        repair_telea(image,
                                     remove_text_mser_v2(image, brightness_thresh=170, dilate_iter=1, inpaint_radius=3)[1],
                                     3),
                        remove_text_mser_v2(image, brightness_thresh=170, dilate_iter=1, inpaint_radius=3)[1]
                    )
                ),
            {}
        ))
    return tasks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
GROUP_BUILDERS = {
    "a": build_group_a,
    "b": build_group_b,
    "c": build_group_c,
    "d": build_group_d,
    "all": lambda: build_group_a() + build_group_b() + build_group_c() + build_group_d(),
}


def main():
    parser = argparse.ArgumentParser(description="Run repair algorithm experiments")
    parser.add_argument("group", choices=["a", "b", "c", "d", "all"], help="Experiment group to run")
    parser.add_argument("--workers", type=int, default=2, help="Max parallel workers (default 2)")
    parser.add_argument("--dry-run", action="store_true", help="Print tasks without running")
    args = parser.parse_args()

    print(f"Building experiment tasks for group '{args.group}'...")
    tasks = GROUP_BUILDERS[args.group]()
    print(f"  Total tasks: {len(tasks)}")

    if args.dry_run:
        for t in tasks:
            print(f"  {t.group}/{t.name} — panel_{t.panel_idx}")
        return

    print(f"Running with {args.workers} workers...")
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_experiment, t): t for t in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                name, elapsed = future.result()
                results.append((name, elapsed, None))
                print(f"  ✓ {name:50s} ({elapsed:.2f}s)")
            except Exception as e:
                results.append((f"{task.group}/{task.name}", 0.0, str(e)))
                print(f"  ✗ {task.group}/{task.name}/{task.panel_path.stem} — {e}")

    # Summary
    total = len(results)
    ok = sum(1 for _, _, err in results if err is None)
    failed = total - ok
    total_time = sum(elapsed for _, elapsed, _ in results)
    print(f"\nDone: {ok}/{total} succeeded, {failed} failed, total time {total_time:.1f}s")
    print(f"Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
