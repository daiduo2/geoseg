"""验证 diff_thresh=15 对 Panel 3 的覆盖率提升."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.segmentation import felzenszwalb


def extract_detail_layer(image, blur_ksize=15, blur_sigma=3.0):
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    blurred = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), sigmaX=blur_sigma)
    diff = np.abs(image.astype(np.float32) - blurred.astype(np.float32))
    return diff.max(axis=2)


def create_overlay_mask(detail, diff_thresh=20.0, expand_radius=15):
    binary = (detail > diff_thresh).astype(np.uint8) * 255
    if expand_radius > 0:
        ksize = expand_radius * 2 + 1
        blurred = cv2.GaussianBlur(binary, (ksize, ksize), sigmaX=expand_radius)
        overlay_mask = blurred > 64
    else:
        overlay_mask = binary > 0
    return overlay_mask


def post_merge(label_img, image, small_ratio=0.015, max_score=0.8, max_color=45.0):
    h, w = label_img.shape
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobel_x**2 + sobel_y**2)
    result = label_img.copy()
    total = h * w

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

    for _ in range(30):
        unique, counts = np.unique(result, return_counts=True)
        mean_colors = {lbl: image[result == lbl].mean(axis=0) for lbl in unique}
        small = unique[counts < total * small_ratio]
        if len(small) == 0:
            break
        for small_lbl in small:
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
                dist = np.linalg.norm(mean_colors[small_lbl] - mean_colors[adj_lbl])
                if dist < min_dist:
                    min_dist = dist
                    nearest = adj_lbl
            if nearest is not None:
                result[mask] = nearest
        renumber()

    for _ in range(30):
        unique = sorted(np.unique(result))
        if len(unique) <= 4:
            break
        mean_colors = {lbl: image[result == lbl].mean(axis=0) for lbl in unique}
        pair_data = {}
        for y in range(h):
            for x in range(w - 1):
                a, b = result[y, x], result[y, x + 1]
                if a != b:
                    pair = (min(a, b), max(a, b))
                    pair_data.setdefault(pair, []).append(gradient[y, x])
            if y < h - 1:
                for x in range(w):
                    a, b = result[y, x], result[y + 1, x]
                    if a != b:
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


def draw_boundaries(image, labels, color=(0, 0, 0), thickness=2):
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


def render_label_fill(labels, overlay_label=-1):
    import colorsys
    unique = sorted(np.unique(labels))
    h, w = labels.shape
    result = np.zeros((h, w, 3), dtype=np.uint8)
    colors = []
    for i, lbl in enumerate(unique):
        if lbl == overlay_label:
            colors.append([128, 128, 128])
        else:
            hue = (i * 0.618033988749895) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.75, 0.92)
            colors.append([int(c * 255) for c in rgb])
    for i, lbl in enumerate(unique):
        mask = labels == lbl
        result[mask] = colors[i]
    return result


def run_pipeline(image, diff_thresh, expand_radius=15):
    detail = extract_detail_layer(image)
    overlay_mask = create_overlay_mask(detail, diff_thresh, expand_radius)

    inpaint_mask = overlay_mask.astype(np.uint8) * 255
    inpainted = cv2.inpaint(image, inpaint_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

    geo_labels = felzenszwalb(inpainted, scale=300, sigma=0.5, min_size=30)
    geo_labels = post_merge(geo_labels, inpainted)

    final_labels = geo_labels.copy()
    final_labels[overlay_mask] = -1

    coverage = overlay_mask.sum() / overlay_mask.size * 100
    n_geo = len(np.unique(geo_labels))
    n_final = len(np.unique(final_labels))

    return {
        "detail": detail,
        "overlay_mask": overlay_mask,
        "geo_labels": geo_labels,
        "final_labels": final_labels,
        "coverage": coverage,
        "n_geo": n_geo,
        "n_final": n_final,
    }


def main():
    base = Path("/Users/daiduo2/geoseg/src/3d_schematic")
    out_dir = base / "diff_overlay_experiments" / "verify_t15"
    out_dir.mkdir(parents=True, exist_ok=True)

    img = np.array(Image.open(base / "panel_3_front.png").convert("RGB"))

    params_list = [
        ("t20_e15", 20.0, 15),
        ("t15_e15", 15.0, 15),
        ("t15_e20", 15.0, 20),
        ("t18_e15", 18.0, 15),
    ]

    results = {}
    for name, thresh, expand in params_list:
        print(f"Running {name}: thresh={thresh}, expand={expand}")
        res = run_pipeline(img, thresh, expand)
        results[name] = res
        print(f"  coverage={res['coverage']:.2f}%, geo_labels={res['n_geo']}, final_labels={res['n_final']}")

    # 生成对比图
    h, w = img.shape[:2]
    n = len(params_list) + 1  # +1 for Original column
    header_h = 30
    cell_h, cell_w = h, w
    canvas = np.ones((5 * cell_h + header_h, n * cell_w, 3), dtype=np.uint8) * 255

    titles = ["Original"] + [f"{name}\ncov={results[name]['coverage']:.1f}%" for name, _, _ in params_list]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for c, title in enumerate(titles):
        x = c * cell_w + 10
        cv2.putText(canvas, title, (x, 20), font, 0.45, (0, 0, 0), 1)

    row_getters = [
        ("Original", None),
        ("Detail", lambda name: (results[name]["detail"] / results[name]["detail"].max() * 255).astype(np.uint8)),
        ("Overlay Mask", lambda name: results[name]["overlay_mask"].astype(np.uint8) * 255),
        ("Label Fill", lambda name: render_label_fill(results[name]["final_labels"], -1)),
        ("Boundaries", lambda name: draw_boundaries(img, results[name]["final_labels"])),
    ]

    for r, (row_name, getter) in enumerate(row_getters):
        y = header_h + r * cell_h
        for c, name in enumerate(["original"] + [n for n, _, _ in params_list]):
            x = c * cell_w
            if row_name == "Original":
                cell = img
            else:
                if name == "original":
                    # Original column: show first param's detail for reference
                    first_name = params_list[0][0]
                    cell = getter(first_name)
                else:
                    cell = getter(name)
                if cell.ndim == 2:
                    cell = cv2.cvtColor(cell, cv2.COLOR_GRAY2RGB)
            canvas[y:y + cell_h, x:x + cell_w] = cell

    out_path = out_dir / "comparison_t15_verify.png"
    Image.fromarray(canvas).save(out_path)
    print(f"Saved: {out_path}")

    # 文字区域特写对比
    # 聚焦左下角文字区域: crop [150:450, 0:250]
    crop_y, crop_x = slice(150, 450), slice(0, 250)
    zoom_h, zoom_w = 300, 250
    zoom_n = len(params_list) + 1
    zoom_canvas = np.ones((5 * zoom_h + header_h, zoom_n * zoom_w, 3), dtype=np.uint8) * 255
    for c, title in enumerate(titles):
        x = c * zoom_w + 5
        cv2.putText(zoom_canvas, title, (x, 20), font, 0.4, (0, 0, 0), 1)

    for r, (row_name, getter) in enumerate(row_getters):
        y = header_h + r * zoom_h
        for c, name in enumerate(["original"] + [n for n, _, _ in params_list]):
            x = c * zoom_w
            if row_name == "Original":
                cell = img[crop_y, crop_x]
            else:
                if name == "original":
                    first_name = params_list[0][0]
                    cell = getter(first_name)[crop_y, crop_x]
                else:
                    cell = getter(name)[crop_y, crop_x]
                if cell.ndim == 2:
                    cell = cv2.cvtColor(cell, cv2.COLOR_GRAY2RGB)
            zoom_canvas[y:y + zoom_h, x:x + zoom_w] = cell

    zoom_path = out_dir / "zoom_text_area.png"
    Image.fromarray(zoom_canvas).save(zoom_path)
    print(f"Saved: {zoom_path}")


if __name__ == "__main__":
    main()
