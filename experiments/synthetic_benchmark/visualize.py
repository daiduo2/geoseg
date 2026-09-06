"""Visualize a benchmark run: per-panel strip + grid montage.

Outputs to `experiments/synthetic_benchmark/visualizations/`:
- per_panel/p<i>_<engine>.png — one image per (panel, engine): left half
  is the noisy input (or clean if noise disabled), right half is the
  engine's segmentation overlay with seeds marked.
- panel_inputs/p<i>_input.png — the input that all engines saw, with
  ground-truth label boundaries drawn for reference.
- overview_grid.png — a single PNG with all (panel × engine) cells in a
  grid for at-a-glance comparison.

This module is intentionally plot-light: it uses numpy + Pillow only
(already required by the rest of the benchmark). No matplotlib.

Why this lives in the benchmark package: visual inspection is the
intended follow-up to `run_benchmark`. The grid PNG is the primary
artefact an agent reads to compare engine behaviour across panels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Pillow is in geoseg's dependency set (overlay.py uses it). If not
# available we fall back gracefully and report the failure.
try:
    from PIL import Image, ImageDraw, ImageFont
    _HAVE_PIL = True
except ImportError:  # pragma: no cover
    _HAVE_PIL = False


def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr


def _label_to_overlay(
    labels: np.ndarray,
    palette: np.ndarray | None = None,
) -> np.ndarray:
    """Convert (H, W) int labels to (H, W, 3) uint8 RGB overlay."""
    h, w = labels.shape
    if palette is None:
        # Tableau-10-ish fixed palette: spread distinct hues.
        palette = np.array(
            [
                (255, 99, 71), (60, 180, 75), (255, 225, 25),
                (0, 130, 200), (245, 130, 48), (145, 30, 180),
                (70, 240, 240), (240, 50, 230), (210, 245, 60),
                (250, 190, 190), (0, 128, 128), (230, 190, 255),
                (170, 110, 40), (128, 0, 0), (170, 255, 195),
                (128, 128, 0), (255, 250, 200), (50, 50, 50),
            ],
            dtype=np.uint8,
        )
    out = np.zeros((h, w, 3), dtype=np.uint8)
    unique = sorted(set(labels.flatten().tolist()))
    for i, lbl in enumerate(unique):
        if lbl == 0:
            continue
        out[labels == lbl] = palette[i % len(palette)]
    return out


def _blend(img: np.ndarray, overlay: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return (img.astype(np.float32) * (1 - alpha)
            + overlay.astype(np.float32) * alpha).astype(np.uint8)


def _hstack_with_caption(
    left: np.ndarray, right: np.ndarray, caption: str,
    caption_height: int = 24, scale: float = 0.5,
) -> np.ndarray | None:
    """Horizontally stack two images, prepend a caption row.

    Returns a single (H', W', 3) uint8 array ready to save.
    Returns None if Pillow isn't available.
    """
    if not _HAVE_PIL:
        return None

    if scale != 1.0:
        new_w_l = int(left.shape[1] * scale)
        new_h_l = int(left.shape[0] * scale)
        left_img = Image.fromarray(left).resize(
            (new_w_l, new_h_l), Image.NEAREST if scale > 1 else Image.BILINEAR,
        )
        new_w_r = int(right.shape[1] * scale)
        new_h_r = int(right.shape[0] * scale)
        right_img = Image.fromarray(right).resize(
            (new_w_r, new_h_r), Image.NEAREST if scale > 1 else Image.BILINEAR,
        )
    else:
        left_img = Image.fromarray(left)
        right_img = Image.fromarray(right)

    w = left_img.width + right_img.width
    h = max(left_img.height, right_img.height) + caption_height

    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((4, 4), caption, fill=(0, 0, 0), font=font)
    canvas.paste(left_img, (0, caption_height))
    canvas.paste(right_img, (left_img.width, caption_height))
    return np.asarray(canvas)


def _gt_boundary_mask(gt: np.ndarray, thickness: int = 2) -> np.ndarray:
    """White pixel mask of GT label boundaries on a black (H,W,3) canvas.

    Vertical boundaries: at column x where gt[:, x] != gt[:, x-1].
    Horizontal boundaries: at row y where gt[y, :] != gt[y-1, :].
    Each is rendered as a 1-pixel line, then dilated by `thickness`.
    """
    from scipy.ndimage import binary_dilation

    h, w = gt.shape
    bnd_2d = np.zeros((h, w), dtype=bool)
    if w >= 2:
        # vertical line at x=1..w-1 where adjacent columns differ
        bnd_2d[:, 1:] |= (gt[:, 1:] != gt[:, :-1])
        bnd_2d[:, :-1] |= (gt[:, 1:] != gt[:, :-1])
    if h >= 2:
        bnd_2d[1:, :] |= (gt[1:, :] != gt[:-1, :])
        bnd_2d[:-1, :] |= (gt[1:, :] != gt[:-1, :])
    if thickness > 1:
        bnd_2d = binary_dilation(
            bnd_2d, iterations=thickness - 1,
            structure=np.ones((3, 3), dtype=bool),
        )
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[bnd_2d] = (255, 255, 255)
    return out


def render_per_engine(
    panel_rgb: np.ndarray,
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    panel_idx: int,
    engine: str,
    seed_markers: list[dict] | None = None,
    out_dir: str | Path = "visualizations/per_engine",
    scale: float = 0.5,
) -> str | None:
    """Render the (panel, engine) pair as a single PNG."""
    if not _HAVE_PIL:
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Crop colorbar out of input for visualization
    cb_w = 0
    if panel_rgb.shape[1] > gt_labels.shape[1]:
        cb_w = panel_rgb.shape[1] - gt_labels.shape[1]
    panel_only = panel_rgb[:, : gt_labels.shape[1]] if cb_w else panel_rgb

    # Pad pred_labels if needed
    if pred_labels.shape[1] != panel_only.shape[1]:
        pad_w = panel_only.shape[1] - pred_labels.shape[1]
        if pad_w > 0:
            pred_labels = np.pad(
                pred_labels, ((0, 0), (0, pad_w)),
                mode="constant", constant_values=0,
            )

    overlay = _label_to_overlay(pred_labels)
    blended = _blend(panel_only, overlay, alpha=0.45)

    # If VLM seeds were used, mark them on the blended image
    if seed_markers:
        # The blended is uint8 HxWx3 — write directly
        for m in seed_markers:
            rp = m.get("representative_point", {})
            x, y = int(rp.get("x", -1)), int(rp.get("y", -1))
            if 0 <= x < blended.shape[1] and 0 <= y < blended.shape[0]:
                cv = m.get("color_name", "?")
                # Cross marker, white border + black interior
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    xx, yy = x + dx, y + dy
                    if 0 <= xx < blended.shape[1] and 0 <= yy < blended.shape[0]:
                        blended[yy, xx] = (255, 255, 255)
                blended[y, x] = (0, 0, 0)

    gt_overlay = _blend(panel_only, _label_to_overlay(gt_labels), alpha=0.35)
    # overlay GT boundaries in white
    gt_bnd = _gt_boundary_mask(gt_labels)
    gt_overlay[gt_bnd[..., 0] > 0] = (255, 255, 255)

    # Side-by-side: input+GT-overlay | input+pred-overlay
    composite = _hstack_with_caption(
        gt_overlay, blended,
        caption=f"panel {panel_idx} | engine={engine} | left=GT, right=pred",
        scale=scale,
    )
    if composite is None:
        return None
    path = out_dir / f"p{panel_idx:02d}_{engine}.png"
    Image.fromarray(composite).save(path)
    return str(path)


def render_panel_input(
    panel_rgb: np.ndarray,
    gt_labels: np.ndarray,
    panel_idx: int,
    out_dir: str | Path = "visualizations/panel_inputs",
    scale: float = 0.5,
) -> str | None:
    """Render the panel input + GT boundary overlay."""
    if not _HAVE_PIL:
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cb_w = 0
    if panel_rgb.shape[1] > gt_labels.shape[1]:
        cb_w = panel_rgb.shape[1] - gt_labels.shape[1]
    panel_only = panel_rgb[:, : gt_labels.shape[1]] if cb_w else panel_rgb
    gt_overlay = _blend(panel_only, _label_to_overlay(gt_labels), alpha=0.35)
    gt_bnd = _gt_boundary_mask(gt_labels)
    gt_overlay[gt_bnd[..., 0] > 0] = (255, 255, 255)
    img = Image.fromarray(gt_overlay)
    if scale != 1.0:
        img = img.resize(
            (int(img.width * scale), int(img.height * scale)),
            Image.BILINEAR,
        )
    path = out_dir / f"p{panel_idx:02d}_input.png"
    img.save(path)
    return str(path)


def render_overview_grid(
    panels_data: list[dict],
    engines: list[str],
    out_path: str | Path = "visualizations/overview_grid.png",
    cell_w: int = 256,
    cell_h: int = 192,
    label_w: int = 110,
    header_h: int = 70,
) -> str | None:
    """Build a (n_panels+1) × (n_engines+1) grid PNG.

    Rows: panels (one per row). Columns: engines.
    Top-left cell: blank. Top row: engine names. Left column: panel
    index + colormap. Each cell shows the engine's segmentation
    overlay blended with the panel.
    """
    if not _HAVE_PIL:
        return None
    n_panels = len(panels_data)
    n_engines = len(engines)
    canvas_w = label_w + cell_w * n_engines
    canvas_h = header_h + cell_h * n_panels
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Header row: engine names
    for j, eng in enumerate(engines):
        draw.text(
            (label_w + cell_w * j + 4, 4),
            eng,
            fill=(0, 0, 0),
            font=font,
        )

    # Each panel row
    for i, pdat in enumerate(panels_data):
        # Row label: idx + colormap
        cfg = pdat["config"]
        row_label = (
            f"p{i}\n{cfg['colormap']}\nL={cfg['n_layers']} "
            f"d={cfg['dip_deg']:.0f}\n"
            f"{int(cfg['has_fault'])}{int(cfg['has_channel'])}"
            f"{int(cfg['has_unconformity'])}"
        )
        draw.multiline_text((4, header_h + cell_h * i + 4), row_label,
                            fill=(0, 0, 0), font=font)

        for j, eng in enumerate(engines):
            edat = pdat["engines"].get(eng)
            if not edat or not edat.get("ok"):
                # Empty/error cell
                draw.rectangle(
                    [
                        (label_w + cell_w * j, header_h + cell_h * i),
                        (label_w + cell_w * (j + 1), header_h + cell_h * (i + 1)),
                    ],
                    outline=(200, 200, 200),
                )
                draw.text(
                    (label_w + cell_w * j + 4, header_h + cell_h * i + 4),
                    "ERR",
                    fill=(200, 0, 0),
                    font=font,
                )
                continue
            labels = edat["labels"]
            overlay = _label_to_overlay(labels)
            # Crop panel_rgb same way
            panel_rgb = pdat["panel_rgb"]
            gt = pdat["gt_labels"]
            cb_w = panel_rgb.shape[1] - gt.shape[1] if panel_rgb.shape[1] > gt.shape[1] else 0
            panel_only = panel_rgb[:, : gt.shape[1]] if cb_w else panel_rgb
            if labels.shape[1] != panel_only.shape[1]:
                pad_w = panel_only.shape[1] - labels.shape[1]
                if pad_w > 0:
                    labels = np.pad(labels, ((0, 0), (0, pad_w)),
                                    mode="constant", constant_values=0)
                overlay = _label_to_overlay(labels)
            blended = _blend(panel_only, overlay, alpha=0.45)
            img = Image.fromarray(blended).resize(
                (cell_w, cell_h), Image.BILINEAR,
            )
            canvas.paste(img, (label_w + cell_w * j, header_h + cell_h * i))

            # Add mIoU cell annotation
            miou = edat["metrics"]["mIoU"]
            draw.text(
                (label_w + cell_w * j + 4, header_h + cell_h * i + 4),
                f"{miou:.2f}",
                fill=(0, 0, 0),
                font=font,
            )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return str(out_path)


def visualize_benchmark(
    summary_or_cfg,
    out_dir: str | Path = "visualizations",
    scale: float = 0.5,
    also_per_engine: bool = True,
) -> dict[str, list[str]]:
    """Render all visualizations from a `run_benchmark` summary OR a
    BenchmarkConfig. Returns {"per_engine": [...paths...], "panel_inputs":
    [...], "overview": path or None}.

    When given a BenchmarkConfig, this re-runs `run_benchmark` with
    keep_artifacts=True so the JSON written elsewhere stays compact.
    """
    if not _HAVE_PIL:
        return {"per_engine": [], "panel_inputs": [], "overview": None}

    # Lazy import to avoid circular dep at module load
    from .runner import run_benchmark as _run  # noqa: PLC0415

    if not isinstance(summary_or_cfg, dict):
        # Treat as BenchmarkConfig
        summary = _run(summary_or_cfg, output_path=None, keep_artifacts=True)
    else:
        summary = summary_or_cfg
        # If artefacts are missing, re-run
        if not summary.get("per_panel"):
            return {"per_engine": [], "panel_inputs": [], "overview": None}
        first = summary["per_panel"][0]
        if "panel_rgb" not in first:
            from .configs import BenchmarkConfig as _Bcfg  # noqa: PLC0415
            cfg = _Bcfg(
                panels=tuple(first["config"] for first in summary["per_panel"]),
                engines=tuple(summary.get("config", {}).get("engines", ())),
                apply_realistic_noise=summary.get("config", {}).get(
                    "apply_realistic_noise", True,
                ),
            )
            summary = _run(cfg, output_path=None, keep_artifacts=True)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    engines = summary.get("config", {}).get("engines", [])
    per_engine_paths: list[str] = []
    panel_input_paths: list[str] = []

    panels_data_for_grid = []
    for record in summary.get("per_panel", []):
        p_idx = record["panel_idx"]
        artefact = record.get("panel_rgb")
        if artefact is None:
            artefact = record.get("rgb")
        gt = record.get("gt_labels")
        if gt is None:
            gt = record.get("gt")
        if artefact is None or gt is None:
            continue

        if also_per_engine:
            for eng, edat in record["engines"].items():
                if not edat.get("ok"):
                    continue
                labels = edat.get("labels")
                if labels is None:
                    continue
                path = render_per_engine(
                    artefact, gt, labels, p_idx, eng,
                    out_dir=out_dir / "per_engine", scale=scale,
                )
                if path:
                    per_engine_paths.append(path)

        path = render_panel_input(
            artefact, gt, p_idx,
            out_dir=out_dir / "panel_inputs", scale=scale,
        )
        if path:
            panel_input_paths.append(path)

        gd = {
            "panel_idx": p_idx,
            "config": record["config"],
            "panel_rgb": artefact,
            "gt_labels": gt,
            "engines": record["engines"],
        }
        panels_data_for_grid.append(gd)

    overview_path = None
    if panels_data_for_grid:
        overview_path = render_overview_grid(
            panels_data_for_grid, engines,
            out_path=out_dir / "overview_grid.png",
        )

    return {
        "per_engine": per_engine_paths,
        "panel_inputs": panel_input_paths,
        "overview": overview_path,
    }


__all__ = [
    "render_per_engine",
    "render_panel_input",
    "render_overview_grid",
    "visualize_benchmark",
]
