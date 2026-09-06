"""Benchmark runner: synthesise panels, run engines, collect metrics.

This module is the single entry point for the synthetic benchmark. It
is intentionally thin: generate → optionally noise → for each engine,
call via `geoseg.experiments.run_engine` → collect metrics → return
JSON-serialisable summary.

Why call geoseg through the experiments facade and not directly? Per
CODEBASE.md: "产品代码不得直接依赖 segment_engines/_shared、segment_engines/internal
或 router._run_engine; ... 需要按名称运行分割引擎时，从 geoseg.modules.segment_engines.runner
import run_engine". The `geoseg.experiments` facade re-exports that.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .configs import BenchmarkConfig, PanelConfig  # noqa: PLC0415
from .evaluate import evaluate  # noqa: PLC0415
from .generator import generate_panel  # noqa: PLC0415
from .noise import apply_noise  # noqa: PLC0415


def _generate_one(
    panel_cfg: PanelConfig,
    apply_noise_flag: bool,
) -> dict[str, Any]:
    """Render one panel and apply optional noise. Returns artefacts dict."""
    panel_rgb, gt_labels, gen_meta = generate_panel(panel_cfg)
    if not apply_noise_flag:
        return {
            "rgb": panel_rgb,
            "gt_labels": gt_labels,
            "text_mask": None,
            "colorbar_rgb": None,
            "gen_meta": gen_meta,
        }

    # Recover the scalar field used for rendering: regenerate cheaply
    # from the labels. We re-run generate_panel and capture the field
    # in a side-band. For now we approximate value_field from the
    # label_to_value map (filled by the generator).
    label_to_value = gen_meta["label_to_value"]
    value_field = np.zeros(gt_labels.shape, dtype=np.float32)
    for lbl, val in label_to_value.items():
        value_field[gt_labels == lbl] = val

    noisy = apply_noise(
        panel_rgb,
        gt_labels,
        jpeg_std=4.0,
        n_text_blocks=4,
        n_contours=6,
        add_colorbar=True,
        colormap=panel_cfg.colormap,
        value_field=value_field,
        seed=panel_cfg.seed + 1000,
    )
    return {
        "rgb": noisy["rgb"],
        "gt_labels": noisy["gt_labels"],
        "text_mask": noisy["text_mask"],
        "colorbar_rgb": noisy["colorbar_rgb"],
        "gen_meta": gen_meta,
    }


def _run_engine_on_panel(
    engine: str,
    rgb: np.ndarray,
    colorbar_rgb: np.ndarray | None,
    n_layers_hint: int,
) -> dict[str, Any]:
    """Run one engine via geoseg.experiments.run_engine.

    Engines that require reps (VLM seed vectors) are skipped here —
    the synthetic benchmark is meant to evaluate the pure pixel-side
    machinery; semantic seed quality is a separate question.

    If `colorbar_rgb` is given, we crop the last `colorbar_rgb.shape[1]`
    columns off `rgb` before passing it as `panel_rgb`. Per the engine
    contract (`v4.colorbar_guided.segment_colorbar_guided` samples
    seeds from `colorbar_rgb` then labels the whole `panel_rgb` in LAB
    space), the panel must NOT contain the appended colorbar — that
    would double-stamp it and confuse the engine.
    """
    from geoseg.experiments import run_engine

    panel = rgb
    if colorbar_rgb is not None and colorbar_rgb.ndim == 3:
        cb_w = colorbar_rgb.shape[1]
        if cb_w > 0 and panel.shape[1] > cb_w:
            panel = panel[:, :-cb_w, :]

    try:
        result = run_engine(
            engine=engine,
            panel_rgb=panel,
            reps=None,
            colorbar_rgb=colorbar_rgb,
            n_layers=n_layers_hint,
            n_color_zones=0,
        )
        # Engine returns labels matching panel shape; if we cropped,
        # the labels cover only the data area.
        return {"ok": True, "labels": result["labels"]}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def run_benchmark(
    cfg: BenchmarkConfig,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the full benchmark. Return a JSON-serialisable summary.

    If `output_path` is given, also dump the summary to that file.
    """
    summary: dict[str, Any] = {
        "config": {
            "n_panels": cfg.n_panels,
            "engines": list(cfg.engines),
            "apply_realistic_noise": cfg.apply_realistic_noise,
        },
        "per_panel": [],
        "aggregated": {},
    }

    for p_idx, panel_cfg in enumerate(cfg.panels):
        artefact = _generate_one(panel_cfg, cfg.apply_realistic_noise)
        rgb = artefact["rgb"]
        gt = artefact["gt_labels"]
        text_mask = artefact["text_mask"]
        colorbar = artefact["colorbar_rgb"]
        gen_meta = artefact["gen_meta"]

        panel_record: dict[str, Any] = {
            "panel_idx": p_idx,
            "seed": panel_cfg.seed,
            "config": {
                "n_layers": panel_cfg.n_layers,
                "colormap": panel_cfg.colormap,
                "has_fault": panel_cfg.has_fault,
                "has_channel": panel_cfg.has_channel,
                "has_unconformity": panel_cfg.has_unconformity,
                "dip_deg": panel_cfg.dip_deg,
                "warp_amplitude": panel_cfg.warp_amplitude,
            },
            "gen_meta": gen_meta,
            "engines": {},
        }

        # Build colorbar mask (rightmost 24 cols when present) so the
        # evaluator excludes the appended colorbar from mIoU/PA/BF1
        # scoring — the colorbar is not part of the segmentation target.
        colorbar_mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=bool)
        if colorbar is not None:
            colorbar_mask[:, -colorbar.shape[1]:] = True

        # n_layers_hint: use the actual ground-truth label count from
        # the generator meta (label_to_value maps every non-zero label
        # to its scalar field value). This avoids kmeans "empty cluster"
        # warnings from over-estimating the layer count and prevents
        # under-segmentation from under-estimating.
        n_layers_hint = max(1, len(gen_meta.get("label_to_value", {})))

        for engine in cfg.engines:
            run_result = _run_engine_on_panel(engine, rgb, colorbar, n_layers_hint)
            if not run_result["ok"]:
                panel_record["engines"][engine] = {
                    "ok": False,
                    "error": run_result["error"],
                }
                continue
            # Pad pred labels to full panel width with 0 (colorbar
            # area stays "background" — it's already masked out from
            # evaluation by `colorbar_mask`).
            pred_labels = run_result["labels"]
            if pred_labels.shape != gt.shape:
                pad_w = gt.shape[1] - pred_labels.shape[1]
                if pad_w > 0:
                    pred_labels = np.pad(
                        pred_labels, ((0, 0), (0, pad_w)),
                        mode="constant", constant_values=0,
                    )
            metrics = evaluate(
                pred=pred_labels,
                gt=gt,
                panel_rgb=rgb,
                text_mask=text_mask,
                colorbar_mask=colorbar_mask,
            )
            panel_record["engines"][engine] = {"ok": True, "metrics": metrics}

        summary["per_panel"].append(panel_record)

    # Aggregate over panels per engine
    summary["aggregated"] = _aggregate(summary["per_panel"])

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=_jsonable)
    return summary


def _jsonable(obj: Any) -> Any:
    """Convert numpy types to plain Python so json.dump doesn't choke."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _aggregate(per_panel: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Mean / median / min / max for each metric across all panels."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for record in per_panel:
        for engine, payload in record["engines"].items():
            if not payload.get("ok"):
                continue
            metrics = payload["metrics"]
            out.setdefault(engine, {})
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    out[engine].setdefault(key, []).append(float(value))

    # Reduce
    aggregated: dict[str, dict[str, float]] = {}
    for engine, kvs in out.items():
        aggregated[engine] = {}
        for key, values in kvs.items():
            arr = np.array(values, dtype=np.float64)
            aggregated[engine][f"{key}_mean"] = round(float(arr.mean()), 4)
            aggregated[engine][f"{key}_median"] = round(float(np.median(arr)), 4)
            aggregated[engine][f"{key}_min"] = round(float(arr.min()), 4)
            aggregated[engine][f"{key}_max"] = round(float(arr.max()), 4)
    return aggregated


def default_benchmark(seed_base: int = 1000) -> BenchmarkConfig:
    """A small, opinionated default: 12 panels covering common variants.

    Mix of:
      - horizontal layers (easy)
      - tilted layers
      - layers + fault
      - layers + channel
      - layers + unconformity
      - everything combined (hard)
    Across jet and viridis colormaps.

    Used as a default smoke test; not as the full benchmark.
    """
    panels: list[PanelConfig] = []
    # When both channel AND unconformity are on, place the channel
    # BELOW the unconformity cut (channel_y_frac > 0.5) so it isn't
    # overwritten by the unconformity band (BUG4 in audit).
    cases = [
        # (n_layers, dip, has_fault, has_channel, has_unconformity, colormap, seed)
        (4, 0.0, False, False, False, "jet", 11),
        (6, 0.0, False, False, False, "viridis", 12),
        (5, 15.0, False, False, False, "jet", 13),
        (5, 0.0, True, False, False, "jet", 14),
        (5, 0.0, False, True, False, "viridis", 15),
        (5, 0.0, False, False, True, "gray", 16),
        (5, 10.0, True, True, False, "plasma", 17),
        (6, 5.0, True, False, True, "jet", 18),
        (4, 0.0, False, True, True, "viridis", 19),
        (7, 8.0, True, True, True, "plasma", 20),
        (3, 0.0, False, False, False, "gray", 21),  # tiny panel
        (8, 12.0, True, True, True, "jet", 22),    # busy
    ]
    for n_layers, dip, fault, ch, unc, cmap, sd in cases:
        ch_y = 0.45
        if ch and unc:
            ch_y = 0.7  # below unconformity cut (0.5) so the
                        # channel survives the unconformity overlay
        panels.append(
            PanelConfig(
                height=384,
                width=512,
                n_layers=n_layers,
                dip_deg=dip,
                warp_amplitude=8.0,
                has_fault=fault,
                has_channel=ch,
                channel_y_frac=ch_y,
                has_unconformity=unc,
                colormap=cmap,
                seed=sd,
            )
        )
    return BenchmarkConfig(
        panels=tuple(panels),
        engines=(
            # Default: only engines that DON'T require VLM seed vectors.
            # kmeans_full / edge_guided / edge_grow all need `reps` and
            # would silently fall back to v4_kmeans without it, which
            # would mask engine differences in the benchmark.
            "v4_kmeans",
            "v4_kmeans_colorbar",
            "v4_kmeans_pastel",
            "grayscale_agglomerative",
            "horizon_refinement",
        ),
        apply_realistic_noise=True,
    )
