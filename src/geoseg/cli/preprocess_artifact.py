"""CLI for artifact absorption preprocessing.

All tunable parameters are exposed as CLI flags or a JSON config file.  The
colorbar ROI should normally come from agent visual review, not from auto-
detection heuristics.

Example:
    python -m geoseg.cli.preprocess_artifact \
        --image ~/geoseg/newimage.jpg \
        --output-dir runs/preprocess_artifact \
        --colorbar-roi 500 1495 400 25 \
        --n-layers 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from geoseg.preprocessing.pipeline import (
    ArtifactAbsorptionConfig,
    process_image,
)


def _parse_bbox(s: str) -> tuple[int, int, int, int]:
    """Parse a bbox string like '500,1495,400,25'."""
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be x,y,w,h")
    x, y, w, h = parts
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("bbox width and height must be positive")
    return (x, y, w, h)  # type: ignore[return-value]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Absorb red fault lines and black crosses before segmentation."
    )
    parser.add_argument("--image", required=True, type=Path, help="Input image path")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory")
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON config file; CLI flags override config values",
    )
    parser.add_argument(
        "--colorbar-roi",
        type=_parse_bbox,
        help="Visual-review colorbar bbox as x,y,w,h",
    )
    parser.add_argument(
        "--panel-bboxes",
        type=_parse_bbox,
        nargs="+",
        help="One or more panel bboxes as x,y,w,h",
    )
    parser.add_argument("--n-layers", type=int, default=5, help="Number of layers")
    parser.add_argument(
        "--skip-segmentation",
        action="store_true",
        help="Skip segmentation comparison outputs",
    )
    parser.add_argument(
        "--skip-red",
        action="store_true",
        help="Skip red-line detection",
    )
    parser.add_argument(
        "--skip-crosses",
        action="store_true",
        help="Skip black-cross detection",
    )
    parser.add_argument("--inpaint-radius", type=int, default=7)
    parser.add_argument("--inpaint-dilate-iters", type=int, default=2)
    parser.add_argument("--merge-min-area-frac", type=float, default=0.001)
    parser.add_argument(
        "--merge-max-brightness",
        type=int,
        default=80,
        help="Brightness threshold for merging dark artifact labels",
    )
    parser.add_argument(
        "--no-merge-max-brightness",
        action="store_true",
        help="Disable brightness-based merging",
    )
    parser.add_argument(
        "--red-params",
        type=json.loads,
        help="JSON dict of parameters for detect_red_lines",
    )
    parser.add_argument(
        "--cross-params",
        type=json.loads,
        help="JSON dict of parameters for detect_black_crosses",
    )
    parser.add_argument(
        "--text-params",
        type=json.loads,
        help="JSON dict of parameters for detect_text",
    )
    parser.add_argument(
        "--per-panel",
        action="store_true",
        help="Run segmentation per panel and assemble a full overlay",
    )
    parser.add_argument(
        "--artifact-labels",
        type=json.loads,
        help="JSON dict mapping panel_id -> list of artifact label IDs to merge",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config_data: dict = {}
    if args.config:
        config_data = json.loads(args.config.read_text(encoding="utf-8"))

    # CLI flags override config file values.
    def _get(name: str, default):
        cli_value = getattr(args, name)
        return cli_value if cli_value is not None else config_data.get(name, default)

    merge_max_brightness = (
        None
        if args.no_merge_max_brightness
        else config_data.get("merge_max_brightness", args.merge_max_brightness)
    )

    run_segmentation = config_data.get("run_segmentation", not args.skip_segmentation)
    detect_red = config_data.get("detect_red", not args.skip_red)
    detect_crosses = config_data.get("detect_crosses", not args.skip_crosses)
    per_panel = config_data.get("per_panel", args.per_panel)

    image_path = Path(_get("image", args.image))
    output_dir = Path(_get("output_dir", args.output_dir))
    if not image_path.exists():
        print(f"error: image not found: {image_path}", file=sys.stderr)
        return 1

    config = ArtifactAbsorptionConfig(
        image_path=image_path,
        output_dir=output_dir,
        colorbar_roi=_get("colorbar_roi", args.colorbar_roi),
        panel_bboxes=_get("panel_bboxes", args.panel_bboxes),
        n_layers=_get("n_layers", args.n_layers),
        run_segmentation=run_segmentation,
        detect_red=detect_red,
        detect_crosses=detect_crosses,
        red_params=_get("red_params", args.red_params),
        cross_params=_get("cross_params", args.cross_params),
        text_params=_get("text_params", args.text_params),
        inpaint_radius=_get("inpaint_radius", args.inpaint_radius),
        inpaint_dilate_iters=_get("inpaint_dilate_iters", args.inpaint_dilate_iters),
        merge_min_area_frac=_get("merge_min_area_frac", args.merge_min_area_frac),
        merge_max_brightness=merge_max_brightness,
        per_panel=per_panel,
        artifact_labels=_get("artifact_labels", args.artifact_labels),
    )

    if not image_path.exists():
        print(f"error: image not found: {image_path}", file=sys.stderr)
        return 1

    result = process_image(config)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
