"""Command-line interface for geoseg batch processing."""

from __future__ import annotations

import argparse

from geoseg.batch.service import export_reviewed, process_directory


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch process figures through geoseg pipeline")
    parser.add_argument("--images_dir", help="Directory containing figure images")
    parser.add_argument("--output_dir", help="Directory to save results")
    parser.add_argument("--session", help="Path to existing session JSON (for resume or export)")
    parser.add_argument("--export_only", action="store_true", help="Export all REVIEWED figures in session")
    parser.add_argument("--n_layers", type=int, default=5)
    parser.add_argument("--quality", type=str, default="balanced", choices=["fast", "balanced", "best"])
    parser.add_argument("--no_vlm", action="store_true", help="Skip VLM calls")
    parser.add_argument("--no_resume", action="store_true", help="Re-process all images")
    parser.add_argument("--properties_json", type=str, default=None, help="Custom property table JSON")
    parser.add_argument(
        "--skip_non_velocity",
        action="store_true",
        default=True,
        help="Skip observational_data and other figure types",
    )
    parser.add_argument(
        "--no_skip_non_velocity",
        action="store_true",
        default=False,
        help="Process all figure types",
    )
    args = parser.parse_args()

    if args.export_only:
        if not args.session:
            parser.error("--session is required with --export_only")
        export_reviewed(args.session, args.output_dir)
        return 0

    if not args.images_dir or not args.output_dir:
        parser.error("--images_dir and --output_dir are required")

    properties_map = None
    if args.properties_json:
        from geoseg.modules.post_process.properties import load_properties_json

        properties_map = load_properties_json(args.properties_json)

    process_directory(
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        n_layers=args.n_layers,
        quality_preference=args.quality,
        use_vlm=not args.no_vlm,
        properties_map=properties_map,
        resume=not args.no_resume,
        skip_non_velocity_model=not args.no_skip_non_velocity,
    )
    return 0


__all__ = ["main"]
