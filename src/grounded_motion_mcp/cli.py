"""Command-line interface for the deterministic service."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .models import (
    CompareRequest,
    Crop,
    ExportRequest,
    InspectRequest,
    TrackRequest,
    ValidateRequest,
)
from .service import GroundedMotionService


def parse_crop(value: str) -> Crop:
    try:
        x, y, width, height = (int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("crop must be x,y,width,height") from exc
    return Crop(x=x, y=y, width=width, height=height)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="grounded-motion")
    parser.add_argument(
        "--workspace",
        default=None,
        help="Root allowed for all input and output paths.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_video = subparsers.add_parser("inspect-video")
    inspect_video.add_argument("source_path")

    track = subparsers.add_parser("track")
    track.add_argument("source_path")
    track.add_argument("--crop", type=parse_crop)
    track.add_argument("--device", default="auto")
    track.add_argument("--model-preset", default="rtmw-x-cocktail14-384x288")
    track.add_argument("--minimum-score", type=float, default=0.5)

    validate = subparsers.add_parser("validate")
    validate.add_argument("track_path")
    validate.add_argument("--production", action="store_true")
    validate.add_argument("--report-path")
    validate.add_argument("--trajectory-path")

    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("track_path")

    compare = subparsers.add_parser("compare")
    compare.add_argument("source_track_path")
    compare.add_argument("candidate_track_path")
    compare.add_argument("--report-path")
    compare.add_argument("--trajectory-path")

    export = subparsers.add_parser("export")
    export.add_argument("job_path")
    export.add_argument("--destination-path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = Path(args.workspace).expanduser().resolve() if args.workspace else None
    service = GroundedMotionService(workspace=workspace)

    if args.command == "inspect-video":
        result = service.inspect_video(args.source_path)
    elif args.command == "track":
        result = service.track_motion(
            TrackRequest(
                source_path=args.source_path,
                crop=args.crop,
                device=args.device,
                model_preset=args.model_preset,
                minimum_score=args.minimum_score,
            )
        )
    elif args.command == "validate":
        result = service.validate_track(
            ValidateRequest(
                track_path=args.track_path,
                production=args.production,
                report_path=args.report_path,
                trajectory_path=args.trajectory_path,
            )
        )
    elif args.command == "inspect":
        result = service.inspect_track(InspectRequest(track_path=args.track_path))
    elif args.command == "compare":
        result = service.compare_motion(
            CompareRequest(
                source_track_path=args.source_track_path,
                candidate_track_path=args.candidate_track_path,
                report_path=args.report_path,
                trajectory_path=args.trajectory_path,
            )
        )
    else:
        result = service.export_artifacts(
            ExportRequest(
                job_path=args.job_path,
                destination_path=args.destination_path,
            )
        )

    print(json.dumps(result, indent=2))
    if args.command in {"validate", "compare"} and not result["pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
