"""Deterministic application service used identically by CLI, STDIO, and HTTP."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import __version__
from .artifacts import export_job, write_manifest
from .audit import compare_tracks, inspect_track, validate_track, write_trajectory_svg
from .backend import PoseBackend
from .constants import RECEIPT_SCHEMA
from .hashing import read_json, sha256_file, sha256_json, write_json
from .mmpose_backend import MMPoseBackend
from .models import (
    CompareRequest,
    ExportRequest,
    InspectRequest,
    TrackRequest,
    ValidateRequest,
    expand_path,
)
from .normalize import normalize_track
from .overlay import render_overlays
from .paths import default_workspace, ensure_within, resolve_output, resolve_source
from .presets import ModelPreset, get_preset
from .video import decode_frames, encode_video, inspect_video, validate_crop

BackendFactory = Callable[[ModelPreset, str, Path], PoseBackend]


def _default_backend_factory(
    preset: ModelPreset,
    device: str,
    model_cache: Path,
) -> PoseBackend:
    return MMPoseBackend(preset, device, model_cache)


class GroundedMotionService:
    def __init__(
        self,
        workspace: Path | None = None,
        backend_factory: BackendFactory = _default_backend_factory,
    ) -> None:
        self.workspace = (workspace or default_workspace()).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.backend_factory = backend_factory

    def inspect_video(self, source_path: str) -> dict[str, Any]:
        source = resolve_source(source_path, self.workspace)
        return {
            "path": str(source),
            "sha256": sha256_file(source),
            **inspect_video(source),
        }

    def track_motion(self, request: TrackRequest) -> dict[str, Any]:
        workspace = (
            expand_path(request.workspace) if request.workspace else self.workspace
        )
        ensure_within(workspace, self.workspace)
        source = resolve_source(request.source_path, workspace)
        preset = get_preset(request.model_preset)
        metadata = inspect_video(source)
        crop = validate_crop(request.crop, metadata)
        source_sha = sha256_file(source)
        job_key = {
            "service_version": __version__,
            "source_sha256": source_sha,
            "source_metadata": metadata,
            "crop": crop.model_dump(),
            "preset": preset.receipt(),
            "device": request.device,
            "minimum_score": request.minimum_score,
        }
        job_id = sha256_json(job_key)[:24]
        jobs_root = workspace / "grounded-motion" / "jobs"
        jobs_root.mkdir(parents=True, exist_ok=True)
        job_dir = jobs_root / job_id
        receipt_path = job_dir / "receipt.json"
        if receipt_path.is_file():
            receipt = read_json(receipt_path)
            if receipt.get("status") in {"completed", "completed-with-findings"}:
                return {
                    "cached": True,
                    "job_id": job_id,
                    "job_path": str(job_dir),
                    "receipt": receipt,
                }

        started = time.time()
        with tempfile.TemporaryDirectory(prefix=f".{job_id}-", dir=jobs_root) as temp_name:
            temp_dir = Path(temp_name)
            frames_dir = temp_dir / "_frames"
            overlay_frames = temp_dir / "_overlay_frames"
            frames = decode_frames(source, frames_dir)
            if len(frames) != metadata["frame_count"]:
                raise RuntimeError(
                    "Decoded frame count does not match source metadata; refusing partial interval"
                )

            model_cache = Path(
                os.environ.get(
                    "GROUNDED_MOTION_MODEL_CACHE",
                    workspace / "grounded-motion" / "models",
                )
            ).expanduser().resolve()
            backend = self.backend_factory(
                preset,
                request.device,
                model_cache,
            )
            raw_frames = backend.infer(frames, metadata["fps"], crop, overlay_frames)
            if len(raw_frames) != len(frames):
                raise RuntimeError(
                    "Backend returned a partial source interval; no job was published"
                )

            raw_payload = {
                "schema": "grounded-motion-raw-predictions/v1",
                "source_sha256": source_sha,
                "backend": backend.receipt,
                "frames": [frame.to_dict() for frame in raw_frames],
            }
            raw_path = temp_dir / "raw-predictions.json"
            write_json(raw_path, raw_payload)
            raw_sha = sha256_file(raw_path)

            source_info = {
                **metadata,
                "sha256": source_sha,
                "path": str(source),
                "crop": crop.model_dump(),
                "orientation": "decoded",
                "resampled": False,
            }
            track = normalize_track(
                track_id=job_id,
                source=source_info,
                backend=dict(backend.receipt),
                raw_frames=raw_frames,
                raw_path="raw-predictions.json",
                raw_sha256=raw_sha,
                minimum_score=request.minimum_score,
            )
            track_path = temp_dir / "pose-track.json"
            write_json(track_path, track)

            report = validate_track(track, production=False)
            report_path = temp_dir / "pose-track-report.json"
            write_json(report_path, report)
            write_trajectory_svg(track, temp_dir / "trajectories.svg")

            render_overlays(frames, track, overlay_frames)
            fps = metadata["fps_rational"]
            encode_video(
                overlay_frames / "frame-%06d.png",
                fps,
                temp_dir / "overlay.mp4",
            )
            encode_video(
                overlay_frames / "frame-%06d.png",
                fps,
                temp_dir / "overlay-slow.mp4",
                slow_factor=4.0,
            )

            status = "completed" if report["pass"] else "completed-with-findings"
            receipt = {
                "schema": RECEIPT_SCHEMA,
                "job_id": job_id,
                "status": status,
                "cached": False,
                "state": "tracked",
                "production_accepted": False,
                "source_sha256": source_sha,
                "backend": backend.receipt,
                "job_key": job_key,
                "started_unix": started,
                "completed_unix": time.time(),
                "duration_seconds": time.time() - started,
                "structural_pass": report["pass"],
                "finding_count": len(report["errors"]) + len(report["warnings"]),
            }
            write_json(temp_dir / "receipt.json", receipt)

            published_names = [
                "raw-predictions.json",
                "pose-track.json",
                "pose-track-report.json",
                "trajectories.svg",
                "overlay.mp4",
                "overlay-slow.mp4",
                "receipt.json",
            ]
            for private_dir in (frames_dir, overlay_frames, temp_dir / "crops"):
                if private_dir.exists():
                    shutil.rmtree(private_dir)
            write_manifest(temp_dir, published_names)

            if job_dir.exists():
                if not request.overwrite_failed:
                    raise FileExistsError(
                        f"Job path exists without a reusable receipt: {job_dir}"
                    )
                shutil.rmtree(job_dir)
            os.replace(temp_dir, job_dir)

        return {
            "cached": False,
            "job_id": job_id,
            "job_path": str(job_dir),
            "state": "tracked",
            "production_accepted": False,
            "receipt": receipt,
            "artifacts": {
                "track": str(job_dir / "pose-track.json"),
                "raw": str(job_dir / "raw-predictions.json"),
                "report": str(job_dir / "pose-track-report.json"),
                "overlay": str(job_dir / "overlay.mp4"),
                "overlay_slow": str(job_dir / "overlay-slow.mp4"),
                "trajectories": str(job_dir / "trajectories.svg"),
                "manifest": str(job_dir / "manifest.json"),
            },
        }

    def validate_track(self, request: ValidateRequest) -> dict[str, Any]:
        track_path = ensure_within(expand_path(request.track_path), self.workspace)
        track = read_json(track_path)
        report = validate_track(track, production=request.production)
        report_path = resolve_output(
            request.report_path,
            self.workspace,
            track_path.with_name(
                "pose-track-production-report.json"
                if request.production
                else "pose-track-report.json"
            ),
        )
        write_json(report_path, report)
        trajectory_path = resolve_output(
            request.trajectory_path,
            self.workspace,
            track_path.with_name("trajectories.svg"),
        )
        write_trajectory_svg(track, trajectory_path)
        return {
            "pass": report["pass"],
            "gate": report["gate"],
            "errors": report["errors"],
            "warnings": report["warnings"],
            "report_path": str(report_path),
            "trajectory_path": str(trajectory_path),
        }

    def inspect_track(self, request: InspectRequest) -> dict[str, Any]:
        track_path = ensure_within(expand_path(request.track_path), self.workspace)
        return inspect_track(read_json(track_path))

    def compare_motion(self, request: CompareRequest) -> dict[str, Any]:
        source_path = ensure_within(expand_path(request.source_track_path), self.workspace)
        candidate_path = ensure_within(
            expand_path(request.candidate_track_path), self.workspace
        )
        source = read_json(source_path)
        candidate = read_json(candidate_path)
        report = compare_tracks(source, candidate)
        report_path = resolve_output(
            request.report_path,
            self.workspace,
            candidate_path.with_name("candidate-motion-report.json"),
        )
        write_json(report_path, report)
        trajectory_path = resolve_output(
            request.trajectory_path,
            self.workspace,
            candidate_path.with_name("candidate-trajectories.svg"),
        )
        write_trajectory_svg(candidate, trajectory_path)
        return {
            "pass": report["mechanical_pass"],
            "judgment_status": report["judgment_status"],
            "mechanical_pass": report["mechanical_pass"],
            "judgment_blockers": report["judgment_blockers"],
            "errors": report["errors"],
            "deviation_count": len(report["deviations"]),
            "diagnostic_deviation_count": report["diagnostic_metrics"][
                "root_relative_mechanics"
            ]["diagnostic_deviation_count"],
            "report_path": str(report_path),
            "trajectory_path": str(trajectory_path),
        }

    def export_artifacts(self, request: ExportRequest) -> dict[str, Any]:
        job_path = ensure_within(expand_path(request.job_path), self.workspace)
        destination = resolve_output(
            request.destination_path,
            self.workspace,
            job_path.with_suffix(".zip"),
        )
        exported = export_job(job_path, destination)
        return {
            "path": str(exported),
            "sha256": sha256_file(exported),
            "size_bytes": exported.stat().st_size,
        }
