"""CPU-only source preflight before a queued L4 tracking execution."""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from .artifacts import verify_manifest, write_manifest
from .audit import validate_track, write_track_set_trajectory_svg
from .constants import INPUT_LOCK_SCHEMA
from .hashing import read_json, sha256_file, sha256_json, write_json
from .models import Crop
from .overlay import render_track_set_overlays
from .review import apply_human_review
from .vanguard_cloud import (
    MAX_INPUT_BYTES,
    MAX_INPUT_FRAMES,
    CloudTasksDispatcher,
    GcsCanaryStore,
    execution_input_lock_object,
    execution_job_spec_object,
    execution_result_object,
    execution_status_object,
    validate_execution_id,
)
from .video import (
    decode_frames,
    encode_video,
    inspect_video,
    require_binary,
    validate_crop,
)


def _count_and_decode(path: Path) -> int:
    ffprobe = require_binary("ffprobe")
    counted = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if counted.returncode != 0:
        raise RuntimeError(counted.stderr.strip() or "ffprobe frame count failed")
    try:
        frame_count = int(counted.stdout.strip())
    except ValueError as exc:
        raise RuntimeError("ffprobe did not return an exact decoded frame count") from exc
    if frame_count <= 0:
        raise RuntimeError("MP4 contains no decodable video frames")
    if frame_count > int(os.environ.get("GROUNDED_MOTION_MAX_INPUT_FRAMES", MAX_INPUT_FRAMES)):
        raise ValueError(
            f"MP4 has {frame_count} frames; the production limit is {MAX_INPUT_FRAMES}"
        )
    decoded = subprocess.run(
        [
            require_binary("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if decoded.returncode != 0:
        raise RuntimeError(decoded.stderr.strip() or "complete MP4 decode failed")
    return frame_count


def _status(
    store: GcsCanaryStore,
    execution_id: str,
    **updates: Any,
) -> dict[str, Any]:
    payload, _ = store.read_json(execution_status_object(execution_id))
    payload.update(updates, updated_unix=time.time())
    store.write_json(execution_status_object(execution_id), payload)
    return payload


def _run_review(
    store: GcsCanaryStore,
    execution_id: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    task = spec["review_task"]
    submission, _ = store.read_json(task["submission_object"])
    try:
        with tempfile.TemporaryDirectory(prefix=f"grounded-motion-review-{execution_id}-") as tmp:
            root = Path(tmp)
            tracked_path = root / "tracked-track-set.json"
            store.download(
                task["track_set_object"],
                tracked_path,
                generation=int(task["track_set_generation"]),
            )
            if sha256_file(tracked_path) != task["track_set_sha256"]:
                raise ValueError("review base track-set hash changed")
            tracked = read_json(tracked_path)
            applied = apply_human_review(
                tracked,
                submission,
                reviewer=str(submission["reviewer"]),
                reviewed_unix=float(submission["submitted_unix"]),
            )
            reviewed = applied["reviewed_track_set"]
            write_json(root / "review-submission.json", submission)
            write_json(root / "reviewed-track-set.json", reviewed)
            report = {
                "schema": "grounded-motion-reviewed-track-set-audit/v1",
                "pass": True,
                "subjects": [],
            }
            for subject in reviewed["subjects"]:
                subject_name = str(subject["subject_id"])
                subject_report = validate_track(subject, production=True)
                if not subject_report["pass"]:
                    report["pass"] = False
                report["subjects"].append(
                    {"subject_id": subject_name, "report": subject_report}
                )
                write_json(root / f"reviewed-{subject_name}.json", subject)
            if not report["pass"]:
                raise ValueError(f"reviewed track set failed production audit: {report}")
            for event_map in applied["event_maps"]:
                write_json(
                    root / f'event-map-{event_map["subject_id"]}.json', event_map
                )
            write_json(root / "review-report.json", report)
            write_track_set_trajectory_svg(reviewed, root / "reviewed-trajectories.svg")

            source_path = root / "source.mp4"
            store.download(
                spec["source"]["object"],
                source_path,
                generation=int(spec["source"]["generation"]),
            )
            frames_dir = root / "_frames"
            overlay_dir = root / "_reviewed_overlay_frames"
            frames = decode_frames(source_path, frames_dir)
            render_track_set_overlays(frames, reviewed, overlay_dir)
            fps = spec["source"]["metadata"]["fps_rational"]
            encode_video(
                overlay_dir / "frame-%06d.png", fps, root / "reviewed-overlay.mp4"
            )
            encode_video(
                overlay_dir / "frame-%06d.png",
                fps,
                root / "reviewed-overlay-slow.mp4",
                slow_factor=4.0,
            )

            published_names = sorted(
                item.name
                for item in root.iterdir()
                if item.is_file() and item.name not in {"source.mp4", "tracked-track-set.json"}
            )
            write_manifest(root, published_names)
            verification = verify_manifest(root)
            if not verification["pass"]:
                raise RuntimeError(
                    f"review manifest verification failed: {verification['errors']}"
                )
            artifacts = []
            for item in sorted(root.iterdir()):
                if not item.is_file() or item.name in {"source.mp4", "tracked-track-set.json"}:
                    continue
                content_type = {
                    ".json": "application/json",
                    ".svg": "image/svg+xml",
                    ".mp4": "video/mp4",
                }.get(item.suffix.lower(), "application/octet-stream")
                artifact = store.upload(
                    item,
                    f"executions/{execution_id}/review/{item.name}",
                    content_type=content_type,
                )
                artifacts.append(
                    {
                        "lane": "reviewed",
                        "name": item.name,
                        "role": (
                            "reviewed-track"
                            if item.name.startswith("reviewed-") and item.suffix == ".json"
                            else "event-map"
                            if item.name.startswith("event-map-")
                            else "reviewed-overlay"
                            if "overlay" in item.name
                            else "review-evidence"
                        ),
                        **artifact,
                    }
                )

        result, result_generation = store.read_json(execution_result_object(execution_id))
        result["artifacts"] = result.get("artifacts", []) + artifacts
        result.update(
            review_status="reviewed",
            event_lock_status="locked",
            event_locked=True,
            human_accepted=False,
            reviewer=submission["reviewer"],
            reviewed_unix=submission["submitted_unix"],
        )
        evidence_index = {
            "schema": "grounded-motion-evidence-index/v2",
            "execution_id": execution_id,
            "revision": result.get("revision"),
            "image_digest": result.get("image_digest"),
            "input_lock": spec.get("input_lock"),
            "tracking_state": "tracked",
            "review_status": "reviewed",
            "event_lock_status": "locked",
            "artifacts": result["artifacts"],
        }
        evidence_path = Path(tempfile.gettempdir()) / f"{execution_id}-reviewed-evidence-index.json"
        write_json(evidence_path, evidence_index)
        index_artifact = store.upload(
            evidence_path,
            f"executions/{execution_id}/review/evidence-index.json",
            content_type="application/json",
        )
        evidence_path.unlink(missing_ok=True)
        result["artifacts"].append(
            {
                "lane": "reviewed",
                "name": "evidence-index.json",
                "role": "reproducibility-evidence",
                **index_artifact,
            }
        )
        store.write_json(
            execution_result_object(execution_id),
            result,
            if_generation_match=result_generation,
        )
        _status(
            store,
            execution_id,
            state="completed",
            terminal=True,
            phase="evidence-ready",
            stage="evidence-ready",
            tracking_state="tracked",
            review_status="reviewed",
            event_lock_status="locked",
        )
        return result
    except Exception as exc:
        _status(
            store,
            execution_id,
            state="completed",
            terminal=True,
            phase="review-failed",
            stage="review-failed",
            review_status="failed",
            event_lock_status="unlocked",
            review_error=str(exc),
        )
        raise


def run(execution_id: str | None = None) -> dict[str, Any]:
    execution_id = validate_execution_id(
        execution_id or os.environ.get("GROUNDED_MOTION_EXECUTION_ID", "")
    )
    store = GcsCanaryStore()
    spec, spec_generation = store.read_json(execution_job_spec_object(execution_id))
    if spec.get("review_task"):
        return _run_review(store, execution_id, spec)
    source = spec["source"]
    _status(store, execution_id, phase="preflighting", stage="verifying-input-lock")
    try:
        if int(source["size_bytes"]) > int(
            os.environ.get("GROUNDED_MOTION_MAX_INPUT_BYTES", MAX_INPUT_BYTES)
        ):
            raise ValueError(f"source exceeds the {MAX_INPUT_BYTES} byte input limit")
        if Path(str(source.get("file_name", ""))).suffix.lower() != ".mp4":
            raise ValueError("source file name must end in .mp4")
        if str(source.get("mime_type", source.get("content_type", ""))).lower() not in {
            "video/mp4",
            "application/mp4",
            "application/octet-stream",
        }:
            raise ValueError("source MIME type does not describe an MP4")

        with tempfile.TemporaryDirectory(prefix=f"grounded-motion-preflight-{execution_id}-") as tmp:
            path = Path(tmp) / "source.mp4"
            store.download(
                source["object"],
                path,
                generation=int(source["generation"]),
            )
            actual_sha = sha256_file(path)
            if actual_sha != source["sha256"]:
                raise ValueError(
                    "source SHA-256 mismatch before GPU allocation: "
                    f"expected {source['sha256']}, got {actual_sha}"
                )
            metadata = inspect_video(path)
            exact_frame_count = _count_and_decode(path)
            metadata["frame_count"] = exact_frame_count
            crop = validate_crop(
                Crop(
                    x=spec["crop"][0],
                    y=spec["crop"][1],
                    width=spec["crop"][2],
                    height=spec["crop"][3],
                )
                if spec.get("crop") is not None
                else None,
                metadata,
            )

        input_lock = {
            "schema": INPUT_LOCK_SCHEMA,
            "execution_id": execution_id,
            "source": {
                "object": source["object"],
                "generation": int(source["generation"]),
                "sha256": actual_sha,
                "size_bytes": int(source["size_bytes"]),
                "file_name": source["file_name"],
                "mime_type": "video/mp4",
                **metadata,
            },
            "crop": crop.model_dump(),
            "minimum_score": float(spec.get("minimum_score", 0.5)),
            "model_preset": spec["model_preset"],
            "complete_decode_pass": True,
            "preflight_revision": os.environ.get("GROUNDED_MOTION_REVISION", "unknown"),
            "created_unix": time.time(),
        }
        input_lock["canonical_sha256"] = sha256_json(input_lock)
        input_generation = store.write_json(
            execution_input_lock_object(execution_id), input_lock, if_generation_match=0
        )
        spec["input_lock"] = {
            "object": execution_input_lock_object(execution_id),
            "generation": input_generation,
            "canonical_sha256": input_lock["canonical_sha256"],
        }
        spec["source"]["metadata"] = metadata
        spec["crop"] = [crop.x, crop.y, crop.width, crop.height]
        store.write_json(
            execution_job_spec_object(execution_id),
            spec,
            if_generation_match=spec_generation,
        )
        queue = store.enqueue_motion(execution_id, time.time_ns())
        _status(
            store,
            execution_id,
            state="queued",
            phase="queued",
            stage="awaiting-dispatch",
            queue_entry=queue["object"],
            input_lock=input_lock,
        )
        dispatcher = CloudTasksDispatcher.from_env()
        if dispatcher is not None:
            dispatcher.enqueue("/internal/dispatch", {})
        return {
            "execution_id": execution_id,
            "state": "queued",
            "input_lock": input_lock,
            "queue": queue,
        }
    except Exception as exc:
        _status(
            store,
            execution_id,
            state="failed",
            phase="preflight-failed",
            stage="preflight-failed",
            terminal=True,
            error=str(exc),
        )
        raise


def main() -> None:
    run()
