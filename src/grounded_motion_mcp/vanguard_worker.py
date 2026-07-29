"""One-task GPU worker for the immutable Vanguard production canary."""

from __future__ import annotations

import json
import mimetypes
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

from .artifacts import verify_manifest, write_manifest
from .audit import validate_track
from .constants import COCO_WHOLEBODY_NAMES, HAND_SUFFIXES
from .hashing import read_json, sha256_file, write_json
from .models import CompareRequest, TrackRequest
from .service import GroundedMotionService
from .vanguard_cloud import (
    CloudTasksDispatcher,
    GcsCanaryStore,
    execution_job_spec_object,
    execution_result_object,
    execution_status_object,
    load_canary_manifest,
    required_env,
    validate_execution_id,
)


def _status(
    store: GcsCanaryStore,
    execution_id: str,
    *,
    state: str,
    stage: str,
    pipeline_pass: bool = False,
    error: str | None = None,
) -> None:
    payload, _ = store.read_json(execution_status_object(execution_id))
    payload.update(
        state=state,
        phase=stage,
        stage=stage,
        terminal=state in {"completed", "failed"},
        pipeline_pass=pipeline_pass,
        updated_unix=time.time(),
    )
    if state == "completed" and payload.get("kind") == "uploaded-track":
        payload["tracking_state"] = "tracked"
    if error:
        payload["error"] = error
    store.write_json(execution_status_object(execution_id), payload)
    store.set_lock_state(execution_id, state)
    if state in {"completed", "failed"} and payload.get("kind") == "uploaded-track":
        dispatcher = CloudTasksDispatcher.from_env()
        if dispatcher is not None:
            try:
                dispatcher.enqueue("/internal/dispatch", {})
            except Exception as exc:  # noqa: BLE001 - status polling is the fallback
                print(f"dispatch wake-up failed: {exc}", file=sys.stderr)


def _verify_cuda() -> dict[str, Any]:
    import torch

    available = bool(torch.cuda.is_available())
    if not available:
        raise RuntimeError("torch.cuda.is_available() is false in the GPU canary worker")
    return {
        "available": available,
        "device_count": int(torch.cuda.device_count()),
        "device_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def _track(
    service: GroundedMotionService,
    path: Path,
) -> dict[str, Any]:
    result = service.track_motion(
        TrackRequest(source_path=str(path), device="cuda:0", minimum_score=0.5)
    )
    job_path = Path(result["job_path"])
    verification = verify_manifest(job_path)
    receipt = read_json(job_path / "receipt.json")
    if not verification["pass"]:
        raise RuntimeError(f"Artifact manifest verification failed: {verification['errors']}")
    if receipt.get("backend", {}).get("name") != "mmpose":
        raise RuntimeError("Canary receipt does not prove the real MMPose backend")
    if receipt.get("backend", {}).get("model_sha256") != required_env(
        "GROUNDED_MOTION_CHECKPOINT_SHA256"
    ):
        raise RuntimeError("Canary receipt checkpoint hash differs from the baked checkpoint")
    if receipt.get("structural_pass") is not True:
        raise RuntimeError("RTMW track failed the existing structural gate at minimum_score=0.5")
    return {
        "service_result": result,
        "job_path": job_path,
        "receipt": receipt,
        "manifest_verification": verification,
    }


def _verify_detector_score_preservation(job_path: Path) -> dict[str, Any]:
    raw = read_json(job_path / "raw-predictions.json")
    raw_frames = raw.get("frames", [])
    track_set_path = job_path / "track-set.json"
    if track_set_path.is_file():
        track_set = read_json(track_set_path)
        normalized_frames = [
            frame
            for subject in track_set.get("subjects", [])
            for frame in subject.get("frames", [])
        ]
        scores: list[float] = []
        for track_frame in normalized_frames:
            frame_index = int(track_frame["index"])
            raw_instance_index = int(track_frame["raw_instance_index"])
            try:
                raw_scores = raw_frames[frame_index]["instances"][raw_instance_index][
                    "keypoint_scores"
                ]
            except (IndexError, KeyError, TypeError) as exc:
                raise RuntimeError(
                    f"Normalized observation does not reference raw evidence: "
                    f"{track_frame.get('observation_id')}"
                ) from exc
            if len(raw_scores) != len(COCO_WHOLEBODY_NAMES):
                raise RuntimeError("Raw score count differs from COCO-WholeBody mapping")
            for name, raw_score in zip(COCO_WHOLEBODY_NAMES, raw_scores):
                landmark = track_frame["landmarks"][name]
                for field in ("score", "detector_score"):
                    if float(landmark[field]) != float(raw_score):
                        raise RuntimeError(
                            f"Detector score changed during normalization: "
                            f"{track_frame['id']} {name} {raw_score} -> {landmark[field]}"
                        )
                scores.append(float(raw_score))
    else:
        track = read_json(job_path / "pose-track.json")
        track_frames = track.get("frames", [])
        if len(raw_frames) != len(track_frames):
            raise RuntimeError("Raw and normalized frame counts differ")
        scores = []
        for raw_frame, track_frame in zip(raw_frames, track_frames):
            instances = raw_frame.get("instances", [])
            if len(instances) != 1:
                raise RuntimeError("Vanguard score verification requires one subject")
            raw_scores = instances[0].get("keypoint_scores", [])
            if len(raw_scores) != len(COCO_WHOLEBODY_NAMES):
                raise RuntimeError("Raw score count differs from COCO-WholeBody mapping")
            for name, raw_score in zip(COCO_WHOLEBODY_NAMES, raw_scores):
                normalized_score = track_frame["landmarks"][name]["score"]
                if float(normalized_score) != float(raw_score):
                    raise RuntimeError(
                        f"Detector score changed during normalization: "
                        f"{track_frame['id']} {name} {raw_score} -> {normalized_score}"
                    )
                scores.append(float(raw_score))
    if not scores:
        raise RuntimeError("No detector scores were published")
    return {
        "pass": True,
        "count": len(scores),
        "minimum": min(scores),
        "maximum": max(scores),
        "above_one_count": sum(score > 1.0 for score in scores),
        "exact_raw_to_track_match": True,
    }


def _apply_vanguard_judgment_context(
    job_path: Path,
    lane: str,
    fixture: dict[str, Any],
) -> dict[str, Any]:
    track_path = job_path / "pose-track.json"
    track = read_json(track_path)
    policy = fixture["judgment"]
    quarantined = [
        f"{side}_hand_{suffix}"
        for side in ("left", "right")
        for suffix in HAND_SUFFIXES
    ]
    judgment = {
        "modality": policy["modality"],
        "review_policy": policy["review_policy"],
        "quarantined_landmarks": quarantined,
        "quarantine_reason": policy["quarantine_reason"],
        "diagnostic_event_alignment": policy["diagnostic_event_alignment"],
    }
    if lane == "candidate":
        judgment["render_registration_to_source"] = policy[
            "render_registration_to_source"
        ]
    track["judgment"] = judgment
    write_json(track_path, track)

    report = validate_track(track, production=False)
    write_json(job_path / "pose-track-report.json", report)
    receipt_path = job_path / "receipt.json"
    receipt = read_json(receipt_path)
    receipt["judgment_context"] = {
        "modality": judgment["modality"],
        "review_policy": judgment["review_policy"],
        "quarantined_landmark_count": len(quarantined),
        "production_accepted": False,
    }
    receipt["structural_pass"] = report["pass"]
    write_json(receipt_path, receipt)
    names = [
        path.name
        for path in job_path.iterdir()
        if path.is_file() and path.name != "manifest.json"
    ]
    write_manifest(job_path, names)
    verification = verify_manifest(job_path)
    if not verification["pass"]:
        raise RuntimeError(
            f"Annotated {lane} manifest verification failed: {verification['errors']}"
        )
    return {
        "track": track,
        "report": report,
        "receipt": receipt,
        "manifest_verification": verification,
    }


def _publish_tree(
    store: GcsCanaryStore,
    execution_id: str,
    lane: str,
    root: Path,
) -> list[dict[str, Any]]:
    artifacts = []
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        content_type, _ = mimetypes.guess_type(path.name)
        artifacts.append(
            {
                "lane": lane,
                "name": path.name,
                **store.upload(
                    path,
                    f"executions/{execution_id}/{lane}/{path.name}",
                    content_type=content_type,
                ),
            }
        )
    return artifacts


def _run_uploaded_track(
    execution_id: str,
    store: GcsCanaryStore,
    spec: dict[str, Any],
) -> dict[str, Any]:
    started = time.time()
    _status(store, execution_id, state="running", stage="verifying-gpu")
    cuda = _verify_cuda()
    with tempfile.TemporaryDirectory(prefix=f"grounded-motion-{execution_id}-") as temporary:
        workspace = Path(temporary)
        inputs = workspace / "inputs"
        inputs.mkdir()
        suffix = Path(str(spec["source"].get("file_name", "source.mp4"))).suffix or ".mp4"
        source_video = inputs / f"source{suffix}"
        _status(store, execution_id, state="running", stage="downloading-source")
        store.download(
            spec["source"]["object"],
            source_video,
            generation=int(spec["source"]["generation"]),
        )
        actual_sha = sha256_file(source_video)
        if actual_sha != spec["source"]["sha256"]:
            raise RuntimeError(f"Uploaded source SHA-256 mismatch: {actual_sha}")

        service = GroundedMotionService(workspace=workspace)
        _status(store, execution_id, state="running", stage="tracking-source")
        request = TrackRequest(
            source_path=str(source_video),
            device="cuda:0",
            minimum_score=float(spec.get("minimum_score", 0.5)),
            model_preset=str(
                spec.get("model_preset", "rtmw-x-cocktail14-multiperson-384x288")
            ),
            crop=(
                {
                    "x": spec["crop"][0],
                    "y": spec["crop"][1],
                    "width": spec["crop"][2],
                    "height": spec["crop"][3],
                }
                if spec.get("crop") is not None
                else None
            ),
        )
        tracked_result = service.track_motion(request)
        job_path = Path(tracked_result["job_path"])
        verification = verify_manifest(job_path)
        receipt = read_json(job_path / "receipt.json")
        if not verification["pass"]:
            raise RuntimeError(
                f"Artifact manifest verification failed: {verification['errors']}"
            )
        if receipt.get("backend", {}).get("name") != "mmpose":
            raise RuntimeError("Uploaded track receipt does not prove the real MMPose backend")
        if receipt.get("backend", {}).get("model_sha256") != required_env(
            "GROUNDED_MOTION_CHECKPOINT_SHA256"
        ):
            raise RuntimeError("Uploaded track checkpoint hash differs from the baked checkpoint")
        detector_hash = receipt.get("backend", {}).get("detector", {}).get("model_sha256")
        expected_detector_hash = required_env("GROUNDED_MOTION_DETECTOR_SHA256")
        if detector_hash != expected_detector_hash:
            raise RuntimeError(
                "Uploaded track detector hash differs from the baked detector checkpoint"
            )
        score_verification = _verify_detector_score_preservation(job_path)

        _status(store, execution_id, state="running", stage="publishing-evidence")
        artifacts = _publish_tree(store, execution_id, "source", job_path)
        input_lock_ref = spec.get("input_lock") or {}
        input_lock_object = input_lock_ref.get("object")
        if not input_lock_object:
            raise RuntimeError("Uploaded tracking job is missing its immutable input lock")
        input_lock_path = workspace / "input-lock.json"
        store.download(
            input_lock_object,
            input_lock_path,
            generation=int(input_lock_ref["generation"]),
        )
        input_lock_info = store.blob_info(input_lock_object)
        artifacts.append(
            {
                "lane": "control",
                "name": "input-lock.json",
                "role": "reproducibility-input-lock",
                **input_lock_info,
                "sha256": sha256_file(input_lock_path),
            }
        )
        evidence_index = {
            "schema": "grounded-motion-evidence-index/v2",
            "execution_id": execution_id,
            "revision": os.environ.get("GROUNDED_MOTION_REVISION", "unknown"),
            "image_digest": os.environ.get("GROUNDED_MOTION_IMAGE_DIGEST", "unknown"),
            "artifacts": artifacts,
        }
        evidence_index_path = workspace / "evidence-index.json"
        write_json(evidence_index_path, evidence_index)
        artifacts.append(
            {
                "lane": "control",
                "name": "evidence-index.json",
                **store.upload(
                    evidence_index_path,
                    f"executions/{execution_id}/evidence-index.json",
                    content_type="application/json",
                ),
            }
        )
        result = {
            "schema": "grounded-motion-uploaded-track-result/v1",
            "execution_id": execution_id,
            "kind": "uploaded-track",
            "state": "completed",
            "pipeline_pass": True,
            "tracking_state": "tracked",
            "review_status": "unreviewed",
            "event_lock_status": "unlocked",
            "event_locked": False,
            "human_accepted": False,
            "source": {
                key: spec["source"][key]
                for key in ("file_id", "file_name", "mime_type", "sha256", "size_bytes")
            },
            "minimum_score": float(spec.get("minimum_score", 0.5)),
            "crop": spec.get("crop"),
            "detector_score_verification": score_verification,
            "cuda": cuda,
            "backend": receipt["backend"],
            "source_structural_pass": receipt.get("structural_pass") is True,
            "revision": os.environ.get("GROUNDED_MOTION_REVISION", "unknown"),
            "image_digest": os.environ.get("GROUNDED_MOTION_IMAGE_DIGEST", "unknown"),
            "started_unix": started,
            "completed_unix": time.time(),
            "duration_seconds": time.time() - started,
            "artifacts": artifacts,
        }
        store.write_json(execution_result_object(execution_id), result, if_generation_match=0)
    _status(
        store,
        execution_id,
        state="completed",
        stage="evidence-ready",
        pipeline_pass=True,
    )
    return result


def _run_canary(
    execution_id: str,
    store: GcsCanaryStore,
) -> dict[str, Any]:
    fixture = load_canary_manifest()
    started = time.time()
    _status(store, execution_id, state="running", stage="verifying-gpu")
    cuda = _verify_cuda()
    with tempfile.TemporaryDirectory(prefix=f"grounded-motion-{execution_id}-") as temporary:
        workspace = Path(temporary)
        inputs = workspace / "inputs"
        inputs.mkdir()
        source_video = inputs / "source.mp4"
        candidate_video = inputs / "candidate.mp4"
        _status(store, execution_id, state="running", stage="downloading-canary-inputs")
        store.download(fixture["source"]["gcs_object"], source_video)
        store.download(fixture["candidate"]["gcs_object"], candidate_video)
        for lane, video in (("source", source_video), ("candidate", candidate_video)):
            expected = fixture[lane]["video_sha256"]
            actual = sha256_file(video)
            if actual != expected:
                raise RuntimeError(f"{lane} canary input SHA-256 mismatch: {actual}")

        service = GroundedMotionService(workspace=workspace)
        _status(store, execution_id, state="running", stage="tracking-source")
        source = _track(service, source_video)
        _status(store, execution_id, state="running", stage="tracking-candidate")
        candidate = _track(service, candidate_video)
        _status(store, execution_id, state="running", stage="applying-judgment-policy")
        source_context = _apply_vanguard_judgment_context(
            source["job_path"], "source", fixture
        )
        candidate_context = _apply_vanguard_judgment_context(
            candidate["job_path"], "candidate", fixture
        )
        if (
            candidate_context["track"]["score_semantics"]
            != source_context["track"]["score_semantics"]
        ):
            raise RuntimeError("Canary lanes disagree on detector score semantics")
        source_score_verification = _verify_detector_score_preservation(
            source["job_path"]
        )
        candidate_score_verification = _verify_detector_score_preservation(
            candidate["job_path"]
        )
        _status(store, execution_id, state="running", stage="comparing-motion")
        comparison = service.compare_motion(
            CompareRequest(
                source_track_path=str(source["job_path"] / "pose-track.json"),
                candidate_track_path=str(candidate["job_path"] / "pose-track.json"),
                report_path=str(workspace / "comparison" / "candidate-motion-report.json"),
                trajectory_path=str(workspace / "comparison" / "candidate-trajectories.svg"),
            )
        )
        comparison_dir = workspace / "comparison"
        comparison_report = read_json(
            comparison_dir / "candidate-motion-report.json"
        )
        comparison_receipt = {
            "schema": "grounded-motion-vanguard-comparison/v2",
            "execution_id": execution_id,
            "source_track_sha256": sha256_file(source["job_path"] / "pose-track.json"),
            "candidate_track_sha256": sha256_file(candidate["job_path"] / "pose-track.json"),
            "judgment_status": comparison["judgment_status"],
            "mechanical_pass": comparison["mechanical_pass"],
            "judgment_blockers": comparison["judgment_blockers"],
            "comparison": comparison,
        }
        write_json(comparison_dir / "comparison-receipt.json", comparison_receipt)

        _status(store, execution_id, state="running", stage="publishing-evidence")
        artifacts = []
        artifacts.extend(_publish_tree(store, execution_id, "source", source["job_path"]))
        artifacts.extend(_publish_tree(store, execution_id, "candidate", candidate["job_path"]))
        artifacts.extend(_publish_tree(store, execution_id, "comparison", comparison_dir))
        evidence_index = {
            "schema": "grounded-motion-vanguard-evidence-index/v2",
            "execution_id": execution_id,
            "revision": os.environ.get("GROUNDED_MOTION_REVISION", "unknown"),
            "image_digest": os.environ.get("GROUNDED_MOTION_IMAGE_DIGEST", "unknown"),
            "artifacts": artifacts,
        }
        evidence_index_path = comparison_dir / "evidence-index.json"
        write_json(evidence_index_path, evidence_index)
        artifacts.append(
            {
                "lane": "comparison",
                "name": "evidence-index.json",
                **store.upload(
                    evidence_index_path,
                    f"executions/{execution_id}/comparison/evidence-index.json",
                    content_type="application/json",
                ),
            }
        )
        result = {
            "schema": "grounded-motion-vanguard-result/v2",
            "execution_id": execution_id,
            "state": "completed",
            "pipeline_pass": True,
            "judgment_status": comparison["judgment_status"],
            "mechanical_pass": comparison["mechanical_pass"],
            "judgment_blockers": comparison["judgment_blockers"],
            "human_accepted": False,
            "fixture": fixture["name"],
            "source_revision": fixture["source"]["revision"],
            "candidate_revision": fixture["candidate"]["revision"],
            "candidate_review_status": fixture["candidate"]["review_status"],
            "minimum_score": 0.5,
            "score_semantics": source_context["track"]["score_semantics"],
            "score_calibrated": source_context["track"]["score_calibrated"],
            "detector_score_verification": {
                "source": source_score_verification,
                "candidate": candidate_score_verification,
            },
            "diagnostic_summary": {
                "render_registration": comparison_report["diagnostic_metrics"][
                    "render_registration"
                ],
                "root_translation": comparison_report["diagnostic_metrics"][
                    "root_translation"
                ]["summary"],
                "root_relative_mechanics": {
                    key: value
                    for key, value in comparison_report["diagnostic_metrics"][
                        "root_relative_mechanics"
                    ].items()
                    if key != "diagnostic_deviations"
                },
                "quarantine": comparison_report["diagnostic_metrics"]["quarantine"],
            },
            "cuda": cuda,
            "backend": source["receipt"]["backend"],
            "source_structural_pass": source_context["report"]["pass"],
            "candidate_structural_pass": candidate_context["report"]["pass"],
            "revision": os.environ.get("GROUNDED_MOTION_REVISION", "unknown"),
            "image_digest": os.environ.get("GROUNDED_MOTION_IMAGE_DIGEST", "unknown"),
            "started_unix": started,
            "completed_unix": time.time(),
            "duration_seconds": time.time() - started,
            "artifacts": artifacts,
        }
        store.write_json(execution_result_object(execution_id), result, if_generation_match=0)
    _status(
        store,
        execution_id,
        state="completed",
        stage="evidence-ready",
        pipeline_pass=True,
    )
    return result


def run(execution_id: str, store: GcsCanaryStore | None = None) -> dict[str, Any]:
    execution_id = validate_execution_id(execution_id)
    store = store or GcsCanaryStore()
    spec, _ = store.read_json_or_none(execution_job_spec_object(execution_id))
    if spec is not None:
        if spec.get("kind") != "uploaded-track":
            raise RuntimeError(f"Unsupported Grounded Motion job kind: {spec.get('kind')}")
        return _run_uploaded_track(execution_id, store, spec)
    return _run_canary(execution_id, store)


def main() -> None:
    execution_id = required_env("GROUNDED_MOTION_EXECUTION_ID")
    store = GcsCanaryStore()
    try:
        result = run(execution_id, store)
        print(json.dumps({"ok": True, "execution_id": execution_id, "pipeline_pass": result["pipeline_pass"]}))
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        try:
            _status(
                store,
                execution_id,
                state="failed",
                stage="worker-failed",
                pipeline_pass=False,
                error=detail,
            )
        finally:
            traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
