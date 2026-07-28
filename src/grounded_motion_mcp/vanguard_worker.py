"""One-task GPU worker for the immutable Vanguard production canary."""

from __future__ import annotations

import json
import mimetypes
import os
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

from .artifacts import verify_manifest
from .hashing import read_json, sha256_file, write_json
from .models import CompareRequest, TrackRequest
from .service import GroundedMotionService
from .vanguard_cloud import (
    GcsCanaryStore,
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
        stage=stage,
        pipeline_pass=pipeline_pass,
        updated_unix=time.time(),
    )
    if error:
        payload["error"] = error
    store.write_json(execution_status_object(execution_id), payload)
    store.set_lock_state(execution_id, state)


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


def run(execution_id: str, store: GcsCanaryStore | None = None) -> dict[str, Any]:
    execution_id = validate_execution_id(execution_id)
    store = store or GcsCanaryStore()
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
        comparison_receipt = {
            "schema": "grounded-motion-vanguard-comparison/v1",
            "execution_id": execution_id,
            "source_track_sha256": sha256_file(source["job_path"] / "pose-track.json"),
            "candidate_track_sha256": sha256_file(candidate["job_path"] / "pose-track.json"),
            "mechanical_pass": bool(comparison["pass"]),
            "comparison": comparison,
        }
        write_json(comparison_dir / "comparison-receipt.json", comparison_receipt)

        _status(store, execution_id, state="running", stage="publishing-evidence")
        artifacts = []
        artifacts.extend(_publish_tree(store, execution_id, "source", source["job_path"]))
        artifacts.extend(_publish_tree(store, execution_id, "candidate", candidate["job_path"]))
        artifacts.extend(_publish_tree(store, execution_id, "comparison", comparison_dir))
        result = {
            "schema": "grounded-motion-vanguard-result/v1",
            "execution_id": execution_id,
            "state": "completed",
            "pipeline_pass": True,
            "mechanical_pass": bool(comparison["pass"]),
            "human_accepted": False,
            "fixture": fixture["name"],
            "source_revision": fixture["source"]["revision"],
            "candidate_revision": fixture["candidate"]["revision"],
            "candidate_review_status": fixture["candidate"]["review_status"],
            "minimum_score": 0.5,
            "cuda": cuda,
            "backend": source["receipt"]["backend"],
            "source_structural_pass": True,
            "candidate_structural_pass": True,
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
