"""Cloud control plane for uploaded motion tracking and the fixed Vanguard canary."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import timedelta
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

TERMINAL_STATES = frozenset({"completed", "failed"})
RUNNING_STATES = frozenset({"queued", "running"})
CANARY_SCOPE = "grounded-motion:vanguard-canary"
ALLOWED_EMAIL = "dclarke1005@gmail.com"
MAX_INPUT_BYTES = 200 * 1024 * 1024
MAX_INPUT_FRAMES = 10_000
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def load_canary_manifest() -> dict[str, Any]:
    resource = files("grounded_motion_mcp.data").joinpath("vanguard_canary.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable is missing: {name}")
    return value


def execution_status_object(execution_id: str) -> str:
    return f"executions/{execution_id}/status.json"


def execution_result_object(execution_id: str) -> str:
    return f"executions/{execution_id}/result.json"


def execution_job_spec_object(execution_id: str) -> str:
    return f"executions/{execution_id}/job-spec.json"


def execution_input_lock_object(execution_id: str) -> str:
    return f"executions/{execution_id}/input-lock.json"


def execution_intake_object(execution_id: str) -> str:
    return f"executions/{execution_id}/private-intake.json"


def execution_upload_intent_object(execution_id: str) -> str:
    return f"executions/{execution_id}/upload-intent.json"


def motion_queue_object(execution_id: str, created_unix_ns: int) -> str:
    return f"queues/motion/{created_unix_ns:020d}-{execution_id}.json"


def idempotency_object(kind: str, request_id: str) -> str:
    digest = hashlib.sha256(f"{kind}\0{request_id}".encode()).hexdigest()
    return f"idempotency/{kind}/{digest}.json"


def validate_execution_id(execution_id: str) -> str:
    try:
        parsed = uuid.UUID(execution_id)
    except (ValueError, AttributeError) as exc:
        raise ValueError("execution_id must be a UUID") from exc
    return str(parsed)


def _validate_sha256(value: str) -> str:
    normalized = value.strip().lower()
    if not SHA256_RE.fullmatch(normalized):
        raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
    return normalized


def _validate_mp4_name(value: str | None) -> str:
    name = Path(str(value or "source.mp4")).name
    if Path(name).suffix.lower() != ".mp4":
        raise ValueError("Grounded Motion accepts hash-locked .mp4 files only")
    return name


class ActiveExecutionError(RuntimeError):
    def __init__(self, execution_id: str):
        self.execution_id = execution_id
        super().__init__(f"Vanguard canary execution is already active: {execution_id}")


class GcsCanaryStore:
    """Private GCS repository with generation-checked state transitions."""

    def __init__(self, bucket_name: str | None = None):
        from google.cloud import storage

        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name or required_env("GROUNDED_MOTION_BUCKET"))

    def read_json(self, name: str) -> tuple[dict[str, Any], int]:
        blob = self.bucket.blob(name)
        blob.reload()
        generation = int(blob.generation or 0)
        payload = json.loads(blob.download_as_text(if_generation_match=generation))
        return payload, generation

    def read_json_or_none(self, name: str) -> tuple[dict[str, Any] | None, int | None]:
        blob = self.bucket.blob(name)
        if not blob.exists(self.client):
            return None, None
        return self.read_json(name)

    def write_json(
        self,
        name: str,
        payload: dict[str, Any],
        *,
        if_generation_match: int | None = None,
    ) -> int:
        blob = self.bucket.blob(name)
        blob.upload_from_string(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            content_type="application/json",
            if_generation_match=if_generation_match,
        )
        blob.reload()
        return int(blob.generation or 0)

    def delete(self, name: str, *, if_generation_match: int | None = None) -> None:
        self.bucket.blob(name).delete(if_generation_match=if_generation_match)

    def acquire_execution(self, execution_id: str, now: float) -> None:
        """Preserve the independent fixed-canary lock."""
        from google.api_core.exceptions import PreconditionFailed

        lock_name = "locks/vanguard-canary.json"
        lock = {
            "schema": "grounded-motion-canary-lock/v1",
            "execution_id": execution_id,
            "state": "queued",
            "updated_unix": now,
        }
        try:
            self.write_json(lock_name, lock, if_generation_match=0)
            return
        except PreconditionFailed:
            current, generation = self.read_json(lock_name)
        if current.get("state") in RUNNING_STATES:
            raise ActiveExecutionError(str(current.get("execution_id", "unknown")))
        try:
            self.write_json(lock_name, lock, if_generation_match=generation)
        except PreconditionFailed as exc:
            winner, _ = self.read_json(lock_name)
            raise ActiveExecutionError(str(winner.get("execution_id", "unknown"))) from exc

    def set_lock_state(self, execution_id: str, state: str) -> None:
        from google.api_core.exceptions import PreconditionFailed

        status, _ = self.read_json_or_none(execution_status_object(execution_id))
        lock_name = (
            "locks/motion-user.json"
            if status and status.get("kind") == "uploaded-track"
            else "locks/vanguard-canary.json"
        )
        lock, generation = self.read_json_or_none(lock_name)
        if not lock or lock.get("execution_id") != execution_id or generation is None:
            return
        lock["state"] = state
        lock["updated_unix"] = time.time()
        try:
            self.write_json(lock_name, lock, if_generation_match=generation)
        except PreconditionFailed:
            return

    def download(
        self,
        object_name: str,
        destination: Path,
        *,
        generation: int | None = None,
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        kwargs = {"if_generation_match": generation} if generation is not None else {}
        self.bucket.blob(object_name).download_to_filename(str(destination), **kwargs)

    def upload(
        self,
        source: Path,
        object_name: str,
        content_type: str | None = None,
        *,
        if_generation_match: int | None = None,
    ) -> dict[str, Any]:
        from .hashing import sha256_file

        blob = self.bucket.blob(object_name)
        blob.upload_from_filename(
            str(source),
            content_type=content_type,
            if_generation_match=if_generation_match,
        )
        blob.reload()
        return {
            "object": object_name,
            "gs_uri": f"gs://{self.bucket.name}/{object_name}",
            "sha256": sha256_file(source),
            "size_bytes": source.stat().st_size,
            "generation": int(blob.generation or 0),
            "content_type": content_type or blob.content_type or "application/octet-stream",
        }

    def blob_info(self, object_name: str) -> dict[str, Any]:
        blob = self.bucket.blob(object_name)
        blob.reload()
        return {
            "object": object_name,
            "gs_uri": f"gs://{self.bucket.name}/{object_name}",
            "size_bytes": int(blob.size or 0),
            "generation": int(blob.generation or 0),
            "content_type": blob.content_type or "application/octet-stream",
            "metadata": dict(blob.metadata or {}),
        }

    def _signing_credentials(self) -> Any:
        import google.auth
        from google.auth import iam
        from google.auth.transport.requests import Request
        from google.oauth2 import service_account

        service_account_email = required_env("GROUNDED_MOTION_SIGNER_SERVICE_ACCOUNT")
        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        request = Request()
        credentials.refresh(request)
        signer = iam.Signer(request, credentials, service_account_email)
        return service_account.Credentials(
            signer=signer,
            service_account_email=service_account_email,
            token_uri="https://oauth2.googleapis.com/token",
        )

    def signed_url(
        self,
        object_name: str,
        expires: timedelta = timedelta(hours=24),
        *,
        method: str = "GET",
        content_type: str | None = None,
        query_parameters: dict[str, str] | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "version": "v4",
            "expiration": expires,
            "method": method,
            "credentials": self._signing_credentials(),
        }
        if content_type:
            kwargs["content_type"] = content_type
        if query_parameters:
            kwargs["query_parameters"] = query_parameters
        return self.bucket.blob(object_name).generate_signed_url(**kwargs)

    def ingest_openai_file(
        self,
        execution_id: str,
        file_payload: dict[str, Any],
        *,
        expected_sha256: str | None = None,
    ) -> dict[str, Any]:
        import requests

        download_url = str(file_payload.get("download_url", "")).strip()
        parsed = urlparse(download_url)
        host = (parsed.hostname or "").lower()
        allowed_suffixes = tuple(
            item.strip().lower()
            for item in os.environ.get(
                "GROUNDED_MOTION_ALLOWED_FILE_HOST_SUFFIXES",
                "oaiusercontent.com,blob.core.windows.net,amazonaws.com",
            ).split(",")
            if item.strip()
        )
        if parsed.scheme != "https" or not any(
            host == suffix or host.endswith(f".{suffix}") for suffix in allowed_suffixes
        ):
            raise ValueError("source_file download_url is not an approved ChatGPT file host")

        max_bytes = int(os.environ.get("GROUNDED_MOTION_MAX_INPUT_BYTES", str(MAX_INPUT_BYTES)))
        file_name = _validate_mp4_name(file_payload.get("file_name"))
        object_name = f"executions/{execution_id}/inputs/source.mp4"
        content_type = str(file_payload.get("mime_type") or "video/mp4").lower()
        if content_type not in {"video/mp4", "application/mp4", "application/octet-stream"}:
            raise ValueError("source_file mime_type must describe an MP4")
        with tempfile.NamedTemporaryFile(prefix="grounded-motion-upload-", suffix=".mp4") as handle:
            total = 0
            with requests.get(download_url, stream=True, timeout=(10, 300)) as response:
                response.raise_for_status()
                declared = int(response.headers.get("content-length", "0") or 0)
                if declared > max_bytes:
                    raise ValueError(f"source_file exceeds the {max_bytes} byte input limit")
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError(f"source_file exceeds the {max_bytes} byte input limit")
                    handle.write(chunk)
            handle.flush()
            if total == 0:
                raise ValueError("source_file is empty")
            artifact = self.upload(
                Path(handle.name),
                object_name,
                content_type="video/mp4",
                if_generation_match=0,
            )
        if expected_sha256 and artifact["sha256"] != _validate_sha256(expected_sha256):
            raise ValueError(
                "source_file SHA-256 mismatch: "
                f"expected {expected_sha256}, got {artifact['sha256']}"
            )
        return {
            **artifact,
            "file_id": str(file_payload["file_id"]),
            "file_name": file_name,
            "mime_type": "video/mp4",
        }

    def enqueue_motion(self, execution_id: str, created_unix_ns: int) -> dict[str, Any]:
        payload = {
            "schema": "grounded-motion-queue-entry/v1",
            "execution_id": execution_id,
            "state": "pending",
            "created_unix_ns": created_unix_ns,
        }
        object_name = motion_queue_object(execution_id, created_unix_ns)
        generation = self.write_json(object_name, payload, if_generation_match=0)
        return {"object": object_name, "generation": generation, **payload}

    def list_motion_queue(self) -> list[dict[str, Any]]:
        entries = []
        for blob in self.client.list_blobs(self.bucket, prefix="queues/motion/"):
            payload, generation = self.read_json(blob.name)
            entries.append({"object": blob.name, "generation": generation, **payload})
        return sorted(entries, key=lambda item: item["object"])


@dataclass(frozen=True)
class CloudRunJobLauncher:
    project: str
    region: str
    job: str

    @classmethod
    def from_env(cls, job_env: str = "GROUNDED_MOTION_JOB") -> CloudRunJobLauncher:
        default_job = (
            "grounded-motion-preflight"
            if job_env == "GROUNDED_MOTION_PREFLIGHT_JOB"
            else "grounded-motion-vanguard-canary"
        )
        return cls(
            project=required_env("GOOGLE_CLOUD_PROJECT"),
            region=os.environ.get("GROUNDED_MOTION_REGION", "us-central1"),
            job=os.environ.get(job_env, default_job),
        )

    def launch(self, execution_id: str, *, timeout_seconds: int = 3600) -> dict[str, Any]:
        import google.auth
        from google.auth.transport.requests import AuthorizedSession

        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        session = AuthorizedSession(credentials)
        name = f"projects/{self.project}/locations/{self.region}/jobs/{self.job}"
        response = session.post(
            f"https://run.googleapis.com/v2/{name}:run",
            json={
                "overrides": {
                    "containerOverrides": [
                        {
                            "env": [
                                {
                                    "name": "GROUNDED_MOTION_EXECUTION_ID",
                                    "value": execution_id,
                                }
                            ]
                        }
                    ],
                    "taskCount": 1,
                    "timeout": f"{timeout_seconds}s",
                }
            },
            timeout=30,
        )
        if response.status_code >= 300:
            raise RuntimeError(
                f"Cloud Run Job launch failed ({response.status_code}): {response.text[:1000]}"
            )
        payload = response.json()
        return {"operation": payload.get("name", ""), "job": name}

    def operation_status(self, operation: str) -> dict[str, Any]:
        if not operation:
            return {"done": False}
        import google.auth
        from google.auth.transport.requests import AuthorizedSession

        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        response = AuthorizedSession(credentials).get(
            f"https://run.googleapis.com/v2/{operation}", timeout=15
        )
        if response.status_code >= 300:
            raise RuntimeError(
                f"Cloud Run operation read failed ({response.status_code}): {response.text[:500]}"
            )
        return response.json()


@dataclass(frozen=True)
class CloudTasksDispatcher:
    project: str
    location: str
    queue: str
    target_base_url: str
    service_account_email: str

    @classmethod
    def from_env(cls) -> CloudTasksDispatcher | None:
        target = os.environ.get("GROUNDED_MOTION_INTERNAL_URL", "").rstrip("/")
        if not target:
            return None
        return cls(
            project=required_env("GOOGLE_CLOUD_PROJECT"),
            location=os.environ.get("GROUNDED_MOTION_REGION", "us-central1"),
            queue=os.environ.get("GROUNDED_MOTION_TASK_QUEUE", "grounded-motion-dispatch"),
            target_base_url=target,
            service_account_email=required_env("GROUNDED_MOTION_CONTROL_SERVICE_ACCOUNT"),
        )

    def enqueue(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        delay_seconds: int = 0,
    ) -> dict[str, Any]:
        import google.auth
        from google.auth.transport.requests import AuthorizedSession

        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        parent = (
            f"projects/{self.project}/locations/{self.location}/queues/{self.queue}"
        )
        task: dict[str, Any] = {
            "dispatchDeadline": "900s",
            "httpRequest": {
                "httpMethod": "POST",
                "url": f"{self.target_base_url}/{path.lstrip('/')}",
                "headers": {"Content-Type": "application/json"},
                "body": base64.b64encode(
                    json.dumps(payload, sort_keys=True).encode()
                ).decode(),
                "oidcToken": {
                    "serviceAccountEmail": self.service_account_email,
                    "audience": self.target_base_url,
                },
            }
        }
        if delay_seconds:
            from datetime import datetime, timezone

            scheduled = datetime.fromtimestamp(
                int(time.time()) + delay_seconds, tz=timezone.utc
            )
            task["scheduleTime"] = scheduled.isoformat().replace("+00:00", "Z")
        response = AuthorizedSession(credentials).post(
            f"https://cloudtasks.googleapis.com/v2/{parent}/tasks",
            json={"task": task},
            timeout=30,
        )
        response.raise_for_status()
        created = response.json()
        return {
            "task": created["name"],
            "schedule_delay_seconds": delay_seconds,
        }


class VanguardCanaryController:
    def __init__(
        self,
        store: GcsCanaryStore | Any | None = None,
        launcher: CloudRunJobLauncher | Any | None = None,
        preflight_launcher: CloudRunJobLauncher | Any | None = None,
        tasks: CloudTasksDispatcher | Any | None = None,
    ) -> None:
        self.store = store or GcsCanaryStore()
        self.launcher = launcher or CloudRunJobLauncher.from_env()
        self.preflight_launcher = preflight_launcher or launcher or CloudRunJobLauncher.from_env(
            "GROUNDED_MOTION_PREFLIGHT_JOB"
        )
        self.tasks = tasks if tasks is not None else CloudTasksDispatcher.from_env()

    @staticmethod
    def _validate_tracking_parameters(crop: list[int] | None, minimum_score: float) -> None:
        if crop is not None and (
            len(crop) != 4
            or any(not isinstance(value, int) for value in crop)
            or crop[0] < 0
            or crop[1] < 0
            or crop[2] <= 0
            or crop[3] <= 0
        ):
            raise ValueError("crop must be [x, y, width, height] with nonnegative origin")
        if not 0 <= minimum_score <= 10:
            raise ValueError("minimum_score must be between 0 and 10")

    def _new_motion_status(self, execution_id: str, now: float, stage: str) -> dict[str, Any]:
        return {
            "schema": "grounded-motion-status/v2",
            "execution_id": execution_id,
            "kind": "uploaded-track",
            "state": "queued",
            "phase": stage,
            "stage": stage,
            "terminal": False,
            "pipeline_pass": False,
            "tracking_state": "pending",
            "review_status": "unreviewed",
            "event_lock_status": "unlocked",
            "revision": os.environ.get("GROUNDED_MOTION_REVISION", "unknown"),
            "image_digest": os.environ.get("GROUNDED_MOTION_IMAGE_DIGEST", "unknown"),
            "created_unix": now,
            "updated_unix": now,
        }

    def start(self) -> dict[str, Any]:
        execution_id = str(uuid.uuid4())
        now = time.time()
        self.store.acquire_execution(execution_id, now)
        revision = os.environ.get("GROUNDED_MOTION_REVISION", "unknown")
        image_digest = os.environ.get("GROUNDED_MOTION_IMAGE_DIGEST", "unknown")
        status = {
            "schema": "grounded-motion-vanguard-status/v1",
            "execution_id": execution_id,
            "state": "queued",
            "stage": "launching-gpu-job",
            "pipeline_pass": False,
            "fixture": load_canary_manifest()["name"],
            "revision": revision,
            "image_digest": image_digest,
            "created_unix": now,
            "updated_unix": now,
        }
        self.store.write_json(execution_status_object(execution_id), status, if_generation_match=0)
        try:
            launch = self.launcher.launch(execution_id)
            status["launch"] = launch
            self.store.write_json(execution_status_object(execution_id), status)
        except Exception as exc:
            status.update(
                state="failed",
                stage="gpu-job-launch-failed",
                error=str(exc),
                updated_unix=time.time(),
            )
            self.store.write_json(execution_status_object(execution_id), status)
            self.store.set_lock_state(execution_id, "failed")
            raise
        return {
            "execution_id": execution_id,
            "state": "queued",
            "status_tool": "get_vanguard_canary_status",
            "result_tool": "get_vanguard_canary_result",
            "fixture": status["fixture"],
            "revision": revision,
            "image_digest": image_digest,
            "launch": launch,
        }

    def _claim_request_id(self, kind: str, request_id: str | None) -> tuple[str, bool]:
        """Return an execution id and whether this is an idempotent replay."""
        if not request_id:
            return str(uuid.uuid4()), False
        normalized = request_id.strip()
        if not normalized or len(normalized) > 128:
            raise ValueError("request_id must contain 1 to 128 characters")
        from google.api_core.exceptions import PreconditionFailed

        object_name = idempotency_object(kind, normalized)
        existing, _ = self.store.read_json_or_none(object_name)
        if existing is not None:
            return validate_execution_id(existing["execution_id"]), True
        execution_id = str(uuid.uuid4())
        payload = {
            "schema": "grounded-motion-idempotency/v1",
            "kind": kind,
            "request_id": normalized,
            "execution_id": execution_id,
            "created_unix": time.time(),
        }
        try:
            self.store.write_json(object_name, payload, if_generation_match=0)
            return execution_id, False
        except PreconditionFailed:
            winner, _ = self.store.read_json(object_name)
            return validate_execution_id(winner["execution_id"]), True

    def _tracking_replay(self, execution_id: str) -> dict[str, Any]:
        status, _ = self.store.read_json(execution_status_object(execution_id))
        return {
            "execution_id": execution_id,
            "kind": "uploaded-track",
            "state": status.get("state"),
            "phase": status.get("phase"),
            "status_tool": "get_motion_status",
            "result_tool": "get_motion_result",
            "revision": status.get("revision"),
            "image_digest": status.get("image_digest"),
            "idempotent_replay": True,
        }

    def start_motion_tracking(
        self,
        source_file: dict[str, Any],
        *,
        expected_sha256: str | None = None,
        crop: list[int] | None = None,
        minimum_score: float = 0.5,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        self._validate_tracking_parameters(crop, minimum_score)
        execution_id, replay = self._claim_request_id("attachment", request_id)
        if replay:
            return self._tracking_replay(execution_id)
        now = time.time()
        status = self._new_motion_status(execution_id, now, "ingesting-source-file")
        status["request_id"] = request_id
        self.store.write_json(execution_status_object(execution_id), status, if_generation_match=0)
        intake = {
            "schema": "grounded-motion-private-intake/v1",
            "execution_id": execution_id,
            "source_file": source_file,
            "expected_sha256": _validate_sha256(expected_sha256) if expected_sha256 else None,
            "crop": crop,
            "minimum_score": minimum_score,
            "created_unix": now,
        }
        self.store.write_json(execution_intake_object(execution_id), intake, if_generation_match=0)
        if self.tasks is not None:
            task = self.tasks.enqueue(
                "/internal/ingest",
                {"execution_id": execution_id},
            )
        else:
            synchronous = self.ingest_attachment(execution_id)
            synchronous.update(
                kind="uploaded-track",
                revision=status["revision"],
                image_digest=status["image_digest"],
                task={"mode": "synchronous-fallback"},
            )
            return synchronous
        return {
            "execution_id": execution_id,
            "kind": "uploaded-track",
            "state": "queued",
            "phase": "ingesting-source-file",
            "status_tool": "get_motion_status",
            "result_tool": "get_motion_result",
            "revision": status["revision"],
            "image_digest": status["image_digest"],
            "task": task,
        }

    def ingest_attachment(self, execution_id: str) -> dict[str, Any]:
        execution_id = validate_execution_id(execution_id)
        intake, intake_generation = self.store.read_json(execution_intake_object(execution_id))
        status, _ = self.store.read_json(execution_status_object(execution_id))
        try:
            source = self.store.ingest_openai_file(
                execution_id,
                intake["source_file"],
                expected_sha256=intake.get("expected_sha256"),
            )
            spec = {
                "schema": "grounded-motion-job-spec/v2",
                "execution_id": execution_id,
                "kind": "uploaded-track",
                "source": source,
                "crop": intake.get("crop"),
                "minimum_score": float(intake.get("minimum_score", 0.5)),
                "model_preset": "rtmw-x-cocktail14-multiperson-384x288",
                "created_unix": intake["created_unix"],
            }
            self.store.write_json(execution_job_spec_object(execution_id), spec, if_generation_match=0)
            self.store.delete(
                execution_intake_object(execution_id), if_generation_match=intake_generation
            )
            return self._launch_preflight(execution_id, status, source)
        except Exception as exc:
            self._fail_status(status, "ingest-failed", exc)
            raise

    def create_motion_upload(
        self,
        *,
        file_name: str,
        size_bytes: int,
        sha256: str,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        file_name = _validate_mp4_name(file_name)
        expected_sha = _validate_sha256(sha256)
        if not 0 < int(size_bytes) <= int(
            os.environ.get("GROUNDED_MOTION_MAX_INPUT_BYTES", str(MAX_INPUT_BYTES))
        ):
            raise ValueError(f"size_bytes must be between 1 and {MAX_INPUT_BYTES}")
        execution_id, replay = self._claim_request_id("direct-upload", request_id)
        if replay:
            status, _ = self.store.read_json(execution_status_object(execution_id))
            intent, _ = self.store.read_json(execution_upload_intent_object(execution_id))
            return {
                "execution_id": execution_id,
                "state": status.get("state"),
                "phase": status.get("phase"),
                "upload_url": self.store.signed_url(
                    intent["object"],
                    expires=timedelta(minutes=15),
                    method="PUT",
                    content_type="video/mp4",
                    query_parameters={"ifGenerationMatch": "0"},
                ),
                "upload_url_expires_minutes": 15,
                "required_headers": {"Content-Type": "video/mp4"},
                "expected_sha256": intent["sha256"],
                "expected_size_bytes": intent["size_bytes"],
                "finalize_tool": "finalize_motion_upload",
                "idempotent_replay": True,
            }
        now = time.time()
        status = self._new_motion_status(execution_id, now, "awaiting-upload")
        status["request_id"] = request_id
        self.store.write_json(execution_status_object(execution_id), status, if_generation_match=0)
        object_name = f"executions/{execution_id}/inputs/source.mp4"
        intent = {
            "schema": "grounded-motion-upload-intent/v1",
            "execution_id": execution_id,
            "file_name": file_name,
            "size_bytes": int(size_bytes),
            "sha256": expected_sha,
            "content_type": "video/mp4",
            "object": object_name,
            "created_unix": now,
        }
        self.store.write_json(
            execution_upload_intent_object(execution_id), intent, if_generation_match=0
        )
        upload_url = self.store.signed_url(
            object_name,
            expires=timedelta(minutes=15),
            method="PUT",
            content_type="video/mp4",
            query_parameters={"ifGenerationMatch": "0"},
        )
        return {
            "execution_id": execution_id,
            "state": "queued",
            "phase": "awaiting-upload",
            "upload_url": upload_url,
            "upload_url_expires_minutes": 15,
            "required_headers": {"Content-Type": "video/mp4"},
            "expected_sha256": expected_sha,
            "expected_size_bytes": int(size_bytes),
            "finalize_tool": "finalize_motion_upload",
        }

    def finalize_motion_upload(
        self,
        execution_id: str,
        *,
        crop: list[int] | None = None,
        minimum_score: float = 0.5,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        self._validate_tracking_parameters(crop, minimum_score)
        execution_id = validate_execution_id(execution_id)
        intent, _ = self.store.read_json(execution_upload_intent_object(execution_id))
        status, _ = self.store.read_json(execution_status_object(execution_id))
        source = self.store.blob_info(intent["object"])
        if source["size_bytes"] != intent["size_bytes"]:
            raise ValueError(
                f"uploaded size mismatch: expected {intent['size_bytes']}, got {source['size_bytes']}"
            )
        if source["content_type"] != "video/mp4":
            raise ValueError(f"uploaded content type is not video/mp4: {source['content_type']}")
        source.update(
            file_id=f"direct:{execution_id}",
            file_name=intent["file_name"],
            mime_type="video/mp4",
            sha256=intent["sha256"],
        )
        spec = {
            "schema": "grounded-motion-job-spec/v2",
            "execution_id": execution_id,
            "kind": "uploaded-track",
            "source": source,
            "crop": crop,
            "minimum_score": minimum_score,
            "model_preset": "rtmw-x-cocktail14-multiperson-384x288",
            "request_id": request_id,
            "created_unix": intent["created_unix"],
        }
        existing, _ = self.store.read_json_or_none(execution_job_spec_object(execution_id))
        if existing is None:
            self.store.write_json(execution_job_spec_object(execution_id), spec, if_generation_match=0)
        elif existing != spec:
            raise ValueError("finalize parameters conflict with the existing immutable job spec")
        return self._launch_preflight(execution_id, status, source)

    def _launch_preflight(
        self,
        execution_id: str,
        status: dict[str, Any],
        source: dict[str, Any],
    ) -> dict[str, Any]:
        if status.get("phase") not in {"awaiting-upload", "ingesting-source-file"}:
            return self._tracking_replay(execution_id)
        status.update(
            phase="preflighting",
            stage="preflighting",
            source={
                key: source[key]
                for key in ("file_id", "file_name", "mime_type", "sha256", "size_bytes", "generation")
            },
            updated_unix=time.time(),
        )
        launch = self.preflight_launcher.launch(execution_id, timeout_seconds=900)
        status["preflight_launch"] = launch
        self.store.write_json(execution_status_object(execution_id), status)
        return {
            "execution_id": execution_id,
            "state": "queued",
            "phase": "preflighting",
            "source": status["source"],
            "status_tool": "get_motion_status",
            "result_tool": "get_motion_result",
            "preflight_launch": launch,
        }

    def _reconcile_operation(
        self,
        status: dict[str, Any],
        launcher: CloudRunJobLauncher | Any,
        launch_key: str,
    ) -> dict[str, Any]:
        launch = status.get(launch_key) or {}
        operation = str(launch.get("operation", ""))
        if not operation or (
            status.get("state") in TERMINAL_STATES
            and status.get("review_status") != "validating"
        ):
            return status
        try:
            operation_state = launcher.operation_status(operation)
        except (AttributeError, RuntimeError) as exc:
            status["operation_warning"] = str(exc)
            return status
        if not operation_state.get("done"):
            return status
        error = operation_state.get("error")
        if error:
            self._fail_status(
                status,
                f"{launch_key.replace('_launch', '')}-operation-failed",
                RuntimeError(json.dumps(error, sort_keys=True)),
            )
            return status
        now = time.time()
        first_seen = status.get("operation_done_seen_unix")
        if first_seen is None:
            status["operation_done_seen_unix"] = now
            status["updated_unix"] = now
            self.store.write_json(execution_status_object(status["execution_id"]), status)
        elif now - float(first_seen) >= 60:
            self._fail_status(
                status,
                f"{launch_key.replace('_launch', '')}-worker-stranded",
                RuntimeError(
                    "Cloud Run operation completed without a terminal Grounded Motion status"
                ),
            )
        return status

    def _fail_status(self, status: dict[str, Any], stage: str, exc: Exception) -> None:
        status.update(
            state="failed",
            phase=stage,
            stage=stage,
            terminal=True,
            error=str(exc),
            updated_unix=time.time(),
        )
        self.store.write_json(execution_status_object(status["execution_id"]), status)
        self.store.set_lock_state(status["execution_id"], "failed")

    def dispatch_next(self) -> dict[str, Any]:
        """Claim and launch at most one queued user GPU execution."""
        from google.api_core.exceptions import PreconditionFailed

        lock_name = "locks/motion-user.json"
        lock, lock_generation = self.store.read_json_or_none(lock_name)
        if lock and lock.get("state") in RUNNING_STATES:
            active_id = str(lock.get("execution_id"))
            active_status, _ = self.store.read_json_or_none(execution_status_object(active_id))
            if active_status:
                active_status = self._reconcile_operation(
                    active_status, self.launcher, "launch"
                )
            if active_status and active_status.get("state") in TERMINAL_STATES:
                lock, lock_generation = self.store.read_json(lock_name)
                if lock.get("execution_id") != active_id:
                    return {"state": "raced"}
                if lock.get("state") != active_status["state"]:
                    lock["state"] = active_status["state"]
                    lock["updated_unix"] = time.time()
                    try:
                        lock_generation = self.store.write_json(
                            lock_name, lock, if_generation_match=lock_generation
                        )
                    except PreconditionFailed:
                        return {"state": "raced"}
            else:
                if self.tasks is not None:
                    self.tasks.enqueue("/internal/dispatch", {}, delay_seconds=30)
                return {"state": "active", "execution_id": active_id}

        queue = self.store.list_motion_queue()
        selected = None
        for entry in queue:
            if entry.get("state") != "pending":
                continue
            candidate_status, _ = self.store.read_json_or_none(
                execution_status_object(str(entry["execution_id"]))
            )
            if candidate_status and candidate_status.get("stage") == "awaiting-dispatch":
                selected = entry
                break
        if selected is None:
            return {"state": "idle"}

        execution_id = str(selected["execution_id"])
        new_lock = {
            "schema": "grounded-motion-user-lock/v1",
            "execution_id": execution_id,
            "state": "queued",
            "updated_unix": time.time(),
        }
        try:
            self.store.write_json(
                lock_name,
                new_lock,
                if_generation_match=0 if lock_generation is None else lock_generation,
            )
        except PreconditionFailed:
            if self.tasks is not None:
                self.tasks.enqueue("/internal/dispatch", {}, delay_seconds=5)
            return {"state": "raced"}

        selected["state"] = "claimed"
        selected["claimed_unix"] = time.time()
        queue_payload = {key: value for key, value in selected.items() if key not in {"object", "generation"}}
        self.store.write_json(
            selected["object"], queue_payload, if_generation_match=selected["generation"]
        )
        status, _ = self.store.read_json(execution_status_object(execution_id))
        try:
            launch = self.launcher.launch(execution_id)
            status.update(
                state="queued",
                phase="gpu-job-launched",
                stage="gpu-job-launched",
                launch=launch,
                updated_unix=time.time(),
            )
            self.store.write_json(execution_status_object(execution_id), status)
            new_lock.update(state="running", launch=launch, updated_unix=time.time())
            current_lock, current_generation = self.store.read_json(lock_name)
            if current_lock.get("execution_id") == execution_id:
                self.store.write_json(
                    lock_name, new_lock, if_generation_match=current_generation
                )
            if self.tasks is not None:
                self.tasks.enqueue("/internal/dispatch", {}, delay_seconds=30)
            return {"state": "launched", "execution_id": execution_id, "launch": launch}
        except Exception as exc:
            self._fail_status(status, "gpu-job-launch-failed", exc)
            if self.tasks is not None:
                self.tasks.enqueue("/internal/dispatch", {}, delay_seconds=1)
            raise

    def submit_motion_review(
        self,
        execution_id: str,
        *,
        tracked_evidence_sha256: str,
        subjects: list[dict[str, Any]],
        excluded_subjects: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
        reviewer: str = ALLOWED_EMAIL,
    ) -> dict[str, Any]:
        execution_id = validate_execution_id(execution_id)
        tracked_evidence_sha256 = _validate_sha256(tracked_evidence_sha256)
        status, _ = self.store.read_json(execution_status_object(execution_id))
        if status.get("state") != "completed" or status.get("tracking_state") != "tracked":
            raise ValueError("review requires a completed tracked execution")
        if status.get("review_status") == "reviewed":
            return {
                "execution_id": execution_id,
                "state": "completed",
                "review_status": "reviewed",
                "idempotent_replay": True,
            }
        result, _ = self.store.read_json(execution_result_object(execution_id))
        track_artifact = next(
            (
                item
                for item in result.get("artifacts", [])
                if item.get("name") == "track-set.json"
            ),
            None,
        )
        if track_artifact is None:
            raise ValueError("execution does not contain a multi-person track-set artifact")
        if track_artifact.get("sha256") != tracked_evidence_sha256:
            raise ValueError("tracked_evidence_sha256 does not match the immutable track set")
        submission = {
            "schema": "grounded-motion-review/v1",
            "execution_id": execution_id,
            "tracked_evidence_sha256": tracked_evidence_sha256,
            "reviewer": reviewer.strip().lower(),
            "subjects": subjects,
            "excluded_subjects": excluded_subjects or [],
            "request_id": request_id,
            "submitted_unix": time.time(),
        }
        submission_object = f"executions/{execution_id}/review/submission.json"
        existing, _ = self.store.read_json_or_none(submission_object)
        if existing is None:
            self.store.write_json(submission_object, submission, if_generation_match=0)
        elif {
            key: existing.get(key)
            for key in ("tracked_evidence_sha256", "reviewer", "subjects", "excluded_subjects", "request_id")
        } != {
            key: submission.get(key)
            for key in ("tracked_evidence_sha256", "reviewer", "subjects", "excluded_subjects", "request_id")
        }:
            raise ValueError("a different immutable review submission already exists")
        spec, spec_generation = self.store.read_json(execution_job_spec_object(execution_id))
        spec["review_task"] = {
            "submission_object": submission_object,
            "track_set_object": track_artifact["object"],
            "track_set_generation": int(track_artifact["generation"]),
            "track_set_sha256": tracked_evidence_sha256,
        }
        self.store.write_json(
            execution_job_spec_object(execution_id), spec, if_generation_match=spec_generation
        )
        status.update(
            phase="review-validating",
            stage="review-validating",
            review_status="validating",
            event_lock_status="unlocked",
            updated_unix=time.time(),
        )
        launch = self.preflight_launcher.launch(execution_id, timeout_seconds=900)
        status["review_launch"] = launch
        self.store.write_json(execution_status_object(execution_id), status)
        return {
            "execution_id": execution_id,
            "state": "completed",
            "tracking_state": "tracked",
            "review_status": "validating",
            "event_lock_status": "unlocked",
            "status_tool": "get_motion_status",
            "result_tool": "get_motion_result",
            "review_launch": launch,
        }

    def status(self, execution_id: str) -> dict[str, Any]:
        execution_id = validate_execution_id(execution_id)
        status, _ = self.store.read_json(execution_status_object(execution_id))
        if status.get("kind") == "uploaded-track":
            if status.get("review_status") == "validating":
                status = self._reconcile_operation(
                    status, self.preflight_launcher, "review_launch"
                )
            elif (
                status.get("state") not in TERMINAL_STATES
                and status.get("phase") == "preflighting"
            ):
                status = self._reconcile_operation(
                    status, self.preflight_launcher, "preflight_launch"
                )
            if status.get("state") not in TERMINAL_STATES:
                try:
                    self.dispatch_next()
                    status, _ = self.store.read_json(execution_status_object(execution_id))
                except Exception as exc:  # noqa: BLE001 - polling remains read-safe
                    status["dispatch_warning"] = str(exc)
                pending = [
                    item
                    for item in self.store.list_motion_queue()
                    if item.get("state") == "pending"
                ]
                pending_ids = [str(item["execution_id"]) for item in pending]
                if execution_id in pending_ids:
                    status["queue_position"] = pending_ids.index(execution_id) + 1
        return status

    def result(self, execution_id: str) -> dict[str, Any]:
        execution_id = validate_execution_id(execution_id)
        status = self.status(execution_id)
        if status.get("state") not in TERMINAL_STATES:
            pending = {
                "execution_id": execution_id,
                "state": status.get("state"),
                "ready": False,
                "stage": status.get("stage"),
            }
            for key in ("phase", "queue_position"):
                if status.get(key) is not None:
                    pending[key] = status[key]
            return pending
        if status.get("state") == "failed":
            return {
                "execution_id": execution_id,
                "state": "failed",
                "ready": True,
                "pipeline_pass": False,
                "error": status.get("error", "worker failed"),
                "stage": status.get("stage"),
            }
        result, _ = self.store.read_json(execution_result_object(execution_id))
        signed = []
        for artifact in result.get("artifacts", []):
            item = dict(artifact)
            item["url"] = self.store.signed_url(item["object"])
            item["url_expires_hours"] = 24
            signed.append(item)
        result["artifacts"] = signed
        result["result_url"] = self.store.signed_url(execution_result_object(execution_id))
        result["status_url"] = self.store.signed_url(execution_status_object(execution_id))
        input_lock, _ = self.store.read_json_or_none(execution_input_lock_object(execution_id))
        if input_lock is not None:
            result["input_lock_url"] = self.store.signed_url(
                execution_input_lock_object(execution_id)
            )
        result["control_urls_expire_hours"] = 24
        result["ready"] = True
        return result
