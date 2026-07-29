"""Cloud control plane for the fixed Vanguard production canary."""

from __future__ import annotations

import json
import mimetypes
import os
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


def validate_execution_id(execution_id: str) -> str:
    try:
        parsed = uuid.UUID(execution_id)
    except (ValueError, AttributeError) as exc:
        raise ValueError("execution_id must be a UUID") from exc
    return str(parsed)


class ActiveExecutionError(RuntimeError):
    def __init__(self, execution_id: str):
        self.execution_id = execution_id
        super().__init__(f"Vanguard canary execution is already active: {execution_id}")


class GcsCanaryStore:
    """Small GCS repository with generation-checked JSON writes."""

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

    def acquire_execution(self, execution_id: str, now: float) -> None:
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
        if generation is None:
            raise RuntimeError("Canary lock disappeared during replacement")
        try:
            self.write_json(lock_name, lock, if_generation_match=generation)
        except PreconditionFailed as exc:
            winner, _ = self.read_json(lock_name)
            raise ActiveExecutionError(str(winner.get("execution_id", "unknown"))) from exc

    def set_lock_state(self, execution_id: str, state: str) -> None:
        from google.api_core.exceptions import PreconditionFailed

        lock, generation = self.read_json("locks/vanguard-canary.json")
        if lock.get("execution_id") != execution_id or generation is None:
            return
        lock["state"] = state
        lock["updated_unix"] = time.time()
        try:
            self.write_json("locks/vanguard-canary.json", lock, if_generation_match=generation)
        except PreconditionFailed:
            # Status/result remain authoritative even if the convenience lock races.
            return

    def download(self, object_name: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.bucket.blob(object_name).download_to_filename(str(destination))

    def upload(self, source: Path, object_name: str, content_type: str | None = None) -> dict[str, Any]:
        from .hashing import sha256_file

        blob = self.bucket.blob(object_name)
        blob.upload_from_filename(str(source), content_type=content_type)
        blob.reload()
        return {
            "object": object_name,
            "gs_uri": f"gs://{self.bucket.name}/{object_name}",
            "sha256": sha256_file(source),
            "size_bytes": source.stat().st_size,
            "generation": int(blob.generation or 0),
            "content_type": content_type or blob.content_type or "application/octet-stream",
        }

    def signed_url(self, object_name: str, expires: timedelta = timedelta(hours=24)) -> str:
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
        signing_credentials = service_account.Credentials(
            signer=signer,
            service_account_email=service_account_email,
            token_uri="https://oauth2.googleapis.com/token",
        )
        return self.bucket.blob(object_name).generate_signed_url(
            version="v4",
            expiration=expires,
            method="GET",
            credentials=signing_credentials,
        )

    def ingest_openai_file(
        self,
        execution_id: str,
        file_payload: dict[str, Any],
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

        max_bytes = int(os.environ.get("GROUNDED_MOTION_MAX_INPUT_BYTES", str(200 * 1024 * 1024)))
        file_name = Path(str(file_payload.get("file_name") or "source.mp4")).name
        suffix = Path(file_name).suffix.lower() or ".mp4"
        object_name = f"executions/{execution_id}/inputs/source{suffix}"
        content_type = str(
            file_payload.get("mime_type")
            or mimetypes.guess_type(file_name)[0]
            or "application/octet-stream"
        )
        with tempfile.NamedTemporaryFile(prefix="grounded-motion-upload-", suffix=suffix) as handle:
            total = 0
            with requests.get(download_url, stream=True, timeout=(10, 120)) as response:
                response.raise_for_status()
                declared = int(response.headers.get("content-length", "0") or 0)
                if declared > max_bytes:
                    raise ValueError(
                        f"source_file exceeds the {max_bytes} byte input limit"
                    )
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError(
                            f"source_file exceeds the {max_bytes} byte input limit"
                        )
                    handle.write(chunk)
            handle.flush()
            if total == 0:
                raise ValueError("source_file is empty")
            artifact = self.upload(Path(handle.name), object_name, content_type=content_type)
        return {
            **artifact,
            "file_id": str(file_payload["file_id"]),
            "file_name": file_name,
            "mime_type": content_type,
        }


@dataclass(frozen=True)
class CloudRunJobLauncher:
    project: str
    region: str
    job: str

    @classmethod
    def from_env(cls) -> CloudRunJobLauncher:
        return cls(
            project=required_env("GOOGLE_CLOUD_PROJECT"),
            region=os.environ.get("GROUNDED_MOTION_REGION", "us-central1"),
            job=os.environ.get("GROUNDED_MOTION_JOB", "grounded-motion-vanguard-canary"),
        )

    def launch(self, execution_id: str) -> dict[str, Any]:
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
                    "timeout": "3600s",
                }
            },
            timeout=30,
        )
        if response.status_code >= 300:
            raise RuntimeError(
                f"Cloud Run Job launch failed ({response.status_code}): {response.text[:1000]}"
            )
        payload = response.json()
        return {
            "operation": payload.get("name", ""),
            "job": name,
        }


class VanguardCanaryController:
    def __init__(
        self,
        store: GcsCanaryStore | Any | None = None,
        launcher: CloudRunJobLauncher | Any | None = None,
    ) -> None:
        self.store = store or GcsCanaryStore()
        self.launcher = launcher or CloudRunJobLauncher.from_env()

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
        self.store.write_json(
            execution_status_object(execution_id), status, if_generation_match=0
        )
        try:
            launch = self.launcher.launch(execution_id)
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

    def start_motion_tracking(
        self,
        source_file: dict[str, Any],
        *,
        crop: list[int] | None = None,
        minimum_score: float = 0.5,
    ) -> dict[str, Any]:
        if crop is not None and (
            len(crop) != 4
            or any(not isinstance(value, int) for value in crop)
            or crop[2] <= 0
            or crop[3] <= 0
        ):
            raise ValueError("crop must be [x, y, width, height] with positive dimensions")
        if not 0 <= minimum_score <= 10:
            raise ValueError("minimum_score must be between 0 and 10")

        execution_id = str(uuid.uuid4())
        now = time.time()
        self.store.acquire_execution(execution_id, now)
        revision = os.environ.get("GROUNDED_MOTION_REVISION", "unknown")
        image_digest = os.environ.get("GROUNDED_MOTION_IMAGE_DIGEST", "unknown")
        status = {
            "schema": "grounded-motion-status/v1",
            "execution_id": execution_id,
            "kind": "uploaded-track",
            "state": "queued",
            "stage": "ingesting-source-file",
            "pipeline_pass": False,
            "revision": revision,
            "image_digest": image_digest,
            "created_unix": now,
            "updated_unix": now,
        }
        self.store.write_json(
            execution_status_object(execution_id), status, if_generation_match=0
        )
        try:
            source = self.store.ingest_openai_file(execution_id, source_file)
            spec = {
                "schema": "grounded-motion-job-spec/v1",
                "execution_id": execution_id,
                "kind": "uploaded-track",
                "source": source,
                "crop": crop,
                "minimum_score": minimum_score,
                "created_unix": now,
            }
            self.store.write_json(
                execution_job_spec_object(execution_id), spec, if_generation_match=0
            )
            status.update(stage="launching-gpu-job", updated_unix=time.time())
            self.store.write_json(execution_status_object(execution_id), status)
            launch = self.launcher.launch(execution_id)
        except Exception as exc:
            status.update(
                state="failed",
                stage="input-or-gpu-launch-failed",
                error=str(exc),
                updated_unix=time.time(),
            )
            self.store.write_json(execution_status_object(execution_id), status)
            self.store.set_lock_state(execution_id, "failed")
            raise
        return {
            "execution_id": execution_id,
            "kind": "uploaded-track",
            "state": "queued",
            "status_tool": "get_motion_status",
            "result_tool": "get_motion_result",
            "source": {
                key: source[key]
                for key in ("file_id", "file_name", "mime_type", "sha256", "size_bytes")
            },
            "revision": revision,
            "image_digest": image_digest,
            "launch": launch,
        }

    def status(self, execution_id: str) -> dict[str, Any]:
        execution_id = validate_execution_id(execution_id)
        status, _ = self.store.read_json(execution_status_object(execution_id))
        return status

    def result(self, execution_id: str) -> dict[str, Any]:
        execution_id = validate_execution_id(execution_id)
        status = self.status(execution_id)
        if status.get("state") not in TERMINAL_STATES:
            return {
                "execution_id": execution_id,
                "state": status.get("state"),
                "ready": False,
                "stage": status.get("stage"),
            }
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
        result["result_url"] = self.store.signed_url(
            execution_result_object(execution_id)
        )
        result["status_url"] = self.store.signed_url(
            execution_status_object(execution_id)
        )
        result["control_urls_expire_hours"] = 24
        result["ready"] = True
        return result
