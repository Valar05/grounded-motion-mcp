"""Authenticated production MCP surface for ChatGPT motion workflows."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from typing import Any

from mcp.server import MCPServer
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict
from starlette.responses import JSONResponse

from .vanguard_cloud import ALLOWED_EMAIL, CANARY_SCOPE, VanguardCanaryController

AsgiApp = Callable[
    [
        dict[str, Any],
        Callable[[], Awaitable[dict[str, Any]]],
        Callable[[dict[str, Any]], Awaitable[None]],
    ],
    Awaitable[None],
]


class OpenAIFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    download_url: str
    file_id: str
    mime_type: str | None = None
    file_name: str | None = None


class HomeCenterTokenVerifier:
    def __init__(self, issuer: str, resource: str):
        self.issuer = issuer.rstrip("/")
        self.resource = resource

    async def verify_token(self, token: str) -> AccessToken | None:
        import requests

        response = await asyncio.to_thread(
            requests.get,
            f"{self.issuer}/oauth/userinfo",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        if response.status_code != 200:
            return None
        try:
            payload = response.json()
        except ValueError:
            return None
        email = str(payload.get("email", "")).strip().lower()
        resource = str(payload.get("resource", ""))
        issuer = str(payload.get("issuer", ""))
        scopes = payload.get("scopes", [])
        if isinstance(scopes, str):
            scopes = scopes.split()
        if (
            email != ALLOWED_EMAIL
            or resource != self.resource
            or issuer != self.issuer
            or CANARY_SCOPE not in scopes
        ):
            return None
        return AccessToken(
            token=token,
            client_id=str(payload.get("client_id", "home-center-chatgpt")),
            scopes=[CANARY_SCOPE],
            expires_at=int(payload.get("expires_at", 0)) or None,
            resource=resource,
            subject=email,
            claims={"iss": issuer, "email": email, "aud": resource},
        )


def add_chatgpt_security_schemes(payload: Any) -> bool:
    """Mirror OAuth schemes onto ChatGPT's top-level tool descriptor field."""
    if not isinstance(payload, dict):
        return False
    result = payload.get("result")
    if not isinstance(result, dict):
        return False
    tools = result.get("tools")
    if not isinstance(tools, list):
        return False
    changed = False
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        meta = tool.get("_meta")
        schemes = meta.get("securitySchemes") if isinstance(meta, dict) else None
        if isinstance(schemes, list) and schemes:
            tool["securitySchemes"] = schemes
            changed = True
    return changed


class ChatGPTToolSecurityMiddleware:
    """Preserve ChatGPT auth metadata that the core MCP wire model omits."""

    def __init__(self, app: AsgiApp):
        self.app = app

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        if scope.get("type") != "http" or scope.get("path") != "/mcp":
            await self.app(scope, receive, send)
            return

        response_start: dict[str, Any] | None = None
        response_body: list[bytes] = []

        async def capture(message: dict[str, Any]) -> None:
            nonlocal response_start
            if message["type"] == "http.response.start":
                response_start = message
                return
            if message["type"] != "http.response.body":
                await send(message)
                return
            response_body.append(message.get("body", b""))
            if message.get("more_body", False):
                return

            body = b"".join(response_body)
            try:
                payload = json.loads(body)
            except (UnicodeDecodeError, json.JSONDecodeError):
                payload = None
            if add_chatgpt_security_schemes(payload):
                body = json.dumps(payload, separators=(",", ":")).encode()

            if response_start is not None:
                headers = [
                    (name, value)
                    for name, value in response_start.get("headers", [])
                    if name.lower() != b"content-length"
                ]
                headers.append((b"content-length", str(len(body)).encode()))
                await send({**response_start, "headers": headers})
            await send({"type": "http.response.body", "body": body, "more_body": False})

        await self.app(scope, receive, capture)


async def verify_internal_request(request: Any) -> dict[str, Any] | None:
    """Verify the Google OIDC token used only by Cloud Tasks callbacks."""
    authorization = str(request.headers.get("authorization", ""))
    if not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(None, 1)[1]
    audience = os.environ.get("GROUNDED_MOTION_INTERNAL_URL", "").rstrip("/")
    allowed = {
        item.strip().lower()
        for item in os.environ.get(
            "GROUNDED_MOTION_INTERNAL_SERVICE_ACCOUNTS",
            os.environ.get("GROUNDED_MOTION_CONTROL_SERVICE_ACCOUNT", ""),
        ).split(",")
        if item.strip()
    }
    if not audience or not allowed:
        return None
    try:
        from google.auth.transport.requests import Request
        from google.oauth2 import id_token

        claims = await asyncio.to_thread(
            id_token.verify_oauth2_token,
            token,
            Request(),
            audience,
        )
    except Exception:  # noqa: BLE001 - reject any token verification failure
        return None
    email = str(claims.get("email", "")).lower()
    if email not in allowed or claims.get("email_verified") is not True:
        return None
    return claims


def create_production_server() -> MCPServer:
    issuer = os.environ.get(
        "GROUNDED_MOTION_OAUTH_ISSUER",
        "https://us-central1-home-center-dclar.cloudfunctions.net/homeCenterMcp",
    ).rstrip("/")
    resource = os.environ.get("GROUNDED_MOTION_RESOURCE", "").rstrip("/")
    if not resource:
        raise RuntimeError("GROUNDED_MOTION_RESOURCE is required for the production profile")
    security = [{"type": "oauth2", "scopes": [CANARY_SCOPE]}]
    server = MCPServer(
        "grounded-motion-vanguard",
        title="Grounded Motion",
        description=(
            "Tracks user-supplied motion clips and runs an immutable Vanguard health canary "
            "on the real pinned RTMW MMPose backend. Tracking is evidence, not acceptance."
        ),
        version=os.environ.get("GROUNDED_MOTION_REVISION", "unknown"),
        auth=AuthSettings(
            issuer_url=issuer,
            resource_server_url=resource,
            required_scopes=[CANARY_SCOPE],
        ),
        token_verifier=HomeCenterTokenVerifier(issuer, resource),
    )

    @server.custom_route("/.well-known/oauth-protected-resource", methods=["GET"])
    async def protected_resource_metadata(_request: Any) -> JSONResponse:
        return JSONResponse(
            {
                "resource": resource,
                "resource_name": "Grounded Motion",
                "authorization_servers": [issuer],
                "scopes_supported": [CANARY_SCOPE],
                "bearer_methods_supported": ["header"],
            },
            headers={"Cache-Control": "no-store"},
        )

    @server.custom_route("/health", methods=["GET"])
    async def health(_request: Any) -> JSONResponse:
        return JSONResponse(
            {
                "ok": True,
                "service": "grounded-motion-vanguard",
                "revision": os.environ.get("GROUNDED_MOTION_REVISION", "unknown"),
                "resource": resource,
            },
            headers={"Cache-Control": "no-store"},
        )

    @server.custom_route("/internal/ingest", methods=["POST"])
    async def internal_ingest(request: Any) -> JSONResponse:
        if await verify_internal_request(request) is None:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        try:
            payload = await request.json()
            result = await asyncio.to_thread(
                VanguardCanaryController().ingest_attachment,
                str(payload["execution_id"]),
            )
            return JSONResponse(result)
        except Exception as exc:  # noqa: BLE001 - task boundary serializes failures
            # Controller records a terminal ingest failure; acknowledge the task so
            # Cloud Tasks does not repeat a rejected or hash-mismatched upload.
            return JSONResponse({"ok": False, "error": str(exc)})

    @server.custom_route("/internal/dispatch", methods=["POST"])
    async def internal_dispatch(request: Any) -> JSONResponse:
        if await verify_internal_request(request) is None:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        try:
            result = await asyncio.to_thread(VanguardCanaryController().dispatch_next)
            return JSONResponse(result)
        except Exception as exc:  # noqa: BLE001 - task retry requires an HTTP failure
            return JSONResponse({"error": str(exc)}, status_code=500)

    @server.tool(
        title="Track a motion clip",
        description=(
            "Start pinned whole-body GPU tracking for a user-provided video. Use this for "
            "actual source motion; the Vanguard canary is only a fixed health check. Returns "
            "an execution id for status and result polling."
        ),
        annotations=ToolAnnotations(
            title="Track a motion clip",
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=True,
        ),
        meta={
            "securitySchemes": security,
            "openai/fileParams": ["source_file"],
        },
    )
    async def start_motion_tracking(
        source_file: OpenAIFile,
        expected_sha256: str | None = None,
        crop: list[int] | None = None,
        minimum_score: float = 0.5,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            VanguardCanaryController().start_motion_tracking,
            source_file.model_dump(exclude_none=True),
            expected_sha256=expected_sha256,
            crop=crop,
            minimum_score=minimum_score,
            request_id=request_id,
        )

    @server.tool(
        title="Create a hash-locked MP4 upload",
        description=(
            "Create a one-use 15-minute signed PUT URL for a declared MP4 byte length and "
            "SHA-256. Call finalize_motion_upload after uploading the exact bytes."
        ),
        annotations=ToolAnnotations(
            title="Create a hash-locked MP4 upload",
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=True,
        ),
        meta={"securitySchemes": security},
    )
    async def create_motion_upload(
        file_name: str,
        size_bytes: int,
        sha256: str,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            VanguardCanaryController().create_motion_upload,
            file_name=file_name,
            size_bytes=size_bytes,
            sha256=sha256,
            request_id=request_id,
        )

    @server.tool(
        title="Finalize a hash-locked MP4 upload",
        description=(
            "Lock the uploaded GCS generation, run CPU SHA/MP4/full-decode preflight, and "
            "enqueue one asynchronous multi-person RTMW GPU execution."
        ),
        annotations=ToolAnnotations(
            title="Finalize a hash-locked MP4 upload",
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=True,
        ),
        meta={"securitySchemes": security},
    )
    async def finalize_motion_upload(
        execution_id: str,
        crop: list[int] | None = None,
        minimum_score: float = 0.5,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            VanguardCanaryController().finalize_motion_upload,
            execution_id,
            crop=crop,
            minimum_score=minimum_score,
            request_id=request_id,
        )

    @server.tool(
        title="Get motion tracking status",
        description="Poll one uploaded motion tracking execution until it reaches a terminal state.",
        annotations=ToolAnnotations(
            title="Get motion tracking status",
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=True,
        ),
        meta={"securitySchemes": security},
    )
    async def get_motion_status(execution_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(VanguardCanaryController().status, execution_id)

    @server.tool(
        title="Get motion tracking result",
        description=(
            "Return the uploaded clip's normalized whole-body track and fresh 24-hour signed "
            "URLs for raw predictions, overlays, trajectories, manifests, and receipts."
        ),
        annotations=ToolAnnotations(
            title="Get motion tracking result",
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=True,
        ),
        meta={"securitySchemes": security},
    )
    async def get_motion_result(execution_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(VanguardCanaryController().result, execution_id)

    @server.tool(
        title="Submit human motion review",
        description=(
            "Submit explicit human identity/interval attestations, exclusions, event maps, "
            "quarantines, and sparse landmark corrections against an immutable track-set hash. "
            "Detector confidence cannot be edited."
        ),
        annotations=ToolAnnotations(
            title="Submit human motion review",
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
        meta={"securitySchemes": security},
    )
    async def submit_motion_review(
        execution_id: str,
        tracked_evidence_sha256: str,
        subjects: list[dict[str, Any]],
        excluded_subjects: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            VanguardCanaryController().submit_motion_review,
            execution_id,
            tracked_evidence_sha256=tracked_evidence_sha256,
            subjects=subjects,
            excluded_subjects=excluded_subjects,
            request_id=request_id,
            reviewer=ALLOWED_EMAIL,
        )

    @server.tool(
        title="Start Vanguard canary",
        description=(
            "Start one fresh asynchronous GPU canary comparing canonical Vanguard Walk v1 "
            "with quarantined WalkSwordCarryV2 candidate 003. Returns an execution id; "
            "never implies human acceptance."
        ),
        annotations=ToolAnnotations(
            title="Start Vanguard canary",
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=True,
        ),
        meta={"securitySchemes": security},
    )
    async def start_vanguard_canary() -> dict[str, Any]:
        return await asyncio.to_thread(VanguardCanaryController().start)

    @server.tool(
        title="Get Vanguard canary status",
        description="Poll the real Cloud Run GPU execution state for a Vanguard canary execution id.",
        annotations=ToolAnnotations(
            title="Get Vanguard canary status",
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=True,
        ),
        meta={"securitySchemes": security},
    )
    async def get_vanguard_canary_status(execution_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(VanguardCanaryController().status, execution_id)

    @server.tool(
        title="Get Vanguard canary result",
        description=(
            "Return v2 pipeline, judgment, registration, root-relative, quarantine findings, "
            "and complete 24-hour signed evidence URLs. pipeline_pass proves real inference "
            "and verified artifacts; blocked judgment returns mechanical_pass null."
        ),
        annotations=ToolAnnotations(
            title="Get Vanguard canary result",
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=True,
        ),
        meta={"securitySchemes": security},
    )
    async def get_vanguard_canary_result(execution_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(VanguardCanaryController().result, execution_id)

    return server


def create_production_app() -> AsgiApp:
    host = os.environ.get("GROUNDED_MOTION_HOST", "0.0.0.0")
    server = create_production_server()
    app = server.streamable_http_app(
        streamable_http_path="/mcp",
        stateless_http=True,
        json_response=True,
        host=host,
    )
    return ChatGPTToolSecurityMiddleware(app)


def main() -> None:
    import uvicorn

    uvicorn.run(
        create_production_app(),
        host=os.environ.get("GROUNDED_MOTION_HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", os.environ.get("GROUNDED_MOTION_PORT", "8080"))),
        log_level="info",
    )
