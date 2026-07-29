"""Authenticated three-tool production MCP surface for ChatGPT."""

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
        title="Grounded Motion Vanguard Canary",
        description=(
            "Runs one immutable source-versus-candidate Vanguard motion tracking canary "
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
                "resource_name": "Grounded Motion Vanguard Canary",
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
