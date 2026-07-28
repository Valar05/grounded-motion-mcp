"""Authenticated three-tool production MCP surface for ChatGPT."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from mcp.server import MCPServer
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.types import ToolAnnotations
from starlette.responses import JSONResponse

from .vanguard_cloud import ALLOWED_EMAIL, CANARY_SCOPE, VanguardCanaryController


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
            "Return structured pipeline and mechanical findings plus 24-hour signed evidence URLs. "
            "pipeline_pass proves real inference and verified artifacts; mechanical_pass may honestly be false."
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


def main() -> None:
    server = create_production_server()
    server.run(
        transport="streamable-http",
        streamable_http_path="/mcp",
        stateless_http=True,
        json_response=True,
        host=os.environ.get("GROUNDED_MOTION_HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", os.environ.get("GROUNDED_MOTION_PORT", "8080"))),
    )
