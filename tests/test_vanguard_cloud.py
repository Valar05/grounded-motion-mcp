from __future__ import annotations

import asyncio
import json
import time
import uuid

import pytest

from grounded_motion_mcp.production_server import HomeCenterTokenVerifier
from grounded_motion_mcp.vanguard_cloud import (
    CANARY_SCOPE,
    VanguardCanaryController,
    load_canary_manifest,
)


class FakeStore:
    def __init__(self):
        self.values = {}
        self.lock = None

    def acquire_execution(self, execution_id, now):
        self.lock = {"execution_id": execution_id, "state": "queued", "updated_unix": now}

    def write_json(self, name, payload, if_generation_match=None):
        if if_generation_match == 0 and name in self.values:
            raise RuntimeError("duplicate")
        self.values[name] = dict(payload)
        return 1

    def read_json(self, name):
        return dict(self.values[name]), 1

    def set_lock_state(self, execution_id, state):
        if self.lock and self.lock["execution_id"] == execution_id:
            self.lock["state"] = state

    def signed_url(self, object_name):
        return f"https://signed.invalid/{object_name}"


class FakeLauncher:
    def launch(self, execution_id):
        return {"operation": f"operations/{execution_id}", "job": "jobs/canary"}


def test_manifest_keeps_exact_revisions_timing_and_no_interpolation():
    fixture = load_canary_manifest()
    assert fixture["source"]["revision"] == "90ca534c46a47c660e7bf5ef7bd2efcf35dbeb9e"
    assert fixture["candidate"]["revision"] == "b2c5bde5d91325726af34e5daea17b96d78b46f3"
    assert fixture["frame_timing_ms"] == [110, 90, 100, 110, 110, 90, 100, 110]
    assert fixture["materialization"] == {
        "fps": 100,
        "repeat_counts": [11, 9, 10, 11, 11, 9, 10, 11],
        "frame_count": 82,
        "interpolation": False,
    }
    assert len(fixture["source"]["frames"]) == 8
    assert len(fixture["candidate"]["frames"]) == 8
    assert fixture["candidate"]["review_status"] == "quarantined-human-verdict-pending"


def test_controller_starts_polls_and_signs_terminal_result(monkeypatch):
    monkeypatch.setenv("GROUNDED_MOTION_REVISION", "abc123")
    monkeypatch.setenv("GROUNDED_MOTION_IMAGE_DIGEST", "sha256:image")
    store = FakeStore()
    controller = VanguardCanaryController(store=store, launcher=FakeLauncher())
    started = controller.start()
    execution_id = started["execution_id"]
    uuid.UUID(execution_id)
    status = controller.status(execution_id)
    assert status["state"] == "queued"
    assert started["launch"]["job"] == "jobs/canary"

    store.values[f"executions/{execution_id}/status.json"] = {
        **status,
        "state": "completed",
        "pipeline_pass": True,
    }
    store.values[f"executions/{execution_id}/result.json"] = {
        "execution_id": execution_id,
        "state": "completed",
        "pipeline_pass": True,
        "mechanical_pass": False,
        "artifacts": [{"object": f"executions/{execution_id}/source/overlay.mp4"}],
    }
    result = controller.result(execution_id)
    assert result["pipeline_pass"] is True
    assert result["mechanical_pass"] is False
    assert result["artifacts"][0]["url"].startswith("https://signed.invalid/")
    assert result["artifacts"][0]["url_expires_hours"] == 24


def test_controller_result_is_not_green_while_running():
    store = FakeStore()
    controller = VanguardCanaryController(store=store, launcher=FakeLauncher())
    execution_id = controller.start()["execution_id"]
    result = controller.result(execution_id)
    assert result == {
        "execution_id": execution_id,
        "state": "queued",
        "ready": False,
        "stage": "launching-gpu-job",
    }


def test_production_server_exposes_only_three_canary_tools(monkeypatch):
    from grounded_motion_mcp.production_server import (
        add_chatgpt_security_schemes,
        create_production_server,
    )

    monkeypatch.setenv("GROUNDED_MOTION_RESOURCE", "https://motion.example.test/mcp")
    server = create_production_server()
    tools = asyncio.run(server.list_tools())
    assert [tool.name for tool in tools] == [
        "start_vanguard_canary",
        "get_vanguard_canary_status",
        "get_vanguard_canary_result",
    ]
    assert all(tool.meta["securitySchemes"] == [
        {"type": "oauth2", "scopes": [CANARY_SCOPE]}
    ] for tool in tools)
    wire_payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "tools": [
                tool.model_dump(mode="json", by_alias=True, exclude_none=True)
                for tool in tools
            ]
        },
    }
    assert add_chatgpt_security_schemes(wire_payload) is True
    assert all(tool["securitySchemes"] == [
        {"type": "oauth2", "scopes": [CANARY_SCOPE]}
    ] for tool in wire_payload["result"]["tools"])


def test_chatgpt_security_middleware_updates_the_actual_wire_response():
    from grounded_motion_mcp.production_server import ChatGPTToolSecurityMiddleware

    original_payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "tools": [{
                "name": "start_vanguard_canary",
                "inputSchema": {"type": "object", "properties": {}},
                "_meta": {"securitySchemes": [
                    {"type": "oauth2", "scopes": [CANARY_SCOPE]}
                ]},
            }]
        },
    }
    original_body = json.dumps(original_payload).encode()

    async def upstream(_scope, _receive, send):
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"application/json"), (b"content-length", b"1")],
        })
        split = len(original_body) // 2
        await send({"type": "http.response.body", "body": original_body[:split], "more_body": True})
        await send({"type": "http.response.body", "body": original_body[split:], "more_body": False})

    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    middleware = ChatGPTToolSecurityMiddleware(upstream)
    asyncio.run(middleware({"type": "http", "path": "/mcp"}, receive, send))

    assert [message["type"] for message in sent] == [
        "http.response.start",
        "http.response.body",
    ]
    body = sent[1]["body"]
    tool = json.loads(body)["result"]["tools"][0]
    assert tool["securitySchemes"] == [
        {"type": "oauth2", "scopes": [CANARY_SCOPE]}
    ]
    headers = dict(sent[0]["headers"])
    assert headers[b"content-length"] == str(len(body)).encode()


def test_home_center_verifier_binds_email_resource_issuer_and_scope(monkeypatch):
    issuer = "https://auth.example.test"
    resource = "https://motion.example.test/mcp"

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "email": "dclarke1005@gmail.com",
                "resource": resource,
                "issuer": issuer,
                "scopes": [CANARY_SCOPE],
                "client_id": "chatgpt-client",
                "expires_at": int(time.time()) + 300,
            }

    import requests

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response())
    token = asyncio.run(HomeCenterTokenVerifier(issuer, resource).verify_token("opaque"))
    assert token is not None
    assert token.resource == resource
    assert token.subject == "dclarke1005@gmail.com"
    assert token.scopes == [CANARY_SCOPE]


@pytest.mark.parametrize(
    "field,value",
    [
        ("email", "other@example.com"),
        ("resource", "https://wrong.example/mcp"),
        ("issuer", "https://wrong.example"),
        ("scopes", []),
    ],
)
def test_home_center_verifier_rejects_unbound_tokens(monkeypatch, field, value):
    issuer = "https://auth.example.test"
    resource = "https://motion.example.test/mcp"
    payload = {
        "email": "dclarke1005@gmail.com",
        "resource": resource,
        "issuer": issuer,
        "scopes": [CANARY_SCOPE],
    }
    payload[field] = value

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return payload

    import requests

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: Response())
    token = asyncio.run(HomeCenterTokenVerifier(issuer, resource).verify_token("opaque"))
    assert token is None
