#!/usr/bin/env python3
"""Launch the real Streamable HTTP server and list tools through the official client."""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time

from mcp import Client

PORT = 18765
EXPECTED = {
    "track_motion",
    "validate_track",
    "inspect_track",
    "compare_motion",
    "export_artifacts",
}


def wait_for_port(process: subprocess.Popen[bytes], timeout: float = 10) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"HTTP server exited early with code {process.returncode}")
        with socket.socket() as probe:
            probe.settimeout(0.2)
            if probe.connect_ex(("127.0.0.1", PORT)) == 0:
                return
        time.sleep(0.05)
    raise TimeoutError("HTTP server did not open its port")


async def list_tools() -> set[str]:
    async with Client(f"http://127.0.0.1:{PORT}/mcp") as client:
        result = await client.list_tools()
        return {tool.name for tool in result.tools}


def main() -> None:
    environment = os.environ.copy()
    for name in (
        "ALL_PROXY",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "all_proxy",
        "http_proxy",
        "https_proxy",
    ):
        environment.pop(name, None)
        os.environ.pop(name, None)
    environment.update(
        {
            "GROUNDED_MOTION_TRANSPORT": "streamable-http",
            "GROUNDED_MOTION_HOST": "127.0.0.1",
            "GROUNDED_MOTION_PORT": str(PORT),
        }
    )
    process = subprocess.Popen(
        [sys.executable, "-m", "grounded_motion_mcp.server"],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_port(process)
        names = asyncio.run(list_tools())
        missing = EXPECTED - names
        if missing:
            raise SystemExit(f"Missing HTTP MCP tools: {sorted(missing)}")
        print(f"HTTP smoke passed: {', '.join(sorted(EXPECTED))}")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


if __name__ == "__main__":
    main()

