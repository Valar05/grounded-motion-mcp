#!/usr/bin/env python3
"""Launch the real STDIO subprocess and verify protocol-clean tool discovery."""

from __future__ import annotations

import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

EXPECTED = {
    "track_motion",
    "validate_track",
    "inspect_track",
    "compare_motion",
    "export_artifacts",
}


async def run() -> None:
    environment = os.environ.copy()
    environment["GROUNDED_MOTION_TRANSPORT"] = "stdio"
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "grounded_motion_mcp.server"],
        env=environment,
        cwd=os.getcwd(),
    )
    async with (
        stdio_client(parameters) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        result = await session.list_tools()
        names = {tool.name for tool in result.tools}
    missing = EXPECTED - names
    if missing:
        raise SystemExit(f"Missing STDIO MCP tools: {sorted(missing)}")
    print(f"STDIO smoke passed: {', '.join(sorted(EXPECTED))}")


if __name__ == "__main__":
    asyncio.run(run())
