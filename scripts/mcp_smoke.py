#!/usr/bin/env python3
"""In-memory MCP smoke test for all required tool registrations."""

from __future__ import annotations

import asyncio

from mcp import Client

from grounded_motion_mcp.server import mcp

EXPECTED = {
    "track_motion",
    "validate_track",
    "inspect_track",
    "compare_motion",
    "export_artifacts",
}


async def run() -> None:
    async with Client(mcp) as client:
        tools = await client.list_tools()
        names = {tool.name for tool in tools.tools}
        missing = EXPECTED - names
        if missing:
            raise SystemExit(f"Missing MCP tools: {sorted(missing)}")
        print(f"MCP smoke passed: {', '.join(sorted(EXPECTED))}")


if __name__ == "__main__":
    asyncio.run(run())
