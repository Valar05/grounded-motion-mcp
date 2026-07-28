from __future__ import annotations

import pytest
from mcp import Client

from grounded_motion_mcp.server import mcp


@pytest.mark.asyncio
async def test_required_mcp_tools_are_registered() -> None:
    async with Client(mcp) as client:
        result = await client.list_tools()
    assert {tool.name for tool in result.tools} == {
        "track_motion",
        "validate_track",
        "inspect_track",
        "compare_motion",
        "export_artifacts",
    }
