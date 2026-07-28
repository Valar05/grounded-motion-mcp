"""MCP server exposing the grounded-motion service."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from mcp.server import MCPServer

from .models import (
    CompareRequest,
    Crop,
    ExportRequest,
    InspectRequest,
    TrackRequest,
    ValidateRequest,
)
from .service import GroundedMotionService

mcp = MCPServer("grounded-motion-mcp")


def _service() -> GroundedMotionService:
    workspace = os.environ.get("GROUNDED_MOTION_WORKSPACE")
    return GroundedMotionService(
        workspace=Path(workspace).expanduser().resolve() if workspace else None
    )


@mcp.tool()
async def track_motion(
    source_path: str,
    crop_x: int | None = None,
    crop_y: int | None = None,
    crop_width: int | None = None,
    crop_height: int | None = None,
    device: str = "auto",
    model_preset: str = "rtmw-x-cocktail14-384x288",
    minimum_score: float = 0.5,
) -> dict[str, Any]:
    """Track every source frame with RTMW whole-body landmarks and emit review artifacts.

    The source must be inside GROUNDED_MOTION_WORKSPACE. Supply all four crop fields or none.
    Inference ends in tracked/unreviewed; it never implies motion acceptance.
    """
    crop_values = (crop_x, crop_y, crop_width, crop_height)
    if any(value is not None for value in crop_values) and not all(
        value is not None for value in crop_values
    ):
        raise ValueError("Supply all crop fields or none")
    crop = (
        Crop(x=crop_x, y=crop_y, width=crop_width, height=crop_height)
        if all(value is not None for value in crop_values)
        else None
    )
    request = TrackRequest(
        source_path=source_path,
        crop=crop,
        device=device,
        model_preset=model_preset,
        minimum_score=minimum_score,
    )
    return await asyncio.to_thread(_service().track_motion, request)


@mcp.tool()
async def validate_track(
    track_path: str,
    production: bool = True,
    report_path: str | None = None,
    trajectory_path: str | None = None,
) -> dict[str, Any]:
    """Validate schema, complete chronology, required joints, review state, and event lock."""
    request = ValidateRequest(
        track_path=track_path,
        production=production,
        report_path=report_path,
        trajectory_path=trajectory_path,
    )
    return await asyncio.to_thread(_service().validate_track, request)


@mcp.tool()
async def inspect_track(track_path: str) -> dict[str, Any]:
    """Summarize coverage gaps, uncertainty, events, review state, and gate results."""
    return await asyncio.to_thread(
        _service().inspect_track,
        InspectRequest(track_path=track_path),
    )


@mcp.tool()
async def compare_motion(
    source_track_path: str,
    candidate_track_path: str,
    report_path: str | None = None,
    trajectory_path: str | None = None,
) -> dict[str, Any]:
    """Compare candidate hips, feet, wrists, and detailed hands to source motion."""
    request = CompareRequest(
        source_track_path=source_track_path,
        candidate_track_path=candidate_track_path,
        report_path=report_path,
        trajectory_path=trajectory_path,
    )
    return await asyncio.to_thread(_service().compare_motion, request)


@mcp.tool()
async def export_artifacts(
    job_path: str,
    destination_path: str | None = None,
) -> dict[str, Any]:
    """Verify the immutable manifest and export a deterministic review bundle."""
    return await asyncio.to_thread(
        _service().export_artifacts,
        ExportRequest(job_path=job_path, destination_path=destination_path),
    )


def main() -> None:
    transport = os.environ.get("GROUNDED_MOTION_TRANSPORT", "stdio")
    if transport == "stdio":
        mcp.run(transport="stdio")
        return
    if transport != "streamable-http":
        raise ValueError("GROUNDED_MOTION_TRANSPORT must be stdio or streamable-http")
    mcp.run(
        transport="streamable-http",
        stateless_http=True,
        json_response=True,
        host=os.environ.get("GROUNDED_MOTION_HOST", "127.0.0.1"),
        port=int(os.environ.get("GROUNDED_MOTION_PORT", "8000")),
    )


if __name__ == "__main__":
    main()

