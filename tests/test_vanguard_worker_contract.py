from __future__ import annotations

import json
from pathlib import Path

from grounded_motion_mcp.vanguard_worker import _publish_tree


class FakeStore:
    def upload(self, source, object_name, content_type=None):
        return {
            "object": object_name,
            "gs_uri": f"gs://fixture/{object_name}",
            "sha256": "hash",
            "size_bytes": source.stat().st_size,
            "generation": 1,
            "content_type": content_type or "application/octet-stream",
        }


def test_publish_tree_keeps_lane_and_object_identity(tmp_path: Path):
    (tmp_path / "overlay.mp4").write_bytes(b"video")
    (tmp_path / "pose-track.json").write_text(json.dumps({"schema": "track"}))
    artifacts = _publish_tree(FakeStore(), "execution", "source", tmp_path)
    assert [item["name"] for item in artifacts] == ["overlay.mp4", "pose-track.json"]
    assert all(item["lane"] == "source" for item in artifacts)
    assert artifacts[0]["object"] == "executions/execution/source/overlay.mp4"
