from __future__ import annotations

from pathlib import Path

import pytest

from grounded_motion_mcp.paths import PathBoundaryError, ensure_within


def test_path_inside_workspace_passes(tmp_path: Path) -> None:
    target = tmp_path / "child"
    assert ensure_within(target, tmp_path) == target.resolve()


def test_path_outside_workspace_fails(tmp_path: Path) -> None:
    with pytest.raises(PathBoundaryError):
        ensure_within(tmp_path.parent / "escape", tmp_path)

