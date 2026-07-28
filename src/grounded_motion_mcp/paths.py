"""Workspace path containment and atomic job helpers."""

from __future__ import annotations

import os
from pathlib import Path


class PathBoundaryError(ValueError):
    pass


def default_workspace() -> Path:
    return Path(os.environ.get("GROUNDED_MOTION_WORKSPACE", Path.cwd())).expanduser().resolve()


def ensure_within(path: Path, root: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = root.expanduser().resolve()
    if not resolved.is_relative_to(root):
        raise PathBoundaryError(f"Path is outside workspace root: {resolved}")
    return resolved


def resolve_source(source_path: str, workspace: Path) -> Path:
    candidate = Path(source_path).expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    candidate = ensure_within(candidate, workspace)
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def resolve_output(path: str | None, workspace: Path, default: Path) -> Path:
    if path is None:
        return ensure_within(default, workspace)
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    return ensure_within(candidate, workspace)

