"""Explicit model acquisition with immutable local hashing."""

from __future__ import annotations

import os
import urllib.request
import uuid
from pathlib import Path
from urllib.parse import urlparse

from .hashing import sha256_file


def checkpoint_name(url: str) -> str:
    name = Path(urlparse(url).path).name
    if not name:
        raise ValueError(f"Checkpoint URL has no filename: {url}")
    return name


def acquire_checkpoint(url: str, cache_root: Path) -> tuple[Path, str]:
    cache_root.mkdir(parents=True, exist_ok=True)
    destination = cache_root / checkpoint_name(url)
    if not destination.is_file():
        temporary = cache_root / f".{destination.name}.{uuid.uuid4().hex}.tmp"
        try:
            with (
                urllib.request.urlopen(url, timeout=120) as response,
                temporary.open("wb") as output,
            ):
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination, sha256_file(destination)
