"""Artifact manifests, receipts, and deterministic exports."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Any

from .constants import MANIFEST_SCHEMA
from .hashing import sha256_file, write_json


def build_manifest(job_dir: Path, names: list[str]) -> dict[str, Any]:
    artifacts = []
    for name in sorted(names):
        path = job_dir / name
        if not path.is_file():
            continue
        artifacts.append(
            {
                "path": name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema": MANIFEST_SCHEMA,
        "job_id": job_dir.name,
        "artifacts": artifacts,
    }


def write_manifest(job_dir: Path, names: list[str]) -> Path:
    manifest = build_manifest(job_dir, names)
    path = job_dir / "manifest.json"
    write_json(path, manifest)
    return path


def verify_manifest(job_dir: Path) -> dict[str, Any]:
    from .hashing import read_json

    path = job_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = read_json(path)
    errors = []
    for artifact in manifest.get("artifacts", []):
        artifact_path = job_dir / artifact["path"]
        if not artifact_path.is_file():
            errors.append({"path": artifact["path"], "error": "missing"})
            continue
        actual = sha256_file(artifact_path)
        if actual != artifact.get("sha256"):
            errors.append(
                {
                    "path": artifact["path"],
                    "error": "sha256",
                    "expected": artifact.get("sha256"),
                    "actual": actual,
                }
            )
    return {"pass": not errors, "errors": errors, "manifest": manifest}


def export_job(job_dir: Path, destination: Path) -> Path:
    verification = verify_manifest(job_dir)
    if not verification["pass"]:
        raise ValueError(f"Artifact manifest failed verification: {verification['errors']}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        names = ["manifest.json"] + [
            item["path"] for item in verification["manifest"].get("artifacts", [])
        ]
        for name in sorted(set(names)):
            path = job_dir / name
            info = zipfile.ZipInfo(f"{job_dir.name}/{name}")
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o644 & 0xFFFF) << 16
            archive.writestr(info, path.read_bytes())
    os.replace(temp, destination)
    return destination

