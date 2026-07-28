#!/usr/bin/env python3
"""Verify the immutable Vanguard PNGs and assemble non-interpolated canary videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from importlib.resources import files
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest() -> dict:
    resource = files("grounded_motion_mcp.data").joinpath("vanguard_canary.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def materialize_lane(lane: dict, frames_root: Path, output: Path, spec: dict) -> dict:
    frames = lane["frames"]
    repeats = spec["repeat_counts"]
    if len(frames) != len(repeats):
        raise ValueError("frame and repeat counts differ")
    with tempfile.TemporaryDirectory(prefix="grounded-motion-canary-") as temporary:
        sequence = Path(temporary)
        output_index = 0
        verified = []
        for frame, repeat in zip(frames, repeats, strict=True):
            source = frames_root / Path(frame["path"]).name
            if not source.is_file():
                raise FileNotFoundError(source)
            actual = sha256_file(source)
            if actual != frame["sha256"]:
                raise ValueError(f"source frame SHA-256 mismatch: {source}: {actual}")
            verified.append({"path": frame["path"], "sha256": actual})
            for _ in range(repeat):
                destination = sequence / f"frame-{output_index:06d}.png"
                shutil.copyfile(source, destination)
                output_index += 1
        if output_index != spec["frame_count"]:
            raise ValueError(f"expected {spec['frame_count']} frames, built {output_index}")
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-framerate", str(spec["fps"]),
                "-i", str(sequence / "frame-%06d.png"),
                "-frames:v", str(spec["frame_count"]),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(output),
            ],
            check=True,
        )
    return {
        "path": str(output),
        "sha256": sha256_file(output),
        "size_bytes": output.stat().st_size,
        "frame_count": spec["frame_count"],
        "fps": spec["fps"],
        "interpolation": False,
        "verified_frames": verified,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-frames", type=Path, required=True)
    parser.add_argument("--candidate-frames", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    fixture = manifest()
    spec = fixture["materialization"]
    result = {
        "schema": "grounded-motion-vanguard-materialization/v1",
        "fixture": fixture["name"],
        "source_revision": fixture["source"]["revision"],
        "candidate_revision": fixture["candidate"]["revision"],
        "source": materialize_lane(
            fixture["source"], args.source_frames,
            args.output_dir / "vanguard-walk-v1.mp4", spec,
        ),
        "candidate": materialize_lane(
            fixture["candidate"], args.candidate_frames,
            args.output_dir / "walk-sword-carry-v2-attempt-003.mp4", spec,
        ),
    }
    receipt = args.output_dir / "materialization.json"
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
