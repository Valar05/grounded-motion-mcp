"""Exact video inspection and frame decoding through ffprobe/ffmpeg."""

from __future__ import annotations

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

from PIL import Image

from .models import Crop


class VideoToolError(RuntimeError):
    pass


def require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise VideoToolError(f"Required executable is missing: {name}")
    return path


def inspect_video(path: Path) -> dict[str, Any]:
    ffprobe = require_binary("ffprobe")
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise VideoToolError(result.stderr.strip() or "ffprobe failed")
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise VideoToolError("No video stream found")
    stream = streams[0]
    rate_text = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    if not rate_text or rate_text == "0/0":
        raise VideoToolError("Video frame rate is unavailable")
    fps_fraction = Fraction(rate_text)
    duration = float(stream.get("duration") or payload.get("format", {}).get("duration") or 0)
    frame_count = int(stream.get("nb_frames") or round(duration * float(fps_fraction)))
    return {
        "codec": stream.get("codec_name"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(fps_fraction),
        "fps_rational": f"{fps_fraction.numerator}/{fps_fraction.denominator}",
        "duration_seconds": duration,
        "frame_count": frame_count,
    }


def validate_crop(crop: Crop | None, metadata: dict[str, Any]) -> Crop:
    if crop is None:
        return Crop(x=0, y=0, width=metadata["width"], height=metadata["height"])
    if crop.x + crop.width > metadata["width"] or crop.y + crop.height > metadata["height"]:
        raise ValueError("Crop extends outside the source frame")
    return crop


def decode_frames(path: Path, destination: Path) -> list[Path]:
    ffmpeg = require_binary("ffmpeg")
    destination.mkdir(parents=True, exist_ok=True)
    pattern = destination / "frame-%06d.png"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-vsync",
        "0",
        "-start_number",
        "0",
        str(pattern),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise VideoToolError(result.stderr.strip() or "ffmpeg frame decode failed")
    frames = sorted(destination.glob("frame-*.png"))
    if not frames:
        raise VideoToolError("Frame decode produced no images")
    return frames


def crop_frame(source: Path, destination: Path, crop: Crop) -> None:
    with Image.open(source) as image:
        cropped = image.crop((crop.x, crop.y, crop.x + crop.width, crop.y + crop.height))
        cropped.save(destination, format="PNG")


def encode_video(frames_pattern: Path, fps: str, destination: Path, slow_factor: float = 1.0) -> None:
    ffmpeg = require_binary("ffmpeg")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        fps,
        "-i",
        str(frames_pattern),
    ]
    if slow_factor != 1.0:
        command.extend(["-vf", f"setpts={slow_factor}*PTS"])
    command.extend(
        [
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ]
    )
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise VideoToolError(result.stderr.strip() or "ffmpeg overlay encode failed")

