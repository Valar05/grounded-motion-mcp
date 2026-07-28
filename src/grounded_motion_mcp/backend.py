"""Detector backend protocol and raw prediction value objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from .models import Crop


@dataclass
class RawInstance:
    keypoints: list[list[float]]
    keypoint_scores: list[float]
    bbox: list[float] | None = None
    bbox_score: float | None = None


@dataclass
class RawFrame:
    frame_id: str
    frame_index: int
    time_seconds: float
    instances: list[RawInstance]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class PoseBackend(Protocol):
    @property
    def receipt(self) -> dict[str, object]: ...

    def infer(
        self,
        frames: list[Path],
        fps: float,
        crop: Crop,
        overlay_dir: Path,
    ) -> list[RawFrame]: ...
