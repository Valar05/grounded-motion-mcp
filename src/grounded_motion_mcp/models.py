"""Pydantic models for tool inputs and normalized artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .constants import SCHEMA


class Crop(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class Landmark(BaseModel):
    x: float | None = None
    y: float | None = None
    score: float = Field(ge=0.0, le=1.0)
    visibility: float | None = Field(default=None, ge=0.0, le=1.0)
    origin: Literal["detector", "manual-source-witnessed", "occluded-unknown"] = "detector"

    @field_validator("x", "y")
    @classmethod
    def unknown_has_no_coordinates(cls, value: float | None) -> float | None:
        return value


class TrackFrame(BaseModel):
    id: str
    index: int = Field(ge=0)
    time_seconds: float = Field(ge=0.0)
    landmarks: dict[str, Landmark]
    subject_score: float | None = Field(default=None, ge=0.0, le=1.0)


class MotionEvent(BaseModel):
    type: str
    start_frame: str
    end_frame: str | None = None
    foot: Literal["left", "right"] | None = None
    evidence: str | None = None
    reviewed: bool = False


class PoseTrack(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    schema_version: str = Field(default=SCHEMA, alias="schema")
    track_id: str
    state: Literal[
        "tracked",
        "reviewed",
        "event-locked",
        "keyed",
        "transferred",
        "mechanically-compared",
        "human-accepted",
    ] = "tracked"
    source: dict[str, Any]
    backend: dict[str, Any]
    minimum_score: float = Field(default=0.5, ge=0.0, le=1.0)
    review: dict[str, Any] = Field(
        default_factory=lambda: {
            "status": "unreviewed",
            "required_groups": ["hips", "feet", "hands"],
        }
    )
    raw_predictions: dict[str, Any]
    events: list[MotionEvent] = Field(default_factory=list)
    frames: list[TrackFrame]


class TrackRequest(BaseModel):
    source_path: str
    workspace: str | None = None
    crop: Crop | None = None
    device: str = "auto"
    model_preset: str = "rtmw-x-cocktail14-384x288"
    minimum_score: float = Field(default=0.5, ge=0.0, le=1.0)
    overwrite_failed: bool = False

    @field_validator("source_path")
    @classmethod
    def source_must_be_file_like(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("source_path is required")
        return value


class ValidateRequest(BaseModel):
    track_path: str
    production: bool = True
    report_path: str | None = None
    trajectory_path: str | None = None


class InspectRequest(BaseModel):
    track_path: str


class CompareRequest(BaseModel):
    source_track_path: str
    candidate_track_path: str
    report_path: str | None = None
    trajectory_path: str | None = None


class ExportRequest(BaseModel):
    job_path: str
    destination_path: str | None = None


def expand_path(value: str) -> Path:
    return Path(value).expanduser().resolve()
