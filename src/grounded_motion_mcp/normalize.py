"""Normalize raw COCO-WholeBody predictions without smoothing or invention."""

from __future__ import annotations

import math
from typing import Any

from .backend import RawFrame, RawInstance
from .constants import COCO_WHOLEBODY_NAMES, SCHEMA


class SubjectAmbiguityError(RuntimeError):
    pass


def choose_subject(frame: RawFrame) -> RawInstance:
    if not frame.instances:
        raise SubjectAmbiguityError(f"No subject detected in {frame.frame_id}")
    if len(frame.instances) > 1:
        raise SubjectAmbiguityError(
            f"Multiple subjects detected in locked crop at {frame.frame_id}; "
            "tighten the crop instead of guessing identity"
        )
    return frame.instances[0]


def normalize_track(
    *,
    track_id: str,
    source: dict[str, Any],
    backend: dict[str, Any],
    raw_frames: list[RawFrame],
    raw_path: str,
    raw_sha256: str,
    minimum_score: float,
) -> dict[str, Any]:
    frames = []
    for raw_frame in raw_frames:
        subject = choose_subject(raw_frame)
        if len(subject.keypoints) != len(COCO_WHOLEBODY_NAMES):
            raise ValueError(
                f"{raw_frame.frame_id} has {len(subject.keypoints)} keypoints; expected 133"
            )
        if len(subject.keypoint_scores) != len(COCO_WHOLEBODY_NAMES):
            raise ValueError(
                f"{raw_frame.frame_id} has {len(subject.keypoint_scores)} scores; expected 133"
            )
        landmarks = {}
        for name, pair, score in zip(
            COCO_WHOLEBODY_NAMES,
            subject.keypoints,
            subject.keypoint_scores,
        ):
            detector_score = float(score)
            if not math.isfinite(detector_score) or detector_score < 0:
                raise ValueError(
                    f"{raw_frame.frame_id} {name} has invalid detector score "
                    f"{detector_score!r}"
                )
            landmarks[name] = {
                "x": float(pair[0]),
                "y": float(pair[1]),
                "score": detector_score,
                "origin": "detector",
            }
        frames.append(
            {
                "id": raw_frame.frame_id,
                "index": raw_frame.frame_index,
                "time_seconds": raw_frame.time_seconds,
                "subject_score": subject.bbox_score,
                "landmarks": landmarks,
            }
        )
    return {
        "schema": SCHEMA,
        "track_id": track_id,
        "state": "tracked",
        "source": source,
        "backend": backend,
        "score_semantics": backend.get(
            "score_semantics", "backend-native-keypoint-score"
        ),
        "score_calibrated": bool(backend.get("score_calibrated", False)),
        "minimum_score": minimum_score,
        "review": {
            "status": "unreviewed",
            "required_groups": ["hips", "feet", "hands"],
            "instruction": (
                "Review against source pixels; convert witnessed corrections to "
                "manual-source-witnessed and hidden joints to occluded-unknown."
            ),
        },
        "raw_predictions": {"path": raw_path, "sha256": raw_sha256},
        "events": [],
        "frames": frames,
    }

