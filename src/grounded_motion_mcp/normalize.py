"""Normalize raw COCO-WholeBody predictions without smoothing or invention."""

from __future__ import annotations

import math
from typing import Any

from .backend import RawFrame, RawInstance
from .constants import (
    COCO_WHOLEBODY_NAMES,
    COCO_WHOLEBODY_SIGMAS,
    SCHEMA,
    TRACK_SET_SCHEMA,
    TRACK_V3_SCHEMA,
)


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


def _instance_sort_key(item: tuple[int, RawInstance]) -> tuple[float, ...]:
    raw_index, instance = item
    bbox = instance.bbox or []
    if len(bbox) >= 4:
        return tuple(float(value) for value in bbox[:4]) + (float(raw_index),)
    usable = [pair for pair in instance.keypoints if len(pair) >= 2]
    if usable:
        x = sum(float(pair[0]) for pair in usable) / len(usable)
        y = sum(float(pair[1]) for pair in usable) / len(usable)
        return (x, y, x, y, float(raw_index))
    return (float("inf"), float("inf"), float("inf"), float("inf"), float(raw_index))


def _instance_area(instance: RawInstance) -> float:
    bbox = instance.bbox or []
    if len(bbox) >= 4:
        width = max(0.0, float(bbox[2]) - float(bbox[0]))
        height = max(0.0, float(bbox[3]) - float(bbox[1]))
        if width > 0 and height > 0:
            return width * height
    xs = [float(pair[0]) for pair in instance.keypoints if len(pair) >= 2]
    ys = [float(pair[1]) for pair in instance.keypoints if len(pair) >= 2]
    if not xs or not ys:
        return 1.0
    return max(1.0, (max(xs) - min(xs)) * (max(ys) - min(ys)))


def wholebody_oks(left: RawInstance, right: RawInstance) -> float:
    """Return COCO-WholeBody OKS while preserving backend-native score semantics."""
    if len(left.keypoints) != 133 or len(right.keypoints) != 133:
        return 0.0
    area = (_instance_area(left) + _instance_area(right)) / 2.0
    values: list[float] = []
    for index, sigma in enumerate(COCO_WHOLEBODY_SIGMAS):
        if index >= len(left.keypoint_scores) or index >= len(right.keypoint_scores):
            continue
        left_score = float(left.keypoint_scores[index])
        right_score = float(right.keypoint_scores[index])
        if not math.isfinite(left_score) or not math.isfinite(right_score):
            continue
        if left_score <= 0 or right_score <= 0:
            continue
        left_pair = left.keypoints[index]
        right_pair = right.keypoints[index]
        if len(left_pair) < 2 or len(right_pair) < 2:
            continue
        dx = float(right_pair[0]) - float(left_pair[0])
        dy = float(right_pair[1]) - float(left_pair[1])
        variance = (sigma * 2.0) ** 2
        exponent = (dx * dx + dy * dy) / variance / max(area, 1e-12) / 2.0
        values.append(math.exp(-exponent))
    return sum(values) / len(values) if values else 0.0


def bbox_iou(left: RawInstance, right: RawInstance) -> float:
    a = left.bbox or []
    b = right.bbox or []
    if len(a) < 4 or len(b) < 4:
        return 0.0
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(a[2]) - float(a[0])) * max(
        0.0, float(a[3]) - float(a[1])
    )
    right_area = max(0.0, float(b[2]) - float(b[0])) * max(
        0.0, float(b[3]) - float(b[1])
    )
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _v3_landmarks(instance: RawInstance, observation_id: str) -> dict[str, dict[str, object]]:
    if len(instance.keypoints) != len(COCO_WHOLEBODY_NAMES):
        raise ValueError(
            f"{observation_id} has {len(instance.keypoints)} keypoints; expected 133"
        )
    if len(instance.keypoint_scores) != len(COCO_WHOLEBODY_NAMES):
        raise ValueError(
            f"{observation_id} has {len(instance.keypoint_scores)} scores; expected 133"
        )
    result: dict[str, dict[str, object]] = {}
    for name, pair, raw_score in zip(
        COCO_WHOLEBODY_NAMES, instance.keypoints, instance.keypoint_scores
    ):
        score = float(raw_score)
        if not math.isfinite(score) or score < 0:
            raise ValueError(f"{observation_id} {name} has invalid detector score {score!r}")
        result[name] = {
            "x": float(pair[0]),
            "y": float(pair[1]),
            "score": score,
            "detector_score": score,
            "origin": "detector",
            "observation_id": observation_id,
        }
    return result


def _new_segment(
    *,
    number: int,
    track_set_id: str,
    source: dict[str, Any],
    backend: dict[str, Any],
    raw_path: str,
    raw_sha256: str,
    minimum_score: float,
) -> dict[str, Any]:
    subject_id = f"subject-{number:04d}"
    return {
        "schema": TRACK_V3_SCHEMA,
        "track_id": f"{track_set_id}:{subject_id}",
        "subject_id": subject_id,
        "state": "tracked",
        "source": source,
        "backend": backend,
        "score_semantics": backend.get(
            "score_semantics", "backend-native-keypoint-score"
        ),
        "score_calibrated": bool(backend.get("score_calibrated", False)),
        "minimum_score": minimum_score,
        "presence_intervals": [],
        "review": {
            "status": "unreviewed",
            "required_groups": ["hips", "feet", "wrists"],
            "identity_continuity_reviewed": False,
        },
        "raw_predictions": {"path": raw_path, "sha256": raw_sha256},
        "events": [],
        "frames": [],
    }


def normalize_track_set(
    *,
    track_set_id: str,
    source: dict[str, Any],
    backend: dict[str, Any],
    raw_frames: list[RawFrame],
    raw_path: str,
    raw_sha256: str,
    minimum_score: float,
    oks_threshold: float = 0.30,
    ambiguity_margin: float = 0.05,
) -> dict[str, Any]:
    """Normalize all people into conservative, reviewable identity segments.

    Association is limited to consecutive frames. Ambiguous matches, detector
    gaps, exits, and re-entry create new segments; review may merge them later.
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:  # pragma: no cover - inference image contract
        raise RuntimeError("scipy is required for deterministic multi-person assignment") from exc

    segments: dict[str, dict[str, Any]] = {}
    active: dict[str, RawInstance] = {}
    next_segment = 1
    findings: list[dict[str, Any]] = []
    frame_index: list[dict[str, Any]] = []

    for raw_frame in raw_frames:
        ordered = sorted(enumerate(raw_frame.instances), key=_instance_sort_key)
        observations: list[tuple[str, RawInstance, int]] = []
        for order, (raw_index, instance) in enumerate(ordered):
            observation_id = f"{raw_frame.frame_id}:o{order:04d}"
            observations.append((observation_id, instance, raw_index))

        active_ids = sorted(active)
        scores = [
            [wholebody_oks(active[segment_id], instance) for _, instance, _ in observations]
            for segment_id in active_ids
        ]
        ious = [
            [bbox_iou(active[segment_id], instance) for _, instance, _ in observations]
            for segment_id in active_ids
        ]
        accepted: dict[int, str] = {}
        if scores and observations:
            import numpy as np

            # The tiny IoU term is only a deterministic tie-break. Admission
            # and ambiguity decisions use the unmodified OKS values.
            utility = np.asarray(scores, dtype=float) + np.asarray(ious, dtype=float) * 1e-9
            row_indices, column_indices = linear_sum_assignment(-utility)
            for row, column in zip(row_indices.tolist(), column_indices.tolist()):
                score = scores[row][column]
                if score < oks_threshold:
                    continue
                row_candidates = sorted(scores[row], reverse=True)
                column_candidates = sorted(
                    (scores[candidate_row][column] for candidate_row in range(len(active_ids))),
                    reverse=True,
                )
                row_gap = score - row_candidates[1] if len(row_candidates) > 1 else 1.0
                column_gap = score - column_candidates[1] if len(column_candidates) > 1 else 1.0
                if row_gap < ambiguity_margin or column_gap < ambiguity_margin:
                    findings.append(
                        {
                            "code": "identity-ambiguous",
                            "frame_id": raw_frame.frame_id,
                            "previous_subject_id": active_ids[row],
                            "observation_id": observations[column][0],
                            "oks": score,
                            "row_gap": row_gap,
                            "column_gap": column_gap,
                        }
                    )
                    continue
                accepted[column] = active_ids[row]

        next_active: dict[str, RawInstance] = {}
        observation_ids: list[str] = []
        for column, (observation_id, instance, raw_index) in enumerate(observations):
            subject_id = accepted.get(column)
            if subject_id is None:
                segment = _new_segment(
                    number=next_segment,
                    track_set_id=track_set_id,
                    source=source,
                    backend=backend,
                    raw_path=raw_path,
                    raw_sha256=raw_sha256,
                    minimum_score=minimum_score,
                )
                next_segment += 1
                subject_id = str(segment["subject_id"])
                segments[subject_id] = segment
            segment = segments[subject_id]
            segment["frames"].append(
                {
                    "id": raw_frame.frame_id,
                    "index": raw_frame.frame_index,
                    "time_seconds": raw_frame.time_seconds,
                    "observation_id": observation_id,
                    "raw_instance_index": raw_index,
                    "bbox": instance.bbox,
                    "subject_score": instance.bbox_score,
                    "landmarks": _v3_landmarks(instance, observation_id),
                }
            )
            next_active[subject_id] = instance
            observation_ids.append(observation_id)
        active = next_active
        frame_index.append(
            {
                "id": raw_frame.frame_id,
                "index": raw_frame.frame_index,
                "time_seconds": raw_frame.time_seconds,
                "observation_ids": observation_ids,
            }
        )

    for segment in segments.values():
        frames = segment["frames"]
        if frames:
            segment["presence_intervals"] = [
                {
                    "start_frame": frames[0]["id"],
                    "end_frame": frames[-1]["id"],
                    "observed_frame_count": len(frames),
                }
            ]

    return {
        "schema": TRACK_SET_SCHEMA,
        "track_set_id": track_set_id,
        "state": "tracked",
        "source": source,
        "backend": backend,
        "minimum_score": minimum_score,
        "association": {
            "method": "consecutive-frame-hungarian-coco-wholebody-oks",
            "oks_threshold": oks_threshold,
            "ambiguity_margin": ambiguity_margin,
            "tie_break": "bbox-iou-then-stable-observation-order",
            "automatic_reidentification": False,
            "sigma_source": "mmpose-1.3.2-coco-wholebody",
        },
        "review": {
            "status": "unreviewed",
            "included_subject_ids": sorted(segments),
            "excluded_subjects": [],
        },
        "raw_predictions": {"path": raw_path, "sha256": raw_sha256},
        "frame_index": frame_index,
        "identity_findings": findings,
        "subjects": [segments[key] for key in sorted(segments)],
    }
