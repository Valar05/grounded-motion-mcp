"""Human-only review promotion over immutable multi-person detector evidence."""

from __future__ import annotations

import copy
import math
import time
from typing import Any

from .audit import validate_track
from .constants import (
    COCO_WHOLEBODY_NAMES,
    EVENT_MAP_SCHEMA,
    REQUIRED_GROUPS,
    REVIEW_SCHEMA,
    STEP_EVENTS,
    TRACK_SET_SCHEMA,
)


class ReviewValidationError(ValueError):
    pass


def _required_landmarks() -> list[str]:
    return REQUIRED_GROUPS["hips"] + REQUIRED_GROUPS["feet"] + REQUIRED_GROUPS["wrists"]


def _validate_attestations(review: dict[str, Any]) -> None:
    for field in (
        "source_interval_complete",
        "identity_continuity_reviewed",
        "required_landmarks_reviewed",
        "event_map_complete",
    ):
        if review.get(field) is not True:
            raise ReviewValidationError(f"review attestation is required: {field}")


def _merge_segments(
    track_set: dict[str, Any],
    segment_ids: list[str],
) -> dict[str, Any]:
    by_id = {str(item["subject_id"]): item for item in track_set.get("subjects", [])}
    if not segment_ids:
        raise ReviewValidationError("segment_ids must be non-empty")
    missing = sorted(set(segment_ids) - set(by_id))
    if missing:
        raise ReviewValidationError(f"unknown subject segments: {missing}")
    ordered_ids = sorted(set(segment_ids))
    base = copy.deepcopy(by_id[ordered_ids[0]])
    frames: dict[str, dict[str, Any]] = {}
    intervals: list[dict[str, Any]] = []
    for segment_id in ordered_ids:
        segment = by_id[segment_id]
        intervals.extend(copy.deepcopy(segment.get("presence_intervals", [])))
        for frame in segment.get("frames", []):
            frame_id = str(frame["id"])
            if frame_id in frames:
                raise ReviewValidationError(
                    f"segments overlap at {frame_id}; they cannot be the same person"
                )
            frames[frame_id] = copy.deepcopy(frame)
    base["subject_id"] = ordered_ids[0]
    base["track_id"] = f'{track_set["track_set_id"]}:{ordered_ids[0]}:reviewed'
    base["merged_segment_ids"] = ordered_ids
    base["presence_intervals"] = sorted(
        intervals, key=lambda item: (str(item.get("start_frame")), str(item.get("end_frame")))
    )
    base["frames"] = sorted(frames.values(), key=lambda item: int(item["index"]))
    return base


def _ensure_frame(
    subject: dict[str, Any],
    frame_id: str,
    source_frames: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    for frame in subject["frames"]:
        if frame["id"] == frame_id:
            return frame
    source_frame = source_frames.get(frame_id)
    if source_frame is None:
        raise ReviewValidationError(f"correction references unknown source frame: {frame_id}")
    frame = {
        "id": frame_id,
        "index": source_frame["index"],
        "time_seconds": source_frame["time_seconds"],
        "observation_id": None,
        "raw_instance_index": None,
        "bbox": None,
        "subject_score": None,
        "landmarks": {},
    }
    subject["frames"].append(frame)
    subject["frames"].sort(key=lambda item: int(item["index"]))
    return frame


def _apply_corrections(
    subject: dict[str, Any],
    corrections: list[dict[str, Any]],
    source_frames: dict[str, dict[str, Any]],
) -> None:
    width = int(subject.get("source", {}).get("width", 0))
    height = int(subject.get("source", {}).get("height", 0))
    seen: set[tuple[str, str]] = set()
    for correction in corrections:
        if "score" in correction or "detector_score" in correction:
            raise ReviewValidationError("review corrections cannot modify detector confidence")
        frame_id = str(correction.get("frame_id", ""))
        landmark = str(correction.get("landmark", ""))
        if landmark not in COCO_WHOLEBODY_NAMES:
            raise ReviewValidationError(f"unknown landmark in correction: {landmark}")
        key = (frame_id, landmark)
        if key in seen:
            raise ReviewValidationError(f"duplicate correction: {frame_id} {landmark}")
        seen.add(key)
        frame = _ensure_frame(subject, frame_id, source_frames)
        existing = frame["landmarks"].get(landmark, {})
        origin = correction.get("origin")
        if origin not in {"manual-source-witnessed", "occluded-unknown"}:
            raise ReviewValidationError(
                "review correction origin must be manual-source-witnessed or occluded-unknown"
            )
        if origin == "occluded-unknown":
            x = None
            y = None
        else:
            try:
                x = float(correction["x"])
                y = float(correction["y"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ReviewValidationError(
                    f"manual correction requires finite x/y: {frame_id} {landmark}"
                ) from exc
            if not math.isfinite(x) or not math.isfinite(y):
                raise ReviewValidationError(
                    f"manual correction requires finite x/y: {frame_id} {landmark}"
                )
            if width and not 0 <= x < width:
                raise ReviewValidationError(f"manual x is outside the source frame: {x}")
            if height and not 0 <= y < height:
                raise ReviewValidationError(f"manual y is outside the source frame: {y}")
        detector_score = existing.get("detector_score", existing.get("score"))
        frame["landmarks"][landmark] = {
            **existing,
            "x": x,
            "y": y,
            "score": detector_score,
            "detector_score": detector_score,
            "origin": origin,
            "review_correction": True,
        }


def _normalize_events(
    events: list[dict[str, Any]],
    valid_frame_ids: set[str],
) -> list[dict[str, Any]]:
    if not events:
        raise ReviewValidationError("each included subject requires at least one reviewed event")
    normalized: list[dict[str, Any]] = []
    for event in events:
        event_type = str(event.get("type", "")).strip()
        start_frame = str(event.get("start_frame", ""))
        end_value = event.get("end_frame")
        end_frame = str(end_value) if end_value is not None else None
        if not event_type:
            raise ReviewValidationError("event type is required")
        if start_frame not in valid_frame_ids:
            raise ReviewValidationError(f"event start frame is outside the subject: {start_frame}")
        if end_frame is not None and end_frame not in valid_frame_ids:
            raise ReviewValidationError(f"event end frame is outside the subject: {end_frame}")
        normalized.append(
            {
                "type": event_type,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "foot": event.get("foot"),
                "evidence": event.get("evidence"),
                "reviewed": True,
            }
        )
    event_types = {item["type"] for item in normalized}
    if event_types & STEP_EVENTS:
        missing = sorted(
            {"foot-release", "foot-travel", "landing", "weight-acceptance"}
            - event_types
        )
        if missing:
            raise ReviewValidationError(f"incomplete step event map: {missing}")
    return normalized


def apply_human_review(
    track_set: dict[str, Any],
    submission: dict[str, Any],
    *,
    reviewer: str,
    reviewed_unix: float | None = None,
) -> dict[str, Any]:
    """Apply a human attestation without rewriting raw detector evidence."""
    if track_set.get("schema") != TRACK_SET_SCHEMA:
        raise ReviewValidationError("review requires a grounded-motion-track-set/v1 artifact")
    if submission.get("schema", REVIEW_SCHEMA) != REVIEW_SCHEMA:
        raise ReviewValidationError(f"review submission schema must be {REVIEW_SCHEMA}")
    reviewer = reviewer.strip().lower()
    if not reviewer:
        raise ReviewValidationError("authenticated reviewer identity is required")

    source_frames = {str(item["id"]): item for item in track_set.get("frame_index", [])}
    original_ids = {str(item["subject_id"]) for item in track_set.get("subjects", [])}
    excluded_items = submission.get("excluded_subjects", [])
    excluded: dict[str, str] = {}
    for item in excluded_items:
        subject_id = str(item.get("subject_id", ""))
        reason = str(item.get("reason", "")).strip()
        if subject_id not in original_ids or not reason:
            raise ReviewValidationError("excluded subjects require a known id and non-empty reason")
        excluded[subject_id] = reason

    reviewed_subjects: list[dict[str, Any]] = []
    event_maps: list[dict[str, Any]] = []
    consumed: set[str] = set()
    now = float(reviewed_unix if reviewed_unix is not None else time.time())
    for subject_review in submission.get("subjects", []):
        _validate_attestations(subject_review)
        segment_ids = sorted(set(str(value) for value in subject_review.get("segment_ids", [])))
        overlap = consumed.intersection(segment_ids)
        if overlap:
            raise ReviewValidationError(f"subject segments reviewed more than once: {sorted(overlap)}")
        if set(segment_ids).intersection(excluded):
            raise ReviewValidationError("an excluded segment cannot also be reviewed")
        subject = _merge_segments(track_set, segment_ids)
        consumed.update(segment_ids)
        _apply_corrections(
            subject,
            list(subject_review.get("corrections", [])),
            source_frames,
        )
        quarantined = sorted(set(str(value) for value in subject_review.get("quarantined_landmarks", [])))
        unknown_quarantine = sorted(set(quarantined) - set(COCO_WHOLEBODY_NAMES))
        if unknown_quarantine:
            raise ReviewValidationError(f"unknown quarantined landmarks: {unknown_quarantine}")
        quarantine_reason = str(subject_review.get("quarantine_reason", "")).strip()
        if quarantined and not quarantine_reason:
            raise ReviewValidationError("quarantined landmarks require a reason")

        for frame in subject["frames"]:
            for landmark in _required_landmarks():
                if landmark in quarantined:
                    continue
                value = frame.get("landmarks", {}).get(landmark)
                if not value:
                    raise ReviewValidationError(
                        f'missing reviewed landmark {landmark} at {frame["id"]}'
                    )
                if value.get("origin") == "occluded-unknown":
                    raise ReviewValidationError(
                        f'occluded required landmark must be corrected or quarantined: '
                        f'{frame["id"]} {landmark}'
                    )
                value["origin"] = "manual-source-witnessed"
                value["review_attested"] = True

        valid_frame_ids = {str(frame["id"]) for frame in subject["frames"]}
        events = _normalize_events(list(subject_review.get("events", [])), valid_frame_ids)
        subject["events"] = events
        subject["state"] = "event-locked"
        subject["review"] = {
            "status": "reviewed",
            "reviewer": reviewer,
            "reviewed_unix": now,
            "source_interval_complete": True,
            "identity_continuity_reviewed": True,
            "required_landmarks_reviewed": True,
            "event_map_complete": True,
            "merged_segment_ids": segment_ids,
        }
        subject["judgment"] = {
            "quarantined_landmarks": quarantined,
            "quarantine_reason": quarantine_reason or None,
        }
        report = validate_track(subject, production=True)
        if not report["pass"]:
            raise ReviewValidationError(
                f'production review gate failed for {subject["subject_id"]}: {report["errors"]}'
            )
        reviewed_subjects.append(subject)
        event_maps.append(
            {
                "schema": EVENT_MAP_SCHEMA,
                "track_set_id": track_set["track_set_id"],
                "subject_id": subject["subject_id"],
                "status": "reviewed",
                "reviewer": reviewer,
                "reviewed_unix": now,
                "events": events,
            }
        )

    accounted = consumed.union(excluded)
    if accounted != original_ids:
        missing = sorted(original_ids - accounted)
        extra = sorted(accounted - original_ids)
        raise ReviewValidationError(
            f"every detected segment must be reviewed or excluded; missing={missing}, extra={extra}"
        )
    if not reviewed_subjects:
        raise ReviewValidationError("at least one subject must remain included for review")

    reviewed_track_set = copy.deepcopy(track_set)
    reviewed_track_set["state"] = "event-locked"
    reviewed_track_set["subjects"] = reviewed_subjects
    reviewed_track_set["review"] = {
        "status": "reviewed",
        "reviewer": reviewer,
        "reviewed_unix": now,
        "included_subject_ids": [item["subject_id"] for item in reviewed_subjects],
        "excluded_subjects": [
            {"subject_id": subject_id, "reason": excluded[subject_id]}
            for subject_id in sorted(excluded)
        ],
    }
    return {
        "schema": REVIEW_SCHEMA,
        "reviewer": reviewer,
        "reviewed_unix": now,
        "submission": copy.deepcopy(submission),
        "reviewed_track_set": reviewed_track_set,
        "event_maps": event_maps,
    }
