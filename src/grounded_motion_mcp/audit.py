"""Fail-closed validation, metrics, trajectories, and source/candidate comparison."""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any

from .constants import (
    ALLOWED_ORIGINS,
    AUDIT_SCHEMA,
    COMPARISON_SCHEMA,
    FORBIDDEN_ACCEPTED_ORIGINS,
    HAND_SUFFIXES,
    PRODUCTION_STATES,
    REQUIRED_GROUPS,
    SCHEMA,
    STEP_EVENTS,
    TRAJECTORY_COLORS,
)


def point(frame: dict[str, Any], name: str) -> dict[str, Any] | None:
    if name == "pelvis":
        left = point(frame, "left_hip")
        right = point(frame, "right_hip")
        if not usable(left) or not usable(right):
            return None
        return {
            "x": (float(left["x"]) + float(right["x"])) / 2,
            "y": (float(left["y"]) + float(right["y"])) / 2,
            "score": min(float(left.get("score", 1)), float(right.get("score", 1))),
            "origin": "derived",
        }
    return frame.get("landmarks", {}).get(name)


def usable(value: dict[str, Any] | None, threshold: float = 0.0) -> bool:
    if not value or value.get("origin") == "occluded-unknown":
        return False
    try:
        return (
            math.isfinite(float(value["x"]))
            and math.isfinite(float(value["y"]))
            and float(value.get("score", 1.0)) >= threshold
        )
    except (KeyError, TypeError, ValueError):
        return False


def distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def body_scale(frame: dict[str, Any]) -> float | None:
    values = []
    for side in ("left", "right"):
        hip = point(frame, f"{side}_hip")
        ankle = point(frame, f"{side}_ankle")
        if usable(hip) and usable(ankle):
            values.append(distance(hip, ankle))
    return sum(values) / len(values) if values else None


def all_named_landmarks() -> set[str]:
    names = {name for group in REQUIRED_GROUPS.values() for name in group}
    names.update(
        f"{side}_hand_{suffix}"
        for side in ("left", "right")
        for suffix in HAND_SUFFIXES
    )
    return names


def calculate_metrics(track: dict[str, Any]) -> dict[str, Any]:
    frames = track.get("frames", [])
    trajectories: dict[str, list[dict[str, float | str]]] = {
        name: [] for name in TRAJECTORY_COLORS
    }
    stance_width: list[dict[str, float | str]] = []
    hand_vector: list[dict[str, float | str]] = []
    pelvis_speed: list[dict[str, float | str]] = []
    previous_pelvis = None
    previous_time = None

    for frame in frames:
        frame_id = str(frame.get("id"))
        timestamp = float(frame.get("time_seconds", 0))
        for name, trajectory in trajectories.items():
            value = point(frame, name)
            if usable(value):
                trajectory.append(
                    {
                        "frame": frame_id,
                        "time": timestamp,
                        "x": float(value["x"]),
                        "y": float(value["y"]),
                    }
                )

        left_heel = point(frame, "left_heel")
        right_heel = point(frame, "right_heel")
        if usable(left_heel) and usable(right_heel):
            stance_width.append(
                {
                    "frame": frame_id,
                    "pixels": abs(float(left_heel["x"]) - float(right_heel["x"])),
                }
            )

        left_wrist = point(frame, "left_wrist")
        right_wrist = point(frame, "right_wrist")
        if usable(left_wrist) and usable(right_wrist):
            dx = float(right_wrist["x"]) - float(left_wrist["x"])
            dy = float(right_wrist["y"]) - float(left_wrist["y"])
            hand_vector.append(
                {
                    "frame": frame_id,
                    "distance": math.hypot(dx, dy),
                    "angle_degrees": math.degrees(math.atan2(dy, dx)),
                }
            )

        pelvis = point(frame, "pelvis")
        if (
            usable(pelvis)
            and previous_pelvis is not None
            and previous_time is not None
            and timestamp > previous_time
        ):
            pelvis_speed.append(
                {
                    "frame": frame_id,
                    "pixels_per_second": distance(pelvis, previous_pelvis)
                    / (timestamp - previous_time),
                }
            )
        if usable(pelvis):
            previous_pelvis = pelvis
            previous_time = timestamp

    return {
        "trajectories": trajectories,
        "stance_width": stance_width,
        "hand_vector": hand_vector,
        "pelvis_speed": pelvis_speed,
    }


def validate_track(track: dict[str, Any], production: bool = False) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    if track.get("schema") != SCHEMA:
        errors.append({"code": "schema", "message": f"Expected {SCHEMA}"})

    source = track.get("source", {})
    for key in ("sha256", "width", "height", "fps", "frame_count", "duration_seconds"):
        if key not in source:
            errors.append({"code": "source-field", "field": key})

    backend = track.get("backend", {})
    for key in (
        "name",
        "model",
        "version",
        "license",
        "device",
        "config_sha256",
        "model_sha256",
    ):
        if not backend.get(key):
            errors.append({"code": "backend-field", "field": key})

    raw = track.get("raw_predictions", {})
    for key in ("path", "sha256"):
        if not raw.get(key):
            errors.append({"code": "raw-predictions-field", "field": key})

    frames = track.get("frames")
    if not isinstance(frames, list) or not frames:
        errors.append({"code": "frames", "message": "frames must be non-empty"})
        return _finish_report(track, production, errors, warnings, {})

    expected_frames = source.get("frame_count")
    if isinstance(expected_frames, int) and expected_frames != len(frames):
        errors.append(
            {
                "code": "incomplete-source-interval",
                "expected_frames": expected_frames,
                "actual_frames": len(frames),
            }
        )

    threshold = float(track.get("minimum_score", 0.5))
    last_time = -math.inf
    ids: set[str] = set()
    indices: set[int] = set()
    coverage_counts = {name: 0 for name in all_named_landmarks()}
    reviewed_counts = {name: 0 for name in all_named_landmarks()}

    for position, frame in enumerate(frames):
        frame_id = frame.get("id")
        index = frame.get("index")
        if not frame_id or frame_id in ids:
            errors.append({"code": "frame-id", "frame": position, "id": frame_id})
        ids.add(frame_id)
        if not isinstance(index, int) or index in indices:
            errors.append({"code": "frame-index", "frame": frame_id, "index": index})
        else:
            indices.add(index)
            if index != position:
                errors.append(
                    {
                        "code": "frame-order",
                        "frame": frame_id,
                        "expected_index": position,
                        "actual_index": index,
                    }
                )
        try:
            timestamp = float(frame["time_seconds"])
            if timestamp <= last_time:
                errors.append({"code": "time-order", "frame": frame_id})
            last_time = timestamp
        except (KeyError, TypeError, ValueError):
            errors.append({"code": "time", "frame": frame_id})

        landmarks = frame.get("landmarks", {})
        if not isinstance(landmarks, dict):
            errors.append({"code": "landmarks", "frame": frame_id})
            continue

        for name, value in landmarks.items():
            origin = value.get("origin", "detector") if isinstance(value, dict) else None
            if origin in FORBIDDEN_ACCEPTED_ORIGINS:
                errors.append(
                    {
                        "code": "forbidden-origin",
                        "frame": frame_id,
                        "landmark": name,
                        "origin": origin,
                    }
                )
            elif origin not in ALLOWED_ORIGINS:
                warnings.append(
                    {
                        "code": "unknown-origin",
                        "frame": frame_id,
                        "landmark": name,
                        "origin": origin,
                    }
                )
            if usable(value, threshold):
                coverage_counts[name] = coverage_counts.get(name, 0) + 1
                if origin == "manual-source-witnessed":
                    reviewed_counts[name] = reviewed_counts.get(name, 0) + 1
            elif name in all_named_landmarks():
                warnings.append(
                    {
                        "code": "low-confidence-or-unknown",
                        "frame": frame_id,
                        "landmark": name,
                    }
                )

    frame_count = len(frames)
    coverage = {name: count / frame_count for name, count in coverage_counts.items()}
    reviewed_coverage = {name: count / frame_count for name, count in reviewed_counts.items()}

    for name in (
        REQUIRED_GROUPS["hips"] + REQUIRED_GROUPS["feet"] + REQUIRED_GROUPS["wrists"]
    ):
        if coverage.get(name, 0) < 1.0:
            errors.append(
                {
                    "code": "required-coverage",
                    "landmark": name,
                    "coverage": round(coverage.get(name, 0), 4),
                }
            )

    for side in ("left", "right"):
        hand_names = [f"{side}_hand_{suffix}" for suffix in HAND_SUFFIXES]
        visible = [coverage.get(name, 0) for name in hand_names]
        if max(visible, default=0) > 0 and min(visible, default=0) < 1.0:
            warnings.append(
                {
                    "code": "partial-hand",
                    "side": side,
                    "minimum_coverage": round(min(visible), 4),
                }
            )
        critical = [
            f"{side}_hand_wrist",
            f"{side}_hand_thumb_cmc",
            f"{side}_hand_thumb_mcp",
            f"{side}_hand_index_mcp",
            f"{side}_hand_pinky_mcp",
        ]
        if any(coverage.get(name, 0) < 1.0 for name in critical):
            warnings.append(
                {
                    "code": "hand-semantic-gap",
                    "side": side,
                    "message": "Grip turnover cannot receive an automatic pass",
                }
            )

    events = track.get("events", [])
    frame_ids = {frame.get("id") for frame in frames}
    event_types = set()
    for event in events:
        event_type = event.get("type")
        event_types.add(event_type)
        if event.get("start_frame") not in frame_ids:
            errors.append({"code": "event-frame", "event": event_type, "field": "start_frame"})
        if event.get("end_frame") and event.get("end_frame") not in frame_ids:
            errors.append({"code": "event-frame", "event": event_type, "field": "end_frame"})

    if event_types & STEP_EVENTS:
        missing = sorted(
            {"foot-release", "foot-travel", "landing", "weight-acceptance"} - event_types
        )
        if missing:
            errors.append({"code": "incomplete-step-event-map", "missing": missing})

    if production:
        state = track.get("state", "tracked")
        review = track.get("review", {})
        if state not in PRODUCTION_STATES or PRODUCTION_STATES.index(state) < PRODUCTION_STATES.index(
            "event-locked"
        ):
            errors.append(
                {
                    "code": "not-event-locked",
                    "state": state,
                    "message": "Inference is not production acceptance",
                }
            )
        if review.get("status") != "reviewed":
            errors.append({"code": "not-reviewed", "status": review.get("status")})
        for field in (
            "source_interval_complete",
            "required_landmarks_reviewed",
            "event_map_complete",
        ):
            if review.get(field) is not True:
                errors.append({"code": "review-attestation-missing", "field": field})
        if not events or any(not event.get("reviewed") for event in events):
            errors.append({"code": "events-not-reviewed"})

    metrics = calculate_metrics(track)
    metrics["coverage"] = coverage
    metrics["manual_review_coverage"] = reviewed_coverage
    return _finish_report(track, production, errors, warnings, metrics)


def compare_tracks(source: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    source_report = validate_track(source, production=False)
    candidate_report = validate_track(candidate, production=False)
    errors: list[dict[str, Any]] = []
    deviations: list[dict[str, Any]] = []

    if not source_report["pass"]:
        errors.append({"code": "invalid-source-track"})
    if not candidate_report["pass"]:
        errors.append({"code": "invalid-candidate-track"})

    source_frames = source.get("frames", [])
    candidate_frames = candidate.get("frames", [])
    if len(source_frames) != len(candidate_frames):
        errors.append(
            {
                "code": "frame-count",
                "source": len(source_frames),
                "candidate": len(candidate_frames),
            }
        )

    thresholds = {"hips": 0.04, "feet": 0.03, "wrists": 0.035, "hands": 0.025}
    thresholds.update(source.get("comparison_thresholds", {}))
    groups = dict(REQUIRED_GROUPS)
    groups["hands"] = [
        f"{side}_hand_{suffix}"
        for side in ("left", "right")
        for suffix in HAND_SUFFIXES
    ]
    source_size = (
        source.get("source", {}).get("width"),
        source.get("source", {}).get("height"),
    )
    candidate_size = (
        candidate.get("source", {}).get("width"),
        candidate.get("source", {}).get("height"),
    )
    transform = candidate.get("coordinate_transform_to_source")
    if source_size != candidate_size and not transform:
        errors.append(
            {
                "code": "coordinate-system-mismatch",
                "source_size": source_size,
                "candidate_size": candidate_size,
                "message": "Declare one fixed candidate coordinate_transform_to_source",
            }
        )
    valid_transform = None
    if transform:
        required_transform = {"scale_x", "scale_y", "offset_x", "offset_y"}
        if not required_transform <= set(transform):
            errors.append(
                {
                    "code": "invalid-coordinate-transform",
                    "missing": sorted(required_transform - set(transform)),
                }
            )
        else:
            valid_transform = transform

    for source_frame, candidate_frame in zip(source_frames, candidate_frames):
        if source_frame.get("id") != candidate_frame.get("id"):
            errors.append(
                {
                    "code": "frame-map",
                    "source": source_frame.get("id"),
                    "candidate": candidate_frame.get("id"),
                }
            )
        if (
            abs(
                float(source_frame.get("time_seconds", 0))
                - float(candidate_frame.get("time_seconds", 0))
            )
            > 1e-6
        ):
            errors.append({"code": "timestamp", "frame": source_frame.get("id")})

        scale = body_scale(source_frame)
        if not scale or scale <= 0:
            errors.append({"code": "body-scale", "frame": source_frame.get("id")})
            continue

        for group, names in groups.items():
            threshold = float(thresholds[group])
            for name in names:
                a = point(source_frame, name)
                b = point(candidate_frame, name)
                if usable(a) and usable(b):
                    if valid_transform:
                        b = {
                            **b,
                            "x": float(b["x"]) * float(valid_transform["scale_x"])
                            + float(valid_transform["offset_x"]),
                            "y": float(b["y"]) * float(valid_transform["scale_y"])
                            + float(valid_transform["offset_y"]),
                        }
                    normalized = distance(a, b) / scale
                    if normalized > threshold:
                        deviations.append(
                            {
                                "frame": source_frame.get("id"),
                                "group": group,
                                "landmark": name,
                                "normalized_error": round(normalized, 6),
                                "threshold": threshold,
                            }
                        )
                elif usable(a) != usable(b):
                    deviations.append(
                        {
                            "frame": source_frame.get("id"),
                            "group": group,
                            "landmark": name,
                            "normalized_error": None,
                            "threshold": threshold,
                            "reason": "visibility-mismatch",
                        }
                    )

    if deviations:
        errors.append({"code": "motion-deviation", "count": len(deviations)})

    return {
        "schema": COMPARISON_SCHEMA,
        "pass": not errors,
        "errors": errors,
        "warnings": [],
        "deviations": deviations,
        "source_validation": source_report,
        "candidate_validation": candidate_report,
    }


def _finish_report(
    track: dict[str, Any],
    production: bool,
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": AUDIT_SCHEMA,
        "track_id": track.get("track_id"),
        "gate": "production" if production else "structural",
        "pass": not errors,
        "errors": errors,
        "warnings": warnings,
        "metrics": metrics,
    }


def write_trajectory_svg(track: dict[str, Any], path: Path) -> None:
    source = track.get("source", {})
    width = int(source.get("width", 1920))
    height = int(source.get("height", 1080))
    metrics = calculate_metrics(track)
    chunks = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}">'
        ),
        '<rect width="100%" height="100%" fill="#111"/>',
    ]
    for name, color in TRAJECTORY_COLORS.items():
        points = metrics["trajectories"].get(name, [])
        if len(points) > 1:
            coords = " ".join(f'{item["x"]:.2f},{item["y"]:.2f}' for item in points)
            chunks.append(
                f'<polyline points="{coords}" fill="none" stroke="{color}" '
                'stroke-width="3" stroke-linejoin="round" opacity="0.9"/>'
            )
        for item in points:
            chunks.append(
                f'<circle cx="{item["x"]:.2f}" cy="{item["y"]:.2f}" r="3" fill="{color}"/>'
            )
    legend_y = 24
    for name, color in TRAJECTORY_COLORS.items():
        chunks.append(f'<circle cx="16" cy="{legend_y}" r="5" fill="{color}"/>')
        chunks.append(
            f'<text x="28" y="{legend_y + 5}" fill="#fff" '
            f'font-family="sans-serif" font-size="14">{html.escape(name)}</text>'
        )
        legend_y += 20
    chunks.append("</svg>")
    path.write_text("\n".join(chunks) + "\n", encoding="utf-8")


def inspect_track(track: dict[str, Any]) -> dict[str, Any]:
    report = validate_track(track, production=False)
    production = validate_track(track, production=True)
    metrics = report.get("metrics", {})
    coverage = metrics.get("coverage", {})
    required = REQUIRED_GROUPS["hips"] + REQUIRED_GROUPS["feet"] + REQUIRED_GROUPS["wrists"]
    gaps = {name: coverage.get(name, 0.0) for name in required if coverage.get(name, 0.0) < 1}
    return {
        "schema": track.get("schema"),
        "track_id": track.get("track_id"),
        "state": track.get("state"),
        "source": track.get("source"),
        "backend": track.get("backend"),
        "frame_count": len(track.get("frames", [])),
        "event_count": len(track.get("events", [])),
        "review": track.get("review"),
        "structural_pass": report["pass"],
        "production_pass": production["pass"],
        "required_coverage_gaps": gaps,
        "warnings": report["warnings"],
        "production_errors": production["errors"],
    }
