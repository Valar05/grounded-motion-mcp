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
    TRACK_V3_SCHEMA,
    TRAJECTORY_COLORS,
)


def point(frame: dict[str, Any], name: str) -> dict[str, Any] | None:
    if name == "pelvis":
        left = point(frame, "left_hip")
        right = point(frame, "right_hip")
        if not usable(left) or not usable(right):
            return None
        scores = [
            value
            for value in (left.get("detector_score", left.get("score")), right.get("detector_score", right.get("score")))
            if value is not None
        ]
        return {
            "x": (float(left["x"]) + float(right["x"])) / 2,
            "y": (float(left["y"]) + float(right["y"])) / 2,
            "score": min(float(value) for value in scores) if scores else None,
            "detector_score": min(float(value) for value in scores) if scores else None,
            "origin": "derived",
        }
    return frame.get("landmarks", {}).get(name)


def usable(value: dict[str, Any] | None, threshold: float = 0.0) -> bool:
    if not value or value.get("origin") == "occluded-unknown":
        return False
    try:
        coordinates_are_finite = math.isfinite(float(value["x"])) and math.isfinite(
            float(value["y"])
        )
        if not coordinates_are_finite:
            return False
        if value.get("origin") == "manual-source-witnessed":
            return True
        score = value.get("detector_score", value.get("score", 1.0))
        return score is not None and float(score) >= threshold
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

    schema = track.get("schema")
    if schema not in {SCHEMA, TRACK_V3_SCHEMA}:
        errors.append(
            {"code": "schema", "message": f"Expected {SCHEMA} or {TRACK_V3_SCHEMA}"}
        )
    if not track.get("score_semantics"):
        errors.append({"code": "score-semantics", "message": "Missing detector score semantics"})
    if not isinstance(track.get("score_calibrated"), bool):
        errors.append({"code": "score-calibration", "message": "Missing calibration flag"})

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
    if (
        schema != TRACK_V3_SCHEMA
        and isinstance(expected_frames, int)
        and expected_frames != len(frames)
    ):
        errors.append(
            {
                "code": "incomplete-source-interval",
                "expected_frames": expected_frames,
                "actual_frames": len(frames),
            }
        )

    threshold = float(track.get("minimum_score", 0.5))
    last_time = -math.inf
    last_index = -1
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
            if schema == TRACK_V3_SCHEMA:
                if index <= last_index:
                    errors.append(
                        {
                            "code": "frame-order",
                            "frame": frame_id,
                            "previous_index": last_index,
                            "actual_index": index,
                        }
                    )
                last_index = index
            elif index != position:
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
            detector_score_value = (
                value.get("detector_score", value.get("score"))
                if isinstance(value, dict)
                else None
            )
            if detector_score_value is None and origin == "detector":
                errors.append(
                    {
                        "code": "missing-detector-score",
                        "frame": frame_id,
                        "landmark": name,
                    }
                )
            elif detector_score_value is not None:
                try:
                    detector_score = float(detector_score_value)
                    if not math.isfinite(detector_score) or detector_score < 0:
                        raise ValueError
                except (TypeError, ValueError):
                    errors.append(
                        {
                            "code": "invalid-detector-score",
                            "frame": frame_id,
                            "landmark": name,
                        }
                    )
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

    quarantined = set(track.get("judgment", {}).get("quarantined_landmarks", []))
    for name in (
        REQUIRED_GROUPS["hips"] + REQUIRED_GROUPS["feet"] + REQUIRED_GROUPS["wrists"]
    ):
        if name in quarantined:
            continue
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
        required_for_judgment = (
            REQUIRED_GROUPS["hips"]
            + REQUIRED_GROUPS["feet"]
            + REQUIRED_GROUPS["wrists"]
        )
        for name in required_for_judgment:
            if name in quarantined:
                continue
            if reviewed_coverage.get(name, 0) < 1.0:
                errors.append(
                    {
                        "code": "required-landmark-not-reviewed",
                        "landmark": name,
                        "reviewed_coverage": round(reviewed_coverage.get(name, 0), 4),
                    }
                )

    quarantined = sorted(
        set(track.get("judgment", {}).get("quarantined_landmarks", []))
    )
    if quarantined:
        warnings.append(
            {
                "code": "landmark-quarantine",
                "count": len(quarantined),
                "reason": track.get("judgment", {}).get(
                    "quarantine_reason", "not eligible for judgment"
                ),
            }
        )

    metrics = calculate_metrics(track)
    metrics["coverage"] = coverage
    metrics["manual_review_coverage"] = reviewed_coverage
    return _finish_report(track, production, errors, warnings, metrics)


def _apply_transform(
    value: dict[str, Any], transform: dict[str, Any] | None
) -> dict[str, Any]:
    if not transform:
        return value
    return {
        **value,
        "x": float(value["x"]) * float(transform["scale_x"])
        + float(transform["offset_x"]),
        "y": float(value["y"]) * float(transform["scale_y"])
        + float(transform["offset_y"]),
    }


def _metric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "maximum": None}
    return {
        "count": len(values),
        "mean": round(sum(values) / len(values), 6),
        "maximum": round(max(values), 6),
    }


def compare_tracks(source: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    source_report = validate_track(source, production=False)
    candidate_report = validate_track(candidate, production=False)
    source_production = validate_track(source, production=True)
    candidate_production = validate_track(candidate, production=True)
    errors: list[dict[str, Any]] = []
    diagnostic_deviations: list[dict[str, Any]] = []
    judgment_blockers: list[dict[str, Any]] = []

    for lane, report in (
        ("source", source_production),
        ("candidate", candidate_production),
    ):
        for error in report["errors"]:
            judgment_blockers.append({"lane": lane, **error})

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
    quarantined = sorted(
        set(source.get("judgment", {}).get("quarantined_landmarks", []))
        | set(candidate.get("judgment", {}).get("quarantined_landmarks", []))
    )
    quarantine_set = set(quarantined)

    source_size = (
        source.get("source", {}).get("width"),
        source.get("source", {}).get("height"),
    )
    candidate_size = (
        candidate.get("source", {}).get("width"),
        candidate.get("source", {}).get("height"),
    )
    transform = candidate.get("judgment", {}).get("render_registration_to_source")
    if transform is None:
        transform = candidate.get("coordinate_transform_to_source")
    valid_transform = None
    if transform:
        required_transform = {"scale_x", "scale_y", "offset_x", "offset_y"}
        if not required_transform <= set(transform):
            errors.append(
                {
                    "code": "invalid-render-registration",
                    "missing": sorted(required_transform - set(transform)),
                }
            )
        else:
            valid_transform = transform
    elif source_size != candidate_size:
        errors.append(
            {
                "code": "coordinate-system-mismatch",
                "source_size": source_size,
                "candidate_size": candidate_size,
                "message": "Declare one fixed candidate render_registration_to_source",
            }
        )

    root_samples: list[dict[str, Any]] = []
    root_magnitudes: list[float] = []
    group_errors: dict[str, list[float]] = {name: [] for name in groups}
    compared_counts = {name: 0 for name in groups}

    for source_frame, candidate_frame in zip(source_frames, candidate_frames):
        frame_id = source_frame.get("id")
        if frame_id != candidate_frame.get("id"):
            errors.append(
                {
                    "code": "frame-map",
                    "source": frame_id,
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
            errors.append({"code": "timestamp", "frame": frame_id})

        source_scale = body_scale(source_frame)
        candidate_scale = body_scale(candidate_frame)
        source_root = point(source_frame, "pelvis")
        candidate_root = point(candidate_frame, "pelvis")
        if (
            not source_scale
            or source_scale <= 0
            or not candidate_scale
            or candidate_scale <= 0
            or not usable(source_root)
            or not usable(candidate_root)
        ):
            errors.append({"code": "body-scale-or-root", "frame": frame_id})
            continue

        registered_root = _apply_transform(candidate_root, valid_transform)
        root_dx = float(registered_root["x"]) - float(source_root["x"])
        root_dy = float(registered_root["y"]) - float(source_root["y"])
        root_normalized = math.hypot(root_dx, root_dy) / source_scale
        root_magnitudes.append(root_normalized)
        root_samples.append(
            {
                "frame": frame_id,
                "time": float(source_frame.get("time_seconds", 0)),
                "dx_pixels": round(root_dx, 6),
                "dy_pixels": round(root_dy, 6),
                "normalized_magnitude": round(root_normalized, 6),
            }
        )

        for group, names in groups.items():
            threshold = float(thresholds[group])
            for name in names:
                if name in quarantine_set:
                    continue
                a = point(source_frame, name)
                b = point(candidate_frame, name)
                if usable(a) and usable(b):
                    a_relative = {
                        "x": (float(a["x"]) - float(source_root["x"])) / source_scale,
                        "y": (float(a["y"]) - float(source_root["y"])) / source_scale,
                    }
                    b_relative = {
                        "x": (float(b["x"]) - float(candidate_root["x"]))
                        / candidate_scale,
                        "y": (float(b["y"]) - float(candidate_root["y"]))
                        / candidate_scale,
                    }
                    normalized = distance(a_relative, b_relative)
                    group_errors[group].append(normalized)
                    compared_counts[group] += 1
                    if normalized > threshold:
                        diagnostic_deviations.append(
                            {
                                "frame": frame_id,
                                "group": group,
                                "landmark": name,
                                "normalized_error": round(normalized, 6),
                                "threshold": threshold,
                            }
                        )
                elif usable(a) != usable(b):
                    diagnostic_deviations.append(
                        {
                            "frame": frame_id,
                            "group": group,
                            "landmark": name,
                            "normalized_error": None,
                            "threshold": threshold,
                            "reason": "visibility-mismatch",
                        }
                    )

    if errors:
        judgment_blockers.extend({"lane": "comparison", **error} for error in errors)

    judgment_status = "completed" if not judgment_blockers else "blocked"
    mechanical_pass: bool | None
    deviations: list[dict[str, Any]]
    authoritative_errors = list(errors)
    if judgment_status == "completed":
        deviations = diagnostic_deviations
        mechanical_pass = not errors and not deviations
        if deviations:
            authoritative_errors.append(
                {"code": "motion-deviation", "count": len(deviations)}
            )
    else:
        deviations = []
        mechanical_pass = None

    registration_public = dict(
        valid_transform
        or {
            "scale_x": 1.0,
            "scale_y": 1.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
        }
    )
    return {
        "schema": COMPARISON_SCHEMA,
        "pass": mechanical_pass,
        "judgment_status": judgment_status,
        "mechanical_pass": mechanical_pass,
        "judgment_blockers": judgment_blockers,
        "errors": authoritative_errors,
        "warnings": (
            [
                {
                    "code": "sprite-hand-landmarks-quarantined",
                    "count": len(quarantined),
                    "landmarks": quarantined,
                }
            ]
            if quarantined
            else []
        ),
        "deviations": deviations,
        "diagnostic_metrics": {
            "render_registration": {
                "transform": registration_public,
                "provenance": registration_public.get("provenance"),
                "used_for_root_translation_only": True,
                "used_for_root_relative_mechanics": False,
            },
            "root_translation": {
                "excluded_from_mechanical_pass": True,
                "summary": _metric_summary(root_magnitudes),
                "samples": root_samples,
            },
            "root_relative_mechanics": {
                "authoritative": judgment_status == "completed",
                "normalization": "per-lane-pelvis-and-hip-to-ankle-scale",
                "thresholds": thresholds,
                "compared_counts": compared_counts,
                "group_error_summary": {
                    name: _metric_summary(values)
                    for name, values in group_errors.items()
                },
                "diagnostic_deviation_count": len(diagnostic_deviations),
                "diagnostic_deviations": diagnostic_deviations,
            },
            "quarantine": {
                "landmarks": quarantined,
                "count": len(quarantined),
                "raw_predictions_preserved": True,
                "excluded_from_mechanical_pass": True,
            },
        },
        "source_validation": source_report,
        "candidate_validation": candidate_report,
        "source_production_validation": source_production,
        "candidate_production_validation": candidate_production,
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



def write_track_set_trajectory_svg(track_set: dict[str, Any], destination: Path) -> Path:
    """Write a deterministic combined pelvis trajectory for all subject segments."""
    width = int(track_set.get("source", {}).get("width", 1280))
    height = int(track_set.get("source", {}).get("height", 720))
    palette = [
        "#00d8ff", "#ff4fd8", "#48e06f", "#ff7248", "#ffcc00", "#9b7bff"
    ]
    body = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="#111"/>',
    ]
    for index, subject in enumerate(track_set.get("subjects", [])):
        points = []
        for frame in subject.get("frames", []):
            pelvis = point(frame, "pelvis")
            if usable(pelvis):
                points.append(f'{float(pelvis["x"]):.3f},{float(pelvis["y"]):.3f}')
        color = palette[index % len(palette)]
        if points:
            body.append(
                f'<polyline points="{" ".join(points)}" fill="none" '
                f'stroke="{color}" stroke-width="3"/>'
            )
        label_y = 24 + index * 20
        label = html.escape(str(subject.get("subject_id", f"subject-{index + 1}")))
        body.append(f'<text x="12" y="{label_y}" fill="{color}">{label}</text>')
    body.append("</svg>")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(body) + "\n", encoding="utf-8")
    return destination
