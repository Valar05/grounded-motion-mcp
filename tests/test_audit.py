from __future__ import annotations

import copy

from grounded_motion_mcp import audit
from grounded_motion_mcp.constants import (
    COCO_WHOLEBODY_NAMES,
    HAND_SUFFIXES,
    REQUIRED_GROUPS,
    SCHEMA,
)


def landmark(x: float, y: float, origin: str = "detector") -> dict[str, object]:
    return {"x": x, "y": y, "score": 6.25, "origin": origin}


def frame(index: int, offset: float = 0) -> dict[str, object]:
    landmarks = {
        name: landmark(20 + point_index * 0.1 + offset, 30 + point_index * 0.1)
        for point_index, name in enumerate(COCO_WHOLEBODY_NAMES)
    }
    return {
        "id": f"f{index:06d}",
        "index": index,
        "time_seconds": index / 10,
        "landmarks": landmarks,
    }


def track() -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "track_id": "fixture",
        "state": "tracked",
        "source": {
            "sha256": "a" * 64,
            "width": 100,
            "height": 100,
            "fps": 10.0,
            "duration_seconds": 0.2,
            "frame_count": 2,
        },
        "backend": {
            "name": "fixture",
            "model": "fixture",
            "version": "1",
            "license": "test",
            "device": "cpu",
            "config_sha256": "c" * 64,
            "model_sha256": "d" * 64,
        },
        "score_semantics": "fixture-score",
        "score_calibrated": False,
        "raw_predictions": {"path": "raw.json", "sha256": "b" * 64},
        "minimum_score": 0.5,
        "review": {"status": "unreviewed"},
        "events": [],
        "frames": [frame(0), frame(1, 1)],
    }


def review_and_lock(value: dict[str, object]) -> dict[str, object]:
    value["state"] = "event-locked"
    value["review"] = {
        "status": "reviewed",
        "source_interval_complete": True,
        "required_landmarks_reviewed": True,
        "event_map_complete": True,
    }
    value["events"] = [
        {
            "type": event_type,
            "start_frame": "f000000",
            "end_frame": "f000001",
            "reviewed": True,
        }
        for event_type in (
            "foot-release",
            "foot-travel",
            "landing",
            "weight-acceptance",
        )
    ]
    required = (
        REQUIRED_GROUPS["hips"]
        + REQUIRED_GROUPS["feet"]
        + REQUIRED_GROUPS["wrists"]
    )
    for frame_value in value["frames"]:
        for name in required:
            frame_value["landmarks"][name]["origin"] = "manual-source-witnessed"
    return value


def detailed_hands() -> list[str]:
    return [
        f"{side}_hand_{suffix}"
        for side in ("left", "right")
        for suffix in HAND_SUFFIXES
    ]


def test_valid_detector_track_passes_structural_gate() -> None:
    report = audit.validate_track(track(), production=False)
    assert report["pass"], report


def test_unreviewed_track_fails_production_gate() -> None:
    report = audit.validate_track(track(), production=True)
    assert not report["pass"]
    assert {"not-reviewed", "not-event-locked", "events-not-reviewed"} <= {
        item["code"] for item in report["errors"]
    }
    assert "required-landmark-not-reviewed" in {
        item["code"] for item in report["errors"]
    }


def test_generated_landmark_fails() -> None:
    value = track()
    value["frames"][0]["landmarks"]["left_heel"]["origin"] = "generated"
    report = audit.validate_track(value)
    assert not report["pass"]
    assert "forbidden-origin" in {item["code"] for item in report["errors"]}


def test_invalid_detector_score_fails_structural_validation() -> None:
    value = track()
    value["frames"][0]["landmarks"]["face_00"]["score"] = -1
    report = audit.validate_track(value)
    assert not report["pass"]
    assert "invalid-detector-score" in {item["code"] for item in report["errors"]}


def test_partial_source_interval_fails() -> None:
    value = track()
    value["source"]["frame_count"] = 3
    report = audit.validate_track(value)
    assert not report["pass"]
    assert "incomplete-source-interval" in {item["code"] for item in report["errors"]}


def test_unreviewed_comparison_is_blocked_without_boolean_verdict() -> None:
    report = audit.compare_tracks(track(), copy.deepcopy(track()))
    assert report["judgment_status"] == "blocked"
    assert report["mechanical_pass"] is None
    assert report["pass"] is None
    assert report["deviations"] == []
    assert report["diagnostic_metrics"]["root_relative_mechanics"][
        "authoritative"
    ] is False


def test_candidate_hip_drift_fails_reviewed_comparison() -> None:
    source = review_and_lock(track())
    candidate = review_and_lock(copy.deepcopy(source))
    candidate["frames"][1]["landmarks"]["left_hip"]["x"] += 30
    candidate["frames"][1]["landmarks"]["right_hip"]["x"] += 30
    report = audit.compare_tracks(source, candidate)
    assert report["judgment_status"] == "completed"
    assert report["mechanical_pass"] is False
    assert "motion-deviation" in {item["code"] for item in report["errors"]}


def test_complete_reviewed_event_map_passes_production() -> None:
    value = review_and_lock(track())
    report = audit.validate_track(value, production=True)
    assert report["pass"], report


def test_review_attestation_without_witnessed_landmarks_fails() -> None:
    value = track()
    value["state"] = "event-locked"
    value["review"] = {
        "status": "reviewed",
        "source_interval_complete": True,
        "required_landmarks_reviewed": True,
        "event_map_complete": True,
    }
    value["events"] = [
        {"type": "landing", "start_frame": "f000000", "reviewed": True}
    ]
    report = audit.validate_track(value, production=True)
    assert "required-landmark-not-reviewed" in {
        item["code"] for item in report["errors"]
    }


def test_different_canvas_requires_fixed_render_registration() -> None:
    source = review_and_lock(track())
    candidate = review_and_lock(copy.deepcopy(source))
    candidate["source"]["width"] = 200
    candidate["source"]["height"] = 200
    report = audit.compare_tracks(source, candidate)
    assert report["judgment_status"] == "blocked"
    assert report["mechanical_pass"] is None
    assert "coordinate-system-mismatch" in {
        item["code"] for item in report["judgment_blockers"]
    }


def test_fixed_render_registration_is_not_used_for_root_relative_mechanics() -> None:
    source = review_and_lock(track())
    candidate = review_and_lock(copy.deepcopy(source))
    candidate["source"]["width"] = 200
    candidate["source"]["height"] = 200
    for frame_value in candidate["frames"]:
        for value in frame_value["landmarks"].values():
            value["x"] *= 2
            value["y"] *= 2
    candidate["judgment"] = {
        "render_registration_to_source": {
            "scale_x": 0.5,
            "scale_y": 0.5,
            "offset_x": 0,
            "offset_y": 0,
            "provenance": "fixture",
        }
    }
    report = audit.compare_tracks(source, candidate)
    assert report["mechanical_pass"] is True, report
    registration = report["diagnostic_metrics"]["render_registration"]
    assert registration["used_for_root_translation_only"] is True
    assert registration["used_for_root_relative_mechanics"] is False


def test_pure_root_translation_does_not_fail_root_relative_mechanics() -> None:
    source = review_and_lock(track())
    candidate = review_and_lock(copy.deepcopy(source))
    for frame_value in candidate["frames"]:
        for value in frame_value["landmarks"].values():
            value["x"] += 30
            value["y"] -= 12
    report = audit.compare_tracks(source, candidate)
    assert report["mechanical_pass"] is True, report
    root = report["diagnostic_metrics"]["root_translation"]
    assert root["excluded_from_mechanical_pass"] is True
    assert root["summary"]["maximum"] > 0


def test_sprite_hands_are_preserved_but_quarantined_from_judgment() -> None:
    source = review_and_lock(track())
    candidate = review_and_lock(copy.deepcopy(source))
    names = detailed_hands()
    policy = {
        "modality": "sprite",
        "quarantined_landmarks": names,
        "quarantine_reason": "sprite detail unreviewed",
    }
    source["judgment"] = copy.deepcopy(policy)
    candidate["judgment"] = copy.deepcopy(policy)
    original_score = candidate["frames"][0]["landmarks"][names[0]]["score"]
    for frame_value in candidate["frames"]:
        for name in names:
            frame_value["landmarks"][name]["x"] += 1000
    report = audit.compare_tracks(source, candidate)
    assert report["mechanical_pass"] is True, report
    quarantine = report["diagnostic_metrics"]["quarantine"]
    assert quarantine["count"] == 42
    assert quarantine["raw_predictions_preserved"] is True
    assert report["diagnostic_metrics"]["root_relative_mechanics"][
        "compared_counts"
    ]["hands"] == 0
    assert candidate["frames"][0]["landmarks"][names[0]]["score"] == original_score


def test_body_wrist_drift_still_fails_when_detailed_hands_are_quarantined() -> None:
    source = review_and_lock(track())
    candidate = review_and_lock(copy.deepcopy(source))
    names = detailed_hands()
    source["judgment"] = {"quarantined_landmarks": names}
    candidate["judgment"] = {"quarantined_landmarks": names}
    candidate["frames"][1]["landmarks"]["right_wrist"]["x"] += 20
    report = audit.compare_tracks(source, candidate)
    assert report["mechanical_pass"] is False
    assert any(
        item["landmark"] == "right_wrist" for item in report["deviations"]
    )
