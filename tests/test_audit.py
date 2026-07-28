from __future__ import annotations

import copy

from grounded_motion_mcp import audit
from grounded_motion_mcp.constants import COCO_WHOLEBODY_NAMES, SCHEMA


def landmark(x: float, y: float, origin: str = "detector") -> dict[str, object]:
    return {"x": x, "y": y, "score": 0.99, "origin": origin}


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
        "raw_predictions": {"path": "raw.json", "sha256": "b" * 64},
        "minimum_score": 0.5,
        "review": {"status": "unreviewed"},
        "events": [],
        "frames": [frame(0), frame(1, 1)],
    }


def test_valid_detector_track_passes_structural_gate() -> None:
    report = audit.validate_track(track(), production=False)
    assert report["pass"], report


def test_unreviewed_track_fails_production_gate() -> None:
    report = audit.validate_track(track(), production=True)
    assert not report["pass"]
    assert {"not-reviewed", "not-event-locked", "events-not-reviewed"} <= {
        item["code"] for item in report["errors"]
    }


def test_generated_landmark_fails() -> None:
    value = track()
    value["frames"][0]["landmarks"]["left_heel"]["origin"] = "generated"
    report = audit.validate_track(value)
    assert not report["pass"]
    assert "forbidden-origin" in {item["code"] for item in report["errors"]}


def test_partial_source_interval_fails() -> None:
    value = track()
    value["source"]["frame_count"] = 3
    report = audit.validate_track(value)
    assert not report["pass"]
    assert "incomplete-source-interval" in {item["code"] for item in report["errors"]}


def test_candidate_hip_drift_fails_comparison() -> None:
    source = track()
    candidate = copy.deepcopy(source)
    candidate["frames"][1]["landmarks"]["left_hip"]["x"] += 30
    candidate["frames"][1]["landmarks"]["right_hip"]["x"] += 30
    report = audit.compare_tracks(source, candidate)
    assert not report["pass"]
    assert "motion-deviation" in {item["code"] for item in report["errors"]}


def test_complete_reviewed_event_map_passes_production() -> None:
    value = track()
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
    report = audit.validate_track(value, production=True)
    assert report["pass"], report


def test_different_canvas_requires_fixed_coordinate_transform() -> None:
    source = track()
    candidate = copy.deepcopy(source)
    candidate["source"]["width"] = 200
    candidate["source"]["height"] = 200
    report = audit.compare_tracks(source, candidate)
    assert not report["pass"]
    assert "coordinate-system-mismatch" in {item["code"] for item in report["errors"]}


def test_fixed_coordinate_transform_maps_candidate_to_source() -> None:
    source = track()
    candidate = copy.deepcopy(source)
    candidate["source"]["width"] = 200
    candidate["source"]["height"] = 200
    for frame_value in candidate["frames"]:
        for value in frame_value["landmarks"].values():
            value["x"] *= 2
            value["y"] *= 2
    candidate["coordinate_transform_to_source"] = {
        "scale_x": 0.5,
        "scale_y": 0.5,
        "offset_x": 0,
        "offset_y": 0,
    }
    report = audit.compare_tracks(source, candidate)
    assert report["pass"], report
