from __future__ import annotations

import pytest

from grounded_motion_mcp.backend import RawFrame, RawInstance
from grounded_motion_mcp.constants import COCO_WHOLEBODY_NAMES
from grounded_motion_mcp.normalize import (
    SubjectAmbiguityError,
    normalize_track,
    normalize_track_set,
)
from grounded_motion_mcp.review import ReviewValidationError, apply_human_review


def instance(score: float = 0.9) -> RawInstance:
    return RawInstance(
        keypoints=[[float(index), float(index + 1)] for index in range(133)],
        keypoint_scores=[score] * 133,
        bbox=[0, 0, 100, 100],
        bbox_score=0.95,
    )


def test_normalize_maps_all_133_landmarks() -> None:
    raw = [RawFrame("f000000", 0, 0.0, [instance()])]
    track = normalize_track(
        track_id="job",
        source={"sha256": "a" * 64},
        backend={"name": "fixture"},
        raw_frames=raw,
        raw_path="raw.json",
        raw_sha256="b" * 64,
        minimum_score=0.5,
    )
    assert list(track["frames"][0]["landmarks"]) == COCO_WHOLEBODY_NAMES
    assert track["state"] == "tracked"
    assert track["review"]["status"] == "unreviewed"
    assert track["score_semantics"] == "backend-native-keypoint-score"


def test_multiple_subjects_fail_instead_of_guessing_identity() -> None:
    raw = [RawFrame("f000000", 0, 0.0, [instance(), instance()])]
    with pytest.raises(SubjectAmbiguityError):
        normalize_track(
            track_id="job",
            source={"sha256": "a" * 64},
            backend={"name": "fixture"},
            raw_frames=raw,
            raw_path="raw.json",
            raw_sha256="b" * 64,
            minimum_score=0.5,
        )



def test_normalize_preserves_unbounded_detector_score_exactly() -> None:
    raw = [RawFrame("f000000", 0, 0.0, [instance(6.842609405517578)])]
    track = normalize_track(
        track_id="job",
        source={"sha256": "a" * 64},
        backend={
            "name": "mmpose",
            "model": "rtmw-x",
            "version": "1.3.2",
            "license": "Apache-2.0",
            "device": "cuda:0",
            "config_sha256": "c" * 64,
            "model_sha256": "d" * 64,
            "score_semantics": "simcc-max-response",
            "score_calibrated": False,
        },
        raw_frames=raw,
        raw_path="raw.json",
        raw_sha256="b" * 64,
        minimum_score=0.5,
    )
    landmark = track["frames"][0]["landmarks"]["nose"]
    assert landmark["score"] == 6.842609405517578
    assert track["score_semantics"] == "simcc-max-response"
    assert track["score_calibrated"] is False


@pytest.mark.parametrize("score", [-0.1, float("nan"), float("inf")])
def test_normalize_rejects_invalid_detector_scores(score: float) -> None:
    raw = [RawFrame("f000000", 0, 0.0, [instance(score)])]
    with pytest.raises(ValueError, match="invalid detector score"):
        normalize_track(
            track_id="job",
            source={"sha256": "a" * 64},
            backend={"name": "fixture"},
            raw_frames=raw,
            raw_path="raw.json",
            raw_sha256="b" * 64,
            minimum_score=0.5,
        )


def positioned_instance(x_offset: float, score: float = 0.9) -> RawInstance:
    return RawInstance(
        keypoints=[
            [x_offset + float(index % 11), 100.0 + float(index // 11)]
            for index in range(133)
        ],
        keypoint_scores=[score] * 133,
        bbox=[x_offset, 90.0, x_offset + 80.0, 220.0],
        bbox_score=0.97,
    )


def multiperson_track_set(*, ambiguous: bool = False) -> dict[str, object]:
    pytest.importorskip("scipy")
    first = [positioned_instance(50.0, 6.5), positioned_instance(450.0, 4.25)]
    second = (
        [positioned_instance(250.0), positioned_instance(250.0)]
        if ambiguous
        else [positioned_instance(54.0, 6.25), positioned_instance(454.0, 4.0)]
    )
    return normalize_track_set(
        track_set_id="multi",
        source={
            "sha256": "a" * 64,
            "width": 800,
            "height": 600,
            "fps": 10.0,
            "duration_seconds": 0.2,
            "frame_count": 2,
        },
        backend={
            "name": "mmpose",
            "model": "rtmw-x",
            "version": "1.3.2",
            "license": "Apache-2.0",
            "device": "cuda:0",
            "config_sha256": "c" * 64,
            "model_sha256": "d" * 64,
            "score_semantics": "simcc-max-response",
            "score_calibrated": False,
        },
        raw_frames=[
            RawFrame("f000000", 0, 0.0, first),
            RawFrame("f000001", 1, 0.1, second),
        ],
        raw_path="raw.json",
        raw_sha256="b" * 64,
        minimum_score=0.5,
    )


def test_multi_person_normalization_preserves_people_and_detector_confidence() -> None:
    track_set = multiperson_track_set()
    assert [len(item["frames"]) for item in track_set["subjects"]] == [2, 2]
    assert track_set["association"]["automatic_reidentification"] is False
    first_nose = track_set["subjects"][0]["frames"][0]["landmarks"]["nose"]
    assert first_nose["score"] == 6.5
    assert first_nose["detector_score"] == 6.5
    assert first_nose["origin"] == "detector"


def test_ambiguous_multi_person_crossing_starts_new_review_segments() -> None:
    track_set = multiperson_track_set(ambiguous=True)
    assert len(track_set["subjects"]) == 4
    assert len(track_set["identity_findings"]) == 2
    assert {item["code"] for item in track_set["identity_findings"]} == {
        "identity-ambiguous"
    }


def test_human_review_requires_accounting_and_locks_reviewed_events() -> None:
    track_set = multiperson_track_set()
    subjects = track_set["subjects"]
    submission = {
        "schema": "grounded-motion-review/v1",
        "subjects": [
            {
                "segment_ids": [subjects[0]["subject_id"]],
                "source_interval_complete": True,
                "identity_continuity_reviewed": True,
                "required_landmarks_reviewed": True,
                "event_map_complete": True,
                "events": [
                    {
                        "type": "pose-anchor",
                        "start_frame": "f000000",
                        "end_frame": "f000001",
                        "evidence": "reviewed against source frames",
                    }
                ],
            }
        ],
        "excluded_subjects": [
            {"subject_id": subjects[1]["subject_id"], "reason": "bystander"}
        ],
    }
    applied = apply_human_review(
        track_set, submission, reviewer="reviewer@example.com", reviewed_unix=42.0
    )
    reviewed = applied["reviewed_track_set"]
    assert reviewed["state"] == "event-locked"
    assert reviewed["review"]["status"] == "reviewed"
    assert reviewed["subjects"][0]["events"][0]["reviewed"] is True
    nose = reviewed["subjects"][0]["frames"][0]["landmarks"]["nose"]
    assert nose["detector_score"] == 6.5

    submission["subjects"][0]["corrections"] = [
        {
            "frame_id": "f000000",
            "landmark": "nose",
            "origin": "manual-source-witnessed",
            "x": 55.0,
            "y": 101.0,
            "detector_score": 1.0,
        }
    ]
    with pytest.raises(ReviewValidationError, match="cannot modify detector confidence"):
        apply_human_review(track_set, submission, reviewer="reviewer@example.com")
