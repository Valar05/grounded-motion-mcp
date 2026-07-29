from __future__ import annotations

import pytest

from grounded_motion_mcp.backend import RawFrame, RawInstance
from grounded_motion_mcp.constants import COCO_WHOLEBODY_NAMES
from grounded_motion_mcp.normalize import SubjectAmbiguityError, normalize_track


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
