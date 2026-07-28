from __future__ import annotations

import pytest

from grounded_motion_mcp.backend import RawFrame, RawInstance
from grounded_motion_mcp.constants import COCO_WHOLEBODY_NAMES
from grounded_motion_mcp.normalize import SubjectAmbiguityError, normalize_track


def instance() -> RawInstance:
    return RawInstance(
        keypoints=[[float(index), float(index + 1)] for index in range(133)],
        keypoint_scores=[0.9] * 133,
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

