"""Schema constants and COCO-WholeBody landmark names."""

from __future__ import annotations

SCHEMA = "grounded-motion-track/v2"
TRACK_V3_SCHEMA = "grounded-motion-track/v3"
TRACK_SET_SCHEMA = "grounded-motion-track-set/v1"
REVIEW_SCHEMA = "grounded-motion-review/v1"
EVENT_MAP_SCHEMA = "grounded-motion-event-map/v1"
INPUT_LOCK_SCHEMA = "grounded-motion-input-lock/v1"
EVIDENCE_INDEX_V2_SCHEMA = "grounded-motion-evidence-index/v2"
LEGACY_SCHEMA = "grounded-motion-track/v1"
AUDIT_SCHEMA = "grounded-motion-audit/v2"
COMPARISON_SCHEMA = "grounded-motion-comparison/v2"
RECEIPT_SCHEMA = "grounded-motion-receipt/v2"
MANIFEST_SCHEMA = "grounded-motion-manifest/v2"

ALLOWED_ORIGINS = {"detector", "manual-source-witnessed", "occluded-unknown"}
FORBIDDEN_ACCEPTED_ORIGINS = {"interpolated", "smoothed", "guessed", "generated"}

BODY_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

FOOT_NAMES = [
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
]

HAND_SUFFIXES = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
]

FACE_NAMES = [f"face_{index:02d}" for index in range(68)]
LEFT_HAND_NAMES = [f"left_hand_{name}" for name in HAND_SUFFIXES]
RIGHT_HAND_NAMES = [f"right_hand_{name}" for name in HAND_SUFFIXES]
COCO_WHOLEBODY_NAMES = BODY_NAMES + FOOT_NAMES + FACE_NAMES + LEFT_HAND_NAMES + RIGHT_HAND_NAMES

# MMPose 1.3.2 COCO-WholeBody labeling uncertainty values. Keeping these
# beside the landmark order makes association reproducible without importing
# the inference-only MMPose package in audit/control installations.
COCO_WHOLEBODY_SIGMAS = [
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
    0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089, 0.068, 0.066, 0.066,
    0.092, 0.094, 0.094, 0.042, 0.043, 0.044, 0.043, 0.040, 0.035, 0.031,
    0.025, 0.020, 0.023, 0.029, 0.032, 0.037, 0.038, 0.043, 0.041, 0.045,
    0.013, 0.012, 0.011, 0.011, 0.012, 0.012, 0.011, 0.011, 0.013, 0.015,
    0.009, 0.007, 0.007, 0.007, 0.012, 0.009, 0.008, 0.016, 0.010, 0.017,
    0.011, 0.009, 0.011, 0.009, 0.007, 0.013, 0.008, 0.011, 0.012, 0.010,
    0.034, 0.008, 0.008, 0.009, 0.008, 0.008, 0.007, 0.010, 0.008, 0.009,
    0.009, 0.009, 0.007, 0.007, 0.008, 0.011, 0.008, 0.008, 0.008, 0.010,
    0.008, 0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024, 0.035,
    0.018, 0.024, 0.022, 0.026, 0.017, 0.021, 0.021, 0.032, 0.020, 0.019,
    0.022, 0.031, 0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024,
    0.035, 0.018, 0.024, 0.022, 0.026, 0.017, 0.021, 0.021, 0.032, 0.020,
    0.019, 0.022, 0.031,
]

if len(COCO_WHOLEBODY_NAMES) != 133:
    raise RuntimeError("COCO-WholeBody mapping must contain exactly 133 landmarks")
if len(COCO_WHOLEBODY_SIGMAS) != 133:
    raise RuntimeError("COCO-WholeBody sigma mapping must contain exactly 133 values")

REQUIRED_GROUPS = {
    "hips": ["left_hip", "right_hip"],
    "feet": [
        "left_ankle",
        "left_heel",
        "left_big_toe",
        "left_small_toe",
        "right_ankle",
        "right_heel",
        "right_big_toe",
        "right_small_toe",
    ],
    "wrists": ["left_wrist", "right_wrist"],
}

TRAJECTORY_COLORS = {
    "pelvis": "#ffcc00",
    "left_wrist": "#00d8ff",
    "right_wrist": "#ff4fd8",
    "left_heel": "#48e06f",
    "right_heel": "#ff7248",
    "left_big_toe": "#198f3a",
    "right_big_toe": "#b8381c",
}

STEP_EVENTS = {
    "foot-release",
    "foot-travel",
    "foot-crossing",
    "landing",
    "weight-acceptance",
}

PRODUCTION_STATES = [
    "tracked",
    "reviewed",
    "event-locked",
    "keyed",
    "transferred",
    "mechanically-compared",
    "human-accepted",
]

