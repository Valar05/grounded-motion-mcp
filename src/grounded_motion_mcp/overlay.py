"""Source-frame overlay rendering with confidence-visible landmarks."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

BODY_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"),
    ("left_heel", "left_big_toe"),
    ("left_big_toe", "left_small_toe"),
    ("left_ankle", "left_big_toe"),
    ("right_ankle", "right_heel"),
    ("right_heel", "right_big_toe"),
    ("right_big_toe", "right_small_toe"),
    ("right_ankle", "right_big_toe"),
]

HAND_CHAINS = [
    ("wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip"),
    ("wrist", "index_mcp", "index_pip", "index_dip", "index_tip"),
    ("wrist", "middle_mcp", "middle_pip", "middle_dip", "middle_tip"),
    ("wrist", "ring_mcp", "ring_pip", "ring_dip", "ring_tip"),
    ("wrist", "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"),
]


def _xy(landmarks: dict[str, Any], name: str, minimum_score: float) -> tuple[float, float] | None:
    value = landmarks.get(name)
    if not value or value.get("origin") == "occluded-unknown":
        return None
    score = value.get("detector_score", value.get("score"))
    if value.get("origin") != "manual-source-witnessed" and (
        score is None or float(score) < minimum_score
    ):
        return None
    if value.get("x") is None or value.get("y") is None:
        return None
    return float(value["x"]), float(value["y"])


def render_overlays(
    frame_paths: list[Path],
    track: dict[str, Any],
    destination: Path,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    minimum_score = float(track.get("minimum_score", 0.5))
    for frame_path, frame in zip(frame_paths, track.get("frames", [])):
        with Image.open(frame_path) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        landmarks = frame["landmarks"]

        for left, right in BODY_EDGES:
            a = _xy(landmarks, left, minimum_score)
            b = _xy(landmarks, right, minimum_score)
            if a and b:
                draw.line((a, b), fill=(255, 220, 0), width=3)

        for side, color in (("left", (0, 216, 255)), ("right", (255, 79, 216))):
            for chain in HAND_CHAINS:
                for start, end in pairwise(chain):
                    a = _xy(landmarks, f"{side}_hand_{start}", minimum_score)
                    b = _xy(landmarks, f"{side}_hand_{end}", minimum_score)
                    if a and b:
                        draw.line((a, b), fill=color, width=2)

        for name, value in landmarks.items():
            if name.startswith("face_"):
                continue
            score = float(value.get("score", 0))
            if value.get("x") is None or value.get("y") is None:
                continue
            x, y = float(value["x"]), float(value["y"])
            if score >= minimum_score:
                color = (90, 255, 120)
                radius = 3
            else:
                color = (255, 80, 80)
                radius = 2
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

        pelvis_points = [
            _xy(landmarks, "left_hip", minimum_score),
            _xy(landmarks, "right_hip", minimum_score),
        ]
        if all(pelvis_points):
            x = sum(item[0] for item in pelvis_points if item) / 2
            y = sum(item[1] for item in pelvis_points if item) / 2
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=(255, 255, 255), width=3)

        label = f'{frame["id"]}  {frame["time_seconds"]:.6f}s'
        draw.rectangle((8, 8, 230, 32), fill=(0, 0, 0))
        draw.text((14, 13), label, fill=(255, 255, 255))
        image.save(destination / frame_path.name, format="PNG")



SUBJECT_PALETTE = [
    (0, 216, 255),
    (255, 79, 216),
    (72, 224, 111),
    (255, 114, 72),
    (255, 204, 0),
    (155, 123, 255),
]


def render_track_set_overlays(
    frame_paths: list[Path],
    track_set: dict[str, Any],
    destination: Path,
) -> None:
    """Render every conservative subject segment with stable color and labels."""
    destination.mkdir(parents=True, exist_ok=True)
    minimum_score = float(track_set.get("minimum_score", 0.5))
    frames_by_id: dict[str, list[tuple[int, dict[str, Any], str]]] = {}
    for subject_index, subject in enumerate(track_set.get("subjects", [])):
        subject_id = str(subject.get("subject_id", f"subject-{subject_index + 1:04d}"))
        for frame in subject.get("frames", []):
            frames_by_id.setdefault(str(frame.get("id")), []).append(
                (subject_index, frame, subject_id)
            )

    for frame_index, frame_path in enumerate(frame_paths):
        frame_id = f"f{frame_index:06d}"
        with Image.open(frame_path) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        for subject_index, frame, subject_id in sorted(
            frames_by_id.get(frame_id, []), key=lambda item: item[2]
        ):
            color = SUBJECT_PALETTE[subject_index % len(SUBJECT_PALETTE)]
            landmarks = frame.get("landmarks", {})
            for left, right in BODY_EDGES:
                a = _xy(landmarks, left, minimum_score)
                b = _xy(landmarks, right, minimum_score)
                if a and b:
                    draw.line((a, b), fill=color, width=3)
            for side in ("left", "right"):
                for chain in HAND_CHAINS:
                    for start, end in pairwise(chain):
                        a = _xy(landmarks, f"{side}_hand_{start}", minimum_score)
                        b = _xy(landmarks, f"{side}_hand_{end}", minimum_score)
                        if a and b:
                            draw.line((a, b), fill=color, width=2)
            bbox = frame.get("bbox") or []
            if len(bbox) >= 4:
                draw.rectangle(tuple(float(value) for value in bbox[:4]), outline=color, width=2)
                x, y = float(bbox[0]), max(0.0, float(bbox[1]) - 18.0)
            else:
                hips = [
                    _xy(landmarks, "left_hip", minimum_score),
                    _xy(landmarks, "right_hip", minimum_score),
                ]
                visible = [item for item in hips if item]
                x = visible[0][0] if visible else 8.0
                y = visible[0][1] if visible else 40.0
            draw.rectangle((x, y, x + 112, y + 17), fill=(0, 0, 0))
            draw.text((x + 3, y + 2), subject_id, fill=color)
        label = f"{frame_id}  {track_set.get('source', {}).get('fps', 0):.6f}fps"
        draw.rectangle((8, 8, 250, 32), fill=(0, 0, 0))
        draw.text((14, 13), label, fill=(255, 255, 255))
        image.save(destination / frame_path.name, format="PNG")
