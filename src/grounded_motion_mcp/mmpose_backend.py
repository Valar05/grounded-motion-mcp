"""MMPose RTMW backend loaded lazily so audit-only installs remain lightweight."""

from __future__ import annotations

import importlib.metadata
import os
import sys
from pathlib import Path
from typing import Any

from PIL import Image

from .backend import RawFrame, RawInstance
from .hashing import sha256_file
from .model_cache import acquire_checkpoint
from .models import Crop
from .presets import ModelPreset


class MMPoseUnavailable(RuntimeError):
    pass


class MMPoseBackend:
    def __init__(
        self,
        preset: ModelPreset,
        device: str = "auto",
        model_cache: Path | None = None,
    ) -> None:
        self.preset = preset
        self.device = None if device == "auto" else device
        try:
            import mmpose
            from mmpose.apis import MMPoseInferencer
        except ImportError as exc:
            raise MMPoseUnavailable(
                "MMPose inference dependencies are not installed. "
                "Use the container or install the inference extra plus a compatible MMCV build."
            ) from exc

        self._versions = {}
        for package in ("mmpose", "mmengine", "mmcv", "torch", "torchvision"):
            try:
                self._versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                self._versions[package] = "missing"

        package_root = Path(mmpose.__file__).resolve().parent
        config_path = package_root / ".mim" / preset.config_relative
        if not config_path.is_file():
            raise MMPoseUnavailable(
                f"Pinned MMPose config is missing from the installed package: {config_path}"
            )
        cache_root = model_cache or Path(
            os.environ.get("GROUNDED_MOTION_MODEL_CACHE", Path.cwd() / "models")
        )
        checkpoint_path, checkpoint_sha = acquire_checkpoint(
            preset.checkpoint_url,
            cache_root.expanduser().resolve(),
        )
        if checkpoint_sha != preset.checkpoint_sha256:
            raise MMPoseUnavailable(
                "Pinned checkpoint SHA-256 mismatch: "
                f"expected {preset.checkpoint_sha256}, got {checkpoint_sha}"
            )
        self._config_path = config_path
        self._config_sha = sha256_file(config_path)
        self._checkpoint_path = checkpoint_path
        self._checkpoint_sha = checkpoint_sha

        self.inferencer = MMPoseInferencer(
            pose2d=str(config_path),
            pose2d_weights=str(checkpoint_path),
            det_model="whole_image",
            device=self.device,
        )

    @property
    def receipt(self) -> dict[str, object]:
        return {
            "name": "mmpose",
            "model": self.preset.model,
            "preset": self.preset.receipt(),
            "version": self._versions.get("mmpose", "unknown"),
            "license": self.preset.license,
            "device": self.device or "auto",
            "config_path": str(self._config_path),
            "config_sha256": self._config_sha,
            "checkpoint_path": str(self._checkpoint_path),
            "model_sha256": self._checkpoint_sha,
            "packages": self._versions,
            "python": sys.version.split()[0],
        }

    def infer(
        self,
        frames: list[Path],
        fps: float,
        crop: Crop,
        overlay_dir: Path,
    ) -> list[RawFrame]:
        overlay_dir.mkdir(parents=True, exist_ok=True)
        crop_dir = overlay_dir.parent / "crops"
        crop_dir.mkdir(parents=True, exist_ok=True)
        results: list[RawFrame] = []

        for index, frame_path in enumerate(frames):
            with Image.open(frame_path) as image:
                crop_image = image.crop(
                    (crop.x, crop.y, crop.x + crop.width, crop.y + crop.height)
                )
                crop_path = crop_dir / frame_path.name
                crop_image.save(crop_path, format="PNG")

            generator = self.inferencer(
                str(crop_path),
                return_vis=False,
                show=False,
            )
            result = next(generator)
            instances = _extract_instances(result)
            translated = [_translate_instance(item, crop.x, crop.y) for item in instances]
            results.append(
                RawFrame(
                    frame_id=f"f{index:06d}",
                    frame_index=index,
                    time_seconds=index / fps,
                    instances=translated,
                )
            )
        return results


def _extract_instances(result: dict[str, Any]) -> list[RawInstance]:
    predictions = result.get("predictions", [])
    while (
        isinstance(predictions, list)
        and len(predictions) == 1
        and isinstance(predictions[0], list)
    ):
        predictions = predictions[0]
    if not isinstance(predictions, list):
        raise TypeError("MMPose returned an unsupported predictions shape")

    instances = []
    for item in predictions:
        if not isinstance(item, dict):
            continue
        keypoints = _to_list(item.get("keypoints", []))
        scores = _to_list(item.get("keypoint_scores", []))
        if keypoints and isinstance(keypoints[0], list) and len(keypoints) == 1:
            keypoints = keypoints[0]
        if scores and isinstance(scores[0], list) and len(scores) == 1:
            scores = scores[0]
        bbox = _to_list(item.get("bbox")) if item.get("bbox") is not None else None
        if bbox and isinstance(bbox[0], list):
            bbox = bbox[0]
        bbox_score = item.get("bbox_score")
        if hasattr(bbox_score, "item"):
            bbox_score = bbox_score.item()
        instances.append(
            RawInstance(
                keypoints=[[float(value) for value in pair[:2]] for pair in keypoints],
                keypoint_scores=[float(value) for value in scores],
                bbox=[float(value) for value in bbox] if bbox else None,
                bbox_score=float(bbox_score) if bbox_score is not None else None,
            )
        )
    return instances


def _to_list(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _translate_instance(instance: RawInstance, offset_x: int, offset_y: int) -> RawInstance:
    return RawInstance(
        keypoints=[
            [float(pair[0]) + offset_x, float(pair[1]) + offset_y]
            for pair in instance.keypoints
        ],
        keypoint_scores=instance.keypoint_scores,
        bbox=(
            [
                instance.bbox[0] + offset_x,
                instance.bbox[1] + offset_y,
                instance.bbox[2] + offset_x,
                instance.bbox[3] + offset_y,
            ]
            if instance.bbox and len(instance.bbox) >= 4
            else instance.bbox
        ),
        bbox_score=instance.bbox_score,
    )
