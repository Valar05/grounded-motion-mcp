"""Pinned detector presets."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelPreset:
    name: str
    backend: str
    package_version: str
    model: str
    config: str
    config_relative: str
    config_url: str
    checkpoint_url: str
    checkpoint_sha256: str
    input_size: tuple[int, int]
    landmark_count: int
    license: str
    detector_config_relative: str | None = None
    detector_config_url: str | None = None
    detector_checkpoint_url: str | None = None
    detector_checkpoint_sha256: str | None = None
    detector_cat_ids: tuple[int, ...] = (0,)

    def receipt(self) -> dict[str, object]:
        return asdict(self)


PRESETS = {
    "rtmw-x-cocktail14-384x288": ModelPreset(
        name="rtmw-x-cocktail14-384x288",
        backend="mmpose",
        package_version="1.3.2",
        model="RTMW-X Cocktail14",
        config="rtmw-x_8xb320-270e_cocktail14-384x288",
        config_relative=(
            "configs/wholebody_2d_keypoint/rtmpose/cocktail14/"
            "rtmw-x_8xb320-270e_cocktail14-384x288.py"
        ),
        config_url=(
            "https://github.com/open-mmlab/mmpose/blob/dev-1.x/configs/"
            "wholebody_2d_keypoint/rtmpose/cocktail14/"
            "rtmw-x_8xb320-270e_cocktail14-384x288.py"
        ),
        checkpoint_url=(
            "https://download.openmmlab.com/mmpose/v1/projects/rtmw/"
            "rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"
        ),
        checkpoint_sha256=(
            "f840f2044fe46cb3821b7cea86be83e1f6cba406ccd28f5475ac010412dcda95"
        ),
        input_size=(384, 288),
        landmark_count=133,
        license="Apache-2.0",
    ),
    "rtmw-x-cocktail14-multiperson-384x288": ModelPreset(
        name="rtmw-x-cocktail14-multiperson-384x288",
        backend="mmpose",
        package_version="1.3.2",
        model="RTMW-X Cocktail14 + RTMDet-m person",
        config="rtmw-x_8xb320-270e_cocktail14-384x288",
        config_relative=(
            "configs/wholebody_2d_keypoint/rtmpose/cocktail14/"
            "rtmw-x_8xb320-270e_cocktail14-384x288.py"
        ),
        config_url=(
            "https://github.com/open-mmlab/mmpose/blob/v1.3.2/configs/"
            "wholebody_2d_keypoint/rtmpose/cocktail14/"
            "rtmw-x_8xb320-270e_cocktail14-384x288.py"
        ),
        checkpoint_url=(
            "https://download.openmmlab.com/mmpose/v1/projects/rtmw/"
            "rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"
        ),
        checkpoint_sha256=(
            "f840f2044fe46cb3821b7cea86be83e1f6cba406ccd28f5475ac010412dcda95"
        ),
        input_size=(384, 288),
        landmark_count=133,
        license="Apache-2.0",
        detector_config_relative="demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py",
        detector_config_url=(
            "https://github.com/open-mmlab/mmpose/blob/v1.3.2/demo/"
            "mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py"
        ),
        detector_checkpoint_url=(
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
            "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
        ),
        detector_checkpoint_sha256=(
            "35b0c7406499e0d141dd6a0235db07c10d2bee8f891f8f4e353c16a009de30e8"
        ),
        detector_cat_ids=(0,),
    ),
}


def get_preset(name: str) -> ModelPreset:
    try:
        return PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown model preset: {name}") from exc
