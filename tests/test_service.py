from __future__ import annotations

import shutil
from pathlib import Path

from grounded_motion_mcp.artifacts import verify_manifest
from grounded_motion_mcp.backend import RawFrame, RawInstance
from grounded_motion_mcp.models import ExportRequest, InspectRequest, TrackRequest
from grounded_motion_mcp.service import GroundedMotionService


class FakeBackend:
    def __init__(self, preset, device: str) -> None:
        self.preset = preset
        self.device = device

    @property
    def receipt(self) -> dict[str, object]:
        return {
            "name": "fixture",
            "model": self.preset.model,
            "version": "1",
            "license": "test",
            "device": self.device,
            "config_sha256": "c" * 64,
            "model_sha256": "d" * 64,
        }

    def infer(self, frames, fps, crop, overlay_dir):
        result = []
        for frame_index, _ in enumerate(frames):
            keypoints = []
            for index in range(133):
                x = crop.x + 20 + (index % 10) * 2 + frame_index
                y = crop.y + 20 + (index // 10) * 2
                keypoints.append([float(x), float(y)])
            result.append(
                RawFrame(
                    frame_id=f"f{frame_index:06d}",
                    frame_index=frame_index,
                    time_seconds=frame_index / fps,
                    instances=[
                        RawInstance(
                            keypoints=keypoints,
                            keypoint_scores=[0.99] * 133,
                            bbox=[0, 0, crop.width, crop.height],
                            bbox_score=0.99,
                        )
                    ],
                )
            )
        return result


def fake_backend_factory(preset, device, model_cache):
    return FakeBackend(preset, device)


def test_real_video_orchestration_is_content_addressed_and_exportable(
    tmp_path: Path,
) -> None:
    fixture = Path(__file__).parent / "fixtures" / "source.mp4"
    source = tmp_path / "source.mp4"
    shutil.copy2(fixture, source)
    service = GroundedMotionService(tmp_path, backend_factory=fake_backend_factory)

    first = service.track_motion(TrackRequest(source_path=str(source), device="cpu"))
    second = service.track_motion(TrackRequest(source_path=str(source), device="cpu"))

    assert first["job_id"] == second["job_id"]
    assert not first["cached"]
    assert second["cached"]
    job_path = Path(first["job_path"])
    assert verify_manifest(job_path)["pass"]
    assert (job_path / "overlay.mp4").stat().st_size > 0
    assert (job_path / "overlay-slow.mp4").stat().st_size > 0

    inspection = service.inspect_track(
        InspectRequest(track_path=str(job_path / "pose-track.json"))
    )
    assert inspection["structural_pass"]
    assert not inspection["production_pass"]
    assert inspection["state"] == "tracked"

    exported = service.export_artifacts(ExportRequest(job_path=str(job_path)))
    assert Path(exported["path"]).is_file()
    assert exported["size_bytes"] > 0
