from __future__ import annotations

import json
from pathlib import Path

from grounded_motion_mcp.constants import COCO_WHOLEBODY_NAMES
from grounded_motion_mcp.vanguard_worker import (
    _publish_tree,
    _verify_detector_score_preservation,
)


class FakeStore:
    def upload(self, source, object_name, content_type=None):
        return {
            "object": object_name,
            "gs_uri": f"gs://fixture/{object_name}",
            "sha256": "hash",
            "size_bytes": source.stat().st_size,
            "generation": 1,
            "content_type": content_type or "application/octet-stream",
        }


def test_publish_tree_keeps_lane_and_object_identity(tmp_path: Path):
    (tmp_path / "overlay.mp4").write_bytes(b"video")
    (tmp_path / "pose-track.json").write_text(json.dumps({"schema": "track"}))
    artifacts = _publish_tree(FakeStore(), "execution", "source", tmp_path)
    assert [item["name"] for item in artifacts] == ["overlay.mp4", "pose-track.json"]
    assert all(item["lane"] == "source" for item in artifacts)
    assert artifacts[0]["object"] == "executions/execution/source/overlay.mp4"



def test_detector_score_preservation_verifier_requires_exact_unbounded_values(
    tmp_path: Path,
):
    scores = [3.25 + index / 100 for index in range(133)]
    raw = {
        "frames": [
            {
                "instances": [{"keypoint_scores": scores}],
            }
        ]
    }
    track = {
        "frames": [
            {
                "id": "f000000",
                "landmarks": {
                    name: {"score": score}
                    for name, score in zip(COCO_WHOLEBODY_NAMES, scores)
                },
            }
        ]
    }
    (tmp_path / "raw-predictions.json").write_text(json.dumps(raw))
    (tmp_path / "pose-track.json").write_text(json.dumps(track))
    result = _verify_detector_score_preservation(tmp_path)
    assert result["pass"] is True
    assert result["exact_raw_to_track_match"] is True
    assert result["above_one_count"] == 133
    assert result["maximum"] == scores[-1]

def test_production_infrastructure_reuses_billed_home_center_project():
    root = Path(__file__).resolve().parents[1]
    bootstrap = (root / "infra" / "bootstrap_gcp.sh").read_text()
    workflow = (root / ".github" / "workflows" / "deploy-production.yml").read_text()
    for source in (bootstrap, workflow):
        assert "home-center-dclar" in source
        assert "grounded-motion-dclar" not in source
        assert "286457226942" not in source
    assert "BILLING_ACCOUNT=0177BC-8C61C8-CD30A1" in bootstrap
    assert "POOL=github-actions" in bootstrap
    assert "PROVIDER=grounded-motion" in bootstrap
    assert "BUCKET=home-center-dclar-grounded-motion-canary" in bootstrap
    assert "BUCKET: home-center-dclar-grounded-motion-canary" in workflow
    assert "GOOGLE_CLOUD_PROJECT=${PROJECT}" in workflow
    assert 'gcloud run jobs add-iam-policy-binding "$JOB"' in workflow
    assert '--member "serviceAccount:$CONTROL_SERVICE_ACCOUNT"' in workflow
    assert "--role roles/run.jobsExecutorWithOverrides" in workflow
    assert 'expected_base="https://${SERVICE}-$(gcloud projects describe "$PROJECT"' in workflow
    assert 'curl --fail --silent --show-error "$expected_base/health"' in workflow
    assert 'EXPECTED_BASE="$expected_base" python3' in workflow
    assert "value(status.url)" not in workflow
