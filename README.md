# grounded-motion-mcp

`grounded-motion-mcp` is a standalone, agent-free motion tracking appliance. It runs pinned
MMPose RTMW whole-body inference, preserves all 133 raw landmarks, normalizes the motion into a
reviewable `grounded-motion-track/v1` graph, and produces evidence for feet, pelvis/root, hands,
confidence, occlusion, and source chronology.

The model is a sensor. It does not author motion or certify its own output.

## What it produces

Every content-addressed job contains:

- `raw-predictions.json` — immutable detector output.
- `pose-track.json` — normalized landmarks with provenance.
- `pose-track-report.json` — structural and production-gate findings.
- `trajectories.svg` — pelvis, wrists, heels, and big-toe paths.
- `overlay.mp4` and `overlay-slow.mp4` — full-speed and slow evidence.
- `manifest.json` — artifact paths, sizes, and SHA-256 hashes.
- `receipt.json` — exact input, backend, versions, device, status, and job identity.

Inference deliberately ends in `tracked/unreviewed`. `validate_track` with the production gate
enabled fails until required landmarks have been reviewed and the event map is locked.

## MCP tools

- `track_motion`
- `validate_track`
- `inspect_track`
- `compare_motion`
- `export_artifacts`

The default transport is local STDIO. The optional Streamable HTTP lane uses the same service
code and expects files to be mounted under the configured workspace root.

## Install

Core tools and MCP server:

```bash
uv sync --extra dev
```

MMPose inference requires OpenMMLab's compiled runtime. The repeatable path is the supplied
container:

```bash
docker build -t grounded-motion-mcp .
```

For a native install, sync the pinned inference stack, then let OpenMIM install the matching
compiled MMCV build into that environment:

```bash
uv sync --extra inference --extra dev
uv run mim install "mmcv==2.1.0"
```

## Local CLI

```bash
uv run grounded-motion --workspace /absolute/path/workspace track \
  /absolute/path/source.mp4 \
  --device cpu

uv run grounded-motion --workspace /absolute/path/workspace validate \
  /absolute/path/workspace/grounded-motion/jobs/<job-id>/pose-track.json \
  --production

uv run grounded-motion --workspace /absolute/path/workspace export \
  /absolute/path/workspace/grounded-motion/jobs/<job-id>
```

Use `--crop x,y,width,height` to lock a single subject crop. Coordinates are source pixels.
The entire source interval is decoded without resampling.

## MCP configuration

STDIO:

```json
{
  "mcpServers": {
    "grounded-motion": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/grounded-motion-mcp",
        "run",
        "grounded-motion-mcp"
      ],
      "env": {
        "GROUNDED_MOTION_WORKSPACE": "/absolute/path/motion-workspace"
      }
    }
  }
}
```

Agent-free HTTP appliance:

```bash
docker run --rm --gpus all -p 8000:8000 \
  -v /absolute/path/data:/data \
  -e GROUNDED_MOTION_TRANSPORT=streamable-http \
  -e GROUNDED_MOTION_WORKSPACE=/data \
  grounded-motion-mcp
```

The endpoint is `/mcp`. Do not expose it publicly without authentication and an origin policy.

## Model preset

The production default is MMPose 1.3.2 RTMW-X Cocktail14 at 384×288:

- 133 COCO-WholeBody landmarks.
- Apache-2.0 MMPose code.
- Explicit body, six foot, face, and 21 landmarks per hand.
- Whole-image top-down inference over a caller-locked single-subject crop.

The preset records the upstream config and checkpoint URL in every receipt. Downloaded weights
must be cached and hashed before production use.

## Completion states

`tracked` → `reviewed` → `event-locked` → `keyed` → `transferred` →
`mechanically-compared` → `human-accepted`

No earlier state implies a later one.

## ChatGPT Vanguard production canary

The production profile is deliberately narrower than the local appliance. It exposes exactly:

- `grounded_motion_v2.start_vanguard_canary()`
- `grounded_motion_v2.get_vanguard_canary_status(execution_id)`
- `grounded_motion_v2.get_vanguard_canary_result(execution_id)`

`start` launches a one-task Cloud Run L4 GPU Job. The job tracks the immutable canonical Vanguard
Walk v1 and quarantined WalkSwordCarryV2 candidate 003 through the same `GroundedMotionService`
used by the CLI, verifies both manifests, runs the existing mechanical comparison, and publishes
private GCS evidence. `result` issues fresh 24-hour signed URLs. `pipeline_pass` proves real pinned
MMPose inference, structural track validity, artifact readback, and comparison completion;
`mechanical_pass` can honestly be false, and neither value means human acceptance.

The fixtures preserve the eight source PNGs at Pose Lab commit
`90ca534c46a47c660e7bf5ef7bd2efcf35dbeb9e` and the eight candidate PNGs at immutable revision
`b2c5bde5d91325726af34e5daea17b96d78b46f3`. They are assembled as 82 frames at 100 fps using
repeats `11/9/10/11/11/9/10/11`, with no interpolation. The paths, Git blob ids, file hashes,
video hashes, candidate quarantine status, and timing live in
`src/grounded_motion_mcp/data/vanguard_canary.json`.

Production authentication delegates identity only to Home Center OAuth. Tokens must be bound to
the production `/mcp` resource, carry `grounded-motion:vanguard-canary`, and identify
`dclarke1005@gmail.com`. Grounded Motion receives no Drive scope or Google refresh token.

Infrastructure is prepared by `infra/bootstrap_gcp.sh` inside the existing billed
`home-center-dclar` project, using isolated Grounded Motion service accounts, Artifact Registry,
bucket, Cloud Run service, GPU job, and a repository/main-constrained provider in the existing
`github-actions` WIF pool. The bootstrap refuses to continue unless the exact materialized videos
pass their stored SHA-256 values. Pull requests run `.github/workflows/ci.yml`. Every merge to `main` runs
`.github/workflows/deploy-production.yml`, builds the exact commit, deploys that image to both the
CPU control service and one-task L4 job, verifies the OAuth challenge, and completes a real GPU
canary before the deployment is green.
