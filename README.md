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
