# Grounded Motion Agent Law

This repository is a deterministic measuring instrument, not an animator.

- No pose graph, no redraw.
- Source pixels own motion, chronology, contacts, and ambiguity.
- Feet establish support; hips answer to the feet; hands define weapon mechanics.
- Unknown joints remain unknown. Never smooth, interpolate, swap, or hallucinate them into acceptance.
- Preserve raw predictions beside reviewed landmarks.
- Track the complete demonstrated interval before selecting sparse keys.
- A successful inference run is `tracked`, not `reviewed`, `event-locked`, or human accepted.
- STDIO writes protocol data only to stdout; diagnostics go to stderr.
- Local and HTTP lanes execute the same service code and produce the same content-addressed artifacts.

Before changing the normalized schema, update fixtures, comparison tests, README examples, and
the schema version together. Before changing a model preset, record the upstream config,
checkpoint URL, hashes, package versions, and license in the receipt.

Required verification:

```bash
uv run --extra dev pytest
uv run grounded-motion inspect-video tests/fixtures/source.mp4
uv run python scripts/mcp_smoke.py
uv run python scripts/stdio_smoke.py
uv run python scripts/http_smoke.py
```

Never report real MMPose inference as proven unless the pinned inference dependencies and model
weights actually ran. Fixture and fake backends prove orchestration only.
