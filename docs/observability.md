# Observability & Support Bundles

Argos now emits structured telemetry for every run. The logging backbone lives in
`projects/argos/panoptes/logging_config.py` and is initialised automatically by the
bootstrapper, CLI entry points, and Lambda handler.

## What gets recorded

- A per-run directory at `projects/argos/tests/results/runs/<timestamp>_<uuid>/`.
- Human-readable console output, plus `argos.log` containing JSONL events.
- `env.json` with system information (OS, Python runtime, Torch/ONNX/runtime provider data).
- Offline runs capture input/output artefacts and errors; live runs report camera backend,
  codec attempts, FPS summaries, and stop reasons.
- Lambda requests log request IDs, phase timings, and S3 uploads.

## Generating support bundles

To compress the key artefacts for sharing, enable the `--support-bundle` flag when
invoking the CLI:

```bash
./run --support-bundle hm tests/raw/assets.jpg
./livevideo synthetic --duration 5 --support-bundle
```

Each command writes `support_<timestamp>.zip` inside the current run directory containing:

- `argos.log` and `env.json`
- Optional additional run files (e.g., diagnostics, cached results)
- A `support.json` manifest listing the included files and metadata
- Any explicitly requested extra artefacts (result images, live output video)

You can also call `projects/argos/panoptes/support_bundle.write_support_bundle()` directly
from scripts or notebooks to capture the current run.

## Log formats and levels

Set `ARGOS_LOG_LEVEL=DEBUG` to increase verbosity, or `ARGOS_LOG_FORMAT=json` to force
JSON output to stdout. The file handler always writes JSON lines for downstream tooling.

## Troubleshooting workflow

1. Reproduce the issue with `--support-bundle`.
2. Attach the generated zip (from the run directory) to your support ticket.
3. For Lambda issues, correlate `lambda_request_id` in CloudWatch with the bundled logs.

These bundles provide enough context for another engineer (or another AI assistant) to
replay what happened without needing direct access to your environment.
