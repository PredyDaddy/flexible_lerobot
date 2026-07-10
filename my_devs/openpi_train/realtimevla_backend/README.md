# RealtimeVLA Backend for OpenPI SO101

This directory contains an isolated RealtimeVLA-style backend for the already
validated OpenPI SO101 LoRA checkpoint. It keeps the original websocket path
untouched and adds a new HTTP `/infer` service plus a robot-side SO101 client.

## Current Scope

- Server: FastAPI HTTP service on port `18080` by default.
- Protocol: internal prototype `pickle` payload over `POST /infer`.
- Model adapters:
  - `openpi_so101`: loads the real SO101 OpenPI checkpoint.
  - `mock_so101`: returns repeated current-state actions for local smoke tests.
- Client: minimal SO101 HTTP loop that reads LeRobot observations, sends JPEG
  images plus 6D state, receives `action_list`, and executes up to
  `action_chunk_steps`.

The client defaults to `execute_actions: false`; set it explicitly before
running a real robot action loop.

## Environments

Server and robot client environments are intentionally separated.

- Server `.venv`: located at `my_devs/openpi_train/realtimevla_backend/.venv`.
  It is created from the existing `openpi_train` conda Python 3.11 environment
  with `--system-site-packages`, so it reuses the validated OpenPI/JAX/CUDA
  stack and only installs HTTP service dependencies locally.
  The server startup path intentionally does not inject this repository's
  top-level `src/` directory, because OpenPI expects the `lerobot.common`
  package from the `openpi_train` environment.
- Client: must run in the `lerobot_flex` conda environment.

Create or refresh the server environment:

```bash
my_devs/openpi_train/realtimevla_backend/scripts/create_server_venv.sh
```

## Server

Start the real OpenPI SO101 backend:

```bash
my_devs/openpi_train/realtimevla_backend/scripts/serve_so101_openpi.sh
```

Useful overrides:

```bash
PORT=18081 CONFIG=/path/to/server.yaml \
  my_devs/openpi_train/realtimevla_backend/scripts/serve_so101_openpi.sh
```

Start a mock server for protocol checks:

```bash
my_devs/openpi_train/realtimevla_backend/scripts/serve_mock.sh
```

The default real server config is:

```text
server/configs/so101_openpi.yaml
```

It points to the known working checkpoint:

```text
/data/cqy_workspace/flexible_lerobot/my_devs/openpi_train/outputs/checkpoints/pi05_so101_eraser_cup_lora/so101_lora_full_v21_10epoch_bs48_20260702_171345/11090
```

## Client

Run the robot-side HTTP client in `lerobot_flex`:

```bash
my_devs/openpi_train/realtimevla_backend/scripts/run_so101_client.sh
```

Override server URL and runtime options:

```bash
REALTIMEVLA_INFER_URL=http://SERVER_HOST:18080 \
  my_devs/openpi_train/realtimevla_backend/scripts/run_so101_client.sh \
  --execute-actions false \
  --timeout-s 60 \
  --action-chunk-steps 50 \
  --control-fps 40 \
  --policy-fps 30 \
  --interpolate-actions true \
  --boundary-blend-steps 3 \
  --max-action-delta-per-step 2.0 \
  --action-ema-alpha 0.7 \
  --async-prefetch true \
  --prefetch-after-steps 25
```

Dry-run without server or hardware access:

```bash
my_devs/openpi_train/realtimevla_backend/scripts/run_so101_client.sh --dry-run true
```

## Smoke Test

With the mock server running, send one synthetic request:

```bash
my_devs/openpi_train/realtimevla_backend/.venv/bin/python \
  my_devs/openpi_train/realtimevla_backend/scripts/smoke_http_client.py
```

Expected output includes:

```text
status=ok action_shape=(30, 6)
```

## Safety Notes

- Do not use port `8000` for this backend. Default is `18080`.
- The real OpenPI server performs a synthetic warmup request at startup. This is
  expected to take several seconds and prevents the robot client from hitting
  the first JAX compilation delay.
- The robot client timeout defaults to `60s`; a `5s` timeout is too short for
  the first real OpenPI request.
- The robot client enables async prefetch by default. With
  `--action-chunk-steps 50`, start with `--prefetch-after-steps 25`; if there is
  still a visible pause between chunks, try `15` or `10` to request the next
  chunk earlier.
- `--control-fps` controls how fast actions are sent to the robot and is
  separate from camera `robot.fps`. Raising it can make motion smoother but also
  compresses each action chunk in time, so start around `40` before trying `50`
  or `60`.
- `--interpolate-actions true` keeps model actions at `--policy-fps` timing and
  resamples them to `--control-fps`. This is usually smoother than simply
  replaying a 30Hz action chunk faster.
- `--boundary-blend-steps` gently blends the first few commands of each chunk
  from the previous command, reducing visible chunk-boundary jumps. Start with
  `3`; use `0` to disable it.
- `--max-action-delta-per-step` rate-limits every joint target between
  consecutive control ticks. This helps when LeRobot reports
  `Relative goal position magnitude had to be clamped to be safe`, because those
  warnings mean the requested target is jumping faster than the robot safety
  limit allows. Start around `2.0` at 40Hz.
- `--action-ema-alpha` applies a light low-pass filter after interpolation.
  `1.0` disables it; `0.7` is a moderate first value; smaller values are
  smoother but laggier.
- Keep the original `openpi_so101` websocket scripts as the rollback path.
- Real success requires robot-side validation by the user; local smoke tests
  only prove importability and HTTP protocol behavior.
