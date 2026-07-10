# RealtimeVLA Backend Implementation Status

## Completed in This Pass

- Created an isolated Python `.venv` inside `realtimevla_backend`.
- Added service-side requirements and a repeatable environment creation script.
- Added a FastAPI `/infer` server with:
  - request validation,
  - global inference lock,
  - `infer_time` reporting,
  - optional startup warmup for real OpenPI/JAX inference,
  - `action_list` and optional `raw_action_list` response fields.
- Added two server adapters:
  - `mock_so101` for protocol smoke tests,
  - `openpi_so101` for the existing 10 epoch LoRA checkpoint.
- Added a minimal SO101 HTTP client that:
  - runs under `lerobot_flex`,
  - collects top/wrist images and 6D robot state,
  - encodes images as JPEG bytes,
  - sends HTTP pickle payloads,
  - can prefetch the next HTTP inference request asynchronously while executing
    the current chunk,
  - supports a separate `control_fps` for action send rate,
  - optionally interpolates 30Hz policy actions to a higher control rate,
  - optionally blends chunk boundaries for smoother transitions,
  - optionally rate-limits and low-pass filters action targets to reduce
    safety clamp induced reversals,
  - executes up to `action_chunk_steps` actions from each returned chunk.
- Added scripts for:
  - server venv creation,
  - real OpenPI server startup,
  - mock server startup,
  - SO101 client startup,
  - synthetic HTTP smoke request.

## Current Boundaries

- All implementation files are inside `my_devs/openpi_train/realtimevla_backend`.
- The original websocket backend remains untouched and should stay available as
  the rollback path.
- The first protocol is intentionally `pickle` over HTTP to match the reference
  backend's prototype shape.
- The real server config enables startup warmup because the first JAX/OpenPI
  request can exceed a short 5-second HTTP timeout.
- MPC, advanced heartbeat scheduling, action prefill, and smoothing are not yet
  implemented; they should be layered on after the minimal loop is validated.

## Manual Robot Validation Still Required

Codex can validate imports and mock HTTP behavior, but cannot prove real robot
success. User-side validation should check:

- real OpenPI server starts on `18080`,
- robot client connects from `lerobot_flex`,
- `action_list` is returned repeatedly,
- a full `action_chunk_steps=30` chunk can execute,
- behavior is no worse than the original websocket baseline.

## Verified Locally

- Server `.venv` exists and uses Python 3.11.15 from `openpi_train` with
  `include-system-site-packages = true`.
- Server `.venv` can import `openpi`, `fastapi`, and `uvicorn`.
- `server/configs/so101_openpi.yaml` parses and points to the expected
  10 epoch checkpoint.
- The real `openpi_so101` server adapter can construct an `InferPipeline` and
  load the 10 epoch checkpoint on CUDA.
- The real `openpi_so101` pipeline can run one synthetic top/wrist image +
  6D-state request and returns a finite `(30, 6)` action chunk.
- `scripts/serve_so101_openpi.sh` starts the real HTTP service, loads the
  checkpoint, and serves a synthetic `POST /infer` request returning
  `action_shape=(30, 6)`.
- The expected checkpoint directory, v2.1 dataset directory, and norm stats file
  are present on disk.
- Client dry-run succeeds under `lerobot_flex` without server or hardware access.
- Client timeout defaults to `60s`, matching the observed real OpenPI first
  request latency more safely than the original `5s`.
- Async prefetch is enabled by default with a configurable
  `prefetch_after_steps` trigger.
- Python compile checks pass for server, client, and smoke-test code.
- `scripts/serve_mock.sh` starts the mock FastAPI server on `18080`.
- `scripts/smoke_http_client.py` receives `action_shape=(30, 6)` from
  `POST /infer` against the mock server.
