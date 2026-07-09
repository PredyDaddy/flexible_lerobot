# SO101 PI0.5 Pure TensorRT

This module is the clean TensorRT deployment workspace for the SO101 PI0.5 policy.
It is meant to replace direct day-to-day use of `my_devs/openpi_trt` for robot
deployment work.

The larger `my_devs/train/pi/so101` workflow is intended to become:

```text
training/checkpoints
  -> pure_trt export/build/validate/runtime assets
  -> rtc_pi05 async RTC inference using the pure_trt engines
```

## Layout

```text
my_devs/train/pi/so101/pure_trt/
  artifacts/                 # staged or newly built ONNX/engine/runtime assets
  runtime/paths.py            # stable artifact path source of truth
  runtime/pure_pi05_trt.py    # facade over the validated TensorRT runtime
  scripts/stage_existing_artifacts.py
  scripts/build_runtime.py
  scripts/check_runtime.py
```

`my_devs/openpi_trt` is still used as the low-level implementation source while
this module is being cleaned up. New commands and RTC defaults should point here.

## 1. Stage Existing Validated Artifacts

This is the fast path. It reuses the already validated engines from
`my_devs/openpi_trt/artifacts`, stages large files as symlinks, and copies the
small runtime assets into this module.

```bash
conda run --no-capture-output -n lerobot_flex \
python my_devs/train/pi/so101/pure_trt/scripts/stage_existing_artifacts.py \
  --mode symlink \
  --force
```

After this, RTC can use:

```text
my_devs/train/pi/so101/pure_trt/artifacts/pi05_runtime_assets
my_devs/train/pi/so101/pure_trt/artifacts/pi05_so101_prefix_cache_b1_fp16_constrained.engine
my_devs/train/pi/so101/pure_trt/artifacts/pi05_so101_denoise_step_b1_fp16_constrained.engine
```

## 2. Load Check

Load the TensorRT engines and processors without touching robot hardware:

```bash
conda run --no-capture-output -n lerobot_flex \
python my_devs/train/pi/so101/pure_trt/scripts/check_runtime.py
```

Run one dummy image/state inference through pure TRT:

```bash
conda run --no-capture-output -n lerobot_flex \
python my_devs/train/pi/so101/pure_trt/scripts/check_runtime.py \
  --dummy-infer
```

## 3. Export / Build / Validate From A Checkpoint

The clean pipeline outputs to `pure_trt/artifacts` by default.

Validate using staged artifacts only:

```bash
conda run --no-capture-output -n lerobot_flex \
python my_devs/train/pi/so101/pure_trt/scripts/build_runtime.py \
  --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
  --profile fp16_constrained \
  --runtime-backend pure_trt \
  --prefix-precision fp16_constrained \
  --validate-only \
  --report my_devs/train/pi/so101/pure_trt/artifacts/validate_fp16_pure_trt.json
```

Re-export ONNX and rebuild both prefix and denoise engines:

```bash
conda run --no-capture-output -n lerobot_flex \
python my_devs/train/pi/so101/pure_trt/scripts/build_runtime.py \
  --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
  --profile fp16_constrained \
  --runtime-backend pure_trt \
  --prefix-precision fp16_constrained \
  --export-prefix-onnx \
  --build-prefix-engine \
  --force-export \
  --force-convert \
  --report my_devs/train/pi/so101/pure_trt/artifacts/build_fp16_pure_trt.json
```

## 4. RTC Async Inference

The RTC TensorRT server defaults now point to this module's artifacts. Start the
server:

```bash
conda run --no-capture-output -n lerobot_flex \
python my_devs/train/pi/so101/rtc_pi05/trt_server/run_trt_policy_server.py \
  --host 127.0.0.1 \
  --port 8090 \
  --enable-rtc true \
  --rtc-execution-horizon 10 \
  --rtc-max-guidance-weight 10.0
```

Run the robot client in another terminal. Real motion still requires
`--confirm-control`.
