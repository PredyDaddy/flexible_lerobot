# openpi_trt

`my_devs/openpi_trt` is the LeRobot PI0.5 TensorRT deployment module for the
trained checkpoint:

```text
outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model
```

The module now keeps one production path:

```text
checkpoint -> prefix_cache.onnx + denoise_step.onnx
           -> prefix_cache fp32 engine + denoise_step fp32/fp16_constrained engine
           -> split TensorRT sample_actions runtime
           -> SO101 robot runner
```

## Current Layout

```text
my_devs/openpi_trt/
  runtime/
    trt_engine.py           # TensorRT engine loader with torch tensor I/O
    protocol.py             # stable tensor-name protocol
    config.py               # split runtime config
    metadata.py             # optional artifact metadata helper
    simple_pi05_split.py    # production compact split TensorRT runtime
    pi05_trt_split.py       # compatibility wrapper for vla_engineering

  scripts/
    pi05_onnx_common.py              # PI0.5 load/export helper code
    simple_pi05_pipeline.py          # export, convert, validate
    simple_pi05_validate.py          # validate-only wrapper
    simple_pi05_run_robot.py         # real SO101 runner
    run_pi05_split_trt_infer_so101.py # old command wrapper
```

`openpi_on_thor/` is kept only as the downloaded upstream reference from Jetson
AI Lab. It is not the active LeRobot deployment path.

## Export, Convert, Validate

Run fp32:

```bash
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/simple_pi05_pipeline.py \
  --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
  --profile fp32 \
  --report my_devs/openpi_trt/artifacts/simple_pipeline_fp32_report.json
```

Run fp16 constrained:

```bash
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/simple_pi05_pipeline.py \
  --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
  --profile fp16_constrained \
  --report my_devs/openpi_trt/artifacts/simple_pipeline_fp16_constrained_report.json
```

Useful switches:

- `--force-export`: rebuild ONNX files.
- `--force-convert`: rebuild TensorRT engines.
- `--validate-only`: reuse existing ONNX/engine artifacts and compare Torch vs TensorRT.
- `--export-only`: only export ONNX.
- `--convert-only`: only build TensorRT engines.

## Real Robot

fp32:

```bash
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/simple_pi05_run_robot.py \
  --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
  --profile fp32 \
  --prefix-engine-path my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp32.engine \
  --denoise-engine-path my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp32.engine \
  --robot-id hfy_follower \
  --robot-type so101_follower \
  --calib-dir /home/cqy/.cache/huggingface/lerobot/calibration/robots/so_follower \
  --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00 \
  --top-cam /dev/video4 \
  --wrist-cam /dev/video6 \
  --top-cam-fourcc YUYV \
  --wrist-cam-fourcc MJPG \
  --img-width 640 \
  --img-height 480 \
  --fps 30 \
  --task "Put the eraser into the small box" \
  --run-time-s 120 \
  --log-interval 1 \
  --motor-write-retries 0 \
  --confirm-control
```

fp16 constrained:

```bash
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/simple_pi05_run_robot.py \
  --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
  --profile fp16_constrained \
  --prefix-engine-path my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp32.engine \
  --denoise-engine-path my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp16_constrained.engine \
  --robot-id hfy_follower \
  --robot-type so101_follower \
  --calib-dir /home/cqy/.cache/huggingface/lerobot/calibration/robots/so_follower \
  --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00 \
  --top-cam /dev/video4 \
  --wrist-cam /dev/video6 \
  --top-cam-fourcc YUYV \
  --wrist-cam-fourcc MJPG \
  --img-width 640 \
  --img-height 480 \
  --fps 30 \
  --task "Put the eraser into the small box" \
  --run-time-s 120 \
  --log-interval 1 \
  --motor-write-retries 0 \
  --confirm-control
```

The runner never touches hardware unless `--confirm-control` is present.

## Compatibility

`my_devs/vla_engineering` imports:

```python
from runtime.pi05_trt_split import patch_sample_actions_with_split_trt
```

That import path is intentionally preserved. The file is now a thin wrapper over
`runtime.simple_pi05_split`, so the async Tesseract backend keeps working while
the implementation stays in one place.
