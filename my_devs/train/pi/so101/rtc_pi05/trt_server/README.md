# PI0.5 RTC Pure TensorRT Server

This directory contains the pure TensorRT policy server for the RTC SO101 runtime.
It does not modify `src/lerobot/policies/rtc`. The server keeps the existing RTC
HTTP protocol and replaces only the policy backend with:

```text
runtime assets + prefix_cache TensorRT engine + denoise_step TensorRT engine
```

RTC guidance is applied in the Python denoise loop. The TensorRT engines still
run the ordinary prefix and denoise boundaries.

Before using this server, stage or build the pure TRT artifacts:

```bash
conda run --no-capture-output -n lerobot_flex \
python my_devs/train/pi/so101/pure_trt/scripts/stage_existing_artifacts.py \
  --mode symlink \
  --force
```

## Safe Checks

Argument and file-path check only:

```bash
conda run --no-capture-output -n lerobot_flex \
python my_devs/train/pi/so101/rtc_pi05/trt_server/run_trt_policy_server.py \
  --runtime-assets-dir my_devs/train/pi/so101/pure_trt/artifacts/pi05_runtime_assets \
  --profile fp16_constrained \
  --prefix-engine-path my_devs/train/pi/so101/pure_trt/artifacts/pi05_so101_prefix_cache_b1_fp16_constrained.engine \
  --denoise-engine-path my_devs/train/pi/so101/pure_trt/artifacts/pi05_so101_denoise_step_b1_fp16_constrained.engine \
  --enable-rtc true \
  --dry-run true
```

Load TensorRT engines and processors, but do not start robot control:

```bash
conda run --no-capture-output -n lerobot_flex \
python my_devs/train/pi/so101/rtc_pi05/trt_server/run_trt_policy_server.py \
  --host 127.0.0.1 \
  --port 8090 \
  --runtime-assets-dir my_devs/train/pi/so101/pure_trt/artifacts/pi05_runtime_assets \
  --profile fp16_constrained \
  --prefix-engine-path my_devs/train/pi/so101/pure_trt/artifacts/pi05_so101_prefix_cache_b1_fp16_constrained.engine \
  --denoise-engine-path my_devs/train/pi/so101/pure_trt/artifacts/pi05_so101_denoise_step_b1_fp16_constrained.engine \
  --enable-rtc true \
  --rtc-execution-horizon 10 \
  --rtc-max-guidance-weight 10.0 \
  --check-policy-load true
```

## Real Robot Run

Terminal A, start the TensorRT policy server:

```bash
conda run --no-capture-output -n lerobot_flex \
python my_devs/train/pi/so101/rtc_pi05/trt_server/run_trt_policy_server.py \
  --host 127.0.0.1 \
  --port 8090 \
  --runtime-assets-dir my_devs/train/pi/so101/pure_trt/artifacts/pi05_runtime_assets \
  --profile fp16_constrained \
  --prefix-engine-path my_devs/train/pi/so101/pure_trt/artifacts/pi05_so101_prefix_cache_b1_fp16_constrained.engine \
  --denoise-engine-path my_devs/train/pi/so101/pure_trt/artifacts/pi05_so101_denoise_step_b1_fp16_constrained.engine \
  --enable-rtc true \
  --rtc-execution-horizon 10 \
  --rtc-max-guidance-weight 10.0
```

Terminal B, run the TensorRT RTC robot client wrapper:

```bash
conda run --no-capture-output -n lerobot_flex \
python my_devs/train/pi/so101/rtc_pi05/trt_server/run_trt_robot_client.py \
  --server-url http://127.0.0.1:8090 \
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
  --camera-fps 30 \
  --control-fps 30 \
  --task "Put the eraser into the small box" \
  --run-time-s 120 \
  --enable-rtc true \
  --queue-low-watermark 4 \
  --queue-target-size 12 \
  --max-queue-size 50 \
  --first-chunk-timeout-s 60 \
  --rtc-execution-horizon 10 \
  --metrics-log-interval-s 2 \
  --assume-calibrated true \
  --motor-io-retries 20 \
  --confirm-control
```

For a camera/robot connection smoke that sends no actions, use the same client
command with `--connect-smoke true` and without `--confirm-control`.
