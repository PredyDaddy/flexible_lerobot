# PI0.5 RTC Server/Client Runtime

This directory contains the service-style PI0.5 RTC runtime for SO101 follower inference.
The policy model runs in one process, and the robot/camera control client runs in another process.
For the current test phase, both processes are expected to run on the same machine through
`127.0.0.1` or `localhost`.

## Architecture

```text
Terminal A
  run_policy_server.py
    - load PI0.5 checkpoint
    - enable RTC on the policy
    - expose /health and /infer on HTTP

Terminal B
  run_robot_client.py
    - connect SO101 follower + cameras
    - sensor loop: capture camera observations at camera_fps
    - producer loop: request action chunks from the policy server
    - actor loop: send robot actions at control_fps
```

The client intentionally separates camera acquisition and robot control:

- `camera_fps` controls how often the client captures a full visual observation.
- `control_fps` controls how often the client consumes queued actions and sends commands to the robot.
- The producer loop tracks request latency and converts it into control-rate `drop_steps`, so higher
  `control_fps` is reflected in RTC chunk alignment.

This makes a setup such as `--camera-fps 30 --control-fps 40` valid even when the wrist camera cannot
capture above 30 FPS.

## Start The Policy Server

```bash
conda run --no-capture-output -n lerobot_flex \
python my_devs/train/pi/so101/rtc_pi05/server/run_policy_server.py \
  --host 127.0.0.1 \
  --port 8088 \
  --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
  --enable-rtc true \
  --rtc-execution-horizon 10 \
  --rtc-max-guidance-weight 10.0
```

For a load-only smoke test:

```bash
conda run --no-capture-output -n lerobot_flex \
python my_devs/train/pi/so101/rtc_pi05/server/run_policy_server.py \
  --check-policy-load true \
  --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
  --enable-rtc true
```

## Start The Robot Client

Use the follower port that has been verified on this machine:

```bash
conda run --no-capture-output -n lerobot_flex \
python my_devs/train/pi/so101/rtc_pi05/server/run_robot_client.py \
  --server-url http://127.0.0.1:8088 \
  --assume-calibrated true \
  --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00 \
  --top-cam /dev/video4 \
  --wrist-cam /dev/video6 \
  --task "Put the eraser into the small box" \
  --run-time-s 120 \
  --camera-fps 30 \
  --control-fps 40 \
  --enable-rtc true \
  --queue-low-watermark 4 \
  --queue-target-size 12 \
  --max-queue-size 50 \
  --first-chunk-timeout-s 60 \
  --rtc-execution-horizon 10 \
  --max-relative-target 10 \
  --metrics-log-interval-s 2
```

The known leader/main-arm port is rejected by default:

```text
/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123582-if00
```

## Non-Motion Checks

Client argument/config check only:

```bash
conda run --no-capture-output -n lerobot_flex \
python my_devs/train/pi/so101/rtc_pi05/server/run_robot_client.py \
  --dry-run true \
  --server-url http://127.0.0.1:8088 \
  --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00 \
  --camera-fps 30 \
  --control-fps 40
```

Connect robot/cameras, read one observation, and exit without sending actions:

```bash
conda run --no-capture-output -n lerobot_flex \
python my_devs/train/pi/so101/rtc_pi05/server/run_robot_client.py \
  --server-url http://127.0.0.1:8088 \
  --connect-smoke true \
  --assume-calibrated true \
  --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00 \
  --top-cam /dev/video4 \
  --wrist-cam /dev/video6 \
  --camera-fps 30 \
  --control-fps 40
```

## Verification

```bash
conda run --no-capture-output -n lerobot_flex \
python -m py_compile my_devs/train/pi/so101/rtc_pi05/server/*.py tests/my_devs/rtc_pi05_server/*.py

conda run --no-capture-output -n lerobot_flex \
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/my_devs/rtc_pi05_server tests/my_devs/rtc_pi05
```

Expected current test result:

```text
12 passed
```
