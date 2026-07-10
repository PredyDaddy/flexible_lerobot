# Head Camera Overlay

Small browser tool for visual calibration with the JZ robot head camera.

It only reads the head camera RTSP stream and serves it as MJPEG. It does not
send robot commands.

## Run

Use the repository conda environment:

```bash
conda run -n lerobot_flex python tests/visual_cali/head_camera_overlay.py
```

Open:

```text
http://127.0.0.1:8090
```

The default camera is `camera_head` from the current UDP RTSP config:

```text
src/lerobot/configs/robot/jz_robot_udp_three_rtsp.yaml
```

## Initial Reference Image

When you already have the reference image path:

```bash
conda run -n lerobot_flex python tests/visual_cali/head_camera_overlay.py \
  --reference-image /path/to/reference.jpg
```

You can also choose an image directly from the browser.

## Useful Overrides

Use another RTSP URL:

```bash
conda run -n lerobot_flex python tests/visual_cali/head_camera_overlay.py \
  --rtsp-url rtsp://192.168.50.10:8554/robot_camera/camera_head
```

Use the older bridge capture config:

```bash
conda run -n lerobot_flex python tests/visual_cali/head_camera_overlay.py \
  --config src/lerobot/configs/robot/jz_bridge_capture.yaml
```

Bind to the LAN:

```bash
conda run -n lerobot_flex python tests/visual_cali/head_camera_overlay.py \
  --host 0.0.0.0 --port 8090
```
