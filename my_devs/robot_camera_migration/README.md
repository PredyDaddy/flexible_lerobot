# Robot / Camera Migration

在 `flexible_lerobot` 里沿用旧 SO101 配置时，先进入指定环境：

```bash
conda activate lerobot_flex
cd /data/cqy_workspace/flexible_lerobot
```

核心沿用参数：

```bash
--robot.type=so101_follower
--robot.id=hfy_follower
--robot.port=/dev/ttyACM0
--teleop.type=so101_leader
--teleop.id=hfy_leader
--teleop.port=/dev/ttyACM1
```

已验证：

```text
/dev/ttyACM0 exists, readable+writable
/dev/ttyACM1 exists, readable+writable

hfy_follower -> /home/cqy/.cache/huggingface/lerobot/calibration/robots/so_follower/hfy_follower.json
hfy_leader   -> /home/cqy/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/hfy_leader.json
```

当前串口 by-id：

```text
/dev/ttyACM0 -> /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00
/dev/ttyACM1 -> /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123582-if00
```

如果现场确认 `/dev/ttyACM0` 是 follower、`/dev/ttyACM1` 是 leader，可以改用 by-id，避免重启后 ACM 编号变化：

```bash
--robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00
--teleop.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123582-if00
```

相机沿用 `top` / `wrist` 两个名字，不要随便改名：

```bash
--robot.cameras='{top: {type: opencv, index_or_path: "/dev/video4", width: 640, height: 480, fps: 30, fourcc: "YUYV"}, wrist: {type: opencv, index_or_path: "/dev/video6", width: 640, height: 480, fps: 30, fourcc: "MJPG"}}'
```

现场先确认端口：

```bash
lerobot-find-port
lerobot-find-cameras opencv
```

标定默认沿用：

```text
/home/cqy/.cache/huggingface/lerobot/calibration/robots/so_follower/hfy_follower.json
/home/cqy/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/hfy_leader.json
```

本目录下的相机测试图命名规则：

```text
captures/top_dev-video4_YYYYMMDD_HHMMSS.png
captures/wrist_dev-video6_YYYYMMDD_HHMMSS.png
```
