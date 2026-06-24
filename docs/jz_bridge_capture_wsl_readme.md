# JZRobot WSL 笔记本跨机采集 README

这份文档面向当前场景：

- 机器人端是 Orin / Ubuntu / ROS 2，负责机械臂、夹爪、相机和 RTSP 推流。
- 笔记本端运行 WSL，作为采集机，负责接收 ROS 2 state/action 和 RTSP 视频，写 raw episode，再转换成 LeRobot v3 数据集。

当前代码位置：

```text
src/lerobot/robots/jz_robot/bridge_capture/
```

主要命令：

```text
jz-lerobot-vector-bridge   # 机器人端：聚合 ROS topic 成 state/action 向量
jz-lerobot-raw-recorder    # WSL 采集机：录制 raw episode
jz-lerobot-validate-raw    # WSL 采集机：校验 raw episode
jz-lerobot-convert-raw     # WSL 采集机：raw -> LeRobot v3
```

默认配置：

```text
src/lerobot/configs/robot/jz_bridge_capture.yaml
```

## 1. RTSP 为什么还需要 receiver

你的机器人里面确实已经有 RTSP 流。RTSP 流的角色是“视频源”或“视频服务器”：

```text
机器人相机 -> 编码 -> RTSP server -> rtsp://192.168.50.10:8554/...
```

但 LeRobot 采集程序不能直接把一个 RTSP URL 当成图片用。采集机还必须做这些事：

1. 连接机器人端 RTSP URL。
2. 持续接收网络视频包。
3. 解码 H.264/H.265/JPEG 等视频数据，得到一帧一帧的图像。
4. 给每帧记录接收时间、frame index、PTS 等时间信息。
5. 放入采集机本地缓冲区。
6. 在每个采样时刻，把最接近的图像、state、action 对齐后落盘。

所以当前的 `rtsp_receiver.py` 不是替代机器人端 RTSP 流，而是笔记本端的“RTSP 客户端/接收解码器”。

当前实现用的是 OpenCV：

```python
cv2.VideoCapture(rtsp_url)
```

也就是：

```text
机器人已有 RTSP 流
        |
        v
WSL 里的 OpenCV RTSP receiver 连接并解码
        |
        v
raw_recorder 按时间对齐并保存 episode
```

如果现场 OpenCV 能稳定拉流，就不需要马上换 GStreamer。  
如果出现卡顿、延迟漂移、断流后不恢复、拿不到可靠时间戳、无法指定 TCP/UDP 等问题，再把 `rtsp_receiver.py` 换成 GStreamer appsink 版本。

换句话说：

- 不需要改机器人端已有 RTSP 推流。
- 需要在 WSL 采集机上有一个稳定的 RTSP 接收器。
- OpenCV receiver 是第一版最小实现。
- GStreamer appsink 是更可控、更适合长期采集的接收实现。

## 2. WSL 网络准备

先确认 WSL 能访问机器人端 IP。假设机器人端 IP 是：

```text
192.168.50.10
```

WSL 里执行：

```bash
ping -c 3 192.168.50.10
```

如果 ping 不通，先检查：

- 笔记本 Windows 是否和机器人在同一网段。
- Windows 防火墙是否拦截。
- WSL2 网络模式是否能访问局域网。
- 机器人端 IP 是否实际是 `192.168.50.10`。

建议采集时用有线网，不要走 Wi-Fi。

## 3. WSL ROS 2 准备

WSL 里要能看到机器人端 ROS 2 topic。两边需要一致：

```bash
export ROS_DOMAIN_ID=50
export ROS_LOCALHOST_ONLY=0
```

WSL 里 source ROS：

```bash
source /opt/ros/humble/setup.bash
```

然后检查 topic：

```bash
ros2 topic list
```

如果机器人端已经启动 `jz-lerobot-vector-bridge`，WSL 应该能看到：

```text
/robot1/lerobot/state
/robot1/lerobot/action
```

验证一次数据：

```bash
ros2 topic echo /robot1/lerobot/state --once
ros2 topic echo /robot1/lerobot/action --once
```

如果 WSL 看不到 ROS 2 topic，但 ping 能通，优先检查：

- `ROS_DOMAIN_ID` 是否一致。
- `ROS_LOCALHOST_ONLY` 是否为 `0`。
- Windows 防火墙。
- WSL2 对 DDS 组播的支持情况。
- 机器人端和 WSL 是否真的在同一张有线网卡路径上。

WSL2 对 ROS 2 DDS 组播有时不稳定。如果 `ros2 topic list` 看不到远端 topic，建议优先在 Windows/WSL 网络层和 DDS 配置上排查；必要时改用 Cyclone DDS 的 peer/static discovery 配置。

## 4. 机器人端启动

在机器人 Orin 上启动相机和 RTSP bridge：

```bash
cd /home/test/workspace/teleop_ws
source install/setup.bash
ros2 launch robot_camera robot_camera.launch.py
ros2 launch camera_bridge camera_bridge.launch.py
```

确认 `camera_bridge.yaml` 里要采集的相机启用了 RTSP：

```yaml
rtsp:
  enabled: true
```

再启动 LeRobot 向量桥：

```bash
cd /home/test/workspace/flexible_lerobot
conda activate lerobot
source /home/test/workspace/teleop_ws/install/setup.bash
export ROS_DOMAIN_ID=50
export ROS_LOCALHOST_ONLY=0

jz-lerobot-vector-bridge \
  --config src/lerobot/configs/robot/jz_bridge_capture.yaml
```

这个进程会把多个机器人 topic 聚合成：

```text
/robot1/lerobot/state
/robot1/lerobot/action
```

## 5. WSL 里验证 RTSP

先不要直接录制，先验证 WSL 能拉到机器人 RTSP 流。

默认配置里的 RTSP URL 类似：

```text
rtsp://192.168.50.10:8554/robot_camera/camera_head
rtsp://192.168.50.10:8554/robot_camera/camera_left
rtsp://192.168.50.10:8554/robot_camera/camera_right
rtsp://192.168.50.10:8554/robot_camera/camera_chest
```

如果 WSL 有图形显示，可以用 `ffplay`：

```bash
ffplay -rtsp_transport tcp rtsp://192.168.50.10:8554/robot_camera/camera_head
```

如果没有图形显示，可以用 OpenCV 做一次读帧测试：

```bash
cd /home/test/workspace/flexible_lerobot
python - <<'PY'
import cv2

url = "rtsp://192.168.50.10:8554/robot_camera/camera_head"
cap = cv2.VideoCapture(url)
ok, frame = cap.read()
print("opened:", cap.isOpened())
print("read:", ok)
if ok:
    print("shape:", frame.shape)
cap.release()
PY
```

预期输出类似：

```text
opened: True
read: True
shape: (720, 1280, 3)
```

如果 `opened: False` 或 `read: False`：

- 先确认 URL 是否正确。
- 在机器人端确认 RTSP 服务已经启动。
- 确认端口 `8554` 没被防火墙拦截。
- 用 `ping 192.168.50.10` 确认网络连通。
- 用 `ffplay` 或 `gst-launch-1.0` 单独排查 RTSP。

## 6. WSL 里试采 ROS-only

先不采相机，只采 state/action，排除 ROS 2 和 topic 问题。

```bash
cd /home/test/workspace/flexible_lerobot
conda activate lerobot
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=50
export ROS_LOCALHOST_ONLY=0

jz-lerobot-raw-recorder \
  --config src/lerobot/configs/robot/jz_bridge_capture.yaml \
  --episode-id 000001 \
  --task "pick cube" \
  --duration-s 10 \
  --no-cameras
```

输出目录默认是：

```text
/home/test/datasets/jz_raw/episode_000001/
```

校验：

```bash
jz-lerobot-validate-raw /home/test/datasets/jz_raw/episode_000001
```

如果 ROS-only 都失败，不要继续测相机，先修 ROS 2 跨机通信。

## 7. WSL 里采 RTSP + ROS

确认 ROS-only 成功、RTSP 单独读帧成功后，再采完整数据：

```bash
cd /home/test/workspace/flexible_lerobot
conda activate lerobot
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=50
export ROS_LOCALHOST_ONLY=0

jz-lerobot-raw-recorder \
  --config src/lerobot/configs/robot/jz_bridge_capture.yaml \
  --episode-id 000002 \
  --task "pick cube" \
  --duration-s 30
```

校验：

```bash
jz-lerobot-validate-raw /home/test/datasets/jz_raw/episode_000002
```

看输出里的：

- `sample_count`
- `valid_count`
- `invalid_count`
- `camera_frame_counts`
- `errors`

如果 `invalid_count` 很高，通常是这些问题：

- RTSP 帧没拉到。
- ROS state/action 没收到。
- 时间同步不准。
- 采样阈值太严格。
- WSL 网络不稳定。

## 8. 转换为 LeRobot v3

转换整个 raw 根目录：

```bash
jz-lerobot-convert-raw \
  --raw-root /home/test/datasets/jz_raw \
  --repo-id local/jz_bridge_capture \
  --output-root /home/test/data/lerobot/jz_bridge_capture
```

只转换一个 episode：

```bash
jz-lerobot-convert-raw \
  --raw-episode /home/test/datasets/jz_raw/episode_000002 \
  --repo-id local/jz_bridge_capture_debug \
  --output-root /home/test/data/lerobot/jz_bridge_capture_debug
```

默认会跳过 `valid=false` 的 sample。正式训练数据不要加 `--include-invalid`。

## 9. 配置需要现场确认

检查这个文件：

```text
src/lerobot/configs/robot/jz_bridge_capture.yaml
```

重点确认：

```yaml
robot:
  orin_ip: 192.168.50.10
  collector_ip: 192.168.50.20

cameras:
  camera_head:
    rtsp_url: rtsp://192.168.50.10:8554/robot_camera/camera_head

state:
  topic: /robot1/lerobot/state

action:
  topic: /robot1/lerobot/action
```

还要确认 action source topic 是否真的带 `/robot1` namespace：

```text
/robot1/telecon/arm_left/joint_commands_input
/robot1/telecon/arm_right/joint_commands_input
```

如果现场实际 topic 是：

```text
/telecon/arm_left/joint_commands_input
/telecon/arm_right/joint_commands_input
```

就要改 YAML。以机器人端 `ros2 topic list` 的实际结果为准。

## 10. OpenCV Receiver 和 GStreamer Receiver 的区别

当前 `rtsp_receiver.py` 用 OpenCV：

```text
src/lerobot/robots/jz_robot/bridge_capture/rtsp_receiver.py
```

优点：

- 代码简单。
- 安装和调试成本低。
- 先验证跨机采集链路足够快。

缺点：

- 不一定能稳定控制 RTSP 使用 TCP 还是 UDP。
- 断流重连能力弱。
- 时间戳信息不如 GStreamer 清楚。
- 长时间采集时更难统计丢帧、延迟和重连。

GStreamer appsink 版本的意义是让 WSL 采集端更可控：

```text
rtspsrc location=... protocols=tcp latency=100 !
rtph264depay !
avdec_h264 !
videoconvert !
appsink
```

它不是因为机器人没有 RTSP。  
它是因为采集机需要更稳定、更可诊断地接收机器人已经提供的 RTSP。

建议顺序：

1. 先用当前 OpenCV receiver 跑通 10 秒、30 秒、10 分钟测试。
2. 如果 `valid_count` 高、图像连续、延迟稳定，就继续用 OpenCV。
3. 如果 OpenCV 读帧不稳定，再替换成 GStreamer appsink receiver。

## 11. 推荐排查顺序

按这个顺序排查，别一开始就同时开所有东西：

1. WSL ping 机器人 IP。
2. WSL 单独拉 RTSP 一帧。
3. WSL `ros2 topic list` 能看到机器人 topic。
4. 机器人端启动 `jz-lerobot-vector-bridge`。
5. WSL `ros2 topic echo /robot1/lerobot/state --once`。
6. WSL `ros2 topic echo /robot1/lerobot/action --once`。
7. WSL 录 ROS-only raw episode。
8. WSL 录 RTSP + ROS raw episode。
9. validate raw episode。
10. convert raw to LeRobot v3。

只有上一步稳定，再做下一步。
