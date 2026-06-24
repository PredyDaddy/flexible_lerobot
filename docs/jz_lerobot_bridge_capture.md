# JZRobot LeRobot 跨机采集使用说明

WSL 笔记本采集机的详细使用说明见：

```text
docs/jz_bridge_capture_wsl_readme.md
```

代码位置：

- 机器人端向量聚合：`lerobot.robots.jz_robot.bridge_capture.ros_action_bridge`
- 笔记本端 raw 采集：`lerobot.robots.jz_robot.bridge_capture.raw_recorder`
- 默认配置：`src/lerobot/configs/robot/jz_bridge_capture.yaml`

## 1. 机器人端 Orin

启动现有相机链路：

```bash
cd /home/test/workspace/teleop_ws
source install/setup.bash
ros2 launch robot_camera robot_camera.launch.py
ros2 launch camera_bridge camera_bridge.launch.py
```

确认 `camera_bridge.yaml` 中需要采集的相机 `rtsp.enabled: true`。

启动 LeRobot 向量桥：

```bash
cd /home/test/workspace/flexible_lerobot
conda activate lerobot
source /home/test/workspace/teleop_ws/install/setup.bash
export ROS_DOMAIN_ID=50
export ROS_LOCALHOST_ONLY=0
jz-lerobot-vector-bridge --config src/lerobot/configs/robot/jz_bridge_capture.yaml
```

它会发布：

```text
/robot1/lerobot/state
/robot1/lerobot/action
```

两个 topic 类型都是 `std_msgs/msg/Float64MultiArray`，向量顺序由 YAML 中 `state.names` 和 `action.names` 固定。

## 2. 笔记本端

先验证网络：

```bash
ping -c 3 192.168.50.10
ros2 topic echo /robot1/lerobot/state --once
ros2 topic echo /robot1/lerobot/action --once
```

试采 ROS-only：

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

确认 ROS-only 成功后，打开 RTSP 采集：

```bash
jz-lerobot-raw-recorder \
  --config src/lerobot/configs/robot/jz_bridge_capture.yaml \
  --episode-id 000002 \
  --task "pick cube" \
  --duration-s 30
```

输出目录默认是：

```text
/home/test/datasets/jz_raw/episode_000002/
```

校验 raw episode：

```bash
jz-lerobot-validate-raw /home/test/datasets/jz_raw/episode_000002
```

转换为 LeRobot v3 数据集：

```bash
jz-lerobot-convert-raw \
  --raw-root /home/test/datasets/jz_raw \
  --repo-id local/jz_bridge_capture \
  --output-root /home/test/data/lerobot/jz_bridge_capture
```

也可以只转换指定 episode：

```bash
jz-lerobot-convert-raw \
  --raw-episode /home/test/datasets/jz_raw/episode_000002 \
  --repo-id local/jz_bridge_capture_debug \
  --output-root /home/test/data/lerobot/jz_bridge_capture_debug
```

默认会跳过 `valid=false` 的 raw sample。若需要调试无效样本，可加 `--include-invalid`，但正式训练数据不建议这样做。

## 3. 配置必须按现场修改

最容易需要改的是：

- `robot.orin_ip`
- `robot.collector_ip`
- `recording.output_root`
- `cameras.*.rtsp_url`
- `state.sources.*.topic`
- `action.sources.*.topic`

如果 `teleop_vr_recv` 没有放在 `/robot1` namespace，action topic 可能不是：

```text
/robot1/telecon/arm_left/joint_commands_input
```

而是：

```text
/telecon/arm_left/joint_commands_input
```

以 `ros2 topic list` 的实际结果为准。
