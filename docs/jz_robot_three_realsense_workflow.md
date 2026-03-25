# JZRobot 三路 RealSense 录制与回放

本文档只保留两个推荐命令：

- 一个完整版录制命令
- 一个完整版回放命令

当前默认工作流：

- 三路 RealSense 图像已经由 ROS2 节点发布出来
- LeRobot 通过 ROS2 image topic 录制和可选回放
- 默认机器人配置文件是 [`src/lerobot/configs/robot/jz_robot_three_realsense_ros2_topics.yaml`](../src/lerobot/configs/robot/jz_robot_three_realsense_ros2_topics.yaml)

固定相机映射：

- `chest` -> Intel RealSense D435 -> `348522073175`
- `left_arm` -> Intel RealSense D405 -> `230422272306`
- `right_arm` -> Intel RealSense D405 -> `230322272819`

推荐入口：

- [`my_devs/jz_robot/run_record_jz_three_realsense.py`](../my_devs/jz_robot/run_record_jz_three_realsense.py)
- [`my_devs/jz_robot/run_replay_jz_three_realsense.py`](../my_devs/jz_robot/run_replay_jz_three_realsense.py)

## 1. 环境准备

```bash
cd /home/test/workspace/flexible_lerobot
conda activate lerobot
source /opt/ros/humble/setup.bash
```

如果不方便 `conda activate`，也可以统一使用 `conda run -n lerobot ...`。

## 2. 录制前先检查 topic

默认录制依赖这些 topic：

- 状态 topic：
  - `/robot1/arm_left/joint_states`
  - `/robot1/arm_right/joint_states`
  - `/robot1/left_gripper/gripper_status`
  - `/robot1/right_gripper/gripper_status`
- 外部命令 topic：
  - `/robot1/telecon/arm_left/joint_commands_input`
  - `/robot1/telecon/arm_right/joint_commands_input`
  - `/robot1/left_gripper/gripper_commands`
  - `/robot1/right_gripper/gripper_commands`
- 相机 topic：
  - `/robot1/camera_head/camera_head/color/image_raw`
  - `/robot1/camera_left/camera_left/color/image_rect_raw`
  - `/robot1/camera_right/camera_right/color/image_rect_raw`

推荐先做最小检查：

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /robot1/arm_left/joint_states --once
ros2 topic echo /robot1/arm_right/joint_states --once
ros2 topic echo /robot1/telecon/arm_left/joint_commands_input --once
ros2 topic echo /robot1/telecon/arm_right/joint_commands_input --once
```

如果你启用了 gripper，再补充检查：

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /robot1/left_gripper/gripper_status --once
ros2 topic echo /robot1/right_gripper/gripper_status --once
ros2 topic echo /robot1/left_gripper/gripper_commands --once
ros2 topic echo /robot1/right_gripper/gripper_commands --once
```

## 3. 录制命令

只推荐这一条完整版录制命令，参数都可以手动改：

```bash
  cd /home/test/workspace/flexible_lerobot
  python -u -m my_devs.jz_robot.run_record_jz_three_realsense \
    --dataset-repo-id local/jz_pick_place_three_rs \
    --dataset-root /home/test/data/lerobot/jz_pick_place_three_rs_run7 \
    --dataset-task "pick and place with dual arms and grippers" \
    --robot-id jz_dual_arm_rs \
    --use-gripper true \
    --init-state-timeout-s 0 \
    --state-timeout-s 0.2 \
    --use-external-commands true \
    --teleop-connect-timeout-s 0 \
    --num-episodes 2 \
    --episode-time-s 15 \
    --reset-time-s 5 \
    --fps 30 \
    --display-data false \
    --play-sounds true
```

这条命令当前的启动行为：

- 先启动 `JZRobot`
- 先等待 robot state topic，就绪日志是 `initialization step 4/5`
- 再启动 `JZCommandTeleop`
- 再等待外部 command topic，就绪日志是 `initialization step 5/5`

默认等待策略：

- `--init-state-timeout-s 0`：启动时一直等待 state topic 首帧
- `--teleop-connect-timeout-s 0`：启动时一直等待 command topic 首帧
- `--state-timeout-s 0.2`：启动完成后，如果后续 state 断流超过 0.2 秒，会按 stale state 处理

## 4. 回放命令

只推荐这一条完整版回放命令，参数都可以手动改：

```bash
cd /home/test/workspace/flexible_lerobot
python -u -m my_devs.jz_robot.run_replay_jz_three_realsense \
  --dataset-repo-id local/jz_pick_place_three_rs \
  --dataset-root /home/test/data/lerobot/jz_pick_place_three_rs_run1 \
  --episode 1 \
  --robot-id jz_dual_arm_rs \
  --use-gripper true \
  --init-state-timeout-s 0 \
  --state-timeout-s 0.2 \
  --use-external-commands false \
  --connect-cameras false \
  --fps 30 \
  --play-sounds false
```

回放默认行为：

- 会等待 robot state topic 首帧
- 会从数据集读取指定 `episode` 的 action 并主动发布到命令 topic
- `--use-external-commands false` 时才会真正发布 JointState 和 gripper 命令
- 默认 `--connect-cameras false`，所以回放时不连接三路相机
- 默认 `--episode 0`，也就是第一段录制；`--episode 1` 是第二段录制

## 5. 常用可调参数

录制和回放时最常改的就是这些：

```bash
--dataset-repo-id local/your_dataset_name
--dataset-root /home/test/data/lerobot/your_dataset_dir
--robot-id your_robot_id
--episode 0
--num-episodes 20
--episode-time-s 45
--reset-time-s 15
--fps 30
```

其中：

- 录制会用到 `--num-episodes`、`--episode-time-s`、`--reset-time-s`
- 回放会用到 `--episode`
- 回放时如果希望机器人真正执行数据集动作，必须保持：

```bash
--use-external-commands false
```

- 回放时如果只想走动作链路，不想连接相机，保持：

```bash
--connect-cameras false
```

Topic 不一致时改这些：

```bash
--left-joint-state-topic ...
--right-joint-state-topic ...
--left-command-topic ...
--right-command-topic ...
--left-gripper-state-topic ...
--right-gripper-state-topic ...
--left-gripper-command-topic ...
--right-gripper-command-topic ...
--head-image-topic ...
--left-image-topic ...
--right-image-topic ...
```

相机参数：

```bash
--img-width 640
--img-height 480
--camera-fps 30
--camera-timeout-ms 5000
```

如果你这次没有 gripper 状态流和命令流，直接改成：

```bash
--use-gripper false
```

如果当前 shell 没有图形环境，比如报：

- `neither WAYLAND_DISPLAY nor WAYLAND_SOCKET nor DISPLAY is set`
- `failed to acquire X connection`

就保持：

```bash
--display-data false
```

## 6. 常见问题

### 6.1 一直卡在 `initialization step 4/5`

说明程序正在等 robot state topic 首帧。

优先检查：

- `/robot1/arm_left/joint_states`
- `/robot1/arm_right/joint_states`
- `/robot1/left_gripper/gripper_status`
- `/robot1/right_gripper/gripper_status`

如果你这次没有 gripper 状态流，录制时加：

```bash
--use-gripper false
```

### 6.2 一直卡在 `initialization step 5/5`

说明 robot state 已经就绪，程序正在等外部 command topic 首帧。

优先检查：

- `/robot1/telecon/arm_left/joint_commands_input`
- `/robot1/telecon/arm_right/joint_commands_input`

如果启用了 gripper，也要检查：

- `/robot1/left_gripper/gripper_commands`
- `/robot1/right_gripper/gripper_commands`

### 6.3 `Timed out waiting for JointState commands`

这通常只会在你手动把 `--teleop-connect-timeout-s` 设成正数时出现。默认现在是一直等待，不会因为 command topic 启动慢直接退出。

这个报错只和录制有关，因为录制会连接 `JZCommandTeleop` 并等待外部 command topic。

回放默认不会等待 teleop command topic；它会直接从数据集读取 action 并发布给机器人。

### 6.4 训练时相机 key 应该填什么

就是数据集里的三个 key：

- `chest`
- `left_arm`
- `right_arm`
