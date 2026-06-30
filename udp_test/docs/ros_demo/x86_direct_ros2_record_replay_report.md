# x86 直接加入 Orin ROS2 网络进行 JZRobot record/replay 对照实验报告

## 1. 背景与目标

当前已有两条相关链路：

1. `src/lerobot/robots/jz_robot/`：直接基于 ROS2 topic 的 LeRobot `Robot` 实现。它订阅左右臂 `JointState`、可选夹爪状态和相机输入，并在需要时向 ROS2 command topic 发布动作。
2. 当前 UDP 方案：x86 运行 LeRobot/UDP robot，Orin 侧运行 UDP bridge / receiver，把 ROS2 状态或命令通过 UDP 转发。该方案适合跨网络隔离、Orin 负载控制和分阶段 dry-run，但多了一层 UDP 协议、receiver/executor 和排错面。

本报告目标是说明一种更直接的对照实验方式：

- 让 x86 机器 `192.168.1.106` 进入 Orin `192.168.1.81` 所在 ROS2 网络。
- 在 x86 上直接运行现有 `src/lerobot/robots/jz_robot`。
- 用标准 `lerobot-record` / `lerobot-replay` 做 record/replay 对照。
- 避免新增 UDP executor，也不新增 `jz_robot_ros` 机器人类型。

本文只提供操作说明和安全边界，不修改代码，不启动 ROS，不操控机器人。

## 2. 为什么不需要新增 jz_robot_ros

不建议新增 `jz_robot_ros`，原因如下：

1. `jz_robot` 已经是 ROS2 direct robot。
   - 配置类注册名是 `type: jz_robot`。
   - 代码依赖 `rclpy`、`sensor_msgs.msg.JointState` 和 `std_msgs.msg.Float64MultiArray`。
   - `connect()` 会创建 ROS2 node、publisher、subscriber 和 `SingleThreadedExecutor`。
   - `get_observation()` 从 ROS2 state topic 和相机读取 observation。
   - `send_action()` 在 `use_external_commands=false` 时会 publish 到 ROS2 command topic。

2. 现有配置已经覆盖 ROS2 topic 场景。
   - `src/lerobot/configs/robot/jz_robot_three_realsense_ros2_topics.yaml` 使用 `/robot1/...` 绝对 topic，并包含 ROS2 image topic camera 配置。
   - `src/lerobot/configs/robot/jz_robot_three_realsense.yaml` 使用不带 namespace 的 topic，并直接打开本机 RealSense。

3. 新增 `jz_robot_ros` 会制造重复抽象。
   - 如果它仍然通过 ROS2 topic 收发状态和命令，本质上和 `jz_robot` 重复。
   - 后续 record/replay/policy 入口仍然只需要一个标准 LeRobot `Robot` 类型。
   - 真正需要变化的是运行环境、ROS2 网络发现、topic namespace 和 `use_external_commands` 策略，而不是新增机器人类型。

因此，对照实验推荐直接使用 `jz_robot`，把 x86 当成 ROS2 网络中的另一台计算节点。

## 3. 前置假设

- Orin IP：`192.168.1.81`
- x86 IP：`192.168.1.106`
- 两台机器在同一二层网络，或至少 ROS2 DDS multicast / unicast discovery 能互通。
- x86 使用本仓库要求的 conda 环境开发与测试，例如 `lerobot_flex`。实际环境名以现场为准。
- x86 已安装 ROS2 Python 依赖，当前 conda 环境中的 Python 可以 `import rclpy`。
- x86 运行命令前需要 `source` ROS2 环境，并设置与 Orin 一致的 `ROS_DOMAIN_ID`。
- 机器人启动、急停、权限和现场安全检查由现场人员完成；本文命令不应在无人监护时执行。

建议先只做网络和 ROS2 只读检查，再做 record，只在人工确认安全后做 replay。

## 4. 网络与 ROS2 前置检查命令

以下命令需要人工在对应机器上运行。不要把检查命令和 record/replay 命令合并成自动脚本。

### 4.1 x86 基础网络检查

在 x86 `192.168.1.106`：

```bash
ip addr
ip route
ping -c 3 192.168.1.81
```

预期：

- x86 网卡上能看到 `192.168.1.106`。
- 能 ping 通 Orin `192.168.1.81`。
- 如果 ping 不通，先处理网线、交换机、IP、路由、防火墙，不要继续 ROS2 检查。

### 4.2 Orin 基础网络检查

在 Orin `192.168.1.81`：

```bash
ip addr
ip route
ping -c 3 192.168.1.106
```

预期：

- Orin 网卡上能看到 `192.168.1.81`。
- 能 ping 通 x86 `192.168.1.106`。

### 4.3 x86 ROS2 环境检查

在 x86：

```bash
conda activate lerobot_flex
source /opt/ros/<ros_distro>/setup.bash
python -c "import rclpy; print('rclpy ok')"
python -c "import sensor_msgs.msg, std_msgs.msg; print('ros msgs ok')"
```

把 `<ros_distro>` 替换为现场 ROS2 发行版，例如 `humble`。如果 conda 环境覆盖了 `PYTHONPATH` 或 `LD_LIBRARY_PATH`，需要保证 `source /opt/ros/<ros_distro>/setup.bash` 后 `python` 仍然是目标 conda 环境里的 Python，并且能 import ROS2 Python 包。

### 4.4 ROS_DOMAIN_ID 对齐

在 Orin 和 x86 分别确认：

```bash
echo "${ROS_DOMAIN_ID:-<unset>}"
```

如果 Orin 使用 `ROS_DOMAIN_ID=50`，则 x86 也设置：

```bash
export ROS_DOMAIN_ID=50
```

实际值必须以机器人现场环境为准。两端 `ROS_DOMAIN_ID` 不一致时，ROS2 topic 通常互相不可见。

### 4.5 x86 查看 Orin ROS2 topic

在 x86：

```bash
ros2 node list
ros2 topic list
ros2 topic list | grep -E 'robot1|arm_left|arm_right|gripper|camera'
```

如果使用 `src/lerobot/configs/robot/jz_robot_three_realsense_ros2_topics.yaml`，重点检查这些 topic 是否存在：

```bash
ros2 topic info /robot1/arm_left/joint_states
ros2 topic info /robot1/arm_right/joint_states
ros2 topic info /robot1/telecon/arm_left/joint_commands_input
ros2 topic info /robot1/telecon/arm_right/joint_commands_input
ros2 topic info /robot1/left_gripper/gripper_status
ros2 topic info /robot1/right_gripper/gripper_status
ros2 topic info /robot1/left_gripper/gripper_commands
ros2 topic info /robot1/right_gripper/gripper_commands
```

只读确认状态流：

```bash
ros2 topic echo --once /robot1/arm_left/joint_states
ros2 topic echo --once /robot1/arm_right/joint_states
```

相机如果也走 ROS2 topic，可检查：

```bash
ros2 topic info /robot1/camera_head/camera_head/color/image_raw
ros2 topic info /robot1/camera_left/camera_left/color/image_rect_raw
ros2 topic info /robot1/camera_right/camera_right/color/image_rect_raw
```

### 4.6 DDS discovery 和防火墙检查

ROS2 默认依赖 DDS discovery。若两机能 ping 通但 topic 不可见，检查：

```bash
env | grep -E 'ROS_|RMW_|CYCLONEDDS|FASTRTPS|FAST_DDS'
sudo ufw status
```

排查方向：

- `ROS_DOMAIN_ID` 是否一致。
- `RMW_IMPLEMENTATION` 是否和现场兼容。
- 是否有防火墙阻挡 DDS multicast / UDP 端口。
- 两机是否在不同 VLAN 或无线 AP 隔离网络中。
- 是否需要 CycloneDDS/FastDDS 的 discovery server 或显式 peer 配置。

## 5. x86 直接 ROS2 record 最小命令

record 阶段建议 `use_external_commands=true`。语义是：LeRobot 读取 ROS2 state 和相机，记录 observation/action，但 `JZRobot.send_action()` 只校验和回传 action，不向机器人 command topic publish。这样可以让外部 teleop、VR 或人工示教系统继续掌握真实控制权，避免 LeRobot record 与外部控制器抢 command topic。

最小命令示例：

```bash
cd /home/data/test/workspace/flexible_lerobot
conda activate lerobot_flex
source /opt/ros/<ros_distro>/setup.bash
export ROS_DOMAIN_ID=<same_as_orin>
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

lerobot-record \
  --robot.type=jz_robot \
  --robot.id=jz_robot_x86_direct_ros2 \
  --robot.left_joint_state_topic=/robot1/arm_left/joint_states \
  --robot.right_joint_state_topic=/robot1/arm_right/joint_states \
  --robot.left_position_command_topic=/robot1/telecon/arm_left/joint_commands_input \
  --robot.right_position_command_topic=/robot1/telecon/arm_right/joint_commands_input \
  --robot.use_gripper=true \
  --robot.left_gripper_state_topic=/robot1/left_gripper/gripper_status \
  --robot.right_gripper_state_topic=/robot1/right_gripper/gripper_status \
  --robot.left_gripper_command_topic=/robot1/left_gripper/gripper_commands \
  --robot.right_gripper_command_topic=/robot1/right_gripper/gripper_commands \
  --robot.state_timeout_s=5.0 \
  --robot.use_external_commands=true \
  --dataset.repo_id=local/jz_robot_x86_direct_ros2_record \
  --dataset.root=tests/outputs/jz_robot_x86_direct_ros2_record \
  --dataset.num_episodes=1 \
  --dataset.episode_time_s=10 \
  --dataset.single_task="x86 direct ros2 record check" \
  --dataset.push_to_hub=false
```

如果现场使用配置文件注入方式，也可以以 `src/lerobot/configs/robot/jz_robot_three_realsense_ros2_topics.yaml` 为模板，但要特别确认 `use_external_commands` 是 `true`。

注意：`lerobot-record` 通常还需要 teleoperator 或 policy 作为 action source。具体命令应按现场已有 teleop/policy 配置补齐。本文重点是 robot 侧参数：x86 直接使用 `jz_robot` 连接 Orin ROS2 topic，不通过 UDP executor。

## 6. x86 直接 ROS2 replay 最小命令

replay 阶段会把 dataset 中的 action 发送给 robot。若要真正通过 ROS2 command topic 回放动作，必须设置 `use_external_commands=false`。

这是会向机器人发布 command topic 的命令，只能在现场人工确认机器人安全、急停可用、工作空间清空、topic 和 dataset 匹配后执行：

```bash
cd /home/data/test/workspace/flexible_lerobot
conda activate lerobot_flex
source /opt/ros/<ros_distro>/setup.bash
export ROS_DOMAIN_ID=<same_as_orin>
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

lerobot-replay \
  --robot.type=jz_robot \
  --robot.id=jz_robot_x86_direct_ros2 \
  --robot.left_joint_state_topic=/robot1/arm_left/joint_states \
  --robot.right_joint_state_topic=/robot1/arm_right/joint_states \
  --robot.left_position_command_topic=/robot1/telecon/arm_left/joint_commands_input \
  --robot.right_position_command_topic=/robot1/telecon/arm_right/joint_commands_input \
  --robot.use_gripper=true \
  --robot.left_gripper_state_topic=/robot1/left_gripper/gripper_status \
  --robot.right_gripper_state_topic=/robot1/right_gripper/gripper_status \
  --robot.left_gripper_command_topic=/robot1/left_gripper/gripper_commands \
  --robot.right_gripper_command_topic=/robot1/right_gripper/gripper_commands \
  --robot.state_timeout_s=5.0 \
  --robot.use_external_commands=false \
  --robot.max_relative_target=0.05 \
  --dataset.repo_id=local/jz_robot_x86_direct_ros2_record \
  --dataset.root=tests/outputs/jz_robot_x86_direct_ros2_record \
  --dataset.episode=0 \
  --dataset.fps=30 \
  --play_sounds=false
```

`--robot.max_relative_target=0.05` 是保守示例，用于限制单步目标相对当前位置的最大变化幅度。真实阈值应按关节单位、控制器语义和现场安全策略确认。若 dataset action 本身不是当前机器人状态附近的目标位置，不能直接 replay。

如果只是验证 `lerobot-replay` 能启动、解析 dataset 和调用 `send_action()`，但不希望 publish ROS2 command topic，则保持：

```bash
--robot.use_external_commands=true
```

这时 replay 是 dry-run 语义，不会驱动机器人，只能用于软件链路对照。

## 7. record/replay 阶段 use_external_commands 的区别

`src/lerobot/robots/jz_robot/jz_robot.py` 中 `send_action()` 的行为可以概括为：

- 总是解析 action，读取当前左右臂/夹爪状态，构造完整 action 字典。
- 如果配置 `use_external_commands=true`，直接返回 action，不 publish ROS2 command。
- 如果配置 `use_external_commands=false`，发布左右臂 `JointState` command；启用夹爪时还会发布左右夹爪 `Float64MultiArray` command。

推荐策略：

| 阶段 | `use_external_commands` | 含义 | 风险 |
| --- | --- | --- | --- |
| record | `true` | 外部 teleop/VR/人工系统控制机器人，LeRobot 只记录并校验 action | 不会由 LeRobot 额外 publish command，适合采集 |
| replay dry-run | `true` | 回放软件链路，但 `send_action()` 不 publish command | 不能证明机器人会真实执行，只能验证 dataset/action path |
| replay real | `false` | LeRobot 通过 ROS2 command topic 直接驱动机器人 | 会动机器人，必须人工安全确认 |

record 阶段如果误设为 `false`，LeRobot 和外部控制源可能同时向 command topic 发布，形成控制冲突。replay 阶段如果误设为 `true`，则不会移动机器人，容易误判为 replay 失效。

## 8. 相机建议：控制走 ROS2，相机可继续 RTSP 或后续混合

本次对照实验的核心是验证 x86 直接加入 Orin ROS2 网络后，`jz_robot` 能否直接读写机器人控制相关 topic。因此建议先把重点放在：

- 左右臂 state topic。
- 左右臂 command topic。
- 可选夹爪 state/command topic。
- `use_external_commands` 对 record/replay 的行为差异。

相机不必强行一次性全部切到 ROS2 topic。可选策略：

1. 控制走 ROS2，相机继续使用已有 RTSP。
   - 优点：复用当前 UDP/bridge capture 中已验证的 RTSP 链路。
   - 缺点：标准 `jz_robot` 配置需要 camera config 支持对应相机类型；如当前 robot config 只写 ROS2 image topic，则需要另行准备匹配的配置。

2. 控制和相机都走 ROS2 topic。
   - 可直接参考 `jz_robot_three_realsense_ros2_topics.yaml`。
   - 优点：record/replay 输入都来自同一个 ROS2 网络。
   - 缺点：x86 需要稳定接收三路 image topic，带宽和 DDS QoS 压力更高。

3. 后续混合方案。
   - 控制状态走 ROS2 direct。
   - 相机根据稳定性在 RTSP、ROS2 topic、或本机 RealSense 之间选择。
   - 最终以 record 数据时间戳质量、丢帧率和调试成本决定。

建议第一轮对照实验优先验证无相机或低带宽相机配置；确认控制链路稳定后，再把三路相机纳入完整 record。

## 9. 常见排错项

### 9.1 rclpy import 失败

现象：

```text
JZRobot requires ROS2 Python dependencies (`rclpy`, `sensor_msgs`).
```

检查：

```bash
which python
python -c "import sys; print(sys.executable)"
python -c "import rclpy; print(rclpy.__file__)"
python -c "import sensor_msgs.msg, std_msgs.msg; print('ok')"
```

处理方向：

- 先 `conda activate lerobot_flex`，再 `source /opt/ros/<ros_distro>/setup.bash`。
- 确认当前 Python 是目标 conda 环境。
- 确认 ROS2 Python 包对当前 Python ABI 可用。
- 如果 conda Python 与系统 ROS2 Python ABI 不兼容，需要使用现场已验证的 ROS2/conda 组合。

### 9.2 PYTHONPATH 找不到本仓库 lerobot

现象：

```text
ModuleNotFoundError: No module named 'lerobot'
```

检查：

```bash
cd /home/data/test/workspace/flexible_lerobot
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python -c "import lerobot; print(lerobot.__file__)"
```

也可以在开发环境中安装：

```bash
python -m pip install -e ".[dev,test]"
```

### 9.3 ROS_DOMAIN_ID 不一致

现象：

- `ros2 topic list` 看不到 Orin 上的 topic。
- x86 `jz_robot.connect()` 一直等待 initial state。

检查：

```bash
echo "${ROS_DOMAIN_ID:-<unset>}"
ros2 topic list
```

处理：

```bash
export ROS_DOMAIN_ID=<same_as_orin>
```

然后重新打开终端或重新启动相关命令，避免旧进程仍使用旧 domain。

### 9.4 topic 不可见或 namespace 不匹配

现象：

- `ros2 topic list` 有 topic，但和配置文件里的名称不一致。
- `jz_robot` 报等待 `/robot1/arm_left/joint_states` 超时。

检查：

```bash
ros2 topic list | sort
ros2 topic info /robot1/arm_left/joint_states
ros2 topic echo --once /robot1/arm_left/joint_states
```

处理：

- 如果现场 topic 不带 `/robot1`，使用 `jz_robot_three_realsense.yaml` 中的相对 topic 风格，或在命令行覆盖 topic。
- 如果现场 topic 带 `/robot1`，使用 `jz_robot_three_realsense_ros2_topics.yaml` 的绝对 topic 风格。
- 以现场 `ros2 topic list` 为准，不要仅凭文件名判断。

### 9.5 state stale 或初始化等待

现象：

```text
Timed out waiting for initial JZRobot state messages
JointState is stale
```

处理方向：

- 确认左右臂 state topic 都有消息。
- 确认 `left_joint_names` / `right_joint_names` 和 `JointState.name` 完全匹配。
- 适当增大 `--robot.state_timeout_s=5.0`，先排除网络抖动。
- 确认 ROS2 QoS 与现场 publisher 兼容。

## 10. 与当前 UDP 方案的对照

| 项目 | x86 直接 ROS2 方案 | 当前 UDP 方案 |
| --- | --- | --- |
| 控制路径 | x86 `jz_robot` 直接 publish ROS2 command topic | x86 经 UDP 发包，Orin receiver/executor 再处理 |
| 状态路径 | x86 直接 subscribe ROS2 state topic | Orin ROS2 state bridge 转 UDP 到 x86 |
| 机器人类型 | 复用 `jz_robot` | 使用 `jz_robot_udp` 或测试脚本 |
| 新增代码 | 不需要 | 需要协议、UDP client、receiver、bridge/executor |
| 排错重点 | ROS2 discovery、domain、topic、QoS、rclpy | UDP 网络、协议、bridge、receiver、executor、ROS2 |
| 安全边界 | `use_external_commands` 决定是否 publish ROS2 command | Phase 2 可 dry-run receiver 不执行；真实执行需 executor |
| 适用场景 | x86 能稳定加入机器人 ROS2 网络，适合快速对照 | x86 不适合进 ROS2 网络、需要隔离或跨网络转发 |
| 风险 | replay real 会直接向机器人发命令 | executor 上线后也会驱动机器人，但多一层安全阀可设计 |

结论：

- 如果 x86 能稳定看到 Orin ROS2 topic，直接 ROS2 方案是最短路径，适合做 record/replay 对照实验。
- UDP 方案仍有价值，适合后续需要网络隔离、协议审计、跨主机部署、Orin 侧安全 executor 或非 ROS2 客户端接入的场景。
- 两者不互斥。直接 ROS2 方案可以作为基准，帮助判断 UDP 层引入的延迟、丢包、字段映射和执行差异。

## 11. 安全注意事项

必须遵守以下边界：

1. 未经现场负责人确认，不执行 `use_external_commands=false` 的 replay。
2. replay real 前必须确认急停可用、机器人工作空间清空、机械臂不接触人员或障碍物。
3. record 阶段默认使用 `use_external_commands=true`，避免 LeRobot 与外部 teleop/VR 同时控制机器人。
4. replay dataset 必须来自同一机器人、同一关节命名、同一控制语义和相近初始姿态；不能把未知 dataset 直接发给实机。
5. 首次 replay 应降低速度、缩短 episode、限制 `max_relative_target`，并由人工实时监控。
6. 任何 topic、namespace、ROS_DOMAIN_ID 不确定时，只做 `ros2 topic list/info/echo` 只读检查，不执行 record/replay。
7. 不要在后台无人值守运行 replay。
8. 不要同时运行 UDP executor 和 x86 direct ROS2 replay，避免两个控制源同时写 command topic。
9. 不要把本报告中的命令封装成开机自启或自动执行脚本。
10. 本报告不授权启动 ROS、启动机器人或操控机器人；所有实机动作必须由现场安全流程批准。

## 12. 推荐实验顺序

1. x86 与 Orin 双向 ping。
2. x86 `rclpy` / ROS message import 检查。
3. x86 与 Orin `ROS_DOMAIN_ID` 对齐。
4. x86 `ros2 topic list/info/echo` 只读确认状态 topic。
5. x86 用 `jz_robot` 做短 record，`use_external_commands=true`。
6. 离线检查 dataset action/observation 字段名和数值范围。
7. `lerobot-replay` dry-run，`use_external_commands=true`，只验证软件链路。
8. 现场人工批准后，才做极短 replay real，`use_external_commands=false`，并设置保守 `max_relative_target`。
9. 与当前 UDP record/replay 或 dry-run 结果对照延迟、稳定性、字段映射和安全边界。
