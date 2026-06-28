# JZRobot VR 摇杆最小控制抽离文档

本文档说明如何从 `src/lerobot/robots/jz_robot` 中抽出最小控制链路，用于后续 VR 摇杆上机和推理控制。目标是尽量轻量，不依赖 LeRobot 的机器人注册、相机、录制、dataset、policy runner 等框架，只保留“能向机器人电机发送目标位置”的部分。

当前文档只写设计和抽离说明，不新增可执行控制代码。

## 1. 结论

`JZRobot` 真正让机器人动起来的路径很短：

1. 创建 ROS2 node。
2. 创建左右臂 command publisher。
3. 将左右臂 7 个关节目标位置打包成 `sensor_msgs/msg/JointState`。
4. publish 到左右臂 command topic。
5. 如果使用夹爪，再将 `[width, force]` 打包成 `std_msgs/msg/Float64MultiArray` publish 到夹爪 command topic。

现有代码中，左右臂不是通过 LeRobot 私有协议控制，也不是通过 dataset 控制，而是直接 publish ROS2 topic。后续 VR 摇杆控制只要能生成左右臂目标关节位置，就可以直接走这条 ROS2 command topic。

推荐最小方案：

```text
VR 摇杆 / 推理程序
  -> 生成 left_goal[7] / right_goal[7] / 可选 gripper
  -> ROS2 publish JointState
  -> 机器人已有 telecon / SmoothMotionEngine / 下游控制节点
  -> 电机动作
```

如果 VR 程序本身能运行在 Orin 或能接入同一个 ROS2 网络，优先直接 publish ROS2 topic，不需要 UDP。只有当 VR 程序在 x86 上、不方便装 ROS2 或不方便进入 ROS_DOMAIN_ID 时，才加一层很薄的 UDP：

```text
x86 VR 程序
  -> UDP 发送目标关节值
  -> Orin UDP receiver
  -> Orin ROS2 publish JointState
  -> 机器人动作
```

无论是否经过 UDP，最终控制机器人的是同一组 ROS2 command topic。

## 2. 现有代码里真正需要抽的部分

主要参考文件：

- `src/lerobot/robots/jz_robot/config_jz_robot.py`
- `src/lerobot/robots/jz_robot/jz_robot.py`
- `src/lerobot/configs/robot/jz_robot_three_realsense_ros2_topics.yaml`
- `src/lerobot/configs/robot/jz_bridge_capture.yaml`

### 2.1 关节名称

左臂 7 个关节：

```text
left_joint1
left_joint2
left_joint3
left_joint4
left_joint5
left_joint6
left_joint7
```

右臂 7 个关节：

```text
right_joint1
right_joint2
right_joint3
right_joint4
right_joint5
right_joint6
right_joint7
```

这两个列表非常关键。publish `JointState` 时，`msg.name` 必须和下游控制节点期待的名字一致，`msg.position` 必须和 `msg.name` 一一对应。

当前 `JZRobot.send_action()` 也是按这个顺序发布：

```python
left_msg.name = list(self.config.left_joint_names)
left_msg.position = [left_goal[joint] for joint in self.config.left_joint_names]

right_msg.name = list(self.config.right_joint_names)
right_msg.position = [right_goal[joint] for joint in self.config.right_joint_names]
```

### 2.2 ROS2 topic

根据当前仓库配置，实际机器人命名空间有两种写法。

不带 namespace 的默认写法：

```text
left state:   arm_left/joint_states
right state:  arm_right/joint_states

left command:  telecon/arm_left/joint_commands_input
right command: telecon/arm_right/joint_commands_input

left gripper state:    left_gripper/gripper_status
right gripper state:   right_gripper/gripper_status
left gripper command:  left_gripper/gripper_commands
right gripper command: right_gripper/gripper_commands
```

带 `/robot1` namespace 的写法：

```text
left state:   /robot1/arm_left/joint_states
right state:  /robot1/arm_right/joint_states

left command:  /robot1/telecon/arm_left/joint_commands_input
right command: /robot1/telecon/arm_right/joint_commands_input

left gripper state:    /robot1/left_gripper/gripper_status
right gripper state:   /robot1/right_gripper/gripper_status
left gripper command:  /robot1/left_gripper/gripper_commands
right gripper command: /robot1/right_gripper/gripper_commands
```

上机前以现场 `ros2 topic list` 为准。当前 `jz_robot_three_realsense_ros2_topics.yaml` 使用的是 `/robot1/...`，`jz_robot_three_realsense.yaml` 使用的是不带 namespace 的相对 topic。

### 2.3 消息类型

左右臂 command topic：

```text
sensor_msgs/msg/JointState
```

只需要用到：

```text
name:     string[]
position: float64[]
```

`velocity` 和 `effort` 在现有 `JZRobot.send_action()` 中没有填。

夹爪 command topic：

```text
std_msgs/msg/Float64MultiArray
```

现有 `JZRobot.send_action()` 发送：

```text
data[0] = width
data[1] = force
```

如果第一版只要求“能动电机”，建议先只做双臂 14 个关节，不把夹爪放进第一版闭环。夹爪可以作为第二步加上。

## 3. 不需要抽出来的 LeRobot 内容

为了轻量化，以下内容不需要进入第一版 VR 控制：

- `RobotConfig.register_subclass("jz_robot")`
- `Robot` 基类
- `make_robot_from_config`
- cameras / RealSense / RTSP
- dataset recording
- `observation_features` / `action_features`
- LeRobot policy runner
- `RobotAction` 字典格式
- calibration 入口
- bridge capture 录制逻辑

这些内容对 LeRobot 录制和训练有用，但对“VR 摇杆发目标位置让电机动”不是必需。

建议保留或重新实现的最小逻辑：

- ROS2 import 检查。
- 创建 publisher。
- 订阅当前 `JointState`，用于启动时拿到当前姿态。
- 等待第一帧 state，避免还没知道当前位置就发零值。
- 按固定顺序 publish 全量 7 关节目标。
- 简单限幅和超时处理。

## 4. 最小 ROS2 publisher 应该长什么样

后续如果要写代码，建议只做一个很薄的类，例如：

```text
JZMinimalRosCommander
  - init ROS2 node
  - subscribe left/right state
  - create left/right command publishers
  - wait_initial_state()
  - publish_arm(side, goal_by_joint)
  - publish_bimanual(left_goal, right_goal)
  - close()
```

### 4.1 初始化

从现有 `JZRobot.connect()` 里抽出来的最小初始化是：

```python
import rclpy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

rclpy.init()
node = rclpy.create_node("pink_vr_jz_minimal_commander")

left_cmd_pub = node.create_publisher(
    JointState,
    "/robot1/telecon/arm_left/joint_commands_input",
    10,
)
right_cmd_pub = node.create_publisher(
    JointState,
    "/robot1/telecon/arm_right/joint_commands_input",
    10,
)
```

如果要读取当前状态，还需要 subscription：

```python
left_state_sub = node.create_subscription(
    JointState,
    "/robot1/arm_left/joint_states",
    left_state_callback,
    10,
)
right_state_sub = node.create_subscription(
    JointState,
    "/robot1/arm_right/joint_states",
    right_state_callback,
    10,
)
```

这就是 `JZRobot.connect()` 中和电机控制直接相关的部分。

### 4.2 发布左右臂命令

从 `JZRobot.send_action()` 抽出来的核心逻辑：

```python
LEFT_JOINT_NAMES = [
    "left_joint1",
    "left_joint2",
    "left_joint3",
    "left_joint4",
    "left_joint5",
    "left_joint6",
    "left_joint7",
]

RIGHT_JOINT_NAMES = [
    "right_joint1",
    "right_joint2",
    "right_joint3",
    "right_joint4",
    "right_joint5",
    "right_joint6",
    "right_joint7",
]

left_msg = JointState()
left_msg.name = LEFT_JOINT_NAMES
left_msg.position = [left_goal[name] for name in LEFT_JOINT_NAMES]

right_msg = JointState()
right_msg.name = RIGHT_JOINT_NAMES
right_msg.position = [right_goal[name] for name in RIGHT_JOINT_NAMES]

left_cmd_pub.publish(left_msg)
right_cmd_pub.publish(right_msg)
```

这里的 `left_goal` / `right_goal` 是普通 dict：

```python
left_goal = {
    "left_joint1": 0.0,
    "left_joint2": 0.0,
    "left_joint3": 0.0,
    "left_joint4": 0.0,
    "left_joint5": 0.0,
    "left_joint6": 0.0,
    "left_joint7": 0.0,
}
```

实际不要上来发全 0。正确做法是先订阅当前 state，把当前 state 当初始 goal，然后在这个基础上做小幅变化。

## 5. VR 摇杆到关节目标的建议数据流

现有 `JZRobot.send_action()` 最终发送的是绝对关节位置，不是速度，也不是增量。因此 VR 控制程序最好在内部维护一份 `goal`：

```text
启动
  -> 等待左右臂 state
  -> current_state 复制为 goal

每个 VR tick，约 80 Hz
  -> 读取摇杆输入
  -> 将摇杆输入转换成关节增量或目标位置
  -> 更新 goal
  -> 限幅
  -> publish 完整 left JointState 和 right JointState
```

如果摇杆输出的是速度或归一化输入，建议按 80 Hz 积分：

```text
period_s = 1 / 80 = 0.0125
delta = joystick_axis * max_joint_speed * period_s
goal[joint] = goal[joint] + delta
```

如果摇杆或推理模型直接输出绝对关节目标，就直接写入 `goal[joint]`，但仍建议做限幅和平滑。

### 5.1 为什么要发布完整 7 关节

`JZRobot.send_action()` 每次都会为左臂发布完整 7 个关节，为右臂发布完整 7 个关节。轻量实现也建议保持这个行为。

不建议第一版只发布某一个关节，原因是下游控制节点是否支持 partial `JointState` 不确定。发布完整 7 关节可以和现有机器人注册代码保持一致。

### 5.2 不要发送零值作为默认值

非常重要：机器人控制里，`0.0` 不是“无动作”，而是一个真实目标位置。

第一版控制程序中，任何未知关节都应该用当前 state 或上一次 goal 补齐，不应该用 `0.0` 补齐。

建议策略：

```text
启动时：
  goal = latest_state

某个 tick 没有 VR 输入：
  goal 保持不变

某个关节没有新目标：
  goal[joint] 保持上一次值

state 或 VR 输入长时间断流：
  停止更新 goal，必要时继续 publish hold position，等待人工处理
```

## 6. 80 Hz 频率建议

你现在的 VR 手柄解算大约是 80 Hz，对应周期：

```text
12.5 ms
```

建议：

- ROS2 publish loop 也按 80 Hz。
- 不要在 publish loop 里做耗时推理或阻塞 IO。
- 如果推理比 80 Hz 慢，publish loop 仍按 80 Hz 发送最近一次 goal。
- 如果 VR 输入偶发抖动，优先使用“latest command buffer + 固定 80 Hz publisher”。
- ROS2 publisher `qos_depth` 先沿用现有配置 `10`。

推荐结构：

```text
线程 / timer A: VR 输入
  - 读取摇杆
  - 更新 latest_goal

线程 / timer B: ROS publish, 80 Hz
  - 取 latest_goal
  - publish JointState
```

第一版也可以单线程：

```text
while running:
  started = monotonic()
  read_vr_once()
  update_goal()
  publish_goal()
  sleep(max(0, 0.0125 - elapsed))
```

如果后面加入模型推理，建议把推理和 ROS publish 解耦，避免推理耗时导致 command topic 频率忽高忽低。

## 7. 直接 ROS2 方案

适用情况：

- VR 程序跑在 Orin 上。
- 或 VR 程序跑在能访问机器人 ROS2 domain 的机器上。
- 或可以正常 `source` ROS2 环境，并设置正确 `ROS_DOMAIN_ID`。

链路：

```text
VR 程序
  -> rclpy publisher
  -> /robot1/telecon/arm_left/joint_commands_input
  -> /robot1/telecon/arm_right/joint_commands_input
```

优点：

- 最简单。
- 延迟最低。
- 少一层协议和进程。
- 和现有 `JZRobot.send_action()` 完全一致。

缺点：

- VR 程序需要 ROS2 Python 环境。
- 网络和 ROS_DOMAIN_ID 必须配置正确。

建议第一版优先尝试这个方案。

## 8. UDP 转 ROS2 方案

适用情况：

- VR 程序跑在 x86。
- x86 不方便接 ROS2。
- 只想让 x86 发一个轻量 UDP 包，Orin 上负责 ROS2 publish。

链路：

```text
x86 VR 程序
  -> UDP packet
  -> Orin pink_vr_udp_receiver
  -> ROS2 JointState publish
  -> robot command topics
```

UDP receiver 也不要依赖 LeRobot。它只需要做四件事：

1. 监听 UDP。
2. 校验来源 IP、seq、时间戳。
3. 把 UDP payload 解析成左右臂关节目标。
4. 调用同一个最小 ROS publisher。

建议 UDP payload 第一版用 JSON 就够了，方便排查：

```json
{
  "version": 1,
  "type": "jz_joint_command",
  "seq": 123,
  "stamp_ns": 1790000000000000000,
  "robot": "robot1",
  "left": {
    "left_joint1": 0.1,
    "left_joint2": -0.2,
    "left_joint3": 0.0,
    "left_joint4": 0.0,
    "left_joint5": 0.0,
    "left_joint6": 0.0,
    "left_joint7": 0.0
  },
  "right": {
    "right_joint1": 0.1,
    "right_joint2": -0.2,
    "right_joint3": 0.0,
    "right_joint4": 0.0,
    "right_joint5": 0.0,
    "right_joint6": 0.0,
    "right_joint7": 0.0
  }
}
```

注意：上面的 `0.0` 只是格式示例，不是推荐上机目标值。真实 payload 应该来自当前 state 初始化后的 goal。

### 8.1 不要直接复用 Phase 2 dry-run receiver 上机

当前 `udp_test/test_scripts/arm_side/orin_udp_command_receiver.py` 明确是 dry-run：

```text
NOT publishing ROS command topics
robot will not move
```

它适合验证 UDP 包格式，不适合直接上机控制。后续如果要做 UDP 转 ROS2，需要在 `udp_test/pink_vr` 下单独写一个 receiver，并且代码和日志里明确标注“会 publish ROS command topic”。

## 9. 上机前确认步骤

只读确认，不会让机器人动：

```bash
ros2 topic list
ros2 topic info /robot1/arm_left/joint_states
ros2 topic info /robot1/arm_right/joint_states
ros2 topic info /robot1/telecon/arm_left/joint_commands_input
ros2 topic info /robot1/telecon/arm_right/joint_commands_input
ros2 topic echo --once /robot1/arm_left/joint_states
ros2 topic echo --once /robot1/arm_right/joint_states
```

需要确认：

- state topic 是否存在。
- command topic 是否存在。
- command topic 类型是否是 `sensor_msgs/msg/JointState`。
- state 中的 `name` 是否包含 `left_joint1..left_joint7` 和 `right_joint1..right_joint7`。
- position 的单位和 command 期望单位一致。现有代码没有做度/弧度转换，只是 float 原样透传。
- 是否需要 `/robot1` namespace。
- `ROS_DOMAIN_ID` 是否和机器人一致。当前 `jz_bridge_capture.yaml` 里写的是 `50`，但现场仍以机器人环境为准。

如果需要观察现有 VR 程序发什么 topic，可以先用：

```bash
bash udp_test/vr_test/probe_vr_topics.sh
```

这个脚本是只读探测，不会 publish command。

## 10. 第一版上机验证建议

为了避免一上来发送错误目标，建议第一版按以下顺序做。

### 10.1 只读 state

先只订阅左右臂 state，打印当前 14 个关节值，不创建 command publisher。

成功标准：

- 能收到左右臂 state。
- 14 个关节名字完整。
- state 刷新稳定。

### 10.2 创建 publisher 但不 publish

启动程序，创建 command publisher，但不发送任何命令。

成功标准：

- 程序能正常启动。
- `ros2 topic info` 能看到 publisher 数量变化。
- 机器人不动作。

### 10.3 publish 当前姿态 hold

读取当前 state，把当前 state 原样作为 goal，以较低频率短时间 publish，例如 5 Hz，确认不会产生明显动作。

成功标准：

- command topic 能收到消息。
- 消息 `name` 和 `position` 顺序正确。
- 机器人保持当前姿态。

### 10.4 提升到 80 Hz hold

仍然 publish 当前姿态 hold，把频率提升到 80 Hz。

成功标准：

- `ros2 topic hz /robot1/telecon/arm_left/joint_commands_input` 接近 80 Hz。
- 机器人仍保持当前姿态。
- CPU 和网络没有明显异常。

### 10.5 小幅单关节测试

在人工确认安全、机器人使能状态正确、急停可用的前提下，只对一个关节加很小的目标变化，并设置严格限幅。

建议先做非常小的增量，不要直接给大目标值：

```text
goal["left_jointX"] = current["left_jointX"] + small_delta
```

具体 `small_delta` 需要按现场单位确认。如果 state/command 是弧度，`0.01 rad` 大约是 `0.57 deg`；如果是其他单位，需要按现场协议重新确认。

## 11. 必须做的安全边界

即使第一版追求简单，也建议保留这些边界：

- 启动时必须先收到当前 state。
- 不允许未知关节用 0 填充。
- 每次 publish 必须包含完整左臂 7 关节和右臂 7 关节。
- 限制每个 tick 的最大增量。
- 限制 goal 相对启动姿态或当前姿态的最大偏移。
- VR 输入超时后不要继续积分。
- UDP 模式下校验 sender IP。
- UDP 模式下校验 seq 单调递增，丢弃旧包。
- 日志里打印 publish topic、频率、seq、目标摘要。
- 不要在没有人工确认的情况下启动会 publish command topic 的程序。

一个简单限幅策略：

```text
per_tick_delta_limit = max_joint_speed * 0.0125
goal[joint] = clamp(goal[joint], previous_goal[joint] - per_tick_delta_limit, previous_goal[joint] + per_tick_delta_limit)
```

另一个相对安全的限幅策略：

```text
goal[joint] = clamp(goal[joint], initial_state[joint] - max_relative, initial_state[joint] + max_relative)
```

`JZRobot` 里已有 `max_relative_target` 的概念，但轻量实现不用依赖它，可以自己写一个很小的 clamp。

## 12. 后续真正写代码时的最小文件建议

如果后续要把本文档落成代码，建议放在：

```text
udp_test/pink_vr/
  minimal_ros_commander.py
  direct_vr_to_ros.py
  udp_to_ros_receiver.py
  docs/
    jz_robot_minimal_vr_control.md
```

第一步只需要：

```text
minimal_ros_commander.py
```

这个文件只负责 ROS2：

- topic 配置
- state 订阅
- command 发布
- 80 Hz publish loop
- close / shutdown

如果 VR 程序可以直接 import 它，那么不需要 UDP。

第二步才考虑：

```text
udp_to_ros_receiver.py
```

它只负责 UDP 到 `minimal_ros_commander.publish_bimanual()` 的转换。

## 13. 最小实现伪代码

后续写代码时，整体结构可以控制在很少几段：

```text
load constants:
  LEFT_JOINT_NAMES
  RIGHT_JOINT_NAMES
  LEFT_STATE_TOPIC
  RIGHT_STATE_TOPIC
  LEFT_COMMAND_TOPIC
  RIGHT_COMMAND_TOPIC

start ROS:
  rclpy.init()
  node = create_node()
  subscribe state topics
  create command publishers

wait initial state:
  while not have all 14 joints:
    spin_once()

initialize goal:
  goal_left = current_left.copy()
  goal_right = current_right.copy()

loop at 80 Hz:
  joystick = read latest VR input
  goal_left, goal_right = update goals from joystick
  goal_left, goal_right = clamp goals
  publish JointState(left)
  publish JointState(right)
```

这就是从 `JZRobot` 抽出来的最小控制闭环。

## 14. 当前最重要的待确认项

后续实现或上机前，需要现场确认这些值：

1. 实际是否使用 `/robot1` namespace。
2. `ROS_DOMAIN_ID` 是否为 `50`。
3. command topic 是否确实是 `sensor_msgs/msg/JointState`。
4. 关节 position 单位是否为弧度。
5. 下游控制节点是否要求 `header.stamp`。当前 `JZRobot.send_action()` 没填 header，如果现场能工作，轻量实现也可以先不填。
6. 夹爪第一版是否需要控制。如果不需要，先跳过。
7. VR 程序运行位置：Orin 本机、x86 但可接 ROS2、还是 x86 只能 UDP。

在这些确认之前，不建议写复杂框架。先把左右臂 `JointState` command publish 跑通，再考虑夹爪、UDP、推理模型和录制。
