# JZRobot 接入说明

> 本文是 `docs/通用添加机器人.md` 和 `docs/ros2_jointstate_realsense接入lerobot指南.md` 的具体落地版本，专门对应 `JZRobot`。

---

## 1. 机器人定位

`JZRobot` 是一个 **双臂 ROS2 机器人**，左右臂都通过 `sensor_msgs/msg/JointState` 读取状态；命令侧也采用 `sensor_msgs/msg/JointState`。

结合你给出的接口和 `SmoothMotionEngine` 的实现，`JZRobot` 实际上有两种接入方式，当前代码默认按**平滑模式**配置：

- 方式 A：**先接入 SmoothMotionEngine，再由它转发到 passthrough**（默认）
  - 左臂状态：`arm_left/joint_states`
  - 左臂命令：`telecon/arm_left/joint_commands_input`
  - 右臂状态：`arm_right/joint_states`
  - 右臂命令：`telecon/arm_right/joint_commands_input`
- 方式 B：**直连机器人 passthrough**
  - 左臂状态：`arm_left/joint_states`
  - 左臂命令：`arm_left/joint_passthrough`
  - 右臂状态：`arm_right/joint_states`
  - 右臂命令：`arm_right/joint_passthrough`

你现在说要先接入平滑模式，因此 `JZRobot` 默认就应该发布到 `telecon/.../joint_commands_input`。
只有当你明确不经过 `SmoothMotionEngine` 时，才需要把命令 topic 改回 `arm_left/right/joint_passthrough`。

相机部分不绑定 ROS2，推荐直接复用仓库现有的 RealSense 相机实现。

---

## 2. 代码位置

本次已经为 `JZRobot` 创建以下代码骨架：

```text
src/lerobot/robots/jz_robot/
├── __init__.py
├── config_jz_robot.py
└── jz_robot.py
```

其中：

- `config_jz_robot.py`：定义 `JZRobotConfig`
- `jz_robot.py`：定义 `JZRobot`

---

## 3. 配置类设计

`JZRobotConfig` 采用双臂结构，分别配置左右臂 joint 名称和 topic：

```python
@RobotConfig.register_subclass("jz_robot")
@dataclass
class JZRobotConfig(RobotConfig):
    left_joint_names: list[str] = field(default_factory=...)
    right_joint_names: list[str] = field(default_factory=...)

    left_joint_state_topic: str = "arm_left/joint_states"
    right_joint_state_topic: str = "arm_right/joint_states"

    left_position_command_topic: str = "telecon/arm_left/joint_commands_input"
    right_position_command_topic: str = "telecon/arm_right/joint_commands_input"

    state_timeout_s: float = 0.2
    qos_depth: int = 10
    max_relative_target: float | dict[str, float] | None = None
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
```

### 3.1 你现在仍需补充的核心信息

虽然骨架已经搭好，但真正运行前仍需明确：

- RealSense 的 `serial_number_or_name`

这是生成稳定 `action_features` / `observation_features` 的必要条件。

对于你当前的 ROS2 工程，`SmoothMotionEngine` 的配置文件已经给出了 joint 名称，分别是：

- 左臂：`left_joint1` ~ `left_joint7`
- 右臂：`right_joint1` ~ `right_joint7`

---

## 4. JZRobot 的 feature 命名

`JZRobot` 使用双臂前缀来区分左右臂关节：

- 左臂：`left_<joint_name>.pos`
- 右臂：`right_<joint_name>.pos`

例如：

```text
left_shoulder_pan.pos
left_elbow_flex.pos
right_shoulder_pan.pos
right_elbow_flex.pos
```

这种设计参考了仓库现有的双臂机器人 `BiSOFollower`，可以避免左右臂键名冲突，也更适合后续录制和回放。

---

## 5. JZRobot 的 ROS2 适配逻辑

### 5.1 connect

`connect()` 会完成以下动作：

1. 初始化 `rclpy`
2. 创建 `JZRobot` 对应的 node
3. 创建左右臂状态订阅
4. 创建左右臂命令 publisher
5. 启动 executor 线程
6. 等待左右臂都收到至少一帧完整 `JointState`
7. 连接 RealSense 相机
8. 调用 `configure()`

### 5.2 get_observation

`get_observation()` 会：

1. 从左右臂最近一次 `JointState` 缓存读取关节位置
2. 转成 LeRobot 约定的平铺字典
3. 从每个相机读取 RGB 图像
4. 合并为一个 observation dict

### 5.3 send_action

`send_action()` 会：

1. 读取 `left_*.pos` 和 `right_*.pos`
2. 与当前状态合并成左右臂完整目标位姿
3. 如有需要应用 `max_relative_target` 限幅
4. 分别发布到命令 topic

这里需要注意：

- 如果你选择**SmoothMotionEngine 模式**，就发布到：
  - `telecon/arm_left/joint_commands_input`
  - `telecon/arm_right/joint_commands_input`
- 如果你选择**直连模式**，就发布到：
  - `arm_left/joint_passthrough`
  - `arm_right/joint_passthrough`

这两种方式的区别不是消息类型，而是系统分层位置不同：

- `joint_commands_input` 是 `SmoothMotionEngine` 的**输入**
- `joint_passthrough` 是 `SmoothMotionEngine` 计算后的**输出**

---

## 6. RealSense 相机接法

推荐直接在 `JZRobotConfig.cameras` 中配置 RealSense，不走 ROS2 图像 topic。

示例：

```python
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig


cameras = {
    "front": RealSenseCameraConfig(
        serial_number_or_name="YOUR_REALSENSE_SERIAL",
        fps=30,
        width=640,
        height=480,
    )
}
```

这样 `JZRobot` 在 `connect()` 时会自动连接相机，在 `get_observation()` 时自动采图。

---

## 7. 一个建议的配置示例

下面是一个更接近你当前系统的配置示例。

### 7.1 平滑模式示例（默认推荐）

```python
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.robots.jz_robot import JZRobotConfig


robot = JZRobotConfig(
    id="jz_dual_arm",
    left_joint_state_topic="arm_left/joint_states",
    right_joint_state_topic="arm_right/joint_states",
    left_position_command_topic="telecon/arm_left/joint_commands_input",
    right_position_command_topic="telecon/arm_right/joint_commands_input",
    state_timeout_s=0.2,
    max_relative_target=5.0,
    cameras={
        "front": RealSenseCameraConfig(
            serial_number_or_name="YOUR_REALSENSE_SERIAL",
            fps=30,
            width=640,
            height=480,
        )
    },
)
```

上面这份配置里，`left_joint_names` 和 `right_joint_names` 已经使用代码默认值：

- 左臂：`left_joint1` ~ `left_joint7`
- 右臂：`right_joint1` ~ `right_joint7`

如果你现场系统里的 joint 名称与 `SmoothMotionEngine` 配置一致，就不需要重复填写。

### 7.2 直连 passthrough 示例

如果你想临时绕过平滑器，才把命令 topic 改成 passthrough：

```python
robot = JZRobotConfig(
    id="jz_dual_arm",
    left_joint_state_topic="arm_left/joint_states",
    right_joint_state_topic="arm_right/joint_states",
    left_position_command_topic="arm_left/joint_passthrough",
    right_position_command_topic="arm_right/joint_passthrough",
)
```

---

## 8. 当前实现的特点

当前 `JZRobot` 骨架具有以下特性：

- 支持双臂 ROS2 `JointState` 状态订阅
- 默认支持双臂 ROS2 `JointState` 平滑模式命令发布
- 也支持切换为双臂 ROS2 `JointState` 直连 passthrough 发布
- 支持双臂统一 observation / action feature
- 支持 `max_relative_target` 安全限幅
- 支持组合 RealSense 相机
- 支持接入 `record / replay`

同时也有几个明确前提：

- 认为控制消息类型与状态消息类型一致，均为 `sensor_msgs/msg/JointState`
- 认为控制语义是“发布目标关节位置”
- 校准逻辑由外部系统负责，因此 `calibrate()` 当前为 no-op

如果以后控制 topic 改成别的消息类型，只需要修改 `send_action()` 的消息构造逻辑即可。

如果你是用 `jz_command_teleop` 作为 `teleop.type`，并且它监听的正是
`telecon/arm_left/right/joint_commands_input`，那么机器人侧必须设置
`use_external_commands=true`。否则 `teleop.get_action()` 读到的外部命令还会被
`robot.send_action()` 再发回同一 topic，形成不必要的回写闭环。

---

## 9. 从 SmoothMotionEngine 代码能确认什么

结合 `/home/lwtcnecz0100684/workspace/teleop_ws/src/SmoothMotionEngine`，可以确认以下事实：

1. 左右臂状态消息类型是 `sensor_msgs/msg/JointState`
2. 左右臂输入命令消息类型也是 `sensor_msgs/msg/JointState`
3. `SmoothMotionEngine` 订阅 `telecon/arm_left/right/joint_commands_input`
4. `SmoothMotionEngine` 发布 `arm_left/right/joint_passthrough`
5. 它会按 joint name 做映射；如果 name 不完整，才退回到 position 顺序

所以对 LeRobot 来说，最关键的不是“能不能发 `JointState`”，这个已经明确能发；真正要保证的是：

- LeRobot 默认经过 `SmoothMotionEngine`
- `JointState.name` 严格使用 `left_joint1..7` / `right_joint1..7`

---

## 10. 推荐的下一步

当前最推荐的推进顺序：

1. 先按默认平滑模式接入 `telecon/.../joint_commands_input`
2. 先不加相机，验证 ROS2 双臂状态和命令链路
3. 再补 RealSense
4. 最后跑 `lerobot_record` 做录制验证

---

## 11. 相关文件

- 通用机器人接入：`docs/通用添加机器人.md`
- ROS2 + RealSense 接入：`docs/ros2_jointstate_realsense接入lerobot指南.md`
- JZRobot 配置：`src/lerobot/robots/jz_robot/config_jz_robot.py`
- JZRobot 实现：`src/lerobot/robots/jz_robot/jz_robot.py`
- JZRobot 平滑模式配置模板：`src/lerobot/configs/robot/jz_robot_smooth.yaml`
- JZCommandTeleop 配置模板：`src/lerobot/configs/teleop/jz_command_teleop.yaml`
