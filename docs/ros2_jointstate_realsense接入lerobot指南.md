# ROS2 JointState + RealSense 接入 LeRobot 指南

> 适用场景：你的机器人状态读取和控制主要通过 ROS2 完成，关节状态使用 `sensor_msgs/msg/JointState`，相机使用 Intel RealSense，并且相机不一定走 ROS2。

本文基于本仓库当前实现方式整理，目标是帮助你把一台“ROS2 机器人”以最小改动接入 LeRobot 的 `record / replay / teleoperate / async_inference` 流程。

---

## 1. 结论先说

对于“**机器人接口走 ROS2，状态话题是 `JointState`，相机直接走 RealSense 驱动**”这个场景，推荐采用下面的接入方案：

- **机器人本体**：实现一个新的 `Robot` 子类，负责 ROS2 节点、关节状态订阅、命令发布、动作限幅、连接生命周期。
- **相机**：直接复用仓库已有的 `RealSenseCameraConfig` 和 `RealSenseCamera`，不需要额外写 ROS2 相机适配层。
- **配置**：实现一个 `RobotConfig` 子类，把 ROS2 topic、关节名、超时、相机配置都放进去。
- **工厂注册**：让 `make_robot_from_config()` 能创建你的机器人类，或者按第三方插件规范暴露。

也就是说，这个场景下你通常**不需要实现 `MotorsBus`**，重点是写一个“**ROS2 -> LeRobot Robot 接口适配器**”。

---

## 2. 这个方案为什么适合当前仓库

LeRobot 的硬件抽象是分层的：

```text
Robot
├─ 负责 connect / get_observation / send_action / disconnect
├─ 负责声明 observation_features / action_features
└─ 可以组合 cameras

Camera
├─ 负责 connect / async_read / disconnect
└─ 负责图像采集实现
```

在本仓库里：

- 机器人统一抽象在 `src/lerobot/robots/robot.py`
- 机器人配置基类在 `src/lerobot/robots/config.py`
- 相机抽象在 `src/lerobot/cameras/camera.py`
- RealSense 已经有现成实现：
  - `src/lerobot/cameras/realsense/configuration_realsense.py`
  - `src/lerobot/cameras/realsense/camera_realsense.py`

因此，你最自然的做法就是：

1. 用 ROS2 实现机器人 `Robot`
2. 用现成 RealSense 实现相机 `Camera`
3. 在机器人里通过 `make_cameras_from_configs()` 组合这些相机

这和仓库现有实现风格完全一致。

---

## 3. 你需要实现哪些接口

### 3.1 必需接口

你必须实现一个 `Robot` 子类，并补齐下面这些成员：

- `observation_features`
- `action_features`
- `is_connected`
- `connect()`
- `is_calibrated`
- `calibrate()`
- `configure()`
- `get_observation()`
- `send_action()`
- `disconnect()`

这是 LeRobot 真正依赖的最小接口集合。

### 3.2 建议额外保留的属性

虽然 `Robot` 抽象类没有强制要求 `self.cameras`，但录制脚本和现有机器人实现都会使用它，因此建议始终保留：

```python
self.cameras = make_cameras_from_configs(config.cameras)
```

如果没有相机，就设为空字典：

```python
self.cameras = {}
```

---

## 4. 推荐的目录结构

建议新建如下目录：

```text
src/lerobot/robots/my_ros2_robot/
├── __init__.py
├── config_my_ros2_robot.py
└── my_ros2_robot.py
```

其中：

- `config_my_ros2_robot.py`：定义 `RobotConfig` 子类
- `my_ros2_robot.py`：定义 `Robot` 子类
- `__init__.py`：导出配置类和机器人类

示例：

```python
from .config_my_ros2_robot import MyRos2RobotConfig
from .my_ros2_robot import MyRos2Robot

__all__ = ["MyRos2Robot", "MyRos2RobotConfig"]
```

---

## 5. 推荐的配置类设计

对于 ROS2 `JointState` 机器人，配置类至少建议包含以下内容：

- `joint_names`：LeRobot 关节顺序定义
- `state_topic`：订阅状态话题
- `command_topic`：发布控制命令话题
- `state_timeout_s`：状态超时阈值
- `max_relative_target`：动作安全限幅
- `cameras`：RealSense 相机配置

建议骨架如下：

```python
from dataclasses import dataclass, field

from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("my_ros2_robot")
@dataclass
class MyRos2RobotConfig(RobotConfig):
    joint_names: list[str] = field(
        default_factory=lambda: [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]
    )
    state_topic: str = "/joint_states"
    command_topic: str = "/joint_command"
    state_timeout_s: float = 0.2
    max_relative_target: float | dict[str, float] | None = None

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "front": RealSenseCameraConfig(
                serial_number_or_name="YOUR_REALSENSE_SERIAL",
                fps=30,
                width=640,
                height=480,
            )
        }
    )
```

### 5.1 关于 `joint_names`

`joint_names` 是这类 ROS2 机器人里最关键的配置之一，它决定：

- LeRobot 对外暴露哪些 action / observation 键
- `send_action()` 下发命令时采用什么顺序
- `get_observation()` 最终输出哪些关节

如果你的 `JointState.name` 顺序会变化，**也没关系**，因为推荐实现会按名字映射，不按数组下标硬编码。

### 5.2 关于相机配置

如果使用仓库现成 RealSense，相机配置直接走：

```python
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
```

已有字段包括：

- `serial_number_or_name`
- `fps`
- `width`
- `height`
- `color_mode`
- `use_depth`
- `rotation`
- `warmup_s`

对于第一版接入，建议先只用 RGB：

- `use_depth=False`
- 分辨率先用 `640x480`
- 帧率先用 `30`

---

## 6. 推荐的特征设计

### 6.1 observation_features

对于 `JointState` 机器人，最建议采用和本仓库机械臂一致的命名风格：

```text
<joint_name>.pos
```

例如：

```text
shoulder_pan.pos
shoulder_lift.pos
elbow_flex.pos
wrist_flex.pos
wrist_roll.pos
gripper.pos
```

如果有相机，再加上：

```text
front
wrist
side
```

因此 `observation_features` 推荐写成：

```python
@property
def _motors_ft(self) -> dict[str, type]:
    return {f"{joint}.pos": float for joint in self.config.joint_names}


@property
def _cameras_ft(self) -> dict[str, tuple]:
    return {
        key: (cfg.height, cfg.width, 3)
        for key, cfg in self.config.cameras.items()
    }


@cached_property
def observation_features(self) -> dict[str, type | tuple]:
    return {**self._motors_ft, **self._cameras_ft}
```

### 6.2 action_features

如果你的控制命令本质上也是关节位置目标，建议 action 直接和 observation 的关节部分保持一致：

```python
@cached_property
def action_features(self) -> dict[str, type]:
    return self._motors_ft
```

这种设计的好处是：

- 最接近现有 `so_follower` / `omx_follower` 风格
- 录制、回放、处理链更顺滑
- 以后接 teleoperator 时键名最容易对齐

### 6.3 为什么不要一上来暴露 velocity / effort

虽然 `JointState` 里可能有：

- `position`
- `velocity`
- `effort`

但第一版建议只暴露 `position`：

- 现有仓库大部分真实机器人 action 都是位置目标
- 多暴露字段会增加处理链复杂度
- velocity / effort 很多时候采样不稳定或语义不统一

等主链路跑通后，再考虑把它们扩展进 observation。

---

## 7. ROS2 接口如何映射到 LeRobot 接口

### 7.1 `connect()`

推荐在 `connect()` 里做这些事：

1. 初始化 `rclpy`
2. 创建 ROS2 node
3. 创建 `JointState` subscriber
4. 创建命令 publisher
5. 启动 executor 线程
6. 等待第一帧 `JointState`
7. 连接所有相机
8. 执行 `configure()`

推荐行为：

- `connect()` 返回前，至少已经收到一帧合法状态
- 否则应抛出超时异常，而不是“假装连上”

### 7.2 `is_connected`

ROS2 机器人不建议只看“node 是否存在”，更建议看：

- ROS2 通信对象已创建
- 最近收到过状态
- 最新状态没有超时

如果你还有 RealSense，相机状态也可以纳入：

```python
return self._ros_ready and all(cam.is_connected for cam in self.cameras.values())
```

### 7.3 `get_observation()`

推荐流程：

1. 从订阅回调缓存里读取最近一次关节状态
2. 将关节位置转成 `{f"{joint}.pos": value}`
3. 调用每个相机的 `async_read()` 获取当前 RGB
4. 合并成单个 `dict`

如果最新状态过旧，应直接报错，不建议静默返回旧数据。

### 7.4 `send_action()`

推荐流程：

1. 从 LeRobot action 中取出每个 `*.pos`
2. 转成 `joint -> goal_position`
3. 如配置了 `max_relative_target`，基于当前状态做安全限幅
4. 构造控制消息并发布
5. 返回“实际下发”的 action dict

这里推荐复用仓库已有的：

```python
from lerobot.robots.utils import ensure_safe_goal_position
```

这样你的动作裁剪逻辑会和现有机器人保持一致。

### 7.5 `calibrate()` 和 `is_calibrated`

如果你的机器人校准逻辑已经在 ROS2 控制栈或底层驱动完成，那么可以采用最简单实现：

```python
@property
def is_calibrated(self) -> bool:
    return True


def calibrate(self) -> None:
    pass
```

如果你有 ROS2 的校准 service，也可以在这里调用它。

### 7.6 `disconnect()`

推荐顺序：

1. 先断开相机
2. 停止 executor
3. 等待 spin 线程退出
4. 销毁 node
5. 清理 publisher / subscriber / 缓存状态

需要注意：如果你未来同一进程里还会实例化别的 ROS2 设备，`rclpy.shutdown()` 的时机要谨慎处理。

---

## 8. 一个推荐的实现骨架

下面给出一版适合当前仓库风格的骨架。假设：

- 状态话题类型：`sensor_msgs/msg/JointState`
- 控制话题类型：`sensor_msgs/msg/JointState`
- 控制语义：`position`

如果你的控制侧不是 `JointState`，只需要改 `send_action()` 构造消息的部分，整体结构不变。

```python
import threading
import time
from functools import cached_property

import rclpy
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import JointState

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.robot import Robot
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config_my_ros2_robot import MyRos2RobotConfig


class MyRos2Robot(Robot):
    config_class = MyRos2RobotConfig
    name = "my_ros2_robot"

    def __init__(self, config: MyRos2RobotConfig):
        super().__init__(config)
        self.config = config

        self.cameras = make_cameras_from_configs(config.cameras)

        self._node = None
        self._executor = None
        self._spin_thread = None
        self._cmd_pub = None
        self._state_sub = None

        self._lock = threading.Lock()
        self._latest_joint_pos: dict[str, float] = {}
        self._last_state_time: float | None = None

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{joint}.pos": float for joint in self.config.joint_names}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            key: (cfg.height, cfg.width, 3)
            for key, cfg in self.config.cameras.items()
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        ros_ready = self._node is not None and self._last_state_time is not None
        cameras_ready = all(cam.is_connected for cam in self.cameras.values())
        return ros_ready and cameras_ready

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def _state_cb(self, msg: JointState) -> None:
        name_to_pos = {}
        for idx, name in enumerate(msg.name):
            if idx < len(msg.position):
                name_to_pos[name] = float(msg.position[idx])

        with self._lock:
            self._latest_joint_pos.update(name_to_pos)
            self._last_state_time = time.monotonic()

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        if not rclpy.ok():
            rclpy.init()

        self._node = rclpy.create_node(f"{self.name}_{self.id or 'default'}")
        self._cmd_pub = self._node.create_publisher(JointState, self.config.command_topic, 10)
        self._state_sub = self._node.create_subscription(JointState, self.config.state_topic, self._state_cb, 10)

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        deadline = time.monotonic() + self.config.state_timeout_s
        while self._last_state_time is None and time.monotonic() < deadline:
            time.sleep(0.01)

        if self._last_state_time is None:
            raise TimeoutError(f"No JointState received from {self.config.state_topic}")

        for cam in self.cameras.values():
            cam.connect()

        self.configure()

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        with self._lock:
            joint_pos = dict(self._latest_joint_pos)
            last_state_time = self._last_state_time

        if last_state_time is None or time.monotonic() - last_state_time > self.config.state_timeout_s:
            raise TimeoutError("Latest JointState is stale")

        obs = {f"{joint}.pos": joint_pos[joint] for joint in self.config.joint_names}

        for key, cam in self.cameras.items():
            obs[key] = cam.async_read()

        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        goal_pos = {
            key.removesuffix(".pos"): float(val)
            for key, val in action.items()
            if key.endswith(".pos")
        }

        if self.config.max_relative_target is not None:
            with self._lock:
                present_pos = {joint: self._latest_joint_pos[joint] for joint in self.config.joint_names}
            goal_present_pos = {joint: (goal_pos[joint], present_pos[joint]) for joint in goal_pos}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        msg = JointState()
        msg.name = list(self.config.joint_names)
        msg.position = [goal_pos[joint] for joint in self.config.joint_names]
        self._cmd_pub.publish(msg)

        return {f"{joint}.pos": goal_pos[joint] for joint in self.config.joint_names}

    @check_if_not_connected
    def disconnect(self) -> None:
        for cam in self.cameras.values():
            cam.disconnect()

        if self._executor is not None:
            self._executor.shutdown()

        if self._spin_thread is not None:
            self._spin_thread.join(timeout=1.0)

        if self._node is not None:
            self._node.destroy_node()

        self._node = None
        self._executor = None
        self._spin_thread = None
        self._cmd_pub = None
        self._state_sub = None
        self._last_state_time = None
```

---

## 9. ROS2 `JointState` 接入时最重要的注意事项

### 9.1 永远按 `name` 建映射，不要依赖数组顺序

`JointState` 的核心坑点在于：

- `msg.name`
- `msg.position`
- `msg.velocity`
- `msg.effort`

这些数组语义上是“**名字对齐**”，不是“固定索引含义”。

因此必须这样处理：

```python
for idx, name in enumerate(msg.name):
    if idx < len(msg.position):
        joint_map[name] = msg.position[idx]
```

不要假设：

```python
msg.position[0] == shoulder_pan
```

### 9.2 你的 `joint_names` 才是 LeRobot 里的标准顺序

LeRobot 输出 observation 和 action 的顺序，应以 `config.joint_names` 为准。

这意味着：

- ROS2 输入可以乱序
- LeRobot 输出仍然保持稳定
- 回放数据、训练数据、策略输出都更稳定

### 9.3 `observation_features` / `action_features` 不能依赖实时 ROS2 数据

`lerobot_record` 会在 `robot.connect()` 之前先读取这两个属性生成数据集 schema。

所以这两个属性只能依赖：

- config
- 本地静态定义

不能依赖“等接上 ROS2 再发现有哪些 joint”。

### 9.4 状态过期要显式报错

ROS2 话题如果卡住，但进程没退出，很容易出现“程序仍然运行，但状态已经不更新”的情况。

因此建议：

- 每次回调更新时间戳
- `get_observation()` 检查是否超时
- 超时直接抛异常，而不是继续使用旧状态

### 9.5 `send_action()` 最好返回实际下发值

如果你做了动作裁剪，`send_action()` 返回值应该是裁剪后的实际动作，而不是原始输入动作。

这和仓库现有 `so_follower`、`omx_follower` 的行为一致。

---

## 10. 相机接入建议：直接用 RealSense，不要先走 ROS2

如果你已经在这台机器上可以直接使用 RealSense，第一版接入最推荐：

- 不走 ROS2 图像 topic
- 直接用 `RealSenseCameraConfig` + `RealSenseCamera`

原因：

- 现成实现已存在，少写很多代码
- 直接采图延迟更低
- 更容易和当前 `robot.cameras` 体系兼容
- `record` 脚本能直接利用现有图像写入逻辑

推荐配置示例：

```python
cameras = {
    "front": RealSenseCameraConfig(
        serial_number_or_name="YOUR_REALSENSE_SERIAL",
        fps=30,
        width=640,
        height=480,
    )
}
```

### 10.1 第一版建议只录 RGB

虽然 RealSense 支持 depth，但第一版建议：

- 先只录 RGB
- observation 里的图像 shape 保持 `(H, W, 3)`

这样最符合当前仓库已有机器人实现风格，也最不容易引入额外处理问题。

---

## 11. 工厂注册方式

### 11.1 直接改当前仓库

最直接做法是在：

```text
src/lerobot/robots/utils.py
```

里的 `make_robot_from_config()` 增加一支：

```python
elif config.type == "my_ros2_robot":
    from .my_ros2_robot import MyRos2Robot
    return MyRos2Robot(config)
```

### 11.2 做成第三方插件

如果你不想改主仓库，也可以按 LeRobot 的第三方插件约定独立成包：

- 让配置类继承 `ChoiceRegistry`
- 命名保持 `MyRos2RobotConfig -> MyRos2Robot`
- 安装后让 CLI 自动发现

对于自定义硬件长期维护，这种方式更干净。

---

## 12. 对 `record / replay / teleoperate` 的影响

只要你的机器人实现满足前面接口，现有脚本可以直接用。

### 12.1 `lerobot_record`

依赖：

- `robot.action_features`
- `robot.observation_features`
- `robot.connect()`
- `robot.get_observation()`
- `robot.send_action()`
- `robot.disconnect()`

### 12.2 `lerobot_replay`

依赖：

- `connect()`
- `get_observation()`
- `send_action()`
- `disconnect()`

### 12.3 `lerobot_calibrate`

如果你没有额外校准逻辑，`calibrate()` 可以是 no-op。

---

## 13. 测试建议

建议先写 mock 单元测试，而不是直接上真机。

至少覆盖以下几类：

- **连接测试**：能否正常 connect / disconnect
- **状态测试**：收到 `JointState` 后 `get_observation()` 是否输出正确键名
- **动作测试**：`send_action()` 是否正确构造消息并发布
- **限幅测试**：`max_relative_target` 是否生效
- **超时测试**：状态过旧时是否抛错
- **相机测试**：有相机时 observation 是否包含图像键

mock 思路：

- mock `rclpy.create_node`
- mock publisher / subscriber
- 直接手动调用 `_state_cb()` 注入 `JointState`
- mock `RealSenseCamera` 或 `make_cameras_from_configs()`

---

## 14. 常见坑点

### 14.1 控制 topic 也叫 `JointState`，但语义不一定可靠

有些 ROS2 系统虽然用 `JointState` 发命令，但这并不是最标准的控制消息定义。

如果你的控制侧后续换成了：

- 自定义 msg
- `trajectory_msgs/msg/JointTrajectory`
- controller 专用 command topic

通常只需要改 `send_action()` 里的消息组装逻辑即可，LeRobot 的 feature 和整体结构不用重写。

### 14.2 不要让 `connect()` 返回太早

如果 `connect()` 还没收到第一帧状态就返回，后面 `record` 或 `replay` 很容易一开始就失败。

### 14.3 不要把图像逻辑混进 ROS2 关节回调

关节状态和相机采集节奏通常不同，建议继续分层：

- 关节状态：ROS2 subscriber 缓存
- 图像：`camera.async_read()`

这样结构更清晰，也更贴合当前仓库架构。

### 14.4 不要让 feature 键名频繁变化

一旦你录过数据，`action_features` / `observation_features` 的键名就尽量保持稳定，否则会影响：

- 数据集复用
- replay
- policy 输出对齐

---

## 15. 最推荐的落地路径

如果你要尽快跑通，建议按下面顺序推进：

1. **先实现无相机版本**
   - 只接 ROS2 `JointState`
   - 跑通 `connect -> get_observation -> send_action -> disconnect`

2. **再接 RealSense 相机**
   - 增加 `cameras`
   - observation 中加入图像键

3. **再跑 `lerobot_record`**
   - 验证数据能否正常录制

4. **最后再考虑 teleoperator 或 policy 接入**
   - 保证链路逐步稳定

这是风险最低、定位问题最快的方式。

---

## 16. 一句话总结

对于“**ROS2 `JointState` 机器人 + 非 ROS2 RealSense 相机**”这个场景，最合理的 LeRobot 接入方式是：

- **用 ROS2 封装一个 `Robot` 子类**
- **用仓库现成 RealSense 封装相机**
- **在机器人里组合两者**

这样既符合当前仓库架构，也最容易复用已有的 `record / replay` 工作流。

如果后续需要落地到真实文件，可以继续实现：

- `src/lerobot/robots/my_ros2_robot/config_my_ros2_robot.py`
- `src/lerobot/robots/my_ros2_robot/my_ros2_robot.py`
- `tests/robots/test_my_ros2_robot.py`

---

## 17. 后续实现所需的最少信息

如果你下一步要正式落代码，至少需要明确下面四项：

- `state_topic` 是什么
- `command_topic` 是什么
- `joint_names` 的标准顺序是什么
- RealSense 的 `serial_number_or_name` 是什么

只要这四项明确，就可以开始写第一版接入代码。

---

## 18. JZRobot 具体落地

如果你的机器人名为 `JZRobot`，并且采用双臂 ROS2 `JointState` 接口，那么根据你当前工程里的 `SmoothMotionEngine`，现在默认优先采用**平滑模式**：

- 通过 SmoothMotionEngine 输入端接入（默认）
  - 左臂状态：`arm_left/joint_states`
  - 左臂命令：`telecon/arm_left/joint_commands_input`
  - 右臂状态：`arm_right/joint_states`
  - 右臂命令：`telecon/arm_right/joint_commands_input`
- 直连 passthrough
  - 左臂状态：`arm_left/joint_states`
  - 左臂命令：`arm_left/joint_passthrough`
  - 右臂状态：`arm_right/joint_states`
  - 右臂命令：`arm_right/joint_passthrough`

同时，你当前工程里已经定义了真实 joint 名称：

- 左臂：`left_joint1` ~ `left_joint7`
- 右臂：`right_joint1` ~ `right_joint7`

这意味着 `JZRobotConfig` 默认就应该使用上面这组 joint 名称，并把命令 topic 默认指向 `telecon/.../joint_commands_input`。

对应的仓库实现路径可以放在：

- `src/lerobot/robots/jz_robot/config_jz_robot.py`
- `src/lerobot/robots/jz_robot/jz_robot.py`

推荐继续阅读：

- `docs/JZRobot接入说明.md`
