# Agilex Piper 双臂机器人 LeRobot 集成完整指南（最终版）

> 本文整合了三个版本的实现方案，提供 Agilex Piper 双臂遥操作机器人与 LeRobot 框架的完整集成指南。
> 
> **核心特性**：7 DOF 双臂（含夹爪）、ROS 通信适配、Mock 测试支持、完整数据采集流程

---

## 目录

1. [硬件规格与接口](#1-硬件规格与接口)
2. [系统架构设计](#2-系统架构设计)
3. [目录结构](#3-目录结构)
4. [配置类实现](#4-配置类实现)
5. [ROS Bridge 实现](#5-ros-bridge-实现)
6. [ROS Camera 适配器](#6-ros-camera-适配器)
7. [AgileXRobot 类](#7-agilexrobot-类)
8. [AgileXTeleoperator 类](#8-agilexteleoperator-类)
9. [工厂注册](#9-工厂注册)
10. [测试方案](#10-测试方案)
11. [部署与使用](#11-部署与使用)
12. [常见问题](#12-常见问题)

---

## 1. 硬件规格与接口

### 1.1 硬件配置

| 组件 | 规格 | 说明 |
|------|------|------|
| **机械臂** | Piper 机械臂（2主 + 2从） | 7 自由度（6关节 + 1夹爪），CAN 总线通信 |
| **主臂（Leader/Master）** | 2x Piper | 遥操作输入端 |
| **从臂（Follower/Puppet）** | 2x Piper | 执行任务端 |
| **相机** | 2x Astra RGB-D | 左右双目视觉 |
| **控制系统** | ROS Noetic | Ubuntu 20.04 |

### 1.2 ROS Topic 列表

```bash
# 主臂状态（遥操作端）
/master/joint_left          # sensor_msgs/JointState - 左主臂 7 个关节
/master/joint_right         # sensor_msgs/JointState - 右主臂 7 个关节

# 从臂状态（执行端）
/puppet/joint_left          # sensor_msgs/JointState - 左从臂状态
/puppet/joint_right         # sensor_msgs/JointState - 右从臂状态
/puppet/joint_left/command  # std_msgs/Float64MultiArray - 左从臂命令
/puppet/joint_right/command # std_msgs/Float64MultiArray - 右从臂命令

# 相机数据
/camera_l/color/image_raw   # sensor_msgs/Image - 左相机 RGB
/camera_r/color/image_raw   # sensor_msgs/Image - 右相机 RGB
/camera_l/depth/image_raw   # sensor_msgs/Image - 左相机深度（可选）
/camera_r/depth/image_raw   # sensor_msgs/Image - 右相机深度（可选）
```

### 1.3 关节命名约定（统一标准）

采用 `{side}_{joint_name}.pos` 格式，便于理解和维护：

| 索引 | 关节名称 | 左臂键名 | 右臂键名 |
|------|----------|----------|----------|
| 0 | 肩偏航 | `left_shoulder_pan.pos` | `right_shoulder_pan.pos` |
| 1 | 肩俯仰 | `left_shoulder_lift.pos` | `right_shoulder_lift.pos` |
| 2 | 肩翻滚 | `left_shoulder_roll.pos` | `right_shoulder_roll.pos` |
| 3 | 肘关节 | `left_elbow.pos` | `right_elbow.pos` |
| 4 | 腕俯仰 | `left_wrist_pitch.pos` | `right_wrist_pitch.pos` |
| 5 | 腕翻滚 | `left_wrist_roll.pos` | `right_wrist_roll.pos` |
| 6 | 夹爪 | `left_gripper.pos` | `right_gripper.pos` |

**总自由度**：14 DOF（左臂 7 + 右臂 7）

---

## 2. 系统架构设计

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LeRobot Framework                             │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    AgileXRobot                               │   │
│  │  ┌─────────────────────┐  ┌─────────────────────┐          │   │
│  │  │  AgileXROSBridge    │  │  RosCamera (x2)     │          │   │
│  │  │  - 订阅 puppet/*    │  │  - /camera_l/...    │          │   │
│  │  │  - 发布 command     │  │  - /camera_r/...    │          │   │
│  │  └─────────────────────┘  └─────────────────────┘          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                 AgileXTeleoperator                           │   │
│  │  ┌─────────────────────┐                                    │   │
│  │  │  AgileXROSBridge    │  ← 订阅 master/* 获取遥操作输入    │   │
│  │  └─────────────────────┘                                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│                         ROS Noetic                                   │
│  roscore + piper_ros + astra_camera                                 │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │       Hardware         │
                    │  2x Piper + 2x Astra   │
                    └────────────────────────┘
```

### 2.2 数据流

```
┌─────────────┐                              ┌─────────────────┐
│   Human     │─────操控主臂──────────────▶│ AgileXTeleop    │
│  Operator   │                              │ get_action()    │
└─────────────┘                              └────────┬────────┘
                                                      │
                                                      ▼ action_dict
┌─────────────────────────────────────────────────────────────────┐
│                        Policy / Direct Map                       │
└─────────────────────────────────────────────────────────────────┘
                                                      │
                                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  AgileXRobot.send_action(action)                                │
│  ├─ 解析 action_dict → left_joints[7], right_joints[7]         │
│  ├─ 安全限幅（max_relative_target）                             │
│  └─ ros_bridge.send_joint_commands(left, right)                 │
└─────────────────────────────────────────────────────────────────┘
                                                      │
                                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  AgileXRobot.get_observation()                                  │
│  ├─ ros_bridge.get_joint_states() → 14 个关节位置              │
│  └─ cameras[*].async_read() → RGB 图像                         │
└─────────────────────────────────────────────────────────────────┘
                                                      │
                                                      ▼
                                             ┌─────────────────┐
                                             │  LeRobotDataset │
                                             │  记录 episode   │
                                             └─────────────────┘
```

---

## 3. 目录结构

```
src/lerobot/
├── robots/
│   └── agilex/
│       ├── __init__.py
│       ├── config_agilex.py          # 配置类
│       ├── agilex.py                 # AgileXRobot 实现
│       └── agilex_ros_bridge.py      # ROS 通信层
│
├── teleoperators/
│   └── agilex/
│       ├── __init__.py
│       ├── config_agilex_teleop.py   # 遥操作配置
│       └── agilex_teleoperator.py    # AgileXTeleoperator 实现
│
└── cameras/
    └── ros_camera/
        ├── __init__.py
        ├── configuration_ros_camera.py
        └── ros_camera.py             # ROS 相机适配器
```

---

## 4. 配置类实现

### 4.1 机器人配置 `config_agilex.py`

```python
# src/lerobot/robots/agilex/config_agilex.py

from dataclasses import dataclass, field
from lerobot.robots.config import RobotConfig
from lerobot.cameras.configs import CameraConfig


# 关节名称常量
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "shoulder_roll",
    "elbow",
    "wrist_pitch",
    "wrist_roll",
    "gripper",
]


@RobotConfig.register_subclass("agilex")
@dataclass
class AgileXConfig(RobotConfig):
    """Agilex Piper 双臂机器人配置"""

    # === ROS 配置 ===
    ros_master_uri: str = "http://localhost:11311"
    node_name: str = "lerobot_agilex"

    # === Topic 配置 ===
    puppet_left_topic: str = "/puppet/joint_left"
    puppet_right_topic: str = "/puppet/joint_right"
    puppet_left_cmd_topic: str = "/puppet/joint_left/command"
    puppet_right_cmd_topic: str = "/puppet/joint_right/command"

    # === 相机配置 ===
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # === 安全参数 ===
    max_relative_target: float = 0.2  # 单步最大关节变化（弧度）

    # === 关节限位（弧度）===
    joint_limits: dict = field(default_factory=lambda: {
        "shoulder_pan": (-2.618, 2.618),
        "shoulder_lift": (-1.571, 1.571),
        "shoulder_roll": (-1.571, 1.571),
        "elbow": (-1.745, 1.745),
        "wrist_pitch": (-1.571, 1.571),
        "wrist_roll": (-2.094, 2.094),
        "gripper": (0.0, 0.085),  # 夹爪行程（米）
    })

    # === Mock 模式（用于测试）===
    mock: bool = False

    def __post_init__(self):
        super().__post_init__()
        # 如果未配置相机，使用默认配置
        if not self.cameras:
            from lerobot.cameras.ros_camera.configuration_ros_camera import RosCameraConfig
            self.cameras = {
                "camera_left": RosCameraConfig(
                    topic_name="/camera_l/color/image_raw",
                    fps=30, width=640, height=480
                ),
                "camera_right": RosCameraConfig(
                    topic_name="/camera_r/color/image_raw",
                    fps=30, width=640, height=480
                ),
            }
```

### 4.2 遥操作配置 `config_agilex_teleop.py`

```python
# src/lerobot/teleoperators/agilex/config_agilex_teleop.py

from dataclasses import dataclass
from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("agilex_teleop")
@dataclass
class AgileXTeleoperatorConfig(TeleoperatorConfig):
    """Agilex 主臂遥操作配置"""

    ros_master_uri: str = "http://localhost:11311"
    node_name: str = "lerobot_agilex_teleop"

    # 主臂 Topic
    master_left_topic: str = "/master/joint_left"
    master_right_topic: str = "/master/joint_right"

    # Mock 模式
    mock: bool = False
```

---

## 5. ROS Bridge 实现

ROS Bridge 封装所有 ROS 通信细节，支持真实硬件和 Mock 模式。

### 5.1 `agilex_ros_bridge.py`

```python
# src/lerobot/robots/agilex/agilex_ros_bridge.py

from __future__ import annotations
import numpy as np
from typing import Protocol, runtime_checkable
from dataclasses import dataclass
import time
import threading


@dataclass
class JointState:
    """关节状态数据结构"""
    position: np.ndarray  # shape: (7,) - 7 个关节位置
    velocity: np.ndarray  # shape: (7,) - 7 个关节速度
    effort: np.ndarray    # shape: (7,) - 7 个关节力矩
    timestamp: float


@runtime_checkable
class AgileXROSBridgeProtocol(Protocol):
    """ROS Bridge 协议定义"""

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def is_connected(self) -> bool: ...
    def get_puppet_state(self) -> tuple[JointState, JointState]: ...
    def get_master_state(self) -> tuple[JointState, JointState]: ...
    def send_joint_commands(self, left: np.ndarray, right: np.ndarray) -> None: ...


class AgileXROSBridge:
    """真实 ROS 通信实现"""

    def __init__(
        self,
        node_name: str = "lerobot_agilex",
        puppet_left_topic: str = "/puppet/joint_left",
        puppet_right_topic: str = "/puppet/joint_right",
        master_left_topic: str = "/master/joint_left",
        master_right_topic: str = "/master/joint_right",
        puppet_left_cmd_topic: str = "/puppet/joint_left/command",
        puppet_right_cmd_topic: str = "/puppet/joint_right/command",
    ):
        self._node_name = node_name
        self._puppet_left_topic = puppet_left_topic
        self._puppet_right_topic = puppet_right_topic
        self._master_left_topic = master_left_topic
        self._master_right_topic = master_right_topic
        self._puppet_left_cmd_topic = puppet_left_cmd_topic
        self._puppet_right_cmd_topic = puppet_right_cmd_topic

        # ROS 对象（延迟初始化）
        self._ros_initialized = False
        self._sub_puppet_left = None
        self._sub_puppet_right = None
        self._sub_master_left = None
        self._sub_master_right = None
        self._pub_left = None
        self._pub_right = None

        # 数据缓存
        self._puppet_left: JointState | None = None
        self._puppet_right: JointState | None = None
        self._master_left: JointState | None = None
        self._master_right: JointState | None = None
        self._lock = threading.Lock()

    def connect(self) -> None:
        """初始化 ROS 节点并订阅 Topic"""
        import rospy
        from sensor_msgs.msg import JointState as RosJointState
        from std_msgs.msg import Float64MultiArray

        if self._ros_initialized:
            raise RuntimeError("Already connected")

        # 初始化 ROS 节点
        rospy.init_node(self._node_name, anonymous=True)
        self._ros_initialized = True

        # 订阅从臂状态
        self._sub_puppet_left = rospy.Subscriber(
            self._puppet_left_topic, RosJointState,
            lambda msg: self._joint_callback(msg, "puppet_left"),
            queue_size=1
        )
        self._sub_puppet_right = rospy.Subscriber(
            self._puppet_right_topic, RosJointState,
            lambda msg: self._joint_callback(msg, "puppet_right"),
            queue_size=1
        )

        # 订阅主臂状态
        self._sub_master_left = rospy.Subscriber(
            self._master_left_topic, RosJointState,
            lambda msg: self._joint_callback(msg, "master_left"),
            queue_size=1
        )
        self._sub_master_right = rospy.Subscriber(
            self._master_right_topic, RosJointState,
            lambda msg: self._joint_callback(msg, "master_right"),
            queue_size=1
        )

        # 创建发布器
        self._pub_left = rospy.Publisher(
            self._puppet_left_cmd_topic, Float64MultiArray, queue_size=1
        )
        self._pub_right = rospy.Publisher(
            self._puppet_right_cmd_topic, Float64MultiArray, queue_size=1
        )

        # 等待首帧数据
        self._wait_for_data(timeout=10.0)
        rospy.loginfo(f"[{self._node_name}] ROS Bridge connected")

    def disconnect(self) -> None:
        """断开连接"""
        if self._sub_puppet_left:
            self._sub_puppet_left.unregister()
        if self._sub_puppet_right:
            self._sub_puppet_right.unregister()
        if self._sub_master_left:
            self._sub_master_left.unregister()
        if self._sub_master_right:
            self._sub_master_right.unregister()
        self._ros_initialized = False

    def is_connected(self) -> bool:
        return self._ros_initialized

    def _joint_callback(self, msg, arm_id: str) -> None:
        """ROS 回调：解析关节状态"""
        state = JointState(
            position=np.array(msg.position[:7], dtype=np.float32),
            velocity=np.array(msg.velocity[:7], dtype=np.float32) if msg.velocity else np.zeros(7),
            effort=np.array(msg.effort[:7], dtype=np.float32) if msg.effort else np.zeros(7),
            timestamp=msg.header.stamp.to_sec() if msg.header.stamp else time.time(),
        )
        with self._lock:
            setattr(self, f"_{arm_id}", state)

    def _wait_for_data(self, timeout: float) -> None:
        """等待首帧数据"""
        import rospy
        start = time.time()
        while time.time() - start < timeout:
            with self._lock:
                if self._puppet_left is not None and self._puppet_right is not None:
                    return
            rospy.sleep(0.01)
        raise TimeoutError("Timeout waiting for puppet joint states")

    def get_puppet_state(self) -> tuple[JointState, JointState]:
        """获取从臂状态"""
        with self._lock:
            if self._puppet_left is None or self._puppet_right is None:
                raise RuntimeError("No puppet state available")
            return self._puppet_left, self._puppet_right

    def get_master_state(self) -> tuple[JointState, JointState]:
        """获取主臂状态"""
        with self._lock:
            if self._master_left is None or self._master_right is None:
                raise RuntimeError("No master state available")
            return self._master_left, self._master_right

    def send_joint_commands(self, left: np.ndarray, right: np.ndarray) -> None:
        """发送关节指令"""
        from std_msgs.msg import Float64MultiArray

        msg_left = Float64MultiArray()
        msg_left.data = left.tolist()
        self._pub_left.publish(msg_left)

        msg_right = Float64MultiArray()
        msg_right.data = right.tolist()
        self._pub_right.publish(msg_right)


class MockAgileXROSBridge:
    """Mock 实现，用于无硬件测试"""

    def __init__(self, **kwargs):
        self._connected = False
        self._puppet_left = np.zeros(7, dtype=np.float32)
        self._puppet_right = np.zeros(7, dtype=np.float32)
        self._master_left = np.zeros(7, dtype=np.float32)
        self._master_right = np.zeros(7, dtype=np.float32)

    def connect(self) -> None:
        self._connected = True
        print("[MockBridge] Connected (mock mode)")

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def get_puppet_state(self) -> tuple[JointState, JointState]:
        return (
            JointState(self._puppet_left, np.zeros(7), np.zeros(7), time.time()),
            JointState(self._puppet_right, np.zeros(7), np.zeros(7), time.time()),
        )

    def get_master_state(self) -> tuple[JointState, JointState]:
        # 模拟主臂有小幅随机运动
        noise = np.random.randn(7) * 0.01
        return (
            JointState(self._master_left + noise, np.zeros(7), np.zeros(7), time.time()),
            JointState(self._master_right + noise, np.zeros(7), np.zeros(7), time.time()),
        )

    def send_joint_commands(self, left: np.ndarray, right: np.ndarray) -> None:
        # Mock 模式下，命令直接更新状态
        self._puppet_left = left.astype(np.float32)
        self._puppet_right = right.astype(np.float32)


def make_ros_bridge(config) -> AgileXROSBridgeProtocol:
    """工厂函数：根据配置创建 Bridge"""
    if config.mock:
        return MockAgileXROSBridge()
    return AgileXROSBridge(
        node_name=config.node_name,
        puppet_left_topic=config.puppet_left_topic,
        puppet_right_topic=config.puppet_right_topic,
        puppet_left_cmd_topic=config.puppet_left_cmd_topic,
        puppet_right_cmd_topic=config.puppet_right_cmd_topic,
    )
```

---

## 6. ROS Camera 适配器

将 ROS Image Topic 转换为 LeRobot 相机接口。

### 6.1 配置类 `configuration_ros_camera.py`

```python
# src/lerobot/cameras/ros_camera/configuration_ros_camera.py

from dataclasses import dataclass
from lerobot.cameras.configs import CameraConfig


@CameraConfig.register_subclass("ros_camera")
@dataclass
class RosCameraConfig(CameraConfig):
    """ROS 相机配置"""

    topic_name: str = "/camera/color/image_raw"

    def __post_init__(self):
        super().__post_init__()
        if not self.topic_name.startswith("/"):
            raise ValueError("topic_name must start with '/'")
```

### 6.2 相机实现 `ros_camera.py`

```python
# src/lerobot/cameras/ros_camera/ros_camera.py

import numpy as np
import threading
from typing import Any

from lerobot.cameras.camera import Camera
from .configuration_ros_camera import RosCameraConfig


class RosCamera(Camera):
    """ROS 图像 Topic → OpenCV 适配器"""

    config_class = RosCameraConfig
    name = "ros_camera"

    def __init__(self, config: RosCameraConfig):
        super().__init__(config)
        self.config = config
        self._sub = None
        self._latest_image: np.ndarray | None = None
        self._lock = threading.Lock()

    @property
    def is_connected(self) -> bool:
        return self._sub is not None

    def connect(self) -> None:
        import rospy
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge

        if self.is_connected:
            raise RuntimeError(f"Camera {self.config.topic_name} already connected")

        self._cv_bridge = CvBridge()
        self._sub = rospy.Subscriber(
            self.config.topic_name,
            Image,
            self._image_callback,
            queue_size=1
        )

        # 等待首帧
        rospy.loginfo(f"Waiting for image from {self.config.topic_name}...")
        start = rospy.Time.now()
        while self._latest_image is None:
            if (rospy.Time.now() - start).to_sec() > 10:
                raise TimeoutError(f"No image from {self.config.topic_name}")
            rospy.sleep(0.01)
        rospy.loginfo(f"Camera {self.config.topic_name} connected")

    def disconnect(self) -> None:
        if self._sub:
            self._sub.unregister()
            self._sub = None
        self._latest_image = None

    def _image_callback(self, msg) -> None:
        """ROS 回调"""
        try:
            cv_image = self._cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            # BGR → RGB
            with self._lock:
                self._latest_image = cv_image[:, :, ::-1].copy()
        except Exception as e:
            import rospy
            rospy.logerr(f"Image conversion error: {e}")

    def read(self) -> np.ndarray | None:
        """同步读取"""
        with self._lock:
            return self._latest_image.copy() if self._latest_image is not None else None

    def async_read(self) -> np.ndarray | None:
        """异步读取（非阻塞）"""
        return self.read()

    @staticmethod
    def find_cameras() -> list[str]:
        """列出可用的 ROS 图像 Topic"""
        try:
            import rospy
            topics = rospy.get_published_topics()
            return [t for t, tp in topics if tp == "sensor_msgs/Image"]
        except:
            return []
```

---

## 7. AgileXRobot 类

核心机器人类，实现 LeRobot Robot 接口。

```python
# src/lerobot/robots/agilex/agilex.py

from __future__ import annotations
import numpy as np
from typing import Any
from functools import cached_property

from lerobot.robots.robot import Robot
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.utils import ensure_safe_goal_position

from .config_agilex import AgileXConfig, JOINT_NAMES
from .agilex_ros_bridge import make_ros_bridge, JointState


class AgileXRobot(Robot):
    """Agilex Piper 双臂机器人"""

    config_class = AgileXConfig
    name = "agilex"

    def __init__(self, config: AgileXConfig):
        super().__init__(config)
        self.config = config
        self.ros_bridge = make_ros_bridge(config)
        self.cameras = make_cameras_from_configs(config.cameras)

    # ===== 特征定义 =====

    @cached_property
    def observation_features(self) -> dict[str, Any]:
        """观测空间定义"""
        features = {}

        # 双臂关节位置（14 DOF）
        for side in ["left", "right"]:
            for joint_name in JOINT_NAMES:
                features[f"{side}_{joint_name}.pos"] = float

        # 相机图像
        for cam_name, cam_cfg in self.config.cameras.items():
            features[cam_name] = (cam_cfg.height, cam_cfg.width, 3)

        return features

    @cached_property
    def action_features(self) -> dict[str, Any]:
        """动作空间定义（仅控制从臂）"""
        features = {}
        for side in ["left", "right"]:
            for joint_name in JOINT_NAMES:
                features[f"{side}_{joint_name}.pos"] = float
        return features

    # ===== 连接管理 =====

    @property
    def is_connected(self) -> bool:
        return self.ros_bridge.is_connected()

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise RuntimeError("Already connected")

        self.ros_bridge.connect()

        for cam in self.cameras.values():
            cam.connect()

    def disconnect(self) -> None:
        for cam in self.cameras.values():
            cam.disconnect()
        self.ros_bridge.disconnect()

    # ===== 核心 I/O =====

    def get_observation(self) -> dict[str, Any]:
        """获取观测数据"""
        obs = {}

        # 获取从臂关节
        left_state, right_state = self.ros_bridge.get_puppet_state()

        for i, joint_name in enumerate(JOINT_NAMES):
            obs[f"left_{joint_name}.pos"] = float(left_state.position[i])
            obs[f"right_{joint_name}.pos"] = float(right_state.position[i])

        # 获取相机图像
        for cam_name, cam in self.cameras.items():
            img = cam.async_read()
            if img is not None:
                obs[cam_name] = img

        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """发送控制指令"""
        # 解析动作
        left_target = np.array([
            action[f"left_{jn}.pos"] for jn in JOINT_NAMES
        ], dtype=np.float32)
        right_target = np.array([
            action[f"right_{jn}.pos"] for jn in JOINT_NAMES
        ], dtype=np.float32)

        # 安全限幅
        if self.config.max_relative_target > 0:
            left_state, right_state = self.ros_bridge.get_puppet_state()
            left_target = self._clip_delta(left_state.position, left_target)
            right_target = self._clip_delta(right_state.position, right_target)

        # 发送指令
        self.ros_bridge.send_joint_commands(left_target, right_target)

        # 返回实际发送的动作
        result = {}
        for i, jn in enumerate(JOINT_NAMES):
            result[f"left_{jn}.pos"] = float(left_target[i])
            result[f"right_{jn}.pos"] = float(right_target[i])
        return result

    def _clip_delta(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        """限制单步变化量"""
        delta = target - current
        delta = np.clip(delta, -self.config.max_relative_target, self.config.max_relative_target)
        return current + delta

    # ===== 校准 =====

    @property
    def is_calibrated(self) -> bool:
        return True  # Agilex 使用内置校准

    def calibrate(self) -> None:
        print("Agilex calibration is handled by ROS driver")
```

---

## 8. AgileXTeleoperator 类

主臂遥操作接口。

```python
# src/lerobot/teleoperators/agilex/agilex_teleoperator.py

from __future__ import annotations
import numpy as np
from typing import Any
from functools import cached_property

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.robots.agilex.agilex_ros_bridge import (
    AgileXROSBridge, MockAgileXROSBridge, JointState
)
from lerobot.robots.agilex.config_agilex import JOINT_NAMES

from .config_agilex_teleop import AgileXTeleoperatorConfig


class AgileXTeleoperator(Teleoperator):
    """Agilex 主臂遥操作"""

    config_class = AgileXTeleoperatorConfig
    name = "agilex_teleop"

    def __init__(self, config: AgileXTeleoperatorConfig):
        super().__init__(config)
        self.config = config

        # 创建 Bridge（仅订阅主臂）
        if config.mock:
            self._bridge = MockAgileXROSBridge()
        else:
            self._bridge = AgileXROSBridge(
                node_name=config.node_name,
                master_left_topic=config.master_left_topic,
                master_right_topic=config.master_right_topic,
                # 不需要 puppet 和 command
                puppet_left_topic="",
                puppet_right_topic="",
                puppet_left_cmd_topic="",
                puppet_right_cmd_topic="",
            )

    @cached_property
    def action_features(self) -> dict[str, Any]:
        """遥操作输出 = 主臂关节"""
        features = {}
        for side in ["left", "right"]:
            for joint_name in JOINT_NAMES:
                features[f"{side}_{joint_name}.pos"] = float
        return features

    @cached_property
    def feedback_features(self) -> dict[str, Any]:
        """反馈特征（可选）"""
        return {}

    @property
    def is_connected(self) -> bool:
        return self._bridge.is_connected()

    def connect(self) -> None:
        self._bridge.connect()

    def disconnect(self) -> None:
        self._bridge.disconnect()

    def get_action(self) -> dict[str, Any]:
        """读取主臂关节位置作为动作"""
        left_state, right_state = self._bridge.get_master_state()

        action = {}
        for i, joint_name in enumerate(JOINT_NAMES):
            action[f"left_{joint_name}.pos"] = float(left_state.position[i])
            action[f"right_{joint_name}.pos"] = float(right_state.position[i])

        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """发送反馈（Agilex 暂不支持力反馈）"""
        pass
```

---

## 9. 工厂注册

### 9.1 注册到机器人工厂

```python
# 在 src/lerobot/robots/__init__.py 或 utils.py 中添加

from lerobot.robots.agilex.config_agilex import AgileXConfig
from lerobot.robots.agilex.agilex import AgileXRobot

# 配置类装饰器已自动注册，无需额外代码
# @RobotConfig.register_subclass("agilex") 已在 config_agilex.py 中定义
```

### 9.2 注册相机

```python
# 在 src/lerobot/cameras/__init__.py 中添加

from lerobot.cameras.ros_camera.configuration_ros_camera import RosCameraConfig
from lerobot.cameras.ros_camera.ros_camera import RosCamera

# 同样通过装饰器自动注册
```

---

## 10. 测试方案

### 10.1 单元测试（Mock 模式）

```python
# tests/robots/test_agilex.py

import pytest
import numpy as np
from lerobot.robots.agilex.config_agilex import AgileXConfig, JOINT_NAMES
from lerobot.robots.agilex.agilex import AgileXRobot


@pytest.fixture
def mock_robot():
    """创建 Mock 模式机器人"""
    config = AgileXConfig(mock=True, cameras={})  # 禁用相机
    robot = AgileXRobot(config)
    robot.connect()
    yield robot
    robot.disconnect()


def test_observation_features(mock_robot):
    """测试观测特征定义"""
    features = mock_robot.observation_features

    # 检查所有关节
    for side in ["left", "right"]:
        for joint_name in JOINT_NAMES:
            assert f"{side}_{joint_name}.pos" in features


def test_action_features(mock_robot):
    """测试动作特征定义"""
    features = mock_robot.action_features
    assert len(features) == 14  # 7 * 2


def test_get_observation(mock_robot):
    """测试观测读取"""
    obs = mock_robot.get_observation()

    # 检查关节数据存在
    assert "left_shoulder_pan.pos" in obs
    assert "right_gripper.pos" in obs


def test_send_action(mock_robot):
    """测试动作发送"""
    action = {}
    for side in ["left", "right"]:
        for joint_name in JOINT_NAMES:
            action[f"{side}_{joint_name}.pos"] = 0.1

    result = mock_robot.send_action(action)

    # 验证返回的动作
    assert len(result) == 14
    assert result["left_shoulder_pan.pos"] == pytest.approx(0.1, abs=0.01)


def test_safety_clipping(mock_robot):
    """测试安全限幅"""
    # 发送一个大动作
    action = {}
    for side in ["left", "right"]:
        for joint_name in JOINT_NAMES:
            action[f"{side}_{joint_name}.pos"] = 10.0  # 超出限幅

    result = mock_robot.send_action(action)

    # 验证被限幅
    max_delta = mock_robot.config.max_relative_target
    assert result["left_shoulder_pan.pos"] <= max_delta
```

### 10.2 硬件集成测试

```python
# tests/manual/test_agilex_hardware.py

"""
运行前确保：
1. roscore 运行中
2. roslaunch piper start_ms_piper.launch 运行中
3. roslaunch astra_camera multi_camera.launch 运行中
"""

from lerobot.robots.agilex.config_agilex import AgileXConfig
from lerobot.robots.agilex.agilex import AgileXRobot
from lerobot.teleoperators.agilex.config_agilex_teleop import AgileXTeleoperatorConfig
from lerobot.teleoperators.agilex.agilex_teleoperator import AgileXTeleoperator
import time


def test_robot_connection():
    """测试机器人连接"""
    config = AgileXConfig()
    robot = AgileXRobot(config)

    robot.connect()
    assert robot.is_connected

    obs = robot.get_observation()
    print(f"Observation keys: {list(obs.keys())}")
    print(f"Left shoulder pan: {obs['left_shoulder_pan.pos']:.4f}")

    robot.disconnect()


def test_teleoperation_loop():
    """测试遥操作闭环"""
    robot_config = AgileXConfig()
    teleop_config = AgileXTeleoperatorConfig()

    robot = AgileXRobot(robot_config)
    teleop = AgileXTeleoperator(teleop_config)

    robot.connect()
    teleop.connect()

    print("Starting teleoperation... Press Ctrl+C to stop")

    try:
        for i in range(300):  # 10 秒 @ 30Hz
            # 读取主臂
            action = teleop.get_action()

            # 发送到从臂
            robot.send_action(action)

            # 读取观测
            obs = robot.get_observation()

            if i % 30 == 0:
                print(f"Step {i}: left_gripper = {obs['left_gripper.pos']:.4f}")

            time.sleep(1/30)
    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()
        teleop.disconnect()


if __name__ == "__main__":
    test_robot_connection()
    test_teleoperation_loop()
```

---

## 11. 部署与使用

### 11.1 环境准备

```bash
# 1. 创建 conda 环境
conda create -n lerobot_agilex python=3.10
conda activate lerobot_agilex

# 2. 安装 LeRobot
cd /path/to/lerobot
pip install -e ".[dev,test]"

# 3. 安装 ROS 依赖
sudo apt install ros-noetic-cv-bridge ros-noetic-image-transport

# 4. 编译 ROS 工作空间
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

### 11.2 启动脚本

创建 `scripts/launch_agilex.sh`：

```bash
#!/bin/bash
# Agilex + LeRobot 启动脚本

set -e

echo "=== Starting Agilex LeRobot System ==="

# 配置 CAN
echo "[1/4] Configuring CAN..."
cd ~/Piper_ros_private-ros-noetic
./can_config.sh

# 启动 roscore
echo "[2/4] Starting roscore..."
gnome-terminal --tab --title="roscore" -- bash -c "roscore; exec bash"
sleep 2

# 启动 Piper 驱动
echo "[3/4] Starting Piper driver..."
gnome-terminal --tab --title="piper" -- bash -c "
source ~/catkin_ws/devel/setup.bash
roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=true
exec bash"
sleep 3

# 启动相机
echo "[4/4] Starting cameras..."
gnome-terminal --tab --title="camera" -- bash -c "
source ~/catkin_ws/devel/setup.bash
roslaunch astra_camera multi_camera.launch
exec bash"
sleep 3

echo "=== All systems ready ==="
echo "Run: python your_script.py"
```

### 11.3 数据采集示例

```python
# scripts/collect_agilex_data.py

"""
使用 LeRobot 采集 Agilex 双臂数据

Usage:
    python collect_agilex_data.py --repo-id user/agilex_demo --num-episodes 10
"""

import argparse
from pathlib import Path

from lerobot.robots.agilex.config_agilex import AgileXConfig
from lerobot.robots.agilex.agilex import AgileXRobot
from lerobot.teleoperators.agilex.config_agilex_teleop import AgileXTeleoperatorConfig
from lerobot.teleoperators.agilex.agilex_teleoperator import AgileXTeleoperator
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    # 创建配置
    robot_config = AgileXConfig()
    teleop_config = AgileXTeleoperatorConfig()

    # 创建设备
    robot = AgileXRobot(robot_config)
    teleop = AgileXTeleoperator(teleop_config)

    # 创建数据集
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        robot=robot,
    )

    # 连接
    robot.connect()
    teleop.connect()

    try:
        for episode in range(args.num_episodes):
            print(f"\n=== Episode {episode + 1}/{args.num_episodes} ===")
            input("Press Enter to start recording...")

            for step in range(args.max_steps):
                # 1. 读取主臂动作
                action = teleop.get_action()

                # 2. 发送到从臂
                robot.send_action(action)

                # 3. 读取观测
                observation = robot.get_observation()

                # 4. 记录到数据集
                dataset.add_frame(
                    observation=observation,
                    action=action,
                    done=(step == args.max_steps - 1),
                )

            print(f"Episode {episode + 1} recorded ({args.max_steps} frames)")

    finally:
        robot.disconnect()
        teleop.disconnect()

    # 保存数据集
    dataset.consolidate()
    print(f"\nDataset saved to {args.repo_id}")


if __name__ == "__main__":
    main()
```

### 11.4 使用 LeRobot CLI

```bash
# 数据采集
lerobot-record \
    --robot.type=agilex \
    --teleop.type=agilex_teleop \
    --dataset-repo-id=user/agilex_stack_cubes \
    --fps=30 \
    --num-episodes=50

# 训练 ACT 策略
lerobot-train \
    --policy.type=act \
    --dataset-repo-id=user/agilex_stack_cubes \
    --output-dir=outputs/agilex_act

# 评估
lerobot-eval \
    --policy.path=outputs/agilex_act \
    --robot.type=agilex \
    --num-episodes=10
```

---

## 12. 常见问题

### Q1: `rospy.init_node()` 失败

**原因**：ROS Master 未运行

**解决**：
```bash
# 检查 roscore
rosnode list

# 如果失败，启动 roscore
roscore
```

### Q2: 订阅不到 Topic

**原因**：Piper 驱动未启动

**解决**：
```bash
# 检查 Topic
rostopic list | grep puppet

# 启动驱动
roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=true
```

### Q3: 相机无数据

**原因**：相机驱动未启动或 USB 连接问题

**解决**：
```bash
# 检查相机 Topic
rostopic hz /camera_l/color/image_raw

# 启动相机
roslaunch astra_camera multi_camera.launch
```

### Q4: 关节角度全为 0

**原因**：机械臂未使能

**解决**：
```bash
# 确保 auto_enable:=true
# 或手动使能
rosservice call /puppet/enable_arm "data: true"
```

### Q5: Mock 模式下测试

**解决**：
```python
config = AgileXConfig(mock=True, cameras={})
robot = AgileXRobot(config)
robot.connect()  # 不需要 ROS
```

---

## 总结

本文档提供了 Agilex Piper 双臂机器人与 LeRobot 框架的完整集成方案：

| 特性 | 实现状态 |
|------|----------|
| ✅ 7 DOF 双臂（含夹爪） | 14 个关节 |
| ✅ ROS 通信适配 | rospy 直接连接 |
| ✅ Mock 测试支持 | 无需硬件即可测试 |
| ✅ ROS Camera 适配器 | Image Topic → OpenCV |
| ✅ 安全限幅 | max_relative_target |
| ✅ 遥操作 | AgileXTeleoperator |
| ✅ CLI 支持 | lerobot-record/train/eval |

### 关键文件清单

```
src/lerobot/
├── robots/agilex/
│   ├── config_agilex.py        # 配置
│   ├── agilex.py               # Robot 实现
│   └── agilex_ros_bridge.py    # ROS 通信
├── teleoperators/agilex/
│   ├── config_agilex_teleop.py # 遥操作配置
│   └── agilex_teleoperator.py  # Teleoperator 实现
└── cameras/ros_camera/
    ├── configuration_ros_camera.py
    └── ros_camera.py           # ROS 相机适配器
```

---

**文档版本**：v1.0 (Final)
**整合日期**：2024-12-02
**基于**：add_agilex_aug.md, add_agilex_codex.md, add_agilex_cc.md

