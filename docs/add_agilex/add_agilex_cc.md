# Agilex 机器人 LeRobot 集成完整指南

> 本文为 Agilex 双臂遥操作机器人系统提供完整的 LeRobot 集成方案，涵盖 ROS 通信适配、双臂协同控制、相机集成和遥操作闭环采集。

## 目录

1. [Agilex 机器人概述](#agilex-机器人概述)
2. [系统架构与接口分析](#系统架构与接口分析)
3. [集成架构设计](#集成架构设计)
4. [详细实现步骤](#详细实现步骤)
5. [ROS 通信适配层](#ros-通信适配层)
6. [双臂控制实现](#双臂控制实现)
7. [相机系统集成](#相机系统集成)
8. [遥操作支持](#遥操作支持)
9. [完整代码实现](#完整代码实现)
10. [测试方案](#测试方案)
11. [常见问题与调试](#常见问题与调试)
12. [部署与使用示例](#部署与使用示例)

---

## Agilex 机器人概述

### 硬件规格

| 组件 | 规格 | 说明 |
|------|------|------|
| **机械臂** | Piper 机械臂（2主 + 2从） | 6自由度协作臂，CAN总线通信 |
| **主臂（Leader）** | 2x Piper | 人手遥操作，无重力补偿 |
| **从臂（Follower）** | 2x Piper | 执行任务，带力反馈 |
| **相机** | 2x Astra RGB-D 相机 | 左右双目视觉，ROS驱动 |
| **通信接口** | USB-CAN + USB相机 | 通过ROS topics发布数据 |
| **控制系统** | ROS Noetic | Ubuntu 20.04，实时控制 |

### 原版ROS接口

#### Topic列表

```bash
# 主臂状态（遥操作端）
/master/joint_left     # 左手臂关节状态
/master/joint_right    # 右手臂关节状态

# 从臂状态（执行端）
/puppet/joint_left     # 左从臂关节状态
/puppet/joint_right    # 右从臂关节状态
/puppet/arm_status     # 手臂状态反馈
/puppet/end_pose_left  # 左臂末端位姿
/puppet/end_pose_right # 右臂末端位姿

# 相机数据
/camera_l/color/image_raw  # 左相机RGB
/camera_r/color/image_raw  # 右相机RGB
/camera_l/depth/image_raw  # 左相机深度
/camera_r/depth/image_raw  # 右相机深度
```

#### 消息格式

**关节状态消息**（/master/joint_* 和 /puppet/joint_*）：
```python
# 近似结构，根据 piper_ros 包定义
std_msgs/Header header
float64[6] position      # 6个关节位置（弧度）
float64[6] velocity      # 关节速度
float64[6] effort        # 关节力矩
bool enabled             # 使能状态
```

**原版数据采集格式**（collect_data.py输出）：
```python
{
    "master_left": [q1, q2, q3, q4, q5, q6],  # 左主臂6个关节角度
    "master_right": [q1, q2, q3, q4, q5, q6], # 右主臂6个关节角度
    "puppet_left": [q1, q2, q3, q4, q5, q6],  # 左从臂6个关节角度
    "puppet_right": [q1, q2, q3, q4, q5, q6], # 右从臂6个关节角度
    "images": {
        "camera_l": np.ndarray,  # (H, W, 3) RGB图像
        "camera_r": np.ndarray,  # (H, W, 3) RGB图像
    }
}
```

### 原版工作流程

1. **数据采集**：
   ```bash
   # Terminal 1: roscore
   roscore

   # Terminal 2: 启动从臂
   roslaunch piper start_ms_piper.launch mode:=0 auto_enable:=false

   # Terminal 3: 启动相机
   roslaunch astra_camera multi_camera.launch

   # Terminal 4: 采集数据
   python collect_data.py --dataset_dir ./data --task_name Dual_arm_manipulation --max_timesteps 400 --episode_idx 0
   ```

2. **模型推理**：
   ```bash
   # Terminal 2: mode:=1（从臂模式）
   roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=true

   # Terminal 3: 相机
   roslaunch astra_camera multi_camera.launch

   # Terminal 4: 模型推理
   python policy_inference.py
   ```

---

## 系统架构与接口分析

### LeRobot 接口要求

LeRobot 的 `Robot` 抽象基类要求实现：

| 方法/属性 | 说明 |
|-----------|------|
| `observation_features` | 返回观测字典的键名和类型定义 |
| `action_features` | 返回动作字典的键名和类型定义 |
| `connect()` | 连接硬件，执行校准和配置 |
| `disconnect()` | 断开连接，释放资源 |
| `get_observation()` | 读取传感器数据，返回字典 |
| `send_action(action)` | 发送控制指令 |
| `is_connected` | 连接状态 |
| `calibrate()` | 执行校准（可选） |

**LeRobot 标准数据格式**：
```python
# 观测字典
{
    "joint_1.pos": float,      # 关节1位置
    "joint_2.pos": float,      # 关节2位置
    ...
    "gripper.pos": float,      # 抓手位置
    "camera_front": (H,W,3),  # 相机图像
}

# 动作字典（与观测的关节键对应）
{
    "joint_1.pos": float,
    "joint_2.pos": float,
    ...
    "gripper.pos": float,
}
```

### Agilex ↔ LeRobot 接口差异

| 差异点 | Agilex 原版 | LeRobot 要求 | 适配方案 |
|--------|-------------|--------------|----------|
| **通信方式** | ROS topics | 直接串口/SDK | 创建 ROS 包装层 |
| **关节命名** | `puppet_left[0-5]` | `joint_X.pos` | 重命名为标准格式 |
| **双臂结构** | 左右臂分离 | 单臂为主 | 统一编码为12个关节 |
| **相机接口** | ROS image topic | OpenCV直接读取 | 创建 ROS camera 适配器 |
| **数据同步** | ROS时间戳 | LeRobot统一时钟 | 使用 ROS time + 缓存 |
| **主从控制** | 主从分离 | 仅控制从臂 | 读取主臂作为观测，控制从臂 |
| **使能控制** | 通过ROS service | 通过 MotorsBus | 封装 service 调用 |

---

## 集成架构设计

### 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                     LeRobot Framework                        │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────┐  │
│  │          AgileXRobot (Robot 子类)                     │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │  ROS Communication Layer                       │  │  │
│  │  │  - rospy.init_node()                          │  │  │
│  │  │  - Subscriber for /puppet/joint_*             │  │  │
│  │  │  - Subscriber for /master/joint_*             │  │  │
│  │  │  - Publisher for /puppet/joint_commands       │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  │                                                       │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │  ROS Camera Adapter                            │  │  │
│  │  │  - Subscribe /camera_l/color/image_raw       │  │  │
│  │  │  - Subscribe /camera_r/color/image_raw       │  │  │
│  │  │  - Bridge ROS Image → OpenCV                │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────┐  │
│  │      AgileXLeader (Teleoperator 子类)               │  │
│  │  - Subscribe /master/joint_left/right              │  │
│  │  - get_action() → 主臂关节角度                    │  │
│  └──────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    ROS Middleware                           │
│  - roscore                                               │
│  - piper_ros driver                                      │
│  - astra_camera driver                                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    ┌────────────────┐
                    │  Hardware      │
                    │  - 2x Piper    │
                    │  - 2x Astra    │
                    └────────────────┘
```

### 数据流设计

#### 数据采集流程（训练）

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐
│             │    │              │    │              │    │             │
│  Human      │───▶│  AgileXLeader│───▶│  Policy      │───▶│ AgileXRobot │
│  Operator   │    │  (读取主臂)  │    │  Network     │    │(控制从臂)   │
│             │    │              │    │              │    │             │
└─────────────┘    └──────────────┘    └──────────────┘    └─────────────┘
                                                            │
                                                            │
                                                            ▼
                                            ┌────────────────────────────┐
                                            │ get_observation()          │
                                            │  - puppet_left[0-5]        │
                                            │  - puppet_right[0-5]       │
                                            │  - master_left[0-5]        │
                                            │  - master_right[0-5]       │
                                            │  - camera_l                │
                                            │  - camera_r                │
                                            └────────────────────────────┘
```

**数据格式转换**：
```python
# ROS消息 → LeRobot观测
ros_joint_left = [0.1, -0.2, 0.3, 0.4, -0.5, 0.6]  # 弧度

observation = {
    "puppet_left_0.pos": 0.1,
    "puppet_left_1.pos": -0.2,
    "puppet_left_2.pos": 0.3,
    "puppet_left_3.pos": 0.4,
    "puppet_left_4.pos": -0.5,
    "puppet_left_5.pos": 0.6,
    # ... 右从臂
    # ... 主臂（用于模仿学习）
    "camera_l": np.ndarray,  # (H, W, 3)
    "camera_r": np.ndarray,  # (H, W, 3)
}
```

#### 控制流程（推理）

```
┌────────────────────────┐
│   Policy Network       │
│   → action dict        │
└────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────┐
│  send_action(action)                    │
│  │                                      │
│  ├─→ 解析action:                        │
│  │    {'puppet_left_0.pos': 0.1, ...}   │
│  │                                      │
│  ├─→ 转换为ROS消息:                     │
│  │    std_msgs/Float64MultiArray       │
│  │    data = [0.1, -0.2, ...]          │
│  │                                      │
│  └─→ pub.publish(msg)                  │
└─────────────────────────────────────────┘
            │
            ▼
    /puppet/joint_left (topic)
            │
            ▼
    Piper_ros driver
            │
            ▼
       机械臂执行
```

### 双臂关节命名策略

采用统一命名空间，区分左右臂：

| 关节索引 | LeRobot键名 | ROS Topic | 说明 |
|----------|-------------|-----------|------|
| 左臂0 | `puppet_left_0.pos` | /puppet/joint_left[0] | 左肩偏航 |
| 左臂1 | `puppet_left_1.pos` | /puppet/joint_left[1] | 左肩俯仰 |
| 左臂2 | `puppet_left_2.pos` | /puppet/joint_left[2] | 左肩翻滚 |
| 左臂3 | `puppet_left_3.pos` | /puppet/joint_left[3] | 左肘 |
| 左臂4 | `puppet_left_4.pos` | /puppet/joint_left[4] | 左腕1 |
| 左臂5 | `puppet_left_5.pos` | /puppet/joint_left[5] | 左腕2 |
| 右臂0 | `puppet_right_0.pos` | /puppet/joint_right[0] | 右肩偏航 |
| 右臂1 | `puppet_right_1.pos` | /puppet/joint_right[1] | 右肩俯仰 |
| 右臂2 | `puppet_right_2.pos` | /puppet/joint_right[2] | 右肩翻滚 |
| 右臂3 | `puppet_right_3.pos` | /puppet/joint_right[3] | 右肘 |
| 右臂4 | `puppet_right_4.pos` | /puppet/joint_right[4] | 右腕1 |
| 右臂5 | `puppet_right_5.pos` | /puppet/joint_right[5] | 右腕2 |

**观测特征定义**：
```python
observation_features = {
    # 从臂关节（12个）
    "puppet_left_0.pos": float,
    "puppet_left_1.pos": float,
    ...
    "puppet_right_5.pos": float,

    # 主臂关节（用于模仿学习，12个）
    "master_left_0.pos": float,
    ...
    "master_right_5.pos": float,

    # 相机
    "camera_l": (480, 640, 3),
    "camera_r": (480, 640, 3),
}
```

**动作特征定义**（仅控制从臂）：
```python
action_features = {
    "puppet_left_0.pos": float,
    "puppet_left_1.pos": float,
    ...
    "puppet_right_5.pos": float,
}
```

---

## 详细实现步骤

### 步骤1：创建目录结构

```
src/lerobot/robots/agilex/
├── __init__.py
├── config_agilex.py
└── agilex.py

src/lerobot/teleoperators/agilex_leader/
├── __init__.py
├── config_agilex_leader.py
└── agilex_leader.py

src/lerobot/cameras/ros_camera/
├── __init__.py
├── ros_camera.py
└── configuration_ros_camera.py
```

### 步骤2：实现 ROS 相机适配器

由于 LeRobot 的相机系统期望直接访问硬件，我们需要创建 ROS 相机适配器：

```python
# src/lerobot/cameras/ros_camera/ros_camera.py

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
from typing import Any

from lerobot.cameras.camera import Camera
from lerobot.cameras.utils import Camera, CameraConfig
from .configuration_ros_camera import RosCameraConfig


class RosCamera(Camera):
    """ROS相机适配器，将ROS Image topic转换为OpenCV图像"""

    def __init__(self, config: RosCameraConfig):
        """
        Args:
            config: RosCameraConfig，包含topic_name等配置
        """
        super().__init__(config)
        self.config = config
        self._cv_bridge = CvBridge()

        # ROS订阅器
        self._sub = None
        self._latest_image = None
        self._image_timestamp = None

    @property
    def is_connected(self) -> bool:
        """检查是否成功订阅topic"""
        return self._sub is not None

    def connect(self) -> None:
        """订阅ROS图像topic"""
        if self.is_connected:
            raise RuntimeError(f"Camera {self.config.topic_name} already connected")

        rospy.loginfo(f"Subscribing to {self.config.topic_name}")
        self._sub = rospy.Subscriber(
            self.config.topic_name,
            Image,
            self._image_callback,
            queue_size=1
        )

        # 等待第一帧图像
        rospy.loginfo("Waiting for first image...")
        start = rospy.Time.now()
        while self._latest_image is None:
            if (rospy.Time.now() - start).to_sec() > 10:
                raise TimeoutError(f"No image received from {self.config.topic_name}")
            rospy.sleep(0.01)

        rospy.loginfo(f"Camera {self.config.topic_name} connected")

    def disconnect(self) -> None:
        """取消订阅"""
        if self._sub:
            self._sub.unregister()
            self._sub = None
        self._latest_image = None

    def read(self) -> np.ndarray | None:
        """同步读取最新图像"""
        if not self.is_connected:
            raise RuntimeError("Camera not connected")
        return self._latest_image

    def async_read(self) -> np.ndarray | None:
        """异步读取（非阻塞），ROS相机本质上都是异步的"""
        return self.read()

    def _image_callback(self, msg: Image) -> None:
        """ROS回调函数，接收图像消息"""
        try:
            # 转换为OpenCV格式
            cv_image = self._cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            # 转换为RGB
            self._latest_image = cv_image[:, :, ::-1]
            self._image_timestamp = msg.header.stamp
        except Exception as e:
            rospy.logerr(f"Error converting image: {e}")

    @staticmethod
    def find_cameras() -> list[str]:
        """列出可用的ROS相机topic"""
        try:
            # 检查ROS master是否运行
            rosgraph.Master('/rostopic').getSystemState()
        except:
            return []

        # 获取所有topic
        topics = rospy.get_published_topics()
        camera_topics = []

        for topic, topic_type in topics:
            if topic_type == "sensor_msgs/Image" and ("camera" in topic or "image" in topic):
                camera_topics.append(topic)

        return camera_topics
```

#### 配置文件

```python
# src/lerobot/cameras/ros_camera/configuration_ros_camera.py

from dataclasses import dataclass
from lerobot.cameras.configs import CameraConfig


@CameraConfig.register_subclass("ros_camera")
@dataclass
class RosCameraConfig(CameraConfig):
    """ROS相机配置"""

    topic_name: str = "/camera/color/image_raw"  # ROS图像topic名称
    """
    其他字段继承自 CameraConfig:
    - fps: int = 30
    - width: int = 640
    - height: int = 480
    """

    def __post_init__(self):
        """校验配置"""
        super().__post_init__()
        if not self.topic_name.startswith("/"):
            raise ValueError("topic_name must start with '/'")
```

### 步骤3：定义机器人配置类

```python
# src/lerobot/robots/agilex/config_agilex.py

from dataclasses import dataclass, field
from lerobot.robots.config import RobotConfig
from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.ros_camera.configuration_ros_camera import RosCameraConfig


@RobotConfig.register_subclass("agilex")
@dataclass
class AgilexConfig(RobotConfig):
    """Agilex双臂机器人配置"""

    # ROS配置
    ros_master_uri: str = "http://localhost:11311"  # ROS master地址
    node_name: str = "lerobot_agilex"             # ROS节点名称

    # 关节topic配置
    puppet_left_topic: str = "/puppet/joint_left"   # 左从臂状态topic
    puppet_right_topic: str = "/puppet/joint_right" # 右从臂状态topic
    master_left_topic: str = "/master/joint_left"   # 左主臂状态topic
    master_right_topic: str = "/master/joint_right" # 右主臂状态topic

    # 命令topic配置
    puppet_left_cmd_topic: str = "/puppet/joint_left/command"   # 左从臂命令topic
    puppet_right_cmd_topic: str = "/puppet/joint_right/command" # 右从臂命令topic

    # 相机配置
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "camera_l": RosCameraConfig(
                topic_name="/camera_l/color/image_raw",
                fps=30,
                width=640,
                height=480,
            ),
            "camera_r": RosCameraConfig(
                topic_name="/camera_r/color/image_raw",
                fps=30,
                width=640,
                height=480,
            )
        }
    )

    # 安全参数
    max_joint_delta: float = 0.2  # 单步最大关节变化（弧度）
    enable_master_observation: bool = True  # 是否在观测中包含主臂状态
    enable_visualization: bool = False      # 是否启用可视化（RVIZ）

    def __post_init__(self):
        """校验配置"""
        super().__post_init__()
        # 相机必须配置宽高
        for cam_name, cam_config in self.cameras.items():
            if cam_config.width is None or cam_config.height is None:
                raise ValueError(f"Camera {cam_name} must have width and height")
```

### 步骤4：实现 AgileXRobot 类

```python
# src/lerobot/robots/agilex/agilex.py

import rospy
import numpy as np
from typing import Any, Dict
from functools import cached_property

from lerobot.robots.robot import Robot
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError
from lerobot.robots.utils import ensure_safe_goal_position

from std_msgs.msg import Float64MultiArray, Bool
from sensor_msgs.msg import JointState

from .config_agilex import AgilexConfig


class AgileXRobot(Robot):
    """Agilex双臂机器人ROS接口"""

    config_class = AgilexConfig
    name = "agilex"

    def __init__(self, config: AgilexConfig):
        super().__init__(config)
        self.config = config

        # ROS节点
        self._ros_node_initialized = False

        # 订阅器
        self._sub_puppet_left = None
        self._sub_puppet_right = None
        self._sub_master_left = None
        self._sub_master_right = None

        # 发布器
        self._pub_puppet_left = None
        self._pub_puppet_right = None

        # 最新数据缓存
        self._puppet_left_state = None
        self._puppet_right_state = None
        self._master_left_state = None
        self._master_right_state = None

        # 相机（使用ROS相机适配器）
        self.cameras = make_cameras_from_configs(config.cameras)

    # ===== 特征声明 =====

    @cached_property
    def observation_features(self) -> Dict[str, type | tuple]:
        """观测特征定义"""
        features = {}

        # 从臂关节（12个）
        for side in ["left", "right"]:
            for i in range(6):
                features[f"puppet_{side}_{i}.pos"] = float

        # 主臂关节（12个，可选）
        if self.config.enable_master_observation:
            for side in ["left", "right"]:
                for i in range(6):
                    features[f"master_{side}_{i}.pos"] = float

        # 相机
        for cam_name in self.cameras:
            cam_cfg = self.config.cameras[cam_name]
            features[cam_name] = (cam_cfg.height, cam_cfg.width, 3)

        return features

    @cached_property
    def action_features(self) -> Dict[str, type]:
        """动作特征定义（仅控制从臂）"""
        features = {}
        for side in ["left", "right"]:
            for i in range(6):
                features[f"puppet_{side}_{i}.pos"] = float
        return features

    # ===== 连接管理 =====

    @property
    def is_connected(self) -> bool:
        """检查ROS连接"""
        return self._ros_node_initialized and self._sub_puppet_left is not None

    def connect(self, calibrate: bool = True) -> None:
        """连接ROS节点并订阅topic"""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        try:
            # 初始化ROS节点
            rospy.init_node(self.config.node_name, anonymous=True, log_level=rospy.INFO)
            self._ros_node_initialized = True
        except rospy.exceptions.ROSException as e:
            # 节点已存在，可能是另一个实例
            rospy.logwarn(f"ROS node initialization warning: {e}")
            self._ros_node_initialized = True

        # 订阅从臂状态
        rospy.loginfo("Subscribing to puppet topics...")
        self._sub_puppet_left = rospy.Subscriber(
            self.config.puppet_left_topic,
            JointState,
            self._puppet_left_callback,
            queue_size=10
        )
        self._sub_puppet_right = rospy.Subscriber(
            self.config.puppet_right_topic,
            JointState,
            self._puppet_right_callback,
            queue_size=10
        )

        # 订阅主臂状态（用于模仿学习）
        if self.config.enable_master_observation:
            rospy.loginfo("Subscribing to master topics...")
            self._sub_master_left = rospy.Subscriber(
                self.config.master_left_topic,
                JointState,
                self._master_left_callback,
                queue_size=10
            )
            self._sub_master_right = rospy.Subscriber(
                self.config.master_right_topic,
                JointState,
                self._master_right_callback,
                queue_size=10
            )

        # 创建命令发布器
        rospy.loginfo("Creating command publishers...")
        self._pub_puppet_left = rospy.Publisher(
            self.config.puppet_left_cmd_topic,
            Float64MultiArray,
            queue_size=10
        )
        self._pub_puppet_right = rospy.Publisher(
            self.config.puppet_right_cmd_topic,
            Float64MultiArray,
            queue_size=10
        )

        # 等待第一帧数据
        rospy.loginfo("Waiting for puppet data...")
        self._wait_for_data()

        # 连接相机
        for cam in self.cameras.values():
            cam.connect()

        rospy.loginfo(f"{self} connected successfully")

    def disconnect(self) -> None:
        """断开ROS连接"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        # 断开相机
        for cam in self.cameras.values():
            cam.disconnect()

        # 取消订阅和发布
        if self._sub_puppet_left:
            self._sub_puppet_left.unregister()
        if self._sub_puppet_right:
            self._sub_puppet_right.unregister()
        if self._sub_master_left:
            self._sub_master_left.unregister()
        if self._sub_master_right:
            self._sub_master_right.unregister()

        self._ros_node_initialized = False
        rospy.loginfo(f"{self} disconnected")

    def _wait_for_data(self, timeout: float = 10.0) -> None:
        """等待接收到第一帧数据"""
        import time
        start_time = time.time()
        while (self._puppet_left_state is None or self._puppet_right_state is None):
            if time.time() - start_time > timeout:
                raise TimeoutError("Timeout waiting for puppet data")
            rospy.sleep(0.01)

    # ===== ROS回调函数 =====

    def _puppet_left_callback(self, msg: JointState) -> None:
        """接收左从臂状态"""
        self._puppet_left_state = np.array(msg.position[:6], dtype=np.float32)

    def _puppet_right_callback(self, msg: JointState) -> None:
        """接收右从臂状态"""
        self._puppet_right_state = np.array(msg.position[:6], dtype=np.float32)

    def _master_left_callback(self, msg: JointState) -> None:
        """接收左主臂状态"""
        self._master_left_state = np.array(msg.position[:6], dtype=np.float32)

    def _master_right_callback(self, msg: JointState) -> None:
        """接收右主臂状态"""
        self._master_right_state = np.array(msg.position[:6], dtype=np.float32)

    # ===== 核心 I/O =====

    def get_observation(self) -> Dict[str, Any]:
        """获取观测数据"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        observation = {}

        # 从臂关节（必须）
        if self._puppet_left_state is not None:
            for i, pos in enumerate(self._puppet_left_state):
                observation[f"puppet_left_{i}.pos"] = float(pos)

        if self._puppet_right_state is not None:
            for i, pos in enumerate(self._puppet_right_state):
                observation[f"puppet_right_{i}.pos"] = float(pos)

        # 主臂关节（可选）
        if self.config.enable_master_observation:
            if self._master_left_state is not None:
                for i, pos in enumerate(self._master_left_state):
                    observation[f"master_left_{i}.pos"] = float(pos)
            if self._master_right_state is not None:
                for i, pos in enumerate(self._master_right_state):
                    observation[f"master_right_{i}.pos"] = float(pos)

        # 相机图像
        for cam_name, cam in self.cameras.items():
            observation[cam_name] = cam.async_read()

        return observation

    def send_action(self, action: Dict[str, Any]) -> Dict[str, float]:
        """发送控制指令"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        # 解析动作字典
        # 格式：{'puppet_left_0.pos': 0.1, 'puppet_left_1.pos': -0.2, ...}
        left_joints = []
        right_joints = []

        for i in range(6):
            left_key = f"puppet_left_{i}.pos"
            right_key = f"puppet_right_{i}.pos"

            if left_key in action:
                left_joints.append(float(action[left_key]))
            else:
                raise KeyError(f"Missing joint in action: {left_key}")

            if right_key in action:
                right_joints.append(float(action[right_key]))
            else:
                raise KeyError(f"Missing joint in action: {right_key}")

        # 安全限幅（可选）
        if self.config.max_joint_delta > 0:
            # 读取当前位置
            curr_left = self._puppet_left_state if self._puppet_left_state is not None else np.zeros(6)
            curr_right = self._puppet_right_state if self._puppet_right_state is not None else np.zeros(6)

            # 限幅
            left_joints = self._clip_joint_delta(curr_left, left_joints, self.config.max_joint_delta)
            right_joints = self._clip_joint_delta(curr_right, right_joints, self.config.max_joint_delta)

        # 发布到ROS
        self._publish_joints(self._pub_puppet_left, left_joints)
        self._publish_joints(self._pub_puppet_right, right_joints)

        # 返回实际发送的动作
        result = {}
        for i, pos in enumerate(left_joints):
            result[f"puppet_left_{i}.pos"] = pos
        for i, pos in enumerate(right_joints):
            result[f"puppet_right_{i}.pos"] = pos

        return result

    def _clip_joint_delta(self, current: np.ndarray, target: list, max_delta: float) -> list:
        """限制关节变化量"""
        current = np.array(current)
        target = np.array(target)
        delta = target - current
        delta_clipped = np.clip(delta, -max_delta, max_delta)
        return (current + delta_clipped).tolist()

    def _publish_joints(self, publisher: rospy.Publisher, joints: list) -> None:
        """发布关节指令"""
        msg = Float64MultiArray()
        msg.data = joints
        publisher.publish(msg)

    # ===== 校准（Agilex通常不需要，使用内置校准）=====

    @property
    def is_calibrated(self) -> bool:
        """Agilex使用ROS驱动内置校准"""
        return True

    def calibrate(self) -> None:
        """Agilex校准通过ROS服务或外部工具完成"""
        rospy.loginfo("Agilex calibration is done via ROS services or external tools")
        rospy.loginfo("Make sure the arms are properly enabled before use")
```

### 步骤5：实现 AgileXLeader（遥操作）

```python
# src/lerobot/teleoperators/agilex_leader/agilex_leader.py

import rospy
import numpy as np
from functools import cached_property
from typing import Dict, Any

from lerobot.teleoperators.teleoperator import Teleoperator
from .config_agilex_leader import AgilexLeaderConfig

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


class AgileXLeader(Teleoperator):
    """Agilex主臂遥操作接口"""

    config_class = AgilexLeaderConfig
    name = "agilex_leader"

    def __init__(self, config: AgilexLeaderConfig):
        super().__init__(config)
        self.config = config

        # ROS节点
        self._ros_node_initialized = False

        # 订阅器
        self._sub_master_left = None
        self._sub_master_right = None

        # 数据缓存
        self._master_left_state = None
        self._master_right_state = None

    # ===== 特征声明 =====

    @cached_property
    def action_features(self) -> Dict[str, type]:
        """遥操作输出 = 主臂关节位置"""
        features = {}
        for side in ["left", "right"]:
            for i in range(6):
                features[f"master_{side}_{i}.pos"] = float
        return features

    @cached_property
    def feedback_features(self) -> Dict[str, type]:
        """遥操作反馈（可选）"""
        return {}

    # ===== 连接管理 =====

    @property
    def is_connected(self) -> bool:
        return self._ros_node_initialized and self._sub_master_left is not None

    def connect(self) -> None:
        """连接主臂"""
        if self.is_connected:
            raise RuntimeError(f"{self} already connected")

        # 初始化ROS节点
        rospy.init_node(self.config.node_name, anonymous=True)
        self._ros_node_initialized = True

        # 订阅主臂状态
        rospy.loginfo("Subscribing to master topics...")
        self._sub_master_left = rospy.Subscriber(
            self.config.master_left_topic,
            JointState,
            self._master_left_callback,
            queue_size=10
        )
        self._sub_master_right = rospy.Subscriber(
            self.config.master_right_topic,
            JointState,
            self._master_right_callback,
            queue_size=10
        )

        # 等待数据
        self._wait_for_data()

        rospy.loginfo(f"{self} connected")

    def disconnect(self) -> None:
        """断开连接"""
        if not self.is_connected:
            raise RuntimeError(f"{self} not connected")

        if self._sub_master_left:
            self._sub_master_left.unregister()
        if self._sub_master_right:
            self._sub_master_right.unregister()

        self._ros_node_initialized = False

    def _wait_for_data(self, timeout: float = 10.0) -> None:
        """等待接收到第一帧数据"""
        import time
        start_time = time.time()
        while (self._master_left_state is None or self._master_right_state is None):
            if time.time() - start_time > timeout:
                raise TimeoutError("Timeout waiting for master data")
            rospy.sleep(0.01)

    # ===== ROS回调 =====

    def _master_left_callback(self, msg: JointState) -> None:
        self._master_left_state = np.array(msg.position[:6], dtype=np.float32)

    def _master_right_callback(self, msg: JointState) -> None:
        self._master_right_state = np.array(msg.position[:6], dtype=np.float32)

    # ===== 核心方法 =====

    def get_action(self) -> Dict[str, float]:
        """读取主臂关节位置作为动作"""
        if not self.is_connected:
            raise RuntimeError(f"{self} not connected")

        action = {}

        if self._master_left_state is not None:
            for i, pos in enumerate(self._master_left_state):
                action[f"master_left_{i}.pos"] = float(pos)

        if self._master_right_state is not None:
            for i, pos in enumerate(self._master_right_state):
                action[f"master_right_{i}.pos"] = float(pos)

        return action

    def send_feedback(self, feedback: Dict[str, Any]) -> None:
        """发送反馈（Agilex暂无反馈机制）"""
        pass
```

#### Leader配置

```python
# src/lerobot/teleoperators/agilex_leader/config_agilex_leader.py

from dataclasses import dataclass
from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("agilex_leader")
@dataclass
class AgilexLeaderConfig(TeleoperatorConfig):
    """Agilex主臂配置"""

    ros_master_uri: str = "http://localhost:11311"
    node_name: str = "lerobot_agilex_leader"

    # 主臂状态topic
    master_left_topic: str = "/master/joint_left"
    master_right_topic: str = "/master/joint_right"
```

### 步骤6：注册到LeRobot工厂

```python
# src/lerobot/robots/utils.py

def make_robot_from_config(config: RobotConfig) -> Robot:
    """创建机器人实例"""
    if config.type == "so100":
        from lerobot.robots.so100_follower import SO100Follower
        return SO100Follower(config)

    elif config.type == "agilex":
        from lerobot.robots.agilex import AgileXRobot
        return AgileXRobot(config)

    # ... 其他机器人


# src/lerobot/teleoperators/utils.py

def make_teleoperator_from_config(config: TeleoperatorConfig) -> Teleoperator:
    """创建遥操作设备实例"""
    if config.type == "so101_leader":
        from lerobot.teleoperators.so101_leader import SO101Leader
        return SO101Leader(config)

    elif config.type == "agilex_leader":
        from lerobot.teleoperators.agilex_leader import AgileXLeader
        return AgileXLeader(config)

    # ... 其他遥操作设备
```

---

## 测试方案

### 单元测试（Mock ROS）

```python
# tests/robots/test_agilex.py

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from lerobot.robots.agilex import AgileXRobot, AgilexConfig


@pytest.fixture
def agilex_robot():
    """创建带mock的Agilex机器人"""
    config = AgilexConfig(
        ros_master_uri="http://localhost:11311",
        puppet_left_topic="/puppet/joint_left",
        puppet_right_topic="/puppet/joint_right",
    )

    # Mock rospy
    with patch("lerobot.robots.agilex.agilex.rospy") as mock_rospy:
        mock_rospy.init_node = MagicMock()
        mock_rospy.Subscriber = MagicMock()
        mock_rospy.Publisher = MagicMock()
        mock_rospy.loginfo = MagicMock()
        mock_rospy.sleep = MagicMock()
        mock_rospy.Time = MagicMock()
        mock_rospy.Time.now = MagicMock(return_value=MagicMock(to_sec=lambda: 0.0))

        robot = AgileXRobot(config)
        yield robot


def test_agilex_observation_features(agilex_robot):
    """测试观测特征"""
    robot = agilex_robot
    features = robot.observation_features

    # 检查从臂关节
    for i in range(6):
        assert f"puppet_left_{i}.pos" in features
        assert f"puppet_right_{i}.pos" in features

    # 检查主臂关节
    for i in range(6):
        assert f"master_left_{i}.pos" in features
        assert f"master_right_{i}.pos" in features


def test_agilex_action_features(agilex_robot):
    """测试动作特征"""
    robot = agilex_robot
    features = robot.action_features

    # 只控制从臂
    for i in range(6):
        assert f"puppet_left_{i}.pos" in features
        assert f"puppet_right_{i}.pos" in features

    assert len(features) == 12


def test_joint_naming_consistency():
    """测试主从臂关节命名一致性"""
    from lerobot.teleoperators.agilex_leader import AgileXLeader
    from lerobot.robots.agilex import AgileXRobot

    # 确保Leader输出与Robot输入匹配
    leader_features = AgileXLeader.action_features
    robot_features = AgileXRobot.action_features

    # 遥操作时，Leader读取主臂，Robot控制从臂
    # 需要通过策略网络进行映射
    assert len(leader_features) == len(robot_features)
```

### 硬件测试

```python
# tests/manual_test_agilex.py

"""手动测试Agilex机器人"""

import rospy
import time
from lerobot.robots.agilex import AgileXRobot, AgilexConfig

def test_agilex_connection():
    """测试连接和基本功能"""
    print("Testing AgileX robot connection...")

    config = AgilexConfig()
    robot = AgileXRobot(config)

    # 连接
    robot.connect()
    print(f"Connected: {robot.is_connected}")

    # 读取观测
    obs = robot.get_observation()
    print(f"Observation keys: {list(obs.keys())}")
    print(f"Puppet left joints: {[obs[f'puppet_left_{i}.pos'] for i in range(6)]}")

    time.sleep(1)

    # 发送零动作
    action = {}
    for side in ["left", "right"]:
        for i in range(6):
            action[f"puppet_{side}_{i}.pos"] = 0.0

    robot.send_action(action)
    print("Zero action sent")

    robot.disconnect()
    print("Disconnected")

if __name__ == "__main__":
    test_agilex_connection()
```

运行测试：
```bash
# Terminal 1: roscore
roscore

# Terminal 2: 启动机器人
roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=true

# Terminal 3: 启动相机
roslaunch astra_camera multi_camera.launch

# Terminal 4: 运行测试
python tests/manual_test_agilex.py
```

### 闭环测试（遥操作 + 数据采集）

```python
# tests/test_agilex_teleop.py

"""测试Agilex遥操作闭环"""

import rospy
from lerobot.robots.agilex import AgileXRobot, AgilexConfig
from lerobot.teleoperators.agilex_leader import AgileXLeader, AgilexLeaderConfig

def test_teleoperation():
    """测试遥操作流程"""
    rospy.loginfo("Starting teleoperation test...")

    # 创建Leader
    leader_config = AgilexLeaderConfig()
    leader = AgileXLeader(leader_config)
    leader.connect()

    # 创建Robot
    robot_config = AgilexConfig()
    robot = AgileXRobot(robot_config)
    robot.connect()

    rospy.loginfo("Starting teleop loop... Press Ctrl+C to stop")
    rate = rospy.Rate(30)  # 30Hz

    try:
        while not rospy.is_shutdown():
            # 1. 读取主臂动作
            action = leader.get_action()

            # 2. 发送到从臂（这里直接映射，实际应通过策略网络）
            robot.send_action(action)

            # 3. 读取观测
            obs = robot.get_observation()

            rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()
        leader.disconnect()
        rospy.loginfo("Teleoperation stopped")

if __name__ == "__main__":
    test_teleoperation()
```

---

## 常见问题与调试

### 问题排查表

| 问题现象 | 可能原因 | 排查步骤 |
|---------|---------|---------|
| `rospy.init_node()` 失败 | ROS master未运行 | 1. 检查 `roscore` 是否启动<br>2. 检查 `ROS_MASTER_URI` 环境变量 |
| 订阅不到topic | 机器人驱动未启动 | 1. 检查 `roslaunch piper ...` 是否运行<br>2. 用 `rostopic list` 查看可用topic |
| 相机无数据 | astra_camera未启动 | 1. 运行 `roslaunch astra_camera ...`<br>2. 检查 `rostopic echo /camera_l/color/image_raw` |
| 关节角度全为0 | 未使能机械臂 | 1. 检查 `auto_enable:=true` 参数<br>2. 手动调用 enable service |
| 动作无响应 | 发布到错误的topic | 1. 检查 `puppet_left_cmd_topic` 配置<br>2. 确认topic名称与ROS驱动匹配 |
| 双臂运动方向相反 | 关节方向定义不一致 | 1. 检查ROS驱动中的关节方向<br>2. 在 `send_action()` 中反转特定关节 |
| 主从延迟高 | ROS网络延迟或负载高 | 1. 检查 `rostopic hz /master/joint_left`<br>2. 关闭不必要的ROS节点 |
| `CvBridge` 导入错误 | cv_bridge未安装 | `sudo apt install ros-noetic-cv-bridge` |

### ROS调试命令

```bash
# 查看所有topic
rostopic list

# 查看topic数据
rostopic echo /puppet/joint_left

# 查看topic发布频率
rostopic hz /master/joint_right

# 查找Image topic
rostopic list | grep image_raw

# 查看ROS master状态
rosnode list

# 查看节点信息
rosnode info /lerobot_agilex

# 查看参数服务器
rosparam list

# 图形化查看
rosrun rqt_graph rqt_graph
```

### 性能优化

**问题**：观测读取延迟高（>50ms）

**解决方案**：
1. **增加缓存**：在回调函数中缓存数据，避免ROS阻塞
```python
def _puppet_left_callback(self, msg: JointState) -> None:
    self._puppet_left_buffer.append(msg)
    if len(self._puppet_left_buffer) > 10:
        self._puppet_left_buffer.pop(0)

# 在 get_observation() 中使用最新缓存
def get_observation(self):
    if self._puppet_left_buffer:
        latest = self._puppet_left_buffer[-1]
        self._puppet_left_state = np.array(latest.position[:6])
```

2. **多线程ROS**：使用 `rospy.async` 或 `multiprocessing`

3. **降低ROS开销**：
   - 减少topic数量
   - 使用 `rospy.TransportHints().reliable().tcp_nodelay()`

### 安全机制

1. **关节限幅**：
```python
def send_action(self, action):
    # 限幅到物理范围
    MIN_JOINT = -3.14
    MAX_JOINT = 3.14

    left_joints = [np.clip(action[f"puppet_left_{i}.pos"], MIN_JOINT, MAX_JOINT)
                   for i in range(6)]
    right_joints = [np.clip(action[f"puppet_right_{i}.pos"], MIN_JOINT, MAX_JOINT)
                    for i in range(6)]
    ...
```

2. **速度限制**：
```python
# 在配置中设置 max_joint_velocity
# 或使用 ensure_safe_goal_position
def send_action(self, action):
    current = self._puppet_left_state
    target = [action[f"puppet_left_{i}.pos"] for i in range(6)]

    # 计算最大允许变化
    max_delta = self.config.max_velocity * dt
    clipped = ensure_safe_goal_position(
        {f"joint_{i}": (target[i], current[i]) for i in range(6)},
        max_delta
    )
    ...
```

3. **急停机制**：
```python
# 监听 /estop topic
self._sub_estop = rospy.Subscriber("/estop", Bool, self._estop_callback)

def _estop_callback(self, msg: Bool):
    if msg.data:  # 急停触发
        rospy.logerr("Emergency stop triggered!")
        self._is_estopped = True

def send_action(self, action):
    if self._is_estopped:
        rospy.logwarn("Cannot send action: estopped")
        return
    ...
```

---

## 部署与使用示例

### 1. 环境准备

```bash
# 创建conda环境
conda create -n agilex_lerobot python=3.10
conda activate agilex_lerobot

# 安装LeRobot（开发模式）
pip install -e ".[dev,test]"

# 安装ROS Noetic（如果未安装）
# 参考：https://wiki.ros.org/noetic/Installation

# 安装ROS依赖
sudo apt install ros-noetic-cv-bridge ros-noetic-image-transport

# 编译工作空间
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

### 2. 启动机器人系统

创建启动脚本 `launch_agilex.sh`：

```bash
#!/bin/bash

# 启动脚本 for Agilex + LeRobot integration

# Terminal 1: roscore
xterm -e "roscore" &
ROS_PID=$!
sleep 2

# Terminal 2: 启动Piper机械臂驱动
xterm -e "
cd ~/Piper_ros_private-ros-noetic
source devel/setup.bash
./can_config.sh
roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=true
" &
PIPER_PID=$!
sleep 3

# Terminal 3: 启动相机
xterm -e "roslaunch astra_camera multi_camera.launch" &
CAMERA_PID=$!
sleep 3

# Terminal 4: LeRobot测试
echo "Starting LeRobot test..."
echo "Run: python test_agilex.py"

# 清理函数
cleanup() {
    echo "\nShutting down..."
    kill $CAMERA_PID 2>/dev/null
    kill $PIPER_PID 2>/dev/null
    kill $ROS_PID 2>/dev/null
}
trap cleanup EXIT

wait
```

赋予执行权限：
```bash
chmod +x launch_agilex.sh
./launch_agilex.sh
```

### 3. 使用 YAML 配置

```yaml
# config/agilex_dual_arm.yaml

robot:
  type: agilex
  ros_master_uri: "http://localhost:11311"
  node_name: "lerobot_agilex"

  enable_master_observation: true
  max_joint_delta: 0.2

  cameras:
    camera_l:
      type: ros_camera
      topic_name: "/camera_l/color/image_raw"
      fps: 30
      width: 640
      height: 480

    camera_r:
      type: ros_camera
      topic_name: "/camera_r/color/image_raw"
      fps: 30
      width: 640
      height: 480

teleoperator:
  type: agilex_leader
  ros_master_uri: "http://localhost:11311"
  node_name: "lerobot_agilex_leader"

policy:
  type: act
  input_dim: 24    # 主臂12 + 从臂12（可选）
  output_dim: 12   # 从臂12个关节
  # ...
```

### 4. 训练脚本

```python
# train_agilex.py

from lerobot.robots.agilex import AgileXRobot, AgilexConfig
from lerobot.teleoperators.agilex_leader import AgileXLeader, AgilexLeaderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 1. 创建数据集
dataset = LeRobotDataset.create(
    repo_id="agilex/dual_arm_stack_cubes",
    fps=30,
    robot=AgilexConfig(),
)

# 2. 创建机器人和遥操作设备
robot = AgileXRobot(AgilexConfig())
leader = AgileXLeader(AgilexLeaderConfig())

# 3. 连接
robot.connect()
leader.connect()

# 4. 数据采集循环
try:
    for episode in range(100):
        timestep_data = []

        for step in range(400):  # 400步
            # 读取主臂动作
            action = leader.get_action()

            # 发送到机器人
            robot.send_action(action)

            # 读取观测
            observation = robot.get_observation()

            # 记录数据
            dataset.add_frame(observation, action, done=False)

finally:
    robot.disconnect()
    leader.disconnect()

# 5. 保存数据集
dataset.consolidate()
```

### 5. 推理脚本

```python
# infer_agilex.py

from lerobot.robots.agilex import AgileXRobot, AgilexConfig
from lerobot.policies.act import ACTPolicy, ACTConfig
import torch

# 1. 加载策略
policy = ACTPolicy(ACTConfig())
policy.load_state_dict(torch.load("path/to/policy.pt"))
policy.eval()

# 2. 创建机器人
robot = AgileXRobot(AgilexConfig())
robot.connect()

# 3. 推理循环
try:
    while True:
        # 读取观测
        observation = robot.get_observation()

        # 转换为batch
        observation_batch = {k: torch.tensor(v).unsqueeze(0) for k, v in observation.items()}

        # 推理
        with torch.no_grad():
            action = policy.select_action(observation_batch)

        # 发送到机器人
        robot.send_action(action)

except KeyboardInterrupt:
    pass
finally:
    robot.disconnect()
```

### 6. 使用 LeRobot CLI

```bash
# 数据采集
lerobot-record \
  --robot.type=agilex \
  --teleop.type=agilex_leader \
  --dataset_repo_id=agilex/dual_arm_demo \
  --fps=30 \
  --max_episodes=100

# 训练
lerobot-train \
  --policy.type=act \
  --dataset_repo_id=agilex/dual_arm_demo \
  --output_dir=outputs/agilex_act \
  --batch_size=32 \
  --epochs=1000

# 评估
lerobot-eval \
  --policy.path=outputs/agilex_act \
  --robot.type=agilex \
  --eval.n_episodes=10
```

---

## 高级功能

### 1. 双臂协同策略

对于需要双臂协同的任务（如双手传递物体），修改策略输入：

```python
# 在 get_observation() 中添加相对位姿
import tf
import tf2_ros

def get_observation(self):
    obs = super().get_observation()

    # 获取左右臂末端相对位姿
    try:
        trans = self._tf_buffer.lookup_transform(
            'puppet_left_end_effector',
            'puppet_right_end_effector',
            rospy.Time(0)
        )
        obs["ee_relative_x"] = trans.transform.translation.x
        obs["ee_relative_y"] = trans.transform.translation.y
        obs["ee_relative_z"] = trans.transform.translation.z
    except:
        pass

    return obs
```

### 2. 力反馈（如支持）

```python
# 订阅力传感器topic
def _force_callback(self, msg):
    self._force_data = msg.data

# 在观测中添加力信息
def get_observation(self):
    obs = super().get_observation()
    if self._force_data is not None:
        for i, force in enumerate(self._force_data):
            obs[f"force_left_{i}"] = force
    return obs
```

### 3. 深度相机集成

```python
# 修改相机配置
cameras: dict[str, CameraConfig] = field(
    default_factory=lambda: {
        "camera_l_rgb": RosCameraConfig(
            topic_name="/camera_l/color/image_raw",
            ...
        ),
        "camera_l_depth": RosCameraConfig(
            topic_name="/camera_l/depth/image_raw",
            ...
        ),
    }
)
```

---

## 总结

本文档提供了完整的Agilex双臂机器人与LeRobot的集成方案：

✅ **通信适配**：通过ROS topic实现与Piper机械臂通信
✅ **双臂支持**：统一12个关节的命名和控制
✅ **遥操作**：主臂读取+从臂控制的闭环系统
✅ **相机集成**：ROS图像topic到OpenCV的桥接
✅ **安全机制**：关节限幅、速度限制、急停
✅ **测试方案**：从单元测试到硬件闭环测试
✅ **生产就绪**：启动脚本、YAML配置、CLI支持

### 核心优势

1. **无缝集成**：Agilex用户无需修改原有ROS驱动即可使用LeRobot
2. **灵活配置**：支持单臂/双臂、主从分离或协同等多种配置
3. **安全可靠**：多层安全保护，适合研究与教学
4. **完整工具链**：数据采集、训练、评估全流程支持

### 下一步工作

- [ ] 实现力反馈支持（如果硬件支持）
- [ ] 添加碰撞检测
- [ ] 优化ROS通信延迟
- [ ] 支持更多相机类型（RealSense等）
- [ ] 添加可视化工具（RViz集成）

---

## 参考资源

- **Piper ROS驱动**：`Piper_ros_private-ros-noetic`
- **Astra相机驱动**：`astra_camera`
- **LeRobot文档**：`docs/source/integrate_hardware.mdx`
- **代码示例**：`src/lerobot/robots/reachy2/`（外部SDK集成参考）
- **ROS-Python教程**：http://wiki.ros.org/rospy

---

**文档版本**：v1.0
**最后更新**：2025-12-02
**维护者**：LeRobot集成团队
