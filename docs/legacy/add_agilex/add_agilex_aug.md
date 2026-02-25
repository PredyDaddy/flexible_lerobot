# Agilex 双臂机器人 LeRobot 集成指南

> **兼容性说明 (lerobot 0.4.3 baseline)**  
> 本文档依赖旧仓库中的 AgileX/ROS Camera 实现（例如 `lerobot.robots.agilex`、`lerobot.cameras.ros_camera`）。  
> 当前仓库的 0.4.3 baseline 默认代码树不包含这些模块，因此本文不能直接照着跑；建议仅作为历史记录/移植参考（路线 B）。  
>
> 本文档面向希望将 Agilex Piper 双臂遥操作系统集成到 LeRobot 框架的开发者。涵盖架构设计、ROS 桥接层实现、代码示例、测试方案及常见问题排查。

## 目录

1. [Agilex 机器人概述](#1-agilex-机器人概述)
2. [集成架构设计](#2-集成架构设计)
3. [前置准备](#3-前置准备)
4. [详细实现步骤](#4-详细实现步骤)
5. [ROS 桥接层实现](#5-ros-桥接层实现)
6. [相机系统集成](#6-相机系统集成)
7. [遥操作支持](#7-遥操作支持)
8. [配置与注册](#8-配置与注册)
9. [测试指南](#9-测试指南)
10. [常见问题与调试](#10-常见问题与调试)
11. [最佳实践](#11-最佳实践)
12. [附录](#12-附录)

---

## 1. Agilex 机器人概述

### 1.1 硬件规格

Agilex Piper 是一套**两主两从**的双臂遥操作系统，主要规格如下：

| 项目 | 规格 |
|------|------|
| **机械臂类型** | Piper 协作机械臂 |
| **自由度** | 每条臂 7-DOF（6关节 + 1夹爪） |
| **总自由度** | 14-DOF（双臂） |
| **通信接口** | CAN 总线（通过 USB 转接） |
| **上位机通信** | ROS Noetic（roslaunch + rostopic） |
| **相机** | Astra RGB-D 相机 × 2-3（左腕、右腕、前方） |
| **图像分辨率** | 640×480 @ 30fps |

### 1.2 原版系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     Agilex 原版系统架构                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   roscore    │  │  piper_ros   │  │ astra_camera │          │
│  │              │  │ (CAN驱动)    │  │ (相机驱动)    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         │                │                  │                   │
│         └────────────────┼──────────────────┘                   │
│                          ▼                                      │
│  ┌────────────────────────────────────────────────────────────┐│
│  │                    ROS Topic 通信                          ││
│  │  /master/joint_left   /master/joint_right  (主臂位置)      ││
│  │  /puppet/joint_left   /puppet/joint_right  (从臂位置)      ││
│  │  /camera_l/color/image_raw  /camera_r/color/image_raw      ││
│  │  /camera_f/color/image_raw  (可选前方相机)                  ││
│  └────────────────────────────────────────────────────────────┘│
│                          ▼                                      │
│  ┌────────────────────────────────────────────────────────────┐│
│  │              collect_data.py / replay_data.py              ││
│  │              (HDF5 数据采集与回放)                          ││
│  └────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 原版 ROS Topic 列表

根据原版代码分析，系统使用以下 ROS Topic：

| Topic | 消息类型 | 方向 | 说明 |
|-------|----------|------|------|
| `/master/joint_left` | `JointState` | 订阅 | 左主臂关节状态（遥操作输入） |
| `/master/joint_right` | `JointState` | 订阅 | 右主臂关节状态 |
| `/puppet/joint_left` | `JointState` | 订阅/发布 | 左从臂关节状态 |
| `/puppet/joint_right` | `JointState` | 订阅/发布 | 右从臂关节状态 |
| `/camera_l/color/image_raw` | `Image` | 订阅 | 左腕相机彩色图像 |
| `/camera_r/color/image_raw` | `Image` | 订阅 | 右腕相机彩色图像 |
| `/camera_f/color/image_raw` | `Image` | 订阅 | 前方相机（可选） |
| `/odom` | `Odometry` | 订阅 | 移动底盘里程计（可选） |

### 1.4 数据格式

原版代码使用 HDF5 格式存储数据：

```python
# 数据字典结构
data_dict = {
    '/observations/qpos': [],      # 从臂位置 [14,] = [left_7 + right_7]
    '/observations/qvel': [],      # 从臂速度 [14,]
    '/observations/effort': [],    # 从臂力矩 [14,]
    '/action': [],                 # 主臂位置（动作） [14,]
    '/base_action': [],            # 底盘速度 [2,] = [linear_x, angular_z]
    '/observations/images/cam_high': [],      # 图像 [480, 640, 3]
    '/observations/images/cam_left_wrist': [],
    '/observations/images/cam_right_wrist': [],
}
```

### 1.5 关节命名约定

每条臂有 7 个关节（6 关节 + 1 夹爪）：

| 原版索引 | 关节名称 | 说明 |
|----------|----------|------|
| 0 | joint0 | 基座旋转 |
| 1 | joint1 | 肩部俯仰 |
| 2 | joint2 | 肩部旋转 |
| 3 | joint3 | 肘部 |
| 4 | joint4 | 腕部旋转 |
| 5 | joint5 | 腕部俯仰 |
| 6 | gripper | 夹爪开合 |

---

## 2. 集成架构设计

### 2.1 核心设计决策

由于 Agilex 系统使用 **ROS 作为底层通信**，与 LeRobot 的标准 MotorsBus 架构不同，我们采用以下集成策略：

| 决策点 | 方案 | 理由 |
|--------|------|------|
| **电机控制** | ROS 桥接层封装 | 原版已有完善的 CAN 驱动，无需重写 |
| **相机接入** | ROS 图像订阅 | 利用 ROS 的时间同步机制 |
| **数据同步** | 基于时间戳对齐 | 复用原版的帧同步逻辑 |
| **校准** | 无需手动校准 | ROS 驱动已处理编码器零点 |

### 2.2 集成架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                    LeRobot 集成架构                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   LeRobot 应用层                          │   │
│  │  lerobot-train / lerobot-record / lerobot-eval            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                            │                                     │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              AgileXRobot (Robot 子类)                     │   │
│  │  ├── observation_features: {joint.pos, cameras}          │   │
│  │  ├── action_features: {joint.pos}                        │   │
│  │  ├── connect() / disconnect()                             │   │
│  │  ├── get_observation() -> dict                            │   │
│  │  └── send_action(action) -> dict                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                            │                                     │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              AgileXROSBridge (ROS 桥接层)                 │   │
│  │  ├── init_ros_node()                                      │   │
│  │  ├── subscribe_topics()                                   │   │
│  │  ├── get_synchronized_frame() -> (joints, images)        │   │
│  │  └── publish_joint_command(action)                        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                            │                                     │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    ROS 底层                               │   │
│  │  roscore + piper_ros + astra_camera                       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 类关系图

```
┌─────────────────────────────────────────────────────────────────┐
│                         Robot (ABC)                              │
│  LeRobot 抽象基类                                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      AgileXRobot                                 │
│  ├── config: AgileXConfig                                       │
│  ├── ros_bridge: AgileXROSBridge                                │
│  ├── left_arm_joints: list[str]                                 │
│  ├── right_arm_joints: list[str]                                │
│  │                                                               │
│  │ Properties:                                                   │
│  ├── observation_features -> dict[str, type | tuple]            │
│  ├── action_features -> dict[str, type]                         │
│  ├── is_connected -> bool                                        │
│  ├── is_calibrated -> bool (always True)                        │
│  │                                                               │
│  │ Methods:                                                      │
│  ├── connect(calibrate=True) -> None                             │
│  ├── calibrate() -> None (no-op)                                 │
│  ├── configure() -> None                                         │
│  ├── get_observation() -> dict[str, Any]                         │
│  ├── send_action(action: dict) -> dict[str, Any]                │
│  └── disconnect() -> None                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
┌────────────────────────┐  ┌────────────────────────┐
│   AgileXROSBridge      │  │    AgileXConfig        │
│   ROS 通信封装          │  │    配置数据类           │
├────────────────────────┤  ├────────────────────────┤
│ - node_name            │  │ + camera_names         │
│ - subscribers          │  │ + image_topics         │
│ - publishers           │  │ + joint_topics         │
│ - deques (缓冲队列)     │  │ + frame_rate           │
│ - cv_bridge            │  │ + use_depth_image      │
├────────────────────────┤  │ + use_robot_base       │
│ + init_ros()           │  │ + max_relative_target  │
│ + get_frame()          │  └────────────────────────┘
│ + publish_command()    │
│ + shutdown()           │
└────────────────────────┘
```

### 2.4 数据流图

```
┌─────────────────────────────────────────────────────────────────┐
│                        数据采集流程                              │
└─────────────────────────────────────────────────────────────────┘

  ┌───────────┐         ┌───────────┐         ┌───────────┐
  │ 主臂(左)  │         │ 主臂(右)  │         │  相机×3   │
  └─────┬─────┘         └─────┬─────┘         └─────┬─────┘
        │                     │                     │
        ▼                     ▼                     ▼
  /master/joint_left    /master/joint_right   /camera_*/image_raw
        │                     │                     │
        └──────────┬──────────┴──────────┬──────────┘
                   │                     │
                   ▼                     ▼
           ┌──────────────────────────────────────┐
           │        AgileXROSBridge               │
           │     (时间戳同步 + 数据对齐)           │
           └──────────────────────────────────────┘
                            │
                            ▼
           ┌──────────────────────────────────────┐
           │           AgileXRobot                │
           │  get_observation() -> {              │
           │    "left_joint0.pos": float,         │
           │    "left_joint1.pos": float,         │
           │    ...                               │
           │    "right_joint6.pos": float,        │
           │    "cam_left_wrist": ndarray,        │
           │    "cam_right_wrist": ndarray,       │
           │  }                                   │
           └──────────────────────────────────────┘
                            │
                            ▼
           ┌──────────────────────────────────────┐
           │         LeRobotDataset               │
           │     (存储为 parquet + mp4)           │
           └──────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────┐
│                        动作执行流程                              │
└─────────────────────────────────────────────────────────────────┘

           ┌──────────────────────────────────────┐
           │          Policy.select_action()      │
           │   action = {"left_joint0.pos": 0.1,  │
           │             "right_joint0.pos": -0.2,│
           │             ...}                     │
           └──────────────────────────────────────┘
                            │
                            ▼
           ┌──────────────────────────────────────┐
           │         AgileXRobot.send_action()    │
           │  1. 解析 action 字典                 │
           │  2. 安全限幅 (max_relative_target)   │
           │  3. 调用 ros_bridge.publish_command()│
           └──────────────────────────────────────┘
                            │
                            ▼
           ┌──────────────────────────────────────┐
           │         AgileXROSBridge              │
           │  publish to /master/joint_left       │
           │  publish to /master/joint_right      │
           └──────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
        ┌───────────┐               ┌───────────┐
        │ 从臂(左)  │               │ 从臂(右)  │
        │ (跟随执行) │               │ (跟随执行) │
        └───────────┘               └───────────┘
```

### 2.5 LeRobot 特征映射

将原版数据格式映射到 LeRobot 的 `observation_features` 和 `action_features`：

| 原版数据 | LeRobot 特征键名 | 类型 | 说明 |
|----------|------------------|------|------|
| `qpos[0:7]` | `left_joint0.pos` ~ `left_gripper.pos` | `float` | 左臂关节位置 |
| `qpos[7:14]` | `right_joint0.pos` ~ `right_gripper.pos` | `float` | 右臂关节位置 |
| `action[0:7]` | 同上 | `float` | 左臂目标位置 |
| `action[7:14]` | 同上 | `float` | 右臂目标位置 |
| `images/cam_left_wrist` | `cam_left_wrist` | `(480, 640, 3)` | 左腕相机 |
| `images/cam_right_wrist` | `cam_right_wrist` | `(480, 640, 3)` | 右腕相机 |
| `images/cam_high` | `cam_high` | `(480, 640, 3)` | 前方相机 |
| `base_action` | `base.vx`, `base.vtheta` | `float` | 底盘速度（可选） |

---

## 3. 前置准备

### 3.1 系统要求

| 项目 | 要求 |
|------|------|
| **操作系统** | Ubuntu 20.04 LTS |
| **ROS 版本** | ROS Noetic |
| **Python** | 3.10+（通过 Conda 管理） |
| **CUDA** | 11.8+（用于训练） |

### 3.2 硬件连接

1. **机械臂连接**：通过 USB-CAN 适配器连接主臂和从臂
2. **相机连接**：Astra 相机通过 USB 连接
3. **串口权限**：
   ```bash
   sudo usermod -aG dialout $USER
   # 重新登录生效
   ```

### 3.3 ROS 环境配置

```bash
# 1. 确保 ROS Noetic 已安装
source /opt/ros/noetic/setup.bash

# 2. 配置 Piper ROS 工作空间
cd /home/agilex/cobot_magic/Piper_ros_private-ros-noetic
./can_config.sh  # 配置 CAN 接口
source devel/setup.bash

# 3. 验证 ROS 环境
roscore &
rostopic list  # 应该能看到基础 topic
```

### 3.4 LeRobot 环境配置

```bash
# 1. 创建 Conda 环境
conda create -y -n lerobot python=3.10
conda activate lerobot
conda install ffmpeg -c conda-forge

# 2. 安装 LeRobot（开发模式）
cd ~/cqy/lerobot_dev/lerobot_4_2/lerobot_my_fork
pip install -e ".[dev,test]"

# 3. 安装 ROS Python 依赖
pip install rospkg catkin_pkg netifaces

# 4. 安装额外依赖
pip install opencv-python h5py dm_env
```

### 3.5 启动原版 ROS 节点

在使用 LeRobot 集成之前，需要先启动 Agilex 的 ROS 驱动：

```bash
# 终端 1: roscore
roscore

# 终端 2: 机械臂驱动（遥操作模式）
conda activate act
cd /home/agilex/cobot_magic/Piper_ros_private-ros-noetic
./can_config.sh
source devel/setup.bash
roslaunch piper start_ms_piper.launch mode:=0 auto_enable:=false

# 终端 3: 相机驱动
roslaunch astra_camera multi_camera.launch

# 终端 4: 验证 Topic
rostopic list
# 应该看到:
# /master/joint_left
# /master/joint_right
# /puppet/joint_left
# /puppet/joint_right
# /camera_l/color/image_raw
# /camera_r/color/image_raw
```

### 3.6 参考实现阅读

建议先阅读以下文件理解实现模式：

| 文件 | 说明 |
|------|------|
| `src/lerobot/robots/bi_so100_follower/` | 双臂机器人实现参考 |
| `src/lerobot/robots/reachy2/robot_reachy2.py` | 外部 SDK 集成示例 |
| `aiglex_origin_code/collect_data.py` | 原版数据采集实现 |
| `aiglex_origin_code/replay_data.py` | 原版数据回放实现 |

---

## 4. 详细实现步骤

### 4.1 文件结构

在 `src/lerobot/robots/` 下创建以下文件结构：

```
src/lerobot/robots/agilex/
├── __init__.py
├── config_agilex.py          # 配置类定义
├── robot_agilex.py           # Robot 子类实现
├── ros_bridge.py             # ROS 桥接层
└── constants.py              # 常量定义
```

### 4.2 常量定义 (constants.py)

```python
#!/usr/bin/env python
"""Agilex Piper 双臂机器人常量定义"""

# 关节名称列表（每臂 7 个关节）
JOINT_NAMES = ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "gripper"]

# LeRobot 特征键名映射
AGILEX_LEFT_ARM_JOINTS = {
    f"left_{name}.pos": i for i, name in enumerate(JOINT_NAMES)
}

AGILEX_RIGHT_ARM_JOINTS = {
    f"right_{name}.pos": i for i, name in enumerate(JOINT_NAMES)
}

# 所有关节特征
AGILEX_ALL_JOINTS = {**AGILEX_LEFT_ARM_JOINTS, **AGILEX_RIGHT_ARM_JOINTS}

# ROS Topic 名称
ROS_TOPICS = {
    "master_left": "/master/joint_left",
    "master_right": "/master/joint_right",
    "puppet_left": "/puppet/joint_left",
    "puppet_right": "/puppet/joint_right",
    "camera_left": "/camera_l/color/image_raw",
    "camera_right": "/camera_r/color/image_raw",
    "camera_front": "/camera_f/color/image_raw",
    "odom": "/odom",
}

# 默认相机配置
DEFAULT_CAMERA_CONFIG = {
    "cam_left_wrist": {
        "topic": "/camera_l/color/image_raw",
        "width": 640,
        "height": 480,
    },
    "cam_right_wrist": {
        "topic": "/camera_r/color/image_raw",
        "width": 640,
        "height": 480,
    },
    "cam_high": {
        "topic": "/camera_f/color/image_raw",
        "width": 640,
        "height": 480,
    },
}

# 关节限位（弧度）- 根据实际硬件调整
JOINT_LIMITS = {
    "joint0": (-3.14, 3.14),
    "joint1": (-1.57, 1.57),
    "joint2": (-3.14, 3.14),
    "joint3": (-1.57, 1.57),
    "joint4": (-3.14, 3.14),
    "joint5": (-1.57, 1.57),
    "gripper": (0.0, 1.0),  # 归一化夹爪开合
}
```

### 4.3 配置类定义 (config_agilex.py)

```python
#!/usr/bin/env python
"""Agilex 机器人配置类"""

from dataclasses import dataclass, field
from typing import Any

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("agilex")
@dataclass
class AgileXConfig(RobotConfig):
    """
    Agilex Piper 双臂机器人配置。

    该机器人通过 ROS 与底层硬件通信，需要先启动 ROS 驱动节点。

    Attributes:
        ros_node_name: ROS 节点名称
        master_left_topic: 左主臂关节状态 Topic
        master_right_topic: 右主臂关节状态 Topic
        puppet_left_topic: 左从臂关节状态 Topic
        puppet_right_topic: 右从臂关节状态 Topic
        use_robot_base: 是否使用移动底盘
        odom_topic: 里程计 Topic（仅当 use_robot_base=True 时使用）
        max_relative_target: 单步最大位移限制（弧度）
        frame_rate: 数据采集帧率
        cameras: 相机配置字典
    """

    # ROS 节点配置
    ros_node_name: str = "lerobot_agilex"

    # 关节状态 Topic
    master_left_topic: str = "/master/joint_left"
    master_right_topic: str = "/master/joint_right"
    puppet_left_topic: str = "/puppet/joint_left"
    puppet_right_topic: str = "/puppet/joint_right"

    # 移动底盘配置（可选）
    use_robot_base: bool = False
    odom_topic: str = "/odom"

    # 安全限制
    max_relative_target: float | None = 0.5  # 弧度/步

    # 数据采集配置
    frame_rate: int = 30

    # 相机配置
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # 图像 Topic 配置（用于 ROS 图像订阅）
    image_topics: dict[str, str] = field(default_factory=lambda: {
        "cam_left_wrist": "/camera_l/color/image_raw",
        "cam_right_wrist": "/camera_r/color/image_raw",
        "cam_high": "/camera_f/color/image_raw",
    })

    # 是否在断开连接时禁用力矩
    disable_torque_on_disconnect: bool = True
```

### 4.4 Robot 子类实现 (robot_agilex.py)

```python
#!/usr/bin/env python
"""Agilex Piper 双臂机器人 LeRobot 实现"""

import logging
import time
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.robots.robot import Robot
from lerobot.robots.utils import ensure_safe_goal_position

from .config_agilex import AgileXConfig
from .constants import AGILEX_LEFT_ARM_JOINTS, AGILEX_RIGHT_ARM_JOINTS, JOINT_NAMES
from .ros_bridge import AgileXROSBridge

logger = logging.getLogger(__name__)


class AgileXRobot(Robot):
    """
    Agilex Piper 双臂机器人实现。

    该类封装了与 ROS 的通信，提供 LeRobot 标准接口。
    需要先启动 Agilex ROS 驱动节点才能使用。

    使用示例:
        config = AgileXConfig(
            cameras={
                "cam_left_wrist": OpenCVCameraConfig(index=0, width=640, height=480),
            }
        )
        robot = AgileXRobot(config)
        robot.connect()
        obs = robot.get_observation()
        robot.send_action(action)
        robot.disconnect()
    """

    config_class = AgileXConfig
    name = "agilex"

    def __init__(self, config: AgileXConfig):
        super().__init__(config)
        self.config = config

        # ROS 桥接层（延迟初始化）
        self.ros_bridge: AgileXROSBridge | None = None

        # 关节名称列表
        self.left_arm_joints = list(AGILEX_LEFT_ARM_JOINTS.keys())
        self.right_arm_joints = list(AGILEX_RIGHT_ARM_JOINTS.keys())
        self.all_joints = self.left_arm_joints + self.right_arm_joints

        # 日志记录
        self.logs: dict[str, float] = {}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """定义观测特征：关节位置 + 相机图像"""
        features = {}

        # 关节位置特征
        for joint in self.all_joints:
            features[joint] = float

        # 相机图像特征
        for cam_name, topic in self.config.image_topics.items():
            # 默认图像尺寸，实际尺寸在连接后更新
            features[cam_name] = (480, 640, 3)

        # 底盘速度特征（可选）
        if self.config.use_robot_base:
            features["base.vx"] = float
            features["base.vtheta"] = float

        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        """定义动作特征：关节目标位置"""
        features = {}

        # 关节位置特征
        for joint in self.all_joints:
            features[joint] = float

        # 底盘速度特征（可选）
        if self.config.use_robot_base:
            features["base.vx"] = float
            features["base.vtheta"] = float

        return features

    @property
    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self.ros_bridge is not None and self.ros_bridge.is_connected

    @property
    def is_calibrated(self) -> bool:
        """Agilex 机器人无需手动校准，ROS 驱动已处理"""
        return True

    def connect(self, calibrate: bool = True) -> None:
        """
        连接机器人。

        初始化 ROS 节点并订阅相关 Topic。

        Args:
            calibrate: 是否执行校准（Agilex 忽略此参数）
        """
        if self.is_connected:
            logger.warning("Robot already connected")
            return

        logger.info("Connecting to Agilex robot via ROS...")

        # 初始化 ROS 桥接层
        self.ros_bridge = AgileXROSBridge(
            node_name=self.config.ros_node_name,
            puppet_left_topic=self.config.puppet_left_topic,
            puppet_right_topic=self.config.puppet_right_topic,
            master_left_topic=self.config.master_left_topic,
            master_right_topic=self.config.master_right_topic,
            image_topics=self.config.image_topics,
            use_robot_base=self.config.use_robot_base,
            odom_topic=self.config.odom_topic,
        )

        # 等待首帧数据
        logger.info("Waiting for first frame from ROS topics...")
        timeout = 10.0
        start_time = time.time()
        while not self.ros_bridge.has_data():
            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Timeout waiting for ROS data. "
                    f"Please ensure ROS nodes are running."
                )
            time.sleep(0.1)

        logger.info("Agilex robot connected successfully")

        # 配置机器人
        self.configure()

    def calibrate(self) -> None:
        """
        校准机器人。

        Agilex 机器人使用绝对编码器，ROS 驱动已处理零点校准，
        因此此方法为空操作。
        """
        logger.info("Agilex robot uses absolute encoders, no calibration needed")

    def configure(self) -> None:
        """配置机器人参数"""
        logger.info("Configuring Agilex robot...")
        # 可以在这里添加额外的配置逻辑
        # 例如设置 PID 参数、速度限制等

    def get_observation(self) -> dict[str, Any]:
        """
        获取当前观测。

        Returns:
            包含关节位置和相机图像的字典
        """
        if not self.is_connected:
            raise RuntimeError("Robot not connected")

        obs_dict = {}

        # 获取关节状态
        before_read_t = time.perf_counter()
        joint_state = self.ros_bridge.get_joint_state()
        self.logs["read_joints_dt_s"] = time.perf_counter() - before_read_t

        # 解析左臂关节
        left_positions = joint_state.get("puppet_left", [0.0] * 7)
        for i, joint_name in enumerate(self.left_arm_joints):
            obs_dict[joint_name] = float(left_positions[i]) if i < len(left_positions) else 0.0

        # 解析右臂关节
        right_positions = joint_state.get("puppet_right", [0.0] * 7)
        for i, joint_name in enumerate(self.right_arm_joints):
            obs_dict[joint_name] = float(right_positions[i]) if i < len(right_positions) else 0.0

        # 获取相机图像
        before_cam_t = time.perf_counter()
        images = self.ros_bridge.get_images()
        self.logs["read_cameras_dt_s"] = time.perf_counter() - before_cam_t

        for cam_name, image in images.items():
            obs_dict[cam_name] = image

        # 获取底盘状态（可选）
        if self.config.use_robot_base:
            base_state = self.ros_bridge.get_base_state()
            obs_dict["base.vx"] = base_state.get("vx", 0.0)
            obs_dict["base.vtheta"] = base_state.get("vtheta", 0.0)

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        发送动作指令。

        Args:
            action: 包含关节目标位置的字典

        Returns:
            实际发送的动作（可能经过安全限幅）
        """
        if not self.is_connected:
            raise RuntimeError("Robot not connected")

        before_write_t = time.perf_counter()

        # 获取当前位置用于安全限幅
        current_state = self.ros_bridge.get_joint_state()
        current_left = current_state.get("puppet_left", [0.0] * 7)
        current_right = current_state.get("puppet_right", [0.0] * 7)

        # 解析并限幅左臂动作
        left_action = []
        for i, joint_name in enumerate(self.left_arm_joints):
            goal = action.get(joint_name, current_left[i] if i < len(current_left) else 0.0)

            # 安全限幅
            if self.config.max_relative_target is not None:
                current = current_left[i] if i < len(current_left) else 0.0
                goal = ensure_safe_goal_position(
                    {joint_name: (goal, current)},
                    self.config.max_relative_target
                )[joint_name]

            left_action.append(float(goal))

        # 解析并限幅右臂动作
        right_action = []
        for i, joint_name in enumerate(self.right_arm_joints):
            goal = action.get(joint_name, current_right[i] if i < len(current_right) else 0.0)

            # 安全限幅
            if self.config.max_relative_target is not None:
                current = current_right[i] if i < len(current_right) else 0.0
                goal = ensure_safe_goal_position(
                    {joint_name: (goal, current)},
                    self.config.max_relative_target
                )[joint_name]

            right_action.append(float(goal))

        # 发送到 ROS
        self.ros_bridge.publish_joint_command(left_action, right_action)

        # 处理底盘动作（可选）
        if self.config.use_robot_base:
            vx = action.get("base.vx", 0.0)
            vtheta = action.get("base.vtheta", 0.0)
            self.ros_bridge.publish_base_command(vx, vtheta)

        self.logs["write_action_dt_s"] = time.perf_counter() - before_write_t

        # 返回实际发送的动作
        sent_action = {}
        for i, joint_name in enumerate(self.left_arm_joints):
            sent_action[joint_name] = left_action[i]
        for i, joint_name in enumerate(self.right_arm_joints):
            sent_action[joint_name] = right_action[i]

        if self.config.use_robot_base:
            sent_action["base.vx"] = action.get("base.vx", 0.0)
            sent_action["base.vtheta"] = action.get("base.vtheta", 0.0)

        return sent_action

    def disconnect(self) -> None:
        """断开机器人连接"""
        if self.ros_bridge is not None:
            logger.info("Disconnecting Agilex robot...")

            if self.config.disable_torque_on_disconnect:
                # 发送零速度指令
                logger.info("Sending zero velocity command before disconnect")
                self.ros_bridge.publish_base_command(0.0, 0.0)

            self.ros_bridge.shutdown()
            self.ros_bridge = None

            logger.info("Agilex robot disconnected")
```

---

## 5. ROS 桥接层实现

### 5.1 ROS 桥接层概述

ROS 桥接层 (`AgileXROSBridge`) 负责：
1. 初始化 ROS 节点
2. 订阅关节状态和图像 Topic
3. 发布关节指令
4. 数据缓冲和时间同步

### 5.2 完整实现 (ros_bridge.py)

```python
#!/usr/bin/env python
"""Agilex ROS 桥接层实现"""

import logging
import threading
import time
from collections import deque
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# 尝试导入 ROS 相关库
try:
    import rospy
    from sensor_msgs.msg import JointState, Image
    from nav_msgs.msg import Odometry
    from cv_bridge import CvBridge
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    logger.warning("ROS not available. Install rospy for hardware control.")


class AgileXROSBridge:
    """
    Agilex 机器人 ROS 通信桥接层。

    负责与 ROS 节点通信，订阅传感器数据，发布控制指令。

    Attributes:
        node_name: ROS 节点名称
        is_connected: 是否已连接
    """

    def __init__(
        self,
        node_name: str = "lerobot_agilex",
        puppet_left_topic: str = "/puppet/joint_left",
        puppet_right_topic: str = "/puppet/joint_right",
        master_left_topic: str = "/master/joint_left",
        master_right_topic: str = "/master/joint_right",
        image_topics: dict[str, str] | None = None,
        use_robot_base: bool = False,
        odom_topic: str = "/odom",
        buffer_size: int = 10,
    ):
        """
        初始化 ROS 桥接层。

        Args:
            node_name: ROS 节点名称
            puppet_left_topic: 左从臂关节状态 Topic
            puppet_right_topic: 右从臂关节状态 Topic
            master_left_topic: 左主臂关节状态 Topic
            master_right_topic: 右主臂关节状态 Topic
            image_topics: 图像 Topic 字典 {cam_name: topic_name}
            use_robot_base: 是否使用移动底盘
            odom_topic: 里程计 Topic
            buffer_size: 数据缓冲区大小
        """
        if not ROS_AVAILABLE:
            raise RuntimeError(
                "ROS is not available. Please install rospy:\n"
                "  pip install rospkg catkin_pkg\n"
                "And source your ROS workspace."
            )

        self.node_name = node_name
        self.puppet_left_topic = puppet_left_topic
        self.puppet_right_topic = puppet_right_topic
        self.master_left_topic = master_left_topic
        self.master_right_topic = master_right_topic
        self.image_topics = image_topics or {}
        self.use_robot_base = use_robot_base
        self.odom_topic = odom_topic
        self.buffer_size = buffer_size

        # 数据缓冲区
        self._puppet_left_buffer: deque = deque(maxlen=buffer_size)
        self._puppet_right_buffer: deque = deque(maxlen=buffer_size)
        self._master_left_buffer: deque = deque(maxlen=buffer_size)
        self._master_right_buffer: deque = deque(maxlen=buffer_size)
        self._image_buffers: dict[str, deque] = {
            name: deque(maxlen=buffer_size) for name in self.image_topics
        }
        self._odom_buffer: deque = deque(maxlen=buffer_size)

        # 线程锁
        self._lock = threading.Lock()

        # CV Bridge 用于图像转换
        self._cv_bridge = CvBridge()

        # 订阅者和发布者
        self._subscribers: list = []
        self._publishers: dict = {}

        # 初始化 ROS 节点
        self._init_ros()

    def _init_ros(self) -> None:
        """初始化 ROS 节点和订阅者"""
        try:
            # 初始化节点（如果尚未初始化）
            if not rospy.core.is_initialized():
                rospy.init_node(self.node_name, anonymous=True, disable_signals=True)

            logger.info(f"ROS node '{self.node_name}' initialized")

            # 订阅从臂关节状态
            self._subscribers.append(
                rospy.Subscriber(
                    self.puppet_left_topic,
                    JointState,
                    self._puppet_left_callback,
                    queue_size=1
                )
            )
            self._subscribers.append(
                rospy.Subscriber(
                    self.puppet_right_topic,
                    JointState,
                    self._puppet_right_callback,
                    queue_size=1
                )
            )

            # 订阅主臂关节状态（用于遥操作）
            self._subscribers.append(
                rospy.Subscriber(
                    self.master_left_topic,
                    JointState,
                    self._master_left_callback,
                    queue_size=1
                )
            )
            self._subscribers.append(
                rospy.Subscriber(
                    self.master_right_topic,
                    JointState,
                    self._master_right_callback,
                    queue_size=1
                )
            )

            # 订阅图像 Topic
            for cam_name, topic in self.image_topics.items():
                self._subscribers.append(
                    rospy.Subscriber(
                        topic,
                        Image,
                        lambda msg, name=cam_name: self._image_callback(msg, name),
                        queue_size=1
                    )
                )
                logger.info(f"Subscribed to image topic: {topic} as {cam_name}")

            # 订阅里程计（可选）
            if self.use_robot_base:
                self._subscribers.append(
                    rospy.Subscriber(
                        self.odom_topic,
                        Odometry,
                        self._odom_callback,
                        queue_size=1
                    )
                )

            # 创建发布者（用于回放/推理时发送指令）
            self._publishers["master_left"] = rospy.Publisher(
                self.master_left_topic,
                JointState,
                queue_size=1
            )
            self._publishers["master_right"] = rospy.Publisher(
                self.master_right_topic,
                JointState,
                queue_size=1
            )

            logger.info("ROS subscribers and publishers initialized")

        except Exception as e:
            logger.error(f"Failed to initialize ROS: {e}")
            raise

    # ==================== 回调函数 ====================

    def _puppet_left_callback(self, msg: "JointState") -> None:
        """左从臂关节状态回调"""
        with self._lock:
            self._puppet_left_buffer.append({
                "timestamp": msg.header.stamp.to_sec() if msg.header.stamp else time.time(),
                "position": list(msg.position),
                "velocity": list(msg.velocity) if msg.velocity else [],
                "effort": list(msg.effort) if msg.effort else [],
            })

    def _puppet_right_callback(self, msg: "JointState") -> None:
        """右从臂关节状态回调"""
        with self._lock:
            self._puppet_right_buffer.append({
                "timestamp": msg.header.stamp.to_sec() if msg.header.stamp else time.time(),
                "position": list(msg.position),
                "velocity": list(msg.velocity) if msg.velocity else [],
                "effort": list(msg.effort) if msg.effort else [],
            })

    def _master_left_callback(self, msg: "JointState") -> None:
        """左主臂关节状态回调"""
        with self._lock:
            self._master_left_buffer.append({
                "timestamp": msg.header.stamp.to_sec() if msg.header.stamp else time.time(),
                "position": list(msg.position),
            })

    def _master_right_callback(self, msg: "JointState") -> None:
        """右主臂关节状态回调"""
        with self._lock:
            self._master_right_buffer.append({
                "timestamp": msg.header.stamp.to_sec() if msg.header.stamp else time.time(),
                "position": list(msg.position),
            })

    def _image_callback(self, msg: "Image", cam_name: str) -> None:
        """图像回调"""
        try:
            # 将 ROS Image 转换为 numpy 数组
            cv_image = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

            with self._lock:
                self._image_buffers[cam_name].append({
                    "timestamp": msg.header.stamp.to_sec() if msg.header.stamp else time.time(),
                    "image": cv_image,
                })
        except Exception as e:
            logger.warning(f"Failed to convert image from {cam_name}: {e}")

    def _odom_callback(self, msg: "Odometry") -> None:
        """里程计回调"""
        with self._lock:
            self._odom_buffer.append({
                "timestamp": msg.header.stamp.to_sec() if msg.header.stamp else time.time(),
                "vx": msg.twist.twist.linear.x,
                "vy": msg.twist.twist.linear.y,
                "vtheta": msg.twist.twist.angular.z,
            })

    # ==================== 数据获取方法 ====================

    @property
    def is_connected(self) -> bool:
        """检查 ROS 是否连接"""
        return not rospy.is_shutdown()

    def has_data(self) -> bool:
        """检查是否有数据"""
        with self._lock:
            has_joints = len(self._puppet_left_buffer) > 0 and len(self._puppet_right_buffer) > 0
            has_images = all(len(buf) > 0 for buf in self._image_buffers.values())
            return has_joints and (not self.image_topics or has_images)

    def get_joint_state(self) -> dict[str, list[float]]:
        """
        获取最新的关节状态。

        Returns:
            包含 puppet_left, puppet_right, master_left, master_right 的字典
        """
        with self._lock:
            result = {
                "puppet_left": self._puppet_left_buffer[-1]["position"] if self._puppet_left_buffer else [0.0] * 7,
                "puppet_right": self._puppet_right_buffer[-1]["position"] if self._puppet_right_buffer else [0.0] * 7,
                "master_left": self._master_left_buffer[-1]["position"] if self._master_left_buffer else [0.0] * 7,
                "master_right": self._master_right_buffer[-1]["position"] if self._master_right_buffer else [0.0] * 7,
            }
        return result

    def get_images(self) -> dict[str, np.ndarray]:
        """
        获取最新的图像。

        Returns:
            包含各相机图像的字典 {cam_name: ndarray}
        """
        images = {}
        with self._lock:
            for cam_name, buffer in self._image_buffers.items():
                if buffer:
                    images[cam_name] = buffer[-1]["image"]
                else:
                    # 返回空图像
                    images[cam_name] = np.zeros((480, 640, 3), dtype=np.uint8)
        return images

    def get_base_state(self) -> dict[str, float]:
        """
        获取底盘状态。

        Returns:
            包含 vx, vy, vtheta 的字典
        """
        with self._lock:
            if self._odom_buffer:
                return {
                    "vx": self._odom_buffer[-1]["vx"],
                    "vy": self._odom_buffer[-1]["vy"],
                    "vtheta": self._odom_buffer[-1]["vtheta"],
                }
            return {"vx": 0.0, "vy": 0.0, "vtheta": 0.0}

    def get_master_action(self) -> dict[str, list[float]]:
        """
        获取主臂位置（用于遥操作数据采集）。

        Returns:
            包含 master_left, master_right 的字典
        """
        with self._lock:
            return {
                "master_left": self._master_left_buffer[-1]["position"] if self._master_left_buffer else [0.0] * 7,
                "master_right": self._master_right_buffer[-1]["position"] if self._master_right_buffer else [0.0] * 7,
            }

    # ==================== 指令发布方法 ====================

    def publish_joint_command(
        self,
        left_positions: list[float],
        right_positions: list[float]
    ) -> None:
        """
        发布关节位置指令。

        在回放/推理模式下，通过发布到 master topic 来控制从臂。

        Args:
            left_positions: 左臂目标位置 [7,]
            right_positions: 右臂目标位置 [7,]
        """
        # 创建 JointState 消息
        left_msg = JointState()
        left_msg.header.stamp = rospy.Time.now()
        left_msg.position = left_positions

        right_msg = JointState()
        right_msg.header.stamp = rospy.Time.now()
        right_msg.position = right_positions

        # 发布
        self._publishers["master_left"].publish(left_msg)
        self._publishers["master_right"].publish(right_msg)

    def publish_base_command(self, vx: float, vtheta: float) -> None:
        """
        发布底盘速度指令。

        Args:
            vx: 线速度 (m/s)
            vtheta: 角速度 (rad/s)
        """
        # 如果需要底盘控制，在这里添加发布逻辑
        # 例如发布到 /cmd_vel topic
        pass

    def shutdown(self) -> None:
        """关闭 ROS 连接"""
        logger.info("Shutting down ROS bridge...")

        # 取消订阅
        for sub in self._subscribers:
            sub.unregister()
        self._subscribers.clear()

        # 关闭发布者
        for pub in self._publishers.values():
            pub.unregister()
        self._publishers.clear()

        logger.info("ROS bridge shutdown complete")
```

### 5.3 Mock 实现（用于测试）

为了在没有 ROS 环境的情况下进行测试，提供 Mock 实现：

```python
#!/usr/bin/env python
"""Agilex ROS 桥接层 Mock 实现"""

import numpy as np
import time


class MockAgileXROSBridge:
    """
    Mock ROS 桥接层，用于单元测试。

    模拟 ROS 通信，返回随机数据。
    """

    def __init__(self, **kwargs):
        self.image_topics = kwargs.get("image_topics", {})
        self._connected = True
        self._start_time = time.time()

    @property
    def is_connected(self) -> bool:
        return self._connected

    def has_data(self) -> bool:
        return True

    def get_joint_state(self) -> dict[str, list[float]]:
        """返回模拟的关节状态"""
        t = time.time() - self._start_time
        # 生成平滑变化的模拟数据
        return {
            "puppet_left": [np.sin(t + i * 0.1) * 0.5 for i in range(7)],
            "puppet_right": [np.cos(t + i * 0.1) * 0.5 for i in range(7)],
            "master_left": [np.sin(t + i * 0.1) * 0.5 for i in range(7)],
            "master_right": [np.cos(t + i * 0.1) * 0.5 for i in range(7)],
        }

    def get_images(self) -> dict[str, np.ndarray]:
        """返回模拟的图像"""
        images = {}
        for cam_name in self.image_topics:
            # 生成随机彩色图像
            images[cam_name] = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        return images

    def get_base_state(self) -> dict[str, float]:
        return {"vx": 0.0, "vy": 0.0, "vtheta": 0.0}

    def get_master_action(self) -> dict[str, list[float]]:
        return self.get_joint_state()

    def publish_joint_command(self, left_positions: list, right_positions: list) -> None:
        pass

    def publish_base_command(self, vx: float, vtheta: float) -> None:
        pass

    def shutdown(self) -> None:
        self._connected = False
```

---

## 6. 相机系统集成

### 6.1 相机集成方案

Agilex 系统使用 **ROS 图像 Topic** 获取相机数据，与 LeRobot 标准的 OpenCV/RealSense 相机接口不同。我们有两种集成方案：

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **ROS 图像订阅** | 利用 ROS 时间同步，与原版兼容 | 依赖 ROS 环境 | 生产环境 |
| **LeRobot 相机** | 独立于 ROS，更灵活 | 需要额外配置 | 独立部署 |

### 6.2 方案一：ROS 图像订阅（推荐）

已在 `AgileXROSBridge` 中实现，通过订阅 ROS Image Topic 获取图像：

```python
# 在 AgileXConfig 中配置图像 Topic
config = AgileXConfig(
    image_topics={
        "cam_left_wrist": "/camera_l/color/image_raw",
        "cam_right_wrist": "/camera_r/color/image_raw",
        "cam_high": "/camera_f/color/image_raw",
    }
)
```

### 6.3 方案二：LeRobot 相机接口

如果需要独立于 ROS 使用相机，可以使用 LeRobot 的标准相机接口：

```python
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig

# 使用 OpenCV 相机
config = AgileXConfig(
    cameras={
        "cam_left_wrist": OpenCVCameraConfig(
            index=0,
            width=640,
            height=480,
            fps=30,
        ),
        "cam_right_wrist": OpenCVCameraConfig(
            index=2,
            width=640,
            height=480,
            fps=30,
        ),
    },
    # 禁用 ROS 图像订阅
    image_topics={},
)
```

### 6.4 混合模式

也可以同时使用两种方式：

```python
# 部分相机使用 ROS，部分使用 LeRobot
config = AgileXConfig(
    # LeRobot 管理的相机
    cameras={
        "cam_external": OpenCVCameraConfig(index=4, width=1280, height=720),
    },
    # ROS 管理的相机
    image_topics={
        "cam_left_wrist": "/camera_l/color/image_raw",
        "cam_right_wrist": "/camera_r/color/image_raw",
    },
)
```

### 6.5 图像预处理

如果需要对图像进行预处理（如裁剪、缩放），可以在 `get_observation()` 中添加：

```python
def get_observation(self) -> dict[str, Any]:
    obs_dict = {}

    # ... 获取关节状态 ...

    # 获取并预处理图像
    images = self.ros_bridge.get_images()
    for cam_name, image in images.items():
        # 可选：裁剪中心区域
        # h, w = image.shape[:2]
        # crop_size = min(h, w)
        # start_h = (h - crop_size) // 2
        # start_w = (w - crop_size) // 2
        # image = image[start_h:start_h+crop_size, start_w:start_w+crop_size]

        # 可选：缩放到目标尺寸
        # image = cv2.resize(image, (224, 224))

        obs_dict[cam_name] = image

    return obs_dict
```

---

## 7. 遥操作支持

### 7.1 遥操作架构

Agilex 系统的遥操作使用**主从臂跟随**模式：
- **主臂**：操作员手持，提供动作输入
- **从臂**：跟随主臂运动，执行实际任务

```
┌─────────────────────────────────────────────────────────────────┐
│                      遥操作数据流                                │
└─────────────────────────────────────────────────────────────────┘

  ┌───────────┐                              ┌───────────┐
  │ 主臂(左)  │ ──── /master/joint_left ───▶ │ 从臂(左)  │
  └───────────┘                              └───────────┘
        │                                          │
        │                                          │
        ▼                                          ▼
  ┌───────────────────────────────────────────────────────────┐
  │                    AgileXTeleoperator                      │
  │  get_action() -> {                                        │
  │    "left_joint0.pos": master_left[0],                     │
  │    "left_joint1.pos": master_left[1],                     │
  │    ...                                                    │
  │    "right_joint6.pos": master_right[6],                   │
  │  }                                                        │
  └───────────────────────────────────────────────────────────┘
                            │
                            ▼
  ┌───────────────────────────────────────────────────────────┐
  │                    LeRobot 数据采集                        │
  │  observation = robot.get_observation()  # 从臂状态        │
  │  action = teleop.get_action()           # 主臂位置        │
  │  dataset.add_frame(observation, action)                   │
  └───────────────────────────────────────────────────────────┘
```

### 7.2 Teleoperator 实现

```python
#!/usr/bin/env python
"""Agilex 遥操作器实现"""

import logging
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from lerobot.teleoperators.teleoperator import Teleoperator, TeleoperatorConfig

from .constants import AGILEX_LEFT_ARM_JOINTS, AGILEX_RIGHT_ARM_JOINTS
from .ros_bridge import AgileXROSBridge

logger = logging.getLogger(__name__)


@TeleoperatorConfig.register_subclass("agilex_teleop")
@dataclass
class AgileXTeleoperatorConfig(TeleoperatorConfig):
    """Agilex 遥操作器配置"""

    ros_node_name: str = "lerobot_agilex_teleop"
    master_left_topic: str = "/master/joint_left"
    master_right_topic: str = "/master/joint_right"


class AgileXTeleoperator(Teleoperator):
    """
    Agilex 主臂遥操作器。

    读取主臂关节位置作为动作输入。
    """

    config_class = AgileXTeleoperatorConfig
    name = "agilex_teleop"

    def __init__(self, config: AgileXTeleoperatorConfig):
        super().__init__(config)
        self.config = config
        self.ros_bridge: AgileXROSBridge | None = None

        # 关节名称
        self.left_arm_joints = list(AGILEX_LEFT_ARM_JOINTS.keys())
        self.right_arm_joints = list(AGILEX_RIGHT_ARM_JOINTS.keys())

    @cached_property
    def action_features(self) -> dict[str, type]:
        """定义动作特征"""
        features = {}
        for joint in self.left_arm_joints + self.right_arm_joints:
            features[joint] = float
        return features

    @property
    def is_connected(self) -> bool:
        return self.ros_bridge is not None and self.ros_bridge.is_connected

    def connect(self) -> None:
        """连接遥操作器"""
        if self.is_connected:
            return

        logger.info("Connecting Agilex teleoperator...")

        self.ros_bridge = AgileXROSBridge(
            node_name=self.config.ros_node_name,
            master_left_topic=self.config.master_left_topic,
            master_right_topic=self.config.master_right_topic,
            puppet_left_topic="",  # 不需要订阅从臂
            puppet_right_topic="",
            image_topics={},
        )

        logger.info("Agilex teleoperator connected")

    def calibrate(self) -> None:
        """校准（无需操作）"""
        pass

    def get_action(self) -> dict[str, Any]:
        """
        获取主臂位置作为动作。

        Returns:
            包含关节目标位置的字典
        """
        if not self.is_connected:
            raise RuntimeError("Teleoperator not connected")

        master_state = self.ros_bridge.get_master_action()

        action = {}

        # 左臂
        left_positions = master_state.get("master_left", [0.0] * 7)
        for i, joint_name in enumerate(self.left_arm_joints):
            action[joint_name] = float(left_positions[i]) if i < len(left_positions) else 0.0

        # 右臂
        right_positions = master_state.get("master_right", [0.0] * 7)
        for i, joint_name in enumerate(self.right_arm_joints):
            action[joint_name] = float(right_positions[i]) if i < len(right_positions) else 0.0

        return action

    def disconnect(self) -> None:
        """断开连接"""
        if self.ros_bridge is not None:
            self.ros_bridge.shutdown()
            self.ros_bridge = None
            logger.info("Agilex teleoperator disconnected")
```

### 7.3 数据采集流程

使用 LeRobot 的 `lerobot-record` 命令进行数据采集：

```bash
# 方式一：使用命令行
lerobot-record \
    --robot.type=agilex \
    --teleop.type=agilex_teleop \
    --dataset.repo_id=user/agilex_demo \
    --dataset.num_episodes=10 \
    --fps=30

# 方式二：使用 Python 脚本
python -c "
from lerobot.scripts.record import record
from lerobot.robots.agilex import AgileXConfig, AgileXRobot
from lerobot.teleoperators.agilex import AgileXTeleoperatorConfig, AgileXTeleoperator

robot_config = AgileXConfig(
    image_topics={
        'cam_left_wrist': '/camera_l/color/image_raw',
        'cam_right_wrist': '/camera_r/color/image_raw',
    }
)
teleop_config = AgileXTeleoperatorConfig()

record(
    robot=AgileXRobot(robot_config),
    teleoperator=AgileXTeleoperator(teleop_config),
    repo_id='user/agilex_demo',
    num_episodes=10,
    fps=30,
)
"
```

### 7.4 与原版数据采集的对比

| 特性 | 原版 collect_data.py | LeRobot 集成 |
|------|---------------------|--------------|
| **数据格式** | HDF5 | Parquet + MP4 |
| **存储位置** | 本地文件 | 本地 + HuggingFace Hub |
| **元数据** | 手动管理 | 自动生成 |
| **可视化** | visualize_episodes.py | lerobot-dataset-viz |
| **版本控制** | 无 | Git LFS |

---

## 8. 配置与注册

### 8.1 工厂函数注册

在 `src/lerobot/robots/agilex/__init__.py` 中导出类：

```python
#!/usr/bin/env python
"""Agilex 机器人模块"""

from .config_agilex import AgileXConfig
from .robot_agilex import AgileXRobot

__all__ = ["AgileXConfig", "AgileXRobot"]
```

在 `src/lerobot/robots/__init__.py` 中添加导入：

```python
# 在文件末尾添加
from lerobot.robots.agilex import AgileXConfig, AgileXRobot
```

### 8.2 YAML 配置示例

创建配置文件 `lerobot/configs/robot/agilex.yaml`：

```yaml
# Agilex Piper 双臂机器人配置
_target_: lerobot.robots.agilex.AgileXConfig

# ROS 节点配置
ros_node_name: lerobot_agilex

# 关节状态 Topic
master_left_topic: /master/joint_left
master_right_topic: /master/joint_right
puppet_left_topic: /puppet/joint_left
puppet_right_topic: /puppet/joint_right

# 移动底盘（可选）
use_robot_base: false
odom_topic: /odom

# 安全限制
max_relative_target: 0.5  # 弧度/步

# 数据采集配置
frame_rate: 30

# 图像 Topic
image_topics:
  cam_left_wrist: /camera_l/color/image_raw
  cam_right_wrist: /camera_r/color/image_raw
  cam_high: /camera_f/color/image_raw

# 断开时禁用力矩
disable_torque_on_disconnect: true
```

### 8.3 命令行使用

```bash
# 使用 YAML 配置训练
lerobot-train \
    --config_path=lerobot/configs/policy/act.yaml \
    --robot.type=agilex \
    --dataset.repo_id=user/agilex_demo

# 使用 YAML 配置评估
lerobot-eval \
    --policy.path=outputs/train/act_agilex/checkpoints/last \
    --robot.type=agilex \
    --env.type=real

# 数据采集
lerobot-record \
    --robot.type=agilex \
    --teleop.type=agilex_teleop \
    --dataset.repo_id=user/agilex_demo \
    --fps=30
```

### 8.4 Python API 使用

```python
from lerobot.robots.agilex import AgileXConfig, AgileXRobot

# 创建配置
config = AgileXConfig(
    ros_node_name="my_robot",
    image_topics={
        "cam_left_wrist": "/camera_l/color/image_raw",
        "cam_right_wrist": "/camera_r/color/image_raw",
    },
    max_relative_target=0.3,
)

# 创建机器人实例
robot = AgileXRobot(config)

# 连接
robot.connect()

# 获取观测
obs = robot.get_observation()
print(f"Left arm joint 0: {obs['left_joint0.pos']}")
print(f"Image shape: {obs['cam_left_wrist'].shape}")

# 发送动作
action = {
    "left_joint0.pos": 0.1,
    "left_joint1.pos": 0.2,
    # ... 其他关节
}
robot.send_action(action)

# 断开
robot.disconnect()
```

### 8.5 第三方插件机制

如果希望将 Agilex 集成作为独立包发布，可以使用 `pyproject.toml` 的 entry-points：

```toml
# pyproject.toml
[project.entry-points."lerobot.robots"]
agilex = "lerobot_agilex:AgileXConfig"

[project.entry-points."lerobot.teleoperators"]
agilex_teleop = "lerobot_agilex:AgileXTeleoperatorConfig"
```

---

## 9. 测试指南

### 9.1 测试策略

| 测试类型 | 目的 | 依赖 |
|----------|------|------|
| **单元测试** | 验证各组件逻辑 | Mock ROS |
| **集成测试** | 验证组件协作 | Mock ROS |
| **硬件测试** | 验证真实硬件 | ROS + 硬件 |
| **端到端测试** | 验证完整流程 | 全部 |

### 9.2 单元测试示例

创建测试文件 `tests/robots/test_agilex.py`：

```python
#!/usr/bin/env python
"""Agilex 机器人单元测试"""

import pytest
import numpy as np

from lerobot.robots.agilex import AgileXConfig, AgileXRobot
from lerobot.robots.agilex.ros_bridge import MockAgileXROSBridge


class TestAgileXConfig:
    """测试配置类"""

    def test_default_config(self):
        """测试默认配置"""
        config = AgileXConfig()

        assert config.ros_node_name == "lerobot_agilex"
        assert config.max_relative_target == 0.5
        assert config.frame_rate == 30
        assert config.use_robot_base is False

    def test_custom_config(self):
        """测试自定义配置"""
        config = AgileXConfig(
            ros_node_name="custom_node",
            max_relative_target=0.3,
            use_robot_base=True,
        )

        assert config.ros_node_name == "custom_node"
        assert config.max_relative_target == 0.3
        assert config.use_robot_base is True


class TestAgileXRobot:
    """测试 Robot 类"""

    @pytest.fixture
    def mock_robot(self, monkeypatch):
        """创建使用 Mock ROS 的机器人实例"""
        config = AgileXConfig(
            image_topics={
                "cam_left_wrist": "/camera_l/color/image_raw",
            }
        )
        robot = AgileXRobot(config)

        # 替换 ROS 桥接层为 Mock
        robot.ros_bridge = MockAgileXROSBridge(
            image_topics=config.image_topics
        )

        return robot

    def test_observation_features(self, mock_robot):
        """测试观测特征定义"""
        features = mock_robot.observation_features

        # 检查关节特征
        assert "left_joint0.pos" in features
        assert "right_gripper.pos" in features
        assert features["left_joint0.pos"] == float

        # 检查相机特征
        assert "cam_left_wrist" in features
        assert features["cam_left_wrist"] == (480, 640, 3)

    def test_action_features(self, mock_robot):
        """测试动作特征定义"""
        features = mock_robot.action_features

        assert "left_joint0.pos" in features
        assert "right_gripper.pos" in features
        assert len(features) == 14  # 7 joints × 2 arms

    def test_get_observation(self, mock_robot):
        """测试获取观测"""
        obs = mock_robot.get_observation()

        # 检查关节数据
        assert "left_joint0.pos" in obs
        assert isinstance(obs["left_joint0.pos"], float)

        # 检查图像数据
        assert "cam_left_wrist" in obs
        assert obs["cam_left_wrist"].shape == (480, 640, 3)

    def test_send_action(self, mock_robot):
        """测试发送动作"""
        action = {
            "left_joint0.pos": 0.1,
            "left_joint1.pos": 0.2,
            "right_joint0.pos": -0.1,
        }

        sent_action = mock_robot.send_action(action)

        # 检查返回的动作
        assert "left_joint0.pos" in sent_action
        assert sent_action["left_joint0.pos"] == 0.1


class TestAgileXROSBridge:
    """测试 ROS 桥接层"""

    def test_mock_bridge(self):
        """测试 Mock 桥接层"""
        bridge = MockAgileXROSBridge(
            image_topics={"cam_test": "/test/image"}
        )

        assert bridge.is_connected
        assert bridge.has_data()

        # 测试关节状态
        state = bridge.get_joint_state()
        assert "puppet_left" in state
        assert len(state["puppet_left"]) == 7

        # 测试图像
        images = bridge.get_images()
        assert "cam_test" in images
        assert images["cam_test"].shape == (480, 640, 3)

    def test_mock_bridge_shutdown(self):
        """测试关闭"""
        bridge = MockAgileXROSBridge()
        bridge.shutdown()

        assert not bridge.is_connected
```

### 9.3 运行测试

```bash
# 运行所有 Agilex 测试
pytest tests/robots/test_agilex.py -v

# 运行特定测试
pytest tests/robots/test_agilex.py::TestAgileXRobot::test_get_observation -v

# 带覆盖率
pytest tests/robots/test_agilex.py --cov=lerobot.robots.agilex --cov-report=term
```

### 9.4 硬件测试

在有真实硬件的环境中进行测试：

```python
#!/usr/bin/env python
"""Agilex 硬件测试脚本"""

import time
from lerobot.robots.agilex import AgileXConfig, AgileXRobot


def test_hardware_connection():
    """测试硬件连接"""
    config = AgileXConfig(
        image_topics={
            "cam_left_wrist": "/camera_l/color/image_raw",
            "cam_right_wrist": "/camera_r/color/image_raw",
        }
    )

    robot = AgileXRobot(config)

    try:
        # 连接
        print("Connecting to robot...")
        robot.connect()
        print("Connected!")

        # 读取观测
        print("\nReading observations...")
        for i in range(10):
            obs = robot.get_observation()
            print(f"Frame {i}: left_joint0={obs['left_joint0.pos']:.3f}")
            time.sleep(0.1)

        print("\nHardware test passed!")

    finally:
        robot.disconnect()


if __name__ == "__main__":
    test_hardware_connection()
```

### 9.5 端到端测试

```bash
# 1. 启动 ROS 节点
roscore &
roslaunch piper start_ms_piper.launch mode:=0 &
roslaunch astra_camera multi_camera.launch &

# 2. 运行数据采集测试
lerobot-record \
    --robot.type=agilex \
    --teleop.type=agilex_teleop \
    --dataset.repo_id=test/agilex_e2e \
    --dataset.num_episodes=1 \
    --fps=30

# 3. 验证数据集
lerobot-dataset-viz --repo-id test/agilex_e2e --episode-index 0
```

---

## 10. 常见问题与调试

### 10.1 问题排查表

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| **ROS 节点无法连接** | roscore 未启动 | 运行 `roscore` |
| **Topic 无数据** | 驱动未启动 | 运行 `roslaunch piper start_ms_piper.launch` |
| **图像全黑** | 相机未连接/驱动问题 | 检查 USB 连接，运行 `roslaunch astra_camera` |
| **关节位置异常** | CAN 配置错误 | 运行 `./can_config.sh` |
| **动作不执行** | mode 参数错误 | 确保 `mode:=0`（遥操作模式） |
| **超时错误** | 网络延迟/节点崩溃 | 检查 ROS 日志，重启节点 |
| **导入错误** | 依赖未安装 | `pip install rospkg catkin_pkg` |
| **权限错误** | 串口权限不足 | `sudo usermod -aG dialout $USER` |

### 10.2 ROS 调试命令

```bash
# 检查 ROS 环境
echo $ROS_MASTER_URI
echo $ROS_IP

# 列出所有 Topic
rostopic list

# 查看 Topic 数据
rostopic echo /puppet/joint_left
rostopic echo /camera_l/color/image_raw --noarr

# 查看 Topic 频率
rostopic hz /puppet/joint_left

# 查看节点列表
rosnode list

# 查看节点信息
rosnode info /piper_node

# 查看 ROS 日志
roscd log
tail -f latest/*.log
```

### 10.3 Python 调试

```python
import logging

# 启用详细日志
logging.basicConfig(level=logging.DEBUG)

# 检查 ROS 是否可用
try:
    import rospy
    print("ROS available")
except ImportError:
    print("ROS not available")

# 检查 Topic 是否有数据
import rospy
from sensor_msgs.msg import JointState

rospy.init_node("debug_node")

def callback(msg):
    print(f"Received: {msg.position}")

sub = rospy.Subscriber("/puppet/joint_left", JointState, callback)
rospy.spin()
```

### 10.4 性能调优

```python
# 1. 减少图像分辨率
config = AgileXConfig(
    image_topics={
        "cam_low_res": "/camera_l/color/image_raw",  # 使用低分辨率 Topic
    }
)

# 2. 降低采集频率
config = AgileXConfig(
    frame_rate=15,  # 从 30fps 降到 15fps
)

# 3. 增加缓冲区
# 在 ros_bridge.py 中调整 buffer_size
bridge = AgileXROSBridge(buffer_size=20)

# 4. 使用异步图像读取
# 图像已经通过 ROS 回调异步获取，无需额外处理
```

### 10.5 常见错误信息

| 错误信息 | 原因 | 解决方案 |
|----------|------|----------|
| `ROSException: roscore cannot run` | roscore 已运行或端口占用 | `killall roscore` |
| `Unable to register with master node` | ROS_MASTER_URI 错误 | `export ROS_MASTER_URI=http://localhost:11311` |
| `No module named 'rospy'` | ROS Python 未安装 | `source /opt/ros/noetic/setup.bash` |
| `CvBridgeError` | 图像编码不匹配 | 检查 `desired_encoding` 参数 |
| `TimeoutError: Timeout waiting for ROS data` | Topic 无数据 | 检查驱动是否启动 |

---

## 11. 最佳实践

### 11.1 代码风格

```python
# ✅ 好的做法
class AgileXRobot(Robot):
    """
    Agilex Piper 双臂机器人实现。

    详细的文档字符串，说明用途和使用方法。
    """

    def get_observation(self) -> dict[str, Any]:
        """获取观测，返回类型明确"""
        if not self.is_connected:
            raise RuntimeError("Robot not connected")
        # ...

# ❌ 避免的做法
class AgileXRobot(Robot):
    def get_observation(self):  # 缺少类型注解和文档
        return self.ros_bridge.get_joint_state()  # 缺少错误处理
```

### 11.2 错误处理

```python
# ✅ 好的做法
def connect(self, calibrate: bool = True) -> None:
    if self.is_connected:
        logger.warning("Robot already connected")
        return

    try:
        self.ros_bridge = AgileXROSBridge(...)
    except Exception as e:
        logger.error(f"Failed to connect: {e}")
        raise ConnectionError(f"Cannot connect to Agilex robot: {e}")

    # 等待数据，带超时
    timeout = 10.0
    start_time = time.time()
    while not self.ros_bridge.has_data():
        if time.time() - start_time > timeout:
            raise TimeoutError("Timeout waiting for ROS data")
        time.sleep(0.1)

# ❌ 避免的做法
def connect(self):
    self.ros_bridge = AgileXROSBridge(...)  # 无错误处理
    time.sleep(5)  # 硬编码等待时间
```

### 11.3 资源管理

```python
# ✅ 好的做法：使用上下文管理器
class AgileXRobot(Robot):
    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False

# 使用
with AgileXRobot(config) as robot:
    obs = robot.get_observation()
    # 自动断开连接

# ✅ 好的做法：确保资源释放
def disconnect(self) -> None:
    if self.ros_bridge is not None:
        try:
            if self.config.disable_torque_on_disconnect:
                self.ros_bridge.publish_base_command(0.0, 0.0)
        finally:
            self.ros_bridge.shutdown()
            self.ros_bridge = None
```

### 11.4 安全考虑

```python
# ✅ 好的做法：使用安全限幅
def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
    # 获取当前位置
    current_state = self.ros_bridge.get_joint_state()

    # 对每个关节进行安全限幅
    for joint_name in self.all_joints:
        goal = action.get(joint_name, 0.0)
        current = current_state.get(joint_name, 0.0)

        # 限制单步最大位移
        if self.config.max_relative_target is not None:
            max_delta = self.config.max_relative_target
            delta = goal - current
            if abs(delta) > max_delta:
                goal = current + max_delta * np.sign(delta)

        action[joint_name] = goal

    return action

# ✅ 好的做法：断开时安全停止
def disconnect(self) -> None:
    if self.config.disable_torque_on_disconnect:
        # 发送零速度指令
        self.ros_bridge.publish_base_command(0.0, 0.0)
```

### 11.5 日志规范

```python
import logging

logger = logging.getLogger(__name__)

# ✅ 好的做法
def connect(self) -> None:
    logger.info("Connecting to Agilex robot via ROS...")
    logger.debug(f"ROS node name: {self.config.ros_node_name}")

    try:
        self.ros_bridge = AgileXROSBridge(...)
        logger.info("Agilex robot connected successfully")
    except Exception as e:
        logger.error(f"Failed to connect: {e}")
        raise

def get_observation(self) -> dict[str, Any]:
    before_read_t = time.perf_counter()
    obs = self._read_sensors()
    dt_ms = (time.perf_counter() - before_read_t) * 1e3
    logger.debug(f"Read observation in {dt_ms:.1f}ms")
    return obs
```

### 11.6 测试友好性

```python
# ✅ 好的做法：支持依赖注入
class AgileXRobot(Robot):
    def __init__(self, config: AgileXConfig, ros_bridge: AgileXROSBridge | None = None):
        self.config = config
        self._ros_bridge = ros_bridge  # 允许注入 Mock

    @property
    def ros_bridge(self) -> AgileXROSBridge:
        if self._ros_bridge is None:
            raise RuntimeError("Not connected")
        return self._ros_bridge

    def connect(self) -> None:
        if self._ros_bridge is None:
            self._ros_bridge = AgileXROSBridge(...)

# 测试时可以注入 Mock
def test_robot():
    mock_bridge = MockAgileXROSBridge()
    robot = AgileXRobot(config, ros_bridge=mock_bridge)
    # 无需真实 ROS 连接即可测试
```

---

## 12. 附录

### 12.1 完整文件清单

| 文件路径 | 说明 |
|----------|------|
| `src/lerobot/robots/agilex/__init__.py` | 模块导出 |
| `src/lerobot/robots/agilex/config_agilex.py` | 配置类定义 |
| `src/lerobot/robots/agilex/robot_agilex.py` | Robot 子类实现 |
| `src/lerobot/robots/agilex/ros_bridge.py` | ROS 桥接层 |
| `src/lerobot/robots/agilex/constants.py` | 常量定义 |
| `src/lerobot/teleoperators/agilex/__init__.py` | 遥操作模块导出 |
| `src/lerobot/teleoperators/agilex/teleop_agilex.py` | 遥操作器实现 |
| `lerobot/configs/robot/agilex.yaml` | YAML 配置文件 |
| `tests/robots/test_agilex.py` | 单元测试 |

### 12.2 参考实现对照表

| 功能 | Agilex 实现 | 参考实现 |
|------|-------------|----------|
| 双臂架构 | `AgileXRobot` | `BiSO100Follower` |
| 外部 SDK 集成 | `AgileXROSBridge` | `Reachy2Robot` (ReachySDK) |
| 关节映射 | `constants.py` | `robot_reachy2.py` (REACHY2_*_JOINTS) |
| 配置类 | `AgileXConfig` | `BiSO100FollowerConfig` |
| 遥操作 | `AgileXTeleoperator` | `SO100Leader` |

### 12.3 核心类速查

#### AgileXConfig

```python
@dataclass
class AgileXConfig(RobotConfig):
    ros_node_name: str = "lerobot_agilex"
    master_left_topic: str = "/master/joint_left"
    master_right_topic: str = "/master/joint_right"
    puppet_left_topic: str = "/puppet/joint_left"
    puppet_right_topic: str = "/puppet/joint_right"
    use_robot_base: bool = False
    odom_topic: str = "/odom"
    max_relative_target: float | None = 0.5
    frame_rate: int = 30
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    image_topics: dict[str, str] = field(default_factory=dict)
    disable_torque_on_disconnect: bool = True
```

#### AgileXRobot

```python
class AgileXRobot(Robot):
    config_class = AgileXConfig
    name = "agilex"

    # Properties
    observation_features: dict[str, type | tuple]
    action_features: dict[str, type]
    is_connected: bool
    is_calibrated: bool  # Always True

    # Methods
    def connect(calibrate: bool = True) -> None
    def calibrate() -> None  # No-op
    def configure() -> None
    def get_observation() -> dict[str, Any]
    def send_action(action: dict) -> dict[str, Any]
    def disconnect() -> None
```

#### AgileXROSBridge

```python
class AgileXROSBridge:
    # Properties
    is_connected: bool

    # Methods
    def has_data() -> bool
    def get_joint_state() -> dict[str, list[float]]
    def get_images() -> dict[str, np.ndarray]
    def get_base_state() -> dict[str, float]
    def get_master_action() -> dict[str, list[float]]
    def publish_joint_command(left: list, right: list) -> None
    def publish_base_command(vx: float, vtheta: float) -> None
    def shutdown() -> None
```

### 12.4 关节名称映射

| 索引 | 原版名称 | LeRobot 左臂 | LeRobot 右臂 |
|------|----------|--------------|--------------|
| 0 | joint0 | `left_joint0.pos` | `right_joint0.pos` |
| 1 | joint1 | `left_joint1.pos` | `right_joint1.pos` |
| 2 | joint2 | `left_joint2.pos` | `right_joint2.pos` |
| 3 | joint3 | `left_joint3.pos` | `right_joint3.pos` |
| 4 | joint4 | `left_joint4.pos` | `right_joint4.pos` |
| 5 | joint5 | `left_joint5.pos` | `right_joint5.pos` |
| 6 | gripper | `left_gripper.pos` | `right_gripper.pos` |

### 12.5 ROS Topic 映射

| 原版 Topic | 用途 | LeRobot 使用 |
|------------|------|--------------|
| `/master/joint_left` | 左主臂状态 | 遥操作动作输入 |
| `/master/joint_right` | 右主臂状态 | 遥操作动作输入 |
| `/puppet/joint_left` | 左从臂状态 | 观测数据 |
| `/puppet/joint_right` | 右从臂状态 | 观测数据 |
| `/camera_l/color/image_raw` | 左腕相机 | 观测图像 |
| `/camera_r/color/image_raw` | 右腕相机 | 观测图像 |
| `/camera_f/color/image_raw` | 前方相机 | 观测图像 |
| `/odom` | 底盘里程计 | 底盘状态（可选） |

### 12.6 相关链接

| 资源 | 链接 |
|------|------|
| **LeRobot 官方文档** | https://github.com/huggingface/lerobot |
| **Agilex 官方网站** | https://www.agilex.ai/ |
| **ROS Noetic 文档** | http://wiki.ros.org/noetic |
| **Piper 机械臂文档** | (内部文档) |
| **HuggingFace Hub** | https://huggingface.co/datasets |

### 12.7 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 1.0.0 | 2025-12-02 | 初始版本，支持双臂遥操作和数据采集 |

---

## 总结

本文档详细介绍了如何将 Agilex Piper 双臂机器人集成到 LeRobot 框架中。主要内容包括：

1. **架构设计**：采用 ROS 桥接层封装原版驱动，保持与原版系统的兼容性
2. **核心实现**：`AgileXConfig`、`AgileXRobot`、`AgileXROSBridge` 三个核心类
3. **遥操作支持**：`AgileXTeleoperator` 实现主臂数据采集
4. **测试方案**：Mock 实现支持无硬件测试
5. **最佳实践**：代码风格、错误处理、安全考虑等

通过本集成，可以使用 LeRobot 的标准工具链（`lerobot-record`、`lerobot-train`、`lerobot-eval`）进行数据采集、模型训练和策略评估，同时保持与 Agilex 原版系统的兼容性。

如有问题，请参考 [常见问题与调试](#10-常见问题与调试) 章节，或联系开发团队。
