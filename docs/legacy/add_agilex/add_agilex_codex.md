# Agilex 机器人集成方案（Codex）

> **兼容性说明 (lerobot 0.4.3 baseline)**  
> 本文档依赖旧仓库中的 AgileX/ROS Camera 实现（例如 `lerobot.robots.agilex`、`lerobot.cameras.ros_camera`）。  
> 当前仓库的 0.4.3 baseline 默认代码树不包含这些模块，因此本文不能直接照着跑；建议仅作为历史记录/移植参考（路线 B）。  
>
> 面向需要在 LeRobot 中集成 Agilex Piper 双臂机器人的开发者，本文档给出了从原版 ROS API 迁移到 `Robot` 抽象的完整方案，包括架构、实现细节、代码样例、测试、部署与排障建议。所有内容均参考 `aiglex_origin_code/*.py` 及 `docs/通用添加机器人.md`。

## 目录
1. [Agilex 机器人概述](#1-agilex-机器人概述)
2. [原版 API 与 LeRobot 接口差异](#2-原版-api-与-lerobot-接口差异)
3. [集成架构设计](#3-集成架构设计)
4. [详细实现步骤](#4-详细实现步骤)
5. [代码示例](#5-代码示例)
6. [测试方案](#6-测试方案)
7. [常见问题与调试](#7-常见问题与调试)
8. [部署与使用](#8-部署与使用)

---

## 1. Agilex 机器人概述

### 1.1 硬件规格与拓扑

| 模块 | 规格 / 接口 | 说明 |
|------|-------------|------|
| 双臂执行器 | Piper 协作臂，每臂 7-DOF（6 旋转关节 + 1 夹爪） | 主臂 (Leader) + 从臂 (Follower) 组成两主两从拓扑 |
| 移动底盘 | 轮式底盘，接受 `cmd_vel` (`geometry_msgs/Twist`) | 采集脚本中仅记录 `linear_x` 、`angular_z` |
| 电机通信 | CAN 总线，通过 `./can_config.sh` 初始化 | `roslaunch piper start_ms_piper.launch` 将 CAN 转为 ROS Topic |
| 上位机 | ROS Noetic，Python 3.8 / Conda 环境 | 所有 SDK 行为通过 Topic 完成，无直接 SDK API |
| 相机 | `astra_camera multi_camera.launch` 管理的三路 RGB-D | `/camera_l`, `/camera_r`, `/camera_f` (可选深度) |
| 数据格式 | HDF5 (`observations/*`, `action`, `base_action`) | 采集脚本 `collect_data.py` 输出 |

**关键特征**：
- 采集与回放全部通过 ROS Topic 完成；动作由主臂 JointState 决定，从臂 JointState 作为观测。
- 每臂 7 维角度向量（包含夹爪），因此观测/动作均为 14 维。
- 数据集包含三路 RGB（可选三路深度）图像，并记录底盘速度，便于多模态策略训练。

### 1.2 原版运行流程

1. **启动底层驱动**（终端 1）：
   ```bash
   roscore
   ```
2. **启动 Piper ROS 驱动**（终端 2）：
   ```bash
   conda activate act
   cd /home/agilex/cobot_magic/Piper_ros_private-ros-noetic
   ./can_config.sh
   source devel/setup.bash
   roslaunch piper start_ms_piper.launch mode:=0 auto_enable:=false  # 采集模式
   ```
3. **启动相机**（终端 3）：
   ```bash
   roslaunch astra_camera multi_camera.launch
   ```
4. （可选）检查 Topic：`rostopic list`。
5. **采集**：`python collect_data.py --dataset_dir ./data --task_name Dual_arm_manipulation --max_timesteps 400 --episode_idx 0`。
6. **回放 / 推理**：切换 `mode:=1 auto_enable:=true`，可运行 `python replay_data.py --only_pub_master` 或模型推理脚本。

### 1.3 ROS Topic 与消息

| Topic | 类型 | 方向 | 用途 |
|-------|------|------|------|
| `/master/joint_left` / `/master/joint_right` | `sensor_msgs/JointState` | 发布 | 作为控制输入（主臂） |
| `/puppet/joint_left` / `/puppet/joint_right` | `sensor_msgs/JointState` | 订阅 | 作为反馈（从臂） |
| `/camera_l/color/image_raw` 等 | `sensor_msgs/Image` | 订阅 | 左/右/前彩色图像 |
| `/camera_*/depth/image_raw` | `sensor_msgs/Image` | 订阅 | 深度（可选） |
| `/odom` | `nav_msgs/Odometry` | 订阅 | 底盘速度观测 |
| `/cmd_vel` | `geometry_msgs/Twist` | 发布 | 底盘目标速度 |

### 1.4 数据格式与命令

- **观测**：`/observations/qpos`、`/qvel`、`/effort`（shape `[T,14]`），`/observations/images/<cam>`（`[T,480,640,3]`）。
- **动作**：`/action`（主臂 14 维），`/base_action`（2 维速度）。
- **采集命令**：见 1.2 第 5 步，`--use_depth_image`、`--use_robot_base` 可扩展观测。原脚本以 `dm_env.TimeStep` 结构缓存数据。
- **可视化 / 回放**：`visualize_episodes.py` 合并三路视频并绘制关节曲线；`replay_data.py` 会插值 `JointState` 并支持只发布主臂或双臂。

---

## 2. 原版 API 与 LeRobot 接口差异

| 维度 | 原版实现 | LeRobot 约束 | 适配方案 |
|------|----------|--------------|----------|
| 观测结构 | `numpy` 向量 + `dict`（HDF5） | `Robot.get_observation()` 返回平坦字典，键需声明在 `observation_features` | 将 `JointState` 拆成 `left_arm.joint0.pos` 等键，图像映射为 `(H,W,C)`，底盘速度映射为 `base.linear_vel`/`base.angular_vel` |
| 动作接口 | 直接向 `/master/joint_*` 发布 `JointState` 数组 | `send_action()` 接收 dict，并支持动作限幅、返回实际命令 | 在 `AgileXRobot` 中解析 dict -> numpy，使用 `ensure_safe_goal_position()` 限幅后推送 ROS Topic |
| 通信协议 | ROS Topic + Piper 驱动（CAN） | `Robot` 抽象不关心底层，但要求 `connect/disconnect` 生命周期 | 实现 `AgileXRosBridge`，封装 ROS 连接、Topic 订阅/发布、Watchdog |
| 相机 | ROS 图像消息 | `Camera` 子类（OpenCV/RealSense/自定义） | 实现 `RosImageCamera` 或将 ROS 图像桥接至共享内存，再由 `Camera` 读取；文本示例沿用 `make_cameras_from_configs` |
| 配置 | `argparse` 参数散落在脚本 | `RobotConfig` + `draccus` 驱动 CLI / YAML | 定义 `AgileXConfig`，集中描述 Topic、限幅、相机、ROS 连接信息，并注册到 `RobotConfig` |
| 校准 | 依赖 Piper 驱动初始零位，无持久化 | `Robot.calibration` 自动读写 JSON | `calibrate()` 通过读取当前从臂位姿、询问用户确认并写入 `HF_LEROBOT_CALIBRATION/robots/agilex/<id>.json` |
| 遥操作 | `master` Topic + `collect_data.py` | 可实现 `Teleoperator`，action_features 必须与 robot 对齐 | 创建 `AgileXTeleoperator` 订阅 `/master/joint_*`，以 `get_action()` 向策略层提供实时动作 |
| 数据存储 | 自定义 HDF5 | LeRobot Dataset（zarr/Parquet） | 在 `data_pipeline` 中添加 converter，将原 HDF5 字段映射为标准 `observation_features/action_features` |

---

## 3. 集成架构设计

### 3.1 类图（抽象）

```
RobotConfig.register_subclass("agilex")
┌───────────────────────────────────────────┐
│              AgileXConfig                 │
│- rosbridge_host/port                      │
│- topic_map (master/puppet/cameras/base)   │
│- cameras: dict[str, CameraConfig]         │
│- safety: max_relative_target, watchdog    │
└───────────────────────────────────────────┘
                 │
                 ▼
┌───────────────────────────────────────────┐      ┌─────────────────────┐
│              AgileXRobot                 │─────▶│  AgileXRosBridge     │
│ connect/compute obs/send_action          │◀─────│  订阅/发布 ROS Topic │
│  ├─ caches calibration/home pose         │      └─────────────────────┘
│  ├─ 调用 make_cameras_from_configs        │
│  └─ 可组合 AgileXTeleoperator             │
└───────────────────────────────────────────┘
                 │
                 ▼
         LeRobot 训练 / 采集脚本
```

### 3.2 数据流

```
策略/采集脚本
    │ actions(dict)
    ▼
AgileXRobot.send_action()
    │ 解析 + 限幅
    ▼
AgileXRosBridge.publish()
    │                         观测流向相反
    ▼                         ┌─────────────────────────┐
ROS Topic (/master, /cmd_vel) │  Piper 节点 + CAN 总线  │
                              └─────────────────────────┘
                                      │
                                      ▼
                               双臂执行器 + 底盘
```

### 3.3 接口映射表

**观测**

| LeRobot 键 | 来源 | 维度/备注 |
|------------|------|-----------|
| `left_arm.joint{i}.pos` (i=0~6) | `/puppet/joint_left.position[i]` | `float`，弧度 |
| `left_arm.joint{i}.vel` | `/puppet/joint_left.velocity[i]` | `float` |
| `left_arm.joint{i}.effort` | `/puppet/joint_left.effort[i]` | `float` |
| `right_arm.*` | `/puppet/joint_right.*` | 同上 |
| `base.linear_vel` | `/odom.twist.twist.linear.x` | `float` |
| `base.angular_vel` | `/odom.twist.twist.angular.z` | `float` |
| `cam_front` | `/camera_f/color/image_raw` | `(480, 640, 3)`，RGB |
| `cam_left` / `cam_right` | `/camera_l` / `/camera_r` | `(480, 640, 3)` |
| `cam_*_depth`（可选） | `/camera_*/depth/image_raw` | `(480, 640)`，`uint16` |

**动作**

| LeRobot 键 | 下发到 | 说明 |
|------------|--------|------|
| `left_arm.joint{i}.pos` | `/master/joint_left.position[i]` | 期望位置，弧度 |
| `right_arm.joint{i}.pos` | `/master/joint_right.position[i]` | 期望位置 |
| `base.linear_cmd` | `/cmd_vel.linear.x` | 单位 m/s，限制在 `[-v_max, v_max]` |
| `base.angular_cmd` | `/cmd_vel.angular.z` | 单位 rad/s |

---

## 4. 详细实现步骤

### 4.1 定义 AgileXConfig

1. 继承 `RobotConfig` 并注册：`@RobotConfig.register_subclass("agilex")`。
2. 字段建议：
   - `rosbridge_host`, `rosbridge_port`, `namespace`（默认 `/`）。
   - `master_topics` / `puppet_topics` / `camera_topics` / `odom_topic` / `cmd_vel_topic`。
   - `max_relative_target`: `float | dict[str, float]`。
   - `command_rate_hz`, `watchdog_timeout_s`, `image_timeout_s`。
   - `cameras: dict[str, CameraConfig]`，默认提供 front/left/right。
   - `enable_depth: bool`、`base_limits: tuple[float, float]`。
3. `__post_init__` 中校验 topic 组合、相机分辨率，并允许关闭某些传感器。

### 4.2 实现 AgileXRobot

- `name = "agilex"`, `config_class = AgileXConfig`。
- 在 `__init__` 中：
  - 创建 `AgileXRosBridge` 实例；
  - `self.cameras = make_cameras_from_configs(config.cameras)`；
  - 解析 `joint_names` 并生成 `_obs_features`、`_action_features`。
- `connect()`：
  1. 启动 `AgileXRosBridge.connect()` 并等待 `/rosout` 心跳；
  2. 连接所有相机；
  3. 若无校准文件则调用 `calibrate()`，否则加载并写入桥接层；
  4. 调用 `configure()` 设置 Watchdog、速度限制。
- `is_connected`：检查 `self.bridge.is_connected` 且全部相机 `is_connected`。
- `calibrate()`：
  - 提示用户将臂移动至中立位；
  - 采样若干帧 `JointState` 求平均，生成 `MotorCalibration`（或自定义结构）并持久化；
  - 写入 `self.bridge.update_offsets()`，保证读数减去零偏。
- `configure()`：设置 `self.bridge.base_limits`, `command_rate_hz` 等；
- `get_observation()`：
  - 调用 `bridge.get_state()`，组装 `pos/vel/effort/base`；
  - 相机调用 `cam.async_read()` / `cam.async_read_depth()`；
  - 返回平坦 dict。
- `send_action()`：
  - 将动作拆解为左右臂数组 + 底盘速度；
  - 读取当前位姿并调用 `ensure_safe_goal_position()` 限幅；
  - `bridge.publish_action(left, right, base)`；
  - 返回实际发送的动作 dict，供数据记录。
- `disconnect()`：停止桥接层、断开相机、清理线程。

### 4.3 原版 API 封装（AgileXRosBridge）

1. 依赖 `roslibpy` 或 `rclpy`，统一使用 websocket `rosbridge` 以便跨平台。
2. 负责：
   - 订阅 `/puppet/joint_*`、`/odom`、相机 Topic；
   - 将最新消息缓存到线程安全队列（`collections.deque`）；
   - 发布 `/master/joint_*`、`/cmd_vel`，可附带平滑插值；
   - Watchdog：若 `send_action` 超过 `watchdog_timeout_s` 未被调用，则自动下发零速并记录警告；
   - 将 `sensor_msgs/Image` 转 `numpy`（可共用 `cv_bridge` 或自定义解码）。
3. 提供 `get_state()` / `publish_action()` / `shutdown()` 等同步方法，被 `AgileXRobot` 调用。

### 4.4 电机控制集成

- Piper ROS 节点负责 CAN 通讯，因此 LeRobot 只需面向 ROS 层：
  - 将 `master` Topic 视为 “虚拟总线” 的写寄存器；
  - 将 `puppet` Topic 视为读寄存器；
  - 在 `AgileXRosBridge` 中增加 `smooth_factor`，对相邻动作做线性插值（类似 `replay_data.py` 中的 20 段插值），降低抖动；
  - 通过 `max_relative_target`、`joint_velocity_limit` 保护从臂安全；
  - 当底盘命令超过 `base_limits` 时自动裁剪。

### 4.5 相机系统集成

- **ROS 图像**：实现 `RosImageCamera`（继承 `Camera`），内部绑定某个 Topic，`connect()` 时订阅，`async_read()` 返回最新 numpy 图像。
- **混合方案**：如果使用 RealSense USB，可直接配置 `RealSenseCameraConfig`，否则在 config 中为每路 ROS 相机提供 `RosImageCameraConfig`。
- **时间同步**：对齐原版 `get_frame()` 的“最小时间戳”逻辑，可在 `RosImageCamera` 内记录 header 并仅返回满足 `stamp >= t` 的帧。

### 4.6 校准流程设计

1. 进入示教模式（Piper `auto_enable:=false`），禁用底盘运动。
2. 用户将双臂移动到中立位 -> `bridge.capture_state()`。
3. 将零位与 `range_min/max` 写入 `MotorCalibration` 或 JSON，自定义字段 `joint_limits_rad`。
4. `send_action()` 根据校准信息做归一化/裁剪，`get_observation()` 也可输出相对于零位的值。
5. 校准文件保存在 `HF_LEROBOT_CALIBRATION/robots/agilex/<id>.json`，切换硬件时仅需更新 `id`。

### 4.7 遥操作与闭环采集

- 新增 `src/lerobot/teleoperators/agilex_leader/`：
  - `AgileXTeleoperatorConfig`：包含 `/master/joint_*` topic、限幅、平滑参数；
  - `AgileXTeleoperator`：订阅主臂 `JointState` 并实现 `get_action()`；
  - 这样可以使用 `lerobot/scripts/teleop_collect.py --robot.type agilex --teleop.type agilex_leader` 实时采集。
- 遥操作与 robot 共享 `observation_features/action_features`，便于数据对齐。

---

## 5. 代码示例

以下示例演示 `AgileXConfig` + `AgileXRosBridge` + `AgileXRobot` 的最小实现（依赖 `roslibpy`、`numpy`）。

```python
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np
import roslibpy

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

logger = logging.getLogger(__name__)
ARM_JOINTS = ("joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "gripper")


class AgileXRosBridge:
    """ROS Topic 适配层，负责缓存 JointState/图像并发布命令。"""

    def __init__(
        self,
        host: str,
        port: int,
        master_topics: dict[str, str],
        puppet_topics: dict[str, str],
        odom_topic: str,
        cmd_vel_topic: str,
    ) -> None:
        self.client = roslibpy.Ros(host=host, port=port)
        self.master_topics = master_topics
        self.puppet_topics = puppet_topics
        self.odom_topic = odom_topic
        self.cmd_vel_topic = cmd_vel_topic
        self._last_action_ts = 0.0
        self.command_rate_hz = 30
        self.watchdog_timeout_s = 0.5
        self._state = {
            "left": {"pos": np.zeros(len(ARM_JOINTS)), "vel": np.zeros(len(ARM_JOINTS)), "effort": np.zeros(len(ARM_JOINTS))},
            "right": {"pos": np.zeros(len(ARM_JOINTS)), "vel": np.zeros(len(ARM_JOINTS)), "effort": np.zeros(len(ARM_JOINTS))},
            "base": np.zeros(2),
        }
        self._queues: dict[str, deque] = {topic: deque(maxlen=2) for topic in puppet_topics.values()}
        self._odom_queue = deque(maxlen=2)
        self._publishers: dict[str, roslibpy.Topic] = {}
        self._subscribers: list[roslibpy.Topic] = []

    # ---- life-cycle ----------------------------------------------------
    def connect(self) -> None:
        self.client.run()
        self.client.on_ready(self._register_topics)

    def _register_topics(self) -> None:
        # 发布者
        for side in ("left", "right"):
            topic = self.master_topics[side]
            self._publishers[side] = roslibpy.Topic(self.client, topic, "sensor_msgs/JointState")
            self._publishers[side].advertise()
        self._publishers["base"] = roslibpy.Topic(self.client, self.cmd_vel_topic, "geometry_msgs/Twist")
        self._publishers["base"].advertise()

        # 订阅者
        for side in ("left", "right"):
            topic = self.puppet_topics[side]
            sub = roslibpy.Topic(self.client, topic, "sensor_msgs/JointState")
            sub.subscribe(lambda msg, s=side: self._handle_joint_state(msg, s))
            self._subscribers.append(sub)
        odom_sub = roslibpy.Topic(self.client, self.odom_topic, "nav_msgs/Odometry")
        odom_sub.subscribe(self._handle_odom)
        self._subscribers.append(odom_sub)

    def shutdown(self) -> None:
        for sub in self._subscribers:
            sub.unsubscribe()
        for pub in self._publishers.values():
            pub.unadvertise()
        if self.client.is_connected:
            self.client.terminate()

    @property
    def is_connected(self) -> bool:
        return self.client.is_connected

    # ---- callbacks -----------------------------------------------------
    def _handle_joint_state(self, msg: dict, side: str) -> None:
        positions = np.array(msg.get("position", []), dtype=np.float32)
        velocities = np.array(msg.get("velocity", []), dtype=np.float32)
        efforts = np.array(msg.get("effort", []), dtype=np.float32)
        self._state[side] = {
            "pos": positions,
            "vel": velocities,
            "effort": efforts,
        }

    def _handle_odom(self, msg: dict) -> None:
        twist = msg.get("twist", {}).get("twist", {})
        self._state["base"] = np.array([twist.get("linear", {}).get("x", 0.0), twist.get("angular", {}).get("z", 0.0)], dtype=np.float32)

    # ---- public helpers ------------------------------------------------
    def get_state(self) -> dict[str, Any]:
        return self._state

    def publish_action(self, left: np.ndarray, right: np.ndarray, base_cmd: tuple[float, float]) -> None:
        stamp = roslibpy.Time.now()
        joint_msg = {
            "header": {"stamp": stamp.to_dict()},
            "name": list(ARM_JOINTS),
            "position": left.tolist(),
        }
        self._publishers["left"].publish(joint_msg)
        joint_msg["position"] = right.tolist()
        self._publishers["right"].publish(joint_msg)
        twist_msg = {
            "linear": {"x": float(base_cmd[0]), "y": 0.0, "z": 0.0},
            "angular": {"x": 0.0, "y": 0.0, "z": float(base_cmd[1])},
        }
        self._publishers["base"].publish(twist_msg)
        self._last_action_ts = time.monotonic()

    def watchdog(self) -> None:
        if time.monotonic() - self._last_action_ts > self.watchdog_timeout_s:
            self.publish_action(self._state["left"]["pos"], self._state["right"]["pos"], (0.0, 0.0))


@RobotConfig.register_subclass("agilex")
@dataclass
class AgileXConfig(RobotConfig):
    rosbridge_host: str = "127.0.0.1"
    rosbridge_port: int = 9090
    master_topics: dict[str, str] = field(
        default_factory=lambda: {"left": "/master/joint_left", "right": "/master/joint_right"}
    )
    puppet_topics: dict[str, str] = field(
        default_factory=lambda: {"left": "/puppet/joint_left", "right": "/puppet/joint_right"}
    )
    odom_topic: str = "/odom"
    cmd_vel_topic: str = "/cmd_vel"
    command_rate_hz: int = 30
    max_relative_target: float | None = 0.25  # 单步最大角度增量 (rad)
    base_speed_limits: tuple[float, float] = (0.5, 1.0)
    cameras: dict[str, Any] = field(
        default_factory=lambda: {
            "cam_front": RealSenseCameraConfig("front", fps=30, width=640, height=480),
            "cam_left": OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=480),
            "cam_right": OpenCVCameraConfig(index_or_path=1, fps=30, width=640, height=480),
        }
    )


class AgileXRobot(Robot):
    config_class = AgileXConfig
    name = "agilex"

    def __init__(self, config: AgileXConfig) -> None:
        super().__init__(config)
        self.config = config
        self.bridge = AgileXRosBridge(
            host=config.rosbridge_host,
            port=config.rosbridge_port,
            master_topics=config.master_topics,
            puppet_topics=config.puppet_topics,
            odom_topic=config.odom_topic,
            cmd_vel_topic=config.cmd_vel_topic,
        )
        self.cameras = make_cameras_from_configs(config.cameras)
        self._calibrated = bool(self.calibration)

    # ---- features ------------------------------------------------------
    @cached_property
    def observation_features(self) -> dict[str, Any]:
        features: dict[str, Any] = {}
        for side in ("left", "right"):
            for joint in ARM_JOINTS:
                base_key = f"{side}_arm.{joint}"
                features[f"{base_key}.pos"] = float
                features[f"{base_key}.vel"] = float
                features[f"{base_key}.effort"] = float
        features["base.linear_vel"] = float
        features["base.angular_vel"] = float
        for cam, cfg in self.config.cameras.items():
            features[cam] = (cfg.height, cfg.width, 3)
        return features

    @cached_property
    def action_features(self) -> dict[str, Any]:
        features: dict[str, Any] = {}
        for side in ("left", "right"):
            for joint in ARM_JOINTS:
                features[f"{side}_arm.{joint}.pos"] = float
        features["base.linear_cmd"] = float
        features["base.angular_cmd"] = float
        return features

    # ---- state ---------------------------------------------------------
    @property
    def is_connected(self) -> bool:
        return self.bridge.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        self.bridge.connect()
        for cam in self.cameras.values():
            cam.connect()
        if calibrate and not self.is_calibrated:
            self.calibrate()
        self.configure()
        logger.info("AgileX bridge connected")

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def calibrate(self) -> None:
        logger.info("Capturing zero pose from puppet joints…")
        state = self.bridge.get_state()
        self.calibration = {f"{side}.{joint}": state[side]["pos"][idx] for side in ("left", "right") for idx, joint in enumerate(ARM_JOINTS)}
        self._save_calibration()
        self._calibrated = True

    def configure(self) -> None:
        self.bridge.command_rate_hz = self.config.command_rate_hz
        logger.info("Set command rate to %shz", self.config.command_rate_hz)

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        state = self.bridge.get_state()
        obs: dict[str, Any] = {}
        for side in ("left", "right"):
            for idx, joint in enumerate(ARM_JOINTS):
                base_key = f"{side}_arm.{joint}"
                obs[f"{base_key}.pos"] = float(state[side]["pos"][idx])
                obs[f"{base_key}.vel"] = float(state[side]["vel"][idx])
                obs[f"{base_key}.effort"] = float(state[side]["effort"][idx])
        obs["base.linear_vel"] = float(state["base"][0])
        obs["base.angular_vel"] = float(state["base"][1])
        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        left_goal = np.array([action[f"left_arm.{joint}.pos"] for joint in ARM_JOINTS], dtype=np.float32)
        right_goal = np.array([action[f"right_arm.{joint}.pos"] for joint in ARM_JOINTS], dtype=np.float32)
        base_cmd = (
            np.clip(action.get("base.linear_cmd", 0.0), -self.config.base_speed_limits[0], self.config.base_speed_limits[0]),
            np.clip(action.get("base.angular_cmd", 0.0), -self.config.base_speed_limits[1], self.config.base_speed_limits[1]),
        )
        if self.config.max_relative_target is not None:
            present = self.bridge.get_state()
            goal_present = {
                f"left_arm.{joint}.pos": (float(left_goal[idx]), float(present["left"]["pos"][idx]))
                for idx, joint in enumerate(ARM_JOINTS)
            }
            goal_present.update(
                {
                    f"right_arm.{joint}.pos": (float(right_goal[idx]), float(present["right"]["pos"][idx]))
                    for idx, joint in enumerate(ARM_JOINTS)
                }
            )
            clipped = ensure_safe_goal_position(goal_present, self.config.max_relative_target)
            left_goal = np.array([clipped[f"left_arm.{joint}.pos"] for joint in ARM_JOINTS])
            right_goal = np.array([clipped[f"right_arm.{joint}.pos"] for joint in ARM_JOINTS])
        self.bridge.publish_action(left_goal, right_goal, base_cmd)
        sent = {
            **{f"left_arm.{joint}.pos": float(left_goal[idx]) for idx, joint in enumerate(ARM_JOINTS)},
            **{f"right_arm.{joint}.pos": float(right_goal[idx]) for idx, joint in enumerate(ARM_JOINTS)},
            "base.linear_cmd": base_cmd[0],
            "base.angular_cmd": base_cmd[1],
        }
        return sent

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        for cam in self.cameras.values():
            cam.disconnect()
        self.bridge.shutdown()
        logger.info("AgileX robot disconnected")
```

> 说明：示例中 `RealSenseCameraConfig("front", ...)` 假设已实现利用 `serial_number_or_name="front"` 查找对应相机；若继续通过 ROS 读取图像，可将 `cameras` 替换为自定义 `RosImageCameraConfig`。

---

## 6. 测试方案

| 层级 | 目标 | 方法 |
|------|------|------|
| Mock 单元测试 | 验证 `AgileXRobot` 行为，无需实际 ROS | 使用 `unittest.mock` 替换 `AgileXRosBridge`，检查 `connect`/`send_action`/`ensure_safe_goal_position` 调用次数；为相机提供虚拟实现 |
| 桥接层测试 | 确保 `AgileXRosBridge` 解析 `JointState`、拼接 `Twist` 正确 | 在 CI 中启动 `rosbridge_server` docker，使用 `roslibpy` 向虚拟 Topic 推送消息，断言队列更新 |
| 硬件自检 | 验证零位、速度裁剪、Watchdog | 脚本：`python -m lerobot.scripts.robot_check --robot.type agilex --check joints`，记录最大延迟 |
| 集成测试 | 训练/推理闭环 | `pytest tests/robots/test_agilex_robot.py -k "hardware" --device=agilex`（标记 slow/hardware），在真机上运行；或执行 `make test-end-to-end DEVICE=cpu -k agilex` |
| 数据管线 | 确保 HDF5 -> LeRobot Dataset | 构造 10 帧样例，运行转换脚本并校验 `observation_features` 键一致 |

测试重点：Topic 断连自动重连、`max_relative_target` 是否限制单步跳变、底盘命令是否裁剪、相机超时是否抛出异常。

---

## 7. 常见问题与调试

| 症状 | 可能原因 | 排查与解决 |
|------|----------|------------|
| `DeviceNotConnectedError` | 未启动 `rosbridge_server` 或 `roscore` | `roslaunch rosbridge_server rosbridge_websocket.launch`，确认 `ws://<host>:9090` 可连接 |
| 关节剧烈抖动 | 未限幅或命令频率过高 | 调整 `max_relative_target`、`command_rate_hz`，启用插值（类似 `replay_data.py`） |
| 相机观测为 `None` | Topic 名错误或没有新帧 | 使用 `rostopic echo <topic>`，检查 `image_timeout_s` 配置，并确保 `astra_camera` 运行 |
| 底盘持续移动 | Watchdog 未触发，保留旧命令 | 在 `AgileXRosBridge.watchdog()` 中加入零速发布，并在 `disconnect()` 调用一次 |
| HDF5 字段缺失 | 原始脚本 `--use_depth_image=False` | 集成前检查 `h5py.File.keys()`，补齐缺失字段或在转换器中填默认值 |
| `/puppet/joint_*` 不更新 | Piper 未能连接 CAN 或主臂插头未断开 | 参考原文档“**需要拔掉主臂的航空插头**”，重新执行 `./can_config.sh` |

调试技巧：`rostopic hz <topic>` 查看频率；`roswtf` 检查节点依赖；`rosparam get` 确认 `mode`；`watch -n1 rosmsg show sensor_msgs/JointState` 了解结构。

---

## 8. 部署与使用

### 8.1 建议运行顺序

1. 启动 `roscore` → `rosbridge_server`。
2. 运行 `./can_config.sh`、`roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=true`。
3. 启动 `astra_camera multi_camera.launch`（或其它相机驱动）。
4. 在 LeRobot 环境中执行 `python -m lerobot.scripts.robot_check --robot.type agilex` 验证连接。
5. 根据需要启动采集、遥操作或策略推理脚本。

### 8.2 命令行示例

- **数据采集**（遥操作闭环）：
  ```bash
  python -m lerobot.scripts.collect_dataset \
      --robot.type agilex --robot.id agx_dualarm_01 \
      --teleop.type agilex_leader --output ./tests/outputs/agilex_demo
  ```
- **策略推理**：
  ```bash
  python -m lerobot.scripts.run_policy \
      --robot.type agilex \
      --policy.checkpoint ./checkpoints/act_agilex.pt \
      --obs.cameras "cam_front,cam_left,cam_right"
  ```

### 8.3 YAML 配置示例

```yaml
# configs/agilex/agilex_default.yaml
robot:
  type: agilex
  id: agx_dualarm_01
  rosbridge_host: 192.168.1.10
  rosbridge_port: 9090
  master_topics:
    left: /master/joint_left
    right: /master/joint_right
  puppet_topics:
    left: /puppet/joint_left
    right: /puppet/joint_right
  odom_topic: /odom
  cmd_vel_topic: /cmd_vel
  command_rate_hz: 30
  max_relative_target: 0.2
  base_speed_limits: [0.5, 1.0]
  cameras:
    cam_front:
      type: intelrealsense
      serial_number_or_name: "front"
      fps: 30
      width: 640
      height: 480
    cam_left:
      type: opencv
      index_or_path: 0
      fps: 30
      width: 640
      height: 480
    cam_right:
      type: opencv
      index_or_path: 1
      fps: 30
      width: 640
      height: 480
```

此 YAML 可被 `draccus` 自动加载：`python -m lerobot.scripts.collect_dataset --config configs/agilex/agilex_default.yaml`。

### 8.4 推理 & 遥操作注意事项

- 推理模式需将 `mode:=1 auto_enable:=true` 并确认主臂航空插头已拔出，否则指令将回环到主臂。
- 若仅想发布主臂轨迹供仿真或复现，可仿照原脚本 `--only_pub_master`，在 `AgileXRobot.send_action()` 中跳过 `/puppet/joint_*` 发布。
- 录制数据前建议运行 `python visualize_episodes.py`（或 LeRobot 等效脚本）检查视频对齐。

---

通过以上步骤即可完成 Agilex 双臂机器人在 LeRobot 框架中的集成：
- **配置驱动**：`AgileXConfig` 统一声明所有 Topic、限幅与相机参数；
- **ROS 适配**：`AgileXRosBridge` 承接原版 API，向外暴露干净的 `Robot` 接口；
- **安全可靠**：动作限幅、Watchdog、零速保护确保硬件安全；
- **易于扩展**：相机、遥操作、数据转换全部模块化，可快速拓展到其它 Agilex 机型。
