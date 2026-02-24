# LeRobot 机器人集成开发指南

本文档详细介绍如何向 LeRobot 项目添加新的机器人支持。通过阅读本指南，开发者将能够独立完成从创建机器人类到测试验证的完整流程。

## 目录

1. [概述](#1-概述)
2. [架构设计](#2-架构设计)
3. [前置准备](#3-前置准备)
4. [详细实现步骤](#4-详细实现步骤)
5. [电机控制器集成](#5-电机控制器集成)
6. [相机系统集成](#6-相机系统集成)
7. [遥操作支持](#7-遥操作支持)
8. [配置文件编写](#8-配置文件编写)
9. [工厂函数注册](#9-工厂函数注册)
10. [测试指南](#10-测试指南)
11. [常见问题与调试](#11-常见问题与调试)
12. [最佳实践](#12-最佳实践)

---

## 1. 概述

### 1.1 LeRobot 机器人集成架构

LeRobot 采用模块化架构设计，通过抽象基类定义标准接口，使得不同类型的机器人可以无缝集成到统一的数据采集和策略训练流程中。

```
src/lerobot/
├── robots/           # 机器人实现
│   ├── robot.py      # Robot 抽象基类
│   ├── config.py     # RobotConfig 基类
│   ├── utils.py      # 工厂函数和工具
│   └── <robot_name>/ # 具体机器人实现
├── motors/           # 电机总线驱动
│   ├── motors_bus.py # MotorsBus 抽象基类
│   ├── dynamixel/    # Dynamixel 电机
│   └── feetech/      # Feetech 电机
├── cameras/          # 相机驱动
│   ├── camera.py     # Camera 抽象基类
│   ├── opencv/       # OpenCV 相机
│   └── realsense/    # Intel RealSense
└── teleoperators/    # 遥操作设备
    ├── teleoperator.py # Teleoperator 抽象基类
    └── <teleop_name>/  # 具体遥操作实现
```

### 1.2 设计理念

- **统一接口**：所有机器人继承自 `Robot` 基类，实现标准化的连接、观测、动作接口
- **组件解耦**：电机、相机、遥操作器作为独立模块，可灵活组合
- **配置驱动**：使用 dataclass 配置类，支持 draccus 命令行解析
- **校准支持**：内置校准机制，自动保存/加载校准数据
- **安全机制**：提供动作限幅、扭矩保护等安全特性

---

## 2. 架构设计

### 2.1 核心类关系图

```
┌─────────────────────────────────────────────────────────────┐
│                       Robot (ABC)                           │
│  ├── config_class: type[RobotConfig]                       │
│  ├── name: str                                              │
│  ├── calibration: dict[str, MotorCalibration]              │
│  │                                                          │
│  │ Abstract Properties:                                     │
│  ├── observation_features -> dict                           │
│  ├── action_features -> dict                                │
│  ├── is_connected -> bool                                   │
│  ├── is_calibrated -> bool                                  │
│  │                                                          │
│  │ Abstract Methods:                                        │
│  ├── connect(calibrate: bool) -> None                       │
│  ├── calibrate() -> None                                    │
│  ├── configure() -> None                                    │
│  ├── get_observation() -> dict[str, Any]                    │
│  ├── send_action(action: dict) -> dict[str, Any]           │
│  └── disconnect() -> None                                   │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ SO100Follower│    │   LeKiwi     │    │ Reachy2Robot │
│  (机械臂)     │    │ (移动机器人)  │    │ (人形机器人)  │
└──────────────┘    └──────────────┘    └──────────────┘
```

### 2.2 数据流

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Teleoperator │────▶│    Robot     │────▶│   Dataset    │
│  get_action() │     │ send_action()│     │  记录数据     │
└──────────────┘     │get_observation│     └──────────────┘
                     └──────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ MotorsBus│ │  Camera  │ │  Sensors │
        │ 电机控制  │ │  图像采集 │ │  其他传感器│
        └──────────┘ └──────────┘ └──────────┘
```

---

## 3. 前置准备

### 3.1 开发环境

```bash
# 克隆仓库并创建开发环境
git clone https://github.com/huggingface/lerobot.git
cd lerobot
conda create -y -n lerobot python=3.10
conda activate lerobot
conda install ffmpeg -c conda-forge

# 以开发模式安装
pip install -e ".[dev,test]"

# 安装预提交钩子
pre-commit install
```

### 3.2 硬件依赖

根据机器人类型，可能需要：
- **电机驱动**：Dynamixel SDK 或 Feetech SDK
- **相机驱动**：OpenCV、pyrealsense2 等
- **通信接口**：USB 串口、网络接口等

```bash
# Dynamixel 电机
pip install dynamixel-sdk

# Intel RealSense 相机
pip install pyrealsense2

# 其他依赖根据具体硬件安装
```

### 3.3 理解现有实现

建议先阅读以下文件理解模式：
- `src/lerobot/robots/so100_follower/so100_follower.py` - 标准机械臂实现
- `src/lerobot/robots/lekiwi/lekiwi.py` - 移动机器人实现
- `src/lerobot/robots/reachy2/robot_reachy2.py` - 外部 SDK 集成示例

---

## 4. 详细实现步骤

### 4.1 创建目录结构

在 `src/lerobot/robots/` 下创建新机器人目录：

```
src/lerobot/robots/my_robot/
├── __init__.py
├── config_my_robot.py    # 配置类
└── my_robot.py           # 机器人实现
```

### 4.2 定义配置类

`config_my_robot.py`:

```python
#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("my_robot")  # 注册机器人类型名称
@dataclass
class MyRobotConfig(RobotConfig):
    """MyRobot 机器人配置类

    Attributes:
        port: 串口路径，如 "/dev/ttyUSB0" 或 "/dev/tty.usbmodem575E0031751"
        disable_torque_on_disconnect: 断开时是否禁用扭矩
        max_relative_target: 安全限制，限制相对目标位置的最大变化量
        cameras: 相机配置字典
        use_degrees: 是否使用角度制（用于向后兼容）
    """
    # 必需参数
    port: str

    # 可选参数（带默认值）
    disable_torque_on_disconnect: bool = True

    # 安全限制参数
    # 设置为正数可限制所有电机，或使用字典为每个电机单独设置
    max_relative_target: float | dict[str, float] | None = None

    # 相机配置
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # 兼容性选项
    use_degrees: bool = False
```

**关键点说明：**
- `@RobotConfig.register_subclass("my_robot")` 装饰器将配置类注册到 draccus 选择注册表
- 继承 `RobotConfig` 获得 `id` 和 `calibration_dir` 属性
- `cameras` 字段会在 `RobotConfig.__post_init__` 中自动验证

### 4.3 实现机器人类

`my_robot.py`:

```python
#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_my_robot import MyRobotConfig

logger = logging.getLogger(__name__)


class MyRobot(Robot):
    """MyRobot 机器人实现

    该类展示了一个标准的 6 自由度机械臂实现模式。
    """

    # 必须设置的类属性
    config_class = MyRobotConfig  # 关联的配置类
    name = "my_robot"             # 机器人类型名称（与注册名一致）

    def __init__(self, config: MyRobotConfig):
        super().__init__(config)  # 初始化基类（处理校准文件）
        self.config = config

        # 根据配置选择归一化模式
        norm_mode = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100

        # 初始化电机总线
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                # 定义电机：名称 -> Motor(ID, 型号, 归一化模式)
                "shoulder_pan": Motor(1, "sts3215", norm_mode),
                "shoulder_lift": Motor(2, "sts3215", norm_mode),
                "elbow_flex": Motor(3, "sts3215", norm_mode),
                "wrist_flex": Motor(4, "sts3215", norm_mode),
                "wrist_roll": Motor(5, "sts3215", norm_mode),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),  # 夹爪用 0-100
            },
            calibration=self.calibration,  # 从基类加载的校准数据
        )

        # 初始化相机
        self.cameras = make_cameras_from_configs(config.cameras)

    # ========== 特征定义属性 ==========

    @property
    def _motors_ft(self) -> dict[str, type]:
        """电机特征：{电机名.pos: float}"""
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """相机特征：{相机名: (高度, 宽度, 通道)}"""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """观测特征：电机位置 + 相机图像"""
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """动作特征：电机目标位置"""
        return self._motors_ft

    # ========== 连接状态属性 ==========

    @property
    def is_connected(self) -> bool:
        """检查机器人是否已连接"""
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        """检查机器人是否已校准"""
        return self.bus.is_calibrated

    # ========== 核心方法实现 ==========

    def connect(self, calibrate: bool = True) -> None:
        """连接机器人

        Args:
            calibrate: 如果未校准，是否自动执行校准
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # 连接电机总线
        self.bus.connect()

        # 检查并执行校准
        if not self.is_calibrated and calibrate:
            logger.info("需要校准，开始校准流程...")
            self.calibrate()

        # 连接相机
        for cam in self.cameras.values():
            cam.connect()

        # 配置电机参数
        self.configure()
        logger.info(f"{self} connected.")

    def calibrate(self) -> None:
        """执行机器人校准

        校准流程：
        1. 如果已有校准文件，询问是否使用
        2. 否则引导用户进行新的校准
        3. 保存校准数据到文件
        """
        if self.calibration:
            user_input = input(
                f"发现校准文件 {self.id}，按 ENTER 使用，输入 'c' 重新校准: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"使用现有校准文件: {self.id}")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\n开始校准 {self}")
        self.bus.disable_torque()

        # 设置操作模式
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        # 记录中位位置
        input("将所有关节移动到运动范围中间位置，按 ENTER 继续...")
        homing_offsets = self.bus.set_half_turn_homings()

        # 记录运动范围
        full_turn_motor = "wrist_roll"  # 可 360° 旋转的电机
        unknown_range_motors = [m for m in self.bus.motors if m != full_turn_motor]

        print(f"依次移动除 '{full_turn_motor}' 外的所有关节到极限位置...")
        print("记录中，按 ENTER 停止...")
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)

        # 全转电机使用完整范围
        range_mins[full_turn_motor] = 0
        range_maxes[full_turn_motor] = 4095

        # 构建校准数据
        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        # 写入电机并保存文件
        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print(f"校准已保存到 {self.calibration_fpath}")

    def configure(self) -> None:
        """配置电机参数"""
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor in self.bus.motors:
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                # 调整 PID 参数减少抖动
                self.bus.write("P_Coefficient", motor, 16)
                self.bus.write("I_Coefficient", motor, 0)
                self.bus.write("D_Coefficient", motor, 32)

                # 夹爪特殊配置
                if motor == "gripper":
                    self.bus.write("Max_Torque_Limit", motor, 500)
                    self.bus.write("Protection_Current", motor, 250)

    def get_observation(self) -> dict[str, Any]:
        """获取当前观测

        Returns:
            包含电机位置和相机图像的字典
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # 读取电机位置
        start = time.perf_counter()
        obs_dict = self.bus.sync_read("Present_Position")
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # 读取相机图像
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """发送动作指令

        Args:
            action: 目标位置字典 {"motor_name.pos": value}

        Returns:
            实际发送的动作（可能经过安全限幅）
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # 提取位置指令
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # 应用安全限幅
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # 发送指令
        self.bus.sync_write("Goal_Position", goal_pos)

        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def disconnect(self) -> None:
        """断开机器人连接"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # 断开电机（可选禁用扭矩）
        self.bus.disconnect(self.config.disable_torque_on_disconnect)

        # 断开相机
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
```

### 4.4 创建 `__init__.py`

```python
from .config_my_robot import MyRobotConfig
from .my_robot import MyRobot
```

---

## 5. 电机控制器集成

### 5.1 支持的电机类型

LeRobot 内置支持两种主流电机总线：

| 类型 | 类名 | 支持型号 |
|------|------|----------|
| Dynamixel | `DynamixelMotorsBus` | XL330, XL430, XM430, XM540 等 |
| Feetech | `FeetechMotorsBus` | STS3215, STS3250, SM8512BL 等 |

### 5.2 Motor 数据类

```python
from lerobot.motors import Motor, MotorNormMode

# 创建电机定义
motor = Motor(
    id=1,                              # 电机 ID
    model="sts3215",                   # 电机型号
    norm_mode=MotorNormMode.RANGE_M100_100  # 归一化模式
)
```

**归一化模式说明：**

| 模式 | 范围 | 适用场景 |
|------|------|----------|
| `RANGE_0_100` | [0, 100] | 夹爪等单向运动 |
| `RANGE_M100_100` | [-100, 100] | 关节位置控制 |
| `DEGREES` | 角度值 | 需要角度的场景 |

### 5.3 MotorsBus 关键方法

```python
# 连接与断开
bus.connect(handshake=True)   # 连接并验证电机
bus.disconnect(disable_torque=True)

# 读取操作
bus.sync_read("Present_Position")  # 同步读取所有电机
bus.sync_read("Present_Position", ["motor1", "motor2"])  # 读取指定电机
bus.read("Present_Position", "motor1")  # 读取单个电机

# 写入操作
bus.sync_write("Goal_Position", {"motor1": 50, "motor2": -30})

# 扭矩控制
bus.enable_torque()
bus.disable_torque()
with bus.torque_disabled():  # 上下文管理器
    # 在这里配置电机
    pass

# 校准相关
bus.set_half_turn_homings()  # 设置半转归零
bus.record_ranges_of_motion(motors)  # 记录运动范围
bus.write_calibration(calibration)  # 写入校准数据
```

### 5.4 添加自定义电机驱动

如果需要支持新的电机类型，可以继承 `MotorsBus`：

```python
from lerobot.motors.motors_bus import MotorsBus

class MyMotorsBus(MotorsBus):
    """自定义电机总线实现"""

    def connect(self, handshake: bool = True) -> None:
        # 实现连接逻辑
        pass

    def disconnect(self, disable_torque: bool = True) -> None:
        # 实现断开逻辑
        pass

    def sync_read(self, register: str, motors: list[str] | None = None) -> dict[str, Any]:
        # 实现同步读取
        pass

    def sync_write(self, register: str, values: dict[str, Any]) -> None:
        # 实现同步写入
        pass

    # 其他必要方法...
```

---

## 6. 相机系统集成

### 6.1 支持的相机类型

| 类型 | 配置类 | 描述 |
|------|--------|------|
| OpenCV | `OpenCVCameraConfig` | USB 摄像头、V4L2 设备 |
| RealSense | `RealSenseCameraConfig` | Intel RealSense D435/D455 |

### 6.2 相机配置示例

```python
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig

# OpenCV 相机
opencv_cam = OpenCVCameraConfig(
    fps=30,
    width=640,
    height=480,
    index_or_path=0,  # 设备索引或路径
    color_mode="rgb",
    rotation=None,  # 可选：Cv2Rotation.ROTATE_90_CW
)

# RealSense 相机
realsense_cam = RealSenseCameraConfig(
    fps=30,
    width=640,
    height=480,
    serial_number="123456789",  # 相机序列号
    color_mode="rgb",
)
```

### 6.3 在机器人配置中使用相机

```python
@dataclass
class MyRobotConfig(RobotConfig):
    port: str
    cameras: dict[str, CameraConfig] = field(default_factory=lambda: {
        "wrist_cam": OpenCVCameraConfig(
            fps=30, width=640, height=480, index_or_path=0
        ),
        "overhead_cam": RealSenseCameraConfig(
            fps=30, width=640, height=480, serial_number="123456789"
        ),
    })
```

### 6.4 Camera 基类接口

```python
class Camera(ABC):
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def read(self) -> np.ndarray: ...

    def async_read(self) -> np.ndarray:
        """异步读取（带后台线程）"""
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...
```

---

## 7. 遥操作支持

### 7.1 遥操作器架构

遥操作器（Teleoperator）用于实时控制机器人，通常是一个 leader 臂或手柄。

```python
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig

class MyTeleoperator(Teleoperator):
    config_class = MyTeleoperatorConfig
    name = "my_teleop"

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def get_action(self) -> dict[str, Any]: ...

    @property
    def action_features(self) -> dict[str, type]: ...
    @property
    def is_connected(self) -> bool: ...
```

### 7.2 Leader 臂示例

参考 `src/lerobot/teleoperators/so100_leader/`:

```python
class SO100Leader(Teleoperator):
    """SO100 Leader 臂遥操作器"""

    config_class = SO100LeaderConfig
    name = "so100_leader"

    def __init__(self, config: SO100LeaderConfig):
        super().__init__(config)
        self.bus = FeetechMotorsBus(
            port=config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
                # ... 其他电机
            },
            calibration=self.calibration,
        )

    def get_action(self) -> dict[str, float]:
        """读取 leader 臂位置作为动作指令"""
        pos = self.bus.sync_read("Present_Position")
        return {f"{motor}.pos": val for motor, val in pos.items()}
```

### 7.3 注册遥操作器

在 `config_my_teleop.py`:

```python
from lerobot.teleoperators.config import TeleoperatorConfig

@TeleoperatorConfig.register_subclass("my_teleop")
@dataclass
class MyTeleoperatorConfig(TeleoperatorConfig):
    port: str
    # 其他配置...
```

---

## 8. 配置文件编写

### 8.1 YAML 配置示例

创建 `lerobot/configs/robot/my_robot.yaml`:

```yaml
# @package _global_
type: my_robot
id: my_robot_001
port: /dev/ttyUSB0
disable_torque_on_disconnect: true
max_relative_target: 10.0

cameras:
  wrist_cam:
    _target_: lerobot.cameras.opencv.OpenCVCameraConfig
    fps: 30
    width: 640
    height: 480
    index_or_path: 0
```

### 8.2 命令行使用

```bash
# 使用 YAML 配置
lerobot-teleoperate --robot.config_path=lerobot/configs/robot/my_robot.yaml

# 覆盖参数
lerobot-teleoperate --robot.config_path=... --robot.port=/dev/ttyUSB1

# 使用注册的类型
lerobot-teleoperate --robot.type=my_robot --robot.port=/dev/ttyUSB0
```

---

## 9. 工厂函数注册

### 9.1 在 utils.py 中注册

编辑 `src/lerobot/robots/utils.py`，添加新机器人类型：

```python
def make_robot_from_config(config: RobotConfig) -> Robot:
    # ... 现有代码 ...
    elif config.type == "my_robot":
        from .my_robot import MyRobot
        return MyRobot(config)
    # ... 其他机器人 ...
```

### 9.2 第三方插件机制

LeRobot 支持通过命名约定自动发现第三方机器人：

```python
# 在 pyproject.toml 中定义入口点
[project.entry-points."lerobot_robot_myrobot"]
myrobot = "my_package.robot:MyRobot"
```

插件包命名规则：
- 机器人：`lerobot_robot_*`
- 相机：`lerobot_camera_*`
- 遥操作器：`lerobot_teleoperator_*`

---

## 10. 测试指南

### 10.1 创建 Mock 实现

在 `tests/mocks/` 创建模拟实现用于单元测试：

```python
# tests/mocks/mock_my_robot.py
from dataclasses import dataclass, field
from typing import Any

from lerobot.robots import Robot, RobotConfig


@RobotConfig.register_subclass("mock_my_robot")
@dataclass
class MockMyRobotConfig(RobotConfig):
    cameras: dict = field(default_factory=dict)


class MockMyRobot(Robot):
    config_class = MockMyRobotConfig
    name = "mock_my_robot"

    def __init__(self, config: MockMyRobotConfig):
        super().__init__(config)
        self._is_connected = False
        self._is_calibrated = True
        self._state = {f"motor_{i}.pos": 0.0 for i in range(6)}

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        return self._state.copy()

    @property
    def action_features(self) -> dict[str, type]:
        return {k: float for k in self._state}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    def connect(self, calibrate: bool = True) -> None:
        self._is_connected = True

    def calibrate(self) -> None:
        self._is_calibrated = True

    def configure(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        return self._state.copy()

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        for key, val in action.items():
            if key in self._state:
                self._state[key] = val
        return action

    def disconnect(self) -> None:
        self._is_connected = False
```

### 10.2 编写单元测试

在 `tests/robots/test_my_robot.py`:

```python
import pytest
from unittest.mock import MagicMock, patch

from lerobot.robots.my_robot import MyRobot, MyRobotConfig


@pytest.fixture
def mock_bus():
    """模拟电机总线"""
    with patch("lerobot.robots.my_robot.my_robot.FeetechMotorsBus") as MockBus:
        bus = MagicMock()
        bus.is_connected = True
        bus.is_calibrated = True
        bus.motors = {
            "shoulder_pan": MagicMock(id=1),
            "shoulder_lift": MagicMock(id=2),
            # ...
        }
        bus.sync_read.return_value = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
        }
        MockBus.return_value = bus
        yield bus


@pytest.fixture
def robot_config():
    return MyRobotConfig(port="/dev/ttyUSB0")


class TestMyRobot:
    def test_init(self, mock_bus, robot_config):
        robot = MyRobot(robot_config)
        assert robot.name == "my_robot"
        assert robot.config == robot_config

    def test_connect(self, mock_bus, robot_config):
        robot = MyRobot(robot_config)
        robot.connect(calibrate=False)
        mock_bus.connect.assert_called_once()

    def test_get_observation(self, mock_bus, robot_config):
        robot = MyRobot(robot_config)
        robot.connect(calibrate=False)
        obs = robot.get_observation()
        assert "shoulder_pan.pos" in obs

    def test_send_action(self, mock_bus, robot_config):
        robot = MyRobot(robot_config)
        robot.connect(calibrate=False)
        action = {"shoulder_pan.pos": 50.0, "shoulder_lift.pos": -30.0}
        result = robot.send_action(action)
        mock_bus.sync_write.assert_called()

    def test_disconnect(self, mock_bus, robot_config):
        robot = MyRobot(robot_config)
        robot.connect(calibrate=False)
        robot.disconnect()
        mock_bus.disconnect.assert_called_once_with(True)
```

### 10.3 运行测试

```bash
# 运行特定测试
pytest tests/robots/test_my_robot.py -v

# 运行所有机器人测试
pytest tests/robots/ -v

# 带覆盖率
pytest tests/robots/test_my_robot.py --cov=lerobot.robots.my_robot --cov-report=term
```

### 10.4 硬件测试（可选）

对于真实硬件测试，添加标记跳过 CI：

```python
import pytest

@pytest.mark.skip_if_no_hardware
class TestMyRobotHardware:
    def test_real_connection(self):
        config = MyRobotConfig(port="/dev/ttyUSB0")
        robot = MyRobot(config)
        robot.connect()
        try:
            obs = robot.get_observation()
            assert obs is not None
        finally:
            robot.disconnect()
```

---

## 11. 常见问题与调试

### 11.1 连接问题

**问题：无法连接电机总线**
```
DeviceNotConnectedError: Could not connect to port /dev/ttyUSB0
```

**解决方案：**
```bash
# 检查串口设备
ls -la /dev/ttyUSB*

# 检查权限
sudo chmod 666 /dev/ttyUSB0

# 添加用户到 dialout 组
sudo usermod -a -G dialout $USER
# 重新登录生效
```

**问题：电机 ID 冲突**
```
RuntimeError: Motor ID 1 not found on bus
```

**解决方案：**
```python
# 使用 setup_motors 方法逐个设置 ID
robot.bus.setup_motor("shoulder_pan")
```

### 11.2 校准问题

**问题：校准数据不匹配**
```
Mismatch between calibration values in the motor and the calibration file
```

**解决方案：**
```bash
# 删除旧校准文件，重新校准
rm -rf ~/.cache/lerobot/calibration/robots/my_robot/

# 或在代码中强制重新校准
robot.connect(calibrate=True)
```

### 11.3 性能问题

**问题：帧率过低**

**解决方案：**
```python
# 1. 使用异步相机读取
obs_dict[cam_key] = cam.async_read()  # 而非 cam.read()

# 2. 减少读取的电机寄存器
# 仅读取必要的寄存器，避免读取速度、电流等

# 3. 优化日志级别
import logging
logging.getLogger("lerobot").setLevel(logging.WARNING)
```

### 11.4 调试技巧

```python
import logging

# 启用详细日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("lerobot.robots.my_robot")
logger.setLevel(logging.DEBUG)

# 打印电机状态
def debug_motor_state(robot):
    for motor in robot.bus.motors:
        pos = robot.bus.read("Present_Position", motor)
        vel = robot.bus.read("Present_Velocity", motor)
        print(f"{motor}: pos={pos}, vel={vel}")
```

---

## 12. 最佳实践

### 12.1 代码风格

```python
# ✅ 好的做法
from lerobot.motors import Motor, MotorNormMode

class MyRobot(Robot):
    """简明的类文档字符串"""

    config_class = MyRobotConfig
    name = "my_robot"

    def __init__(self, config: MyRobotConfig):
        super().__init__(config)
        self.config = config  # 类型明确

# ❌ 避免的做法
class myrobot(Robot):  # 类名应为 PascalCase
    def __init__(self, config):  # 缺少类型注解
        self.cfg = config  # 命名不清晰
```

### 12.2 错误处理

```python
from lerobot.utils.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError

def connect(self, calibrate: bool = True) -> None:
    # ✅ 在操作前检查状态
    if self.is_connected:
        raise DeviceAlreadyConnectedError(f"{self} already connected")

    try:
        self.bus.connect()
    except Exception as e:
        logger.error(f"Failed to connect {self}: {e}")
        raise

def get_observation(self) -> dict[str, Any]:
    # ✅ 确保连接状态
    if not self.is_connected:
        raise DeviceNotConnectedError(f"{self} is not connected.")

    # ... 读取操作
```

### 12.3 资源管理

```python
# ✅ 使用上下文管理器
with self.bus.torque_disabled():
    self.bus.configure_motors()

# ✅ 确保断开连接
try:
    robot.connect()
    # ... 操作
finally:
    robot.disconnect()

# ✅ 或使用 contextmanager
from contextlib import contextmanager

@contextmanager
def robot_session(config):
    robot = make_robot_from_config(config)
    robot.connect()
    try:
        yield robot
    finally:
        robot.disconnect()

# 使用
with robot_session(config) as robot:
    obs = robot.get_observation()
```

### 12.4 安全考虑

```python
# ✅ 设置动作限幅
@dataclass
class MyRobotConfig(RobotConfig):
    max_relative_target: float = 10.0  # 限制每步移动量

# ✅ 夹爪保护
self.bus.write("Max_Torque_Limit", "gripper", 500)
self.bus.write("Protection_Current", "gripper", 250)

# ✅ 断开时禁用扭矩
def disconnect(self):
    self.bus.disconnect(disable_torque=True)
```

### 12.5 文档与注释

```python
class MyRobot(Robot):
    """MyRobot 6-DoF 机械臂实现

    该机器人使用 Feetech STS3215 伺服电机，支持：
    - 位置控制模式
    - USB 相机集成
    - 自动校准

    Attributes:
        bus: 电机总线实例
        cameras: 相机字典
        arm_motors: 手臂电机名称列表

    Example:
        >>> config = MyRobotConfig(port="/dev/ttyUSB0")
        >>> robot = MyRobot(config)
        >>> robot.connect()
        >>> obs = robot.get_observation()
        >>> robot.disconnect()
    """
```

---

## 附录 A：完整文件清单

添加新机器人需要创建/修改的文件：

```
src/lerobot/robots/my_robot/
├── __init__.py              # 导出类
├── config_my_robot.py       # 配置类
└── my_robot.py              # 机器人实现

src/lerobot/robots/utils.py  # 添加工厂函数分支

tests/robots/
└── test_my_robot.py         # 单元测试

lerobot/configs/robot/
└── my_robot.yaml            # (可选) YAML 配置
```

## 附录 B：参考实现

| 机器人 | 类型 | 特点 | 文件位置 |
|--------|------|------|----------|
| SO100Follower | 机械臂 | 标准 6-DoF 臂 | `so100_follower/` |
| SO101Follower | 机械臂 | 带腕关节扩展 | `so101_follower/` |
| LeKiwi | 移动机器人 | 全向底盘 + 臂 | `lekiwi/` |
| HopeJrArm | 机械臂 | 7-DoF 高精度臂 | `hope_jr/` |
| Reachy2Robot | 人形机器人 | 外部 SDK 集成 | `reachy2/` |

## 附录 C：相关链接

- [LeRobot GitHub 仓库](https://github.com/huggingface/lerobot)
- [Dynamixel SDK 文档](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_sdk/overview/)
- [Feetech 电机文档](https://www.feetechrc.com/)
- [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense)

