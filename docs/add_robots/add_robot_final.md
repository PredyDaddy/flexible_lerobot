# LeRobot 新机器人集成完全指南

> 本文档面向希望在 LeRobot 项目中集成新机器人硬件的开发者，综合现有实现（SO100/SO101、Reachy2、HopeJr、LeKiwi 等）的最佳实践，提供从架构理解到测试验证的完整指南。

## 目录

1. [概述与架构设计](#1-概述与架构设计)
2. [前置准备](#2-前置准备)
3. [集成流程总览](#3-集成流程总览)
4. [详细实现步骤](#4-详细实现步骤)
5. [电机控制器集成](#5-电机控制器集成)
6. [相机系统集成](#6-相机系统集成)
7. [遥操作支持](#7-遥操作支持)
8. [配置与注册](#8-配置与注册)
9. [测试指南](#9-测试指南)
10. [常见问题与调试](#10-常见问题与调试)
11. [最佳实践](#11-最佳实践)
12. [附录](#12-附录)

---

## 1. 概述与架构设计

### 1.1 模块化架构

LeRobot 采用分层模块化设计，通过抽象基类定义标准接口，使不同类型的机器人可以无缝集成到统一的数据采集和策略训练流程中。

```
src/lerobot/
├── robots/               # 机器人实现
│   ├── robot.py          # Robot 抽象基类
│   ├── config.py         # RobotConfig 基类
│   ├── utils.py          # 工厂函数和工具
│   └── <robot_name>/     # 具体机器人实现
├── motors/               # 电机总线驱动
│   ├── motors_bus.py     # MotorsBus 抽象基类
│   ├── feetech/          # Feetech 电机
│   └── dynamixel/        # Dynamixel 电机
├── cameras/              # 相机驱动
│   ├── camera.py         # Camera 抽象基类
│   ├── opencv/           # OpenCV 相机
│   └── realsense/        # Intel RealSense
└── teleoperators/        # 遥操作设备
    ├── teleoperator.py   # Teleoperator 抽象基类
    └── <teleop_name>/    # 具体遥操作实现
```

### 1.2 核心类关系图

```
┌─────────────────────────────────────────────────────────────────┐
│                         Robot (ABC)                              │
│  ├── config_class: type[RobotConfig]                            │
│  ├── name: str                                                   │
│  ├── calibration: dict[str, MotorCalibration]                   │
│  │                                                               │
│  │ Abstract Properties:                                          │
│  ├── observation_features -> dict[str, type | tuple]            │
│  ├── action_features -> dict[str, type]                         │
│  ├── is_connected -> bool                                        │
│  ├── is_calibrated -> bool                                       │
│  │                                                               │
│  │ Abstract Methods:                                             │
│  ├── connect(calibrate: bool) -> None                            │
│  ├── calibrate() -> None                                         │
│  ├── configure() -> None                                         │
│  ├── get_observation() -> dict[str, Any]                         │
│  ├── send_action(action: dict) -> dict[str, Any]                │
│  └── disconnect() -> None                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ SO100Follower│    │    LeKiwi    │    │ Reachy2Robot │
│   (机械臂)    │    │ (移动机器人)  │    │ (人形机器人)  │
└──────────────┘    └──────────────┘    └──────────────┘
```

### 1.3 数据流

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Teleoperator │────▶│    Robot     │────▶│   Dataset    │
│  get_action() │     │ send_action()│     │   记录数据    │
└──────────────┘     │get_observation│     └──────────────┘
                     └──────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ MotorsBus│ │  Camera  │ │  Sensors │
        │  电机控制 │ │  图像采集 │ │ 其他传感器 │
        └──────────┘ └──────────┘ └──────────┘
```

### 1.4 设计原则

| 原则 | 说明 |
|------|------|
| **统一接口** | 所有机器人继承自 `Robot` 基类，实现标准化的连接、观测、动作接口 |
| **组件解耦** | 电机、相机、遥操作器作为独立模块，可灵活组合 |
| **配置驱动** | 使用 dataclass 配置类，支持 draccus 命令行解析 |
| **声明型特征** | `observation_features` / `action_features` 提供平坦字典结构定义 |
| **校准支持** | 内置校准机制，自动保存/加载校准数据到 `HF_LEROBOT_CALIBRATION/` |
| **安全机制** | 提供动作限幅 `ensure_safe_goal_position()`、扭矩保护等安全特性 |
| **显式错误处理** | 统一使用 `DeviceAlreadyConnectedError`、`DeviceNotConnectedError` 等异常 |

---

## 2. 前置准备

### 2.1 开发环境搭建

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

### 2.2 硬件依赖安装

根据机器人类型安装对应驱动：

```bash
# Feetech 电机
pip install scservo-sdk

# Dynamixel 电机
pip install dynamixel-sdk

# Intel RealSense 相机
pip install pyrealsense2

# 其他依赖根据具体硬件安装
```

### 2.3 串口权限配置 (Linux)

```bash
# 添加用户到 dialout 组
sudo usermod -a -G dialout $USER

# 或临时修改权限
sudo chmod 666 /dev/ttyUSB*

# 重新登录以生效
```

### 2.4 参考实现阅读

建议先阅读以下文件理解实现模式：

| 文件 | 说明 |
|------|------|
| `src/lerobot/robots/so100_follower/so100_follower.py` | 标准 Feetech 六轴机械臂实现 |
| `src/lerobot/robots/so101_follower/so101_follower.py` | 带不同运动范围流程的变体 |
| `src/lerobot/robots/lekiwi/lekiwi.py` | 移动机器人（底盘+机械臂）实现 |
| `src/lerobot/robots/reachy2/robot_reachy2.py` | 外部 SDK 集成示例 |
| `src/lerobot/robots/hope_jr/*` | 多路总线与 GUI 校准示例 |
| `src/lerobot/motors/motors_bus.py` | MotorsBus 抽象与核心 API |

---

## 3. 集成流程总览

向 LeRobot 增加一个新机器人，一般包含以下步骤：

```
┌─────────────────────────────────────────────────────────────────┐
│  1. 创建目录结构                                                  │
│     src/lerobot/robots/my_robot/                                │
│     ├── __init__.py                                             │
│     ├── config_my_robot.py    # 配置类                          │
│     └── my_robot.py           # 机器人实现                       │
├─────────────────────────────────────────────────────────────────┤
│  2. 定义配置类                                                    │
│     @RobotConfig.register_subclass("my_robot")                  │
│     继承 RobotConfig，定义端口、相机、安全参数等                   │
├─────────────────────────────────────────────────────────────────┤
│  3. 实现 Robot 子类                                               │
│     实现所有抽象属性和方法                                        │
│     组合 MotorsBus、Camera 等组件                                │
├─────────────────────────────────────────────────────────────────┤
│  4. 注册工厂函数                                                  │
│     在 robots/utils.py 的 make_robot_from_config() 中添加分支    │
├─────────────────────────────────────────────────────────────────┤
│  5. 添加遥操作支持（可选）                                         │
│     在 teleoperators/ 下实现对应的 Teleoperator                  │
├─────────────────────────────────────────────────────────────────┤
│  6. 编写测试                                                      │
│     在 tests/robots/ 下添加单元测试                              │
├─────────────────────────────────────────────────────────────────┤
│  7. 添加配置示例                                                  │
│     在 lerobot/configs/ 中添加示例配置                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. 详细实现步骤

### 4.1 创建目录结构

```bash
mkdir -p src/lerobot/robots/my_robot
touch src/lerobot/robots/my_robot/__init__.py
touch src/lerobot/robots/my_robot/config_my_robot.py
touch src/lerobot/robots/my_robot/my_robot.py
```

### 4.2 定义配置类

`config_my_robot.py`:

```python
#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("my_robot")  # 注册机器人类型名称
@dataclass
class MyRobotConfig(RobotConfig):
    """MyRobot 机器人配置类

    Attributes:
        port: 串口路径，如 "/dev/ttyUSB0" 或 "/dev/tty.usbmodem*"
        disable_torque_on_disconnect: 断开时是否禁用扭矩（安全考虑，默认True）
        max_relative_target: 安全限制，限制每步相对目标位置的最大变化量
        cameras: 相机配置字典，键为相机名称
        use_degrees: 是否使用角度制（用于向后兼容）
    """
    # 必需参数
    port: str

    # 安全参数
    disable_torque_on_disconnect: bool = True
    max_relative_target: float | dict[str, float] | None = 5.0  # 设置为 None 禁用限幅

    # 相机配置（可选）
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "front": OpenCVCameraConfig(
                index_or_path=0,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    # 兼容性选项
    use_degrees: bool = False
```

**关键点说明：**

| 要点 | 说明 |
|------|------|
| `@RobotConfig.register_subclass()` | 将配置类注册到 draccus 选择注册表，使 CLI 可通过 `--robot.type=my_robot` 创建 |
| 继承 `RobotConfig` | 自动获得 `id` 和 `calibration_dir` 属性 |
| `cameras` 字段 | 会在 `RobotConfig.__post_init__` 中自动验证 width/height/fps 不为 None |
| `max_relative_target` | 支持 float（所有电机）或 dict（每个电机单独设置）|

### 4.3 实现 Robot 子类

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
    支持 Feetech 电机总线、相机集成、自动校准。
    """

    # ==================== 类属性 ====================
    config_class = MyRobotConfig  # 关联的配置类
    name = "my_robot"             # 机器人类型名称（与注册名一致）

    # ==================== 初始化 ====================
    def __init__(self, config: MyRobotConfig):
        super().__init__(config)  # 初始化基类（处理校准文件加载）
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

    # ==================== 特征定义 ====================
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

    # ==================== 连接状态 ====================
    @property
    def is_connected(self) -> bool:
        """检查机器人是否已连接（电机 + 所有相机）"""
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        """检查机器人是否已校准"""
        return self.bus.is_calibrated

    # ==================== 核心方法 ====================
    def connect(self, calibrate: bool = True) -> None:
        """连接机器人

        Args:
            calibrate: 如果未校准，是否自动执行校准流程
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # 1. 连接电机总线
        self.bus.connect()

        # 2. 检查并执行校准
        if not self.is_calibrated and calibrate:
            logger.info("机器人未校准，开始校准流程...")
            self.calibrate()

        # 3. 连接相机
        for cam in self.cameras.values():
            cam.connect()

        # 4. 配置电机参数
        self.configure()

        logger.info(f"{self} connected.")

    def calibrate(self) -> None:
        """执行机器人校准

        校准流程：
        1. 如果已有校准文件，询问是否使用
        2. 否则引导用户进行新的校准
        3. 保存校准数据到文件
        """
        # 检查是否有已保存的校准
        if self.calibration:
            user_input = input(
                f"发现校准文件 {self.calibration_fpath}，"
                "按 ENTER 使用，输入 'c' 重新校准: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"使用现有校准文件")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\n========== 开始校准 {self} ==========")
        self.bus.disable_torque()

        # 设置操作模式
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        # 步骤1：记录中位位置
        input("\n将所有关节移动到运动范围中间位置，按 ENTER 继续...")
        homing_offsets = self.bus.set_half_turn_homings()

        # 步骤2：记录运动范围
        full_turn_motor = "wrist_roll"  # 可 360° 旋转的电机
        unknown_range_motors = [m for m in self.bus.motors if m != full_turn_motor]

        print(f"\n依次将除 '{full_turn_motor}' 外的所有关节移动到极限位置...")
        print("完成后按 ENTER 停止记录...")
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
        print(f"\n校准已保存到 {self.calibration_fpath}")

    def configure(self) -> None:
        """配置电机参数（操作模式、PID、安全限制等）"""
        with self.bus.torque_disabled():
            self.bus.configure_motors()

            for motor in self.bus.motors:
                # 设置位置控制模式
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

                # 调整 PID 参数减少抖动
                self.bus.write("P_Coefficient", motor, 16)
                self.bus.write("I_Coefficient", motor, 0)
                self.bus.write("D_Coefficient", motor, 32)

                # 夹爪特殊配置：限制扭矩和电流
                if motor == "gripper":
                    self.bus.write("Max_Torque_Limit", motor, 500)
                    self.bus.write("Protection_Current", motor, 250)

    def get_observation(self) -> dict[str, Any]:
        """获取当前观测

        Returns:
            包含电机位置和相机图像的字典
            键名格式：电机 -> "{motor_name}.pos", 相机 -> "{camera_name}"
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # 读取电机位置
        start = time.perf_counter()
        obs_dict = self.bus.sync_read("Present_Position")
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read motors: {dt_ms:.1f}ms")

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
            action: 目标位置字典 {"motor_name.pos": value, ...}

        Returns:
            实际发送的动作（可能经过安全限幅）
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # 提取位置指令
        goal_pos = {
            key.removesuffix(".pos"): val
            for key, val in action.items()
            if key.endswith(".pos")
        }

        # 应用安全限幅
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {
                key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()
            }
            goal_pos = ensure_safe_goal_position(
                goal_present_pos, self.config.max_relative_target
            )

        # 发送指令到电机
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

__all__ = ["MyRobot", "MyRobotConfig"]
```

---

## 5. 电机控制器集成

### 5.1 支持的电机类型

LeRobot 内置支持两种主流电机总线：

| 类型 | 类名 | 支持型号 | SDK |
|------|------|----------|-----|
| Feetech | `FeetechMotorsBus` | STS3215, STS3250, SM8512BL 等 | `scservo-sdk` |
| Dynamixel | `DynamixelMotorsBus` | XL330, XL430, XM430, XM540 等 | `dynamixel-sdk` |

### 5.2 Motor 数据类

```python
from lerobot.motors import Motor, MotorNormMode

# 创建电机定义
motor = Motor(
    id=1,                                    # 电机 ID（总线上的唯一标识）
    model="sts3215",                         # 电机型号
    norm_mode=MotorNormMode.RANGE_M100_100   # 归一化模式
)
```

**归一化模式说明：**

| 模式 | 输出范围 | 适用场景 |
|------|----------|----------|
| `RANGE_0_100` | [0, 100] | 夹爪等单向运动 |
| `RANGE_M100_100` | [-100, 100] | 关节位置控制（推荐） |
| `DEGREES` | 角度值 | 需要绝对角度的场景 |

### 5.3 MotorsBus 核心 API

```python
# ==================== 连接与断开 ====================
bus.connect(handshake=True)          # 连接并验证电机
bus.disconnect(disable_torque=True)   # 断开（可选禁用扭矩）

# ==================== 读取操作 ====================
bus.sync_read("Present_Position")                      # 同步读取所有电机
bus.sync_read("Present_Position", ["motor1", "motor2"]) # 读取指定电机
bus.read("Present_Position", "motor1")                  # 读取单个电机

# ==================== 写入操作 ====================
bus.sync_write("Goal_Position", {"motor1": 50, "motor2": -30})
bus.write("Operating_Mode", "motor1", OperatingMode.POSITION.value)

# ==================== 扭矩控制 ====================
bus.enable_torque()       # 启用扭矩
bus.disable_torque()      # 禁用扭矩
with bus.torque_disabled():  # 上下文管理器（配置时使用）
    bus.configure_motors()

# ==================== 校准相关 ====================
bus.set_half_turn_homings()           # 设置半转归零（当前位置为中点）
bus.record_ranges_of_motion(motors)   # 交互式记录运动范围
bus.write_calibration(calibration)    # 写入校准数据到电机
bus.read_calibration()                # 从电机读取校准数据
```

### 5.4 校准数据结构

```python
from lerobot.motors import MotorCalibration

calibration = {
    "shoulder_pan": MotorCalibration(
        id=1,               # 电机 ID
        drive_mode=0,       # 驱动模式
        homing_offset=2048, # 零点偏移
        range_min=500,      # 最小位置（原始值）
        range_max=3500,     # 最大位置（原始值）
    ),
    # ... 其他电机
}
```

### 5.5 自定义电机总线

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

    # 其他必要的抽象方法...
```

---

## 6. 相机系统集成

### 6.1 支持的相机类型

| 类型 | 配置类 | 描述 |
|------|--------|------|
| OpenCV | `OpenCVCameraConfig` | USB 摄像头、V4L2 设备 |
| RealSense | `RealSenseCameraConfig` | Intel RealSense D435/D455 |
| Reachy2 | `Reachy2CameraConfig` | Reachy2 机器人专用相机 |

### 6.2 相机配置示例

```python
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig

# OpenCV 相机（USB 摄像头）
opencv_cam = OpenCVCameraConfig(
    fps=30,
    width=640,
    height=480,
    index_or_path=0,      # 设备索引或路径
    color_mode="rgb",
    rotation=None,        # 可选：Cv2Rotation.ROTATE_90_CW
)

# RealSense 深度相机
realsense_cam = RealSenseCameraConfig(
    fps=30,
    width=640,
    height=480,
    serial_number="123456789",  # 相机序列号
    color_mode="rgb",
)
```

### 6.3 在机器人中使用相机

```python
from lerobot.cameras.utils import make_cameras_from_configs

# 在 Robot.__init__ 中
self.cameras = make_cameras_from_configs(config.cameras)

# 在 connect() 中
for cam in self.cameras.values():
    cam.connect()

# 在 get_observation() 中
for cam_key, cam in self.cameras.items():
    obs_dict[cam_key] = cam.async_read()  # 非阻塞读取

# 在 disconnect() 中
for cam in self.cameras.values():
    cam.disconnect()
```

### 6.4 Camera 基类接口

```python
class Camera(ABC):
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def read(self) -> np.ndarray:
        """同步读取（阻塞）"""
        ...

    def async_read(self) -> np.ndarray:
        """异步读取（带后台线程，推荐使用）"""
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...

    @staticmethod
    def find_cameras() -> list:
        """发现可用相机（用于调试）"""
        ...
```

---

## 7. 遥操作支持

### 7.1 遥操作器架构

遥操作器（Teleoperator）用于实时控制机器人，通常是 Leader 臂、手柄或其他输入设备。

```
┌─────────────────────────────────────────────────────────────────┐
│                      Teleoperator (ABC)                          │
│  ├── config_class: type[TeleoperatorConfig]                     │
│  ├── name: str                                                   │
│  │                                                               │
│  │ Abstract Properties:                                          │
│  ├── action_features -> dict[str, type]                         │
│  ├── feedback_features -> dict[str, type]                       │
│  ├── is_connected -> bool                                        │
│  │                                                               │
│  │ Abstract Methods:                                             │
│  ├── connect() -> None                                           │
│  ├── disconnect() -> None                                        │
│  ├── get_action() -> dict[str, Any]   # 获取操作者输入           │
│  └── send_feedback() -> None           # 发送反馈（可选）        │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 实现 Teleoperator 子类

`config_my_teleop.py`:

```python
from dataclasses import dataclass
from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("my_teleop")
@dataclass
class MyTeleoperatorConfig(TeleoperatorConfig):
    port: str
    use_present_position: bool = True  # 使用当前位置还是目标位置
```

`my_teleop.py`:

```python
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors import Motor, MotorNormMode


class MyTeleoperator(Teleoperator):
    config_class = MyTeleoperatorConfig
    name = "my_teleop"

    def __init__(self, config: MyTeleoperatorConfig):
        super().__init__(config)
        self.config = config
        self.bus = FeetechMotorsBus(
            port=config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
                "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
                # ... 与 follower 机器人关节名称保持一致
            },
            calibration=self.calibration,
        )

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}  # 如不需要反馈

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self) -> None:
        self.bus.connect()
        # 配置为被动模式（可自由移动）
        self.bus.disable_torque()

    def disconnect(self) -> None:
        self.bus.disconnect()

    def get_action(self) -> dict[str, float]:
        """读取 Leader 臂位置作为动作指令"""
        register = "Present_Position" if self.config.use_present_position else "Goal_Position"
        pos = self.bus.sync_read(register)
        return {f"{motor}.pos": val for motor, val in pos.items()}

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        # 可选：实现力反馈或震动等
        raise NotImplementedError("该遥操作器不支持反馈")
```

### 7.3 遥操作数据采集流程

```python
# 典型的遥操作数据采集循环
teleop = make_teleoperator_from_config(teleop_config)
robot = make_robot_from_config(robot_config)

teleop.connect()
robot.connect()

try:
    while running:
        # 1. 从遥操作器获取动作
        action = teleop.get_action()

        # 2. 发送动作到机器人
        robot.send_action(action)

        # 3. 获取观测并记录
        obs = robot.get_observation()
        dataset.record(obs, action)
finally:
    teleop.disconnect()
    robot.disconnect()
```

---

## 8. 配置与注册

### 8.1 工厂函数注册

编辑 `src/lerobot/robots/utils.py`，添加新机器人类型：

```python
def make_robot_from_config(config: RobotConfig) -> Robot:
    """从配置创建机器人实例"""
    # ... 现有代码 ...
    elif config.type == "my_robot":
        from .my_robot import MyRobot
        return MyRobot(config)
    # ... 其他机器人 ...
    else:
        raise ValueError(f"Unknown robot type: {config.type}")
```

同样地，如果有 Teleoperator，编辑 `src/lerobot/teleoperators/utils.py`：

```python
def make_teleoperator_from_config(config: TeleoperatorConfig) -> Teleoperator:
    # ... 现有代码 ...
    elif config.type == "my_teleop":
        from .my_teleop import MyTeleoperator
        return MyTeleoperator(config)
    # ...
```

### 8.2 YAML 配置示例

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
  overhead_cam:
    _target_: lerobot.cameras.opencv.OpenCVCameraConfig
    fps: 30
    width: 640
    height: 480
    index_or_path: 1
```

### 8.3 命令行使用

```bash
# 使用 YAML 配置
lerobot-teleoperate --robot.config_path=lerobot/configs/robot/my_robot.yaml

# 覆盖参数
lerobot-teleoperate --robot.config_path=... --robot.port=/dev/ttyUSB1

# 使用注册的类型
lerobot-teleoperate --robot.type=my_robot --robot.port=/dev/ttyUSB0
```

### 8.4 第三方插件机制

LeRobot 支持通过 entry-points 自动发现第三方机器人：

```toml
# 在你的 pyproject.toml 中定义入口点
[project.entry-points."lerobot_robot_myrobot"]
myrobot = "my_package.robot:MyRobot"
```

插件包命名规则：
- 机器人：`lerobot_robot_*`
- 相机：`lerobot_camera_*`
- 遥操作器：`lerobot_teleoperator_*`

---

## 9. 测试指南

### 9.1 创建 Mock 实现

在 `tests/mocks/mock_my_robot.py` 创建模拟实现用于单元测试：

```python
from dataclasses import dataclass, field
from typing import Any

from lerobot.robots import Robot, RobotConfig


@RobotConfig.register_subclass("mock_my_robot")
@dataclass
class MockMyRobotConfig(RobotConfig):
    cameras: dict = field(default_factory=dict)


class MockMyRobot(Robot):
    """用于测试的模拟机器人"""
    config_class = MockMyRobotConfig
    name = "mock_my_robot"

    def __init__(self, config: MockMyRobotConfig):
        super().__init__(config)
        self._is_connected = False
        self._is_calibrated = True
        self._state = {f"joint_{i}.pos": 0.0 for i in range(6)}

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        return {k: float for k in self._state}

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

### 9.2 编写单元测试

`tests/robots/test_my_robot.py`:

```python
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from contextlib import contextmanager

from lerobot.robots.my_robot import MyRobot, MyRobotConfig


def _make_bus_mock():
    """创建模拟电机总线"""
    bus = MagicMock()
    bus.is_connected = True
    bus.is_calibrated = True
    bus.motors = {
        "shoulder_pan": MagicMock(id=1),
        "shoulder_lift": MagicMock(id=2),
        "elbow_flex": MagicMock(id=3),
        "wrist_flex": MagicMock(id=4),
        "wrist_roll": MagicMock(id=5),
        "gripper": MagicMock(id=6),
    }
    bus.sync_read.return_value = {
        "shoulder_pan": 0.0,
        "shoulder_lift": 10.0,
        "elbow_flex": -5.0,
        "wrist_flex": 0.0,
        "wrist_roll": 0.0,
        "gripper": 50.0,
    }

    @contextmanager
    def torque_disabled():
        yield

    bus.torque_disabled = torque_disabled
    return bus


@pytest.fixture
def mock_bus():
    """模拟 FeetechMotorsBus"""
    with patch("lerobot.robots.my_robot.my_robot.FeetechMotorsBus") as MockBus:
        bus = _make_bus_mock()
        MockBus.return_value = bus
        yield bus


@pytest.fixture
def mock_cameras():
    """模拟相机"""
    with patch("lerobot.robots.my_robot.my_robot.make_cameras_from_configs") as mock:
        mock.return_value = {}  # 无相机
        yield mock


@pytest.fixture
def robot_config():
    """测试用配置"""
    return MyRobotConfig(port="/dev/ttyUSB0", cameras={})


class TestMyRobot:
    def test_init(self, mock_bus, mock_cameras, robot_config):
        """测试初始化"""
        robot = MyRobot(robot_config)
        assert robot.name == "my_robot"
        assert robot.config == robot_config

    def test_connect(self, mock_bus, mock_cameras, robot_config):
        """测试连接"""
        robot = MyRobot(robot_config)
        robot.connect(calibrate=False)
        mock_bus.connect.assert_called_once()

    def test_connect_already_connected(self, mock_bus, mock_cameras, robot_config):
        """测试重复连接抛出异常"""
        from lerobot.utils.errors import DeviceAlreadyConnectedError

        robot = MyRobot(robot_config)
        robot.connect(calibrate=False)

        with pytest.raises(DeviceAlreadyConnectedError):
            robot.connect()

    def test_get_observation(self, mock_bus, mock_cameras, robot_config):
        """测试获取观测"""
        robot = MyRobot(robot_config)
        robot.connect(calibrate=False)

        obs = robot.get_observation()

        # 验证返回的键名格式
        assert "shoulder_pan.pos" in obs
        assert "gripper.pos" in obs
        mock_bus.sync_read.assert_called_with("Present_Position")

    def test_send_action(self, mock_bus, mock_cameras, robot_config):
        """测试发送动作"""
        robot = MyRobot(robot_config)
        robot.connect(calibrate=False)

        action = {
            "shoulder_pan.pos": 10.0,
            "shoulder_lift.pos": -5.0,
        }
        result = robot.send_action(action)

        mock_bus.sync_write.assert_called()
        assert "shoulder_pan.pos" in result

    def test_send_action_with_safety_limit(self, mock_bus, mock_cameras):
        """测试动作安全限幅"""
        config = MyRobotConfig(
            port="/dev/ttyUSB0",
            cameras={},
            max_relative_target=5.0,  # 每步最大移动 5
        )
        robot = MyRobot(config)
        robot.connect(calibrate=False)

        # 尝试发送超出限制的动作
        action = {"shoulder_pan.pos": 100.0}  # 当前值为 0，跳变 100
        result = robot.send_action(action)

        # 验证动作被限幅
        assert result["shoulder_pan.pos"] <= 5.0

    def test_disconnect(self, mock_bus, mock_cameras, robot_config):
        """测试断开连接"""
        robot = MyRobot(robot_config)
        robot.connect(calibrate=False)
        robot.disconnect()

        mock_bus.disconnect.assert_called_once_with(True)

    def test_disconnect_not_connected(self, mock_bus, mock_cameras, robot_config):
        """测试未连接时断开抛出异常"""
        from lerobot.utils.errors import DeviceNotConnectedError

        robot = MyRobot(robot_config)
        mock_bus.is_connected = False

        with pytest.raises(DeviceNotConnectedError):
            robot.disconnect()
```

### 9.3 运行测试

```bash
# 运行特定测试
pytest tests/robots/test_my_robot.py -v

# 运行所有机器人测试
pytest tests/robots/ -v

# 带覆盖率
pytest tests/robots/test_my_robot.py --cov=lerobot.robots.my_robot --cov-report=term

# 仅运行特定测试方法
pytest tests/robots/test_my_robot.py -k "test_connect" -v
```

### 9.4 硬件测试（可选）

对于真实硬件测试，添加标记跳过 CI：

```python
import pytest

@pytest.mark.skip_if_no_hardware
class TestMyRobotHardware:
    """需要真实硬件的测试"""

    def test_real_connection(self):
        config = MyRobotConfig(port="/dev/ttyUSB0")
        robot = MyRobot(config)
        robot.connect()
        try:
            obs = robot.get_observation()
            assert obs is not None
            assert all(isinstance(v, float) for k, v in obs.items() if k.endswith(".pos"))
        finally:
            robot.disconnect()
```

---

## 10. 常见问题与调试

### 10.1 常见问题排查表

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| `DeviceAlreadyConnectedError` | 重复调用 `connect()` | 使用 `try/finally` 或检查 `if not robot.is_connected` |
| `DeviceNotConnectedError` | 硬件已断开或 `connect()` 失败 | 检查串口/网络，捕获异常记录日志 |
| `Could not connect on port` | 串口权限或被占用 | 检查 `ls -la /dev/ttyUSB*`，加入 dialout 组 |
| 波特率/ID 不匹配 | 工厂未配置正确 ID | 使用 `MotorsBus.scan_port()` 和 `setup_motor()` |
| 校准文件与实际不符 | 机械结构调整或更换电机 | 删除校准文件，重新 `calibrate()` |
| 相机未连接 | 配置缺少 `width/height/fps` | `RobotConfig.__post_init__` 会抛错，检查配置 |
| 动作抖动或延迟 | PID 参数或 `max_relative_target` 过小 | 在 `configure()` 中调整 PID 参数 |
| 遥操作映射不一致 | Teleoperator 与 Robot 键名不同 | 保证双方 `action_features` 键名完全一致 |

### 10.2 串口调试

```bash
# 查找串口设备
ls -la /dev/ttyUSB*
ls -la /dev/ttyACM*

# 使用 lerobot 工具查找端口
python -m lerobot.scripts.find_port

# 检查串口是否被占用
fuser /dev/ttyUSB0

# 临时修改权限
sudo chmod 666 /dev/ttyUSB0
```

### 10.3 校准问题调试

```bash
# 查看校准文件位置
echo $HF_LEROBOT_CALIBRATION  # 默认 ~/.cache/lerobot/calibration

# 删除校准文件强制重新校准
rm -rf ~/.cache/lerobot/calibration/robots/my_robot/

# 在代码中强制重新校准
robot.connect(calibrate=True)
```

### 10.4 性能调试

```python
import logging
import time

# 启用详细日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("lerobot.robots.my_robot")
logger.setLevel(logging.DEBUG)

# 测量采样耗时
def benchmark_robot(robot, n_samples=100):
    times = []
    for _ in range(n_samples):
        start = time.perf_counter()
        obs = robot.get_observation()
        times.append(time.perf_counter() - start)

    avg_ms = sum(times) / len(times) * 1000
    max_ms = max(times) * 1000
    print(f"平均: {avg_ms:.1f}ms, 最大: {max_ms:.1f}ms, FPS: {1000/avg_ms:.1f}")
```

### 10.5 电机状态调试

```python
def debug_motor_state(robot):
    """打印详细的电机状态"""
    for motor in robot.bus.motors:
        pos = robot.bus.read("Present_Position", motor)
        vel = robot.bus.read("Present_Velocity", motor)
        load = robot.bus.read("Present_Load", motor)
        print(f"{motor}: pos={pos:.2f}, vel={vel:.2f}, load={load:.2f}")

# 使用
robot.connect()
debug_motor_state(robot)
```

---

## 11. 最佳实践

### 11.1 代码风格

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

### 11.2 错误处理

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

### 11.3 资源管理

```python
# ✅ 使用上下文管理器配置电机
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

### 11.4 安全考虑

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

### 11.5 日志规范

```python
import logging
logger = logging.getLogger(__name__)

# ✅ 使用 logger 而非 print
logger.info(f"{self} connected.")
logger.debug(f"{self} read state: {dt_ms:.1f}ms")
logger.error(f"Failed to connect {self}: {e}")

# ✅ 在 __repr__ 中返回有用信息
def __repr__(self) -> str:
    return f"{self.__class__.__name__}(id={self.config.id}, port={self.config.port})"
```

### 11.6 测试友好性

```python
# ✅ 不要在模块导入时访问硬件
# 硬件访问应放在 __init__/connect 中，方便测试时 patch

# ❌ 避免
class MyRobot(Robot):
    bus = FeetechMotorsBus(...)  # 类级别实例化

# ✅ 推荐
class MyRobot(Robot):
    def __init__(self, config):
        self.bus = FeetechMotorsBus(...)  # 实例级别
```

---

## 12. 附录

### 12.1 完整文件清单

添加新机器人需要创建/修改的文件：

```
src/lerobot/robots/my_robot/
├── __init__.py              # 导出 MyRobot, MyRobotConfig
├── config_my_robot.py       # 配置类定义
└── my_robot.py              # 机器人实现

src/lerobot/robots/utils.py  # 添加工厂函数分支

tests/robots/
└── test_my_robot.py         # 单元测试

lerobot/configs/robot/
└── my_robot.yaml            # (可选) YAML 配置示例
```

### 12.2 参考实现对照表

| 机器人 | 类型 | 特点 | 文件位置 |
|--------|------|------|----------|
| SO100Follower | 机械臂 | 标准 6-DoF 臂，Feetech 电机 | `so100_follower/` |
| SO101Follower | 机械臂 | 带腕关节扩展 | `so101_follower/` |
| LeKiwi | 移动机器人 | 全向底盘 + 机械臂 | `lekiwi/` |
| HopeJrArm | 机械臂 | 7-DoF，GUI 校准 | `hope_jr/` |
| Reachy2Robot | 人形机器人 | 外部 SDK 集成 | `reachy2/` |

### 12.3 核心类速查

| 类名 | 模块 | 用途 |
|------|------|------|
| `Robot` | `lerobot.robots.robot` | 机器人抽象基类 |
| `RobotConfig` | `lerobot.robots.config` | 机器人配置基类 |
| `Teleoperator` | `lerobot.teleoperators.teleoperator` | 遥操作器抽象基类 |
| `TeleoperatorConfig` | `lerobot.teleoperators.config` | 遥操作器配置基类 |
| `MotorsBus` | `lerobot.motors.motors_bus` | 电机总线抽象基类 |
| `FeetechMotorsBus` | `lerobot.motors.feetech` | Feetech 电机总线 |
| `DynamixelMotorsBus` | `lerobot.motors.dynamixel` | Dynamixel 电机总线 |
| `Motor` | `lerobot.motors` | 电机定义数据类 |
| `MotorCalibration` | `lerobot.motors` | 电机校准数据类 |
| `MotorNormMode` | `lerobot.motors` | 电机归一化模式枚举 |
| `Camera` | `lerobot.cameras.camera` | 相机抽象基类 |
| `CameraConfig` | `lerobot.cameras.configs` | 相机配置基类 |

### 12.4 常用工具函数

| 函数 | 模块 | 用途 |
|------|------|------|
| `make_robot_from_config()` | `lerobot.robots.utils` | 从配置创建机器人 |
| `make_teleoperator_from_config()` | `lerobot.teleoperators.utils` | 从配置创建遥操作器 |
| `make_cameras_from_configs()` | `lerobot.cameras.utils` | 从配置创建相机字典 |
| `ensure_safe_goal_position()` | `lerobot.robots.utils` | 动作安全限幅 |

### 12.5 相关链接

- [LeRobot GitHub 仓库](https://github.com/huggingface/lerobot)
- [LeRobot 官方文档](https://huggingface.co/docs/lerobot)
- [Dynamixel SDK 文档](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_sdk/overview/)
- [Feetech 电机文档](https://www.feetechrc.com/)
- [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense)

---

> **文档版本**: 1.0
> **最后更新**: 2024年
> **基于**: LeRobot 项目现有实现（SO100/SO101、Reachy2、HopeJr、LeKiwi 等）

