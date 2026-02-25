# LeRobot 新机器人集成完整指南

> 面向希望在 LeRobot 库中集成新真实机器人硬件的开发者，本指南结合架构理念、实践步骤与最佳实践，提供从驱动开发到遥操作的全流程参考。

## 目录

1. [概述：架构与设计理念](#概述架构与设计理念)
2. [前置准备](#前置准备)
3. [集成流程总览](#集成流程总览)
4. [详细实现步骤](#详细实现步骤)
5. [电机控制器集成](#电机控制器集成)
6. [相机系统集成](#相机系统集成)
7. [遥操作支持](#遥操作支持)
8. [配置与注册](#配置与注册)
9. [完整代码模板](#完整代码模板)
10. [测试与验证指南](#测试与验证指南)
11. [常见问题与调试技巧](#常见问题与调试技巧)
12. [最佳实践](#最佳实践)

---

## 概述：架构与设计理念

### 模块化架构

LeRobot 将机器人硬件堆栈拆分为多层抽象，确保统一接口和组件解耦：

```
src/lerobot/
├── robots/               # Robot 抽象基类及具体实现
│   ├── robot.py          # Robot 抽象基类（connect/get_observation/send_action）
│   ├── config.py         # RobotConfig 配置基类
│   ├── utils.py          # 工厂函数和安全工具
│   └── <robot_name>/     # 具体机器人实现目录
├── motors/               # MotorsBus 抽象 + Feetech/Dynamixel实现
│   └── feetech/          # FeetechMotorsBus、OperatingMode等
├── cameras/              # Camera 抽象及OpenCV/RealSense实现
└── teleoperators/        # Teleoperator 抽象与实现
```

### 核心设计原则

1. **统一接口**：所有机器人都继承 `Robot` 基类，实现标准化方法，使训练脚本与硬件解耦
2. **声明型特征描述**：`observation_features` 和 `action_features` 提供平坦字典结构定义（如 `<joint_name>.pos`）
3. **配置驱动**：通过 `RobotConfig` 和 `draccus.ChoiceRegistry` 实现CLI参数解析
4. **安全优先**：提供 `ensure_safe_goal_position()` 等工具限制动作跳变，自动管理扭矩
5. **可测试性**：所有硬件依赖都可以通过 mock 隔离，便于CI测试

---

## 前置准备

### 硬件需求

- **机器人本体**：关节式机械臂、移动平台或其他形态
- **电机与总线**：
  - 首选 **Feetech** 或 **Dynamixel** 伺服电机（已有现成驱动）
  - 若使用其他电机/控制板，需支持：端口打开、寄存器读写、批量同步操作
- **相机（可选）**：OpenCV相机、Intel RealSense或自定义相机后端
- **遥操作设备（可选）**：Leader机械臂、游戏手柄等

### 软件与环境

```bash
# 创建虚拟环境
conda create -y -n lerobot python=3.10
conda activate lerobot
conda install ffmpeg -c conda-forge

# 以开发模式安装
pip install -e ".[dev,test]"
pre-commit install
```

**额外驱动**：
- Feetech: `pip install scservo-sdk`
- Dynamixel: `pip install dynamixel-sdk`
- RealSense: `pip install pyrealsense2`

### 阅读参考实现

重点阅读以下文件以掌握通用模式：
- `src/lerobot/robots/so_follower/so_follower.py` - 典型Feetech六轴机械臂
- `src/lerobot/robots/so_follower/so_follower.py` - 类似结构，不同运动范围
- `src/lerobot/robots/reachy2/robot_reachy2.py` - 如何包装外部SDK并驱动移动底盘
- `src/lerobot/robots/hope_jr/*` - 多路总线（arm/hand）与GUI校准

---

## 集成流程总览

集成新机器人包含以下9个步骤：

1. **定义配置类**：创建 `RobotConfig` 子类，使用 `@RobotConfig.register_subclass("your_robot")` 注册
2. **实现 Robot 子类**：继承 `Robot` 基类，实现所有抽象方法
3. **组合硬件组件**：实例化 `MotorsBus` 和相机，或通过SDK连接硬件
4. **实现核心方法**：补全 `connect/calibrate/configure/get_observation/send_action/disconnect`
5. **遥操作支持（可选）**：如需闭环采集，同步实现 `Teleoperator` 子类
6. **注册工厂函数**：在 `robots/utils.py` 中添加新分支，或确保自动发现
7. **添加配置示例**：在 `lerobot/configs/` 中提供使用示例
8. **编写测试**：使用 mock 编写单元测试，验证连接、观测、动作逻辑
9. **验证与调试**：运行测试，在真实硬件上验证功能

---

## 详细实现步骤

### 步骤1：创建目录结构

```
src/lerobot/robots/my_robot/
├── __init__.py
├── config_my_robot.py
└── my_robot.py
```

在 `__init__.py` 中导出主要类：
```python
from .config_my_robot import MyRobotConfig
from .my_robot import MyRobot

__all__ = ["MyRobot", "MyRobotConfig"]
```

### 步骤2：定义配置类

在 `config_my_robot.py` 中：

```python
from dataclasses import dataclass, field
from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("my_robot")
@dataclass
class MyRobotConfig(RobotConfig):
    """MyRobot 配置类"""

    port: str = "/dev/ttyUSB0"  # 串口路径
    use_degrees: bool = False   # 是否使用角度单位

    # 安全参数
    max_relative_target: float | None = 5.0  # 单步最大移动量
    disable_torque_on_disconnect: bool = True

    # 相机配置
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "front": OpenCVCameraConfig(
                index_or_path=0, fps=30, width=640, height=480
            ),
        }
    )
```

**要点**：
- 使用 `@RobotConfig.register_subclass()` 注册类型字符串
- 相机配置必须包含 `fps/width/height`（基类会自动校验）
- 建议添加安全参数如 `max_relative_target`

### 步骤3：实现构造函数

在 `my_robot.py` 中：

```python
import logging
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.robots.robot import Robot
from lerobot.robots.utils import ensure_safe_goal_position

from .config_my_robot import MyRobotConfig

logger = logging.getLogger(__name__)


class MyRobot(Robot):
    config_class = MyRobotConfig
    name = "my_robot"

    def __init__(self, config: MyRobotConfig):
        super().__init__(config)  # 初始化校准相关逻辑
        self.config = config

        # 创建电机总线
        norm_mode = (
            MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        )
        self.bus = FeetechMotorsBus(
            port=config.port,
            motors={
                "joint_1": Motor(1, "sts3215", norm_mode),
                "joint_2": Motor(2, "sts3215", norm_mode),
                "joint_3": Motor(3, "sts3215", norm_mode),
                "joint_4": Motor(4, "sts3215", norm_mode),
                "joint_5": Motor(5, "sts3215", norm_mode),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

        # 创建相机实例
        self.cameras = make_cameras_from_configs(config.cameras)
```

### 步骤4：实现特征声明

```python
    @property
    def _motors_ft(self) -> dict[str, type]:
        """电机特征：关节位置"""
        return {f"{m}.pos": float for m in self.bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """相机特征：图像尺寸"""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """观测特征 = 电机特征 + 相机特征"""
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """动作特征（通常与电机特征一致）"""
        return self._motors_ft
```

### 步骤5：实现连接管理

```python
    from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

    @property
    def is_connected(self) -> bool:
        """检查连接状态：电机总线 + 所有相机"""
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        """检查校准状态"""
        return self.bus.is_calibrated

    def connect(self, calibrate: bool = True) -> None:
        """连接硬件"""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # 1. 连接电机总线
        self.bus.connect()

        # 2. 必要时执行校准
        if not self.is_calibrated and calibrate:
            self.calibrate()

        # 3. 连接相机
        for cam in self.cameras.values():
            cam.connect()

        # 4. 配置电机参数
        self.configure()

        logger.info(f"{self} connected.")

    def disconnect(self) -> None:
        """断开连接"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        # 断开电机总线（可选择是否释放扭矩）
        self.bus.disconnect(self.config.disable_torque_on_disconnect)

        # 断开相机
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
```

### 步骤6：实现校准

```python
    def calibrate(self) -> None:
        """交互式校准流程"""
        # 如果已有校准文件，询问是否重新校准
        if self.calibration:
            user_input = input(
                "Press ENTER to use stored calibration, or type 'c' to re-run: "
            )
            if user_input.strip().lower() != "c":
                self.bus.write_calibration(self.calibration)
                return

        # 禁用扭矩
        self.bus.disable_torque()

        # 设置半转归位（当前位置作为中点）
        homing_offsets = self.bus.set_half_turn_homings()

        # 交互式记录运动范围
        print("Move each joint to its MINIMUM position, then press Enter...")
        input()
        mins = self.bus.sync_read("Present_Position")

        print("Move each joint to its MAXIMUM position, then press Enter...")
        input()
        maxs = self.bus.sync_read("Present_Position")

        # 构建校准数据
        from lerobot.motors import MotorCalibration

        self.calibration = {
            motor: MotorCalibration(
                id=m.id,
                drive_mode=0,  # 0=normal, 1=reverse
                homing_offset=homing_offsets[motor],
                range_min=mins[motor],
                range_max=maxs[motor],
            )
            for motor, m in self.bus.motors.items()
        }

        # 写入硬件并保存到文件
        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        logger.info("Calibration completed and saved.")
```

### 步骤7：实现配置

```python
    def configure(self) -> None:
        """配置电机参数"""
        with self.bus.torque_disabled():
            # 配置通用参数（返回延迟、加速度等）
            self.bus.configure_motors()

            # 设置操作模式为位置模式
            for motor in self.bus.motors:
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

            # 对抓手设置扭矩限制（可选）
            if "gripper" in self.bus.motors:
                self.bus.write("Max_Torque_Limit", self.bus.motors["gripper"], 300)
                self.bus.write("Protection_Current", self.bus.motors["gripper"], 800)
```

### 步骤8：实现观测和动作

```python
    def get_observation(self) -> dict[str, Any]:
        """获取观测数据"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        # 1. 读取电机位置
        obs = self.bus.sync_read("Present_Position")
        obs = {f"{m}.pos": v for m, v in obs.items()}  # 重命名：joint_name.pos

        # 2. 读取相机图像（异步，非阻塞）
        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()

        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """发送动作指令"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        # 1. 解析动作字典：提取关节目标位置
        # 格式：{"joint_1.pos": value, "joint_2.pos": value, ...}
        goal_pos = {k.removesuffix(".pos"): v for k, v in action.items()}

        # 2. 安全限幅：限制单步移动量
        if self.config.max_relative_target is not None:
            present = self.bus.sync_read("Present_Position")
            goal_present = {m: (goal_pos[m], present[m]) for m in goal_pos}
            goal_pos = ensure_safe_goal_position(
                goal_present, self.config.max_relative_target
            )

        # 3. 下发到电机总线
        self.bus.sync_write("Goal_Position", goal_pos)

        # 4. 返回实际发送的动作（用于记录数据集）
        return {f"{m}.pos": v for m, v in goal_pos.items()}
```

---

## 电机控制器集成

### 使用现有 MotorsBus 实现

LeRobot 提供了两种现成实现：

1. **FeetechMotorsBus** (`src/lerobot/motors/feetech/`)
   - 支持 STS/SCS 系列舵机
   - 使用 `scservo_sdk`
   - 自动解析控制表
   - 参考：`SO100Follower`, `SO101Follower`, `HopeJrArm`

2. **DynamixelMotorsBus** (`src/lerobot/motors/dynamixel/`)
   - 支持 Dynamixel 舵机
   - 使用 `dynamixel_sdk`
   - 类似接口设计

**构造示例**：
```python
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

motors = {
    "joint_1": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
    "joint_2": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}
bus = FeetechMotorsBus(
    port="/dev/ttyUSB0",
    motors=motors,
    calibration=self.calibration
)
```

### 实现自定义 MotorsBus

如果使用自研驱动板或其他品牌电机：

1. 在 `src/lerobot/motors/` 中创建新模块
2. 继承 `MotorsBus` 抽象基类 (`motors/motors_bus.py`)
3. 实现以下抽象方法：
   - `_assert_protocol_is_compatible` - 协议兼容性检查
   - `_handshake` - 硬件握手
   - `_find_single_motor` - 扫描单个电机
   - `configure_motors` - 配置参数
   - `read/write` - 寄存器读写
   - `sync_read/sync_write` - 批量读写
   - `disable_torque/enable_torque` - 扭矩管理
   - `read_calibration/write_calibration` - 校准管理

4. 在机器人类中替换为新的总线实例

---

## 相机系统集成

### 使用现有 Camera 实现

LeRobot 提供多种相机后端：
- **OpenCVCamera** - USB摄像头
- **RealSenseCamera** - Intel RealSense RGB-D相机
- **Reachy2Camera** - Reachy2头部相机
- 可自定义 `Camera` 子类

**配置示例**：
```python
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

cameras = {
    "front": OpenCVCameraConfig(
        index_or_path=0, fps=30, width=640, height=480
    ),
    "wrist": RealSenseCameraConfig(
        serial_number="123456", fps=30, width=640, height=480
    ),
}
```

### 在 Robot 中使用相机

```python
from lerobot.cameras.utils import make_cameras_from_configs

class MyRobot(Robot):
    def __init__(self, config: MyRobotConfig):
        super().__init__(config)
        # 自动创建相机实例
        self.cameras = make_cameras_from_configs(config.cameras)

    def connect(self, calibrate: bool = True):
        self.bus.connect()
        # 连接所有相机
        for cam in self.cameras.values():
            cam.connect()
        self.configure()

    def get_observation(self):
        obs = self.bus.sync_read("Present_Position")
        obs = {f"{m}.pos": v for m, v in obs.items()}
        # 异步读取图像
        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()
        return obs

    def disconnect(self):
        self.bus.disconnect()
        # 断开相机
        for cam in self.cameras.values():
            cam.disconnect()
```

---

## 遥操作支持

### 实现 Teleoperator 子类

如果需要 Leader 机械臂或手柄进行遥操作：

1. **创建目录结构**：
```
src/lerobot/teleoperators/my_teleop/
├── __init__.py
├── config_my_teleop.py
└── my_teleop.py
```

2. **定义配置类**：
```python
from lerobot.teleoperators.config import TeleoperatorConfig

@TeleoperatorConfig.register_subclass("my_teleop")
@dataclass
class MyTeleopConfig(TeleoperatorConfig):
    port: str = "/dev/ttyUSB1"
    use_present_position: bool = True
```

3. **实现 Teleoperator**：
```python
from lerobot.teleoperators.teleoperator import Teleoperator

class MyTeleop(Teleoperator):
    config_class = MyTeleopConfig
    name = "my_teleop"

    def __init__(self, config: MyTeleopConfig):
        super().__init__(config)
        self.config = config
        # 创建电机总线
        self.bus = FeetechMotorsBus(port=config.port, motors=..., calibration=...)

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{m}.pos": float for m in self.bus.motors}

    def get_action(self) -> dict[str, Any]:
        """从Leader设备读取动作"""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        # 读取当前位置或目标位置
        key = "Present_Position" if self.config.use_present_position else "Goal_Position"
        positions = self.bus.sync_read(key)
        return {f"{m}.pos": v for m, v in positions.items()}

    # 实现 connect/disconnect/is_connected 等...
```

参考实现：
- `src/lerobot/teleoperators/so_leader/`
- `src/lerobot/teleoperators/reachy2_teleoperator/`

---

## 配置与注册

### 1. 注册 RobotConfig

确保在配置类上使用装饰器：
```python
@RobotConfig.register_subclass("my_robot")
@dataclass
class MyRobotConfig(RobotConfig):
    ...
```

### 2. 注册到工厂函数

在 `src/lerobot/robots/utils.py` 的 `make_robot_from_config()` 中添加：
```python
def make_robot_from_config(config: RobotConfig) -> Robot:
    if config.type == "so100_follower":
        from lerobot.robots.so_follower import SO100Follower
        return SO100Follower(config)
    elif config.type == "so101_follower":
        from lerobot.robots.so_follower import SO101Follower
        return SO101Follower(config)
    elif config.type == "my_robot":
        from lerobot.robots.my_robot import MyRobot
        return MyRobot(config)
    # ...
```

### 3. 添加配置示例

在 `src/lerobot/configs/` 中创建示例：
```python
from dataclasses import dataclass
from lerobot.robots.my_robot.config_my_robot import MyRobotConfig

@dataclass
class MyRobotExperimentConfig:
    robot: MyRobotConfig = MyRobotConfig(port="/dev/ttyUSB0")
    # 其他配置...
```

---

## 完整代码模板

以下是完整的 `my_robot.py` 模板，整合了前面所有步骤：

```python
# src/lerobot/robots/my_robot/my_robot.py

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.robots.robot import Robot
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_my_robot import MyRobotConfig

logger = logging.getLogger(__name__)


class MyRobot(Robot):
    """MyRobot 实现 - 6自由度机械臂 + 抓手"""

    config_class = MyRobotConfig
    name = "my_robot"

    def __init__(self, config: MyRobotConfig):
        super().__init__(config)
        self.config = config

        # 初始化电机总线
        norm_mode_body = (
            MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        )
        self.bus = FeetechMotorsBus(
            port=config.port,
            motors={
                "joint_1": Motor(1, "sts3215", norm_mode_body),
                "joint_2": Motor(2, "sts3215", norm_mode_body),
                "joint_3": Motor(3, "sts3215", norm_mode_body),
                "joint_4": Motor(4, "sts3215", norm_mode_body),
                "joint_5": Motor(5, "sts3215", norm_mode_body),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

        # 初始化相机
        self.cameras = make_cameras_from_configs(config.cameras)

    # ===== 特征声明 =====

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{m}.pos": float for m in self.bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    # ===== 连接管理 =====

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected")

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()
        logger.info(f"{self} disconnected")

    # ===== 校准 =====

    def calibrate(self) -> None:
        """交互式校准"""
        if self.calibration:
            user_input = input("Use stored calibration? Press ENTER to use, 'c' to recalibrate: ")
            if user_input.strip().lower() != "c":
                self.bus.write_calibration(self.calibration)
                return

        self.bus.disable_torque()

        # 半转归位
        homing_offsets = self.bus.set_half_turn_homings()

        # 记录运动范围
        print("Move all joints to MINIMUM position, then press Enter...")
        input()
        mins = self.bus.sync_read("Present_Position")

        print("Move all joints to MAXIMUM position, then press Enter...")
        input()
        maxs = self.bus.sync_read("Present_Position")

        # 保存校准
        self.calibration = {
            motor: MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=mins[motor],
                range_max=maxs[motor],
            )
            for motor, m in self.bus.motors.items()
        }
        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        logger.info("Calibration completed")

    # ===== 配置 =====

    def configure(self) -> None:
        """配置电机参数"""
        with self.bus.torque_disabled():
            self.bus.configure_motors()

            for motor in self.bus.motors:
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

            # 抓手扭矩限制
            if "gripper" in self.bus.motors:
                self.bus.write("Max_Torque_Limit", self.bus.motors["gripper"], 300)
                self.bus.write("Protection_Current", self.bus.motors["gripper"], 800)

    # ===== 核心 I/O =====

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        # 读取电机位置
        obs = self.bus.sync_read("Present_Position")
        obs = {f"{m}.pos": v for m, v in obs.items()}

        # 读取相机
        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()

        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        # 解析动作
        goal_pos = {k.removesuffix(".pos"): v for k, v in action.items()}

        # 安全限幅
        if self.config.max_relative_target is not None:
            present = self.bus.sync_read("Present_Position")
            goal_present = {m: (goal_pos[m], present[m]) for m in goal_pos}
            goal_pos = ensure_safe_goal_position(
                goal_present, self.config.max_relative_target
            )

        # 下发动作
        self.bus.sync_write("Goal_Position", goal_pos)

        return {f"{m}.pos": v for m, v in goal_pos.items()}
```

---

## 测试与验证指南

### 单元测试（Mock）

在 `tests/robots/test_my_robot.py` 中：

```python
import pytest
from unittest.mock import MagicMock, patch
from lerobot.robots.my_robot import MyRobot, MyRobotConfig


@pytest.fixture
def robot():
    """创建带 mock 总线的机器人"""
    config = MyRobotConfig(port="/dev/ttyUSB0")

    # mock FeetechMotorsBus
    with patch("lerobot.robots.my_robot.my_robot.FeetechMotorsBus") as mock_bus_class:
        mock_bus = MagicMock()
        mock_bus.motors = {
            "joint_1": MagicMock(id=1),
            "joint_2": MagicMock(id=2),
            # ...
        }
        mock_bus.is_connected = False
        mock_bus.is_calibrated = True
        mock_bus_class.return_value = mock_bus

        robot = MyRobot(config)
        yield robot, mock_bus


def test_connect(robot):
    robot, mock_bus = robot
    robot.connect()

    mock_bus.connect.assert_called_once()
    assert robot.is_connected is True


def test_get_observation(robot):
    robot, mock_bus = robot
    robot.connect()

    # Mock 电机读数
    mock_bus.sync_read.return_value = {
        "joint_1": 0.5,
        "joint_2": -0.3,
        # ...
    }

    obs = robot.get_observation()

    # 验证键名
    assert "joint_1.pos" in obs
    assert "joint_2.pos" in obs
    # 验证调用了 sync_read
    mock_bus.sync_read.assert_called_with("Present_Position")


def test_send_action(robot):
    robot, mock_bus = robot
    robot.connect()

    # Mock 当前位置用于安全限幅
    mock_bus.sync_read.return_value = {
        "joint_1": 0.0,
        "joint_2": 0.0,
    }

    action = {"joint_1.pos": 0.5, "joint_2.pos": -0.3}
    robot.send_action(action)

    # 验证动作下发
    mock_bus.sync_write.assert_called_once()
    call_args = mock_bus.sync_write.call_args[0]
    assert call_args[0] == "Goal_Position"  # 寄存器名
    assert "joint_1" in call_args[1]
    assert "joint_2" in call_args[1]
```

### 测试覆盖率建议

- **连接/断开**：状态切换、重复连接异常
- **观测**：返回键名与 `observation_features` 一致、数值正确
- **动作**：正确解析、安全限幅、下发到总线
- **校准**：交互式流程、校准文件读写
- **异常**：未连接时调用方法抛出 `DeviceNotConnectedError`

### 真实硬件验证

1. **连接测试**：
```bash
pytest tests/robots/test_my_robot.py::test_connect -s
```

2. **手动验证观测**：
```python
from lerobot.robots.my_robot import MyRobot, MyRobotConfig

robot = MyRobot(MyRobotConfig(port="/dev/ttyUSB0"))
robot.connect()
obs = robot.get_observation()
print(f"Obs keys: {list(obs.keys())}")
robot.disconnect()
```

3. **遥操作闭环**（如适用）：
```bash
python examples/my_robot_teleop/teleoperate.py
```

---

## 常见问题与调试技巧

### 问题排查表

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| `DeviceAlreadyConnectedError` | 重复调用 `connect()` | 使用 `if not robot.is_connected:` 保护或上下文管理器 |
| `DeviceNotConnectedError` | 硬件断开或连接失败 | 检查串口/网络，确认无其他进程占用 |
| 串口连接失败 | 波特率/ID不匹配 | 运行 `lerobot-find-port` 确认端口，检查 `protocol_version` |
| 校准状态为 False | 校准文件与电机实际状态不符 | 删除校准文件重新执行 `calibrate()` |
| 相机无法连接 | 配置缺少 `width/height/fps` | 检查 `RobotConfig` 校验，安装对应驱动 |
| 动作抖动/延迟 | PID参数或 `max_relative_target` 过小 | 调整 `configure()` 中的PID，增大限幅 |
| 遥操作映射不一致 | Leader和Follower键名不匹配 | 确保双方 `action_features` 完全一致 |

### 调试技巧

1. **日志记录**：
```python
import time

def get_observation(self):
    t0 = time.perf_counter()
    obs = self.bus.sync_read("Present_Position")
    dt = time.perf_counter() - t0
    logger.debug(f"Read motors took {dt*1000:.2f}ms")
    return obs
```

2. **串口调试**：
```bash
# Linux/macOS
ls -l /dev/ttyUSB*
# 测试串口权限
cat /dev/ttyUSB0
```

3. **MotorBus 调试工具**：
```python
# 扫描端口
from lerobot.motors.feetech import FeetechMotorsBus
FeetechMotorsBus.scan_port("/dev/ttyUSB0")

# 读取单个寄存器
bus = FeetechMotorsBus(port="/dev/ttyUSB0", motors={"test": Motor(1, "sts3215")})
bus.connect()
pos = bus.read("Present_Position", bus.motors["test"])
print(f"Position: {pos}")
```

4. **校准文件位置**：
```
HF_LEROBOT_CALIBRATION/
└── robots/
    └── my_robot/
        └── <robot_id>.json
```

---

## 最佳实践

### 1. 命名规范
- **关节键名**：使用 `<joint_name>.pos` 格式（如 `joint_1.pos`, `gripper.pos`）
- **机器人名称**：小写字母 + 下划线（`my_robot`），与注册名称保持一致
- **文件命名**：`config_my_robot.py`, `my_robot.py`

### 2. 错误处理
- 优先抛出 `DeviceNotConnectedError` 和 `DeviceAlreadyConnectedError`
- 使用 `try/except` 包装硬件调用，提供有意义的错误信息
- 不要吞掉底层异常，使用 `raise ... from ...` 保留调用链

### 3. 日志记录
- 使用 `logging.getLogger(__name__)` 创建日志器
- 在关键操作（连接/断开/校准）记录 `info` 级别
- 在数据采样记录 `debug` 级别，便于性能分析
- 日志消息包含设备标识：
```python
logger.info(f"{self}: Connected to {self.config.port}")
```

### 4. 安全守则
- 默认在 `disconnect()` 时禁用扭矩（可配置）
- 对抓手或高扭矩关节设置 `Max_Torque_Limit` 和 `Protection_Current`
- 始终实现 `max_relative_target` 限制单步移动量
- 校准流程中提示用户保持安全距离

### 5. 代码质量
- 遵循仓库 ruff 配置（4空格、双引号、110字符行宽）
- 添加类型注解，使用 `type | None` 而不是 `Optional[type]`
- 复杂逻辑添加注释，说明设计决策
- 不要在模块顶层实例化硬件（便于 mock）

### 6. 测试友好
- 所有硬件初始化放在 `__init__` 或 `connect()`，不在 import 时执行
- 复杂机器人提供灵活的 config 选项，支持部分组件禁用
- 为可选功能（如相机）提供跳过机制：
```python
if self.config.cameras:
    self.cameras = make_cameras_from_configs(config.cameras)
else:
    self.cameras = {}
```

### 7. 文档同步
- 更新 `docs/source/integrate_hardware.mdx` 提及新机器人
- 在 `examples/` 提供最小运行示例
- PR描述中说明硬件版本、依赖、测试命令

### 8. 提交建议

```bash
# 1. 运行测试
pytest tests/robots/test_my_robot.py -xvs

# 2. 代码检查
pre-commit run --files src/lerobot/robots/my_robot/*.py

# 3. 类型检查
mypy src/lerobot/robots/my_robot/

# 4. 提交信息
feat(robots): add MyRobot support

- Add MyRobotConfig with port/cameras/safety parameters
- Implement 6-dof arm with gripper control
- Support interactive calibration via MotorsBus
- Add comprehensive unit tests with mock
- Include example configuration in lerobot/configs/
```

---

## 参考资源

- **官方文档**：`docs/source/integrate_hardware.mdx`
- **现有实现**：
  - `src/lerobot/robots/so_follower/` - Feetech机械臂
  - `src/lerobot/robots/reachy2/` - 外部SDK示例
  - `src/lerobot/robots/hope_jr/` - 多总线+GUI校准
- **测试示例**：`tests/robots/test_so100_follower.py`
- **遥操作**：`src/lerobot/teleoperators/so_leader/`

---

## 总结

通过遵循本指南，你可以系统地将新机器人集成到 LeRobot 中：

✅ **架构清晰**：统一接口、组件解耦、配置驱动
✅ **功能完整**：电机控制、相机采集、遥操作闭环
✅ **安全可靠**：扭矩管理、动作限幅、校准持久化
✅ **测试充分**：mock隔离、单元测试、硬件验证
✅ **文档完善**：代码注释、使用示例、调试指南

开始集成你的机器人吧！🚀
