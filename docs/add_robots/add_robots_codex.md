# LeRobot 新机器人集成技术说明（Codex版）

> 面向想为 LeRobot（`src/lerobot/`）贡献新硬件支持的开发者。本文以现有实现（如 `so100_follower`、`so101_follower`、`reachy2`、`hope_jr` 等）为参照，梳理架构理念、所需依赖、实现步骤、测试方法以及常见陷阱，以便独立完成从驱动集成到遥操作的全流程工作。

## 目录

1. [概述：架构与设计理念](#概述架构与设计理念)
2. [前置准备](#前置准备)
3. [集成流程总览](#集成流程总览)
4. [实现机器人类的详细步骤](#实现机器人类的详细步骤)
5. [电机控制器集成策略](#电机控制器集成策略)
6. [相机系统集成策略](#相机系统集成策略)
7. [遥操作支持与闭环采集](#遥操作支持与闭环采集)
8. [配置与注册](#配置与注册)
9. [完整代码模板](#完整代码模板)
10. [测试与验证指南](#测试与验证指南)
11. [常见问题与调试技巧](#常见问题与调试技巧)
12. [最佳实践与提交建议](#最佳实践与提交建议)

---

## 概述：架构与设计理念

### 模块化架构

LeRobot 将机器人堆栈拆分为多层抽象：

```
src/lerobot/
├── robots/
│   ├── robot.py          # Robot 抽象基类（connect/get_observation/send_action...）
│   ├── config.py         # RobotConfig 抽象配置，内置 camera 配置校验
│   ├── utils.py          # make_robot_from_config、ensure_safe_goal_position 等工具
│   └── <robot_name>/     # 具体机器人实现（so100_follower、reachy2、hope_jr...）
├── motors/               # MotorsBus 抽象 + Feetech/Dynamixel 等具体实现
│   └── feetech/          # FeetechMotorsBus、OperatingMode、RangeFinderGUI...
├── cameras/              # Camera 抽象、OpenCV/RealSense/Reachy2 Camera 等
└── teleoperators/        # Teleoperator 抽象与 so100_leader、reachy2_teleoperator 等实现
```

该结构保证：
- **统一接口**：`robots/robot.py` 和 `teleoperators/teleoperator.py` 定义必须实现的接口，使训练脚本与硬件解耦。
- **组件解耦**：电机总线、相机、遥操作均为独立模块，可按需组合（例如 `SO100Follower` 既复用 Feetech 总线，也通过 `make_cameras_from_configs` 插入任意相机）。
- **配置驱动**：所有机器人配置继承自 `RobotConfig`，通过 `@RobotConfig.register_subclass("<type>")` 向 `draccus` CLI 注册。
- **校准和安全**：基类负责在 `HF_LEROBOT_CALIBRATION/robots/<name>/<id>.json` 自动读写 `MotorCalibration`，并提供 `ensure_safe_goal_position()` 限制动作跳变。像 `so100_follower.py` 会在连接时比对校准并可选择重新标定。

### 设计原则

1. **声明型特征描述**：`observation_features` / `action_features` 提供平坦字典结构定义（键名通常为 `<joint>.pos`），使数据记录和策略编码保持一致。
2. **面向硬件异常的显式错误处理**：统一使用 `DeviceAlreadyConnectedError`、`DeviceNotConnectedError` 等异常（来自 `lerobot.utils.errors`），便于上层脚本退避。
3. **可复现性**：校准文件和配置均为可序列化 dataclass，配合 `lerobot/configs/parser.py` 的 CLI 解析，支持 `lerobot-record --robot.id=my_arm ...` 或 `python -m lerobot.scripts.lerobot_record --robot.id=my_arm ...` 等命令。
4. **可测试性**：所有公开机器人都对应 pytest 基准，如 `tests/robots/test_so100_follower.py` 通过 mock 模拟 Feetech 总线；`tests/robots/test_reachy2.py` 则模拟 SDK 与摄像头。

---

## 前置准备

### 硬件需求

- **驱动链路**：Serial-USB 连接（Feetech/Dynamixel 总线）、以太网（Reachy2 SDK）或自定义控制器。
- **电机与编码器**：需确认型号、协议版本、可用的控制表；`Motor` dataclass 需要 `id`、`model`、`norm_mode`（`MotorNormMode.RANGE_M100_100`、`RANGE_0_100`、`DEGREES`）。
- **相机**：RGB-D、UVC 或 SDK 特定相机（Intel RealSense、Reachy 头部相机等）。`RobotConfig.__post_init__` 要求所有摄像头配置提供 `width/height/fps`。
- **遥操作器（可选）**：若需要闭环采集，提前规划匹配的 teleoperator（如 `so100_leader`、`gamepad`）。

### 软件与环境

```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
pip install -e ".[dev,test]"
pre-commit install
```

额外驱动：
- Feetech：`pip install scservo-sdk`
- Dynamixel：`pip install dynamixel-sdk`
- RealSense：`pip install pyrealsense2`
- Reachy2：安装官方 `reachy2_sdk`

建议在 Linux/macOS 上开发，Windows 请确认串口名称格式。

### 阅读参考实现

重点阅读下列文件以掌握通用模式：
- `src/lerobot/robots/so_follower/so_follower.py`：典型 Feetech 六轴机械臂，覆盖连接、校准、动作限幅与相机读流。
- `src/lerobot/robots/so_follower/so_follower.py`：同一文件内包含 SO101 变体（`SO101Follower`），具备不同的运动范围流程。
- `src/lerobot/robots/reachy2/robot_reachy2.py`：演示如何包装外部 SDK 并同时驱动移动底盘。
- `src/lerobot/robots/hope_jr/*`：展示多路总线（arm/hand）与 GUI 校准 (`RangeFinderGUI`) 的使用方式。
- `src/lerobot/motors/motors_bus.py`：理解 `MotorsBus` 抽象提供的 `connect/sync_read/sync_write/configure_motors` 等 API 及扫描、校准辅助函数。
- `src/lerobot/cameras/*` 与 `make_cameras_from_configs()`：掌握相机工厂与异步采集接口。
- `src/lerobot/teleoperators/*`：了解闭环采集所需的 action/feedback 标准。

---

## 集成流程总览

1. **定义配置**：为新机器人撰写 `@RobotConfig.register_subclass("my_robot")` 的 dataclass，并声明端口/相机/安全参数。
2. **实现 Robot 子类**：在 `src/lerobot/robots/my_robot/my_robot.py` 继承 `Robot`，实现所有抽象属性和方法。
3. **组合硬件组件**：根据硬件类型实例化 `FeetechMotorsBus`、`DynamixelMotorsBus` 或自定义电机控制器，并调用 `make_cameras_from_configs()` 生成相机实例。
4. **实现核心流程**：补全 `connect/calibrate/configure/get_observation/send_action/disconnect`，合理使用 `ensure_safe_goal_position()`、`MotorCalibration`、`MotorNormMode`。
5. **注册工厂**：在 `robots/utils.py` 的 `make_robot_from_config()` 中增加 `elif config.type == "my_robot": ...`，以便 CLI / dataset loader 自动实例化。
6. **遥操作（可选）**：若需要搭配新的 teleoperator，同步在 `src/lerobot/teleoperators/` 下新增实现，并在其 `utils.make_teleoperator_from_config()` 注册。
7. **配置示例**：在 `lerobot/configs/default.py` 或 `docs/examples` 中给出 `robot_config=MyRobotConfig(...)` 的使用示例。
8. **编写测试**：遵循 `tests/robots/test_so100_follower.py` 模式使用 mock 验证连接、观测、动作与异常，必要时补充集成测试脚本。
9. **文档与演示**：在 `docs/` 或 `examples/` 增加说明，确保贡献者能复现你的流程。

---

## 实现机器人类的详细步骤

### 1. 创建目录与 `__init__.py`

```
src/lerobot/robots/my_robot/
├── __init__.py
├── config_my_robot.py
└── my_robot.py
```

`__init__.py` 中导出 `MyRobot` 与 `MyRobotConfig`，以便 `from lerobot.robots.my_robot import MyRobot` 生效。

### 2. 定义配置（`config_my_robot.py`）

- 继承 `RobotConfig` 并使用 `@RobotConfig.register_subclass("my_robot")` 装饰。
- 明确硬件端口/IP、相机配置、可选参数（扭矩释放、单位、最大移动量等）。
- 如果需要多模块（如 `HopeJrArm` + `HopeJrHand`），可以创建多个配置类并共享基础字段。

示例参见 `SO100FollowerConfig`（`src/lerobot/robots/so_follower/config_so_follower.py`）。

### 3. 构造函数

在 `MyRobot.__init__()` 中：
- 调用 `super().__init__(config)` 以初始化 `robot_type/id/calibration`。
- 构建电机总线：通过 `FeetechMotorsBus(port=..., motors={"joint": Motor(id, model, norm_mode)})` 实例化；
  - `Motor` 使用 `MotorNormMode` 声明输出单位（与策略输出一致）。
  - 如果使用自研控制器，可实现 `MotorsBus` 子类并在此引用。
- 初始化相机：`self.cameras = make_cameras_from_configs(config.cameras)`。
- 其他资源：外部 SDK 客户端（如 `ReachySDK`）。

### 4. 特征声明

- `observation_features`：返回 {"joint.pos": float, "cam_front": (H, W, 3)} 等；可使用 `cached_property` 避免重复构造。
- `action_features`：通常与电机特征一致（SO100 直接复用 `_motors_ft`）。
- 视觉/额外传感器：如 `Reachy2Robot.motors_features` 同时覆盖关节和移动底盘速度。

### 5. 连接与断开

- 在 `connect()` 中：
  - 检查 `self.is_connected`，避免重复连接。
  - 打开 MotorBus (`self.bus.connect()`)，根据需要执行握手/校准。
  - 对每个相机场景调用 `cam.connect()`。
  - 执行 `self.configure()`，设置电机模式、PID、力矩限制等。
  - 记录日志：`logger.info(f"{self} connected.")`。
- `disconnect()`：
  - 如果 `not self.is_connected` 则抛出 `DeviceNotConnectedError`。
  - `self.bus.disconnect(self.config.disable_torque_on_disconnect)`。
  - 断开所有相机或第三方 SDK，必要时释放扭矩（`Reachy2Robot` 会根据配置调用 `turn_off_smoothly()`）。

### 6. 校准

- 若总线支持 `MotorCalibration`，在 `calibrate()` 中：
  - 可提示用户选择沿用旧文件还是重新标定（SO100 的交互式 input）。
  - 使用 `self.bus.disable_torque()`、`set_half_turn_homings()`、`record_ranges_of_motion()` 或 `RangeFinderGUI`（`hope_jr_arm.py`）。
  - 将结果写入电机并 `_save_calibration()`。
- 对于不需要校准的机器人（如 `Reachy2Robot`），`is_calibrated` 始终返回 `True`。

### 7. configure()

- 在 `with self.bus.torque_disabled():` 块内调用 `self.bus.configure_motors()`。
- 根据硬件写入控制模式、PID/加速度限制、安全阈值。
- 例如 `SO100Follower.configure()` 将操作模式设为位置模式，并降低 P/I/D 以避免震荡。

### 8. 观测与动作

- `get_observation()`：
  - 确保连接，否则抛出 `DeviceNotConnectedError`。
  - 使用 `self.bus.sync_read("Present_Position")` 或 SDK API 获取状态，并重命名为 `<joint>.pos`。
  - 对每台相机调用 `cam.async_read()`；注意记录执行时间（`logger.debug`）。
  - 如需额外状态（里程计、电池），按同样的键值加入字典。
- `send_action()`：
  - 解析 `action` 字典为 `goal_pos`。
  - 可读取当前位姿并调用 `ensure_safe_goal_position()` 进行限幅。
  - 使用 `self.bus.sync_write("Goal_Position", goal_pos)` 或调用 SDK 的 `goal_position` 接口。
  - 返回实际发送的动作（用于日志与数据集对齐）。

### 9. 其他辅助

- `setup_motors()`：当设备首次使用且需要为每个电机写入 ID/波特率时，可照搬 `SO100Follower.setup_motors()` 的交互式流程。
- `logs`：对于 SDK 设备，可记录 `read_pos_dt_s` 和 `write_pos_dt_s` 等计时指标以便后续监控。

---

## 电机控制器集成策略

### MotorsBus 抽象（`motors/motors_bus.py`）

- **统一 API**：`connect() / disconnect() / sync_read() / sync_write() / write() / read()`。
- **安全工具**：`torque_disabled()` 上下文管理器、`set_half_turn_homings()`、`record_ranges_of_motion()`、`scan_port()` 等。
- **校准结构**：`MotorCalibration` 包含 `drive_mode/homing_offset/range_min/range_max`；`MotorNormMode` 控制归一化单位。

### Feetech 示例

- `FeetechMotorsBus`（`motors/feetech/feetech.py`）支持 Protocol 0/1，包含 `_handshake()`、`OperatingMode`、`DriveMode` 等；相关例子：`SO100Follower`、`HopeJrArm`、`HopeJrHand`。
- 常用流程：
  1. `self.bus.connect()`；
  2. `self.bus.configure_motors()` 设置返回延迟/加速度；
  3. `self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)`；
  4. 通过 `RangeFinderGUI` 或 `record_ranges_of_motion()` 获取校准数据；
  5. `self.bus.sync_read("Present_Position")` 与 `self.bus.sync_write("Goal_Position", goal_pos)` 完成动作闭环。

### Dynamixel 或自定义电机

- 若使用 Dynamixel，需要在 `motors/dynamixel/` 中参考现有子类（若仓库已提供）。若需要新控制器：
  - 继承 `MotorsBus`，实现 `_handshake/_find_single_motor/configure_motors/read/write` 等抽象。
  - 在机器人类中替换为新的总线实例。
- 也可直接与厂商 SDK 通信（例如 `Reachy2Robot` 直接持有 `ReachySDK` 实例），但要确保 `send_action()` 与 `get_observation()` 输出符合 LeRobot 的特征规范。

### 控制模式与安全

- 使用 `MotorNormMode` 与 `MotorCalibration` 配合 `self.bus.normalize/unnormalize`，确保策略输出和原始编码一致。
- `ensure_safe_goal_position()`（`robots/utils.py`）可对 `goal_present_pos` 字典进行限幅，`max_relative_target` 支持 float 或 dict。
- 需要额外安全机制（扭矩限制、保护电流）时，可参考 `SO100Follower.configure()` 对抓手设置 `Max_Torque_Limit/Protection_Current`。

---

## 相机系统集成策略

- 相机配置继承 `cameras/configs.py` 中的 `CameraConfig`。现有类型包括：
  - `opencv.OpenCVCamera`（UVC 相机）；
  - `realsense.RealSenseCamera`；
  - `reachy2_camera.Reachy2Camera`；
  - 自定义 `Camera` 子类（需实现 `is_connected/find_cameras/connect/read/async_read/disconnect`）。
- 机器人中使用 `make_cameras_from_configs()` 自动创建实例；如配置不合法，会抛出 `ValueError`，需在配置阶段确保参数完整。
- `get_observation()` 中调用 `cam.async_read()`（非阻塞）或 `cam.read()`。建议捕获超时并记录日志，以免数据管线阻塞。
- 示例 `config`：

```python
from lerobot.cameras.opencv import OpenCVCameraConfig
cameras = {
    "front": OpenCVCameraConfig(device_index=0, width=640, height=480, fps=30),
    "wrist": OpenCVCameraConfig(device_index=1, width=640, height=480, fps=30, rotation=90),
}
```

- 如需特殊相机（如双目）、深度图，可扩展 `CameraConfig` 字段并在自定义 `Camera` 子类中处理。

---

## 遥操作支持与闭环采集

- 所有 teleoperator 需继承 `teleoperators/teleoperator.py`，实现与 `Robot` 对应的 `action_features/get_action`、`feedback_features/send_feedback` 等接口。
- 示例：`so_leader/so_leader.py` 与 `SO100Follower` 搭配，二者的关节键名完全一致，使 `teleoperate.py` 示例可以直接映射 leader/follower。
- `Reachy2Teleoperator`（`teleoperators/reachy2_teleoperator/`) 展示了如何使用同一个 SDK 控制不同驱动（移动底盘/手臂）。
- 若新增 teleoperator：
  1. 创建 `src/lerobot/teleoperators/my_teleop/`；
  2. 定义配置（继承 `TeleoperatorConfig`，参见 `config.py`）；
  3. 实现动作/反馈逻辑，必要时共享 `MotorCalibration`；
  4. 在 `teleoperators/utils.py` 的工厂函数中注册。
- 采集脚本可在 `examples/` 仿照 `examples/so100_to_so100_EE/record.py`，通过 `teleop.get_action()` -> `robot.send_action()` -> `robot.get_observation()` 完成数据记录。

---

## 配置与注册

1. **配置类**：`RobotConfig` 与 `TeleoperatorConfig` 均继承自 `draccus.ChoiceRegistry`，必须使用 `@register_subclass` 指定唯一类型字符串（用于 CLI `--robot.type=my_robot`）。
2. **工厂函数**：
   - `robots/utils.py` 内的 `make_robot_from_config()` 需要 `elif config.type == "my_robot": from .my_robot import MyRobot; return MyRobot(config)`。
   - 若想避免频繁修改中枢文件，可通过 `lerobot.utils.import_utils.make_device_from_device_class()` 动态创建，但要保证模块在 `PYTHONPATH` 中可见。
3. **配置入口**：
   - 训练/评估配置 (`lerobot/configs/train.py`, `eval.py`) 会经由 `parser.wrap()` 自动解析 `robot` 字段；请确保新类型能够从 CLI 创建。
   - 如果需要默认配置，可在 `lerobot/configs/default.py` 中追加字段或在 `docs/source/*.mdx` 介绍使用方式。
4. **校准存储**：`Robot` 基类会根据 `self.name` + `config.id` 决定校准文件存放路径。提供 `calibration_dir` 可覆盖默认位置。

---

## 完整代码模板

以下模板综合 `SO100Follower` 的常见逻辑，可按需修改（请根据硬件替换注释段落）：

```python
# src/lerobot/robots/my_robot/my_robot.py

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
    config_class = MyRobotConfig
    name = "my_robot"

    def __init__(self, config: MyRobotConfig):
        super().__init__(config)
        self.config = config
        self.bus = FeetechMotorsBus(
            port=config.port,
            motors={
                "joint_1": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
                "joint_2": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
                # ...
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{m}.pos": float for m in self.bus.motors}

    @cached_property
    def observation_features(self) -> dict[str, Any]:
        camera_ft = {
            cam: (cfg.height, cfg.width, 3) for cam, cfg in self.config.cameras.items()
        }
        return {**self._motors_ft, **camera_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            self.calibrate()
        for cam in self.cameras.values():
            cam.connect()
        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            user_input = input("Press ENTER to use stored calibration, or type 'c' to re-run: ")
            if user_input.strip().lower() != "c":
                self.bus.write_calibration(self.calibration)
                return
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
        homing_offsets = self.bus.set_half_turn_homings()
        range_mins, range_maxes = self.bus.record_ranges_of_motion()
        self.calibration = {
            motor: MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )
            for motor, m in self.bus.motors.items()
        }
        self.bus.write_calibration(self.calibration)
        self._save_calibration()

    def configure(self) -> None:
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor in self.bus.motors:
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        obs = {f"{m}.pos": val for m, val in self.bus.sync_read("Present_Position").items()}
        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        goal_pos = {key[:-4]: val for key, val in action.items() if key.endswith(".pos")}
        if self.config.max_relative_target is not None:
            present = self.bus.sync_read("Present_Position")
            goal_present = {k: (goal_pos[k], present[k]) for k in goal_pos}
            goal_pos = ensure_safe_goal_position(goal_present, self.config.max_relative_target)
        self.bus.sync_write("Goal_Position", goal_pos)
        return {f"{m}.pos": v for m, v in goal_pos.items()}

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()
        logger.info(f"{self} disconnected.")
```

根据硬件特性替换：
- 外部 SDK（如 `ReachySDK`）可替代 `self.bus`
- 若需要移动底盘或其他状态，扩展 `observation_features` 和 `get_observation()`

---

## 测试与验证指南

### 单元测试（mock）

- 参考 `tests/robots/test_so100_follower.py`：
  - 使用 `unittest.mock.patch` 替换 `FeetechMotorsBus`，模拟 `connect/sync_read/sync_write` 行为。
  - 编写 pytest fixture 负责 `robot.connect()` 与 teardown。
  - 覆盖以下用例：
    1. `connect()`/`disconnect()` 状态切换；
    2. `get_observation()` 返回期望键名和数值；
    3. `send_action()` 将动作正确发送到总线；
    4. 异常路径（未连接时调用方法）。
- SDK 型设备（`Reachy2Robot`）可模拟完整对象（MagicMock + 属性），`tests/robots/test_reachy2.py` 展示如何覆盖可选配置。

### 模拟/软件在环

- 若硬件 SDK 提供仿真器（如 Reachy2），可在 CI 中运行软仿测试。
- 对于多电机系统，可提供 `MockMotorsBus`，在测试中验证 `ensure_safe_goal_position()` 逻辑。

### 真实硬件验收

1. 运行 `python examples/<...>/record.py --robot.type=my_robot ...` 检查实时动作。
2. 使用 `lerobot-find-port`（或 `python -m lerobot.scripts.lerobot_find_port`）确定串口并测试波特率。
3. 执行 `pytest tests/robots/test_my_robot.py -k connect -s` 确认基本行为。
4. 若涉及遥操作，运行 `examples/<...>/teleoperate.py` 验证闭环采集。

### 持续集成

- 所有新测试需加入 `tests/robots/` 或 `tests/teleoperators/`。
- 避免真实硬件依赖：使用 mock 或环境变量跳过需要物理设备的测试。

---

## 常见问题与调试技巧

| 问题 | 可能原因 | 排查建议 |
|------|----------|----------|
| `DeviceAlreadyConnectedError` | 重复调用 `connect()` | 在脚本中使用上下文管理或 `try/finally`，必要时包裹在 `if not robot.is_connected` |
| `DeviceNotConnectedError` | 硬件已断开或 `connect()` 失败 | 检查串口/网络，捕获异常记录日志，必要时自动重连 |
| 波特率/ID 不匹配 | 工厂未配置正确 ID 或 `setup_motors()` 未执行 | 使用 `MotorsBus.scan_port()` 和 `setup_motor()` 重新写入 ID/波特率 |
| 校准文件与实际不符 | 机械结构调整或更换电机 | 删除对应 `HF_LEROBOT_CALIBRATION/robots/<name>/<id>.json`，重新 `calibrate()` |
| 相机未连接 | 配置缺少 `width/height/fps` 或驱动未安装 | `RobotConfig.__post_init__` 会抛错；使用 `Camera.find_cameras()` 辅助定位 |
| 动作抖动或延迟 | PID 参数或 `max_relative_target` 过小 | 在 `configure()` 中调整 PID，在配置中放宽 `max_relative_target` |
| 遥操作映射不一致 | Teleoperator 与 Robot 的键名不同 | 保证双方 `action_features`/`observation_features` 完全一致；可定义映射字典 |

调试技巧：
- 使用 `logger.debug` + `time.perf_counter()` 记录采样/下发耗时（参见 `SO100Follower`、`Reachy2Robot`）。
- 在硬件交互前后打印 `self.bus.sync_read` 的原始值，可快速定位归一化/限幅问题。
- 利用 `MotorsBus.set_timeout()` 和 `set_baudrate()` 调整通信容错。

---

## 最佳实践与提交建议

1. **代码风格**：遵循仓库 ruff 配置（4 空格、双引号、110 字符行宽），并尽可能补充类型标注。
2. **日志优先**：使用 `logging` 替代 `print`，并以 `self` 标识设备，便于多机器人环境诊断。
3. **错误处理**：对硬件调用包裹 try/except 并抛出语义化异常；避免吞掉底层错误。
4. **配置清晰**：为所有字段提供 docstring 或注释，说明单位、取值范围。复杂配置可拆分为多个 dataclass。
5. **安全守则**：默认在 `disconnect()` 时禁用扭矩；对抓手或高力矩关节设置合适的限流/限压。
6. **重用工具**：尽量复用 `ensure_safe_goal_position()`、`RangeFinderGUI` 等已有工具，避免重复实现。
7. **测试先行**：新机器人至少具备一个 mock 单测；当修改公共逻辑时同步更新相关测试。
8. **文档同步**：更新 `docs/` 或 `examples/`，并在 PR 描述中给出硬件版本、依赖和测试命令。

通过以上步骤和建议，即可在 LeRobot 中稳健地集成新的机器人硬件，实现与现有训练、评估、遥操作管线的顺畅衔接。
