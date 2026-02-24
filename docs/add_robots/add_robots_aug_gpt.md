# 在 LeRobot 中集成新的真实机器人：开发者技术指南

> 本文面向希望在 **LeRobot** 仓库中直接增加新机器人支持（而非外部插件包）的开发者，结合 `src/lerobot/robots/`、`motors/`、`cameras/`、`teleoperators/` 与 `tests/` 中的实现，总结一套可复用的集成流程与代码模板。

## 1. 架构概述

LeRobot 的硬件集成围绕几类抽象基类展开：

- **Robot 抽象类**（`src/lerobot/robots/robot.py`）
  - 代表“被控制的实体”（机械臂、移动底盘、人形机器人等）。
  - 负责：连接/断开、标定、配置、读取观测 `get_observation()`、发送动作 `send_action()`。
- **RobotConfig 抽象配置类**（`src/lerobot/robots/config.py`）
  - 通过 `draccus.ChoiceRegistry` 注册具体机器人配置，字段包括 `id`、`calibration_dir` 以及自定义字段（串口、是否带某些部件、相机配置等）。
- **MotorsBus 抽象类**（`src/lerobot/motors/motors_bus.py`）
  - 抽象一条串联的电机总线，负责对多个伺服电机进行批量读写（`sync_read`/`sync_write`）。
  - 具体实现：`FeetechMotorsBus`、`DynamixelMotorsBus` 等。
- **Camera 抽象类**（`src/lerobot/cameras/camera.py` + `configs.py`）
  - 抽象相机后端，实现 `connect/read/async_read/disconnect`；配置通过 `CameraConfig` 子类描述。
- **Teleoperator 抽象类**（`src/lerobot/teleoperators/teleoperator.py`）
  - 抽象“人类操作端”（游戏手柄、Leader 机械臂、手机端等），负责产出动作 `get_action()`，接受反馈 `send_feedback()`。

此外还存在若干 **工厂工具**：

- `lerobot.robots.utils.make_robot_from_config(config)`：由 `RobotConfig` 创建具体机器人实例。
- `lerobot.cameras.utils.make_cameras_from_configs(configs)`：由相机配置字典创建相机实例字典。
- `lerobot.teleoperators.utils.make_teleoperator_from_config(config)`：由 `TeleoperatorConfig` 创建遥操作设备。

典型的数据流如下：

1. 用户根据 `RobotConfig`/`TeleoperatorConfig` 构建配置对象（通常从 `lerobot/configs/` 读取）。
2. 通过 `make_robot_from_config` / `make_teleoperator_from_config` 实例化设备。
3. 调用 `connect()` 完成硬件连接和标定/配置。
4. 在控制循环中，Policy → `send_action()` 控制机器人；遥操作设备通过 `get_action()` 生成动作。

## 2. 前置准备

### 2.1 硬件与通信

- 一台可控的机器人：
  - 关节式机械臂、移动平台或其他形态。
- 电机与总线：
  - 首选 **Feetech** 或 **Dynamixel** 伺服电机，总线接口与 SDK 已在 `lerobot.motors` 中实现。
  - 若使用其他电机/控制板，需要能够：
    - 打开端口（串口/CAN/TCP 等）；
    - 读写电机位置/速度等寄存器；
    - 支持批量同步读写更佳。
- 可选相机：
  - OpenCV 相机（USB 摄像头等）或 Intel RealSense，或自定义相机后端。

### 2.2 软件与环境

- Python 3.10+，已安装本仓库：
  - `pip install -e .` 或 `pip install -e ".[dev,test]"`。
- 若使用 Feetech/Dynamixel：
  - 安装对应厂商 SDK（`scservo_sdk`、`dynamixel_sdk`），参见 `CLAUDE.md` 与对应 `tests/motors/` 用法。
- 已正确配置串口权限（Linux 下通常需要将用户加入 `dialout` 组或修改 `/dev/tty*` 权限）。

## 3. 整体集成流程总览

向 LeRobot 增加一个新机器人，一般包含如下步骤：

1. 在 `src/lerobot/robots/<your_robot>/` 下创建：
   - `config_<your_robot>.py`：定义 `RobotConfig` 子类；
   - `<your_robot>.py`：定义具体 `Robot` 子类实现。
2. 在 `RobotConfig` 上使用 `@RobotConfig.register_subclass("your_robot_name")` 注册类型字符串，并定义端口、相机等字段。
3. 在 `Robot` 子类中：
   - 在 `__init__` 中构造 `MotorsBus` 与 `cameras`；
   - 实现 `observation_features` / `action_features`；
   - 实现 `connect/is_connected/disconnect`；
   - 实现 `is_calibrated/calibrate/configure`；
   - 实现核心 I/O：`get_observation`、`send_action`。
4. 若需要遥操作：在 `src/lerobot/teleoperators/<your_teleop>/` 下实现对应的 `TeleoperatorConfig` 与 `Teleoperator` 子类。
5. 在 `lerobot/robots/utils.py` / `lerobot/teleoperators/utils.py` 中接入 `make_*_from_config` 分支，或确保通过插件机制自动发现。
6. 在 `lerobot/configs/` 中添加示例配置，方便 `lerobot-teleoperate` / `lerobot-record` / 训练脚本直接调用。
7. 在 `tests/robots/`、`tests/teleoperators/` 中添加单元测试，使用 `unittest.mock` 与 `tests/mocks/` 中的 mock 类隔离真实硬件。

后续章节将结合现有实现（如 `SO100Follower`、`SO101Follower`、`Reachy2Robot`、`HopeJr` 等）详细展开。

## 4. 在 robots 模块中创建新的机器人类

### 4.1 定义配置类：RobotConfig 子类

参考：`src/lerobot/robots/so101_follower/config_so101_follower.py`、`lekiwi/config_lekiwi.py`。

关键点：

- 继承 `RobotConfig` 并使用 `@RobotConfig.register_subclass("your_robot_name")` 注册；
- 字段中至少包括：
  - 通信端口：如 `port: str`；
  - 相机配置：`cameras: dict[str, CameraConfig]`（可选）；
  - 安全相关参数：例如 `max_relative_target`、`disable_torque_on_disconnect` 等。
- `RobotConfig.__post_init__` 会自动检查 `cameras` 中的相机宽高/FPS 不为 `None`。

示例（简化版）：

```python
from dataclasses import dataclass, field

from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("my_cool_robot")
@dataclass
class MyCoolRobotConfig(RobotConfig):
    port: str = "/dev/ttyUSB0"
    use_degrees: bool = False
    max_relative_target: float | None = 5.0
    disable_torque_on_disconnect: bool = True
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "front": OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=480),
        }
    )
```

### 4.2 实现 Robot 子类

参考：

- `src/lerobot/robots/so101_follower/so101_follower.py`
- `src/lerobot/robots/so100_follower/so100_follower.py`
- `tests/mocks/mock_robot.py`

关键要素：

- 类签名与元数据：
  - 继承 `Robot`；
  - 设置 `config_class = MyCoolRobotConfig`；
  - 设置 `name = "my_cool_robot"`。
- 在 `__init__` 中：
  - 调用 `super().__init__(config)`，以便基类处理校准文件路径 `calibration_dir`、`calibration` 加载；
  - 构造电机总线 `self.bus = FeetechMotorsBus(...)` 或 `DynamixelMotorsBus(...)`；
  - 通过 `make_cameras_from_configs(config.cameras)` 构建 `self.cameras`（如有）。
- 实现特征描述：
  - `observation_features`：描述 `get_observation()` 返回字典的 key 与 shape/type；
  - `action_features`：描述 `send_action()` 期望的动作字典结构。
- 实现生命周期方法：
  - `is_connected`：综合电机与相机连接状态；
  - `connect(calibrate: bool = True)`：连接总线与相机，必要时触发标定并调用 `configure()`；
  - `disconnect()`：断电机总线与相机，考虑是否关闭扭矩；
  - `is_calibrated` / `calibrate()` / `configure()`：复用 `MotorsBus` 的标定与配置能力。
- 实现核心 I/O：
  - `get_observation()`：
    - 调用 `self.bus.sync_read("Present_Position")` 获取关节位置；
    - 为每个电机生成 `"joint_name.pos"` 键；
    - 对每个相机调用 `cam.async_read()`，将图像放入同一字典；
  - `send_action(action)`：
    - 从 `action` 中抽取各关节目标位置；
    - 可通过 `ensure_safe_goal_position()` 对动作做安全裁剪；
    - 调用 `self.bus.sync_write("Goal_Position", goal_pos)` 下发至总线。

一个简化的模板（省略日志与异常处理细节）：

```python
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.robots.robot import Robot
from lerobot.robots.utils import ensure_safe_goal_position
from .config_my_cool_robot import MyCoolRobotConfig


class MyCoolRobot(Robot):
    config_class = MyCoolRobotConfig
    name = "my_cool_robot"

    def __init__(self, config: MyCoolRobotConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = (
            MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        )
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "joint_1": Motor(1, "sts3250", norm_mode_body),
                "joint_2": Motor(2, "sts3215", norm_mode_body),
                "joint_3": Motor(3, "sts3215", norm_mode_body),
                "joint_4": Motor(4, "sts3215", norm_mode_body),
                "joint_5": Motor(5, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

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

    def calibrate(self) -> None:
        # 参考 SO101Follower / integrate_hardware 文档实现完整标定逻辑
        ...

    def configure(self) -> None:
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor in self.bus.motors:
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        obs = self.bus.sync_read("Present_Position")
        obs = {f"{m}.pos": v for m, v in obs.items()}
        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        goal_pos = {k.removesuffix(".pos"): v for k, v in action.items() if k.endswith(".pos")}
        if self.config.max_relative_target is not None:
            present = self.bus.sync_read("Present_Position")
            goal_present = {m: (g, present[m]) for m, g in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present, self.config.max_relative_target)
        self.bus.sync_write("Goal_Position", goal_pos)
        return {f"{m}.pos": v for m, v in goal_pos.items()}

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()
```

## 5. 集成电机控制器

### 5.1 使用 FeetechMotorsBus / DynamixelMotorsBus

- **Feetech**：`lerobot.motors.feetech.FeetechMotorsBus`
  - 使用 Feetech 官方 `scservo_sdk`；
  - 通过 `MODEL_CONTROL_TABLE` 等控制表自动进行寄存器地址解析；
  - 提供 `configure_motors()`、`is_calibrated`、`read_calibration`、`write_calibration` 等帮助函数。
- **Dynamixel**：`lerobot.motors.dynamixel.DynamixelMotorsBus`
  - 使用 Robotis `dynamixel_sdk`；
  - 同样通过控制表驱动读写逻辑。

构造总线的通用形式为：

```python
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

motors = {
    "joint_1": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
    "joint_2": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
}
bus = FeetechMotorsBus(port="/dev/ttyUSB0", motors=motors, calibration=self.calibration)
```

在 `configure()`、`calibrate()` 中优先复用 `MotorsBus` 的能力，例如：

- 用 `set_half_turn_homings()` 让当前位置对应到行程中点；
- 用 `record_ranges_of_motion()` 交互式记录各关节的最大/最小角度；
- 基于这些信息构造 `MotorCalibration` 并调用 `write_calibration()` 与 `_save_calibration()`。

### 5.2 自定义电机总线

若使用自研驱动板或其他品牌电机，可以：

1. 在 `src/lerobot/motors/` 中新增子模块，继承 `MotorsBus`；
2. 实现其抽象方法（参考 `FeetechMotorsBus` / `DynamixelMotorsBus`）：
   - `_assert_protocol_is_compatible`、`_handshake`、`_find_single_motor`、`configure_motors`、
     `disable_torque/_disable_torque/enable_torque`、`read_calibration/write_calibration`、
     `_get_half_turn_homings`、`_encode_sign/_decode_sign`、`_split_into_byte_chunks`、`broadcast_ping`；
3. 在测试中参考 `tests/mocks/mock_motors_bus.py` 给出最小可用 mock。

## 6. 集成相机系统

### 6.1 相机配置与基类

- 抽象配置类：`CameraConfig`（`src/lerobot/cameras/configs.py`）
  - 字段：`fps/width/height`；
  - 使用 `ChoiceRegistry` 注册具体实现（如 `opencv`、`intelrealsense`）。
- 抽象相机类：`Camera`（`src/lerobot/cameras/camera.py`）
  - 必须实现：
    - `is_connected` 属性；
    - `connect()` / `disconnect()`；
    - `read()` / `async_read()`；
    - `@staticmethod find_cameras()`：用于自动发现可用相机。

### 6.2 在机器人中使用相机

- 在 `RobotConfig` 中添加 `cameras: dict[str, CameraConfig]` 字段；
- 在 `Robot.__init__` 中：
  - `self.cameras = make_cameras_from_configs(config.cameras)`；
- 在 `connect()` 中：
  - 遍历 `self.cameras.values()` 调用 `cam.connect()`；
- 在 `get_observation()` 中：
  - 对每个相机调用 `cam.async_read()`，将结果以键名（如 `"front"` 或 `"teleop_left"`）写入返回字典；
- 在 `disconnect()` 中关闭相机。

参见 `Reachy2Robot` 的实现（`src/lerobot/robots/reachy2/robot_reachy2.py`）以及其测试 `tests/robots/test_reachy2.py` 中对图像形状的断言。

## 7. 添加遥操作支持（Teleoperator）

如果你的机器人需要通过 Leader 机械臂或手柄进行遥操作，建议实现对应的 `Teleoperator`：

### 7.1 TeleoperatorConfig 与 Teleoperator 子类

参考：

- `src/lerobot/teleoperators/so101_leader/` 与 `so100_leader/`；
- mock 实现：`tests/mocks/mock_teleop.py`。

基本步骤：

1. 在 `src/lerobot/teleoperators/<your_teleop>/config_<your_teleop>.py` 中：
   - 继承 `TeleoperatorConfig`；
   - 使用 `@TeleoperatorConfig.register_subclass("your_teleop_name")` 注册；
   - 添加需要的字段（端口、使用 present / goal position、是否带移动底盘等）。
2. 在 `<your_teleop>.py` 中：
   - 继承 `Teleoperator`；
   - 设置 `config_class` 与 `name`；
   - 构造 `MotorsBus` 或其他输入设备 SDK；
   - 实现：
     - `action_features` / `feedback_features`（通常使用 `f"{motor}.pos"` 作为键）；
     - `is_connected/connect/disconnect`；
     - `is_calibrated/calibrate/configure`（如需）；
     - `get_action()`：从设备读取当前关节/速度命令；
     - `send_feedback()`：如需要力反馈或震动等效果。
3. 在 `teleoperators/utils.py` 中为新类型添加分支，或通过插件机制自动发现。

`SO101Leader` 的典型 `get_action` 实现：

- 通过 `bus.sync_read("Present_Position")` 读取当前位置；
- 将 `{"shoulder_pan.pos": val, ...}` 形式的字典作为动作返回；
- 不实现 `send_feedback`（抛出 `NotImplementedError`）。

## 8. 在 lerobot/configs/ 中编写配置

为了让训练脚本与 CLI 轻松使用你的机器人，需要提供相应的配置文件。常见方式：

1. 在 `src/lerobot/configs/` 下创建新的配置模块（例如 `robots/my_cool_robot_config.py`），其中引用 `MyCoolRobotConfig` 并与策略/环境配置组合。
2. 或在现有示例配置中增加一个 robot 选项，让 `robot.type="my_cool_robot"` 即可实例化你的机器人。

典型片段（伪代码）：

```python
from dataclasses import dataclass

from lerobot.robots.my_cool_robot.config_my_cool_robot import MyCoolRobotConfig


@dataclass
class MyCoolRobotExperimentConfig:
    robot: MyCoolRobotConfig = MyCoolRobotConfig(port="/dev/ttyUSB0")
    # 这里还可以加入 policy/env/dataset 等子配置
```

## 9. 测试指南

### 9.1 单元测试模式

LeRobot 的测试对硬件部分高度 mock 化，避免 CI 依赖真实设备。

推荐模式：

- 在 `tests/robots/test_<your_robot>.py` 中：
  - 使用 `unittest.mock.patch` 替换 `FeetechMotorsBus` / `DynamixelMotorsBus` / 第三方 SDK 为 MagicMock；
  - 仅保留你在 `Robot` 中实际调用到的属性与方法；
  - 针对以下场景断言行为：
    - `connect()` / `disconnect()` 正确修改 `is_connected` 状态并调用底层对象；
    - `get_observation()` 返回的键集合与 `observation_features` 一致，数值来源于 mock；
    - `send_action()` 按预期调用 `sync_write` / 下游库，动作裁剪逻辑正确。

可以参考：

- `tests/robots/test_so100_follower.py`
  - 自定义 `_make_bus_mock()`，仅实现 `is_connected`、`connect`、`disconnect`、`torque_disabled`、`sync_read/sync_write` 等；
  - 使用 `patch("lerobot.robots.so100_follower.so100_follower.FeetechMotorsBus", ...)` 注入测试总线；
  - 在 `test_get_observation` / `test_send_action` 中严格检查键空间与调用次数。
- `tests/robots/test_reachy2.py`
  - 使用嵌套 MagicMock 模拟 SDK 对象 `ReachySDK` 与 `Reachy2Camera`；
  - 检查开启/关闭扭矩、移动底盘速度指令是否被调用。

### 9.2 遥操作测试

参考 `tests/teleoperators/test_reachy2_teleoperator.py`：

- mock `ReachySDK`，设置 joints present/goal position 与移动底盘速度；
- 在 `get_action()` 测试中，根据 `config.use_present_position` 切换返回 present/goal 值；
- 校验配置非法组合时是否抛出 `ValueError`。

### 9.3 电机层测试（可选）

若你实现了新的 `MotorsBus` 子类：

- 在 `tests/motors/` 中添加对应测试文件，参考 `test_feetech.py`：
  - 使用 vendor SDK 的模拟对象（见 `tests/mocks/mock_feetech.py`）；
  - 覆盖 `_split_into_byte_chunks/_read/_write/sync_read/sync_write/is_calibrated/reset_calibration` 等核心逻辑。

### 9.4 运行测试

- 仅运行单个机器人测试：
  - `pytest tests/robots/test_my_cool_robot.py -q`
- 仅运行 teleop 测试：
  - `pytest tests/teleoperators/test_my_cool_teleop.py -q`
- 运行全量测试（较耗时）：
  - `pytest tests` 或参考 `CLAUDE.md` 中的 `make test-end-to-end`。

## 10. 常见问题与调试技巧

1. **`DeviceNotConnectedError` / `DeviceAlreadyConnectedError`**
   - 确保 `is_connected` 属性仅依赖底层总线/相机状态，而不是额外逻辑；
   - 在 `connect()` / `disconnect()` 中严格检查并抛出对应错误，有助于快速发现重复连接或错误调用顺序。
2. **串口连接失败（`ConnectionError: Could not connect on port`）**
   - 使用 `lerobot-find-port` 辅助确认端口；
   - 检查电源与线缆，确认没有其他进程占用串口；
   - 确认 `FeetechMotorsBus` / `DynamixelMotorsBus` 的 `protocol_version` 与硬件一致。
3. **标定状态不一致（`is_calibrated` 为 False）**
   - 当 `calibration` 字典中的 range/homing 与电机实际寄存器不一致时会被判定为未标定；
   - 按照 SO 系列 follower/leader 实现重新执行交互式标定流程，并确认 `.json` 校准文件成功写入。
4. **相机图像尺寸不匹配**
   - 确保 `CameraConfig` 中 `width/height/fps` 不为 `None`，否则 `RobotConfig.__post_init__` 会抛错；
   - 在测试中通过 `obs[cam_key].shape == (height, width, 3)` 进行断言。
5. **CLI 无法识别新的机器人类型**
   - 检查是否在配置类上正确使用了 `@RobotConfig.register_subclass("my_cool_robot")`；
   - 如需在 `lerobot-setup-motors` 等脚本中直接支持，需要修改脚本导入的 robots/teleoperators 列表与兼容设备常量。

## 11. 最佳实践与风格建议

1. **命名规范**
   - 关节观测/动作键推荐使用 `<joint_name>.pos` 形式，与 SO/Reachy2 系列保持一致；
   - Robot 名称（`name` 字段）应与 `RobotConfig.register_subclass` 中的类型字符串一致，保持小写+下划线。
2. **错误处理与日志**
   - 避免直接使用 `print`，使用 `logging.getLogger(__name__)` 输出调试信息（见 SO/Reachy2 实现）；
   - 对硬件相关错误（连接、通信超时）优先抛出 `ConnectionError` 或仓库中已有的异常类型。
3. **安全性**
   - 对新机器人建议提供 `max_relative_target` 配置，并在 `send_action` 中调用 `ensure_safe_goal_position()` 做动作裁剪；
   - 标定流程中务必提醒用户保持关节在安全范围内，且必要时降低抓手最大扭矩（参考 SO100/101 中对 `gripper` 的限制）。
4. **测试友好性**
   - 不要在模块 import 顶部直接访问硬件（串口扫描、SDK 实例化等），而应放在 `__init__`/`connect` 中，方便在测试里 `patch`；
   - 为复杂机器人（如带移动底盘、多个相机）提供灵活的 config 选项，并在测试中覆盖各种子集组合。
5. **与官方文档保持一致**
   - 本文是对 `docs/source/integrate_hardware.mdx` 的仓库级补充，建议在阅读本指南的同时通读该文档，以获得更完整的背景与示例代码。

通过上述步骤与规范，你应当能够在 LeRobot 仓库中稳健地集成一个新的真实机器人，并完整接入电机、相机与遥操作管线，同时具备可维护的测试覆盖率。
