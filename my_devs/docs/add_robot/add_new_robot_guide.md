# LeRobot 新机器人接入指南

本指南不是抽象架构说明，而是按当前仓库真实代码路径整理的一份“可落地接入”文档。

目标是回答三个问题：

1. 机器人类本身应该怎么写，才能符合 `Robot` / `Teleoperator` 接口？
2. 写完类之后，还要补哪些接入点，`lerobot-teleoperate`、`lerobot-record`、`lerobot-replay`、`lerobot-calibrate` 才能真正识别？
3. 哪些脚本是主流程支持，哪些脚本还有额外白名单或额外限制？

本文基于当前仓库代码核对，重点参考：

- `src/lerobot/robots/robot.py`
- `src/lerobot/robots/utils.py`
- `src/lerobot/teleoperators/teleoperator.py`
- `src/lerobot/teleoperators/utils.py`
- `src/lerobot/utils/import_utils.py`
- `src/lerobot/configs/parser.py`
- `src/lerobot/scripts/lerobot_calibrate.py`
- `src/lerobot/scripts/lerobot_teleoperate.py`
- `src/lerobot/scripts/lerobot_record.py`
- `src/lerobot/scripts/lerobot_replay.py`
- `src/lerobot/scripts/lerobot_setup_motors.py`
- `src/lerobot/scripts/lerobot_find_joint_limits.py`
- `src/lerobot/robots/so_follower/`
- `src/lerobot/robots/reachy2/`
- `tests/robots/test_so100_follower.py`
- `tests/test_control_robot.py`

## 1. 先分清两条接入路径

在当前仓库里，新增机器人有两种路径：

### 路径 A：直接集成到当前仓库

适合你准备修改 `src/lerobot/` 内代码，并希望当前仓库自带脚本直接支持。

典型结果：

- 机器人代码放在 `src/lerobot/robots/<your_robot>/`
- 可以通过 `lerobot-teleoperate`、`lerobot-record`、`lerobot-replay`、`lerobot-calibrate` 使用
- 如有需要，再额外接入 `lerobot-setup-motors`、`lerobot-find-joint-limits`

### 路径 B：做成第三方插件包

适合你不想改动主仓库，而是想单独维护一个 `pip install -e .` 的扩展包。

典型结果：

- 包名以 `lerobot_robot_`、`lerobot_teleoperator_`、`lerobot_camera_` 开头
- 安装到环境后，主流程脚本会通过插件发现自动导入
- 但不是所有辅助脚本都支持插件发现，后文会单独说明

如果你只是想“先在当前仓库里把机器人跑通”，建议先走路径 A。

## 2. 当前仓库的真实发现机制

这部分是最容易写错的地方。

当前仓库里，一个新机器人要能被 CLI 正常使用，通常同时依赖三层机制：

1. `RobotConfig` 子类必须通过 `@RobotConfig.register_subclass("your_robot")` 注册。
2. 对应模块必须先被导入，这样 `draccus` 在解析 `--robot.type=your_robot` 时才知道这个类型存在。
3. `make_robot_from_config()` 必须能实例化这个对象。

第三层又分两种情况：

- 情况 A：你显式在 `src/lerobot/robots/utils.py` 里加一个 `elif config.type == ...`
- 情况 B：你遵守命名约定，让 `make_device_from_device_class()` 自动找到类

### 2.1 自动实例化的命名约定

如果你不想改 `src/lerobot/robots/utils.py`，推荐严格遵守下面这套约定：

- 配置类名以 `Config` 结尾，例如 `MyRobotConfig`
- 设备类名和配置类同名去掉 `Config`，例如 `MyRobot`
- 目录结构类似：

```text
src/lerobot/robots/my_robot/
├── __init__.py
├── config_my_robot.py
└── my_robot.py
```

- `__init__.py` 要导出 `MyRobotConfig` 和 `MyRobot`
- `config_my_robot.py` 中定义 `MyRobotConfig`
- `my_robot.py` 中定义 `MyRobot`

为什么这套命名能工作：

- `make_device_from_device_class()` 会先根据配置类模块路径推断候选模块
- 如果配置文件名是 `config_xxx.py`，它还会额外尝试导入 `xxx.py`
- 如果包级 `__init__.py` 已经导出设备类，它也能从父包直接拿到类

所以，推荐组合是：

- `config_my_robot.py`
- `my_robot.py`
- `__init__.py` 导出两者

这样最稳。

### 2.2 为什么“只注册 config”还不够

很多文档会把“注册 config”写成唯一步骤，但在当前仓库里这不够。

原因是：

- `draccus` 只有在配置类所属模块已经被导入时，才能看到这个注册项
- `lerobot-teleoperate`、`lerobot-record`、`lerobot-replay`、`lerobot-calibrate` 会在脚本顶部显式导入一些已知机器人模块，以此触发注册
- 第三方插件则依赖 `register_third_party_plugins()` 自动导入安装包

因此：

- 如果你是 in-tree 新机器人，通常要把它加入目标脚本的 import 列表
- 如果你是第三方插件，需要保证包已安装到当前环境，并遵守插件命名约定

还有一个容易忽略的点：

- CLI 里真正使用的 `type` 是 `register_subclass("...")` 里注册的字符串
- 它不是目录名，也不是类名自动推导值

例如现有代码里：

- 目录是 `so_follower/`
- 但 `type` 是 `so100_follower` / `so101_follower`

### 2.3 什么时候必须改 `make_robot_from_config()`

以下情况建议直接改 `src/lerobot/robots/utils.py`：

- 你的类命名不遵守 `SomethingConfig` / `Something`
- 你的模块布局不是 `config_xxx.py` + `xxx.py`
- 你希望像现有内建机器人一样走显式分支
- 你需要特殊构造逻辑，不想依赖通用反射实例化

否则，通常不需要改 factory。

## 3. 主流程支持与辅助脚本支持不是一回事

这是接入时第二个常见误区。

### 3.1 主流程脚本

以下脚本是新增机器人最优先要打通的主流程：

- `lerobot-calibrate`
- `lerobot-teleoperate`
- `lerobot-record`
- `lerobot-replay`

这些脚本都能配合当前的机器人抽象使用。

其中：

- `teleoperate` / `record` / `replay` 会调用 `register_third_party_plugins()`
- `teleoperate` / `record` / `replay` 使用的是 `@parser.wrap()`
- `calibrate` 虽然用的是 `@draccus.wrap()`，但它的 `main()` 里也会先执行 `register_third_party_plugins()`

### 3.2 辅助脚本

以下脚本有额外限制，不应该默认认为“新机器人自然支持”：

- `lerobot-setup-motors`
- `lerobot-find-joint-limits`

如果你的目标不仅是主流程，还包括更广的运行面，还要额外关注：

- `lerobot-eval`
- `python -m lerobot.rl.gym_manipulator`
- RL actor / learner 相关 CLI
- async inference 的 robot client

原因：

- `lerobot-setup-motors` 有 `COMPATIBLE_DEVICES` 白名单
- `lerobot-find-joint-limits` 逻辑本身依赖 teleop + kinematics + `robot.bus.motors`
- 它们没有走主流程那套完整插件发现路径

结论：

- 对大多数新机器人，先打通主流程即可
- 如果你还想支持这些辅助脚本，要额外补白名单、导入和设备特定方法

## 4. 路径 A：直接集成到当前仓库的推荐步骤

本节假设你准备修改 `src/lerobot/`。

### 4.1 目录布局

推荐结构：

```text
src/lerobot/robots/my_robot/
├── __init__.py
├── config_my_robot.py
└── my_robot.py
```

`__init__.py` 推荐写法：

```python
from .config_my_robot import MyRobotConfig
from .my_robot import MyRobot

__all__ = ["MyRobot", "MyRobotConfig"]
```

### 4.2 配置类写法

推荐先从最小配置开始，不要默认带真实相机，避免单测变成硬件依赖。

```python
from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("my_robot")
@dataclass
class MyRobotConfig(RobotConfig):
    port: str

    disable_torque_on_disconnect: bool = True
    max_relative_target: float | dict[str, float] | None = None
    use_degrees: bool = False

    # 推荐默认留空，真实运行时再从 CLI 传入
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
```

为什么推荐 `cameras` 默认留空：

- 单元测试只 patch 电机总线时，不会误连真实摄像头
- 真实运行时可以从 CLI 覆盖：

```bash
--robot.cameras='{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}'
```

### 4.3 设备类最小骨架

下面这个模板遵守了当前仓库的真实接口与命名约定。

```python
import logging
import time
from functools import cached_property

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots import Robot
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config_my_robot import MyRobotConfig

logger = logging.getLogger(__name__)


class MyRobot(Robot):
    config_class = MyRobotConfig
    name = "my_robot"

    def __init__(self, config: MyRobotConfig):
        super().__init__(config)
        self.config = config

        norm_mode_body = (
            MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        )
        self.bus = FeetechMotorsBus(
            port=self.config.port,
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
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple[int, int, int]]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()

        if not self.is_calibrated and calibrate:
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info("%s connected.", self)

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            user_input = input(
                f"Press ENTER to use calibration for id={self.id}, or type 'c' to recalibrate: "
            )
            if user_input.strip().lower() != "c":
                self.bus.write_calibration(self.calibration)
                return

        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move {self} to the middle of its range and press ENTER...")
        homing_offsets = self.bus.set_half_turn_homings()

        print("Move all joints through their ranges of motion. Press ENTER to stop...")
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
        print("Calibration saved to", self.calibration_fpath)

    def configure(self) -> None:
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor in self.bus.motors:
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                self.bus.write("P_Coefficient", motor, 16)
                self.bus.write("I_Coefficient", motor, 0)
                self.bus.write("D_Coefficient", motor, 32)

                if motor == "gripper":
                    self.bus.write("Max_Torque_Limit", motor, 500)
                    self.bus.write("Protection_Current", motor, 250)
                    self.bus.write("Overload_Torque", motor, 25)

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        start = time.perf_counter()
        obs = self.bus.sync_read("Present_Position")
        obs = {f"{motor}.pos": val for motor, val in obs.items()}
        logger.debug("%s read state in %.1fms", self, (time.perf_counter() - start) * 1e3)

        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()

        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        self.bus.sync_write("Goal_Position", goal_pos)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def disconnect(self) -> None:
        if not self.is_connected:
            return
        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()
        logger.info("%s disconnected.", self)
```

### 4.4 这个模板为什么比“泛用示例”更可靠

上面模板专门贴合了当前仓库的真实做法：

- `connect()` / `get_observation()` / `send_action()` 使用当前仓库常见的连接状态装饰器模式
- `configure()` 中给 `bus.write()` 传的是电机名字符串，不是 `Motor` 对象
- `get_observation()` / `send_action()` 用 `RobotObservation` / `RobotAction`
- `cameras` 默认可为空，测试更容易隔离
- `observation_features` 和 `action_features` 不依赖运行时连接状态

`disconnect()` 这里故意做成幂等 no-op：

- 当前仓库里很多内建设备的 `disconnect()` 仍然使用 `@check_if_not_connected`
- 但主流程脚本在 `finally` 中会无条件调用 `disconnect()`
- 如果 `connect()` 在把连接状态置真之前就抛错，幂等 `disconnect()` 往往更不容易把原始异常覆盖掉

## 5. 哪些字段和行为最容易踩坑

### 5.1 `id` 不填也能跑，但会共享到 `None.json`

`Robot.__init__()` 里校准文件路径是：

```text
<calibration_dir>/<id>.json
```

如果你不传 `id`，最终文件名会是 `None.json`。

这不会立刻报错，但多个设备很容易共用同一份校准文件。

建议：

- 真机运行时总是显式传 `--robot.id=...`
- leader / follower 分别使用不同 id

### 5.2 相机配置缺 `width/height/fps` 会在 `RobotConfig.__post_init__()` 报错

如果配置里有 `cameras`，每个相机配置都必须给出：

- `width`
- `height`
- `fps`

否则 `RobotConfig.__post_init__()` 会直接报错。

### 5.3 `observation_features` / `action_features` 必须在未连接时也能调用

不要依赖：

- 运行时硬件返回
- `connect()` 之后才会存在的属性

推荐像现有实现一样，在 `__init__()` 里就创建总线对象和相机对象，然后从 `self.bus.motors` / `self.config.cameras` 派生 features。

### 5.4 `calibrate()` 不能只在 `connect(calibrate=True)` 里成立

`lerobot-calibrate` 的调用顺序是：

1. `device.connect(calibrate=False)`
2. `device.calibrate()`
3. `device.disconnect()`

所以：

- `calibrate()` 要能独立工作
- 不要把关键初始化只写在 `connect(calibrate=True)` 的分支里

### 5.5 并不是所有机器人都必须实现校准

如果你的机器人根本不需要校准，可以这样写：

```python
@property
def is_calibrated(self) -> bool:
    return True

def calibrate(self) -> None:
    pass
```

这对纯 SDK 机器人很常见，例如网络化或厂商 SDK 已自带绝对位姿系统。

## 6. 写完 Robot 类后，in-tree 还要补哪些接入点

这是让脚本真正能识别你的关键步骤。

### 6.1 让目标脚本先 import 你的模块

如果你要让主流程脚本支持 `--robot.type=my_robot`，通常要把 `my_robot` 加到这些脚本顶部的机器人 import 列表里：

- `src/lerobot/scripts/lerobot_calibrate.py`
- `src/lerobot/scripts/lerobot_teleoperate.py`
- `src/lerobot/scripts/lerobot_record.py`
- `src/lerobot/scripts/lerobot_replay.py`

例如：

```python
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
    my_robot,
    so_follower,
)
```

为什么要这么做：

- 导入模块时，`@RobotConfig.register_subclass("my_robot")` 才会执行
- `draccus` 才能识别 `--robot.type=my_robot`

同一个原则也适用于你在 CLI 里用到的其他注册型配置：

- 新的 `TeleoperatorConfig` 子类
- 新的 `CameraConfig` 子类

如果目标脚本没有 import 对应模块，也没有插件发现把它们拉进来，CLI 解析一样会失败。

### 6.2 通常不需要改 `src/lerobot/robots/__init__.py`

只要你的模块路径是 `src/lerobot/robots/my_robot/`，脚本里可以直接：

```python
from lerobot.robots import my_robot
```

通常不需要再去改 `src/lerobot/robots/__init__.py`。

### 6.3 通常也不需要改 `src/lerobot/robots/utils.py`

如果你遵守了：

- `MyRobotConfig`
- `MyRobot`
- `config_my_robot.py`
- `my_robot.py`
- `__init__.py` 导出

那么 `make_robot_from_config()` 的 fallback 通常就能实例化成功。

只有在以下情况下才建议显式改 factory：

- 命名不标准
- 模块布局不标准
- 需要特殊构造分支

### 6.4 如果你想支持辅助脚本，还要额外处理

#### `lerobot-setup-motors`

这个脚本除了 import 你的模块，还需要：

- 在 `COMPATIBLE_DEVICES` 里加入你的 `config.type`
- 你的设备实现 `setup_motors()`

注意：

- `setup_motors()` 不是 `Robot` 抽象基类的一部分
- 这是设备专项能力，不是通用机器人接口

#### `lerobot-find-joint-limits`

这个脚本还依赖：

- 一个兼容的 teleoperator
- `robot.bus.motors`
- URDF 与运动学模型

所以它并不适合所有机器人。

如果你的机器人不是基于这种关节级总线机械臂结构，不要强行接这个脚本。

还要注意，这个脚本的约束比上面三条更严：

- `teleop.get_action()` 的原始输出必须能直接喂给 `robot.send_action()`
- `robot.get_observation()` 必须为 `robot.bus.motors` 中的每个关节都提供 `"{motor}.pos"` 键
- 这里没有 processor / rename / key remap 过渡层

所以它更像是面向特定关节机械臂形态的专项脚本，而不是通用机器人接入面。

## 7. 推荐测试策略

### 7.1 单元测试先做纯 bus mock

推荐从 `tests/robots/test_so100_follower.py` 这种模式开始。

最重要的原则：

- patch 电机总线构造
- 让 `cameras` 默认为空
- 不要让测试依赖本机真实串口或摄像头

示例：

```python
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.my_robot import MyRobot, MyRobotConfig


def _make_bus_mock() -> MagicMock:
    bus = MagicMock(name="MyRobotBusMock")
    bus.is_connected = False

    def _connect():
        bus.is_connected = True

    def _disconnect(_disable=True):
        bus.is_connected = False

    bus.connect.side_effect = _connect
    bus.disconnect.side_effect = _disconnect

    @contextmanager
    def _dummy_cm():
        yield

    bus.torque_disabled.side_effect = _dummy_cm
    return bus


@pytest.fixture
def robot():
    bus_mock = _make_bus_mock()

    def _bus_side_effect(*_args, **kwargs):
        bus_mock.motors = kwargs["motors"]
        motors_order = list(bus_mock.motors)
        bus_mock.sync_read.return_value = {motor: idx for idx, motor in enumerate(motors_order, 1)}
        bus_mock.sync_write.return_value = None
        bus_mock.write.return_value = None
        bus_mock.disable_torque.return_value = None
        bus_mock.enable_torque.return_value = None
        bus_mock.is_calibrated = True
        return bus_mock

    with (
        patch("lerobot.robots.my_robot.my_robot.FeetechMotorsBus", side_effect=_bus_side_effect),
        patch.object(MyRobot, "configure", lambda self: None),
    ):
        cfg = MyRobotConfig(port="/dev/null")
        yield MyRobot(cfg), bus_mock
```

最少覆盖：

- `connect()` / `disconnect()`
- `get_observation()` 键名与 `observation_features` 一致
- `send_action()` 会把 `.pos` 键转换为总线目标键
- `max_relative_target` 会触发裁剪

### 7.2 命令行烟测要分两层

#### 层 1：类级别单测

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n lerobot_flex pytest -q tests/robots/test_my_robot.py
```

这里加 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 是因为某些本机环境可能带 ROS 或别的 pytest 插件，导致和仓库无关的导入失败。

#### 层 2：脚本接入验证

如果你改了主流程脚本的 import 列表，建议再做一次真实 CLI 解析烟测。

这里要区分两类测试：

1. 直接构造 `cfg`，再调用 `calibrate(cfg)`、`teleoperate(cfg)`、`record(cfg)`、`replay(cfg)`：
   这类测试可以覆盖主流程函数本身，但不能验证 `--robot.type=...` / `--teleop.type=...` 的 CLI 注册与发现链路。
2. 真实走命令行参数解析：
   这类测试才会暴露“目标脚本没 import 你的模块，导致类型根本没注册”这类错误。

`tests/test_control_robot.py` 属于第 1 类，所以它可以作为流程 smoke test 参考，但不能替代 CLI 解析验证。

如果你想验证发现链路，至少要实际执行一次目标脚本命令，或者写一个显式覆盖 CLI 参数解析的测试。

### 7.3 真机验证建议顺序

推荐按下面顺序验证：

1. `lerobot-calibrate`
2. `lerobot-teleoperate`
3. `lerobot-record`
4. `lerobot-replay`

理由：

- `calibrate` 最先暴露串口、总线、校准问题
- `teleoperate` 最先暴露 action key 对齐问题
- `record` 最先暴露 observation features / camera 问题
- `replay` 最先暴露动作重放和数据集兼容问题

## 8. 路径 B：做成第三方插件包

如果你不打算修改主仓库，而是想单独维护包，建议按下面做。

### 8.1 包名约定

安装包名必须带前缀：

- 机器人：`lerobot_robot_<name>`
- teleoperator：`lerobot_teleoperator_<name>`
- 相机：`lerobot_camera_<name>`

主仓库会扫描环境里这些前缀的已安装包，并自动 import。

但这里还有一个实现细节要说清楚：

- 代码遍历的是安装分发的 `Name`
- 然后直接执行 `importlib.import_module(dist_name)`

因此，除了前缀正确之外，分发名本身还必须是一个可 import 的模块名。

最稳的实践是：

- 分发名和顶层 Python 包名保持一致
- 两者都使用 `lerobot_robot_my_robot` 这种下划线形式

### 8.2 推荐目录

例如机器人插件：

```text
lerobot_robot_my_robot/
├── pyproject.toml
└── lerobot_robot_my_robot/
    ├── __init__.py
    ├── config_my_robot.py
    └── my_robot.py
```

### 8.3 推荐约定

- 配置类：`MyRobotConfig`
- 设备类：`MyRobot`
- `config_my_robot.py` 定义配置类
- `my_robot.py` 定义设备类
- `__init__.py` 导出二者

### 8.4 插件包什么时候会被发现

主流程脚本中，以下场景通常可自动发现已安装插件：

- `lerobot-calibrate`
- `lerobot-teleoperate`
- `lerobot-record`
- `lerobot-replay`
- `lerobot-eval`

### 8.5 插件包的限制

不要默认认为“插件包安装好了，所有脚本都支持”。

以下脚本当前不是这条主路径：

- `lerobot-setup-motors`
- `lerobot-find-joint-limits`

如果你要让插件也支持这些脚本，通常需要额外改主仓库脚本本身。

## 9. 可选：新增 Teleoperator 的正确姿势

如果你的机器人需要新的 leader 设备或手柄，teleoperator 也要按真实抽象来实现。

### 9.1 当前 `Teleoperator` 抽象的必需项

不是只实现 `get_action()` 就够了。当前基类要求：

- `action_features`
- `feedback_features`
- `is_connected`
- `connect()`
- `is_calibrated`
- `calibrate()`
- `configure()`
- `get_action()`
- `send_feedback()`
- `disconnect()`

### 9.2 最小 teleop 骨架

```python
from dataclasses import dataclass
from functools import cached_property

from lerobot.processor import RobotAction
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected


@TeleoperatorConfig.register_subclass("my_teleop")
@dataclass
class MyTeleopConfig(TeleoperatorConfig):
    port: str


class MyTeleop(Teleoperator):
    config_class = MyTeleopConfig
    name = "my_teleop"

    def __init__(self, config: MyTeleopConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            "joint_1.pos": float,
            "joint_2.pos": float,
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self._is_connected = True
        if calibrate:
            self.calibrate()
        self.configure()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        return {
            "joint_1.pos": 0.0,
            "joint_2.pos": 0.0,
        }

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, float]) -> None:
        pass

    def disconnect(self) -> None:
        if not self.is_connected:
            return
        self._is_connected = False
```

### 9.3 如果你要让 teleoperate / record 真正工作

还必须保证：

- teleop 的 action keys 和机器人 action keys 对齐
- `teleop.get_action()` 返回的结构能直接喂给 robot pipeline

关于多 teleop，要非常保守地理解当前支持边界：

- `TeleoperateConfig.teleop` 和 `RecordConfig.teleop` 对外都只接受单个 `TeleoperatorConfig`
- 当前公开 CLI 并没有通用的“多个 teleop 同时接入”配置入口
- `record_loop()` 虽然内部接受 `list[Teleoperator]`，但马上又把它限制成一个硬编码特例：
  一个 `KeyboardTeleop` + 一个 arm teleop + `robot.name == "lekiwi_client"`

所以，对新 teleoperator 来说，不要把“多 teleop 组合”当成现成通用能力，而应把它视为当前代码里的 LeKiwi 特例。

如果你是 in-tree 新 teleoperator，而不是插件包，还要记得把模块加入目标脚本的 import 列表。

通常涉及：

- `src/lerobot/scripts/lerobot_calibrate.py`
- `src/lerobot/scripts/lerobot_teleoperate.py`
- `src/lerobot/scripts/lerobot_record.py`
- 如果还想支持专项脚本，再看 `src/lerobot/scripts/lerobot_setup_motors.py`
- 如果还想支持关节边界查找，再看 `src/lerobot/scripts/lerobot_find_joint_limits.py`

## 10. 自定义 MotorsBus 时该实现什么

如果你不是复用 Feetech / Dynamixel，而是自己写 `MotorsBus` 子类，注意不要照着过时文档去实现错接口。

### 10.1 真正的抽象方法

当前 `MotorsBus` 真正要求你实现的核心抽象包括：

- `_assert_protocol_is_compatible`
- `_handshake`
- `_find_single_motor`
- `configure_motors`
- `disable_torque`
- `_disable_torque`
- `enable_torque`
- `is_calibrated`
- `read_calibration`
- `write_calibration`
- `_get_half_turn_homings`
- `_encode_sign`
- `_decode_sign`
- `_split_into_byte_chunks`
- `broadcast_ping`

### 10.2 不要误以为 `read()` / `write()` / `sync_read()` / `sync_write()` 是抽象方法

在当前仓库里，这些方法已经在 `MotorsBus` 基类中实现好了。

你真正需要补的是更底层的协议相关能力和校准相关能力。

如果你误把 `read()` / `write()` 当成抽象方法来实现，最后仍然会因为漏掉真正的抽象方法而无法实例化。

## 11. 一份实际可执行的接入清单

### 11.1 只求主流程可用

如果你的目标只是让机器人走通主流程，建议按这个最小清单执行：

1. 新建 `src/lerobot/robots/my_robot/`
2. 写 `config_my_robot.py`
3. 写 `my_robot.py`
4. 在 `__init__.py` 导出类
5. 保证命名遵守 `MyRobotConfig` / `MyRobot`
6. 在以下脚本 import 列表加入 `my_robot`
7. 新增 `tests/robots/test_my_robot.py`
8. 先跑类级单测
9. 再跑真机 `calibrate -> teleoperate -> record -> replay`

要改的脚本通常是：

- `src/lerobot/scripts/lerobot_calibrate.py`
- `src/lerobot/scripts/lerobot_teleoperate.py`
- `src/lerobot/scripts/lerobot_record.py`
- `src/lerobot/scripts/lerobot_replay.py`

### 11.2 希望支持辅助脚本

在主流程之外，再按需补：

- `src/lerobot/scripts/lerobot_setup_motors.py`
- `src/lerobot/scripts/lerobot_find_joint_limits.py`

以及：

- `setup_motors()` 方法
- 白名单
- 运动学 / URDF / teleop 兼容逻辑

如果你的目标再往外扩一层，还要按需检查：

- `src/lerobot/scripts/lerobot_eval.py`
- `src/lerobot/rl/gym_manipulator.py`
- `src/lerobot/rl/actor.py`
- `src/lerobot/rl/learner.py`
- `src/lerobot/async_inference/robot_client.py`
- `src/lerobot/async_inference/constants.py`

这些入口不是“新增一个 Robot 类就自然支持”的。

### 11.3 如果走插件包

最小清单是：

1. 新建独立包 `lerobot_robot_my_robot`
2. 写 `MyRobotConfig` / `MyRobot`
3. 遵守 `config_my_robot.py` / `my_robot.py`
4. `__init__.py` 导出类
5. `pip install -e .`
6. 用主流程脚本验证

## 12. 常见错误对照表

| 症状 | 常见原因 | 应对方式 |
| --- | --- | --- |
| `--robot.type=my_robot` 解析失败 | 模块没被导入，注册没发生 | in-tree 时把模块加入脚本 import；插件时确认包已安装 |
| CLI 里的 `type` 和目录名对不上 | `type` 取的是 `register_subclass("...")`，不是目录名 | 以注册字符串为准，不要把目录名当成 CLI 类型 |
| `--robot.cameras=...` 里的相机类型解析失败 | 目标脚本没 import 该 `CameraConfig` 子类，也没有插件发现 | 给目标脚本补 import，或把相机做成插件包并安装 |
| factory 找不到类 | 命名或模块布局不符合约定 | 改成 `MyRobotConfig` / `MyRobot` + `config_my_robot.py` / `my_robot.py`，或显式修改 `make_robot_from_config()` |
| 单测连接了真实摄像头 | config 默认创建了 OpenCV 相机 | 默认把 `cameras` 设为空，测试只 mock bus |
| `configure()` 写寄存器时报错 | 把 `self.bus.motors["gripper"]` 当成 `write()` 的 motor 参数 | 传字符串电机名，如 `"gripper"` |
| 校准文件混用 | 没显式设置 `id` | 真机运行始终传 `--robot.id=` |
| `lerobot-calibrate` 不工作 | `calibrate()` 依赖了 `connect(calibrate=True)` 的隐含逻辑 | 保证 `connect(calibrate=False)` 后 `calibrate()` 也能独立执行 |
| 连接失败后 `finally` 又报 `disconnect()` 错误 | `disconnect()` 不是幂等的，清理时把原始连接异常覆盖了 | 优先把 `disconnect()` 设计成已断开时直接返回 |
| 插件在 `setup_motors` 不可用 | 该脚本不走主流程插件发现 | 修改主仓库脚本或暂不支持 |
| 想用 `--robot.config_path=...` 或 `--teleop.config_path=...` | 当前解析器没有通用支持这种写法 | 以当前脚本真实支持的参数为准，不要假设所有嵌套字段都有 `.path` 入口 |

## 13. 推荐验证命令

以下命令按仓库规范使用 `lerobot_flex` 环境：

### 13.1 文档相关或代码变更后的最小检查

```bash
conda run -n lerobot_flex python -m compileall src/lerobot/robots/my_robot
```

### 13.2 机器人单测

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n lerobot_flex pytest -q tests/robots/test_my_robot.py
```

### 13.3 主流程 smoke test

先人工确认：

- 串口或网络连通
- 校准文件 id 正确
- 相机配置包含 `width/height/fps`

然后按顺序执行：

```bash
conda run -n lerobot_flex lerobot-calibrate --robot.type=my_robot --robot.port=/dev/ttyUSB0 --robot.id=my_robot_01
```

```bash
conda run -n lerobot_flex lerobot-teleoperate \
  --robot.type=my_robot \
  --robot.port=/dev/ttyUSB0 \
  --robot.id=my_robot_01 \
  --teleop.type=my_teleop \
  --teleop.port=/dev/ttyUSB1
```

```bash
conda run -n lerobot_flex lerobot-record \
  --robot.type=my_robot \
  --robot.port=/dev/ttyUSB0 \
  --robot.id=my_robot_01 \
  --teleop.type=my_teleop \
  --teleop.port=/dev/ttyUSB1 \
  --dataset.repo_id=<user>/<dataset_name> \
  --dataset.single_task="Demo task" \
  --dataset.num_episodes=1 \
  --dataset.push_to_hub=false
```

```bash
conda run -n lerobot_flex lerobot-replay \
  --robot.type=my_robot \
  --robot.port=/dev/ttyUSB0 \
  --robot.id=my_robot_01 \
  --dataset.repo_id=<user>/<dataset_name> \
  --dataset.episode=0
```

## 14. 最后的建议

如果你只打算把一个新机器人接入当前仓库，最稳的路线不是“先做大而全抽象”，而是：

1. 仿照 `so_follower` 或 `reachy2` 先做一个最小、可连接、可读状态、可发动作的实现。
2. 先让 `calibrate` / `teleoperate` / `record` / `replay` 这四条主流程打通。
3. 只有在确实需要时，再去补 `setup_motors`、`find_joint_limits`、插件化发布、多 teleop、复杂相机后端等增强项。

把“接口实现正确”和“脚本入口真正可发现”同时完成，才算接入完成。
