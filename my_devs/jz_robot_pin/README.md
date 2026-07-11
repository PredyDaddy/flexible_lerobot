# jz_robot_pin 快速入口

本目录集中放置 `jz_robot_pin` 的 x86 侧启动、录制、回放脚本，以及机器人/边缘端包装脚本。

## x86 服务器

进入目录：

```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot/my_devs/jz_robot_pin
```

启动遥操作：

```bash
JZ_ROBOT_PIN_ARMED=1 \
I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT=1 \
bash start_teleop.sh
```

停止遥操作：

```bash
bash stop_teleop.sh
```

只启动 LeRobot 控制主循环：

```bash
JZ_ROBOT_PIN_ARMED=1 \
I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT=1 \
bash x86/start_pin_control.sh
```

只启动 VR/摇杆发布端：

```bash
bash x86/start_pin_joystick.sh
```

该入口默认持续发布当前保持目标（`publish_mode=always`），运行时长 24 小时。Meshcat 默认以
30 Hz 更新整机模型并关闭 VR debug 标记；IK 和 target action 发布使用 90 Hz。
启动后应看到 `target_frequency=30Hz ... model=whole-robot` 和 `visual meshes loaded: 24`；
运行状态中的 `meshcat_hz` 是墙钟实测刷新率，应稳定在约 30 Hz。

录制数据：

先在另一个终端启动 VR/摇杆发布端，并保持它在录制和 episode reset 期间持续发布：

```bash
bash x86/start_pin_joystick.sh
```

再启动录制。录制入口会等待第一帧 target action；录制中 target action stale 会终止本次录制，避免保存 hold-current 污染帧：

```bash
JZ_ROBOT_PIN_ARMED=1 \
I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT=1 \
DATASET_NAME=jz_robot_pin_vr_001 \
bash record.sh
```

回放数据，默认 dry-run：

```bash
DATASET_NAME=jz_robot_pin_vr_001 \
EPISODE=0 \
bash replay.sh
```

连续录制 3 条并在录制完成后自动检查 18 维 action/state、第一帧安全差值、连续动作步长和未来 1～6 帧跟随误差：

```bash
JZ_ROBOT_PIN_ARMED=1 \
I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT=1 \
bash data_check/record_and_check_3.sh
```

详细说明见 `data_check/README.md`。

真实 armed 回放：

```bash
JZ_ROBOT_PIN_ARMED=1 \
I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT=1 \
EXECUTION=armed \
DATASET_NAME=jz_robot_pin_vr_001 \
EPISODE=0 \
bash replay.sh
```

## 机器人/边缘端

进入机器人/边缘端仓库：

```bash
cd /home/data/test/workspace/flexible_lerobot
```

只启动 state 通信，适合只录状态/相机：

```bash
bash my_devs/jz_robot_pin/edge/start_pin_state.sh
```

启动 state + command executor，适合 x86 遥操作/录制：

```bash
JZ_UDP_EXECUTOR_ARMED=1 \
bash my_devs/jz_robot_pin/edge/start_pin_replay.sh
```

停止：

```bash
bash my_devs/jz_robot_pin/edge/stop_pin_replay.sh
```

查看状态：

```bash
bash my_devs/jz_robot_pin/edge/status_pin_replay.sh
```

## 默认环境

- x86 LeRobot 主进程默认使用 `lerobot_flex`。
- x86 VR/摇杆发布端默认使用 `light_tp`。
- 机器人/边缘端包装脚本默认使用 `lerobot`，和当前 Orin 端日志保持一致。

更完整的实现方案见 `IMPLEMENTATION_PLAN.md`。
