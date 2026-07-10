# jz_robot_pin 接入 LeRobot 实现方案

本文记录 `jz_robot_pin` 的设计方案。目标是把当前“机器人/边缘端 UDP 通信服务 + x86 侧 VR/摇杆遥操作 + LeRobot 录制/控制”整理成一套新的、独立于 `jz_robot_udp` 的机器人接入路径。

当前只做方案设计，不启动机器人、不启动遥操作、不改动现有 `jz_robot_udp` 行为。

## 背景和结论

当前仓库里已经有两套 JZ 机器人路径：

- `src/lerobot/robots/jz_robot`
  - x86 直接作为 ROS2 节点连接机器人 ROS topic。
  - 适合 x86 能直接访问 ROS2 graph 的场景。
- `src/lerobot/robots/jz_robot_udp`
  - x86 通过 UDP 接收 Orin/边缘端转发的机器人状态，并通过 UDP 发动作命令。
  - 目前已经有不少代码基于它工作，不能为了新遥操作链路直接重构它。

这次新链路的关键点是：

- 机器人和 x86 服务器部署同一套代码。
- 机器人/边缘端先启动 ROS state UDP bridge 和 Phase 3 command executor。
- x86 侧再启动当前 VR/摇杆遥操作程序。
- 当前遥操作不是机器人内部原来的那版遥操作，而是 `my_devs/my_var_tp/live_vr_replay_bridge` 这一路：
  - `visual_publisher` 接收 VR/摇杆 UDP，跑 IK，发 `target_action`。
  - `robot_receiver` 接收 `target_action`，读机器人状态，再通过 UDP command 发给边缘端 executor。

为了不影响现有 `jz_robot_udp` 及其依赖代码，建议新增一套 robot backend：

```text
src/lerobot/robots/jz_robot_pin
```

LeRobot 类型名：

```yaml
type: jz_robot_pin
```

`jz_robot_pin` 的目标不是立刻替换旧链路，而是把“当前这版 x86 控制 + VR/摇杆遥操作 + 录制”做成一个边界清晰、可测试、可回滚的新入口。

## 当前实际启动命令梳理

### 机器人/边缘端：启动通信和控制服务

在机器人/边缘端仓库根目录执行：

```bash
cd /home/data/test/workspace/flexible_lerobot

bash udp_test/all/start_replay.sh
```

这个命令做两件事：

- 启动 ROS state UDP bridge：
  - 机器人/边缘端 `192.168.1.81`
  - x86 服务器 `192.168.1.106`
  - state UDP port `39010`
  - 默认 `STATE_HZ=20`
- 启动 Phase 3 command executor：
  - bind `192.168.1.81:39020`
  - allowed sender `192.168.1.106`
  - 默认 `EXECUTION=armed`
  - 默认配置 `udp_test/test_scripts/arm_side/orin_phase3_executor_config_armed_hold.yaml`

停止机器人/边缘端 replay 服务：

```bash
cd /home/data/test/workspace/flexible_lerobot

bash udp_test/all/stop_replay.sh
```

### x86 服务器：启动和关闭当前 VR/摇杆遥操作

在 x86 服务器执行：

```bash
bash /home/luzhuang/cqy/aaa/flexible_lerobot/my_devs/my_var_tp/start_live_vr_bridge.sh
```

停止当前 VR/摇杆遥操作：

```bash
bash /home/luzhuang/cqy/aaa/flexible_lerobot/my_devs/my_var_tp/stop_live_vr_bridge.sh
```

`start_live_vr_bridge.sh` 内部实际启动两个进程：

```text
robot_replay_receiver
  运行环境: lerobot_flex
  接收 target_action: 0.0.0.0:39030
  接收 robot state: 0.0.0.0:39010
  发送 robot command: 192.168.1.81:39020
  默认执行: EXECUTION=armed
  默认频率: LIVE_REPLAY_FPS=80

vr_visual_publisher
  运行环境: light_tp
  接收 VR/摇杆 UDP: 10.1.42.3:8080
  发布 target_action: 127.0.0.1:39030
  Meshcat: http://127.0.0.1:7000/static/
```

当前这两个命令可以继续作为过渡期入口。`jz_robot_pin` 做完之后，目标是让 x86 侧用新的 `my_devs/jz_robot_pin` 包装命令启动，而不是继续直接依赖 `my_var_tp` 的实验路径。

## 网络拓扑

默认网络参数：

```text
机器人/边缘端 ORIN_IP = 192.168.1.81
x86 服务器 X86_IP    = 192.168.1.106
VR/摇杆发送端 HOST   = 10.1.42.3

robot state UDP      = 39010   Orin -> x86
robot command UDP    = 39020   x86 -> Orin
target_action UDP    = 39030   visual_publisher -> x86 LeRobot control loop
VR input UDP         = 8080    VR/摇杆 -> visual_publisher
Meshcat              = 7000    x86 local visualization
```

数据流：

```text
机器人 ROS state topics
  -> orin_ros_state_udp_bridge
  -> UDP 39010
  -> JZRobotPin.get_observation()

VR/摇杆 UDP
  -> vr_visual_publisher / IK
  -> target_action UDP 39030
  -> jz_robot_pin target-action teleop/control source
  -> JZRobotPin.send_action()
  -> UDP 39020
  -> orin_phase3_command_executor
  -> ROS command topics
  -> 机器人运动
```

## 新模块目标

新增 robot backend：

```text
src/lerobot/robots/jz_robot_pin/
  __init__.py
  config_jz_robot_pin.py
  jz_robot_pin.py
  protocol.py
  udp_client.py
  state_cache.py
  rtsp_camera.py
  safety.py
```

推荐初期做成和 `jz_robot_udp` 隔离的实现：

- 不直接改 `src/lerobot/robots/jz_robot_udp/*`。
- 不改变 `jz_robot_udp` 的默认配置和行为。
- 可以复制一份必要协议和 UDP 收发逻辑到 `jz_robot_pin`，先保证新链路可控、可回滚。
- 后续稳定后，再考虑把公共 UDP/RTSP/cache 逻辑抽到共享模块。

还需要注册入口：

```text
src/lerobot/robots/__init__.py
src/lerobot/robots/utils.py
```

目标配置文件：

```text
src/lerobot/configs/robot/jz_robot_pin.yaml
```

类型名：

```yaml
type: jz_robot_pin
id: jz_robot_pin_default
```

## Robot 接口设计

`JZRobotPin` 仍然实现 LeRobot 标准 `Robot` 接口：

```python
connect()
get_observation()
send_action(action)
disconnect()
observation_features
action_features
is_connected
```

### observation_features

初始版本沿用当前双臂 + 双夹爪 + 可选 RTSP 相机：

```text
left_left_joint1.pos
left_left_joint2.pos
...
right_right_joint7.pos
left_gripper.width
left_gripper.force
right_gripper.width
right_gripper.force
camera_head
camera_left
camera_right
```

注意：现有 key 命名来自 `jz_robot_udp`，实际是：

```text
left_{joint_name}.pos
right_{joint_name}.pos
```

如果 joint name 本身已经包含 `left_` 或 `right_`，就会出现：

```text
left_left_joint1.pos
right_right_joint1.pos
```

`jz_robot_pin` 初期建议保持这个 key 兼容，避免 policy、dataset、target_action 转换逻辑同时变化。

### action_features

和 observation 里的关节/夹爪 action 对齐：

```text
left_left_joint1.pos
...
right_right_joint7.pos
left_gripper.width
left_gripper.force
right_gripper.width
right_gripper.force
```

每次 `send_action()` 发完整目标，不发增量。

### connect()

`connect()` 只连接已经启动的服务：

- 开始监听 `state_port=39010`。
- 等待第一帧 state。
- 可选连接 RTSP 相机。
- 不自动 ssh 到机器人。
- 不自动执行 `start_replay.sh`。
- 不自动切 armed。

原因：`start_replay.sh` 会启动 armed executor，真实机器人可能运动。这个动作必须保持显式。

### get_observation()

逻辑：

- 从 UDP state cache 读取最新 packet。
- 检查 sender IP。
- 检查 state age，不接受 stale state。
- 检查 joint/gripper 字段完整性。
- 读取可选相机帧。
- 返回 LeRobot flat observation。

关键配置：

```yaml
bind_ip: 0.0.0.0
state_port: 39010
allowed_state_sender_ip: 192.168.1.81
connect_timeout_s: 5.0
state_timeout_s: 0.5
```

### send_action()

逻辑：

- 检查 action key 是否完整。
- 检查 action value 是有限数值。
- 可选做 x86 侧限幅：
  - 单步最大关节变化。
  - 夹爪宽度范围。
  - action 过期检查。
- 编码为 command UDP packet。
- 发送到 `192.168.1.81:39020`。

关键配置：

```yaml
command_target_ip: 192.168.1.81
command_target_port: 39020
send_action_transport: udp
send_action_execution: dry_run | armed
command_robot: robot1
command_timeout_s: 0.2
```

安全要求：

- 默认配置必须是 `dry_run` 或 `send_action_enabled=false`。
- armed 配置必须显式设置，例如：

```bash
JZ_ROBOT_PIN_ARMED=1
```

- 文档和脚本里不能把 armed 隐藏成默认行为。

## target_action / VR 摇杆接入设计

虽然用户希望新建的是 robot，但 VR/摇杆本身仍然不是 robot，而是 action source。

为了把当前遥操作完整接入 LeRobot，建议新增或迁移一套 pin 专用 target-action teleoperator：

```text
src/lerobot/teleoperators/jz_robot_pin_target_action/
  __init__.py
  config_jz_robot_pin_target_action.py
  jz_robot_pin_target_action.py
```

类型名：

```yaml
type: jz_robot_pin_target_action
```

职责：

- 监听 `target_action_port=39030`。
- 只接受允许 IP 的 packet。
- 检查 seq/stamp/type/version。
- 转成 LeRobot flat action。
- 提供给 `lerobot-teleoperate` / `lerobot-record`。

这样最终 x86 侧标准流程可以变成：

```text
visual_publisher 只负责 VR/摇杆 -> target_action
lerobot-teleoperate 或 lerobot-record 负责:
  JZRobotPin.get_observation()
  JZRobotPinTargetActionTeleop.get_action()
  processors / safety
  JZRobotPin.send_action()
```

这会替代当前 `robot_replay_receiver.py` 的大部分职责。

## x86 侧目标启动命令

### 过渡期命令

当前可继续使用：

```bash
bash /home/luzhuang/cqy/aaa/flexible_lerobot/my_devs/my_var_tp/start_live_vr_bridge.sh
```

停止：

```bash
bash /home/luzhuang/cqy/aaa/flexible_lerobot/my_devs/my_var_tp/stop_live_vr_bridge.sh
```

### jz_robot_pin 完成后的建议命令

建议在 `my_devs/jz_robot_pin` 下提供 x86 包装脚本：

```text
my_devs/jz_robot_pin/x86/start_pin_control.sh
my_devs/jz_robot_pin/x86/start_pin_joystick.sh
my_devs/jz_robot_pin/x86/start_pin_teleop.sh
my_devs/jz_robot_pin/x86/start_pin_record.sh
my_devs/jz_robot_pin/x86/stop_pin_teleop.sh
```

其中最常用的两个启动命令是：

```bash
# 1. 启动 x86 侧 LeRobot 控制主循环：接机器人状态、接 target_action、发机器人命令
bash /home/luzhuang/cqy/aaa/flexible_lerobot/my_devs/jz_robot_pin/x86/start_pin_control.sh

# 2. 启动 x86 侧 VR/摇杆发布端：接 VR/摇杆、跑 IK、发 target_action
bash /home/luzhuang/cqy/aaa/flexible_lerobot/my_devs/jz_robot_pin/x86/start_pin_joystick.sh
```

为了日常使用，也可以提供一键脚本：

```bash
bash /home/luzhuang/cqy/aaa/flexible_lerobot/my_devs/jz_robot_pin/x86/start_pin_teleop.sh
```

它内部按顺序启动：

```text
start_pin_control.sh
sleep 2
start_pin_joystick.sh
```

停止：

```bash
bash /home/luzhuang/cqy/aaa/flexible_lerobot/my_devs/jz_robot_pin/x86/stop_pin_teleop.sh
```

### start_pin_control.sh 目标行为

启动 LeRobot 标准控制循环，等价于：

```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot

conda run --no-capture-output -n lerobot_flex python -m lerobot.scripts.lerobot_teleoperate \
  --robot.type=jz_robot_pin \
  --robot.id=jz_robot_pin_default \
  --robot.bind_ip=0.0.0.0 \
  --robot.state_port=39010 \
  --robot.allowed_state_sender_ip=192.168.1.81 \
  --robot.command_target_ip=192.168.1.81 \
  --robot.command_target_port=39020 \
  --robot.send_action_transport=udp \
  --robot.send_action_execution=armed \
  --teleop.type=jz_robot_pin_target_action \
  --teleop.id=jz_robot_pin_target_action \
  --teleop.bind_ip=0.0.0.0 \
  --teleop.target_action_port=39030 \
  --fps=80 \
  --display_data=false
```

实际脚本必须加安全确认：

```bash
JZ_ROBOT_PIN_ARMED=1
I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT=1
```

如果没有这两个环境变量，`start_pin_control.sh` 应拒绝 armed 启动。

### start_pin_joystick.sh 目标行为

启动当前 `visual_publisher` 能力，等价于：

```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot/my_devs/my_var_tp/live_vr_replay_bridge

TARGET_ACTION_IP=127.0.0.1 \
TARGET_ACTION_PORT=39030 \
VISUALIZE_WHOLE_ROBOT=true \
conda run --no-capture-output -n light_tp python -m live_vr_replay_bridge.vr_visual_publisher \
  --host 10.1.42.3 \
  --port 8080 \
  --target-action-ip 127.0.0.1 \
  --target-action-port 39030 \
  --left-ee-frame left_arm_link7 \
  --right-ee-frame right_arm_link7 \
  --tcp-control-offset 0.0 \
  --vr-debug-target-forward-offset 0.0 \
  --frequency 80 \
  --target-max-speed 0.4 \
  --publish-every 1 \
  --reset-publish-duration-s 0.5 \
  --joint-motion-cost 0.0 \
  --joint-motion-cost-profile uniform \
  --no-arm-meshes-only \
  --gripper-input trigger \
  --gripper-open-width 0 \
  --gripper-closed-width 100 \
  --gripper-force 80 \
  --gripper-publish-mode on-change
```

后续可以把 `visual_publisher` 代码从 `my_var_tp` 迁移或包装到：

```text
my_devs/jz_robot_pin/x86/
```

短期不必复制 IK 代码，先包装现有能力，降低风险。

## 录制方案

### 机器人/边缘端怎么启动

录制分两种：

#### 1. 只录状态和相机，不让 x86 发机器人命令

机器人/边缘端只启动 state bridge：

```bash
cd /home/data/test/workspace/flexible_lerobot

ORIN_IP=192.168.1.81 \
X86_IP=192.168.1.106 \
STATE_PORT=39010 \
STATE_HZ=20 \
PYTHON_CMD="conda run --no-capture-output -n lerobot python" \
AUTO_TAIL=0 \
bash udp_test/server_bash/orin_arm/start.sh
```

停止：

```bash
cd /home/data/test/workspace/flexible_lerobot

bash udp_test/server_bash/orin_arm/stop.sh
```

适用场景：

- 外部已有机器人内部遥操作在控制机器人。
- x86 只想通过 `JZRobotPin.get_observation()` 记录 observation。
- x86 不应该额外 publish command。

#### 2. 录制当前 x86 VR/摇杆遥操作，同时让 x86 发机器人命令

机器人/边缘端启动完整 replay 服务：

```bash
cd /home/data/test/workspace/flexible_lerobot

bash udp_test/all/start_replay.sh
```

停止：

```bash
cd /home/data/test/workspace/flexible_lerobot

bash udp_test/all/stop_replay.sh
```

适用场景：

- x86 的 LeRobot 控制主循环负责把 VR/摇杆 action 发到机器人。
- `lerobot-record` 同时保存 observation 和 action。

### x86 怎么启动录制

`jz_robot_pin` 完成后的推荐录制命令：

```bash
bash /home/luzhuang/cqy/aaa/flexible_lerobot/my_devs/jz_robot_pin/x86/start_pin_record.sh
```

它内部等价于：

```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot

conda run --no-capture-output -n lerobot_flex python -m lerobot.scripts.lerobot_record \
  --robot.type=jz_robot_pin \
  --robot.id=jz_robot_pin_default \
  --robot.bind_ip=0.0.0.0 \
  --robot.state_port=39010 \
  --robot.allowed_state_sender_ip=192.168.1.81 \
  --robot.command_target_ip=192.168.1.81 \
  --robot.command_target_port=39020 \
  --robot.send_action_transport=udp \
  --robot.send_action_execution=armed \
  --teleop.type=jz_robot_pin_target_action \
  --teleop.id=jz_robot_pin_target_action \
  --teleop.bind_ip=0.0.0.0 \
  --teleop.target_action_port=39030 \
  --dataset.repo_id=<user>/<dataset_name> \
  --dataset.single_task="<task description>" \
  --dataset.num_episodes=10 \
  --dataset.fps=30 \
  --display_data=false
```

录制时还需要单独启动 VR/摇杆发布端：

```bash
bash /home/luzhuang/cqy/aaa/flexible_lerobot/my_devs/jz_robot_pin/x86/start_pin_joystick.sh
```

如果想一键启动录制 + joystick，可以后续增加：

```bash
bash /home/luzhuang/cqy/aaa/flexible_lerobot/my_devs/jz_robot_pin/x86/start_pin_record_with_joystick.sh
```

## 安全设计

必须保留三层安全门：

### 1. 机器人/边缘端 executor 安全门

当前已有：

```bash
JZ_UDP_EXECUTOR_ARMED=1
```

`start_phase3_executor.sh` 在 armed 模式下没有这个环境变量会拒绝启动。

### 2. x86 robot backend 安全门

`JZRobotPin.send_action()` armed 模式必须要求：

```bash
JZ_ROBOT_PIN_ARMED=1
```

如果 `send_action_execution=armed` 但没有该环境变量，应直接报错。

### 3. x86 启动脚本安全门

`start_pin_control.sh` / `start_pin_record.sh` 如果要 armed，必须要求：

```bash
I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT=1
```

日志里明确打印：

```text
JZ ROBOT PIN ARMED
WILL send UDP commands to Orin executor
emergency stop is physical fallback, not a replacement for software limits
```

### 运行时保护

`jz_robot_pin` 应实现：

- state stale 直接拒绝继续使用旧 observation。
- target_action stale 后不再使用旧 target；实时遥操作改为根据最新 observation 发送 hold-current
  action，严格录制入口则直接失败，避免把 hold-current 帧写入数据集。
- UDP command packet 带 `seq` 和 `stamp_ns`。
- x86 侧检查 action 是否有限数值。
- 可配置单步最大关节变化，例如 `max_joint_step_rad=0.02`。
- 可配置第一帧最大跳变，例如 `max_initial_joint_delta_rad`。
- 夹爪宽度和 force 限幅。
- 发送端 IP 白名单。
- executor 端继续保留 allowed sender 和 ROS 发布限幅。

## 实施步骤

### Phase 0：文档和边界确认

- 本文档确认：
  - 新 robot 名称 `jz_robot_pin`。
  - 不改旧 `jz_robot_udp`。
  - 不自动启动机器人端 armed 服务。
  - x86 侧目标命令和机器人端目标命令。

### Phase 1：新增 robot backend

新增：

```text
src/lerobot/robots/jz_robot_pin/
```

实现：

- config dataclass。
- UDP state receiver。
- UDP command sender。
- protocol validator。
- action/observation features。
- connect/get_observation/send_action/disconnect。
- 基础单元测试。

注册：

```text
src/lerobot/robots/__init__.py
src/lerobot/robots/utils.py
```

新增配置：

```text
src/lerobot/configs/robot/jz_robot_pin.yaml
```

### Phase 2：新增 target-action teleoperator

新增：

```text
src/lerobot/teleoperators/jz_robot_pin_target_action/
```

实现：

- `target_action` UDP receiver。
- stale/allowed sender 检查。
- packet 到 flat action 的转换。
- 单元测试。

注册：

```text
src/lerobot/teleoperators/utils.py
```

新增配置：

```text
src/lerobot/configs/teleop/jz_robot_pin_target_action.yaml
```

### Phase 3：新增 x86 包装脚本

新增：

```text
my_devs/jz_robot_pin/x86/start_pin_control.sh
my_devs/jz_robot_pin/x86/start_pin_joystick.sh
my_devs/jz_robot_pin/x86/start_pin_teleop.sh
my_devs/jz_robot_pin/x86/start_pin_record.sh
my_devs/jz_robot_pin/x86/stop_pin_teleop.sh
```

要求：

- 所有 Python 运行显式使用 conda 环境。
- x86 侧 LeRobot 主进程默认使用 `lerobot_flex` 环境；机器人/边缘端包装脚本继续按当前 Orin 日志默认使用 `lerobot` 环境。
- 当前 visual publisher 可继续使用 `light_tp`，但脚本必须允许通过 `CONDA_ENV_LIGHT_TP` 覆盖。
- 写 PID 文件。
- 写 log 文件。
- stop 脚本必须先按 PID 停，再按进程名兜底。

### Phase 4：新增机器人/边缘端包装脚本

可以先包装现有 `udp_test`：

```text
my_devs/jz_robot_pin/edge/start_pin_state.sh
my_devs/jz_robot_pin/edge/start_pin_replay.sh
my_devs/jz_robot_pin/edge/stop_pin_replay.sh
my_devs/jz_robot_pin/edge/status_pin_replay.sh
```

其中：

- `start_pin_state.sh` 只启动 state bridge，适合只记录。
- `start_pin_replay.sh` 启动 state bridge + command executor，适合 x86 控制/遥操作/录制。
- `stop_pin_replay.sh` 调用 `udp_test/all/stop_replay.sh`。

### Phase 5：联调和验收

不操控真实机器人前，先做 dry-run：

```bash
conda run --no-capture-output -n lerobot_flex \
  env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
  pytest -q tests/robots/test_jz_robot_pin.py
```

本地 UDP 假包测试：

```bash
conda run --no-capture-output -n lerobot_flex python udp_test/test_scripts/x86_side/x86_jz_robot_pin_observation_check.py
```

边缘端 dry-run executor：

```bash
EXECUTION=dry_run bash udp_test/server_bash/orin_arm/start_phase3_executor.sh
```

确认：

- x86 能收到 state。
- x86 能接收 target_action。
- x86 dry-run send_action packet 能被 executor 打印。
- 不发布 ROS command topic。

最后才做 armed：

```bash
# 机器人/边缘端
JZ_UDP_EXECUTOR_ARMED=1 bash udp_test/all/start_replay.sh

# x86
JZ_ROBOT_PIN_ARMED=1 \
I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT=1 \
bash /home/luzhuang/cqy/aaa/flexible_lerobot/my_devs/jz_robot_pin/x86/start_pin_teleop.sh
```

### 三条连续录制验收

提供：

```text
my_devs/jz_robot_pin/data_check/record_and_check_3.sh
my_devs/jz_robot_pin/data_check/check_3_episodes.py
```

包装器在同一个 `lerobot-record` 进程中连续录制 3 个 episode，强制使用新数据集目录、
`RESUME=false`、18 维 action/state 和三路 RTSP 视频。录制成功后自动检查：

- 三个 episode 和 metadata 完整。
- action/state 均为 18 维且字段顺序一致。
- 第一帧 14 个关节与当前 state 的最大差值不超过 `0.02 rad`。
- 相邻 action 的最大关节步长不超过 `0.02 rad`。
- 在未来 1～6 帧中搜索 action 到 state 的最佳机械跟随延迟，而不是要求
  `action_t == observation.state_(t+1)`。
- 默认最佳延迟 MAE 不超过 `0.01 rad`、P95 不超过 `0.03 rad`。

运行前必须由操作者显式启动机器人/边缘端服务和 joystick publisher；包装器不会自动启动或
armed 机器人服务。

## 完成后的使用方式汇总

### 机器人/边缘端：只启动通信，给 x86 录状态

```bash
cd /home/data/test/workspace/flexible_lerobot

bash my_devs/jz_robot_pin/edge/start_pin_state.sh
```

停止：

```bash
cd /home/data/test/workspace/flexible_lerobot

bash my_devs/jz_robot_pin/edge/stop_pin_replay.sh
```

### 机器人/边缘端：启动通信 + 控制 executor，给 x86 遥操作/录制

```bash
cd /home/data/test/workspace/flexible_lerobot

JZ_UDP_EXECUTOR_ARMED=1 \
bash my_devs/jz_robot_pin/edge/start_pin_replay.sh
```

停止：

```bash
cd /home/data/test/workspace/flexible_lerobot

bash my_devs/jz_robot_pin/edge/stop_pin_replay.sh
```

### x86 服务器：启动遥操作

```bash
JZ_ROBOT_PIN_ARMED=1 \
I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT=1 \
bash /home/luzhuang/cqy/aaa/flexible_lerobot/my_devs/jz_robot_pin/x86/start_pin_teleop.sh
```

停止：

```bash
bash /home/luzhuang/cqy/aaa/flexible_lerobot/my_devs/jz_robot_pin/x86/stop_pin_teleop.sh
```

### x86 服务器：启动录制

```bash
JZ_ROBOT_PIN_ARMED=1 \
I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT=1 \
bash /home/luzhuang/cqy/aaa/flexible_lerobot/my_devs/jz_robot_pin/x86/start_pin_record.sh
```

录制时另开 VR/摇杆发布端：

```bash
bash /home/luzhuang/cqy/aaa/flexible_lerobot/my_devs/jz_robot_pin/x86/start_pin_joystick.sh
```

## 验收标准

认为 `jz_robot_pin` 完成，需要满足：

- `type: jz_robot_pin` 能被 `make_robot_from_config` 创建。
- 不依赖 ROS2 Python 环境也能在 x86 侧运行 robot client。
- 不修改 `jz_robot_udp` 的行为。
- dry-run 下可以完整跑通：
  - state UDP 接收。
  - target_action UDP 接收。
  - action 转换。
  - command UDP 发送。
  - executor dry-run 打印。
- armed 模式有双重显式确认。
- x86 有一键启动和停止脚本。
- 机器人/边缘端有只启动 state 和启动完整 replay 两种脚本。
- `lerobot-record` 能用 `jz_robot_pin + jz_robot_pin_target_action` 录制 observation/action。
- `lerobot-teleoperate` 能用 `jz_robot_pin + jz_robot_pin_target_action` 做实时遥操作。

## 待确认问题

- `jz_robot_pin` 是否继续沿用 `left_left_joint1.pos` 这类 key，还是趁新 robot 改成更干净的 key。初期建议沿用。
- 正式 LeRobot 主进程 conda 环境是否统一为 `lerobot_flex`。当前服务器可用的是 `lerobot_flex`，`visual_publisher` 默认是 `light_tp`，机器人/边缘端日志里使用的是 `lerobot`。
- 当前 VR/摇杆源 `10.1.42.3:8080` 是否固定，还是要做成脚本参数。
- `STATE_HZ=20` 和 x86 控制 `fps=80` 是否长期保留。当前 receiver 允许 state stale fallback，标准 LeRobot 控制循环里要明确策略。
- 录制时是否必须保存三路 RTSP 相机。若保存，`jz_robot_pin.yaml` 需要默认启用 `camera_head/camera_left/camera_right`。
