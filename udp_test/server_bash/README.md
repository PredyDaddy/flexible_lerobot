# UDP Server Bash

这些脚本用于手动联调 Orin 和 x86 的 UDP 链路。

Phase 1 只读链路做：

```text
Orin: ROS2 state topic -> UDP
x86: UDP state + RTSP cameras -> JZRobotUDP.get_observation()
```

Phase 2 command dry-run 链路做：

```text
x86: JZRobotUDP.send_action() -> UDP command packet
Orin: UDP command packet -> 校验、解析、打印日志
```

当前不做：

```text
不 publish ROS command topic
不 publish cmd_vel
不控制机器人
```

Phase 2 的 Orin command receiver 是 dry-run receiver。它不会创建 ROS publisher，不会发布控制 topic，不会让机器人动作。人工急停只能作为最后兜底，不能替代这个软件安全边界。

## 启动顺序

### 1. x86 上先启动接收和 observation 监控

```bash
cd /path/to/flexible_lerobot
bash udp_test/server_bash/x86/start.sh
```

默认使用：

```text
conda run --no-capture-output -n lerobot_flex python
```

### 2. Orin 上启动 ROS 状态 UDP 桥

```bash
cd /home/data/test/workspace/flexible_lerobot
bash udp_test/server_bash/orin_arm/start.sh
```

Orin 端需要在 ROS2/rclpy 可用的终端里运行。默认使用当前终端的 `python`，如果你要显式指定：

```bash
PYTHON_CMD="python" bash udp_test/server_bash/orin_arm/start.sh
```

默认不跑本地 readiness 检查，只启动只读 ROS -> UDP bridge。需要启动前顺带做本地只读检查时：

```bash
RUN_READINESS=1 bash udp_test/server_bash/orin_arm/start.sh
```

## 停止

x86：

```bash
bash udp_test/server_bash/x86/stop.sh
```

Orin：

```bash
bash udp_test/server_bash/orin_arm/stop.sh
```

stop 脚本会读取对应 `pids/` 文件并停止脚本启动的后台进程。

注意：

```text
Ctrl-C 只会停止 tail -f，不会停止后台 Python 服务。
停止服务必须运行 stop.sh。
```

`start.sh` 默认会在启动后台服务后自动进入 `tail -f` 看日志。只想启动、不想自动看日志时：

```bash
AUTO_TAIL=0 bash udp_test/server_bash/x86/start.sh
AUTO_TAIL=0 bash udp_test/server_bash/orin_arm/start.sh
```

查看是否还有残留进程：

```bash
bash udp_test/server_bash/x86/status.sh
bash udp_test/server_bash/orin_arm/status.sh
```

## Phase 2 command dry-run

### Orin 上启动 command dry-run receiver

```bash
cd /home/data/test/workspace/flexible_lerobot
AUTO_TAIL=1 bash udp_test/server_bash/orin_arm/start_command_receiver.sh
```

默认参数：

```text
ORIN_IP=192.168.1.81
X86_IP=192.168.1.106
COMMAND_PORT=39020
PYTHON_CMD="conda run --no-capture-output -n lerobot python"
AUTO_TAIL=1
```

启动脚本会先调用 `stop_command_receiver.sh` 停掉旧 receiver，然后后台 `nohup` 启动，PID 写入：

```text
udp_test/server_bash/orin_arm/pids/orin_udp_command_receiver.pid
```

日志写入：

```text
udp_test/server_bash/orin_arm/logs/orin_udp_command_receiver.log
```

停止：

```bash
bash udp_test/server_bash/orin_arm/stop_command_receiver.sh
```

状态：

```bash
bash udp_test/server_bash/orin_arm/status.sh
```

`status.sh` 会同时显示 `ros_state_udp_bridge.py` 和 `orin_udp_command_receiver.py`。

### x86 上运行 send_action check

本脚本默认 `--command-only`，只测试 `send_action()` action path，不要求 RTSP 相机或最新 UDP state。

local dry-run：

```bash
PYTHONPATH=src conda run --no-capture-output -n lerobot_flex python \
  udp_test/test_scripts/x86_side/x86_jz_robot_udp_send_action_check.py \
  --robot-config src/lerobot/configs/robot/jz_robot_udp_three_rtsp.yaml \
  --transport local \
  --execution dry_run \
  --command-only \
  --count 20 \
  --hz 5
```

x86 -> Orin UDP dry-run：

```bash
PYTHONPATH=src conda run --no-capture-output -n lerobot_flex python \
  udp_test/test_scripts/x86_side/x86_jz_robot_udp_send_action_check.py \
  --robot-config src/lerobot/configs/robot/jz_robot_udp_three_rtsp.yaml \
  --transport udp \
  --execution dry_run \
  --command-only \
  --command-target-ip 192.168.1.81 \
  --command-target-port 39020 \
  --count 20 \
  --hz 5
```

### 本机 127.0.0.1 command dry-run

终端 1 启动 receiver：

```bash
PYTHONPATH=src conda run --no-capture-output -n lerobot_flex python \
  udp_test/test_scripts/arm_side/orin_udp_command_receiver.py \
  --bind-ip 127.0.0.1 \
  --port 39020 \
  --allowed-sender-ip 127.0.0.1 \
  --count 20
```

终端 2 发送 command：

```bash
PYTHONPATH=src conda run --no-capture-output -n lerobot_flex python \
  udp_test/test_scripts/x86_side/x86_jz_robot_udp_send_action_check.py \
  --robot-config src/lerobot/configs/robot/jz_robot_udp_three_rtsp.yaml \
  --transport udp \
  --execution dry_run \
  --command-only \
  --command-target-ip 127.0.0.1 \
  --command-target-port 39020 \
  --count 20 \
  --hz 5
```

预期摘要格式：

```text
x86: SUMMARY: PASS commands=20
receiver: SUMMARY: received=20 invalid=0 unexpected_sender=0
```

如果核心 `JZRobotUDP.send_action()` 或 command protocol helper 还没有合入，脚本会在运行时给出缺失接口错误；这不代表 Orin 端启用了任何控制能力。

## 默认 IP

```text
Orin lan2: 192.168.1.81
x86  enp130s0: 192.168.1.106
UDP state port: 39010
UDP command dry-run port: 39020
x86 connect timeout: 300s
```

临时改 IP：

```bash
X86_IP=192.168.1.xxx bash udp_test/server_bash/orin_arm/start.sh
```

## 单独跑 observation 检查

在 x86 上：

```bash
python udp_test/test_scripts/x86_side/x86_jz_robot_udp_observation_check.py \
  --robot-config src/lerobot/configs/robot/jz_robot_udp_three_rtsp.yaml \
  --count 20 \
  --hz 5
```

## lerobot-record dry-run 说明

Phase 2 允许用标准 `lerobot-record` 验证 record 流程会调用 `send_action()`，但 Orin 端仍只运行 command dry-run receiver。这个流程保存的数据只适合做链路和数据格式检查，不是最终可训练示教数据，因为 action 没有真实执行，下一帧 observation 不代表动作后的机器人状态。

record dry-run 必须满足：

```text
push_to_hub=false
本地 dataset root
安全的 dry-run teleop/policy action source
Orin receiver 日志显示 DRY_RUN / NOT publishing
机器人不动
```
