# UDP Server Bash

这些脚本用于手动联调 Orin 和 x86 的只读 UDP 链路。

当前只做：

```text
Orin: ROS2 state topic -> UDP
x86: UDP state + RTSP cameras -> JZRobotUDP.get_observation()
```

当前不做：

```text
不 send_action
不 publish command topic
不控制机器人
```

## 启动顺序

### 1. x86 上先启动接收和 observation 监控

```bash
cd /path/to/flexible_lerobot
bash udp_test/server_bash/x86/start.sh
tail -f udp_test/server_bash/x86/logs/jz_robot_udp_observation_check.log
```

默认使用：

```text
conda run -n lerobot_flex python
```

### 2. Orin 上启动 ROS 状态 UDP 桥

```bash
cd /home/data/test/workspace/flexible_lerobot
bash udp_test/server_bash/orin_arm/start.sh
tail -f udp_test/server_bash/orin_arm/logs/ros_state_udp_bridge.log
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

## 默认 IP

```text
Orin lan2: 192.168.1.81
x86  enp130s0: 192.168.1.106
UDP state port: 39010
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
