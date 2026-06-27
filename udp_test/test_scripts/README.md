# UDP Test Scripts

只做 UDP 联调，不发机器人控制命令。

```text
第 1/2 节：不接 ROS
第 3 节：只读 subscribe ROS state topic，不 publish command topic
```

## 目录

```text
arm_side/   在 Orin / ARM 上运行
x86_side/   拷贝到 x86 笔记本上运行
```

## 1. 测 UDP ping/pong

Orin 上运行：

```bash
cd /home/data/test/workspace/flexible_lerobot/udp_test/test_scripts/arm_side
python orin_udp_ping_server.py --bind-ip 192.168.1.81 --port 39001
```

x86 上运行：

```bash
cd x86_side
python3 x86_udp_ping_client.py --orin-ip 192.168.1.81 --port 39001 --count 20
```

把 x86 输出里的 summary 发回来。

## 2. 测 Orin -> x86 状态流

x86 上先查 IP：

```bash
ip addr
```

找到 `192.168.1.xxx`，下面用 `<X86_IP>` 代替。

x86 上先运行：

```bash
cd x86_side
python3 x86_udp_state_receiver.py --bind-ip 0.0.0.0 --port 39002 --print-every 20
```

Orin 上再运行：

```bash
cd /home/data/test/workspace/flexible_lerobot/udp_test/test_scripts/arm_side
python orin_udp_state_sender.py --bind-ip 192.168.1.81 --target-ip 192.168.1.106 --target-port 39002 --hz 20
```

把 x86 receiver 的前几行和跑 30 秒后的 lost / loss_percent / hz 发回来。

## 3. 测 ROS 状态 -> UDP -> JZRobotUDP.get_observation()

x86 上先运行：

```bash
cd /path/to/flexible_lerobot
conda run -n lerobot_flex python udp_test/test_scripts/x86_side/x86_jz_robot_udp_observation_check.py \
  --robot-config src/lerobot/configs/robot/jz_robot_udp_three_rtsp.yaml \
  --count 20 \
  --hz 5
```

Orin 上再运行：

```bash
cd /home/data/test/workspace/flexible_lerobot
# 需要在 ROS2/rclpy 可用的终端里运行
python udp_test/test_scripts/arm_side/orin_ros_state_udp_bridge.py \
  --bind-ip 192.168.1.81 \
  --target-ip 192.168.1.106 \
  --target-port 39010 \
  --hz 20
```

看 x86 输出里是否出现：

```text
SUMMARY: PASS observations=20
```

如果只想先测 UDP state，不测 RTSP 相机：

```bash
conda run -n lerobot_flex python udp_test/test_scripts/x86_side/x86_jz_robot_udp_observation_check.py \
  --robot-config src/lerobot/configs/robot/jz_robot_udp_three_rtsp.yaml \
  --count 20 \
  --hz 5 \
  --skip-cameras
```

没有 ROS 环境时，也可以用 fake state 先测 `JZRobotUDP`：

```bash
conda run -n lerobot_flex python udp_test/test_scripts/arm_side/orin_udp_state_sender.py \
  --bind-ip 127.0.0.1 \
  --target-ip 127.0.0.1 \
  --target-port 39010 \
  --hz 20 \
  --count 100 \
  --schema jz_robot_udp
```

如果 x86 check 也在同一台机器上用 `127.0.0.1` 收 fake state，需要加：

```bash
--allowed-sender-ip 127.0.0.1
```

## 文件怎么拷到 x86

只需要把这个目录拷到 x86：

```text
udp_test/test_scripts/x86_side
```

如果要测试 `JZRobotUDP`，还需要把最新仓库代码一起拉到 x86，因为它依赖：

```text
src/lerobot/robots/jz_robot_udp
src/lerobot/configs/robot/jz_robot_udp_three_rtsp.yaml
```
