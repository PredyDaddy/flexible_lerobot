# UDP Test Scripts

只做 UDP 联调，不接 ROS，不发机器人控制命令。

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

## 文件怎么拷到 x86

只需要把这个目录拷到 x86：

```text
udp_test/test_scripts/x86_side
```

里面已经包含运行需要的公共文件。
