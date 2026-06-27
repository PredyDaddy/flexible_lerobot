# UDP 外置通信方案背景

本文档是 `udp_test/docs/` 下的入口文档。后续其他 agent 或团队成员接手时，建议先读这份，再读其他细节文档。

## 为什么要做这件事

当前机器人主控是 NVIDIA Orin AGX。Orin 上已经运行 ROS2，负责和机器人硬件、驱动、底层通信系统交互。

现在的问题是：

```text
Orin 同时承担了太多职责：

1. ROS2 通信
2. 机器人硬件 I/O
3. lerobot 运行
4. 数据处理
5. 后续可能还有模型推理、录制、日志、调试工具
```

这会导致：

```text
Orin CPU/GPU/内存压力过大
lerobot 或模型相关逻辑占资源时，机器人通信处理可能受到影响
```

所以现在希望把系统拆开：

```text
Orin:
  继续负责 ROS2
  继续负责机器人硬件通信
  继续负责安全兜底
  尽量轻量化

x86 笔记本:
  负责 lerobot
  负责后续模型推理
  负责数据处理、录制、调试
```

简单说：

```text
Orin 做机器人通信网关
x86 做外部计算节点
```

## 为什么不用 ROS2 直接跨机器

x86 笔记本是 Ubuntu 24.04，ROS2 版本和 Orin 上的 ROS2 环境不容易完全对齐。

如果强行让 x86 也跑同一套 ROS2，可能会遇到：

```text
ROS2 版本不一致
DDS 配置问题
多网卡/组播/发现机制问题
环境维护复杂
```

所以当前方案选择：

```text
Orin 本机继续跑 ROS2
x86 不强依赖 ROS2
两边通过固定 IP + UDP 通信
```

## 当前网络

```text
        Orin / ARM                                  x86 Laptop
  +---------------------+                     +---------------------+
  | NVIDIA Orin AGX     |                     | Ubuntu 24.04         |
  |                     |                     |                     |
  | lan2                |                     | enp130s0             |
  | 192.168.1.81/24     |                     | 192.168.1.106/24     |
  +----------+----------+                     +----------+----------+
             |                                           |
             +---------------- Ethernet -----------------+
```

已验证：

```text
Orin IP: 192.168.1.81
x86 IP:  192.168.1.106
```

## 当前总体方向

目标结构：

```text
Robot hardware
      |
      v
Orin ROS2
      |
      v
Orin UDP bridge
      |
      | Ethernet UDP
      v
x86 UDP client
      |
      v
lerobot / policy / recording / tools
```

当前阶段只做只读验证：

```text
ROS/状态数据方向:
  Orin -> x86

控制命令方向:
  暂不做
```

## 当前安全边界

当前明确禁止：

```text
不发送机器人控制命令
不 publish ROS command topic
不发送 cmd_vel
不控制夹爪
不操控机器人
```

后续如果进入控制阶段，必须先设计并实现：

```text
read-only / active-control 模式切换
heartbeat
command timeout
seq 检查
action limit
robot connected 检查
has_control 检查
异常时停止接受命令
```

## 当前已经做了什么

已经完成两类 UDP 通信测试：

```text
1. x86 -> Orin -> x86 ping/pong
2. Orin -> x86 fake state stream
```

关键结果：

```text
ping/pong:
  sent=20
  received=20
  lost=0
  avg_rtt_ms=1.545
  max_rtt_ms=1.937

state stream:
  sent seq=1 ... 6000
  received=6000
  lost=0
  reordered=0
  measured hz=19.14
```

结论：

```text
Orin 和 x86 之间的 UDP 通信链路已经验证可用。
当前链路可以作为后续 ROS 只读状态桥的基础。
```

## 文档阅读顺序

建议其他 agent 按这个顺序阅读：

```text
1. background.md
   先理解为什么做 UDP 外置通信，以及当前安全边界。

2. 暂定通信结论.md
   看当前已经完成的测试、日志结论、网络信息和阶段判断。

3. udp_readonly_architecture.md
   看只读 UDP bridge 的结构图、阶段图、状态图。

4. udp_robot_interface_tech_stack.md
   看后续新增 jz_robot_udp 机器人接口时的技术栈和目录建议。
```

## 当前目录说明

```text
udp_test/docs/
  background.md
    背景入口文档，先读这个。

  暂定通信结论.md
    当前通信测试结果和阶段性结论。

  udp_readonly_architecture.md
    UDP 只读通信架构图和阶段说明。

  udp_robot_interface_tech_stack.md
    后续新增 jz_robot_udp 机器人接口的技术栈建议。
```

测试脚本目录：

```text
udp_test/test_scripts/
  README.md

  arm_side/
    Orin / ARM 上运行的脚本

  x86_side/
    x86 笔记本上运行的脚本
```

## 后续建议

下一步建议继续保持只读：

```text
把 Orin 上的 ROS2 topic:

  /robot1/arm_left/joint_states
  /robot1/arm_right/joint_states

只读桥接到 x86 UDP receiver。
```

验证真实 ROS 状态流稳定后，再考虑：

```text
新增 src/lerobot/robots/jz_robot_udp
先只实现 get_observation()
send_action() 保持禁用
```
