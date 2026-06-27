# UDP Robot Interface Tech Stack

本文档用于讨论后续新增 `jz_robot_udp` 机器人接口时的技术栈、模块边界和推荐落地方式。

当前文档只描述方案，不代表已经接入控制能力。

## 目标

新增一个面向 x86 的远程机器人接口：

```text
jz_robot_udp
```

它的目标不是替代 Orin 上的 ROS2 系统，而是让 x86 上的 lerobot 能通过 UDP 访问 Orin 提供的机器人状态。

第一阶段只做：

```text
get_observation()
```

不做：

```text
send_action()
```

## 总体结构

```text
        Orin / ARM                                  x86 Laptop
  +---------------------+                     +-------------------------+
  | ROS2                |                     | lerobot                 |
  | /robot1/...         |                     |                         |
  +----------+----------+                     | +---------------------+ |
             |                                | | JZRobotUDP          | |
             | subscribe only                 | | Robot interface     | |
             v                                | +----------+----------+ |
  +----------+----------+                     |            |            |
  | ROS -> UDP bridge   |                     |            v            |
  | readonly first      |                     | +----------+----------+ |
  +----------+----------+                     | | UDP client/cache    | |
             |                                | +----------+----------+ |
             | UDP state packets              |            ^            |
             +--------------------------------+------------+            |
                                              |                         |
                                              +-------------------------+
```

## 推荐命名

推荐：

```text
jz_robot_udp
```

备选：

```text
jz_robot_remote
```

不优先推荐：

```text
jz_robot_web
```

原因：

```text
当前核心通信方式是 UDP，不是 HTTP/WebSocket/Web UI。
使用 web 命名容易让人误解技术边界。
```

## 和现有 jz_robot 的区别

现有：

```text
src/lerobot/robots/jz_robot
```

定位：

```text
ROS2 direct robot
依赖 rclpy / sensor_msgs / std_msgs
直接 subscribe ROS state topic
直接 publish ROS command topic
适合 Orin 或完整 ROS2 环境
```

新增：

```text
src/lerobot/robots/jz_robot_udp
```

定位：

```text
remote robot client
x86 侧运行
不依赖 ROS2
通过 UDP 获取 observation
后续通过 UDP 发送 action
```

## 技术栈

### x86 侧

```text
Python 3
lerobot Robot interface
socket UDP
dataclasses
json / msgpack / protobuf
threading or asyncio
```

第一阶段建议：

```text
Python standard library + JSON
```

原因：

```text
容易调试
容易 tcpdump / 日志查看
方便快速验证 observation 格式
```

稳定后可升级：

```text
MessagePack
Protobuf
FlatBuffers
```

### Orin 侧

```text
Python 3
ROS2 rclpy
sensor_msgs / std_msgs
socket UDP
```

Orin 侧职责：

```text
subscribe ROS topic
组装 UDP state packet
发送到 x86
后续控制阶段做 safety gate
```

## 推荐目录结构

正式接入阶段建议：

```text
src/lerobot/robots/jz_robot_udp/
  __init__.py
  config_jz_robot_udp.py
  jz_robot_udp.py
  protocol.py
  udp_client.py
  state_cache.py
```

Orin bridge 可以单独放：

```text
src/lerobot/robots/jz_robot_udp_bridge/
  __init__.py
  config.py
  protocol.py
  ros_readonly_bridge.py
  safety_gate.py
```

启动脚本建议单独放：

```text
scripts/jz_robot_udp/
  arm_start.sh
  arm_stop.sh
  arm_status.sh
  x86_start.sh
  x86_stop.sh
  x86_status.sh
```

实验阶段继续保留：

```text
udp_test/test_scripts/
  arm_side/
  x86_side/
```

## x86 侧 Robot 接口

`JZRobotUDP` 应该实现 lerobot 标准 Robot 接口：

```text
connect()
disconnect()
is_connected
is_calibrated
calibrate()
configure()
get_observation()
send_action()
observation_features
action_features
```

第一阶段实现重点：

```text
connect()
disconnect()
is_connected
get_observation()
observation_features
```

第一阶段 `send_action()`：

```text
明确禁用
调用时报错
不发送任何 UDP 控制包
```

## x86 侧模块职责

```text
config_jz_robot_udp.py
  定义 IP、端口、timeout、joint names、feature 配置

protocol.py
  定义 UDP packet schema
  encode/decode
  version 检查
  message type 检查

udp_client.py
  绑定本地 UDP 端口
  接收 Orin state packet
  后续发送 command packet

state_cache.py
  缓存最新 observation
  记录 seq
  记录 last_receive_time
  检查 stale state

jz_robot_udp.py
  实现 lerobot Robot 接口
  把 UDP state 映射成 lerobot observation
```

## Orin 侧模块职责

```text
config.py
  定义 ROS topic、x86 IP、端口、发送频率

protocol.py
  与 x86 侧保持一致

ros_readonly_bridge.py
  subscribe ROS topic
  聚合 state snapshot
  UDP 发送到 x86

safety_gate.py
  后续控制阶段使用
  当前阶段可以先只占位
```

## UDP Packet 初始建议

第一阶段 state packet：

```text
{
  "version": 1,
  "type": "state",
  "seq": 123,
  "robot_id": "robot1",
  "timestamp_ns": 123456789,
  "payload": {
    "arm_left": {
      "joint_names": ["..."],
      "position": [...]
    },
    "arm_right": {
      "joint_names": ["..."],
      "position": [...]
    }
  }
}
```

后续可以扩展：

```text
gripper
body
head
waist
agv
battery
has_control
connected
```

## observation_features 建议

第一阶段只接双臂关节状态：

```text
left_<joint>.pos: float
right_<joint>.pos: float
```

示例：

```text
left_left_joint1.pos
left_left_joint2.pos
...
right_right_joint1.pos
right_right_joint2.pos
...
```

后续根据实际 joint name 再调整命名，目标是和 dataset / policy 的 feature key 保持稳定。

## get_observation() 行为

推荐行为：

```text
1. 检查 UDP client 是否已连接
2. 从 state cache 取最新 state
3. 检查 state 是否超时
4. 检查必要字段是否齐全
5. 转成 lerobot observation dict
6. 返回 observation
```

如果状态超时：

```text
raise TimeoutError
```

如果字段缺失：

```text
raise RuntimeError
```

## send_action() 阶段规划

第一阶段：

```text
send_action() disabled
```

行为：

```text
raise RuntimeError("JZRobotUDP action sending is disabled in readonly mode")
```

控制阶段再做：

```text
x86 send_action()
  -> UDP command packet
  -> Orin command receiver
  -> safety gate
  -> ROS command topic
```

控制阶段必须有：

```text
heartbeat
command timeout
seq 检查
mode: readonly / active_control
action limit
robot connected 检查
has_control 检查
异常时停止接受命令
```

## start.sh / stop.sh 建议

可以提供启动脚本，但职责要清晰。

```text
arm_start.sh:
  启动 Orin ROS -> UDP bridge
  绑定 192.168.1.81
  指定 x86 IP
  第一阶段只读

arm_stop.sh:
  停止 Orin bridge 进程

x86_start.sh:
  启动 x86 UDP receiver 或 lerobot client

x86_stop.sh:
  停止 x86 侧进程
```

注意：

```text
start.sh / stop.sh 负责进程管理
JZRobotUDP 负责 lerobot Robot 接口
二者不要混在一起
```

## 推荐开发顺序

```text
1. 继续 udp_test 验证 ROS 只读 topic -> UDP
2. 固定第一版 state packet schema
3. 新增 jz_robot_udp 的 config/protocol/udp_client/state_cache
4. 实现 readonly JZRobotUDP.get_observation()
5. 跑 lerobot 只读 observation 测试
6. 再讨论 send_action 和安全门
```

## 当前禁止事项

```text
不发送机器人控制命令
不 publish ROS command topic
不接 cmd_vel
不控制夹爪
不操控机器人
```

