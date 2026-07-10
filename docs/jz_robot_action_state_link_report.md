# JZRobot 动作下发与状态上报链路报告

> 静态审查日期：2026-07-10
>
> LeRobot 工作区：`/home/data/test/workspace/flexible_lerobot`
>
> Teleop/机器人侧工作区：`/home/data/test/workspace/teleop_ws`
>
> LeRobot 基线提交：`46ad9ed723d23d6c08515c9e9eb43de451a031b8`

本文结合 `flexible_lerobot` 与 `teleop_ws` 当前源码，梳理 JZRobot 双臂、双夹爪的动作下发、状态上报、录制、回放和跨机 UDP 链路。结论来自静态代码与配置检查；本次未启动 ROS 节点、未运行机器人控制程序、未向任何控制 topic 发布消息，也未做实机时序和网络抓包验证。

文中的“状态”分为三类，不能混用：

- **物理反馈状态**：由机械臂控制器或夹爪硬件读回的值。
- **目标动作**：VR、数据集、策略或 Web 界面希望机器人达到的值。
- **命令回显**：系统把已收到或已发送的目标值重新发布；它不证明硬件已经执行到位。

## 1. 结论摘要

当前系统的主链可以概括为：

```text
动作源（VR / LeRobot replay / Web replay / UDP executor）
  -> /robot1/telecon/arm_{left,right}/joint_commands_input
  -> SmoothMotionEngine：状态初始化、名称映射、限位、轨迹生成、平滑
  -> /robot1/arm_{left,right}/joint_passthrough
  -> armcontrol：rad -> degree、写队列
  -> 华成 SDK RobotManager::PluseToServo
  -> 双臂控制器

双臂控制器
  -> 华成 SDK RobotManager::getJoints
  -> armcontrol 最新缓存、degree -> rad
  -> /robot1/arm_{left,right}/joint_states
  -> SmoothMotionEngine / LeRobot / UDP 状态桥 / 采集桥
```

夹爪不经过 SmoothMotionEngine：

```text
[width_percent, force_percent]
  -> /robot1/{left,right}_gripper/gripper_commands
  -> gripper_node -> DualGripperController -> RS485Gripper
  -> /dev/ttyTHS1 或 /dev/ttyTHS4

夹爪硬件反馈或命令回显
  -> /robot1/{left,right}_gripper/gripper_status
  -> LeRobot / UDP 状态桥 / 采集桥
```

需要优先关注的事实：

1. **动作入口没有统一的所有权仲裁。** VR、LeRobot、UDP executor、RoboWeb 等组件可以成为同一 topic 的并行 publisher，最后到达的消息可能改变目标。
2. **右夹爪当前不是物理状态。** `status_source="command"` 使右夹爪 `gripper_status` 发布上一帧 ROS 命令的延迟回显；左夹爪则读取硬件反馈。
3. **ROS Domain 配置冲突。** 机器人 systemd 服务和 `teleop_ws` 环境使用 `ROS_DOMAIN_ID=42`，而 `jz_bridge_capture.yaml` 配置为 `50`；两端不一致时 DDS topic 完全不可见。
4. **Web 回放可能运行另一份 LeRobot 代码。** RoboWeb 硬编码 `/home/test/workspace/flexible_lerobot`，该目录与本文审查的 `/home/data/test/workspace/flexible_lerobot` 是两个独立工作树，提交也不同。
5. **UDP 是三条分开的链。** `39010` 是实际状态上行，`39030` 是示教目标动作上行，`39020` 是回放/控制动作下行；三者的可靠性和时间语义不同。
6. **MindHubBridge 目前不是 JZ 双臂控制链。** 当前状态采集和任务执行以模拟实现为主，没有把 WebSocket 任务转换成本文所述的双臂 ROS 动作，也没有上传关节级真实状态。
7. **当前 armcontrol 透传会自动关闭 SDK 保护。** SDK 要求切换后等待 3 秒以上，代码仅等待 500 ms；该路径必须依赖更上层的限位、所有权和硬件安全联锁。

## 2. 组件边界与总体拓扑

### 2.1 两个工作区的职责

| 工作区 | 主要职责 |
|---|---|
| `flexible_lerobot` | LeRobot Robot/Teleoperator 适配、数据录制与回放、策略动作、Web 回放 worker、跨机 UDP 协议与桥、RTSP 采集、可选 ROS 向量聚合 |
| `teleop_ws` | ROS 2 bringup、VR 命令入口、SmoothMotionEngine、armcontrol、夹爪 RS485 驱动、RoboWeb、MindHubBridge |

`teleop_ws` 中没有 `/robot1/lerobot/state` 或 `/robot1/lerobot/action` 的原生发布实现。这两个 topic 由 `flexible_lerobot` 的 `bridge_capture` 可选桥生成，只用于聚合/采集，不是硬件执行 topic。

### 2.2 端到端拓扑

```text
                    x86 / LeRobot 侧
┌───────────────────────────────────────────────────────────────┐
│ dataset / policy / teleop                                    │
│   ├─ 直接 ROS JZRobot ------------------------------------┐  │
│   ├─ RoboWeb -> Unix socket -> jz_replay_worker ----------┤  │
│   └─ JZRobotUDP -> UDP 39020 --------------------------┐   │  │
│                                                        │   │  │
│ UDP 39010 -> JZRobotUDP observation                    │   │  │
│ UDP 39030 -> target-action teleop                      │   │  │
│ RTSP 8554 -> 图像                                      │   │  │
└────────────────────────────────────────────────────────┼───┼──┘
                                                         │   │
                    Orin / ROS 2 侧                       │   │
┌────────────────────────────────────────────────────────┼───┼──┐
│ UDP 39020 -> Phase 3 executor -> ROS command topics ---┘   │  │
│ VR UDP 8080 -> teleop_vr_recv -> ROS command topics -------┤  │
│                                                            v  │
│          SmoothMotionEngine -> armcontrol -> 华成双臂硬件     │
│                   gripper_node -> RS485 夹爪硬件              │
│                          │                                    │
│                          ├─ ROS state -> UDP 39010             │
│ 外部 command topics -------┴---------------> UDP 39030        │
└───────────────────────────────────────────────────────────────┘
```

“直接 ROS”要求两端处于可互相发现的 ROS 2 DDS 网络；“UDP 模式”则显式跨机传输状态和动作，摄像头单独走 RTSP。

## 3. ROS topic 与数据契约

### 3.1 双臂与夹爪主 topic

| 用途 | 完整 topic | ROS 类型 | 当前语义 |
|---|---|---|---|
| 左臂目标入口 | `/robot1/telecon/arm_left/joint_commands_input` | `sensor_msgs/msg/JointState` | 7 个目标关节角，rad |
| 右臂目标入口 | `/robot1/telecon/arm_right/joint_commands_input` | `sensor_msgs/msg/JointState` | 7 个目标关节角，rad |
| 左臂平滑输出 | `/robot1/arm_left/joint_passthrough` | `sensor_msgs/msg/JointState` | 送往 armcontrol 的位置/速度目标 |
| 右臂平滑输出 | `/robot1/arm_right/joint_passthrough` | `sensor_msgs/msg/JointState` | 送往 armcontrol 的位置/速度目标 |
| 左臂反馈 | `/robot1/arm_left/joint_states` | `sensor_msgs/msg/JointState` | SDK 最新缓存，degree 转 rad |
| 右臂反馈 | `/robot1/arm_right/joint_states` | `sensor_msgs/msg/JointState` | SDK 最新缓存，degree 转 rad |
| 左夹爪命令 | `/robot1/left_gripper/gripper_commands` | `std_msgs/msg/Float64MultiArray` | `[width_percent, force_percent]` |
| 右夹爪命令 | `/robot1/right_gripper/gripper_commands` | `std_msgs/msg/Float64MultiArray` | `[width_percent, force_percent]` |
| 左夹爪状态 | `/robot1/left_gripper/gripper_status` | `std_msgs/msg/Float64MultiArray` | 当前配置下为硬件查询反馈 |
| 右夹爪状态 | `/robot1/right_gripper/gripper_status` | `std_msgs/msg/Float64MultiArray` | 当前配置下为上一帧命令回显 |

双臂标准名称是 `left_joint1`…`left_joint7` 和 `right_joint1`…`right_joint7`。SmoothMotionEngine 和 LeRobot 都支持按 `JointState.name` 映射；部分代码也兼容带下划线的名称变体。消息缺少完整名称时，SmoothMotionEngine 会退回按 `position` 数组顺序解析，因此发布端必须保持固定的 7 关节顺序。

夹爪 `Float64MultiArray` 没有 header，无法携带硬件采样时间。`gripper_node` 还会把 ROS 宽度换算为 `abs(width - 100)` 后再送硬件；左侧硬件状态回传路径没有看到对应的反向换算，而右侧保持命令原值回显。因此除“左硬件/右回显”的来源差异外，左右宽度方向语义也需要现场标定确认。

### 3.2 LeRobot feature 名称

当前默认 joint name 已经包含 `left_`/`right_`，而 feature 构造时又添加一次侧别前缀，所以标准 JZRobot 与 JZRobotUDP 当前实际关节 key 形如：

```text
left_left_joint1.pos ... left_left_joint7.pos
right_right_joint1.pos ... right_right_joint7.pos
left_gripper.width
left_gripper.force
right_gripper.width
right_gripper.force
```

启用双夹爪时，共有 18 个数值字段：14 个关节、2 个夹爪宽度、2 个夹爪力度。该命名应以数据集 metadata 为准，不能仅根据 ROS joint name 推断。

### 3.3 可选聚合 topic

`bridge_capture` 可发布：

| Topic | 类型 | 固定维度 | 用途 |
|---|---|---:|---|
| `/robot1/lerobot/state` | `std_msgs/msg/Float64MultiArray` | 16 | 14 关节 + 左右夹爪宽度的状态镜像 |
| `/robot1/lerobot/action` | `std_msgs/msg/Float64MultiArray` | 16 | 14 关节 + 左右夹爪宽度的目标动作镜像 |

这两个向量排除了夹爪 force，与标准 JZRobot 的 18 维 schema 不同；它们也没有进入 SmoothMotionEngine 或 armcontrol，不能用作“已执行”的证据。

## 4. 双臂动作下发链

### 4.1 动作源

当前代码中至少存在以下动作生产者：

| 动作源 | 进入方式 | 默认状态 | 是否经过 Smooth |
|---|---|---|---|
| VR | `teleop_vr_recv` 接收 UDP `8080` 后发布 telecon topic | bringup 启动节点，但 `enable_udp_receive=false` | 是 |
| LeRobot 直接回放/策略 | `JZRobot.send_action()` 发布 telecon topic | 回放会设置 `use_external_commands=false` | 是 |
| LeRobot 外部命令录制 | `JZCommandTeleop` 订阅已有 telecon/gripper 命令 | `use_external_commands=true`，LeRobot 不重复发布 | 外部生产者本身走原链 |
| UDP 回放/控制 | Orin Phase 3 executor 接收 `39020` 后发布 telecon/gripper topic | dry-run 或 armed 取决于启动参数 | 是 |
| RoboWeb 九宫格回放 | HTTP -> replay manager -> Unix socket -> LeRobot worker | worker 强制 `use_external_commands=false` | 是 |
| RoboWeb 直接控制 | `multi_robot_bridge` 可发布 normal command 或 passthrough topic | 取决于 Web 调用 | normal 可绕过 Smooth；passthrough 直接进入 armcontrol |

系统没有发现一个位于这些生产者之前的 command mux、lease、owner token 或互斥锁。录制、回放、VR、Web 操作必须在部署层保证只有一个控制源拥有发布权。

### 4.2 LeRobot `JZRobot.send_action()`

直接 ROS 适配器的处理顺序是：

1. 将 action 字典拆成左右臂和左右夹爪目标。
2. action 只提供部分关节时，用当前状态补齐缺失关节。
3. 可选应用 `max_relative_target`。
4. `use_external_commands=true` 时，只校验并返回目标，不发布 ROS 消息。
5. 否则发布左右臂 `JointState` 和夹爪 `[width, force]`。

因此：

- **录制外部 VR/示教动作**时应使用 `true`，避免 LeRobot 把监听到的命令再写回相同 topic。
- **数据集回放或策略控制**时必须使用 `false`，否则调用成功但不会有 ROS 动作输出。

当前三 RealSense ROS 配置默认是 `use_external_commands: true`，专用 replay 和 Web worker 会显式覆盖为 `false`。

### 4.3 SmoothMotionEngine

双臂目标进入 SmoothMotionEngine 后：

1. 必须先收到对应手臂的真实状态并完成初始化，否则忽略目标。
2. 按 joint name 映射，必要时按数组顺序回退，并要求恰好 7 个位置。
3. 检查重复命令、关节绝对限位和单次输入步长。
4. 使用反馈状态进行 Kalman 更新。
5. 经 `TrajectoryInterpolator`、`TrajectoryGenerator`、`StepLimiter` 生成位置/速度命令。
6. 以约 100 Hz 发布至 `/robot1/arm_{left,right}/joint_passthrough`。

当前关键配置：

- 输入标称 100 Hz，轨迹控制定时器 100 Hz。
- 外层 system spin 配置 200 Hz，但不是轨迹发布频率。
- `max_step_per_command_deg=30.0`。
- `command_timeout_ms=100.0`。
- `enable_position_limits=true`。
- `publish_when_inactive=false`。
- Smooth 自身 `passthrough_mode=false`，即仍执行轨迹生成和平滑。

命令超时后，代码把内部目标重置/保持到当前位置；在 `publish_when_inactive=false` 时不会把“不活跃保持”持续作为新消息下发。因此 100 ms 应理解为 Smooth 输入活性门限，不应直接宣传为硬件级急停或端到端停止确认。

### 4.4 armcontrol 与华成 SDK

armcontrol 当前配置：

- 本机 IP `192.168.4.80`，控制器 IP `192.168.4.4`，端口 `12398`。
- 左臂 ID `2`，右臂 ID `1`。
- `enable_passthrough=true`。
- `auto_disable_protection=true`。
- 状态循环标称 200 Hz。

这里有两个容易混淆的开关：

- Smooth 的 `passthrough_mode=false`：上游仍做平滑和轨迹生成。
- armcontrol 的 `enable_passthrough=true`：下游消费 `joint_passthrough`，调用 `PluseToServo` 直送伺服，而不是 normal command 路径的 `moveJoints`。

armcontrol 对 passthrough 消息执行 rad 转 degree，压入 SDK 写 FIFO；独立写线程取出后调用：

```text
RobotManager::PluseToServo(robot_id, joints)
```

SDK 头文件明确说明：非保护状态配合 `PluseToServo` 时，机器人完全交给用户控制，SDK 不提供保护；切换保护状态后还要求等待 3 秒以上再发指令。当前 `SetPassthroughMode()` 自动关闭左右臂保护后只等待 500 ms，再查询保护状态。因此不能只依赖 SDK 保护，且该等待时序应按 SDK 要求整改并经过实机安全评审。

ROS publish 和本地 UDP `sendto()` 都是 fire-and-forget；当前 LeRobot 主链没有消费一个能够证明“控制器已接收且硬件已达到目标”的端到端 ACK。

## 5. 双臂与夹爪状态上报链

### 5.1 双臂物理反馈

双臂状态路径是：

```text
机械臂控制器
  -> RobotManager::getJoints
  -> armcontrol SDK 读线程
  -> 按 robot ID 更新共享缓存
  -> degree 转 rad
  -> JointState(left_joint1...7 / right_joint1...7)
  -> /robot1/arm_{left,right}/joint_states
```

同一份原始状态 topic 被多个组件并行消费：

- SmoothMotionEngine：初始化、状态估计和平滑控制。
- 直接 ROS `JZRobot`：生成 LeRobot observation。
- Orin UDP 状态桥：打包并发送到 x86 `39010`。
- `bridge_capture`：聚合为 `/robot1/lerobot/state`。
- 其他 Web/诊断组件。

需要注意状态时间语义：`JointState.header.stamp` 是 ROS 发布时刻，不是控制器硬件采样时刻；armcontrol 缓存也没有独立 freshness/seq。SDK 读取停滞时，理论上可能以新 ROS 时间戳再次发布旧缓存。因此它是“最近一次可用 SDK 状态”，不是严格硬件采样对齐的数据。

### 5.2 夹爪状态

夹爪状态发布频率配置为 100 Hz，但左右来源不一致：

- 左夹爪调用 `get_status_percent("left")`，来自 RS485 接收/查询缓存。
- 右夹爪在 `status_source="command"` 时，通过 `popDelayedRightCommandStatus()` 发布上一帧命令。

所以上层录制到的 `right_gripper.width/force` 当前应标注为**命令回显**，不能用于分析跟踪误差、堵转、抓取是否成功或硬件通信健康。若需要真实闭环状态，应将右侧改为硬件 RX，并同时确认左右宽度方向换算一致。

### 5.3 LeRobot observation 缓存

直接 ROS `JZRobot.connect()` 为左右臂和夹爪创建 subscriber，在后台 `SingleThreadedExecutor` 线程更新受锁保护的最新值。`get_observation()` 检查首帧与 stale timeout，再读取当前缓存和相机帧。

这是一种“调用时最新值”采样，而不是所有传感器共享硬件触发或统一时间戳。标准 record 循环通常先取 observation，再取 teleop/policy action，图像也可能有独立缓存延迟。

## 6. LeRobot 直接录制、回放与 Web 回放

### 6.1 外部命令录制

推荐三 RealSense 录制链为：

```text
armcontrol/夹爪状态 -> JZRobot.get_observation()
外部 telecon/gripper 命令 -> JZCommandTeleop.get_action()
observation + target action -> LeRobotDataset.add_frame()
```

`JZCommandTeleop` 只镜像已有命令 topic。`JZRobot.send_action()` 仍会被通用 record loop 调用，但在 `use_external_commands=true` 时不会再次向机器人发布。

标准 record 实现还有一个需要修正的语义差异：代码调用 `_sent_action = robot.send_action(...)`，但数据集实际写入的是调用前的 `action_values`，没有使用 `_sent_action`。若 `send_action()` 内部补齐或限幅，数据集中的 action 可能不是最终返回/发布的 action，与旁边“保存实际发送动作”的注释不一致。

### 6.2 数据集回放

直接回放链为：

```text
LeRobotDataset action vector
  -> 按 dataset feature name 还原 action dict
  -> robot.get_observation()
  -> robot action processor
  -> JZRobot.send_action()
  -> ROS telecon/gripper command topics
```

回放需要 `use_external_commands=false`。专用 `run_replay_jz_three_realsense.py` 默认不连接相机，但仍等待机器人状态首帧，因为 action 补齐/限幅和连接状态依赖 observation。

### 6.3 RoboWeb 九宫格回放

Web 触发路径为：

```text
POST /api/jz-replay/actions/{slot_id}/trigger
  -> JZReplayActionManager
  -> JZReplayWorkerClient
  -> Unix domain socket JSON
  -> my_devs.jz_robot.jz_replay_worker
  -> JZRobot(use_external_commands=false)
  -> ROS 主动作链
```

worker 常驻后加载数据集、连接 robot，并在线程中逐帧调用 `get_observation()`、processor 和 `send_action()`。

当前部署风险是 RoboWeb 管理器硬编码：

```text
/home/test/workspace/flexible_lerobot
/home/test/miniconda3/envs/lerobot/bin/python
```

`/home/test/workspace/teleop_ws` 是当前 `/home/data/test/workspace/teleop_ws` 的符号链接，但 `/home/test/workspace/flexible_lerobot` 不是当前工作区的符号链接。静态检查时两份 LeRobot 工作树提交分别为：

```text
/home/data/test/workspace/flexible_lerobot  46ad9ed723d23d6c08515c9e9eb43de451a031b8
/home/test/workspace/flexible_lerobot       4a640ac9403656d89dea7e9dff5e45d551f27e1d
```

因此 Web 页面实际执行的 worker、配置和数据集路径可能与本文档代码不一致。该路径应改为部署配置或环境变量，并在 Web 状态接口中暴露实际 commit/workspace。

## 7. 跨机 UDP 链路

UDP 模式把“状态上行”“目标动作上行”和“执行动作下行”明确拆成三个端口：

| 端口 | 方向 | packet type | 含义 |
|---:|---|---|---|
| `39010/udp` | Orin -> x86 | `state` | 实际/缓存状态上行 |
| `39030/udp` | Orin -> x86 | `target_action` | 外部示教目标动作镜像上行 |
| `39020/udp` | x86 -> Orin | `command` | 回放或控制动作下行 |

视频不走这些 JSON packet，而是由 Orin RTSP `8554` 提供给 x86。

### 7.1 状态上行：39010

```text
ROS arm/gripper state
  -> Orin ReadonlyStateCollector 最新值缓存
  -> 固定频率构造 state JSON（默认 20 Hz）
  -> UDP 39010
  -> x86 UDPReceiver 后台线程
  -> StateCache
  -> JZRobotUDP.get_observation()
```

x86 会校验 packet schema、源 IP 和“本机收到包距现在的时间”。但 Orin 桥没有记录每个 ROS source 的最后更新时间：底层 ROS 状态断流后，它仍可能反复发送旧缓存并写入新的 packet 时间。此时 x86 的 freshness 只证明 UDP 包持续到达，不能证明机械臂/夹爪反馈仍然新鲜。

此外，x86 状态接收侧没有把 packet `seq` 单调性和 `robot` 身份作为完整门禁，乱序晚到包仍可能覆盖最新缓存。

### 7.2 目标动作上行：39030

```text
VR/其他上游发布的 telecon 和 gripper commands
  -> Orin TargetActionCollector
  -> 固定 30 Hz 构造 target_action JSON
  -> UDP 39030
  -> x86 JZRobotUDPTargetActionTeleop
  -> record loop 的 action
```

这是“目标动作”，不是执行反馈。桥同样会周期性重发最后缓存，且 ready 条件主要保证左右臂完整，夹爪首帧不足时可能使用默认 `0.0/0.0`。录制端应保存源 seq/stamp/freshness，或至少把这些质量字段写入旁路日志。

### 7.3 动作下行：39020

```text
dataset/policy action
  -> JZRobotUDP.send_action()
  -> version/type/robot/seq/stamp_ns/mode/actions JSON
  -> UDP 39020
  -> Orin Phase 3 command executor
  -> sender/schema/robot/freshness/seq/mode/数值/rate gates
  -> ROS telecon 和 gripper command topics
  -> Smooth/armcontrol/RS485 主执行链
```

`JZRobotUDP` 的 `send_action_transport=local` 只构包和记录，不发网络；`udp` 才会 `sendto()`。`execution=dry_run` 与 `armed` 也必须区分。

Orin executor 有源 IP、schema、robot、时间戳、seq、armed、topic allowlist 和 timeout 等门禁；但当前 `orin_phase3_executor_config_armed_hold.yaml` 同时设置：

- `allow_publish_rate_limit_bypass: true`；
- 关节绝对位置、关节 delta、夹爪范围、夹爪 delta 的 bypass 全为 `true`。

因此配置中的数值上下限和 2 Hz 字段不能被当作当前 armed 模式的硬安全边界。`udp_test/all/start_replay.sh` 还会按默认 armed 路径包装启动 executor；任何现场回放都应另行做显式人员确认和单一控制源隔离。

UDP command 没有执行 ACK、重传、鉴权或加密。x86 `sendto()` 成功只表示数据交给本机 UDP 栈，不证明 Orin 接收、ROS 发布、SDK 执行或硬件到位。

### 7.4 UDP 录制与回放模式

| 模式 | observation | action 来源 | x86 是否向 Orin 发动作 | 备注 |
|---|---|---|---|---|
| UDP 录制默认 | `39010` + RTSP | `39030` target action | 否，`local + dry_run` | 机器人可能仍由外部 VR 控制 |
| UDP hold 调试 | `39010` + RTSP | 当前 observation 复制 | 否，默认 local | 不等价于真实示教目标 |
| UDP replay dry-run | `39010` | dataset | 可构包但不应执行 | 用于协议和数据检查 |
| UDP replay armed | `39010` | dataset | `39020` | 具备实机动作能力 |

标准 UDP record 是按主循环依次取“最新状态、三路 RTSP、最新 target action”，packet seq/stamp 没有进入标准 dataset。状态默认 20 Hz、target action 30 Hz、record 常用 30 FPS，因此相邻样本会复用状态；三路 RTSP 的 `async_read()` 当前实际同步读取，还会影响跨模态时间关系。

## 8. `bridge_capture` 原始采集旁路

可选采集流程为：

```text
原始 ROS state/action sources
  -> ActionAggregator
  -> /robot1/lerobot/state 与 /robot1/lerobot/action
  -> RawRecorder ring buffers

RTSP cameras -> image buffers

固定采样时刻 latest_not_after 匹配
  -> raw episode
  -> offline converter
  -> LeRobotDataset
```

它与标准 `lerobot-record`、JZRobotUDP 录制互不依赖，主要价值是显式缓存和时间窗口匹配。需要注意：

- state/action 只有 16 维，不含 gripper force。
- 聚合 action 仍是输入命令镜像，不是执行确认。
- 当前配置写 `ros_domain_id: 50`，与机器人运行域 `42` 冲突。
- 机器人 systemd 服务显式设置 `ROS_DOMAIN_ID=42`；`teleop_ws/src/robot_bringup/scripts/env.sh` 也为 `42`。
- `start_robot.sh` 本身不 source `env.sh`，手工启动若没有预设环境，可能进入默认域 `0`。

凡通过 ROS 2 DDS 直连的机器人侧、采集侧和推理侧都必须统一 Domain。按当前现场 systemd 配置，应先以 `42` 为部署基准，或一次性统一修改所有组件后再使用其他值。

## 9. RoboWeb 与 MindHub 管理面边界

### 9.1 RoboWeb

RoboWeb 同时包含两类能力：

- 管理/状态 API、WebSocket、心跳等控制面能力。
- 可直接发布 arm normal command、arm passthrough、gripper command，或触发 LeRobot replay worker 的动作能力。

其中 direct normal command 可绕过 SmoothMotionEngine，passthrough 更会直接进入 armcontrol。报告和部署日志必须记录“命令实际来自哪个 publisher、进入哪个 topic”，不能仅写成笼统的“Web 控制”。

### 9.2 MindHubBridge

MindHubBridge 设计了 HTTP 心跳/状态上传、WebSocket 任务下发和执行状态回传，但当前代码的 `StatusCollector` 和本地任务执行器主要是模拟实现；任务只覆盖模拟巡检、停止和截图等，没有转换成 JZ 双臂 joint/gripper command。bringup 所需的实际 `robot.json` 也可能缺失并导致组件跳过启动。

因此目前不能画成：

```text
云端 task -> JZ 双臂动作
JZ 真实关节状态 -> 云端状态
```

RoboWeb 的 Hub reporter 上传的也主要是 AGV/媒体/任务摘要，而不是本文 14 关节 + 夹爪的实时闭环状态。若后续需要“云端动作下发与状态上传”，必须新增明确的 task-to-action adapter、命令所有权、安全门禁、真实状态 collector 和端到端 ACK。

## 10. 运行模式判定表

| 场景 | LeRobot/桥的作用 | `use_external_commands` 或 execution | 本组件是否主动输出硬件动作 |
|---|---|---|---|
| 直接 ROS 外部示教录制 | 订阅状态与外部命令并写 dataset | `use_external_commands=true` | LeRobot 不输出；外部控制源可能输出 |
| 直接 ROS 数据集回放 | 从 dataset 发布 ROS 目标 | `use_external_commands=false` | 是 |
| VR 示教 | UDP 8080 转 ROS 目标 | `enable_udp_receive=true` 后 | 是 |
| UDP 录制默认 | 收 `39010/39030`、RTSP，写 dataset | `local + dry_run` | x86 不输出；外部控制源可能输出 |
| UDP 回放 dry-run | 校验和构包 | `dry_run` | 否 |
| UDP 回放 armed | `39020` 下发到 Orin executor | `armed` | 是 |
| bridge_capture | 聚合/缓存/离线转换 | 只读源 topic | 否 |
| RoboWeb replay | 启动 LeRobot worker | worker 使用 external=false | 是 |
| MindHubBridge 当前实现 | 心跳、模拟任务、截图 | 无 JZ action adapter | 否 |

## 11. 风险清单与建议

### 11.1 P0：部署或安全阻断项

| 风险 | 当前证据 | 建议 |
|---|---|---|
| 多个动作 publisher 无所有权仲裁 | VR、LeRobot、UDP executor、RoboWeb 可写相同或下游 topic | 增加 command mux/lease；每次只允许一个 owner；记录 publisher GID/来源 |
| ROS Domain 42/50 冲突 | systemd/env 为 42，bridge config 为 50 | 统一部署配置；启动时打印并校验 Domain |
| Web worker 指向旧工作树 | RoboWeb 硬编码 `/home/test/workspace/flexible_lerobot` | 改成配置项；API 暴露 workspace、commit、conda python |
| 右夹爪状态不是物理反馈 | `status_source="command"` | 改为 RX 或重命名为 command_echo；数据集 metadata 标明来源 |
| armed executor 绕过数值/rate 限制 | armed hold 配置多个 bypass=true | 恢复硬限位、delta 和速率门禁；旁路只能用于受控调试且要审计 |
| RoboWeb 可绕过 Smooth | direct normal/passthrough publisher | 生产环境只开放统一安全入口，限制或移除直达 publisher |
| passthrough 会关闭 SDK 保护且等待时序不足 | SDK 注释要求切换后等待 3 秒以上，代码仅等待 500 ms | 阻止过早发指令；按 SDK 规范整改并增加硬件级安全联锁 |

### 11.2 P1：数据正确性与可观测性

| 风险 | 影响 | 建议 |
|---|---|---|
| Orin 状态桥用新 packet 时间重复旧 ROS 缓存 | stale 反馈看起来仍新鲜 | 每个 source 保存 ROS recv time；任何 source 超时则标 invalid/停止发包 |
| target-action 桥周期重发旧目标 | 录制 action 的真实发生时刻丢失 | 上报原始 recv stamp、seq、valid mask |
| 无端到端执行 ACK | 无法区分发布成功与执行成功 | 增加 accepted/executing/reached/fault 状态及 command_id |
| UDP 明文、无鉴权、无重传 | 丢包、伪造、网络故障不可闭环 | 控制网隔离；至少加签名/会话 token/ACK，必要时使用可靠传输 |
| command freshness 依赖两机墙钟 | executor 用 `stamp_ns` 计算 age/skew | 为 Orin/x86 部署时钟同步和偏差监控；超差拒绝 armed |
| record 忽略 `_sent_action` | 限幅/补齐后动作可能与 dataset 不一致 | 数据集写入实际返回动作，并可同时保存 raw target |
| 状态/action/图像没有严格同步 | 训练样本存在相位偏差 | 保存各源采样时间、接收时间和 seq；离线按统一时钟对齐 |
| feature 双前缀且 schema 不统一 | 模型/数据转换易错 | 明确 schema version；迁移为单一 canonical name；转换时校验 metadata |
| 16 维 bridge 与 18 维标准 schema 不同 | 数据和模型不能直接混用 | 为两套 schema 使用不同名称/version，禁止静默拼接 |
| armcontrol/夹爪状态缺少硬件采样时间 | 无法量化闭环延迟 | 驱动层增加 controller stamp/seq；至少保存 SDK 完成时间 |

### 11.3 P2：维护性问题

- `teleop_vr_recv` 的旧 README 对 arm command 类型描述可能与当前 `JointState` 实现不一致，应以源码和 `ros2 topic info` 为准。
- Smooth 配置中个别注释与数值/布尔值不一致，例如 100 ms 附近的“10 秒”描述、`publish_when_inactive=false` 旁的“仍发布”描述，容易误导运维。
- MindHub 模拟状态和真实机器人状态需使用不同 schema/字段标识，避免管理平台把 mock 数据当作反馈。
- armcontrol SDK 读写采用 FIFO；若 SDK 变慢，应监控队列深度、丢弃策略和命令 age，避免旧目标积压。

## 12. 建议的端到端观测字段

若后续统一直连 ROS、UDP 和云端链路，建议每个动作至少携带并记录：

```text
command_id
source_id / owner_id
schema_version
source_stamp_ns
receive_stamp_ns
publish_stamp_ns
execution_mode
raw_target
limited_target
accepted / rejected + reason
controller_ack_stamp_ns
latest_state_seq / latest_state_stamp_ns
reached / fault
```

每份 observation 至少记录：

```text
hardware_or_sdk_stamp_ns
ros_publish_stamp_ns
collector_receive_stamp_ns
source_age_ms
valid_mask
state_source = hardware | cache | command_echo | simulated
```

其中 `state_source` 对当前右夹爪尤其重要。

## 13. 只读现场核对建议

以下检查不会发布动作，但仍应在机器人现场规程允许的终端执行；不要调用 replay、trigger、enable service 或 `ros2 topic pub`：

```bash
printenv ROS_DOMAIN_ID
ros2 topic info -v /robot1/telecon/arm_left/joint_commands_input
ros2 topic info -v /robot1/telecon/arm_right/joint_commands_input
ros2 topic info -v /robot1/arm_left/joint_passthrough
ros2 topic info -v /robot1/arm_right/joint_passthrough
ros2 topic hz /robot1/arm_left/joint_states
ros2 topic hz /robot1/arm_right/joint_states
ros2 topic echo /robot1/left_gripper/gripper_status --once
ros2 topic echo /robot1/right_gripper/gripper_status --once
ss -lunp
```

重点核对：

1. telecon topic 是否存在多个 publisher。
2. 实际 ROS Domain 是否为 42，采集端是否一致。
3. Smooth 输出是否约 100 Hz、arm state 是否约 200 Hz、夹爪 status 是否约 100 Hz。
4. 右夹爪 status 是否只随命令变化而变化。
5. 监听 `39010/39020/39030` 的进程和目标 IP 是否符合当前部署。
6. Web replay worker 的实际 cwd、Python 和 Git commit 是否为预期版本。

## 14. 关键源码索引

### 14.1 `flexible_lerobot`

- 直接 ROS Robot 配置：`src/lerobot/robots/jz_robot/config_jz_robot.py:23-83`
- 直接 ROS 状态缓存、连接和动作发布：`src/lerobot/robots/jz_robot/jz_robot.py:186-219,449-542,544-639`
- 三 RealSense ROS 配置：`src/lerobot/configs/robot/jz_robot_three_realsense_ros2_topics.yaml:28-82`
- 外部命令 teleop：`src/lerobot/teleoperators/jz_command_teleop/jz_command_teleop.py:175-229,417-484`
- record 循环：`src/lerobot/scripts/lerobot_record.py:336-399`
- replay 循环：`src/lerobot/scripts/lerobot_replay.py:96-134`
- 专用录制/回放入口：`my_devs/jz_robot/run_record_jz_three_realsense.py`、`my_devs/jz_robot/run_replay_jz_three_realsense.py`
- UDP 顶层录制/回放入口：`record.sh`、`replay.sh`、`udp_test/all/start_record.sh`、`udp_test/all/start_replay.sh`
- Web replay worker：`my_devs/jz_robot/jz_replay_worker.py:116-192,240-328,354-420`
- UDP Robot：`src/lerobot/robots/jz_robot_udp/jz_robot_udp.py:123-307`
- UDP 协议：`src/lerobot/robots/jz_robot_udp/protocol.py:39-76,107-216`
- UDP receiver/sender/cache：`src/lerobot/robots/jz_robot_udp/udp_client.py:17-128`、`state_cache.py:11-59`
- Orin 状态桥：`udp_test/test_scripts/arm_side/orin_ros_state_udp_bridge.py:39-187`
- Orin target-action 桥：`udp_test/test_scripts/arm_side/orin_ros_target_action_udp_bridge.py:47-223`
- Orin command executor：`udp_test/test_scripts/arm_side/orin_phase3_command_executor.py:357-576,732-907`
- armed 配置：`udp_test/test_scripts/arm_side/orin_phase3_executor_config_armed_hold.yaml`；若部署路径不同，以启动脚本实参为准
- bridge_capture 配置：`src/lerobot/configs/robot/jz_bridge_capture.yaml:1-119`
- ROS 聚合桥：`src/lerobot/robots/jz_robot/bridge_capture/ros_action_bridge.py:25-101`
- raw recorder：`src/lerobot/robots/jz_robot/bridge_capture/raw_recorder.py:91-201`

### 14.2 `teleop_ws`

- 现场 systemd 启动环境（非仓库文件）：`/etc/systemd/system/robot_bringup.service:13-17`
- 总 bringup：`src/robot_bringup/launch/bringup.launch.py:43-108,155-292,330-364`
- 机器人启动环境：`src/robot_bringup/scripts/env.sh:1-4`
- VR 配置与发布：`src/teleop_vr_recv/config/teleop_vr_recv.toml:7-48`、`src/teleop_vr_recv/src/teleop_node.cpp:376-484`
- Smooth 配置：`src/SmoothMotionEngine/config/joint_executor_config.toml:3-99`
- Smooth ROS 输入、状态和输出：`src/SmoothMotionEngine/src/joint_executor_ros.cpp:315-455,504-759,1066-1453`
- armcontrol 配置：`src/armcontrol/config/arm_controller_config.toml:3-54`
- armcontrol SDK 读写与 ROS 桥：`src/armcontrol/src/arm_control_ros.cpp:744-847,1474-1522,1808-2029`
- SDK 保护模式约束：`src/armcontrol/lib/huachengSDK/include/robot.h:87-92`、`src/armcontrol/src/arm_control_ros.cpp:1273-1310`
- 夹爪配置：`src/gripper/config/gripper_config.toml:1-60`
- 夹爪 ROS 命令和状态：`src/gripper/src/gripper_node.cpp:201-234,261-311,375-403`
- RS485 驱动：`src/gripper/src/driver/rs485_gripper.cpp:152-210,257-303,336-456`
- RoboWeb replay API：`src/roboweb/roboweb/api/jz_replay_endpoints.py:54-98`
- RoboWeb replay manager：`src/roboweb/roboweb/core/jz_replay_action_manager.py:51-282`
- RoboWeb worker client：`src/roboweb/roboweb/core/jz_replay_worker_client.py:20-253`
- RoboWeb 直接 ROS bridge：`src/roboweb/roboweb/core/multi_robot_bridge.py`
- MindHub status/task/control：`src/mindHubBridge/robot_edge_agent/status.py`、`task_executor.py`、`control_ws.py`

## 15. 最终判定

当前可确认的生产级硬件主路径是“telecon 目标 topic -> SmoothMotionEngine -> arm passthrough -> armcontrol -> 华成 SDK”和相反方向的“华成 SDK -> armcontrol state topic”。LeRobot、VR、UDP executor 与 RoboWeb 是这条主路径的不同上游；`bridge_capture` 是采集旁路；MindHub 是尚未接入真实关节控制/反馈的管理面。

在进行任何实机回放或策略下发前，至少应先完成：统一 ROS Domain、锁定唯一命令 owner、确认运行的是当前 LeRobot 工作树、恢复 armed 数值安全门禁、把右夹爪状态来源改成真实反馈或明确标注为命令回显。否则即使单个组件日志显示“发布成功”，也无法证明整条链路的动作归属、状态真实性和端到端执行结果。
