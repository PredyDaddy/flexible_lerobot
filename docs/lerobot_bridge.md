# 跨电脑 LeRobot 数据采集方案

本文档描述如何把机器人端产生的 **RTSP/GStreamer 视频流** 和 **ROS 2 关节数据**，通过一根网线传到另一台电脑（下称“采集机”），并由采集机生成可转换为 LeRobot 数据集的原始数据。

> 核心原则：**共享内存不能跨电脑使用。** 它只能是单机进程间通信手段。视频必须由采集机直接接收网络 GStreamer 流，在采集机本地解码；若现有程序必须从共享内存读图，再由采集机上的接收程序把解码后的图像写入“采集机本地”的共享内存。

## 目标架构

```text
机器人端（控制机，192.168.50.10）                 采集机（192.168.50.20）
┌──────────────────────────────────┐              ┌────────────────────────────────────┐
│ 相机 → GStreamer → RTSP 服务     │── 视频流 ───→│ 接收、解码、按时间戳缓存帧           │
│                                  │              │     └─（可选）写入本机共享内存       │
│ ROS 2 /joint_states、夹爪状态     │── DDS/以太网 →│ ROS 2 订阅、按时间戳缓存关节状态     │
│ 遥操作/控制命令（推荐也发布）     │── DDS/以太网 →│ 采集协调器：对齐、落盘、生成 episode │
└──────────────────────────────────┘              └────────────────────────────────────┘
                                                        │
                                                        ▼
                                             原始 episode → 校验 → 转为 LeRobot
```

建议由**采集机统一落盘**。不要让机器人端和采集机同时写同一个 episode，也不要在采集过程中把数据直接写到网络共享目录。

## 先做的决定

### 1. 哪台机器采集

- **机器人端**：连接相机、机械臂和夹爪，发布状态与遥操作命令，发送视频流。
- **采集机**：具备足够 SSD 空间和 GPU（训练也可在此机），接收全部数据、完成同步、保存 episode，并执行 LeRobot 转换和训练。

### 2. 记录 `state` 与 `action`

训练模仿学习策略时，二者不可混淆：

| LeRobot 字段               | 应记录的内容                             | 典型来源                        |
| -------------------------- | ---------------------------------------- | ------------------------------- |
| `observation.images.cam_*` | 相机帧                                   | GStreamer 接收端                |
| `observation.state`        | 实际关节角、夹爪实际开度等反馈           | `/joint_states`、夹爪状态 topic |
| `action`                   | 操作者下发的目标关节角/末端位姿/夹爪命令 | 遥操作控制命令 topic            |

**不要把 `/joint_states` 同时当成 `action`。** 它是机器人反馈。若当前系统没有发布操作命令，请先在控制链路中增加一个明确的命令 topic（例如 `/teleop/action`），再开始正式采集；否则只有观测数据，不能直接用于行为克隆训练。

## 网络与时间准备

### 1. 有线网络

使用千兆或更高的有线网。直连时为两台电脑配置静态地址：

```bash
# 机器人端：192.168.50.10/24
# 采集机：192.168.50.20/24
ping -c 3 192.168.50.10   # 在采集机执行
```

视频高码率且 ROS 2 发现依赖网络广播；采集时关闭 Wi-Fi 或确保 ROS 2 使用有线网卡，可减少走错网卡的情况。

### 2. 时间同步（必须）

两台机器必须同步系统时钟。先使用 `chrony`；追求更高精度时使用 PTP。同步后检查：

```bash
chronyc tracking
timedatectl status
```

目标是在采集前让两机偏差低于 5 ms。所有记录使用 `time.time_ns()`（墙钟，用于跨机器对齐）和 `time.monotonic_ns()`（本机单调时钟，用于排序与诊断）两种时间戳。ROS header stamp、视频 PTS 与接收时刻都应保留，**不要只保存一种时间戳**。

### 3. ROS 2 跨机配置

两台机器使用同一 ROS 2 发行版、相同 `ROS_DOMAIN_ID`，并允许非本机通信：

```bash
export ROS_DOMAIN_ID=50
export ROS_LOCALHOST_ONLY=0
```

在机器人端启动控制系统后，采集机验证：

```bash
ros2 topic list
ros2 topic echo /joint_states --once
ros2 topic hz /joint_states
```

若 `topic list` 看不到机器人端的话题，优先检查防火墙、网卡选择、`ROS_DOMAIN_ID` 和 DDS 的组播/单播发现配置。生产环境建议为 DDS 配置固定有线网卡和对端 peer；不要依赖办公室网络中偶然可用的组播。

## 视频链路（RTSP）

### 机器人端：提供 RTSP 流

保留机器人端现有的 RTSP 服务即可。采集机只需要知道每个相机的 RTSP URL，例如：

```bash
rtsp://192.168.50.10:8554/cam_front
# 或：rtsp://<机器人端 IP>:<端口>/<路径>
```

为稳定性起见，机器人端应使用 H.264/H.265 编码并合理设置关键帧间隔（例如 1 秒一个关键帧）。采集期间固定视频的分辨率、帧率和编码参数；改变它们会使同一个数据集的图像 feature 不一致。

### 采集机：接收并验证

先只验证画面，再接入采集程序。默认先用 TCP 传输，避免 UDP 丢包；若追求更低延迟再评估 UDP：

```bash
gst-launch-1.0 -v \
  rtspsrc location=rtsp://192.168.50.10:8554/cam_front protocols=tcp latency=100 ! \
  rtph264depay ! avdec_h264 ! videoconvert ! autovideosink sync=false
```

正式采集程序将上述管线末端替换为 `appsink` 来获取帧，并记录：`camera_name`、RTSP URL、分辨率、像素格式、GStreamer PTS、接收墙钟时间、单调时间和帧序号。若旧图像消费者依赖共享内存，让该程序在**采集机上**把帧写入共享内存；不要尝试让两台电脑挂载同一段 POSIX shared memory。

RTSP/TCP 更容易保证完整性，但仍要记录帧序号、重连次数和接收时间。每个 episode 需要记录丢帧数；若网络环境稳定且追求更低延迟，可测试 `protocols=udp`，但应重新测量端到端延迟和丢帧率。

## 采集协调器的职责

建议新增一个运行在采集机的 `dataset_recorder`（Python 或 C++ 均可）。它不直接控制机器人，只负责接收、对齐和落盘：

1. 从 GStreamer `appsink` 持续接收视频帧，保存一个短环形缓冲区。
2. 订阅 `/joint_states`、夹爪状态和**遥操作命令**，分别保存带 ROS stamp 的环形缓冲区。
3. 以固定采样频率（建议先 20 Hz；相机可保持 30 Hz）产生样本时刻 `t`。
4. 对每个 `t`，选择时间最接近且不超过阈值的图像、状态和动作；不要盲目使用“最新一条”。
5. 写入 episode 临时目录；停止后校验，再原子改名为完成状态。

推荐的起步阈值：相机匹配误差 ≤ 25 ms，关节状态匹配误差 ≤ 10 ms，动作匹配误差 ≤ 10 ms。超过阈值的样本标为无效并在元数据中计数；不要悄悄用旧数据补齐。

> 动作通常要和观测在同一个采样时刻配对。若控制器存在已知的执行延迟，应在元数据记录 `action_delay_ms`，并在转换阶段统一补偿；不要每个 episode 手工“调感觉”。

## 原始 episode 格式

在转换成 LeRobot 前，先使用简单、可审计的原始格式。一个 episode 一个目录：

```text
datasets/raw/2026-06-23_pick_cube/
└── episode_000001/
    ├── metadata.json
    ├── samples.parquet             # 每个采样时刻一行：timestamps、state、action、有效标记
    ├── frames/
    │   ├── cam_front/000000.jpg
    │   └── cam_wrist/000000.jpg
    └── events.jsonl                # start/stop、错误、掉帧、急停等事件
```

`samples.parquet` 至少应包含：

```text
sample_index, capture_time_ns, monotonic_time_ns,
cam_front_frame_index, cam_front_pts_ns, cam_front_delta_ms,
state_time_ns, state_delta_ms, state[...],
action_time_ns, action_delta_ms, action[...],
valid, invalid_reason
```

`metadata.json` 至少应包含任务描述、机器人型号、关节名称与顺序、状态/动作单位、相机内参（若有）、编码参数、软件 Git commit、两机主机名、IP、时钟偏差、采样频率和 `action_delay_ms`。关节顺序必须写死在元数据中，不能依赖字典遍历顺序。

## 采集流程

### 每次采集前

1. 检查急停、限位和手动操控安全。
2. 确认两机时间同步、网络连通、可看到 ROS topic 和视频。
3. 检查采集机磁盘余量；1080p JPEG 或视频会很快占满空间。
4. 检查关节名、单位（弧度/米/百分比）以及 `state`/`action` 向量维度与配置一致。
5. 运行 10 秒试采集，检查图像、状态、动作和各自时间偏差。

### 一次 episode

```bash
# 采集机：示例命令，参数名以实际 recorder 实现为准
dataset_recorder start \
  --task "pick up the red cube" \
  --episode-id 000001 \
  --rate-hz 20 \
  --camera cam_front=rtsp://192.168.50.10:8554/cam_front \
  --state-topic /joint_states \
  --action-topic /teleop/action

# 完成动作后
dataset_recorder stop --validate
```

建议每个 episode 保存任务成功/失败、失败原因和操作者备注。失败 episode 不要立即删除：先保留并标注，后续按训练策略筛选。

## 转换为 LeRobot 与训练

把“原始采集”和“LeRobot 转换”拆成两个独立命令。转换器负责读取完成的 raw episode、拒绝不合格数据、输出 LeRobot 数据集；它不应再读取实时 ROS topic 或实时视频流。

转换前逐项校验：

- 每个样本均有有效的图像、state、action，或被明确标记无效。
- 关节名称、顺序、维度和单位在所有 episode 一致。
- 图片能解码，帧索引连续，时间匹配误差在阈值内。
- `observation.state` 为实际反馈，`action` 为操作者命令。
- 任务文本、相机键名和 feature schema 全部一致。

将转换器固定在项目仓库中，并把使用的 LeRobot 版本写入 `metadata.json`。LeRobot 的数据 API 与版本会演进，因此不要把未经版本锁定的命令硬编码到采集端；以当前安装版本的官方文档为准，转换器只生成该版本要求的 feature schema。

训练与采集可以在同一台采集机完成，但建议先把原始数据和转换后的数据放在本地 NVMe；训练结束后再用 `rsync` 或对象存储备份。采集期间不要通过 Wi-Fi/NFS 远程写大文件——那是掉帧小精灵最爱住的地方。

## 验收标准

在开始正式采集前，连续运行 10 分钟，满足以下条件：

- 视频显示稳定，帧率达到目标的 95% 以上。
- `/joint_states` 和动作 topic 无明显断流，维度与关节顺序正确。
- 采样对齐误差的 P95：相机 ≤ 25 ms，state/action ≤ 10 ms。
- episode 停止后校验可通过，随机抽查图像、状态和动作能正确对应。
- 采集机没有磁盘写入瓶颈；CPU/GPU 占用留有余量。
- 断网、重启视频发送端或停止 ROS 节点后，recorder 会明确报错并关闭/标记 episode，不生成看似正常的坏数据。

## 最小实施顺序

1. 配置两机有线 IP、`chrony` 和 ROS 2 通信。
2. 在采集机跑通 GStreamer 接收预览。
3. 在采集机订阅并记录 `/joint_states` 与遥操作 `action` topic。
4. 实现或接入 `dataset_recorder` 的环形缓冲、时间对齐和 raw episode 落盘。
5. 录制 10 个短 episode，先做可视化和时间误差统计。
6. 编写确定性的 raw → LeRobot 转换器，验证一批数据后再扩大采集。

## 待替换的项目配置

开始实现前，请把下列占位项写入一份版本控制的 YAML/TOML 配置，而不是散落在代码中：

```yaml
robot_ip: 192.168.50.10
collector_ip: 192.168.50.20
ros_domain_id: 50
sample_rate_hz: 20
cameras:
  cam_front:
    rtsp_url: rtsp://192.168.50.10:8554/cam_front
    transport: tcp
    expected_width: 1280
    expected_height: 720
    max_delta_ms: 25
state_topic: /joint_states
action_topic: /teleop/action   # 必须替换为实际的遥操作命令 topic
joint_names: [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6]
state_unit: rad
action_unit: rad
```

这份配置中最需要你确认的是：实际视频协议/端口、状态 topic、**动作 topic**、关节与夹爪的顺序和单位。确认它们以后，采集系统的实现就可以非常直接地落地。