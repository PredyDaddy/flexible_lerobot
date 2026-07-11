# JZ Robot Pin 视频编码与时序接收设计

本文记录两个后续改进方向：

1. 将 Pin 数据录制的最终 H.264 编码从隐式 `CRF=30` 改为显式、可配置的 `CRF=18`。
2. 新增 `jz_robot_pin_timed` Robot 类型，复用现有 Pin/UDP 控制代码，只替换带时间戳的相机接收与采样对齐部分。

本文最初是设计说明，目前对应实现已经落在新实验类型 `jz_robot_pin_timed`。旧
`jz_robot_pin` 保持原行为，作为现场回退路径；正式录制是否切换到新类型，仍需完成上机的
只读时序验证和短数据验收。

## 1. CRF 对训练数据的影响

### 1.1 当前图像链路

当前采集到的临时 PNG 不是传感器 RAW。完整链路是：

```text
相机 / Orin H.264 RTSP（第一次有损）
    -> X86 OpenCV 解码为 RGB
    -> 每个采样时刻保存临时 PNG（无损）
    -> X86 libx264, CRF=30, yuv420p, GOP=2（第二次有损）
    -> 删除临时 PNG
    -> 训练时从 MP4 解码为 RGB Tensor
```

训练读取的是最终 MP4 解码后的像素，不会读取录制阶段的临时 PNG。因此第二次 H.264
编码产生的模糊、色块、色偏和隔帧量化闪烁都会进入视觉模型。

数值 `action` 和 `observation.state` 保存在 Parquet 中，不经过视频编码，不会受到 CRF
影响。CRF 只影响视觉输入。

### 1.2 为什么此前使用了 CRF 30

这不是一个单独的编码器故障，而是通用默认参数与 Pin 实际编码路径不匹配：

1. LeRobot 视频 benchmark 选出的通用组合是 `libsvtav1 + yuv420p + GOP=2 + CRF=30`。
2. Pin 的 `record.sh` 为编码速度和兼容性将 codec 显式改成了 `h264`。
3. 录制配置只暴露了 `vcodec`，没有暴露 `crf`、`gop` 和 `pix_fmt`。
4. `LeRobotDataset` 调用编码 worker 时只传 `vcodec`，所以 H.264 继续使用
   `encode_video_frames()` 的默认 `CRF=30`。
5. 不同 codec 的 CRF 数字不能简单横向等价。适合 AV1 benchmark 的数值不能未经验证地
   直接当作 libx264 的 Pin 数据质量目标。
6. 原始 RTSP 已经是 H.264，最终又用 H.264 编码一次，形成二次有损压缩。
7. 现有数据检查主要验证 shape、帧数和 action/state，过去没有把编码前后画质和色彩元数据
   纳入验收条件。

因此此前的问题可以归纳为：**codec 被改成 H.264 后，质量参数没有随之显式配置和验收。**

### 1.3 当前实测

使用同一段右腕 RTSP 原流生成 GOP=2、yuv420p 的不同 CRF 版本：

| CRF | 实际码率 | SSIM | PSNR |
|---:|---:|---:|---:|
| 18 | 3.60 Mbps | 0.9818 | 44.59 dB |
| 20 | 2.69 Mbps | 0.9796 | 43.95 dB |
| 22 | 2.01 Mbps | 0.9770 | 43.27 dB |
| 30 | 0.75 Mbps | 0.9613 | 39.69 dB |

这些指标证明 CRF 30 相对 RTSP 解码帧丢失了更多像素信息，但不能单独证明策略成功率下降。
CRF 18 的作用是减少 X86 最终落盘阶段的额外损失，它不能修复 RTSP 源流已经存在的运动
模糊、过曝、色偏或源编码损失。

现有 `jz_robot_pin_real_20260710_180651` 中，Parquet 共 896 行，三路 MP4 也各 896 帧。
目前没有证据表明 PNG 转 MP4 的编码阶段删除了已保存帧。播放器中 episode 边界的硬切和
采集阶段可能发生的源帧跳过，需要与编码画质问题分开判断。

### 1.4 推荐的正式编码参数

在已经完成视觉检查且存储空间允许的前提下，Pin 正式录制推荐：

```text
codec    = h264 / libx264
crf      = 18
pix_fmt  = yuv420p
gop      = 2（第一阶段保留，避免改变 LeRobot 随机取帧性能）
fps      = 30
```

CRF 是编码参数，不是解码参数。训练端不会“解码成 CRF 18”，而是解码一个使用 CRF 18
生成的视频。已经使用 CRF 30 编码的数据重新转成 CRF 18，不能恢复之前丢失的细节。

### 1.5 当前代码如何实现

没有直接修改 `video_utils.py` 的全仓库默认值，因为这会影响所有机器人和数据转换工具。
当前实现新增显式参数，并由 timed 录制入口选择 18：

```text
my_devs/jz_robot_pin_timed/data_check/record_and_check_3.sh
    -> VIDEO_CRF=18
my_devs/jz_robot_pin_timed/record.sh
    -> --dataset.video_crf=18
src/lerobot/scripts/lerobot_record.py
    -> DatasetRecordConfig.video_crf
src/lerobot/datasets/lerobot_dataset.py
    -> 保存 codec/CRF 元数据，并传给所有 encoding worker
src/lerobot/datasets/video_utils.py
    -> 继续使用已有 encode_video_frames(..., crf=...) 接口
```

当前已经完成：

- 校验不同 codec 的 CRF 取值范围。
- 在录制开始日志打印 codec、CRF、GOP、pix_fmt 和 FPS。
- 在数据集 metadata 中记录实际编码参数，避免只看到 codec/pix_fmt 而不知道 CRF。
- 为参数从 CLI 到 worker 的完整传递增加单元测试。
- timed 三条录制包装器在录制后验证 MP4、Parquet 和时序 sidecar。

仍建议在相机或光照改变后，用同一批无损母版重新生成 CRF 18/20/30 做质量 A/B。旧
`jz_robot_pin/record.sh` 没有被静默改成 18；新参数由 `jz_robot_pin_timed` 的脚本显式启用。

## 2. 新 Robot 类型与带时间戳接收器

### 2.1 当前继承关系

当前 `JZRobotPin` 继承 `JZRobotUDP`：

```text
JZRobotUDP
    -> UDP state receiver
    -> UDP command sender
    -> joint/gripper feature mapping
    -> OpenCV RTSPCamera

JZRobotPin(JZRobotUDP)
    -> armed gate
    -> first-action / step safety
    -> Pin-specific defaults
```

`src/lerobot/robots/jz_robot_pin/rtsp_camera.py` 当前只是对
`jz_robot_udp.rtsp_camera.RTSPCamera` 的重新导出。只修改这个包装文件不会改变运行时行为，因为
相机对象是在 `JZRobotUDP.__init__()` 中直接创建的。

### 2.2 是否应该新建 Robot

新增一个实验 Robot 类型是合理的，但不应复制整套 Pin 实现。推荐名称暂定为：

```text
jz_robot_pin_timed
```

它应继承 `JZRobotPin`，复用：

- UDP state/command 协议；
- joint 和 gripper 字段映射；
- armed 环境变量保护；
- 第一帧和连续动作安全检查；
- connect/disconnect 和命令发送逻辑；
- Pin 现有配置默认值。

新类型只负责：

- 使用 GStreamer appsink 或 PyAV 的带时间戳 RTSP receiver；
- 保存每帧 source PTS、接收墙钟、接收单调时间、frame ID 和 frame age；
- 保存 state packet 的 seq/stamp/接收时间；
- 将 Camera/State 对齐到明确的采样时刻；
- 输出重复帧、跳帧、stale 和跨模态时间差诊断。

### 2.3 推荐的复用方式

为避免在新 Robot 中复制 `JZRobotUDP.__init__()`，建议先在公共基类增加一个保持默认行为的
相机工厂钩子：

```text
JZRobotUDP._make_camera(camera_config)
    默认返回现有 OpenCV RTSPCamera

JZRobotPinTimed._make_camera(camera_config)
    返回新的 TimestampedRTSPCamera
```

这样公共 UDP Robot 的默认行为不变，新类型可以替换 receiver，同时继续复用全部状态、动作和
安全逻辑。若新 receiver 验证稳定，后续可以将它变成 `JZRobotPinConfig` 的可选 backend，最终
不一定需要长期保留两个 Robot 类型。

### 2.4 新 receiver 的实际接口

```text
connect()
disconnect()
read() -> ndarray                       # 保持现有 Robot API
read_timed() -> TimestampedFrame        # 新的诊断/对齐接口

TimestampedFrame:
    image
    decoder_pts_ns
    receive_wall_ns
    receive_monotonic_ns
    decoder_sequence
    reconnect_generation
```

`decoder_pts_ns` 是当前 RTSP 解码流自己的相对时间轴，不是 Unix 时间，也不能直接减去 state
packet 的 `stamp_ns`。`decoder_sequence` 只是 X86 解码器本地序号，不是相机源 frame ID。
当前 `receive_*` 在 decoder 交出 frame 后、像素转换前采样，并以
`timestamp_stage=decoder_output_before_pixel_conversion` 标记；它仍不是 RTSP 网络包到达或传感器
曝光时间。跨进程续录使用新的 `session_id`，observation sequence 只要求在同一 session 内递增。
如果 RTSP 发布端没有把 ROS `header.stamp`/frame ID 映射到 RTP PTS，需要 Orin 端增加映射或
旁路 metadata。仅依赖 X86 收包时间可以检查接收侧对齐、重复和 stale，但不能还原严格曝光时刻。

### 2.5 新 Robot 仍不能单独解决的部分

新增 receiver 只能解决相机时间可见性和 Camera/State 对齐。Action 来自独立 Teleoperator，
标准 `record_loop` 是先取 observation、再取 action。因此完整的 Camera/State/Action 时间对齐
还需要录制层保存 action packet 的 seq/stamp 和发送时间，不能只在 Robot 类内部完成。

CRF 18 也属于 Dataset 编码层，不应放进 Robot 类。

### 2.6 实施与验收顺序

1. 已实现 CRF 参数传递，新 timed 录制脚本显式使用 H.264 CRF 18。
2. 已新增 `jz_robot_pin_timed`，继承现有 Pin Robot，只替换相机接收与观测时序诊断。
3. 先运行 `x86/check_timed_observation.sh` 做 observation-only 验证；该流程不发送动作。
4. 使用同一个 RTSP 源比较旧/新 receiver 的帧数、重复率、延迟和重连行为。
5. 再录制三条短数据，检查视频帧数、Parquet 行数以及 Camera/State/Action sidecar。
6. 验证稳定后再决定：保留独立 Robot 类型，或者将 timed backend 合并为 `JZRobotPinConfig` 选项。

### 2.7 初始验收标准

- 30 FPS 视频帧数与对应数据行数一致。
- 每个 episode 记录源帧间隔、重复数、跳号数和重连次数。
- Camera/State 匹配误差 P95 有明确统计；初始目标可参考 camera 25 ms、state 10 ms，但应根据
  实际 RTSP 编码延迟重新标定。
- stale 数据不得静默作为新样本保存；超阈值样本应标记无效或终止 episode。
- 新 Robot 的 armed gate、动作安全检查和停止流程必须与现有 `JZRobotPin` 完全一致。
- 旧 Robot 类型保持可运行，作为现场回退路径。

## 3. 当前建议

- 新 timed 录制入口正式选择 CRF 18；旧 Pin 脚本保持原行为，不能把两者日志混看。
- 继续使用 RTSP/TCP 传输图像，不使用 RDP，也不直接通过千兆网传三路 ROS raw RGB。
- `jz_robot_pin_timed` 已继承并复用现有 Pin/UDP 实现，先作为实验验证入口。
- 长期目标不是维护两份重复 Robot，而是验证新的 timestamped receiver 后，将其作为 Pin 可选或
  默认相机 backend。
- 相机 receiver、Dataset CRF 和 Orin ROS/RTSP 发布端属于三个不同层级，不能只修改一个目录
  就宣称全部问题已经解决。
