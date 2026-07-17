# jz_robot_pin_timed 三条录制与时序检查

完整的录制中断保存语义、PASS/WARN/FAIL 判定、200 ms 采集容错与 100 ms 训练质量门、部分源精选合并
以及 2026-07-15 当前数据整理结果，统一记录在
[`DATASET_INTEGRITY_AND_CURATION.md`](DATASET_INTEGRITY_AND_CURATION.md)。后续智能体在删除、修复、
合并或训练任何 JZ timed 数据前，应先阅读该文档。

本目录用于连续录制 3 个 episode，并在录制结束后执行三类离线检查：

- `check_3_episodes.py`：复用 `jz_robot_pin` 的 18 维 action/state、episode、视频文件和跟随延迟检查，默认要求 `robot_type=jz_robot_pin_timed`。
- `check_timing.py`：检查 `video_encoding.codec=h264` 和 `video_encoding.crf=18`，并将数据 parquet 中的每个 `(episode_index, frame_index)` 与 `meta/timing/episode-XXXXXX.jsonl` 严格对应。
- `check_training_projection.py`：按 schema 字段名验证 raw18 到 model16 的映射、来源/方向语义、
  force 排除、全量数值行、gripper generation 推进以及 raw 数据未被修改。来源为 `unavailable`
  时只能产生显式 `AUDIT`，不能作为训练通过。

时序 sidecar 不会改变 raw18 `observation.state` 和 `action`。模型训练通过独立 schema view 投影为
16D，原始 parquet 仍保持 18D。timing sidecar 单独保存 session、状态包、target-action 包、实际
command 发送完成时间和三路相机时序。新 ZMQ schema 记录 sequence gap、Orin capture/JPEG timing、
X86 receive/decode timing、接收 FPS 和 source FPS；旧 RTSP episode 的 decoder PTS/reconnect schema
继续兼容。

只要 state 中出现 `source_timing`，检查器就按 v1 协议完整校验；正式三条包装器进一步要求每个
dataset frame 都携带有效对象。默认要求四路 source age 不超过 50 ms、接收 skew 不超过 20 ms，
并从每路 `recv_monotonic_ns + age_ms` 反推同一个 Orin snapshot，误差不超过 1 ns。X86 与 Orin 的
timed 修复必须一起部署；旧 Orin bridge 仍可用兼容模式读取，但不能通过正式三条数据检查。

## 连续录制并自动检查

先启动机器人端服务和 X86 joystick publisher，再运行：

```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot

JZ_ROBOT_PIN_ARMED=1 \
I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT=1 \
bash my_devs/jz_robot_pin_timed/data_check/record_and_check_3.sh
```

包装器固定 `NUM_EPISODES=3`、`RESUME=false`、三路直连 ZMQ 相机、`TIMING_SIDECAR=true` 和
H.264；默认 `VIDEO_CRF=18`。常用覆盖参数：

```bash
DATASET_NAME=jz_robot_pin_timed_check_001 \
EPISODE_TIME_S=10 \
RESET_TIME_S=5 \
RECORD_FPS=30 \
VIDEO_CRF=18 \
LEFT_GRIPPER_OBSERVATION_SOURCE=measured_opening \
RIGHT_GRIPPER_OBSERVATION_SOURCE=commanded_opening \
MAX_INITIAL_JOINT_DELTA_RAD=10.0 \
MAX_JOINT_STEP_RAD=10.0 \
MAX_TIMING_SOURCE_AGE_MS=50 \
MAX_TIMING_SOURCE_SKEW_MS=20 \
JZ_ROBOT_PIN_ARMED=1 \
I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT=1 \
bash my_devs/jz_robot_pin_timed/data_check/record_and_check_3.sh
```

上述来源值对应当前已审计部署；如果不显式设置，默认 `unavailable` 会被如实写入 schema sidecar，
projection checker 返回 `AUDIT` 而不是训练 `PASS`。

自动检查默认生成：

```text
<dataset_root>/data_check_report.json
<dataset_root>/timing_check_report.json
<dataset_root>/training_projection_report.json
```

Timed 三条流程的最佳 lag P95 默认上限为 `0.05 rad`，可通过 `MAX_LAG_P95_RAD` 显式覆盖；
MAE、18 维 schema、视频和 source timing 检查不受该参数影响。

当前 Orin timed bridge 联调基线为 commit `188f64a8ef7c08615a1f5b30b2b9539f1a264ca7`。双夹爪
status 订阅使用 BEST_EFFORT/VOLATILE/KEEP_LAST(depth=1)；它恢复状态源约 30 Hz，并保持 minimum
ratio `0.9`、source age `50 ms`、source skew `20 ms` 和四源推进要求。X86 检查器不会因该 QoS 修复
自动放宽任何阈值。

## 相机蓝偏编码诊断

`record_color_diagnostic.sh` 默认只录一条 10 秒、20 FPS episode，同时保留最终 MP4 和编码前 PNG：

```bash
DATASET_NAME="jz_robot_pin_timed_color_diag_$(date +%Y%m%d_%H%M%S)" \
EPISODE_TIME_S=10 \
RECORD_FPS=20 \
VIDEO_CRF=18 \
DISPLAY_DATA=true \
MAX_INITIAL_JOINT_DELTA_RAD=10.0 \
MAX_JOINT_STEP_RAD=10.0 \
JZ_ROBOT_PIN_ARMED=1 \
I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT=1 \
bash my_devs/jz_robot_pin_timed/data_check/record_color_diagnostic.sh
```

该入口用于实际运动下的颜色诊断，默认启用屏幕相机显示，并显式使用 `10 rad` 的 initial/step joint
guard，以免 joystick 当前目标与实测姿态的小偏差在首帧中止。双重 armed 确认仍然必需，底层 command
executor、关节检查实现和夹爪范围没有修改。屏幕显示需要运行终端具有有效的 `DISPLAY`；Meshcat 可视化
由 `start_pin_joystick.sh` 独立提供，不受该变量影响。

产物位置：

- `images/<camera>/episode-000000/frame-*.png`：Orin JPEG 经 X86 解码后、LeRobot 二次编码前的 PNG；
- `videos/<camera>/chunk-000/file-000.mp4`：最终数据集视频；
- `color_encoding_comparison.json`：5 个对应抽样帧的 RGB 均值、逐通道 MAE 和蓝色偏移变化。

如果 PNG 正常、MP4 发蓝，重点检查 X86 的 H.264/pixel-format 编码链；如果 PNG 已经发蓝，则蓝偏在
LeRobot 二次编码之前，应继续检查 Orin RealSense 采集、JPEG 编码或 X86 JPEG 解码。这里的 PNG
不是传感器 RAW，而是已通过 Orin JPEG 编码并在 X86 解码后的图像。

## 只检查已有数据

完整数据检查：

```bash
conda run --no-capture-output -n lerobot_flex \
  python my_devs/jz_robot_pin_timed/data_check/check_3_episodes.py \
  --dataset-root tests/outputs/<dataset_name>
```

CRF 和时序检查：

```bash
conda run --no-capture-output -n lerobot_flex \
  python my_devs/jz_robot_pin_timed/data_check/check_timing.py \
  --dataset-root tests/outputs/<dataset_name> \
  --expected-codec h264 \
  --expected-crf 18 \
  --require-source-timing \
  --max-source-age-ms 50 \
  --max-source-skew-ms 20
```

训练投影检查（纯离线，不改变 raw18）：

```bash
conda run --no-capture-output -n lerobot_flex \
  python my_devs/jz_robot_pin_timed/data_check/check_training_projection.py \
  --dataset-root tests/outputs/<dataset_name> \
  --manifest tests/outputs/<dataset_name>/meta/jz_pin_training_schema.json
```

这些命令除写入对应 JSON 报告外只读数据集，不连接机器人，也不会发送控制命令。检查 dry-run 数据时，需要给时序检查器补充 `--expected-command-mode dry_run`；非 UDP 本地测试还要补充 `--expected-command-transport local`。

## 时序检查判定

默认会失败的情况包括：

- `meta/info.json` 缺少 `video_encoding`，编码器不是 H.264，或 CRF 不是 18；
- 机器人类型不是 `jz_robot_pin_timed`；
- dataset frame 与 timing JSONL 缺失、多余、重复、乱序或 episode 内不连续；
- 缺少合法 `session_id`，或同一 session 内 observation sequence 重复/倒退；
- episode/frame 与 sidecar 文件名不一致，JSON 损坏或关键字段类型错误；
- 缺少三路相机时序、状态包时序或 target-action 包时序；
- 任一已有 `state.source_timing` 不符合 v1 协议、记录的 skew 不能从四路接收时间复算，或四路不能
  反推出同一个 snapshot；
- 正式三条流程的任一帧缺少有效的 Orin `state.source_timing` 对象，source age 超过 50 ms，或
  source skew 超过 20 ms；
- 同一 X86 session 内 state seq 倒退；seq 推进时四路 generation/receive monotonic time 未严格推进，
  或双臂非零 ROS header stamp 未严格推进；seq 复用时 packet stamp 或完整 source timing 发生变化；
- 缺少实际 command 时序，command 未关联到同一 observation，或 mode/transport/18 维字段数不符合本次录制配置；
- 正式录制帧的相机 age 超过 1000 ms，或相机与状态的本机接收时间偏差超过 100 ms；
- 默认录制出现 `hold_current` 动作来源。

episode reset 使用 18 维 state-only control observation，不读取或保存相机，因此 reset 期间不执行上述
相机 age/skew 检查。state freshness、`source_timing`、target-action stale 和 command 安全门仍然生效。
reset 因这些控制链路错误退出时，recorder 会先尝试保存已经录完的 episode，再保留原异常退出；三条包装器
仍返回失败，不会把不足三条的 partial dataset 误报为通过。

相机复用默认只统计并告警，因为两个独立 30 Hz 循环可能偶发读取同一解码帧。需要把它变成硬限制时，可传入例如：

```bash
--max-reuse-fraction 0.05
```

state 包复用比例也会按同一 X86 session 内的相邻 frame 统计。新 Timed 录制默认在采集前等待本机
state revision 推进，因此持续复用会在写入 frame/发送 action 之前失败；checker 仍保留复用统计，
用于审计旧数据和显式关闭该门禁的兼容流程。离线检查时也可以显式启用比例限制，例如：

```bash
--max-state-reuse-fraction 0.05
```

现场阈值也可以显式覆盖：

```bash
--max-camera-age-ms 1000 \
--max-camera-state-skew-ms 100
```

报告按 session 检查 observation sequence 和 state sequence，并给出 state reuse fraction、四路 source
age、source skew、snapshot 反推偏差。双臂 `header_stamp_ns=0` 符合协议但会统计比例并告警，不直接
判失败。相机报告包含 age、带符号的 state-to-decoder-output delta、绝对 skew、decoder sequence、
复用比例、缺失 PTS 和重连 generation。decoder PTS 只在同一个 episode/reconnect generation 内统计
间隔，不会与本机 wall/monotonic 时间或 state stamp 做跨时钟域比较。同时报告 target-action 包和
command 的 sequence、gap、重复、reset，以及 action receive 到 command send、state receive 到
command send 的本机单调时钟延迟。
