# jz_robot_pin_timed 三条录制与时序检查

本目录用于连续录制 3 个 episode，并在录制结束后执行两类离线检查：

- `check_3_episodes.py`：复用 `jz_robot_pin` 的 18 维 action/state、episode、视频文件和跟随延迟检查，默认要求 `robot_type=jz_robot_pin_timed`。
- `check_timing.py`：检查 `video_encoding.codec=h264` 和 `video_encoding.crf=18`，并将数据 parquet 中的每个 `(episode_index, frame_index)` 与 `meta/timing/episode-XXXXXX.jsonl` 严格对应。

时序 sidecar 不会改变训练使用的 18 维 `observation.state` 和 `action`。它单独保存 session、状态包、target-action 包、实际 command 发送完成时间和三路相机 decoder 输出时间，用于发现漏帧元数据、相机陈旧、相机/状态偏差、控制链路延迟、帧复用或 RTSP 重连。

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

包装器固定 `NUM_EPISODES=3`、`RESUME=false`、三路 RTSP、`TIMING_SIDECAR=true` 和 H.264；默认 `VIDEO_CRF=18`。常用覆盖参数：

```bash
DATASET_NAME=jz_robot_pin_timed_check_001 \
EPISODE_TIME_S=10 \
RESET_TIME_S=5 \
RECORD_FPS=30 \
VIDEO_CRF=18 \
MAX_INITIAL_JOINT_DELTA_RAD=10.0 \
MAX_JOINT_STEP_RAD=10.0 \
MAX_TIMING_SOURCE_AGE_MS=50 \
MAX_TIMING_SOURCE_SKEW_MS=20 \
JZ_ROBOT_PIN_ARMED=1 \
I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT=1 \
bash my_devs/jz_robot_pin_timed/data_check/record_and_check_3.sh
```

自动检查默认生成：

```text
<dataset_root>/data_check_report.json
<dataset_root>/timing_check_report.json
```

Timed 三条流程的最佳 lag P95 默认上限为 `0.05 rad`，可通过 `MAX_LAG_P95_RAD` 显式覆盖；
MAE、18 维 schema、视频和 source timing 检查不受该参数影响。

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

这两个命令除写入对应 JSON 报告外只读数据集，不连接机器人，也不会发送控制命令。检查 dry-run 数据时，需要给时序检查器补充 `--expected-command-mode dry_run`；非 UDP 本地测试还要补充 `--expected-command-transport local`。

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
- 相机 age 超过 1000 ms，或相机与状态的本机接收时间偏差超过 100 ms；
- 默认录制出现 `hold_current` 动作来源。

相机复用默认只统计并告警，因为两个独立 30 Hz 循环可能偶发读取同一解码帧。需要把它变成硬限制时，可传入例如：

```bash
--max-reuse-fraction 0.05
```

state 包复用比例也会按同一 X86 session 内的相邻 frame 统计。正式三条包装器暂不设置硬阈值；离线
检查时可以按现场结果显式启用，例如：

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
