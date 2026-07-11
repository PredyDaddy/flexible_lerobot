# jz_robot_pin_timed 实现与上线边界

本文记录当前已经落地的实现、sidecar 语义、仍需外部条件才能解决的问题，以及回退方式。

## 目标与继承关系

```text
JZRobotUDP
    -> JZRobotPin
        -> JZRobotPinTimed
```

`jz_robot_pin_timed` 不复制控制协议和 18 维字段，继续复用：

- UDP state receiver/cache 和 sender IP 校验；
- UDP/local command 发送及 packet 协议；
- 18 维双臂、双夹爪 state/action key；
- armed 环境变量、initial delta、step delta 和夹爪限制；
- target-action teleoperator、录制和回放主循环。

新增行为通过小型 hook 接入：

- `JZRobotUDP._make_camera()`：timed 子类替换为 PyAV receiver；
- `JZRobotUDP._read_camera()`：按 state 的本机接收时刻选择最近缓存帧；
- `JZRobotUDP._after_observation()`：生成 state/camera timing；
- `JZRobotUDP._after_action_sent()`：记录实际 command packet 和 send completion；
- `lerobot_record.record_loop()`：可选调用 Robot 的 `save_frame_timing()`。

旧 `jz_robot_pin` 不实现这些 hook 的 timed 行为，公开 observation/action schema 保持不变。

## 已实现的数据路径

每个成功保存的 dataset frame 对应一条：

```text
meta/timing/episode-XXXXXX.jsonl
```

一条记录包含：

```text
session_id, episode_index, frame_index, observation_sequence
state:
  packet_seq, packet_stamp_ns, receive_wall_ns, receive_monotonic_ns
cameras.<camera>:
  timestamp_stage=decoder_output_before_pixel_conversion
  decoder_pts_ns, receive_wall_ns, receive_monotonic_ns
  decoder_sequence, reconnect_generation, age_ms
  reused_by_observation_loop, state_receive_delta_ms, state_receive_skew_ms
action:
  source, packet_seq, packet_stamp_ns, receive_wall_ns, receive_monotonic_ns, age_ms
command:
  observation_sequence, packet_seq, packet_stamp_ns, mode, transport
  send_completed_wall_ns, send_completed_monotonic_ns, action_key_count
```

约束：

- sidecar 的 `(episode_index, frame_index)` 必须与 Parquet frame 一一对应；
- 同一 `session_id` 内 observation sequence 严格递增；跨进程 resume 使用新的 session；
- `command.observation_sequence` 必须等于本行 observation；
- command 记录的是安全处理后真正传给 `send_action()` 的 18 维动作；
- episode 重录从 frame 0 开始时覆盖该 episode 的旧 sidecar，避免重复行；
- H.264 codec 和 CRF 写入 `meta/info.json.video_encoding`，默认正式流程使用 CRF 18。
- resume 同时校验 codec/CRF；缺少该元数据的旧数据只允许按历史默认 CRF 30 续录，并写入
  `legacy_assumed=true`，不能静默混入 CRF 18。

## 时间语义边界

已经能够诊断：

- observation loop 是否复用或跳过 decoder sequence；
- RTSP 是否缺 PTS、重连或出现 generation 内 PTS 非递增/前向跳变；
- state、camera decoder output、target action、command 在 x86 本机时间轴上的相对关系；
- target action 到 command send、state receive 到 command send 的本机延迟；
- dataset frame、视频和 sidecar 是否缺失或错位。

不能从当前 sidecar 证明：

- `decoder_pts_ns` 不是 ROS header stamp，也不保证是传感器曝光时刻；
- camera `receive_*` 在 decoder 交出 frame 后、像素转换前采样，不是 RTSP 网络包到达时刻；
- 三路相机 PTS 不一定来自同一硬件时钟，跨相机不能直接当作硬同步；
- `packet_stamp_ns` 与 x86 wall/monotonic clock 未经 PTP/chrony 校准时不能直接相减；
- `send_completed_*` 表示 x86 本地 send 返回，不是 Orin 收包、ROS publish 或电机执行确认；
- 网络单向延迟、相机曝光延迟和机器人实际闭环响应仍需源时间戳、时钟同步和 executor ack。

因此当前实现解决的是“可观测、可核对”，不是传感器和控制器的硬件同步。

## 分阶段上线

1. 离线门禁

   - 运行 timed Robot 单测、record timing hook 单测和 Shell/Python 静态检查；
   - 不连接机器人。

2. 只读现场探针

   - Orin 只启动 `edge/start_pin_state.sh`；
   - x86 单独运行 `check_timed_observation.sh`；
   - 检查三路 PTS、reuse、skew、age 和 reconnect，不启动 recorder/control。
   - timed edge 默认 `STATE_HZ=30`；旧 Pin 默认仍为 20 Hz。

3. 本地 dry-run 短采集

   - 使用 `EXECUTION=dry_run SEND_ACTION_TRANSPORT=local`；
   - 用通用 `record.sh` 录制三条短数据，再运行 `data_check/check_timing.py`；
   - 检查器显式传 `--expected-command-mode dry_run --expected-command-transport local`；
   - 确认每帧都有 matching command，且 command mode/transport 符合本次配置。

4. armed 小样本

   - 现场急停和人员到位后，仅录 3 个短 episode；
   - 通过 18 维、视频、sidecar、skew/reuse 和 command timing 检查后再扩大采集。

5. 批量采集

   - 固定代码版本、CRF、相机参数、网络 IP 和阈值；
   - 每批数据保留 `data_check_report.json` 与 `timing_check_report.json`。

任何阶段都不能同时运行 probe、recorder 和 control，因为三者绑定同一个 x86 UDP state port。
正式录制只运行 joystick publisher + recorder，不能再启动 `start_teleop.sh`。

## 验收门槛

- `robot_type == jz_robot_pin_timed`；
- action/state 均为 18 维，episode/video/frame 数符合配置；
- `video_encoding.codec == h264` 且 `crf == 18`；
- Parquet 与 sidecar frame key 完全一致；
- state/action/command sequence 和字段类型通过检查；
- resume 数据按 `session_id` 分段校验，不把合法的新进程序号重置判为损坏；
- command 与 observation 同序，mode/transport 符合本次执行配置；
- camera age/skew 在现场确认的阈值内，reuse 比例被记录并评估；
- 不存在静默 hold-current，除非某次实验明确允许并单独标注。

## 回退到 jz_robot_pin

Orin edge transport 是共享的，无需修改机器人端协议。回退步骤：

1. 停止 timed recorder/control/joystick；
2. 使用 `my_devs/jz_robot_pin/` 的 x86 脚本；
3. 新数据使用新的 dataset root，不能向 timed dataset 继续追加旧 Robot episode；
4. 如需保留画质设置，在旧流程中也显式使用相同 CRF；
5. timed 失败数据保留 sidecar 和报告用于定位，不用重新编码覆盖原证据。

回退后会失去 PyAV timing、最近 state 帧选择和 command sidecar，但 18 维动作/状态及现有 Orin
UDP bridge/executor 不变。这样 timed receiver 的问题不会迫使控制协议一起回滚。

当前 timed sidecar 要求 target-action packet timing，因此这个入口用于 VR/teleop 采集；直接用
policy 录制会因缺少 `action_timing` 明确失败，不能视为已支持的 policy 采集模式。

另一个已知部署前置条件是共享 joystick 仍依赖 `my_devs/my_var_tp`。该目录当前被 Git 忽略，
本机存在所以能运行，但干净仓库必须先按现场版本部署它；本次没有把这项后续工作混入 timed
receiver 改动。
