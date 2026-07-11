# jz_robot_pin_timed 快速入口

`jz_robot_pin_timed` 继承 `jz_robot_pin` 的 18 维 state/action、UDP command 和安全检查，只把三路
RTSP 接收器替换为保留 PTS、decoder 输出时刻和 sequence 的 PyAV 接收器。正式录制还会为每个
dataset frame 写一条 timing sidecar，并记录这一轮实际生成并发送完成的 command packet。

## 目录

| 路径 | 用途 |
|---|---|
| `record.sh` | x86 正式录制，默认 H.264 CRF 18 |
| `replay.sh` | x86 单 episode 回放，默认 dry-run |
| `x86/start_pin_joystick.sh` | 复用现有 VR/IK/Meshcat 发布端 |
| `x86/start_pin_control.sh` | timed Robot 遥操控制循环，不开相机 |
| `x86/start_pin_teleop.sh` | 同时启动 control 和 joystick |
| `x86/stop_pin_teleop.sh` | 只停止 timed PID/命令指纹匹配的进程 |
| `x86/probe_timed_observation.sh` | 只读 state/RTSP 时序探针 |
| `check_timed_observation.sh` | 只读探针的兼容别名 |
| `data_check/record_and_check_3.sh` | 连录三条并检查数据、CRF 和 timing sidecar |
| `recv_vr_udp.py` | 复用原 Pin 的 VR UDP 包解析诊断 |
| `edge/` | 复用 Orin 的 state bridge/executor |
| `video_check/` | 复用只读 RTSP/CRF 对照工具 |

## 环境

- x86 LeRobot、录制、回放、时序探针：`lerobot_flex`
- x86 VR/IK/Meshcat：`light_tp`
- Orin bridge/executor：沿用机器人上的 `lerobot`

VR/IK/Meshcat 包装器继续复用本机 `my_devs/my_var_tp`。该目录目前不在 Git 中，干净部署时必须
单独提供与现场一致的版本；state/RTSP 只读探针不依赖它。

## 正式录制最短流程

Orin：

```bash
cd /home/data/test/workspace/flexible_lerobot

JZ_UDP_EXECUTOR_ARMED=1 \
bash my_devs/jz_robot_pin_timed/edge/start_pin_replay.sh
```

x86 终端 1，持续运行 VR/可视化发布：

```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot
bash my_devs/jz_robot_pin_timed/x86/start_pin_joystick.sh
```

录制流程只启动这个 joystick 发布端和下面的 recorder，**不要启动 `start_teleop.sh`**。后者用于
不录制时的独立遥操，会额外启动 control 并抢占同一个 state/command 链路。

办公机通过 SSH 使用 x86 时，Meshcat 打开 `http://<X86_IP>:7000/static/`；例如 x86 地址为
`10.1.42.3` 时使用 `http://10.1.42.3:7000/static/`。

x86 终端 2，新建并录制三条：

```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot

DATASET_NAME="jz_robot_pin_timed_real_$(date +%Y%m%d_%H%M%S)" \
EPISODE_TIME_S=10 \
RESET_TIME_S=5 \
RECORD_FPS=30 \
VIDEO_CRF=18 \
JZ_ROBOT_PIN_ARMED=1 \
I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT=1 \
bash my_devs/jz_robot_pin_timed/data_check/record_and_check_3.sh
```

该包装器固定三条新数据、`RESUME=false`、CRF 18，并在录制后生成 `data_check_report.json` 和
`timing_check_report.json`。它沿用当前已接受的现场流程，默认把 initial/step joint delta guard
都设为 `10.0 rad`；通用 `record.sh` 仍保留保守的 `0.02 rad` 默认值。手动使用 `record.sh`
新建数据时必须设置 `RESUME=false`；只有继续写同一个已存在的数据集时才使用 `RESUME=true`，
且 codec/CRF 必须与 `meta/info.json` 一致；每个续录进程会生成新的 timing `session_id`。
正式三条检查要求 Orin timed bridge 在每个 state packet 中提供完整且合法的 `source_timing v1`；
recorder 会在第一条 command 之前校验它，并在录制后复核四源 50 ms age、20 ms skew、generation、
接收时间和非零关节 header 的推进关系。因此 Orin 与 X86 修复必须成对部署，旧 bridge 不能通过
这条面向训练数据的正式检查。

时序记录位于：

```text
tests/outputs/<dataset>/meta/timing/episode-000000.jsonl
```

每行对应同 episode、同 `frame_index` 的 Parquet 行和视频帧，包含 session、state packet seq/stamp、
三路 camera PTS/decoder 输出时刻/sequence、frame age、state skew、target action timing，以及
实际 command 的 seq/stamp/mode/transport 和 send-completion wall/monotonic 时间。

完整操作和诊断说明见 [整体使用记录.md](整体使用记录.md)，回放见 [回放使用记录.md](回放使用记录.md)。

## 只读探针

探针不会调用 `send_action()`，配置也固定为 `local + dry_run`：

```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot
bash my_devs/jz_robot_pin_timed/probe_timed_observation.sh
```

probe、recorder 和 control 都会绑定 x86 的 UDP `39010`，三者不能同时运行。timed 相机会从
buffer 中选择 decoder 输出时刻最接近 state 接收时刻的帧。camera `receive_*` 在 decoder 交出
frame 后、像素转换前采样，不是网络包到达或曝光时刻。探针之前应停止
所有 x86 recorder/control，只在 Orin 启动 `edge/start_pin_state.sh` 即可。

只检查 VR 设备是否向 x86 发包，不启动 IK、Robot 或 command 链路：

```bash
conda run --no-capture-output -n lerobot_flex \
  python my_devs/jz_robot_pin_timed/recv_vr_udp.py \
  --host 10.1.42.3 \
  --port 8080
```

## 停止

x86 timed teleop：

```bash
bash my_devs/jz_robot_pin_timed/stop_teleop.sh
```

Orin bridge/executor：

```bash
bash my_devs/jz_robot_pin_timed/edge/stop_pin_replay.sh
```

Orin transport 与原 `jz_robot_pin` 共用服务，不能把两套 edge 服务同时启动。
