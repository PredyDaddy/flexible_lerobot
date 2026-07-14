# JZ Robot Pin Timed 三终端录制流程

本流程用于正式录制 3 条 timed 数据。开始前确认现场有人守急停，机器人工作空间已清空。

不要同时运行 `start_teleop.sh`、`start_pin_control.sh`、probe 或另一个 recorder：它们会争用 X86
的 state UDP `39010` 或 command 链路。

## 终端 1：Orin state bridge + executor

在 Orin（`192.168.1.81`）执行：

```bash
cd /home/data/test/workspace/flexible_lerobot

JZ_UDP_EXECUTOR_ARMED=1 \
EXECUTION=armed \
bash my_devs/jz_robot_pin_timed/edge/start_pin_replay.sh
```

等待并确认出现：

```text
PHASE3 COMMAND EXECUTOR ARMED
replay services ready.
```

Timed bridge 默认要求 30 Hz；启动失败时不要继续启动 X86 recorder，先检查 Orin 日志中的 state
发送率、source freshness 和 source skew。

## 终端 2：X86 VR、IK 和整机 Meshcat

在 X86 执行：

```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot

bash my_devs/jz_robot_pin_timed/x86/start_pin_joystick.sh
```

该终端使用 `light_tp` 环境，持续运行。它负责：

- 接收 VR UDP 输入；
- 运行 Pink/Pinocchio IK；
- 向本机 `127.0.0.1:39030` 发布 target action；
- 启动整机 Meshcat。

办公机通过 SSH 访问时，在浏览器打开 X86 实际 IP，例如：

```text
http://10.1.42.3:7000/static/
```

启动日志应显示整机模型和约 30 Hz 的 Meshcat 显示目标。保持该终端在全部三条录制和两个 reset
阶段持续运行。

## 终端 3：X86 recorder

在另一个 X86 终端执行：

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

20fps
```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot

DATASET_NAME="jz_robot_pin_timed_real_$(date +%Y%m%d_%H%M%S)" \
EPISODE_TIME_S=10 \
RESET_TIME_S=5 \
RECORD_FPS=20 \
VIDEO_CRF=18 \
JZ_ROBOT_PIN_ARMED=1 \
I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT=1 \
bash my_devs/jz_robot_pin_timed/data_check/record_and_check_3.sh
```

这个终端使用 `lerobot_flex`。recorder 本身负责读取 state、读取 target action、发送 command 和保存
数据；它不需要、也不允许再配套启动 X86 control。

默认行为：

- 录制 3 条，每条 10 秒，30 FPS；
- 两条之间继续执行固定时长的 `RESET_TIME_S` 原有流程；
- X86 不连接 `39040`，不调用 `/robot1/choreographer/execute`，也不自动触发底层编排；
- action/state 为 18 维；
- 三路视频使用 H.264、CRF 18、yuv420p、GOP 2；
- target action stale 默认立即终止录制，不保存 hold-current 数据；
- 首帧要求完整 `source_timing v1`；
- 录制后自动运行 `data_check_report.json` 与 `timing_check_report.json`。

`jz_pin_reset_control/39040` 自动 reset 已回退。若旧启动命令仍设置
`AUTOMATIC_EPISODE_RESET=true`，recorder 会在启动阶段明确拒绝，不会静默忽略。

三条包装器为现场联调将 initial/step joint delta guard 设为 `10.0 rad`。这是已知的高风险现场
取舍，不是 Robot 全局默认；必须依赖急停和现场人员。

## 成功判据

终端 3 最后应出现：

```text
[timed/record_and_check_3] PASS data_report=...
[timed/record_and_check_3] PASS timing_report=...
```

保存终端输出中打印的 `dataset_root`。报告位于：

```text
tests/outputs/<dataset_name>/data_check_report.json
tests/outputs/<dataset_name>/timing_check_report.json
```

若 recorder 报 target action 首包超时，先确认终端 2 仍在运行且正在向 `39030` 发布，不要直接重启
Orin executor。

## 停止顺序

1. 等待 recorder 自然结束；异常时只在 recorder 终端按 `Ctrl+C`。
2. 在 joystick/Meshcat 终端按 `Ctrl+C`。
3. 在 Orin 执行：

```bash
cd /home/data/test/workspace/flexible_lerobot
bash my_devs/jz_robot_pin_timed/edge/stop_pin_replay.sh
```

只有本次确实单独运行过 `start_teleop.sh` 时，才额外执行 timed 的 `stop_teleop.sh`；上面的正式录制
流程不需要它。
