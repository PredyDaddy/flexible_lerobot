# jz_robot_pin 三条连续录制检查

本目录提供一个安全包装命令，用同一个 `lerobot-record` 进程连续录制 3 个 episode，录制成功后自动检查三条数据。

开始前需要：

- 机器人/边缘端服务已由操作者显式启动。
- `x86/start_pin_joystick.sh` 已启动，并在录制和 episode reset 期间持续发布 target action。
- 真实 armed 录制需要现有的两层确认环境变量。

pin joystick 包装器默认使用 `publish_mode=always`，持续发布当前保持目标；默认运行 24 小时，
因此 recorder 不再依赖启动后 5 秒内人工按 Grip。Grip 仍负责绑定并移动遥操目标。
Meshcat 默认以 30 Hz 显示整机模型，target action 以 90 Hz 发布。
启动日志应显示 `model=whole-robot`、`visual meshes loaded: 24`，周期日志的 `meshcat_hz` 应约为 30。

运行：

```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot

JZ_ROBOT_PIN_ARMED=1 \
I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT=1 \
bash my_devs/jz_robot_pin/data_check/record_and_check_3.sh
```

常用覆盖参数：

```bash
DATASET_NAME=jz_robot_pin_check_001 \
EPISODE_TIME_S=10 \
RESET_TIME_S=5 \
RECORD_FPS=30 \
MAX_INITIAL_JOINT_DELTA_RAD=10.0 \
MAX_JOINT_STEP_RAD=10.0 \
JZ_ROBOT_PIN_ARMED=1 \
I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT=1 \
bash my_devs/jz_robot_pin/data_check/record_and_check_3.sh
```

包装器固定：

- `NUM_EPISODES=3`
- `RESUME=false`，避免新旧 action 语义混合
- 18 维 action/state（14 个关节 + 4 个夹爪字段）
- 三路 RTSP 视频
- 现场三条录制的第一帧和连续 action 关节差值上限默认均为 `10 rad`，对正常关节范围等效于放开
- armed 确认、状态新鲜度、发送端 IP/序列和数值有限性检查仍然保留
- 录制和自动检查使用同一组 `MAX_INITIAL_JOINT_DELTA_RAD` / `MAX_JOINT_STEP_RAD`；需要更严格时可显式覆盖
- 录制开始前等待 target action；录制中 target action stale 会令本次命令失败

只检查已经存在的数据：

```bash
conda run --no-capture-output -n lerobot_flex \
  python my_devs/jz_robot_pin/data_check/check_3_episodes.py \
  --dataset-root tests/outputs/<dataset_name>
```

检查报告默认写入：

```text
<dataset_root>/data_check_report.json
```

检查器不会要求 `action_t == observation.state_(t+1)`。它会在未来 1～6 帧中寻找最佳关节跟随延迟，并检查 MAE/P95、第一帧差值、连续 action 步长、18 维字段、episode 完整性和三路视频文件。
