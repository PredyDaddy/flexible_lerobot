# jz_robot_pin_timed 三条数据回放流程

本文只记录命令，不会自动启动机器人。目标数据集：

```text
/home/luzhuang/cqy/aaa/flexible_lerobot/tests/outputs/jz_robot_pin_timed_real_20260711_190502
```

数据集共有 episode 0、1、2。每次只回放一条，不要并发启动 recorder、control 或另一个 replay。

## 1. 先做 dry-run

### Orin 终端

```bash
cd /home/data/test/workspace/flexible_lerobot

env -u JZ_UDP_EXECUTOR_ARMED \
EXECUTION=dry_run \
bash my_devs/jz_robot_pin_timed/edge/start_pin_replay.sh
```

等待 `replay services ready.`，并确认 executor 日志包含 `DRY-RUN` 和
`NOT publishing ROS command topics`。

### X86 终端

```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot

DATASET_NAME=jz_robot_pin_timed_real_20260711_190502 \
DATASET_ROOT=/home/luzhuang/cqy/aaa/flexible_lerobot/tests/outputs/jz_robot_pin_timed_real_20260711_190502 \
DATASET_REPO_ID=local/jz_robot_pin_timed_real_20260711_190502 \
EPISODE=0 \
EXECUTION=dry_run \
REPLAY_FPS=30 \
bash my_devs/jz_robot_pin_timed/x86/start_pin_replay.sh
```

依次把 `EPISODE` 改成 `0`、`1`、`2`，确认三条均能完整读取。dry-run 完成后在 X86
前台按 `Ctrl+C`，再在 Orin 停止服务：

```bash
cd /home/data/test/workspace/flexible_lerobot
bash my_devs/jz_robot_pin_timed/edge/stop_pin_replay.sh
```

## 2. 真实 armed 回放

armed 回放会让机器人运动。开始前必须由现场人员确认急停、工作空间和机器人起始姿态。

### Orin 终端

```bash
cd /home/data/test/workspace/flexible_lerobot

JZ_UDP_EXECUTOR_ARMED=1 \
EXECUTION=armed \
bash my_devs/jz_robot_pin_timed/edge/start_pin_replay.sh
```

必须看到 `PHASE3 COMMAND EXECUTOR ARMED` 和 `replay services ready.`。

### X86 终端

先回放 episode 0：

```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot

DATASET_NAME=jz_robot_pin_timed_real_20260711_190502 \
DATASET_ROOT=/home/luzhuang/cqy/aaa/flexible_lerobot/tests/outputs/jz_robot_pin_timed_real_20260711_190502 \
DATASET_REPO_ID=local/jz_robot_pin_timed_real_20260711_190502 \
EPISODE=0 \
EXECUTION=armed \
REPLAY_FPS=30 \
JZ_ROBOT_PIN_ARMED=1 \
I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT=1 \
bash my_devs/jz_robot_pin_timed/x86/start_pin_replay.sh
```

episode 0 完成并确认机器人状态正常后，再分别把 `EPISODE=0` 改成 `EPISODE=1`、
`EPISODE=2`。每条回放前都要让机器人回到该 episode 第一帧附近；脚本默认保留
`0.02 rad` 的 initial/step joint delta 检查，不建议为了绕过失败直接放大阈值。

## 3. 停止顺序

1. X86 replay 前台自然结束；异常时按 `Ctrl+C`。
2. Orin 执行：

```bash
cd /home/data/test/workspace/flexible_lerobot
bash my_devs/jz_robot_pin_timed/edge/stop_pin_replay.sh
```

回放不需要启动 joystick 或 Meshcat。脚本会关闭 RTSP 相机读取，只使用数据集 action 和实时
robot state。
