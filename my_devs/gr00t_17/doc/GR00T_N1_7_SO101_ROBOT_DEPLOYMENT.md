# GR00T N1.7 SO101 真机部署与 Smoke Runbook

## 1. 当前状态

本文对应正式模型：

```text
my_devs/gr00t_17/outputs/formal/so101_n17_b2_e10_20260711/train/checkpoint-63600
```

截至 2026-07-12：

1. 正式训练已完成并通过 `../reports/formal_training_validation.json` 验收；
2. checkpoint 已在 GPU 上由 N1.7 policy server 完整加载；
3. SO101 串口、已有校准、top/wrist 两路真实相机和当前关节状态均已验证；
4. 真实机器人上的两轮 policy prediction Smoke 已通过，未发送电机命令；
5. 带动作 Smoke 脚本已经实现，但必须在操作员确认现场安全后才允许执行；
6. 正式 120 秒入口已经实现，缺少双重动作确认时会在加载模型前退出。
7. 上机开发后的不可变输入复核已通过，原数据、PI checkpoint、reference 和转换数据 hash 均未变化。

加固后的最新无动作 Smoke 证据位于：

```text
my_devs/gr00t_17/outputs/inference/smoke/so101_final_no_actuation_20260712_0958/
```

## 2. 部署架构和环境隔离

推理由两个本地进程组成：

```text
SO101 + top/wrist cameras
        |
        | LeRobot device API
        v
lerobot_flex robot client
        |
        | ZeroMQ, tcp://127.0.0.1:<port>
        v
project-local GR00T N1.7 policy server
        |
        v
checkpoint-63600 on CUDA GPU
```

- robot client 必须由 conda 环境 `lerobot_flex` 启动；
- N1.7 server 使用 `my_devs/gr00t_17/env/gr00t_n17/`；
- pyzmq 安装在 `my_devs/gr00t_17/tools/robot_client_deps/`，不写入 conda 环境；
- server 只接受 `127.0.0.1`/`localhost`，不暴露到局域网；
- cache、日志、快照、报告和推理输出全部留在 `my_devs/gr00t_17/`；
- 原始数据、PI 权重、原校准文件和 reference 均不会被修改。

## 3. 模型输入输出契约

robot client 生成的单样本输入如下：

| 模态 | key | shape | dtype/单位 |
| --- | --- | --- | --- |
| top camera | `video.top` | `(1, 1, 480, 640, 3)` | RGB `uint8` |
| wrist camera | `video.wrist` | `(1, 1, 480, 640, 3)` | RGB `uint8` |
| arm state | `state.single_arm` | `(1, 1, 5)` | SO101 range-normalized position |
| gripper state | `state.gripper` | `(1, 1, 1)` | SO101 `[0, 100]` position |
| language | `annotation.human.task_description` | `(1, 1)` | exact training task text |

关节顺序固定为：

```text
shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
```

policy response 必须同时满足：

```text
single_arm: (1, 16, 5)
gripper:    (1, 16, 1)
all values: finite
```

训练时前五维 arm 使用 relative action，gripper 使用 absolute action。N1.7 保存的 processor 会先完成
反归一化，再利用本次请求中的当前 state 把 relative arm action 恢复成绝对关节目标。因此 robot client
收到的 16×6 chunk 已是可供安全层检查的绝对目标，不能再次手工累加 relative action。

## 4. 本次正式训练配置

本次正式 run 的真实参数不是早期 micro-batch-1 baseline，而是：

| 项目 | 实际值 |
| --- | ---: |
| per-device micro batch | 2 |
| gradient accumulation | 4 |
| effective batch | 8 |
| valid horizon-16 windows | 50,880 |
| optimizer steps | 63,600 |
| nominal sample epochs | 10.0 |
| train runtime | 32,231.7025 s |
| average train loss | 0.0691617722 |
| final logged loss | 0.0358 |

参数训练范围为：

| 模块 | 是否训练 |
| --- | --- |
| visual / ViT | 冻结 |
| language model / LM | 冻结 |
| multimodal projector | 训练 |
| diffusion model / DiT | 训练 |
| VLLN/self-attention action module | 训练 |

## 5. 动作安全层

### 5.1 校准和设备契约

使用的项目内校准文件为：

```text
my_devs/gr00t_17/configs/calibration/so_follower/hfy_follower.json
```

它与现有 `hfy_follower` 校准 JSON 语义逐项相同，只是格式化不同。client 使用
`robot.connect(calibrate=False)`，校准不匹配时直接失败，不进入交互式重标定，也不覆盖原校准。

连接前会验证 robot action feature 的完整顺序和 observation keys。即使只连接了电机总线而某个相机随后
失败，清理逻辑也会单独断开总线并显式禁用扭矩。

### 5.2 限幅

早期入口把 checkpoint `statistics.json` 的训练 action min/max 当成推理硬边界。真机验证表明这不适合
已经从 relative action 反解出的绝对目标：当当前 state 位于训练 action 极值之外时，目标会被长期钉在
同一个统计边界上。正式默认现改为 SO101 物理边界：

```text
lower = [-100, -100, -100, -100, -100, 0]
upper = [ 100,  100,  100,  100,  100, 100]
```

训练统计仍用于 processor 的归一化/反归一化及越界监控，但不再对解码后的绝对目标做第二次统计截断。

每条命令还经过两级相对变化限制：

1. client 的 `max_command_delta` 针对上一条安全目标限幅；
2. LeRobot SOFollower 的 `max_relative_target` 会读取电机当前位置再次限幅。

Smoke 动作阶段固定为 1 秒、5 Hz、horizon 1、每关节每次最多 0.25。正式真机验证已经确认单步 1.0
会导致持续限幅，而物理边界 + 单步 3.0 能产生正确方向的轨迹，因此正式默认使用 3.0。

### 5.3 动作授权

只有以下两个条件同时出现才可发送动作：

```text
--enable-actuation
--confirm-actuation SO101_GR00T_N17
```

Smoke wrapper 还要求对应的两个环境变量。所有时间、频率、推理时限和动作限幅必须为有限正数，动作运行
最长 600 秒。动作模式下 ZMQ request timeout 和推理 watchdog 默认均为 2 秒，且不允许 request timeout
大于 watchdog，避免 server 卡住时长时间保持扭矩等待。报告路径不允许复用，避免覆盖先前证据。

## 6. 上机前现场检查

每次动作 Smoke 或正式运行前，人工逐项确认：

1. SO101 底座固定，机械臂不会带动底座翻倒或滑动；
2. 电缆在完整运动范围内不会缠绕、拉扯或进入夹爪；
3. 人手、脸部和无关物体离开机械臂运动范围；
4. 操作员能立即断开机器人电源或串口控制；
5. top 是训练时的俯视相机，wrist 是训练时的腕部相机，没有交换；
6. top/wrist 的视角、方向、曝光和训练数据一致；
7. 橡皮、小盒和杯子按对应训练任务的初始分布摆放；
8. 机械臂初始姿态位于训练数据覆盖范围，不在硬限位上受力；
9. GPU 没有新的大显存任务，串口和相机没有被其他进程占用；
10. 先查看本次 Smoke 保存的两张 snapshot，再决定是否授权电机动作。

设备只读检查示例：

```bash
ls -l /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00
ls -l /dev/video4 /dev/video6
nvidia-smi
```

## 7. 安装 robot client 的项目内依赖

该命令只写 `my_devs/gr00t_17/tools/robot_client_deps/`：

```bash
cd /data/cqy_workspace/flexible_lerobot
/home/cqy/miniconda3/bin/conda run --no-capture-output -n lerobot_flex \
  bash my_devs/gr00t_17/scripts/setup_robot_client_deps.sh
```

## 8. 分阶段 Smoke

### 8.1 无动作 Smoke

该阶段先逐个只读检查六个舵机并在退出预检时禁用扭矩，然后加载正式 checkpoint，连接真实机器人和
双相机，读取关节状态，保存相机快照并做两次模型推理，但不会调用 `send_action` 或写 Goal_Position：

```bash
cd /data/cqy_workspace/flexible_lerobot
RUN_ID=so101_predict_$(date +%Y%m%d_%H%M%S) \
  /home/cqy/miniconda3/bin/conda run --no-capture-output -n lerobot_flex \
  bash my_devs/gr00t_17/scripts/run_so101_smoke.sh
```

通过条件：

- `reports/summary.json` 为 `status=passed`；
- `reports/bus_preflight.json` 证明六个舵机可读且预检退出时禁用扭矩；
- `level=live_observation_and_prediction`；
- `actuation_performed=false`、`sent_action_count=0`；
- 两路 camera shape 均为 `[480, 640, 3]`；
- 两个 16×6 prediction 均为有限值；
- 人工查看 `reports/snapshots/top.png` 和 `wrist.png` 后确认场景正确。

### 8.2 1 秒极小动作 Smoke

只有第 6 节全部确认后运行：

```bash
cd /data/cqy_workspace/flexible_lerobot
RUN_ID=so101_guarded_motion_$(date +%Y%m%d_%H%M%S) \
ENABLE_ACTUATION=1 \
ACTUATION_CONFIRM=SO101_GR00T_N17 \
  /home/cqy/miniconda3/bin/conda run --no-capture-output -n lerobot_flex \
  bash my_devs/gr00t_17/scripts/run_so101_smoke.sh
```

该 wrapper 会先重复无动作两轮预测，再执行：

```text
run_time_s=1
execution_horizon=1
control_hz=5
max_command_delta=0.25
max_relative_target=0.25
```

通过条件：`reports/summary.json` 必须为 `level=guarded_actuation`、`actuation_performed=true` 且
`sent_action_count>0`。任何异常、反向运动、明显抖动或碰撞趋势都应立即断电，不进入正式运行。

记录该次 `<GUARDED_RUN_ID>`。正式 wrapper 会重新读取它的 `reports/summary.json`，并核对 status、任务文本、
checkpoint-63600、实际发送动作数，以及 Smoke 参数不超过 1 秒、5 Hz、horizon 1、delta 0.25。无动作
Smoke、其他 checkpoint/任务或手工缺字段的 JSON 都不能解除正式门禁。

## 9. 正式 120 秒指令

仅在带动作 Smoke 通过后执行。以下命令使用与 PI 示例相同的设备和任务；基于真机验证，先使用
horizon 8、5 Hz、物理边界和单步双重限幅 3.0：

```bash
cd /data/cqy_workspace/flexible_lerobot
export ACTUATED_SMOKE_REPORT=/absolute/path/to/passed/guarded-smoke/reports/summary.json
RUN_ID=so101_n17_eraser_$(date +%Y%m%d_%H%M%S) \
  /home/cqy/miniconda3/bin/conda run --no-capture-output -n lerobot_flex \
  bash my_devs/gr00t_17/scripts/run_so101_infer.sh \
    --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00 \
    --top-cam /dev/video4 \
    --wrist-cam /dev/video6 \
    --task "Put the eraser into the small box" \
    --run-time-s 120 \
    --execution-horizon 8 \
    --control-hz 5 \
    --camera-warmup-s 2 \
    --bounds-mode physical \
    --max-command-delta 3 \
    --max-relative-target 3 \
    --request-timeout-s 2 \
    --max-inference-s 2 \
    --enable-actuation \
    --confirm-actuation SO101_GR00T_N17
```

按 `Ctrl-C` 时，终端信号会传给 robot client；client 的 `finally` 会断开设备并禁用扭矩，wrapper 随后
停止 policy server 和 GPU monitor。紧急情况优先物理断电，不要等待软件清理。

正式输出位于：

```text
my_devs/gr00t_17/outputs/inference/formal/<RUN_ID>/
├── logs/gpu_usage.log
├── logs/policy_server.log
├── logs/robot_client.log
├── reports/actuation_gate.json
├── reports/bus_preflight.json
├── reports/inference.json
└── reports/snapshots/{top,wrist}.png
```

## 10. 已完成的无动作 Smoke 结果

`so101_final_no_actuation_20260712_0958` 的实测结果：

| 检查项 | 结果 |
| --- | --- |
| checkpoint | step 63,600，3 个 shard，10,343,115,064 model bytes |
| robot serial | `/dev/ttyACM0`（by-id 解析结果） |
| top/wrist | RGB uint8 480×640，非黑屏/非过曝 |
| initial state | `[-6.6469, -99.5746, 93.8154, 66.4330, 11.7460, 1.6529]` |
| first inference | 0.3697 s（warm-up） |
| second inference | 0.0578 s |
| output | 两个 `(16, 6)` finite chunks |
| motor commands | 0 |
| summary | passed |

首动作安全预览中，第一个 chunk 不需要限幅；第二个 chunk 的 shoulder lift 为 `-100.5840`，安全层会
夹到 `-100`。因为本阶段没有动作，实际 `clipping_events=0`、`sent_action_count=0`。

现场快照与训练 episode 0 的 top/wrist 方向和相机映射一致。当前快照中任务物体并不清晰，因此该结果只
证明设备和模型链路正常，不构成自动执行动作的现场授权。复测前曾出现一次 6 号舵机
`Incorrect status packet`；流程按设计失败并清理。随后两次逐舵机只读检查均通过且六个扭矩寄存器为 0，
单次受控重试通过。若该错误重复出现，应检查夹爪供电、串联线缆和接头，不应持续重试。

## 11. 故障处理

- server 120 秒内未 ready：查看 `logs/policy_server.log`，确认显存和 checkpoint shard；
- 串口失败：确认没有另一个 LeRobot/串口进程占用 by-id 设备；
- 相机 shape 或 blank 检查失败：确认 `/dev/video4`、`/dev/video6` 映射和 fourcc；
- calibration mismatch：停止，不运行交互校准；先核对项目内校准副本和机器人 ID；
- prediction 出现 NaN/Inf 或 shape 不是 16×6：停止，不发送动作；
- clipping_events 很高：说明当前姿态/场景可能超出训练分布，回到无动作 Smoke 检查；
- 第二次稳定推理明显超过 0.2 秒：先排查 GPU 竞争，不通过提高 horizon 掩盖异常；
- 动作方向错误：立即断电，重新核对 joint order、相机映射和 checkpoint，不继续尝试。

## 12. 上机开发后的输入完整性

`../reports/post_deployment_integrity_20260712.json` 在全部上机脚本和无动作 Smoke 完成后重新计算并通过：

| 输入 | 条目数 | aggregate SHA-256 |
| --- | ---: | --- |
| 原始 v3 dataset | 14 | `5c18fe2ecf334f451f1d6c5cd01b9c393cc7bf742bd5c26726e154bf595227e8` |
| 原 PI0.5 checkpoint | 7 | `f85fc56ece26154920725faa43ed54a432e06f0ae5cddd80ca684e6e6f4cf846` |
| GR00T reference | 302 | `bbf3a837d8b0f8cd3fdd885ef01de65e170399287be6b907050025e8b0140c6f` |
| 转换后的训练副本 | 478 | `fde7fe09a84af986ddabe02e0d8d6e0ea5dec9ee17181acfdfb347e0f70b4815` |

同一门禁还复核了 55 个基础模型文件、19 个 backbone 文件和内部符号链接。所有结果均为 passed。
