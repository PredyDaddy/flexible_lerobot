# OpenPI SO101 10 Epoch LoRA 训练与实机推理工作报告

日期：2026-07-03

## 1. 模型训练过程

本次工作目标是在 `my_devs/openpi_train` 中完成 SO101 桌面整理任务的 OpenPI PI0.5 LoRA 训练，并将训练好的模型通过 websocket policy server 接入真实 SO101 follower 机器人。

训练数据来自已有 LeRobot 数据集：

```text
原始数据集:
/data/cqy_workspace/flexible_lerobot/datasets/desk_cleanup_v1/eraser_cup_multi_task

OpenPI 训练使用的 v2.1 转换数据:
/data/cqy_workspace/flexible_lerobot/my_devs/openpi_train/easy_use/data/lerobot_v21/desk_cleanup_v1/eraser_cup_multi_task
```

数据集规模：

```text
total_episodes = 157
total_frames   = 53235
fps            = 30
state_dim      = 6
action_dim     = 6
camera_keys    = observation.images.top, observation.images.wrist
```

任务文本包含三类：

```text
Put the eraser into the small box
Move the cup back to the upper-right corner
First put the eraser into the small box, then move the cup back to the upper-right corner
```

训练使用 OpenPI 的 JAX PI0.5 LoRA 路径。主配置由 `openpi_so101/config.py` 生成：

```text
model              = pi0.5
paligemma_variant = gemma_2b_lora
action_expert     = gemma_300m_lora
action_dim         = 32   # 模型内部维度，SO101 输出时裁剪前 6 维
action_horizon     = 50
use_delta_actions  = False
norm               = quantile norm
```

正式训练脚本：

```bash
cd /data/cqy_workspace/flexible_lerobot/my_devs/openpi_train
bash easy_use/train_full_v21_lora.sh
```

本次正式 run：

```text
EXP_NAME = so101_lora_full_v21_10epoch_bs48_20260702_171345
```

关键训练参数：

```text
batch_size       = 48
num_train_steps  = 11091
save_interval    = 1000
learning_rate    = 5e-5
base_params      = assets/openpi_cache/openpi-assets/checkpoints/pi05_base/params
asset_id         = desk_cleanup_v1/eraser_cup_multi_task_v21_full
```

训练步数与 10 epoch 的关系：

```text
53235 frames * 10 epoch / 48 batch_size = 11090.625 steps
```

OpenPI 训练循环保存最后一步时使用 step index，因此最终 checkpoint 是：

```text
step 11090 ~= 第 10 个 epoch 结束
```

最终权重目录：

```text
/data/cqy_workspace/flexible_lerobot/my_devs/openpi_train/outputs/checkpoints/pi05_so101_eraser_cup_lora/so101_lora_full_v21_10epoch_bs48_20260702_171345/11090
```

训练日志最后指标：

```text
Step 11090: grad_norm=0.0158, loss=0.0020, param_norm=1806.1256
```

模型服务端脚本：

```bash
cd /data/cqy_workspace/flexible_lerobot/my_devs/openpi_train && \
  DEFAULT_PROMPT="Put the eraser into the small box" \
  PORT=8000 \
  bash easy_use/serve_robot_policy_10epoch_bs48.sh
```

服务端实际加载的 checkpoint：

```text
outputs/checkpoints/pi05_so101_eraser_cup_lora/so101_lora_full_v21_10epoch_bs48_20260702_171345/11090
```

环境划分：

```text
服务端 / OpenPI 训练与模型推理: openpi_train
真实机器人客户端 / SO101 硬件控制: lerobot_flex
```

机器人客户端仅复用 `openpi-main/packages/openpi-client/src` 中的轻量 websocket client，不在 `lerobot_flex` 中安装完整 OpenPI 包，避免影响机器人环境依赖。

## 2. 最终推理效果

服务端启动后，客户端可以正常连接 websocket policy server，并读取服务端 metadata：

```text
robot       = so101_follower
state_dim   = 6
action_dim  = 6
camera_keys = observation.images.top, observation.images.wrist
```

服务端输入 observation：

```text
observation.images.top    HWC uint8
observation.images.wrist  HWC uint8
observation.state         shape=(6,) float32
prompt                    task string
```

服务端输出 action chunk：

```text
actions shape = (50, 6)
```

6 维动作顺序：

```text
shoulder_pan.pos
shoulder_lift.pos
elbow_flex.pos
wrist_flex.pos
wrist_roll.pos
gripper.pos
```

初始实机推理效果不理想。主要表现是动作不够连续，执行过程容易显得犹豫或碎片化。排查后发现，最初客户端每次只执行服务端返回 action chunk 的第 0 步：

```text
policy 输出 50 步动作
客户端只执行第 1 步
剩余 49 步丢弃
然后立刻重新请求 policy
```

这会破坏 OpenPI 训练时学习到的 50-step 连续轨迹结构。将客户端改为连续执行 action chunk 的前 N 步后，实机效果明显提升。

最终实机推理中，`--action-chunk-steps 30` 的效果明显更好：

```text
action_horizon      = 50
fps                 = 30
action_chunk_steps  = 30
实际执行窗口        = 30 / 30fps = 1 秒
```

也就是说，模型每次预测约 1.67 秒动作，客户端执行前约 1 秒动作后重新观测和规划。这个节奏在当前 “Put the eraser into the small box” 任务上更稳定，动作连续性和任务完成效果都明显改善。

推荐实机推理命令：

```bash
cd /data/cqy_workspace/flexible_lerobot/my_devs/openpi_train && \
  HOST=localhost \
  PORT=8000 \
  EXECUTE_ACTIONS=true \
  bash easy_use/run_robot_remote_client.sh \
    --task "Put the eraser into the small box" \
    --robot-id hfy_follower \
    --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00 \
    --calib-dir /home/cqy/.cache/huggingface/lerobot/calibration/robots/so_follower \
    --top-cam /dev/video4 \
    --wrist-cam /dev/video6 \
    --top-cam-fourcc YUYV \
    --wrist-cam-fourcc MJPG \
    --img-width 640 \
    --img-height 480 \
    --fps 30 \
    --run-time-s 60 \
    --max-relative-target 10 \
    --motor-io-retries 10 \
    --action-chunk-steps 30
```

## 3. 改动原因及效果提升分析

### 3.1 服务端与客户端解耦

OpenPI 模型推理依赖 JAX、OpenPI、checkpoint 和较重的 GPU 环境；SO101 实机控制依赖 LeRobot、串口、电机和相机环境。为了避免环境互相污染，本次实现拆分为：

```text
服务端: openpi_train 环境，负责加载 PI0.5 LoRA checkpoint 并提供 websocket policy server
客户端: lerobot_flex 环境，负责采集机器人 observation 并执行动作
```

这样做的好处：

```text
1. GPU 模型环境和机器人控制环境独立，减少依赖冲突。
2. 客户端只需要轻量 openpi-client websocket 代码。
3. 后续可以将服务端放在 GPU 主机，客户端放在机器人控制主机。
```

### 3.2 新增远程推理客户端

新增文件：

```text
openpi_so101/remote_infer_client.py
openpi_so101/robot_remote_client.py
easy_use/infer_robot_policy_client.sh
easy_use/run_robot_remote_client.sh
easy_use/serve_robot_policy_10epoch_bs48.sh
```

其中：

```text
remote_infer_client.py
  用于不连接机器人时，用数据集样本或随机 observation 测试服务端是否可用。

robot_remote_client.py
  用于真实 SO101 机器人 websocket 推理。

serve_robot_policy_10epoch_bs48.sh
  固定加载 10 epoch / bs48 / full v21 的最终 checkpoint。
```

### 3.3 增加 SO101 电机通信重试

实机调试时曾出现 Feetech 总线通信错误：

```text
Failed to write 'Lock' on id_=3 with '1'
[TxRxResult] Incorrect status packet!
```

该错误发生在 `robot.connect()` 阶段，早于 policy 推理和动作执行。它不是模型问题，而是 SO101 电机总线在 configure/enable torque 时的通信稳定性问题。

为此，客户端增加：

```bash
--motor-io-retries 10
```

对 motor bus 的 `read/write/sync_read/sync_write` 设置最小重试次数，提升串口、电机总线瞬时通信错误下的鲁棒性。同时增加诊断输出，方便定位具体失败 motor id。当前 SO follower 映射中：

```text
id=3 -> elbow_flex
```

### 3.4 修正 action chunk 执行策略

这是本次效果提升最关键的改动。

OpenPI/PI0.5 的输出不是单步 action，而是 action chunk：

```text
actions shape=(50, 6)
```

训练时模型学习的是连续 50 步动作轨迹。如果实机端只执行第一步，就会变成：

```text
高频重新规划
轨迹被频繁打断
动作连续性变差
模型预测的后续轨迹信息被浪费
```

因此客户端新增：

```bash
--action-chunk-steps N
```

用于控制每次 policy 返回 50 步后，实际连续执行前 N 步。经过实机调试：

```text
N=10: 比只执行 1 步更合理，但仍偏保守
N=30: 当前任务效果明显变好
```

原因分析：

```text
1. N=30 对应约 1 秒动作，保留了模型输出轨迹的连续性。
2. 每 1 秒重新观测一次，仍保留了一定闭环纠偏能力。
3. 相比 N=1，避免了过度重新规划导致的动作碎片化。
4. 相比直接执行 50 步，N=30 又不会完全开环执行 1.67 秒，风险更低。
```

### 3.5 关于 LoRA 参数是否不足

从本次现象看，效果差的首要原因不是 LoRA 参数太少。证据包括：

```text
1. 10 epoch 最终训练 loss 约 0.0020，训练过程正常收敛。
2. LoRA 参数确实参与训练，日志中可见 lora_a/lora_b 参数。
3. 将 action_chunk_steps 调到 30 后，实机效果明显提升。
```

因此当前优先级判断是：

```text
第一优先级: 实机 action chunk 执行策略
第二优先级: 相机视角、光照、物体初始位置与训练分布一致性
第三优先级: 数据质量与任务覆盖
第四优先级: LoRA 容量或继续训练轮数
```

如果后续继续优化，可以考虑：

```text
1. 固定 action_chunk_steps=30 作为当前默认实机推理参数。
2. 补充更多失败场景和边界初始位置数据。
3. 分任务单独训练或按任务均衡采样。
4. 对比 10 epoch、15 epoch、20 epoch checkpoint。
5. 在确定执行策略无误后，再考虑提升 LoRA 容量或解冻更多 action expert 参数。
```

## 当前结论

本次 OpenPI SO101 LoRA 训练链路已经跑通：

```text
数据转换 -> norm stats -> 10 epoch LoRA 训练 -> policy server -> websocket client -> SO101 实机推理
```

最终服务端使用：

```text
so101_lora_full_v21_10epoch_bs48_20260702_171345/11090
```

实机效果的关键提升来自：

```text
将 action chunk 执行策略从 “只执行第 1 步” 改为 “连续执行前 30 步”。
```

当前推荐默认实机参数：

```text
--action-chunk-steps 30
--motor-io-retries 10
--max-relative-target 10
```
