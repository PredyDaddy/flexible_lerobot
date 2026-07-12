# GR00T N1.7 SO101 正式训练 Runbook

> 状态更新（2026-07-12）：`so101_n17_b2_e10_20260711` 正式 run 已以 micro batch 2、gradient
> accumulation 4 完成 63,600 steps（10.0 nominal sample epochs）。本文件保留训练入口说明；实际结果
> 见执行报告和 `../reports/formal_training_validation.json`，真机流程见
> `GR00T_N1_7_SO101_ROBOT_DEPLOYMENT.md`。

## 1. 唯一推荐入口

在仓库根目录执行：

```bash
cd /data/cqy_workspace/flexible_lerobot
/home/cqy/miniconda3/bin/conda run --no-capture-output -n lerobot_flex \
  bash my_devs/gr00t_17/scripts/run_train.sh
```

不要直接调用 `launch_finetune.py`。包装脚本会固定本地环境、模型、backbone 运行时资产、转换数据、
modality config、离线模式、输出根目录和可恢复 checkpoint，并在启动前重新校验只读输入。

## 2. 启动门禁

正式训练只有同时满足以下条件才会启动：

1. 当前 host conda 环境是 `lerobot_flex`，实际 Python 是
   `my_devs/gr00t_17/env/gr00t_n17/bin/python`；
2. `uv.lock` 已固定本机构建的 FlashAttention wheel；
3. 环境、数据、模型、真实 batch 和 smoke 报告均为 `passed`；
4. 原始 v3 数据、PI0.5 checkpoint 和 `reference/` 的完整清单哈希未改变；
5. 转换后的训练副本与 478 条派生 manifest 完全一致；
6. smoke 的 step/resume/loss/权重 delta 和 checkpoint 状态证据仍完整；
7. N1.7 模型文件和 Qwen3-VL runtime 资产 SHA-256 未改变；
8. 输出路径解析后仍位于 `my_devs/gr00t_17/`；
9. 新训练目录为空，或 resume 模式下存在完整 `trainer_state.json`；
10. GPU 空闲显存不少于 40,000 MiB。

若存在 VLLM、其他训练进程或大显存推理服务，预检会失败。不要降低门槛后与其他任务抢显存。

## 3. 默认正式参数

| 参数 | 默认值 |
| --- | ---: |
| 数据 | 全部 157 episodes / 50,880 个有效 horizon-16 windows |
| video | 当前 top + wrist 双视角 |
| state | 当前 5 维 arm + 1 维 gripper |
| action | 16 步；arm relative，gripper absolute |
| tune LLM / visual | false / false |
| tune projector / diffusion | true / true |
| per-device micro batch | 1 |
| gradient accumulation | 8 |
| effective batch | 8 |
| max steps | 2,000 optimizer steps |
| learning rate | 1e-4 |
| weight decay | 1e-5 |
| warmup ratio | 0.05 |
| state dropout | 0.2 |
| shard size | 1,024 |
| dataloader workers | 4 |
| save interval / retained | 500 / 5 |
| precision | BF16 + TF32 |
| W&B | disabled |
| resumable state | enabled（model + optimizer + scheduler + RNG） |

`episode_sampling_rate=0.1` 在该上游实现中用于把每个 episode 拆成 10 个子序列以平衡 shard，
不会丢弃 90% 数据。预检还会验证 shard 数不超过可分配子序列数。

micro batch 1 的真实两阶段 smoke 采样峰值总显存为 37,852 MiB，单个训练 micro-step 计算约
8.5-9.2 秒。正式默认的 gradient accumulation 8 因此约需 8 个 micro-step 才产生一个 optimizer
step；实际总时长还受视频解码、shard 等待和 checkpoint I/O 影响，不应把 2,000 steps 理解为
2,000 个单样本前向。仅按 smoke 计算耗时外推约为 38-41 小时，实际 wall time 通常更长。

上游 new-embodiment 示例使用 effective batch 32。当前默认 effective batch 8 是单卡保守基线，
`2,000 * 8 / 50,880` 约为 0.314 个窗口 epoch，并非与上游相同的样本预算。先运行该基线并检查
checkpoint；若要严格对齐上游 batch 32，可设置 `GRADIENT_ACCUMULATION_STEPS=32`，但计算时间也会
接近默认的四倍。不要同时擅自线性放大学习率。

## 4. 输出位置

默认 `RUN_ID` 是启动时间：

```text
my_devs/gr00t_17/outputs/formal/<RUN_ID>/
├── train/                       # root final model + checkpoint-*
├── logs/
│   ├── train_terminal.log
│   └── gpu_usage.log
└── reports/
    ├── preflight_<timestamp>.json
    └── launch_parameters.env
```

启动时终端会打印真实 `TRAIN_DIR`。不要把 `train/` 指向基础模型、原数据或仓库顶层
`outputs/`。

实测每个完整可恢复 checkpoint 为 23,314,772,866 bytes。默认保留 5 个 checkpoint，另有根目录
final model、日志和临时写入，启动正式训练前应至少为该 run 预留约 130 GB；保存期间出现明显磁盘
等待属于预期现象。

## 5. 中断恢复

假设原 run ID 是 `20260711_150000`：

```bash
cd /data/cqy_workspace/flexible_lerobot
RUN_ID=20260711_150000 RESUME=1 \
  /home/cqy/miniconda3/bin/conda run --no-capture-output -n lerobot_flex \
  bash my_devs/gr00t_17/scripts/run_train.sh
```

恢复前会重新校验全部输入，并要求找到最高编号且带 `trainer_state.json` 的 checkpoint。
`save_only_model=false` 已固定，optimizer、scheduler 和 RNG 状态都会恢复。不要手工把残缺目录改名为
`checkpoint-N`。

## 6. 参数覆盖

所有覆盖仍通过同一包装脚本。例如把有效 batch 提高到 16：

```bash
GRADIENT_ACCUMULATION_STEPS=16 \
  /home/cqy/miniconda3/bin/conda run --no-capture-output -n lerobot_flex \
  bash my_devs/gr00t_17/scripts/run_train.sh
```

可覆盖变量：

```text
RUN_ID
MAX_STEPS
GLOBAL_BATCH_SIZE
GRADIENT_ACCUMULATION_STEPS
DATALOADER_NUM_WORKERS
SHARD_SIZE
SAVE_STEPS
SAVE_TOTAL_LIMIT
RESUME
```

单卡不建议提高 `GLOBAL_BATCH_SIZE`，优先用 gradient accumulation 扩大有效 batch。任何 horizon、
modality、动作表示或 state 单位变更都不是运行参数调整，必须重新生成 relative stats 并重新完成数据验收。

## 7. 训练中检查

终端日志至少应持续出现有限值 loss、step time 和 dataloader timing。另一个终端可只读查看：

```bash
tail -f my_devs/gr00t_17/outputs/formal/<RUN_ID>/logs/train_terminal.log
```

GPU 记录位于同一 run 的 `logs/gpu_usage.log`。出现下列任一情况应停止并保留现场：

- loss 为 NaN/Inf；
- TorchCodec 解码或 worker 异常；
- CUDA OOM；
- 输出目录越界或预检报告失败；
- checkpoint 缺 optimizer/scheduler/RNG；
- 原始输入清单变化。

## 8. 训练结束边界

本 runbook 只负责正式微调。正式训练结束后仍需做 checkpoint 离线加载、三任务 open-loop 指标和真机
dry-run。模型输出中的 arm action 是相对表示，必须由保存的 processor 反归一化并结合当前 state 恢复为
绝对目标，禁止直接发送给 SO101。

## 9. 已执行的 10-epoch 正式 run

用户授权后实际采用 `GLOBAL_BATCH_SIZE=2` 和 `GRADIENT_ACCUMULATION_STEPS=4`，每个 optimizer step
处理 8 个样本。`MAX_STEPS=63600` 对 50,880 个有效窗口恰好是 10.0 个样本 epoch。最终 checkpoint 为
`outputs/formal/so101_n17_b2_e10_20260711/train/checkpoint-63600`，完整验收状态为 passed。早期章节的
micro-batch-1/2,000-step 内容是启动前的保守 baseline，不是最终正式 run 的实际值。
