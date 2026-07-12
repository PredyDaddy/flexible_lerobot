# GR00T N1.7 SO101 多任务训练执行报告

## 1. 范围和结论

本次工作为现有 SO101 双相机三任务数据建立一条完全位于
`/data/cqy_workspace/flexible_lerobot/my_devs/gr00t_17/` 的 GR00T N1.7 微调链路。

执行遵守以下边界：

- 原始 LeRobot v3 数据只读；
- 原 PI0.5 checkpoint 只读；
- 用户提供的 `reference/` 只读；
- N1.7 的可修改代码只位于 `workspace/` 副本；
- venv、依赖缓存、模型、转换数据、编译产物、日志、报告和训练输出均位于本项目根目录；
- 正式训练默认离线，不接收 Hub 模型 ID；
- 所有训练输出路径先做 resolve 和根目录边界检查。

## 2. 只读输入完整性基线

完整 manifest 位于 `../data/source_manifests/`。最终训练预检和 smoke 后检查都会重新计算全部条目，
而不是只比较文件数量或 mtime。

| 输入 | 条目 | aggregate SHA-256 |
| --- | ---: | --- |
| 原始数据 | 14 | `5c18fe2ecf334f451f1d6c5cd01b9c393cc7bf742bd5c26726e154bf595227e8` |
| PI0.5 checkpoint | 7 | `f85fc56ece26154920725faa43ed54a432e06f0ae5cddd80ca684e6e6f4cf846` |
| GR00T reference | 302 | `bbf3a837d8b0f8cd3fdd885ef01de65e170399287be6b907050025e8b0140c6f` |

PI manifest 的解析根目录固定为
`outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/020000/pretrained_model`，
不会因 `last` 符号链接以后改变而漂移。

## 3. 隔离环境

环境路径：

```text
my_devs/gr00t_17/env/gr00t_n17
```

最终关键版本：

| 组件 | 版本/结果 |
| --- | --- |
| Python | 3.10.19 |
| uv | 0.11.28 |
| lock packages | 178 |
| PyTorch | 2.7.1+cu128 |
| torch CXX11 ABI | true |
| CUDA runtime | 12.8 |
| Transformers | 4.57.3 |
| TorchCodec | 0.4.0 |
| FlashAttention | 2.7.4.post1 |
| GPU | NVIDIA GeForce RTX 4090，50,875,924,480 bytes |

所有 cache 环境变量由 `../scripts/common.sh` 固定到本项目 `cache/`。W&B、HF telemetry 和
Albumentations update check 已禁用，训练阶段启用 `HF_HUB_OFFLINE=1` 和
`TRANSFORMERS_OFFLINE=1`。

### 3.1 FlashAttention 本机兼容处理

上游预编译 wheel 同时存在两个问题：要求 host 没有的 `GLIBC_2.32`，且 wheel 名称声明的 CXX11
ABI 与本地 Torch 不一致。没有降低版本或修改其他 conda 环境，而是在本项目中从
`flash-attn==2.7.4.post1` sdist 构建 host wheel：

```text
tools/wheels/flash_attn-2.7.4.post1-cp310-cp310-linux_x86_64.whl
SHA-256 79e2eeb9ff257d17a6f0c591655812d5289debfab301a4e7615f297f23735aa9
```

构建使用 CUDA 12.6 toolchain、ninja 8 jobs、`sm_80` cubin；生成扩展最高只要求
`GLIBC_2.14`。Ada/RTX 4090 上实际完成以下 BF16 测试：

- head dimension 64、non-causal：forward + backward passed；
- head dimension 128、causal：forward + backward passed。

wheel 已写入 workspace `pyproject.toml` 和 `uv.lock`，并通过
`uv sync --locked --offline` 复核。详见 `../reports/flash_attn_build.json`。

### 3.2 TorchCodec/libffi 处理

`lerobot_flex` 基础解释器的 RPATH 会优先找到一个名为 `libffi.so.7`、实际指向 libffi 8 的兼容
链接，导致系统 FFmpeg 4 加载 `libgobject` 时缺少 `LIBFFI_BASE_7.0` 符号。

`common.sh` 显式预加载系统真实 `/lib/x86_64-linux-gnu/libffi.so.7`。最终 TorchCodec 0.4 已完成
300 帧 AV1 视频首/中/尾解码，输出 `(3, 480, 640, 3) uint8`。

## 4. 非破坏性数据转换

源：

```text
datasets/desk_cleanup_v1/eraser_cup_multi_task
```

目标：

```text
my_devs/gr00t_17/data/converted_v21/desk_cleanup_v1/eraser_cup_multi_task
```

转换器使用显式只读 `source_root -> partial -> final` 流程，没有调用会移动输入目录的上游默认 CLI。
输出为 GR00T 所需的 LeRobot v2.1 per-episode 布局：

- 157 个 per-episode parquet；
- top/wrist 各 157 个 AV1 视频，共 314 个；
- 53,235 行；
- 3 个任务；
- 30 FPS，640 x 480；
- state/action 均为原始 6 维 float32 值；
- 新增 `meta/modality.json`；
- 新增 `(16, 5)` 的 arm relative action 逐 horizon 统计。

动作语义保持官方 SO100/N1.7 配置：前五维 arm 在 processor 内转 relative，第六维 gripper 保持
absolute；原 parquet 中的 absolute target 没有被改写。

### 4.1 最终独立深度校验

最终数据在所有 processor/statistics 操作结束后重新完整验证：

| 检查 | 结果 |
| --- | ---: |
| 与 v3 逐值比较的 parquet rows | 53,235 / 53,235 |
| FFprobe 完整计帧的视频 | 314 / 314 |
| TorchCodec 源/目标逐像素相等帧 | 942 / 942 |
| task 0 frames | 18,600 |
| task 1 frames | 16,500 |
| task 2 frames | 18,135 |
| relative action values | `(50,880, 16, 5)` |

报告：`../reports/dataset_validation.json`。

所有验证和 smoke 写入结束后，另生成训练副本的 478 条全量 manifest：

```text
data/derived_manifests/converted_dataset.json
aggregate SHA-256 fde7fe09a84af986ddabe02e0d8d6e0ea5dec9ee17181acfdfb347e0f70b4815
```

该 manifest 位于数据目录外，避免把自身纳入哈希；每次 smoke 和正式 preflight 都会重新计算并逐项比较，
因此转换后的 parquet、视频或 metadata 在验证后发生变化也会阻止训练。

## 5. 模型和 backbone 资产

### 5.1 N1.7 主 checkpoint

```text
repo     nvidia/GR00T-N1.7-3B
revision 2fc962b973bccdd5d8ce4f67cc63b264d6886495
path     my_devs/gr00t_17/models/GR00T-N1.7-3B
files    55
bytes    6,927,373,531
```

两个 safetensors shard、index、config、processor config、statistics 和 experiment config 均已做
SHA-256 清单，正式训练预检会逐文件复核。

### 5.2 Cosmos/Qwen3-VL runtime 资产

N1.7 checkpoint 包含完整 VLM backbone 权重，但原代码构造模型和 processor 时还会访问
`nvidia/Cosmos-Reason2-2B` 的 config/tokenizer/image processor。该仓库需要接受许可并使用授权 token，
当前环境没有 token，因此没有尝试绕过 gate，也没有从非官方来源拉取权重。

HF 文件元数据证明，Cosmos pinned revision 与公开 Qwen pinned revision 的九个运行时文件具有完全相同的
git blob ID 和 size：

```text
Qwen/Qwen3-VL-2B-Instruct revision
89644892e4d85e24eaac8bacfd4f463576704203

nvidia/Cosmos-Reason2-2B revision
9ce19a195e423419c349abfc86fd07178b230561
```

因此本项目只下载公开 Qwen 仓库中逐字节相同的九个 runtime 文件，不下载 Qwen 权重：

```text
models/Qwen3-VL-2B-Instruct-assets
```

workspace 增加显式 `backbone_init_from_config` 路径：先用这些资产构造 Qwen3-VL 架构，再由 N1.7
safetensors 严格填充全部参数。任何 missing、unexpected 或 mismatched key 都直接报错。

### 5.3 严格加载结果

| 项目 | 结果 |
| --- | ---: |
| 总参数 | 3,144,016,000 |
| 默认可训练参数 | 1,620,515,968 |
| language layers（N1.7 select） | 16 |
| missing keys | 0 |
| unexpected keys | 0 |
| mismatched keys | 0 |
| attention implementation | `flash_attention_2` |

另从两个 safetensors shard 选择 vision patch、LLM q-proj、token embedding 和 action decoder 四个张量，
与加载后 `state_dict` 做逐值 `torch.equal`，四项均完全相等。

报告：`../reports/model_validation.json`。

## 6. 真实训练 batch

使用转换后的真实数据、保存的统计量、Qwen3-VL processor 和项目 modality 构造 batch，结果：

| tensor | shape | dtype |
| --- | --- | --- |
| input IDs | `(1, 157)` | int64 |
| pixel values | `(512, 1536)` | float32 |
| image grid | `(2, 3)` | int64 |
| state | `(1, 1, 132)` | float32 |
| action | `(1, 40, 132)` | float32 |
| action mask | `(1, 40, 132)` | float64 |
| embodiment ID | `(1,)` | int64 |

两路图像和组合任务文本都进入 prompt。16 步 x 6 维数据在 N1.7 最大 action tensor 中产生 96 个 active
mask values；所有浮点 tensor 为有限值。50,880 个有效窗口在 smoke `shard_size=64` 下生成 795 个
非空 shard。

报告：`../reports/training_batch.json`。

## 7. Smoke Train

<!-- SMOKE_RESULTS_START -->
实际 run：

```text
RUN_ID    smoke_20260711_144453
train dir my_devs/gr00t_17/outputs/smoke/smoke_20260711_144453/train
GPU       NVIDIA GeForce RTX 4090，49,140 MiB
```

两阶段训练均使用真实双相机数据、micro batch 1、BF16/TF32、FlashAttention 2，并保存完整
model/optimizer/scheduler/RNG 状态：

| 阶段 | 行为 | global step | terminal train loss | trainer runtime |
| --- | --- | ---: | ---: | ---: |
| 1 | 从 N1.7 基座启动并保存 `checkpoint-1` | 1 | 1.4883127213 | 45.4055 s |
| 2 | 明确从 `checkpoint-1` 恢复并保存 `checkpoint-2` | 2 | 0.5696015954 | 105.6169 s |

`warmup_ratio=0.05` 在两步 smoke 中形成一个 warmup step。第 1 步的实际学习率为 0，因此选定的
action-head 张量从基座到 `checkpoint-1` 的 delta 为 0；`checkpoint-1` 的 scheduler 已推进到 epoch 1，
并把下一步学习率置为 `1e-4`。从该 checkpoint 恢复后，第 2 步产生真实更新：

| 权重检查 | max absolute difference |
| --- | ---: |
| 可训练 `action_head.action_decoder.layer1.W`，step 1 -> 2 | `0.00010013580322265625` |
| 冻结 LLM q-proj，base -> step 2 | `0.0` |

可训练参数从基础 checkpoint 的 BF16 提升为 FP32 optimizer 参数后保存；冻结 LLM 参数仍为 BF16。
续训日志中的 `lm_head.weight` missing-key 提示来自 Qwen3-VL 的 tied embedding 保存：runtime config
明确设置 `tie_word_embeddings=true`，基础 checkpoint 的 lm-head 和 token embedding 逐值完全相等，且
LLM 在本训练策略中冻结，不代表丢失可训练权重。

其他验收结果：

- 两个 checkpoint 的 `trainer_state.json` 分别为 step 1 和 2；
- 两者都包含非空 `optimizer.pt`、`scheduler.pt` 和 `rng_state.pth`；
- 每个完整 checkpoint 为 23,314,772,866 bytes；整个 smoke train 目录为 56,979,711,110 bytes；
- 2 秒间隔监控的峰值总显存为 37,852 MiB；
- 两阶段 loss 均为有限值；
- smoke 后再次全量计算原数据、PI checkpoint 和 reference manifest，三个 aggregate SHA-256 均未改变；
- 模型文件、runtime 资产和根目录内符号链接也再次通过预检。

最终报告 `../reports/smoke_train.json` 的状态为 `passed`。该结果在当时用于验证正式训练链路、保存和恢复；
后续 10-epoch 正式训练结果见第 9 节。
<!-- SMOKE_RESULTS_END -->

## 8. 正式训练入口

正式命令、默认参数、输出布局和恢复方式见：

`GR00T_N1_7_SO101_TRAINING_RUNBOOK.md`

包装脚本：

- `../scripts/run_train.sh`；
- `../scripts/preflight.py`；
- `../scripts/common.sh`。

正式入口要求 smoke 报告为 `passed`，默认使用新时间戳 run 目录；resume 必须显式设置同一 `RUN_ID`
和 `RESUME=1`。

以正式默认 `shard_size=1024`、40,000 MiB 空闲显存门槛和 `--require-smoke` 做的最终只读演练已通过，
报告为 `../reports/preflight_formal_ready.json`。该门禁同时重算 478 条转换数据 manifest，并核验 smoke
的 checkpoint step、resume、有限 loss、可训练/冻结权重 delta 和 optimizer/scheduler/RNG 文件；演练
没有创建模型训练输出或启动 optimizer step。

## 9. 关键报告索引

| 文件 | 内容 |
| --- | --- |
| `data_conversion.json` | 转换发布结果 |
| `dataset_validation.json` | parquet/video/pixel 全量校验 |
| `environment.json` | venv、lock 和 package 快照 |
| `environment_validation.json` | CUDA/FlashAttention/TorchCodec kernel 检查 |
| `flash_attn_build.json` | host wheel 构建与二进制要求 |
| `model_download.json` | N1.7 pinned snapshot SHA 清单 |
| `backbone_assets.json` | Cosmos/Qwen runtime blob 等价证明 |
| `model_validation.json` | 严格加载和张量逐值比较 |
| `training_batch.json` | 真实 processor/collator batch |
| `preflight_probe.json` | 路径、输入、模型和显存预检探针 |
| `smoke_train.json` | 两阶段训练、保存、恢复、权重更新和冻结参数证明 |
| `preflight_formal_ready.json` | 正式默认参数的最终启动门禁演练 |
| `formal_training_validation.json` | 10-epoch 正式训练、最终 checkpoint 和真实 server 加载证明 |
| `post_deployment_integrity_20260712.json` | 上机开发后原数据/权重/reference/转换数据完整性复核 |

全部报告位于 `my_devs/gr00t_17/reports/`。

## 10. 10-epoch 正式训练与 checkpoint 验收

正式 run：

```text
RUN_ID=so101_n17_b2_e10_20260711
micro batch=2
gradient accumulation=4
effective batch=8
max steps=63600
save steps=6360
```

由于转换数据包含 50,880 个有效 horizon-16 windows，样本预算为：

```text
63600 * 8 / 50880 = 10.0 nominal sample epochs
```

训练完成指标：

```text
train_runtime=32231.7025 s
train_samples_per_second=15.786
train_steps_per_second=1.973
train_loss=0.06916177219213367
final logged loss=0.0358
```

最终 `checkpoint-63600` 包含 3 个模型 shard（10,343,115,064 bytes）、processor、statistics、
12,964,594,710-byte optimizer state、scheduler 和 RNG state。训练参数为冻结 LLM 与 visual/ViT，训练
projector、diffusion/DiT 和 VLLN self-attention action module。

`../reports/formal_training_validation.json` 重新核对训练预算、trainer state、最终 metrics、全部关键文件
和真实 policy server 加载证据，状态为 `passed`。

## 11. 真机无动作 Smoke

部署实现和正式命令见 `GR00T_N1_7_SO101_ROBOT_DEPLOYMENT.md`。

`outputs/inference/smoke/so101_final_no_actuation_20260712_0958` 已在真实 SO101 上完成：

- by-id 串口解析为 `/dev/ttyACM0`，项目内校准成功使用；
- top/wrist 均为 RGB uint8 480×640 且通过非黑屏检查；
- checkpoint-63600 三个 shard 在 CUDA 上完整加载；
- 两次输出均为 finite 16×6 action chunk；
- warm-up 推理 0.3697 秒，第二次推理 0.0578 秒；
- `sent_action_count=0`，未向电机发送命令；
- summary 状态为 `passed`。

带动作 Smoke 必须在机器人固定、运动范围清空、任务物体摆放完成且操作员可立即断电后执行，不能把
无动作预测通过等同于动作安全验收。

## 12. 上机开发后不可变输入复核

完成真机脚本和无动作 Smoke 后再次运行完整 preflight。原始 dataset、PI checkpoint、GR00T reference
和 478 项转换数据的 aggregate SHA-256 均与训练前 manifest 完全一致；55 个模型文件、19 个 backbone
文件和内部符号链接也通过。报告 `../reports/post_deployment_integrity_20260712.json` 状态为 `passed`。
