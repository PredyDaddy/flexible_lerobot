# GR00T N1.7 SO101 多任务训练工作计划

## 1. 文档状态

- 编写日期：2026-07-11
- 工作目录：`/data/cqy_workspace/flexible_lerobot/my_devs/gr00t_17`
- 目标模型：`nvidia/GR00T-N1.7-3B`
- 原始数据：`datasets/desk_cleanup_v1/eraser_cup_multi_task`
- 参考训练：`my_devs/train/pi/so101/easy_train.sh`
- 当前阶段（2026-07-12 更新）：环境、模型、全量数据转换、真实 batch、两阶段 save/resume smoke
  和正式训练均已完成。实际正式 run 使用 micro batch 2、gradient accumulation 4、63,600 steps，
  对应 50,880 个有效窗口的 10 个样本 epoch；checkpoint-63600 已通过真机 server 加载和无动作预测
  Smoke。带动作 Smoke 等待操作员完成现场安全确认。

## 2. 目标与完成定义

目标是在不修改原始数据、原 PI 权重和 N1.7 参考源码的前提下，使用同一份 SO101 双相机多任务数据，完成 GR00T N1.7 的数据转换、训练、离线评估和后续真机推理准备。

本任务只有同时满足以下条件才算完成：

1. 所有新环境、依赖缓存、模型、转换数据、配置、日志、检查点和评估结果都位于 `my_devs/gr00t_17/`。
2. 原始 v3 数据、PI 权重和 `reference/` 内容在工作前后哈希一致。
3. 转换数据通过结构、schema、逐 episode 行数、逐视频帧数、任务映射、双相机同步和随机解码检查。
4. N1.7 数据加载器能读取真实样本，状态、动作、语言和两路图像的键及 shape 全部正确。
5. 先通过 1 batch 前向/反向和短 smoke train，再允许正式训练。
6. 正式训练可恢复、日志完整、检查点可独立加载，并有离线评估结果。
7. 真机动作在发送前经过反归一化、相对转绝对、关节顺序和安全限幅验证。

## 3. 不可破坏边界

### 3.1 只读输入

以下路径只能读取，禁止写入、移动、改名、覆盖、删除、原地转换或创建辅助文件：

- `/data/cqy_workspace/flexible_lerobot/datasets/desk_cleanup_v1/eraser_cup_multi_task`
- `/data/cqy_workspace/flexible_lerobot/outputs/pi05_eraser_cup_multi_task_runs/20260602_200955`
- `/data/cqy_workspace/flexible_lerobot/my_devs/gr00t_17/reference`

已有 PI checkpoint 的 `last/pretrained_model` 实际解析到：

`outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/020000/pretrained_model`

它只用于理解现有输入输出契约和做结果对照，不作为 N1.7 的初始化权重，也不允许被 N1.7 训练覆盖。

### 3.2 唯一允许的新写入根目录

所有工作产物必须位于：

`/data/cqy_workspace/flexible_lerobot/my_devs/gr00t_17/`

任何脚本启动前都应解析所有输出路径，并拒绝下列情况：

- 输出路径不在上述根目录内；
- 输出路径经过符号链接后逃逸到根目录外；
- 输出路径等于任一只读输入路径或其父目录；
- 目标数据目录已存在但未显式指定安全续跑模式；
- 训练输出目录与基础模型目录、数据目录重合。

### 3.3 官方转换脚本的禁用规则

参考仓的 `scripts/lerobot_conversion/convert_v3_to_v2.py` 会在结束时：

1. 将输入目录移动为 `<name>_v3.0`；
2. 将生成的 v2.1 目录移动到原输入路径。

因此，禁止直接把原始数据路径传给该脚本。后续应实现显式的只读 `source_root -> output_root` 转换入口，或只在 `my_devs/gr00t_17/tmp/` 内的完整副本上运行官方逻辑。首选前者，因为它更容易审计，也避免不必要的 626 MiB 中间副本。

## 4. 已完成的只读现状审计

### 4.1 原始数据概况

| 项目 | 实际值 |
| --- | --- |
| LeRobot 版本 | `v3.0` |
| robot type | `so_follower` |
| episode 数 | 157 |
| 总帧数 | 53,235 |
| FPS | 30 |
| 任务数 | 3 |
| 状态维度 | 6 |
| 动作维度 | 6 |
| 相机 | `observation.images.top`、`observation.images.wrist` |
| 图像尺寸 | 480 x 640 x 3 |
| 视频编码 | AV1, yuv420p |
| 活跃数据体积 | 约 626 MiB |

三个任务及分布：

| task_index | 指令 | episodes | frames |
| --- | --- | ---: | ---: |
| 0 | `Put the eraser into the small box` | 62 | 18,600 |
| 1 | `Move the cup back to the upper-right corner` | 55 | 16,500 |
| 2 | `First put the eraser into the small box, then move the cup back to the upper-right corner` | 40 | 18,135 |

状态和动作顺序一致：

1. `shoulder_pan.pos`
2. `shoulder_lift.pos`
3. `elbow_flex.pos`
4. `wrist_flex.pos`
5. `wrist_roll.pos`
6. `gripper.pos`

当前数据没有状态、动作、episode、frame、timestamp 或 task 的空值，所有状态和动作行都是固定 6 维。

### 4.2 视频完整性现状

v3 聚合视频共 5 个文件，两路相机各自总帧数都等于 53,235：

- top：18,600 + 25,618 + 9,017 = 53,235
- wrist：35,100 + 18,135 = 53,235

已有的独立 v2.1 副本抽样显示，AV1 stream-copy 切分可以得到精确 episode 帧数，例如：

- episode 0：300 帧，10.0 秒；
- episode 156：438 帧，14.6 秒。

该已有副本位于其他开发目录，只用于只读交叉核验，不作为本项目训练输入。

### 4.3 硬件和软件现状

- GPU：1 x NVIDIA GeForce RTX 4090，运行时报告约 48 GiB 显存；
- Driver：570.133.20；
- `lerobot_flex`：Python 3.10.19；
- 项目隔离环境 PyTorch：2.7.1+cu128，CUDA 可用；
- 磁盘：`/data` 剩余约 4.1 TiB；
- host `lerobot_flex` 不依赖全局 uv；项目本地工具固定为 uv 0.11.28；
- 参考仓要求 Python 3.10、PyTorch 2.7.1，并通过 lockfile 指向 CUDA 12.8 wheel。

硬件满足官方“单卡 40 GiB+、默认只训练 projector + diffusion head”的最低条件，但双相机实际 batch 上限必须通过显存阶梯测试决定，不能直接假定 `global_batch_size=32` 一定可用。

### 4.4 当前基线差异

现有 PI0.5 配置使用：

- 2 路 480 x 640 图像，模型内处理为 224 x 224；
- 6 维 state/action；
- action chunk 50；
- batch size 16；
- 33,300 steps；
- bfloat16；
- action/state 使用 mean/std normalization。

N1.7 官方 SO100 方案使用 16 步 action horizon、臂相对动作和夹爪绝对动作。模型架构和动作处理器不同，所以“使用同样的数据”不应被误解为复制 PI 的 50 步 chunk 和 33,300 steps。

## 5. 目标目录设计

后续实现必须遵循下列布局：

```text
my_devs/gr00t_17/
├── doc/                         # 计划、技术文档、最终报告
├── reference/                   # 用户提供的只读参考源码和压缩包
├── workspace/                   # 可修改的 N1.7 工作副本
├── env/                         # 本地虚拟环境/本地工具，不写全局 conda
├── cache/
│   ├── huggingface/
│   ├── transformers/
│   ├── torch/
│   ├── uv/
│   ├── pip/
│   ├── triton/
│   └── pycache/
├── models/
│   └── GR00T-N1.7-3B/          # 基础模型的本地快照
├── data/
│   ├── source_manifests/        # 原数据只读哈希和清单
│   ├── derived_manifests/       # 转换数据发布后的完整哈希清单
│   ├── converted_v21/           # 完整 GR00T LeRobot v2.1 数据
│   └── splits/                  # 开发 train/val 清单或派生数据
├── configs/                     # modality 和实验配置
├── scripts/                     # 转换、检查、训练、评估、推理入口
├── tests/                       # 本项目定向测试
├── outputs/
│   ├── smoke/
│   ├── pilot/
│   ├── train/
│   ├── eval/
│   └── inference/
├── logs/
├── tmp/
└── reports/
```

禁止把 Hugging Face 默认缓存留在 `$HOME/.cache`，也禁止把训练输出写到仓库顶层 `outputs/`。

## 6. 分阶段实施计划

### 阶段 0：冻结输入和建立安全护栏

工作内容：

1. 生成原数据活跃文件清单，记录相对路径、大小、mtime 和 SHA-256。
2. 生成 PI checkpoint 文件清单和 SHA-256。
3. 记录参考 zip 和参考源码清单。
4. 编写路径守卫，所有写操作先做 `Path.resolve()` 和允许根目录校验。
5. 给转换器加入“源目录只读、目标目录必须为空、临时目录原子替换”的约束。
6. 所有删除操作只能针对 `my_devs/gr00t_17/tmp/` 或带有本项目 marker 的未完成目标。

已记录的审计基准：

- N1.7 release zip SHA-256：`bee284943b06d5bce1c28d415020db007ce031d515867bb1877df785ef8dcc4a`
- 原数据 `meta/info.json` SHA-256：`42c84e468bc4c9e361a5c29b84caf448538c80e36dffdef5d5e833bedacc9eb4`
- 原数据主 parquet SHA-256：`892c75c7980cedd744e737d94fbd620e03b309eed25301d7a289f6489900333b`
- 原数据最终 14 条 manifest aggregate SHA-256：`5c18fe2ecf334f451f1d6c5cd01b9c393cc7bf742bd5c26726e154bf595227e8`
- PI checkpoint 最终 7 条 manifest aggregate SHA-256：`f85fc56ece26154920725faa43ed54a432e06f0ae5cddd80ca684e6e6f4cf846`
- reference 最终 302 条 manifest aggregate SHA-256：`bbf3a837d8b0f8cd3fdd885ef01de65e170399287be6b907050025e8b0140c6f`

阶段门禁：输入清单生成后再次校验一致；任何不一致立即停止。

### 阶段 1：建立完全本地的 N1.7 工作环境

工作内容：

1. 从 `reference/Isaac-GR00T-n1.7-release` 复制出 `workspace/Isaac-GR00T-n1.7`，不修改 reference。
2. 从 `lerobot_flex` 环境启动本地环境构建工具。
3. 将虚拟环境固定到 `my_devs/gr00t_17/env/`。
4. 使用参考仓 `uv.lock` 的锁定版本安装，优先 `uv sync --locked`。
5. 将所有下载和编译缓存重定向到本项目 `cache/`。
6. 保存 `pip freeze`、Python/PyTorch/CUDA/FFmpeg/TorchCodec 版本报告。
7. 做 import、CUDA、flash-attn、torchcodec 和 1 帧 AV1 解码检查。

必须设置或等价覆盖的环境变量：

```bash
GR00T17_ROOT=/data/cqy_workspace/flexible_lerobot/my_devs/gr00t_17
HF_HOME=$GR00T17_ROOT/cache/huggingface
HF_HUB_CACHE=$GR00T17_ROOT/cache/huggingface/hub
HF_DATASETS_CACHE=$GR00T17_ROOT/cache/huggingface/datasets
TRANSFORMERS_CACHE=$GR00T17_ROOT/cache/transformers
TORCH_HOME=$GR00T17_ROOT/cache/torch
UV_CACHE_DIR=$GR00T17_ROOT/cache/uv
PIP_CACHE_DIR=$GR00T17_ROOT/cache/pip
TRITON_CACHE_DIR=$GR00T17_ROOT/cache/triton
PYTHONPYCACHEPREFIX=$GR00T17_ROOT/cache/pycache
TMPDIR=$GR00T17_ROOT/tmp
WANDB_MODE=disabled
WANDB_DIR=$GR00T17_ROOT/logs/wandb
```

阶段门禁：环境外不产生新缓存；CPU/GPU import 和 AV1 随机帧解码全部通过。

### 阶段 2：下载并冻结 N1.7 基础模型

工作内容：

1. 将 `nvidia/GR00T-N1.7-3B` 下载到 `models/GR00T-N1.7-3B/.partial`。
2. 使用固定 revision，禁止训练时临时从 Hub 拉取漂移版本。
3. 下载完成后原子改名为正式目录。
4. 记录模型文件、大小、SHA-256、Hub revision 和许可证信息。
5. 离线模式加载 config、processor 和权重。
6. 禁止用字符串 Hub ID 直接启动正式训练，正式训练只接收本地绝对路径。

阶段门禁：断网/离线变量启用后仍可完整加载基础模型。

### 阶段 3：非破坏性转换 v3 数据到 GR00T LeRobot v2.1

工作内容：

1. 只读解析 v3 `info.json`、tasks parquet、episodes parquet、聚合 data parquet 和视频索引。
2. 在 `data/converted_v21/...partial` 写出每 episode 一个 parquet。
3. 写出 `meta/info.json`、`episodes.jsonl`、`tasks.jsonl` 和保留的 `stats.json`。
4. 按 episode 时间区间从 AV1 聚合视频无损切分为 314 个视频。
5. 新增 GR00T 必需的 `meta/modality.json`。
6. 生成相对臂动作所需的 `meta/relative_stats.json`。
7. 完成全量验证后将 `.partial` 原子改名为正式数据目录。
8. 转换前后重新校验原始数据清单，证明原数据未变化。
9. 在数据目录外冻结转换结果的全量 manifest，后续每次训练预检重新计算。

首选视频策略：保留 AV1 并使用 stream copy，因为 N1.7 自带 demo 也是 AV1，且已有相同数据的 v2.1 样本精确匹配帧数。若全量验证发现 seek、帧数或随机解码问题，才在目标目录内对全套视频统一重编码为 H.264，并同步修正 metadata；不得混用错误的 codec metadata。

阶段门禁详见第 7 节，任何一个 episode 不匹配都禁止训练。

### 阶段 4：建立 SO101 modality 配置与单样本验证

计划配置：

- video：当前时刻 `[0]`，键为 `top`、`wrist`；
- state：当前时刻 `[0]`，键为 `single_arm`、`gripper`；
- action：未来 `0..15`，共 16 步；
- `single_arm`：`RELATIVE + NON_EEF + DEFAULT`；
- `gripper`：`ABSOLUTE + NON_EEF + DEFAULT`；
- language：`annotation.human.task_description`；
- 不启用 sin/cos state embedding，因为当前值是旧 SO follower 的度制位置，不是弧度；
- 原 parquet 始终保存绝对目标，不预先改成相对值，相对变换由 N1.7 processor 完成。

必须做的单样本测试：

1. 读取 episode 0、62、117，覆盖三个 task。
2. 校验两路图像为 `uint8`、HWC、480 x 640 x 3。
3. 校验 state `(1, 5) + (1, 1)`。
4. 校验 action `(16, 5) + (16, 1)`。
5. 校验语言文本与 `task_index` 精确对应。
6. 校验 relative action 首步等于 action target 减当前 state。
7. 校验 processor 反变换恢复到原绝对关节目标，误差在浮点容差内。
8. 校验 episode 尾部不会跨 episode 采样；默认不 padding 时有效窗口为 `length - 15`。

157 个 episode 在 horizon 16 下共有 50,880 个有效训练窗口。

阶段门禁：三个任务的样本都完成正向和反向变换闭环。

### 阶段 5：数据加载与模型 smoke test

按以下顺序执行，不允许跳级：

1. dataset metadata-only 检查；
2. 单 episode parquet 加载；
3. 单相机随机帧加载；
4. 双相机同一 index 加载；
5. collator 单样本；
6. processor 单 batch；
7. 模型 forward；
8. 单 batch backward；
9. 从基础模型运行到 step 1，并保存完整可恢复 `checkpoint-1`；
10. 明确从 `checkpoint-1` 恢复到 step 2，并保存 `checkpoint-2`；
11. 比较可训练和冻结权重，复核 optimizer/scheduler/RNG 及全部只读输入 manifest。

实际采用两步而不是原计划的 5/20/25 steps，是因为每个完整可恢复 checkpoint 实测为 23.31 GB；
两步已经覆盖 forward/backward、warmup、真实参数更新、完整保存和跨进程恢复。峰值总显存为
37,852 MiB，step 1/2 terminal loss 分别为 1.4883127213 和 0.5696015954。

阶段门禁：loss 有限、梯度有限、checkpoint 可恢复、无根目录外写入。

### 阶段 6：显存阶梯和训练参数定标

单卡双相机按 batch 1、2、4、8、16、32 逐级尝试，某一级 OOM 后停止继续放大。每一级只跑足够稳定的短测试，并记录：

- allocated/reserved/peak VRAM；
- samples/s 和 steps/s；
- dataloader wait；
- CPU RAM；
- 是否发生 TorchCodec/worker 错误；
- 是否有 NaN/Inf。

注意：N1.7 代码把 `global_batch_size / num_gpus` 直接传给 `per_device_train_batch_size`。如果同时设置 `gradient_accumulation_steps > 1`，实际有效 batch 会继续乘以累积步数。训练脚本必须打印三者：

```text
per_device_batch
gradient_accumulation_steps
effective_batch = per_device_batch * num_gpus * gradient_accumulation_steps
```

首轮参数候选：

| 参数 | 首选值 | 调整规则 |
| --- | --- | --- |
| tune_llm | false | 单卡不启用 |
| tune_visual | false | 单卡不启用 |
| tune_projector | true | 保持官方默认 |
| tune_diffusion_model | true | 保持官方默认 |
| dtype | bf16 | 官方默认 |
| action horizon | 16 | SO100 官方基线 |
| learning rate | 1e-4 | 若不稳定降到 5e-5 |
| state dropout | 0.2 | 先用 CLI 默认；再评估 0.0/0.1 |
| warmup ratio | 0.05 | 官方默认 |
| weight decay | 1e-5 | 官方默认 |
| dataloader workers | 2 起步 | 按 RAM/稳定性增到 4 |
| W&B | false | 使用本地日志 |
| save_only_model | false | 保留可恢复状态 |

阶段门禁：选定不会紧贴显存上限、可连续稳定运行的 batch，至少保留 10% 显存余量。

### 阶段 7：开发训练、评估与最终全量训练

#### 7.1 开发划分

先在本项目派生目录内按 task 分层、固定 seed 42 生成开发 train/validation 清单。建议 validation 约 10%，三个任务均有样本，split manifest 必须保存 episode ID，不能只保存随机 seed。

开发训练只用于选择：

- batch 和学习率；
- 训练步数；
- state dropout；
- 是否需要额外图像增强；
- checkpoint 选择。

#### 7.2 训练层级

| 层级 | 目的 | 建议 steps |
| --- | --- | ---: |
| smoke | 验证全链路 | 1、2 resume（已通过） |
| pilot | 观察 loss、吞吐和显存 | 100-200 |
| baseline | 官方量级基线 | 2,000 |
| extension | 仅在验证指标继续改善时 | 4,000 / 8,000 |

不直接复制 PI 的 33,300 steps。数据只有 157 个 episode，N1.7 仅训练 projector 和 diffusion head，过长训练可能过拟合。是否延长必须由 held-out open-loop 指标和真机结果决定。

#### 7.3 最终训练

开发参数冻结后，最终模型有两个明确选项：

1. 保留 validation episode，不参与最终训练，得到可持续比较的离线指标；
2. 使用全部 157 episode 重训生产模型，最终以真机评估为准。

默认先执行选项 1。只有 held-out 和真机基线通过后，再执行选项 2，并明确标注它不再拥有独立的同分布离线 holdout。

每个 run 使用不可复用的时间戳目录，保存：

- 完整命令和环境变量快照；
- 输入数据 manifest hash；
- 基础模型 revision/hash；
- modality config hash；
- git diff 或工作副本 manifest；
- stdout/stderr；
- GPU/RAM 监控；
- checkpoint；
- loss/throughput 汇总；
- 结束状态和失败原因。

### 阶段 8：离线评估

评估不只看训练 loss，至少包含：

1. 三个任务分别报告 action MSE/MAE；
2. 前五个臂关节和夹爪分开报告；
3. 每个 horizon step 分别报告误差，检查远期动作退化；
4. 相对动作空间和反变换后的绝对动作空间都报告；
5. 预测范围、q01/q99 越界率和速度/加速度突变率；
6. 多个 episode 的 GT vs prediction 曲线；
7. 基础模型 zero-shot、N1.7 finetune checkpoint 和现有 PI checkpoint 做同任务定性/定量对照；
8. checkpoint 选择使用验证结果，不使用训练集最低 loss 作为唯一标准。

阶段门禁：三类任务都不能只靠总体平均数掩盖失败；组合任务必须单独报告。

### 阶段 9：真机推理准备与安全验证

本阶段在训练完成后实施，仍然只在 `my_devs/gr00t_17/` 增加代码和产物。

推理契约：

- top camera -> `top`；
- wrist camera -> `wrist`；
- state 顺序严格为训练时 6 维顺序；
- 模型输出的前五维相对动作经 processor 恢复为绝对目标；
- gripper 保持绝对目标；
- 控制频率目标 30 Hz，但必须根据实际 N1.7 推理延迟设计 action chunk 消费策略；
- 禁止把相对输出直接发送给 `robot.send_action()`。

安全顺序：

1. 离线 checkpoint load；
2. 录制 observation 的离线 inference；
3. 无电机 dry-run，只打印目标；
4. 单步、低速、带急停；
5. 小关节限幅；
6. 短时单任务；
7. 扩展到三个任务；
8. 最后才运行 120 秒任务。

## 7. 数据转换验收矩阵

### 7.1 结构验收

- `meta/info.json`：`codebase_version == v2.1`；
- `meta/episodes.jsonl`：157 行，episode 连续 0..156；
- `meta/tasks.jsonl`：3 行，文本和索引完全一致；
- `meta/modality.json`：state/action/video/annotation 完整；
- `data/chunk-000/episode_XXXXXX.parquet`：157 个；
- 两路视频：各 157 个，共 314 个；
- `meta/stats.json` 存在且维度正确；
- `meta/relative_stats.json` 中 `single_arm` 为 `(16, 5)` 的逐 horizon 统计。

### 7.2 内容验收

- parquet 总行数 53,235；
- 每个 parquet 行数等于对应 episode length；
- 每行 state/action 为 6 维 float32；
- `episode_index`、`frame_index`、`index`、`task_index` 连续且合法；
- timestamp 单 episode 从 0 开始、单调、步长约 1/30 秒；
- v3 与 v2.1 对应行的 state/action/timestamp/task 完全一致；
- 三个 task 的 frame/episode 数与原数据一致。

### 7.3 视频验收

- 每路每 episode 的帧数等于 parquet 行数；
- FPS 30、640 x 480、无音频；
- 两路相机同 episode 帧数相等；
- 首/中/尾帧可解码；
- 随机抽样至少覆盖每个聚合源文件边界前后；
- PyAV、FFprobe 和训练所用 TorchCodec 至少各完成一轮解码检查；
- 不允许黑帧、0 字节、缺 moov/索引或 seek 错误。

### 7.4 不可破坏验收

- 转换前后重新计算原始 10 个活跃文件哈希；
- PI checkpoint 7 个文件哈希不变；
- reference zip 和源码 manifest 不变；
- `git status` 不出现原数据、原权重或 reference 内的新修改；
- 根目录外无新增本项目缓存或输出。

## 8. 失败处理与恢复策略

### 数据转换失败

- 只删除带本项目 marker 的 `.partial` 目录；
- 不清理任何源路径；
- 保存失败 episode、FFmpeg stderr 和已完成清单；
- 修复后重新创建新的 `.partial`，不在不完整正式目录上续写。

### 训练 OOM

- 先减 batch；
- 再降低 dataloader workers/预加载；
- 再考虑梯度累积；
- 不启用 `tune_llm` 或 `tune_visual`；
- 不把 OOM 后的残缺 checkpoint 当成可恢复点。

### 训练中断

- 仅从包含 optimizer/scheduler/RNG state 的完整 checkpoint 恢复；
- 恢复前核对数据、基础模型、配置 hash；
- 新日志明确记录 parent checkpoint；
- 不覆盖旧 run。

### loss 异常

- 先检查 language 映射、relative stats shape 和 action/state 顺序；
- 再检查 batch 内图像 dtype/range；
- 再检查学习率和 outlier；
- 不通过“继续训练更久”掩盖数据错误。

## 9. 预期交付物

1. 非破坏性 v3 -> GR00T v2.1 转换器；
2. 全量数据验证器及 JSON 报告；
3. SO101 N1.7 modality config；
4. 环境安装与缓存隔离脚本；
5. 基础模型下载/校验脚本；
6. smoke、pilot、正式训练脚本；
7. GPU/RAM/训练日志监控；
8. checkpoint 恢复检查；
9. open-loop 评估脚本和图表；
10. 真机推理入口及 dry-run/safety guard；
11. 工作报告、训练报告、评估报告和使用说明；
12. 原数据、原权重、reference 前后哈希证明。

## 10. 下一步执行顺序

当前执行状态和后续顺序：

1. 阶段 0-6：输入保护、隔离环境、模型、数据转换、modality、训练 smoke 和显存定标，已完成；
2. 阶段 7：micro batch 2、effective batch 8、10 epoch 正式训练和 checkpoint 验收，已完成；
3. 阶段 8：正式 checkpoint 的真机 server 加载、真实双相机/状态读取和无动作预测，已完成；
4. 阶段 9：1 秒、5 Hz、单步最大 0.25 的带动作 Smoke，等待现场安全确认；
5. 阶段 10：带动作 Smoke 通过后，按真机部署 Runbook 执行正式 120 秒任务。

任何阶段门禁未通过，都不能进入下一阶段。
