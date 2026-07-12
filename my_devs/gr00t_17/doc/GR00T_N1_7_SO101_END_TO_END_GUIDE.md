# GR00T N1.7 SO101 从数据、训练、真机 RTC 到 TensorRT 的完整工程记录

## 0. 文档定位

本文是 `my_devs/gr00t_17` 工作的总入口，目标是让后续维护者只阅读这一份文档，就能够回答以下问题：

1. 原始 PI0.5 训练链路、数据和真机接口是什么；
2. 为什么原始 LeRobot v3 数据不能直接训练 GR00T N1.7；
3. 如何在不修改原数据、原权重和参考代码的前提下建立隔离环境；
4. SO101 的 state、action、双相机和语言任务如何映射到 N1.7 modality；
5. 本次到底训练了哪些模块，哪些模块被冻结；
6. micro batch、gradient accumulation、effective batch、step 和 epoch 如何换算；
7. 两阶段 smoke train 为什么必要，如何证明恢复训练和参数冻结确实有效；
8. 10 epoch 正式训练使用了什么参数，最终 checkpoint 如何验收；
9. 为什么第一次真机部署出现“机械臂抖动但不运动”，如何定位到限制策略；
10. RTC 在 N1.7 模型内部和异步 Client 时间轴上分别做了什么；
11. TensorRT full pipeline 如何导出、构建、验证，为什么不能直接照搬参考 forward；
12. 当前推荐的 Server/Client 上机命令是什么；
13. 哪些文件应该进入 Git，哪些大文件必须只保留在本机。

本文记录的最终状态日期为 2026-07-12。各阶段的更细证据仍保存在同目录其他专题文档和
`my_devs/gr00t_17/reports/` 中，但本文给出完整主线、关键实现和实际结果。

## 1. 最终交付状态

当前链路已经完成：

- 原始 LeRobot v3 多任务数据审计和不可变哈希基线；
- 非破坏性转换为 GR00T 可读的 LeRobot v2.1 per-episode 数据；
- SO101 双相机、6 维 state/action、3 个语言任务的 modality 配置；
- 项目内独立 Python 环境、缓存、模型和依赖；
- FlashAttention 和 TorchCodec 本机兼容处理；
- 基础 N1.7 checkpoint 严格加载；
- 真实 batch 检查；
- 两阶段训练和 resume smoke；
- micro batch 2、gradient accumulation 4、effective batch 8 的 10 epoch 正式训练；
- 最终 `checkpoint-63600` 完整验收；
- 同步 Server/Client 真机推理；
- 双摄像头实拍检查；
- 机械臂动作限制问题定位；
- N1.7 原生 model-level RTC；
- 30 Hz 异步动作队列；
- 独立 RTC Server 和 Client；
- 7-engine TensorRT full pipeline；
- RTC-aware TensorRT action head 修正；
- PyTorch/TensorRT 普通动作与 RTC 动作数值对照；
- 真实设备零动作 30 Hz TensorRT RTC smoke。

最终训练 checkpoint 位于本机：

```text
my_devs/gr00t_17/outputs/formal/so101_n17_b2_e10_20260711/train/checkpoint-63600
```

最终 TensorRT engine 位于本机：

```text
my_devs/gr00t_17/artifacts/tensorrt/so101_n17_b1_bf16_full/engines
```

这两个目录都是生成物，不进入普通 Git 仓库。

## 2. 不可破坏边界

本项目把“不能伤害原始数据和权重”作为硬约束，而不是操作习惯。

### 2.1 只读输入

以下内容只允许读取和计算哈希，不允许覆盖、移动、重编码或原地转换：

- 原始数据：`datasets/desk_cleanup_v1/eraser_cup_multi_task`；
- 原 PI0.5 checkpoint；
- 用户提供的 `my_devs/gr00t_17/reference/`；
- 下载完成后的基础 GR00T N1.7 模型；
- 正式训练完成后的 `checkpoint-63600`。

### 2.2 唯一工作根目录

与本任务相关的环境、缓存、数据副本、模型、训练输出、TensorRT 产物、日志和报告全部放在：

```text
/data/cqy_workspace/flexible_lerobot/my_devs/gr00t_17
```

脚本通过 `require_path_within_root()` 和 Python `resolve()/relative_to()` 检查输出路径。即使调用者传入
`..`、符号链接或绝对路径，解析后的写入目标也必须仍位于该根目录。

### 2.3 不可变 manifest

输入不是只靠“没有执行删除命令”来保护，而是建立完整文件清单：

| 输入 | 条目数 | aggregate SHA-256 |
| --- | ---: | --- |
| 原始 LeRobot 数据 | 14 | `5c18fe2ecf334f451f1d6c5cd01b9c393cc7bf742bd5c26726e154bf595227e8` |
| PI0.5 checkpoint | 7 | `f85fc56ece26154920725faa43ed54a432e06f0ae5cddd80ca684e6e6f4cf846` |
| GR00T reference | 302 | `bbf3a837d8b0f8cd3fdd885ef01de65e170399287be6b907050025e8b0140c6f` |
| 转换后训练数据 | 478 | `fde7fe09a84af986ddabe02e0d8d6e0ea5dec9ee17181acfdfb347e0f70b4815` |

正式训练前、smoke 后、真机开发后以及 TensorRT 开发后都重新计算清单。任何 parquet、视频、metadata、
参考代码或 PI 权重变化都会让 preflight 失败。

## 3. 目录职责

```text
my_devs/gr00t_17/
├── configs/                  # SO101 modality 和本机 calibration
├── data/                     # 转换数据与输入 manifest，本体不进 Git
├── doc/                      # 设计、执行、部署和本文档
├── scripts/                  # 转换、训练、验证、RTC、TensorRT、真机入口
├── tests/                    # 快速单元测试
├── reports/                  # 小型验证证据，可以选择进入 Git
├── env/                      # 项目内 Python venv，不进 Git
├── cache/                    # 所有依赖和编译缓存，不进 Git
├── models/                   # 基础模型和运行时资产，不进 Git
├── outputs/                  # 训练、smoke、真机输出，不进 Git
├── artifacts/                # ONNX 和 TensorRT engine，不进 Git
├── workspace/                # 可修改的上游代码副本，不进 Git
└── reference/                # 用户提供的只读参考代码，不进 Git
```

Git 只需要保存能够重建流程的脚本、配置、测试、文档和小型清单。模型、数据、checkpoint、optimizer、
ONNX、engine、虚拟环境和缓存都必须在目标机器重新准备。

## 4. 原始系统背景

### 4.1 原 PI0.5 训练

原训练入口为 `my_devs/train/pi/so101/easy_train.sh`。它使用同一个桌面清理多任务数据集，输入包括：

- top camera；
- wrist camera；
- SO101 6 维当前位置；
- 英文任务文本；
- 6 维绝对关节目标。

PI0.5 的动作 chunk 是 50 步，而本次 N1.7 配置选择 16 步。不能把 PI checkpoint 直接当 N1.7
checkpoint 使用；PI 链路的价值是提供已验证的数据、相机、机器人端口、任务文本和真机控制契约。

### 4.2 原真机接口

原推理命令证明了以下外部契约：

```text
robot port = /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00
top camera = /dev/video4
wrist camera = /dev/video6
task = Put the eraser into the small box
```

N1.7 适配保持同样的 SO follower action feature 顺序：

```text
shoulder_pan.pos
shoulder_lift.pos
elbow_flex.pos
wrist_flex.pos
wrist_roll.pos
gripper.pos
```

## 5. 原始数据审计

### 5.1 数据规模

| 项目 | 数值 |
| --- | ---: |
| episodes | 157 |
| frames | 53,235 |
| FPS | 30 |
| top frames | 53,235 |
| wrist frames | 53,235 |
| task 0 frames | 18,600 |
| task 1 frames | 16,500 |
| task 2 frames | 18,135 |

任务文本为：

```text
Put the eraser into the small box
Move the cup back to the upper-right corner
First put the eraser into the small box, then move the cup back to the upper-right corner
```

### 5.2 parquet schema

关键列：

```text
action: list<float32>[6]
observation.state: list<float32>[6]
timestamp: float32
frame_index: int64
episode_index: int64
index: int64
task_index: int64
```

state 和 action 都是旧 SO follower 位置空间。前五维近似角度制位置，夹爪是位置值，不是弧度、速度或力矩。

### 5.3 视频格式

原视频为 AV1、`yuv420p`、640 x 480、30 FPS、无音频。AV1 本身不是 GR00T 的阻塞项，因为参考
SO101 数据也使用 AV1。真正需要验证的是目标 Python/TorchCodec/FFmpeg 组合能否正确解码。

## 6. 为什么必须转换数据格式

原数据是 LeRobot v3 聚合布局：多个 episode 共用 parquet 和视频文件，episode 边界保存在 metadata
parquet 中。GR00T N1.7 的 `LeRobotEpisodeLoader` 要求 LeRobot v2.1 per-episode 布局：

```text
meta/info.json
meta/episodes.jsonl
meta/tasks.jsonl
meta/modality.json
meta/stats.json
meta/relative_stats.json
data/chunk-000/episode_000000.parquet
videos/chunk-000/<video_key>/episode_000000.mp4
```

因此直接把原 v3 路径传给 N1.7 会在 metadata 和文件模板层面失败。转换不是为了改数值，而是为了改变
物理布局和 metadata 表达。

## 7. 非破坏性 v3 到 v2.1 转换

入口：`scripts/convert_v3_to_gr00t_v21.py`。

```bash
cd /data/cqy_workspace/flexible_lerobot

/home/cqy/miniconda3/bin/conda run --no-capture-output -n lerobot_flex \
  python my_devs/gr00t_17/scripts/convert_v3_to_gr00t_v21.py \
    --source-root datasets/desk_cleanup_v1/eraser_cup_multi_task \
    --output-root my_devs/gr00t_17/data/converted_v21/desk_cleanup_v1/eraser_cup_multi_task \
    --allowed-write-root my_devs/gr00t_17 \
    --modality-json my_devs/gr00t_17/configs/modality.json \
    --report-path my_devs/gr00t_17/reports/data_conversion.json
```

### 7.1 转换原子性

脚本先写 `.<dataset-name>.partial-<pid>`。只有所有 parquet、视频、metadata 和统计通过检查后才 rename
成正式目录。失败时保留 partial 供检查，不覆盖已有正式目录，更不修改 source。

### 7.2 parquet 转换

转换器根据 v3 episode metadata 中的 `dataset_from_index` 和 `dataset_to_index` 从聚合 parquet 切出
每个 episode。输出保持原始列、dtype、值和顺序。明确不做：

- 不修改状态或动作单位；
- 不把绝对 action 原地改成 relative；
- 不重排关节；
- 不修改 timestamp；
- 不把模型 padding 写回数据；
- 不向原数据添加列。

### 7.3 视频转换

视频按 metadata 的 `from_timestamp` 和 `to_timestamp` 用 FFmpeg stream copy 切分。之后全量 FFprobe
验证帧数、编码、分辨率和 FPS，并用 TorchCodec 对每个 episode 的首、中、尾帧与源聚合视频逐像素比较。

最终结果：157 个 parquet、314 个视频、53,235 行、942 个抽样帧逐像素相等。

### 7.4 relative stats

数据文件仍保存绝对 action。针对前五维 arm，转换器计算：

```text
relative_action[t, offset] = action[t + offset] - state[t]
```

对 horizon 16 的每个 offset 分别统计 min、max、q01、q99、mean、std，输出形状为 `(16, 5)`。
夹爪保持 absolute，不进入 arm relative stats。

## 8. SO101 Modality 设计

### 8.1 数据字段映射

```text
state.single_arm   = observation.state[0:5]
state.gripper      = observation.state[5:6]
action.single_arm  = action[0:5]
action.gripper     = action[5:6]
video.top          = observation.images.top
video.wrist        = observation.images.wrist
language           = task_index -> tasks.jsonl
```

### 8.2 模型时序

```text
video delta indices    = [0]
state delta indices    = [0]
action delta indices   = [0..15]
language delta indices = [0]
```

30 Hz 下 16 步动作覆盖约 `16 / 30 = 0.533` 秒。

### 8.3 动作表示

```text
single_arm: RELATIVE, NON_EEF
gripper:    ABSOLUTE, NON_EEF
```

训练数据中的 absolute arm target 由 processor 根据当前 state 转为 relative 并归一化。推理结果由同一个
checkpoint processor 反归一化，再与当前 state 组合成绝对关节目标。不能把模型内部 132 维 padded
action 或归一化 relative action 直接发给机器人。

## 9. 隔离环境

### 9.1 双层环境

仓库规范要求从 `lerobot_flex` conda 环境启动；GR00T 的实际依赖安装到项目内：

```text
my_devs/gr00t_17/env/gr00t_n17
```

典型调用：

```bash
/home/cqy/miniconda3/bin/conda run --no-capture-output -n lerobot_flex \
  bash my_devs/gr00t_17/scripts/<script>.sh
```

### 9.2 缓存隔离

`scripts/common.sh` 把 HF、Transformers、Torch、uv、pip、Ruff、Triton、CUDA、Numba、Matplotlib、临时目录
和 W&B 目录全部重定向到 `my_devs/gr00t_17`。训练默认开启 HF/Transformers offline，禁用 W&B 和 telemetry。

### 9.3 最终关键版本

| 组件 | 版本 |
| --- | --- |
| Python | 3.10.19 |
| PyTorch | 2.7.1+cu128 |
| Transformers | 4.57.3 |
| FlashAttention | 2.7.4.post1 |
| TorchCodec | 0.4.0 |
| ONNX | 1.20.1 |
| TensorRT | 10.15.1.29 |
| GPU | NVIDIA GeForce RTX 4090, SM 8.9, 49,140 MiB |

### 9.4 FlashAttention 处理

上游 wheel 与本机 glibc/CXX11 ABI 不匹配，所以在项目内从 `flash-attn==2.7.4.post1` sdist 构建 wheel。
构建后完成 BF16 head dimension 64/128、causal/non-causal forward 和 backward 验证。

### 9.5 TorchCodec/libffi 处理

外层 conda 的 RPATH 中存在名字为 `libffi.so.7`、实际 ABI 不匹配的链接。`common.sh` 显式预加载真实系统
`/lib/x86_64-linux-gnu/libffi.so.7`，最终完成 AV1 视频解码。

## 10. 模型和运行时资产

### 10.1 基础模型

```text
repo: nvidia/GR00T-N1.7-3B
revision: 2fc962b973bccdd5d8ce4f67cc63b264d6886495
path: my_devs/gr00t_17/models/GR00T-N1.7-3B
```

下载脚本要求 pinned revision，下载到 partial 目录，检查必需文件，计算每个文件 SHA-256，最后原子发布。

### 10.2 Qwen3-VL runtime 资产

N1.7 checkpoint 包含 VLM 权重，但构造 processor 和架构仍需要 config、tokenizer 和 image processor。
原配置指向受许可的 Cosmos 仓库。本项目证明 pinned Cosmos 和公开 pinned Qwen3-VL 对应的 9 个 runtime
文件具有相同 blob ID 和 size，只下载这些公开、逐字节等价的运行时文件，不下载 Qwen 权重。

### 10.3 严格加载

```text
total parameters: 3,144,016,000
missing keys: 0
unexpected keys: 0
mismatched keys: 0
attention implementation: flash_attention_2
```

另外从 safetensors 读取 vision patch、LLM q-proj、token embedding 和 action decoder 样本张量，与加载后
state dict 做 `torch.equal`，确保不是“能加载但权重错位”。

## 11. 真实训练 Batch

| tensor | shape | dtype |
| --- | --- | --- |
| input IDs | `(1, 157)` | int64 |
| pixel values | `(512, 1536)` | float32 |
| image grid | `(2, 3)` | int64 |
| state | `(1, 1, 132)` | float32 |
| action | `(1, 40, 132)` | float32 |
| action mask | `(1, 40, 132)` | float64 |
| embodiment ID | `(1,)` | int64 |

132 和 40 是 N1.7 最大 embodiment/action padding，不代表 SO101 真有 132 个关节或执行 40 步。SO101
有效 action 是 16 x 6，共 96 个 active mask value。processor 只解码前 16 步、前 6 个有效维度。

## 12. 训练哪些参数

正式配置：

```text
tune_llm             = false
tune_visual          = false
tune_projector       = true
tune_diffusion_model = true
tune_vlln             = true
```

- LLM 冻结；
- ViT/visual backbone 冻结；
- 视觉语言到动作空间的 projector 训练；
- DiT/diffusion action head 训练；
- VLLN/self-attention action-side 模块训练。

数据只有 157 episodes，且单张 4090 不适合 3B 全量训练。任务变化主要在 embodiment 和动作映射，冻结
backbone 也更容易验证基础视觉语言能力没有被破坏。

Smoke 中冻结 LLM q-proj 从基础权重到 step 2 最大差为 0；可训练 action decoder 从 step 1 到 step 2
最大差为 `0.0001001358`，证明冻结和训练策略实际生效。

## 13. Batch、Step 和 Epoch 的精确换算

单卡上：

```text
effective_batch = GLOBAL_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
```

本次正式训练：

```text
GLOBAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
effective_batch = 8
```

有效 horizon-16 windows 为 50,880。10 个 nominal sample epoch 需要：

```text
optimizer_steps = 50,880 * 10 / 8 = 63,600
```

若以后改 batch，保持相同样本预算：

```text
max_steps = valid_windows * epochs / effective_batch
```

例如 effective batch 16 时，10 epoch 是 31,800 steps。不能在增大 accumulation 后仍保留 63,600 steps，
否则样本预算会翻倍。

## 14. Smoke Train

```bash
/home/cqy/miniconda3/bin/conda run --no-capture-output -n lerobot_flex \
  bash my_devs/gr00t_17/scripts/run_smoke_train.sh
```

Smoke 分两阶段：

1. 从基础 N1.7 运行到 step 1，保存完整 `checkpoint-1`；
2. 从 `checkpoint-1` 恢复，运行到 step 2，保存 `checkpoint-2`。

只跑 forward 不能证明 optimizer、scheduler、RNG、checkpoint 恢复和参数冻结正确。第一个 warmup step 的
learning rate 还可能为 0，因此必须运行第二步并比较权重。

| 阶段 | global step | train loss | runtime |
| --- | ---: | ---: | ---: |
| stage 1 | 1 | 1.4883127213 | 45.4055 s |
| stage 2 resume | 2 | 0.5696015954 | 105.6169 s |

两个 checkpoint 都包含 optimizer、scheduler 和 RNG。峰值显存约 37,852 MiB。

## 15. 正式训练

最终实际命令等价于：

```bash
cd /data/cqy_workspace/flexible_lerobot

RUN_ID=so101_n17_b2_e10_20260711 \
MAX_STEPS=63600 \
GLOBAL_BATCH_SIZE=2 \
GRADIENT_ACCUMULATION_STEPS=4 \
SAVE_STEPS=6360 \
  /home/cqy/miniconda3/bin/conda run --no-capture-output -n lerobot_flex \
  bash my_devs/gr00t_17/scripts/run_train.sh
```

固定优化参数：

```text
learning rate = 1e-4
weight decay = 1e-5
warmup ratio = 0.05
state dropout = 0.2
precision = BF16 + TF32
W&B = disabled
save_only_model = false
```

训练结果：

```text
global_step = 63600
nominal sample epochs = 10.0
train_runtime = 32231.7025 s (约 8 小时 57 分)
train_samples_per_second = 15.786
train_steps_per_second = 1.973
train_loss = 0.0691617722
final logged loss = 0.0358
final grad_norm = 1.00798
```

最终模型权重为 3 个 shard，共 10,343,115,064 bytes。optimizer state 约 12.96 GB，另有 processor、
statistics、scheduler 和 RNG。

### 15.1 恢复训练

```bash
RUN_ID=<existing_run_id> RESUME=1 \
  /home/cqy/miniconda3/bin/conda run --no-capture-output -n lerobot_flex \
  bash my_devs/gr00t_17/scripts/run_train.sh
```

preflight 会寻找最高编号且有 `trainer_state.json` 的 checkpoint，并重新校验所有输入。不要手工把不完整目录
改名为 checkpoint。

## 16. 训练前门禁和最终验收

`scripts/preflight.py` 检查外层 conda、项目内 Python、离线模式、必需报告、输入 manifest、转换数据、模型
哈希、内部 symlink、shard 数量、输出目录、resume 状态、CUDA、FlashAttention、TorchCodec 和至少
40,000 MiB 空闲显存。

`reports/formal_training_validation.json` 重新验证 global step 63,600、batch/epoch 预算、trainable/frozen
配置、有限 loss/grad norm、模型 shards、optimizer/scheduler/RNG、processor/statistics 和真实 Server 加载。

## 17. 第一版同步真机推理

最初实现是同步 ZeroMQ Server/Client：

```text
读取 observation
    -> 请求 Server
    -> 等待模型完整推理
    -> 收到 16 步 chunk
    -> 执行前 8 步
    -> 再次读取 observation
```

这能验证模型和硬件，但推理期间机器人控制循环暂停，因此不是严格 30 Hz realtime。它仍然是重要的分层
验证工具，因为 RTC 出问题时可以回退到同步路径，区分“基础模型输出错误”和“异步时间轴错误”。

同步部署由以下文件组成：

```text
scripts/run_so101_policy_server.sh
scripts/so101_robot_client.py
scripts/run_so101_smoke.sh
scripts/run_so101_infer.sh
```

Client 对 observation 做严格检查：

- state 必须是有限的 6 维 float；
- top/wrist 必须是 `(480, 640, 3)` uint8；
- 图像不能近似全黑、全白或无方差；
- action 必须包含 `single_arm` 和 `gripper`；
- 解码后必须是 16 x 6 finite chunk；
- checkpoint 必须位于 `my_devs/gr00t_17`，且三个 model shard 完整。

## 18. 相机排障

第一次真机效果异常时，没有直接认定训练失败，而是先单独抓取：

- `/dev/video4` top；
- `/dev/video6` wrist。

两张图确认内容和映射正常。随后发现客户端连接相机后立即取第一帧，top 首帧亮度明显偏离训练分布。
增加 `--camera-warmup-s 2` 后，在约 2 秒内读取 58 帧，top mean 回到约 113，与训练/正常画面一致。

这一步说明：相机设备“能打开”不等于第一帧“可用于模型”。USB 摄像头曝光和白平衡需要 warmup。排障时
至少要检查：

```text
设备映射
RGB/BGR 顺序
shape 和 dtype
mean/std/min/max
实际保存图片
warmup 前后差异
```

## 19. 为什么机械臂最初只抖不动

最初使用：

```text
bounds_mode = dataset_minmax
max_command_delta = 1
max_relative_target = 1
```

实际机器人 elbow state 接近 99.8，而数据集 action max 约 96.659。模型即使预测了有意义的轨迹，也先被
dataset max 截到 96.659，再被每步 1 的限制二次截断。夹爪也持续触发 relative clamp。结果表现为：

- 日志不断出现 `Relative goal position magnitude had to be clamped`；
- 机械臂基本不沿轨迹走；
- 电机在相邻受限目标附近抖动。

逐步放开后，使用：

```text
bounds_mode = physical
max_command_delta = 200
max_relative_target = 200
```

机械臂开始沿正确方向运动。这里的 200 对 SO101 标定域而言等价于关闭额外逐步 relative limit，但仍保留：

- NaN/Inf 拒绝；
- arm 物理标定域 `[-100, 100]`；
- gripper 物理标定域 `[0, 100]`；
- 用户可直接急停。

因此根因是部署限制与当前机器人姿态/数据边界不匹配，不是摄像头坏掉，也不能仅凭这个现象判定模型训练
无效。以后正式上机保持 physical/200/200，不再回到 dataset_minmax/1/1。

## 20. N1.7 原生 RTC 原理

### 20.1 模型内部能力

N1.7 action head 的 `get_action_with_features()` 已包含 RTC 分支。当 `action_input` 中存在上一动作块时：

1. 生成当前 40 步 padded action noise；
2. 用上一条有效 16 步动作块的尾部覆盖当前 overlap 前缀；
3. 将推理延迟期间必然已经执行的 frozen steps 的 velocity strength 设为 0；
4. 对其余 overlap 使用指数 ramp，从低重采样强度渐进到正常强度；
5. 在 4 个 flow-matching denoising step 中使用 `velocity * velocity_strength` 更新。

核心参数：

```text
action_horizon = 16
rtc_advance_steps = 8
rtc_overlap_steps = 16 - 8 = 8
rtc_frozen_steps = 2
rtc_ramp_rate = 2.0
```

### 20.2 Ramp 数学

令 overlap 为 `O`、frozen 为 `F`，中间可重采样步数 `M = O - F`。对 `M + 2` 个均匀点：

```text
t = linspace(0, 1, M + 2)
ramp = 1 - exp(-rate * t)
ramp = ramp / ramp[-1]
```

去掉开头 0 和末尾 1，只把中间 `M` 个值赋给 overlap 的非 frozen 区。这样不会在旧块和新块边界突然从
完全冻结跳到完全重采样。

### 20.3 为什么保存上一块物理动作

Server 不直接保存上一块归一化 tensor，而是保存 processor 解码后的物理动作。下一次请求时把上一物理动作
和新的当前 state 一起交回 processor。这样 arm relative action 会相对于最新 state 重新编码。

若直接复用上一时刻的 relative tensor，它仍以旧 state 为参考，机器人已经移动后会产生错误的绝对目标。

## 21. RTC Server 实现

文件：`scripts/so101_rtc_policy_server.py`。

Server 每个会话保存：

```text
previous physical action chunk
request index
backend type
TensorRT mode
```

第一次请求没有历史，因此执行普通 action generation。后续请求：

- 验证 batch size 必须为 1；
- 将上一动作块填入 `VLAStepData.actions`；
- processor 根据当前 state 重新归一化；
- 构造 RTC model options；
- 调用 PyTorch 或 TensorRT action head；
- 解码前 16 步 x 6 维物理动作；
- 更新会话历史。

Client 每次新 run 都先调用 reset，因此同一个常驻 Server 可以顺序服务多次任务，不会继承上次动作块。

Server 的 `ping` 不只返回“进程活着”，还返回：

```text
inference_backend = pytorch | tensorrt
trt_mode = n17_full_pipeline | null
```

这用于防止 Client 连到错误后端。

## 22. 异步 RTC Client

文件：`scripts/so101_rtc_robot_client.py`。

### 22.1 两条执行路径

```text
主控制线程:
  30 Hz tick -> 从时间轴取动作 -> send_action -> 更新全局 step

推理线程:
  抓取最新 observation -> ZeroMQ 请求 -> 等待 Server -> 返回新 chunk
```

ZeroMQ socket 不跨线程共享。每个异步 worker 创建并拥有自己的短连接，避免 ZMQ thread-affinity 问题。

### 22.2 时间轴合并

假设上一 chunk 覆盖全局 step 0 到 15：

```text
step 0: 取得第一块，开始执行
step 8: 发出下一次请求，但继续执行旧块 step 8、9...
step 10: 新块返回，真实 delay = 10 - 8 = 2
         丢弃新块索引 0、1
         从全局 step 10 开始用新块索引 2
step 16: 发出下一次请求
```

如果返回时 delay 已经达到 16，整个新块都过期，客户端报 `stale`/queue underrun 并停止，不重复执行旧动作。

### 22.3 Frozen step 估计

Client 使用上一条稳定端到端 latency 估计下一请求的 frozen steps：

```text
frozen_steps = ceil(previous_latency * control_hz)
```

结果再 clamp 到 `[0, overlap]`。实际队列合并使用“推理期间真实经过了多少全局控制步”，不是仅依赖 wall-clock
估算。模型内部 frozen 和客户端真实丢弃分别解决生成连续性和时间轴过期问题。

### 22.4 Warmup

启动后先做一次不进入动作队列、不发送给电机的模型 warmup，然后 reset Server RTC 状态，再请求第一条正式
动作块。这样 CUDA kernel/engine 初始化延迟不会被误当成运行期 frozen steps。

### 22.5 PyTorch RTC 验收

```text
control steps = 90 / 3 s
achieved control rate = 30.0 Hz
queue underruns = 0
sent actions = 0
steady end-to-end inference = 58-68 ms
```

## 23. 独立 Server/Client 架构

本地统一使用 `127.0.0.1:5556`。

```text
SO101 + cameras
      |
      v
RTC Client (lerobot_flex, motor/camera ownership)
      |
      | ZeroMQ/msgpack on localhost
      v
RTC Policy Server (project GR00T venv, GPU/model/engines ownership)
```

Server 常驻并持有模型/engine，Client 只持有摄像头、机器人和动作时间轴。模型只加载一次，Client 可以多次
启动。当前只允许 localhost，避免在尚未验证网络时延和认证的情况下直接远程控制机械臂。

## 24. TensorRT 参考实现审计

参考仓库提供：

```text
dit_only
action_head
full_pipeline / n17_full_pipeline
```

本项目采用 full pipeline。7 个 engine：

| Engine | 输入/作用 | 精度 |
| --- | --- | --- |
| ViT | 512 x 1536 image patches -> image/deepstack embeddings | FP32 export |
| LLM | dynamic seq x 2048 embeddings | BF16 |
| VL self-attention | dynamic seq x 2048 | BF16 |
| state encoder | 1 x 1 x 132 -> state feature | BF16 |
| action encoder | 1 x 40 x 132 -> action feature | BF16 |
| DiT | state/action + VL feature -> flow feature | BF16 |
| action decoder | flow feature -> 132 dim velocity | BF16 |

以下轻量逻辑仍保留 PyTorch：token embedding、masked scatter、RoPE index、VLLN、processor
normalization/decode、RTC overlap/frozen/ramp 和异步时间轴。这仍是参考实现定义的 full pipeline，不是把
整个 Python 进程编译为单一 engine。

## 25. 参考 TensorRT RTC 缺陷与修正

参考 `action_head_tensorrt_forward()` 虽然函数签名接受 `options`，但内部直接从随机 noise 开始，Euler
update 是：

```text
actions = actions + dt * pred_velocity
```

它没有读取上一动作、没有 overlap 初始化、没有 frozen prefix、没有 exponential ramp，也没有
`velocity_strength`。若原样接入，接口仍会运行，但 RTC 会静默退化成普通 chunking。

项目文件 `scripts/so101_rtc_trt.py` 在加载参考 7 个 engine 后重新绑定 action head forward，恢复：

```text
actions[:overlap] = previous[input_horizon-overlap:input_horizon]
velocity_strength[:frozen] = 0
velocity_strength[frozen:overlap] = exponential_ramp
actions += dt * velocity * velocity_strength
```

单元测试还直接验证 previous tail、frozen prefix、ramp 区间和非法参数拒绝。

## 26. TensorRT 构建

```bash
cd /data/cqy_workspace/flexible_lerobot

/home/cqy/miniconda3/bin/conda run --no-capture-output -n lerobot_flex \
  bash my_devs/gr00t_17/scripts/build_so101_trt.sh
```

固定参数：

```text
checkpoint = checkpoint-63600
dataset = converted SO101 v2.1
batch size = 1
precision = BF16
ViT export precision = FP32
builder workspace = 8192 MiB
export mode = full_pipeline
steps = export,build,verify
```

产物：

```text
artifacts/tensorrt/so101_n17_b1_bf16_full/onnx
artifacts/tensorrt/so101_n17_b1_bf16_full/engines
```

实测 export 49 秒，build 68 秒，verify 12 秒；ONNX 和 engines 分别约 6.1 GiB。Engine 与 GPU compute
capability、TensorRT、CUDA 版本绑定。换 GPU 或升级主版本后必须重建。

## 27. TensorRT 数值验证

### 27.1 参考 full-pipeline

```text
ViT cosine = 0.997885
backbone cosine = 0.999951
final action cosine = 0.999999
final action L1 = 0.076156
final action Linf = 0.361862
```

### 27.2 RTC 专项

`scripts/verify_so101_rtc_trt.py` 在同一 observation、同一随机种子上依次运行 PyTorch 首块、PyTorch RTC
第二块，再原地加载 TensorRT 并重复同一序列。

```text
initial chunk cosine = 0.9999990463
initial L1 = 0.0761561
initial Linf = 0.3618622

RTC chunk cosine = 0.9999990463
RTC L1 = 0.0625477
RTC Linf = 0.4858780
```

报告保存 7 个 engine 的 size 和 SHA-256：`reports/so101_rtc_trt_verification.json`。

## 28. TensorRT 实时效果

真实相机和机器人状态、零电机命令的 3 秒 dry-run：

```text
backend = tensorrt
trt_mode = n17_full_pipeline
control steps = 90
control rate = 30.0 Hz
queue underruns = 0
sent actions = 0
warmup = 147 ms
first formal request = 35 ms
steady model inference = about 29 ms
steady end-to-end = about 35-47 ms
```

对比 PyTorch steady model inference 约 50 ms，TensorRT 降到约 29 ms。RTC buffer 仍按实际控制步计算，
本机稳定表现为约 2 步 delay。

## 29. 后端握手

TensorRT 接入后增加了显式 backend handshake。正式 Client 传入：

```text
--expected-backend tensorrt
```

如果 5556 仍是旧 PyTorch Server，ping 只返回普通 status 或明确返回 pytorch，Client 会在连接机器人动作循环前
失败。这样不会出现“命令能跑，但实际上测试的是旧后端”的假阳性。

## 30. 当前正式上机命令

### 30.1 终端一: TensorRT RTC Server

先确保旧 PyTorch Server 已停止：

```bash
cd /data/cqy_workspace/flexible_lerobot

SERVER_HOST=127.0.0.1 SERVER_PORT=5556 \
  /home/cqy/miniconda3/bin/conda run --no-capture-output -n lerobot_flex \
  bash my_devs/gr00t_17/scripts/run_so101_rtc_policy_server.sh
```

必须看到：

```text
[RTC SERVER] ready on 127.0.0.1:5556 backend=tensorrt
```

Server 加载 checkpoint 后再反序列化 7 个 engine，删除被 TensorRT 替代的 PyTorch ViT、LLM layers 和 action
modules，以释放显存。第一次启动看到每个 engine 的 input/output binding 属于正常日志。

### 30.2 终端二: SO101 Client

```bash
cd /data/cqy_workspace/flexible_lerobot

RUN_ID=so101_n17_rtc_trt_$(date +%Y%m%d_%H%M%S) \
SERVER_HOST=127.0.0.1 SERVER_PORT=5556 \
  /home/cqy/miniconda3/bin/conda run --no-capture-output -n lerobot_flex \
  bash my_devs/gr00t_17/scripts/run_so101_rtc_client.sh \
    --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00 \
    --top-cam /dev/video4 \
    --wrist-cam /dev/video6 \
    --task "Put the eraser into the small box"
```

Client 内部固定：

```text
run time = 120 s
control rate = 30 Hz
execution horizon = 8
bounds = physical
max command delta = 200
robot max relative target = 200
camera warmup = 2 s
request timeout = 2 s
expected backend = tensorrt
```

Client 退出后 Server 继续运行。再次启动 Client 时会自动 reset RTC 会话。Server 用 Ctrl+C 停止。

### 30.3 单脚本组合入口

需要一次性启动和退出 Server 时：

```bash
RUN_ID=so101_n17_rtc_trt_$(date +%Y%m%d_%H%M%S) \
  /home/cqy/miniconda3/bin/conda run --no-capture-output -n lerobot_flex \
  bash my_devs/gr00t_17/scripts/run_so101_rtc_infer.sh \
    --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00 \
    --top-cam /dev/video4 \
    --wrist-cam /dev/video6 \
    --task "Put the eraser into the small box"
```

组合入口内部也默认 TensorRT，并在退出时停止它启动的 Server。

### 30.4 诊断性 PyTorch 回退

只在比较后端或 TensorRT 故障诊断时使用：

```bash
INFERENCE_BACKEND=pytorch SERVER_HOST=127.0.0.1 SERVER_PORT=5556 \
  /home/cqy/miniconda3/bin/conda run --no-capture-output -n lerobot_flex \
  bash my_devs/gr00t_17/scripts/run_so101_rtc_policy_server.sh
```

正式 Client 会拒绝该后端。若要做只读诊断，应直接调用 RTC Client 的 `--expected-backend pytorch`，不要修改
正式 wrapper 的 TensorRT 要求。

## 31. 从新机器复现的推荐顺序

以下是流程顺序，不代表模型和数据能从 Git 自动获得。

### 31.1 准备本地输入

```text
1. 放置只读 reference
2. 放置原始 LeRobot v3 数据
3. 放置或下载原 PI checkpoint（若需要做基线完整性对照）
4. 建立 source manifests
5. 准备可修改 workspace 副本
```

不要从 Git 期望获得视频、parquet、N1.7 权重或 TensorRT engine。

### 31.2 准备依赖

```text
1. 从 lerobot_flex 启动
2. 准备项目内 FlashAttention wheel
3. 运行 setup_env.sh
4. 运行 setup_robot_client_deps.sh
5. 运行 validate_environment.py
```

环境入口：

```bash
/home/cqy/miniconda3/bin/conda run --no-capture-output -n lerobot_flex \
  bash my_devs/gr00t_17/scripts/setup_env.sh
```

### 31.3 准备模型

```text
1. 使用 pinned revision 下载 N1.7 基础模型
2. 下载经过 blob 等价证明的 Qwen3-VL runtime assets
3. 严格加载模型
4. 检查 missing/unexpected/mismatch
5. 写 model reports
```

模型下载必须指定 revision，不使用会漂移的 latest/main。下载报告是后续 preflight 的输入，不是可选日志。

### 31.4 准备数据

```text
1. 转换 v3 -> v2.1 partial
2. 生成 modality metadata
3. 生成 relative stats
4. 全量 parquet/video 检查
5. TorchCodec 像素检查
6. 原子发布 final
7. 生成 converted manifest
```

### 31.5 训练

```text
1. 验证真实 batch
2. 两阶段 smoke + resume
3. 比较 trainable/frozen tensor
4. 正式 preflight
5. 按 effective batch 计算 max steps
6. 正式训练
7. 验收 final checkpoint
```

### 31.6 部署

```text
1. ping Server
2. 只读 motor bus preflight
3. 两摄像头快照
4. no-actuation prediction
5. physical/200/200 短时间动作验证
6. RTC dry-run
7. 正式 RTC 上机
```

### 31.7 TensorRT

```text
1. export full pipeline ONNX
2. build 7 engines
3. reference numerical verification
4. RTC-specific numerical verification
5. backend-aware ping
6. real-device no-actuation 30 Hz dry-run
7. restart formal Server with TensorRT default
```

## 32. 常见故障树

### 32.1 数据加载失败

检查顺序：

1. `meta/info.json` 是否为 v2.1；
2. `episodes.jsonl/tasks.jsonl/modality.json` 是否存在；
3. per-episode 文件模板是否与 info 一致；
4. video key 是否为 top/wrist；
5. TorchCodec 是否能解码 AV1；
6. relative stats 是否为 horizon 16 x arm dim 5。

不要通过修改 loader 去“兼容”一个尚未验证的数据副本，优先修复转换和 metadata。

### 32.2 CUDA OOM

训练优先降低 micro batch，不要先删除 checkpoint 或污染其他环境。扩大 effective batch 应增加 gradient
accumulation。正式训练 preflight 要求至少 40,000 MiB 空闲显存。

TensorRT 构建时还要考虑 builder workspace 和同时运行的推理服务。不要擅自杀死不属于本任务的 GPU 进程；
若剩余显存不足，应先由操作者停止自己的服务再重建。

### 32.3 loss 不正常

检查：

- action 是否被重复转 relative；
- 关节顺序是否错；
- state/action 单位是否改变；
- active mask 是否仍为 96；
- 两路图像是否进入 prompt；
- 任务文本是否正确；
- 恢复训练是否加载了错误 run。

### 32.4 摄像头打开但模型效果差

检查实际快照、设备映射、RGB/BGR、shape、dtype、均值/标准差和 warmup。不能只看 `/dev/video*` 存在。

### 32.5 机械臂只抖

先看 Client 和 robot 日志中的 clamp。若 dataset bounds 与当前姿态冲突，模型轨迹会被完全覆盖。当前正式
配置必须是 physical/200/200。

### 32.6 RTC queue underrun

检查模型端到端延迟、execution horizon、控制线程是否被 camera read 阻塞、请求是否错误并发、Server 是否
发生首次编译、网络是否为 localhost。不要在 underrun 时重复最后动作掩盖问题。

### 32.7 TensorRT Server 无法启动

检查：

1. 7 个 engine 是否非空；
2. export metadata 是否 batch 1、NEW_EMBODIMENT；
3. GPU 是否与构建机器一致；
4. TensorRT/CUDA 版本是否改变；
5. engine path 是否仍在 `my_devs/gr00t_17`；
6. 端口是否被旧 Server 占用。

必要时重新运行 `build_so101_trt.sh`，不要自动回退而不打印 backend。

### 32.8 Client 拒绝 Server

新 Client 要求 ping 返回 TensorRT。如果旧 PyTorch Server 占用 5556，先在旧终端 Ctrl+C，再启动新 Server。

### 32.9 TensorRT 数值漂移

先区分 ViT、backbone、普通 final action 和 RTC action。只看最终 cosine 不能定位模块。若换 GPU/版本后漂移：

```text
重新 export
重新 build
运行参考 verify
运行 verify_so101_rtc_trt.py
最后才做真机 dry-run
```

## 33. 测试与验证命令

```bash
cd /data/cqy_workspace/flexible_lerobot

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/cqy/miniconda3/bin/conda run --no-capture-output -n lerobot_flex \
  pytest -q my_devs/gr00t_17/tests
```

当前结果：`18 passed`。

关闭 pytest plugin autoload 是因为系统 ROS Python 3.8 插件会被 Python 3.10 pytest 自动发现，并因缺少
`lark` 失败。这是系统插件污染，不是项目测试失败。

静态检查：

```bash
my_devs/gr00t_17/tools/dev_deps/bin/ruff check \
  my_devs/gr00t_17/scripts \
  my_devs/gr00t_17/tests
```

Shell 脚本至少运行 `bash -n`。真机脚本的最终验证不能由单元测试替代，仍需要真实 camera/robot dry-run。

## 34. 报告索引

| 报告 | 说明 |
| --- | --- |
| `data_conversion.json` | v3 -> v2.1 转换结果 |
| `dataset_validation.json` | parquet/video/pixel 验证 |
| `environment.json` | 环境和 package 快照 |
| `environment_validation.json` | CUDA/FlashAttention/TorchCodec 验证 |
| `flash_attn_build.json` | 本机构建 wheel 证据 |
| `model_download.json` | pinned N1.7 文件哈希 |
| `backbone_assets.json` | Qwen/Cosmos runtime blob 等价证明 |
| `model_validation.json` | 严格加载和张量比较 |
| `training_batch.json` | 真实 batch shape/mask |
| `smoke_train.json` | 两阶段训练和恢复 |
| `formal_training_validation.json` | 10 epoch checkpoint 验收 |
| `post_deployment_integrity_20260712.json` | 真机开发后的不可变复核 |
| `so101_rtc_trt_verification.json` | 7 engine 哈希和 RTC 数值对照 |

运行时大型输出报告在 `outputs/` 下，不进 Git；上表的小型摘要报告可以进入 Git，作为复现目标和审计证据。

## 35. 脚本职责索引

| 脚本 | 职责 |
| --- | --- |
| `common.sh` | 路径、缓存、离线模式和训练公共参数 |
| `create_manifest.py` | 创建/验证不可变文件清单 |
| `convert_v3_to_gr00t_v21.py` | 非破坏性数据转换 |
| `validate_gr00t_dataset.py` | 独立深度数据验证 |
| `validate_training_batch.py` | 真实 processor/collator batch |
| `run_smoke_train.sh` | 两阶段训练恢复 smoke |
| `run_train.sh` | 正式训练唯一包装入口 |
| `preflight.py` | fail-closed 训练门禁 |
| `validate_formal_training.py` | 正式 checkpoint 验收 |
| `so101_robot_client.py` | 同步推理和基础硬件适配 |
| `so101_rtc_policy_server.py` | stateful model-level RTC Server |
| `so101_rtc_robot_client.py` | 异步 30 Hz RTC Client |
| `run_so101_rtc_policy_server.sh` | 独立 Server，默认 TensorRT |
| `run_so101_rtc_client.sh` | 独立正式上机 Client |
| `run_so101_rtc_smoke.sh` | 真实设备零动作 RTC smoke |
| `build_so101_trt.sh` | full-pipeline ONNX/engine 构建验证 |
| `so101_rtc_trt.py` | RTC-aware TensorRT action forward |
| `verify_so101_rtc_trt.py` | 普通/RTC PyTorch-TRT 数值对照 |

## 36. Git 发布边界

应该提交：

- `configs/so101_modality.py`；
- `configs/modality.json`；
- `scripts/*.py` 和 `scripts/*.sh`；
- `tests/*.py`；
- `doc/*.md`；
- 小型 source/derived manifests；
- 小型验证 reports；
- `my_devs/gr00t_17/.gitignore`。

不应该提交：

- `.safetensors`；
- checkpoint 和 optimizer；
- 原始或转换后视频/parquet；
- `.onnx` 和 `.onnx.data`；
- `.engine`/`.trt`；
- `env/`、`cache/`、`tools/`；
- `workspace/` 和 `reference/` 副本；
- `outputs/`、日志、相机快照；
- 机器特定 calibration，除非明确决定共享该机器人标定。

## 37. 修改系统时的强制检查清单

### 37.1 改数据或 modality

- 重新转换数据；
- 重新生成 relative stats；
- 重新验证 478 项 manifest；
- 重新 smoke train；
- 旧 checkpoint 不能默认继续使用。

### 37.2 改 action horizon

- 更新 Python modality；
- 更新 relative stats；
- 更新 RTC overlap/advance；
- 更新 Client chunk validation；
- 重新训练；
- 重新导出全部 TensorRT engine。

### 37.3 改相机或任务文本

- 更新训练数据和 modality；
- 检查 TensorRT LLM/DiT dynamic sequence profile；
- 重新做快照和 dry-run；
- 新文本不应绕过已知任务检查直接上机。

### 37.4 改 checkpoint

- 重新严格加载；
- 重新生成 TensorRT engines；
- 重新做普通和 RTC 数值验证；
- Server health 和报告必须指向新 checkpoint。

### 37.5 改 GPU/CUDA/TensorRT

- 重新构建 engines；
- 重新记录 SHA；
- 重新做 real-device dry-run；
- 不复用旧 engine。

## 38. 已知边界和后续工作

当前完成的是单客户端、单 GPU、localhost、batch 1 的真实 SO101 部署。以下内容没有被本次结果证明：

- 多客户端并发 Server；
- 跨主机网络 RTC；
- 网络认证、TLS 或远程急停；
- batch 大于 1 的实时推理；
- 新相机分辨率或第三路相机；
- 未出现在训练数据中的自由文本任务；
- 在不同 GPU 架构上直接复用 engines；
- TensorRT INT8/FP8 量化；
- 自动任务成功率评估。

这些扩展都不能仅靠修改命令行参数完成，需要各自的验证设计。

## 39. 最终工程结论

这次工作的关键不是单独跑通一次训练命令，而是建立了可审计的完整闭环：

```text
只读输入哈希
  -> 非破坏性数据转换
  -> modality/action 语义验证
  -> 隔离环境和严格模型加载
  -> 真实 batch
  -> smoke + resume + 权重差异
  -> 10 epoch 正式训练
  -> checkpoint 加载
  -> 相机/机器人分层 smoke
  -> 限制策略排障
  -> model-level RTC
  -> async 30 Hz timeline
  -> independent Server/Client
  -> TensorRT full pipeline
  -> RTC-aware TensorRT 修正
  -> 数值、延迟和真实设备零动作验收
```

后续维护时最重要的原则：

1. 不把数据格式转换和动作语义转换混为一件事；
2. 不把“模型有输出”当作训练、RTC 或 TensorRT 正确的证据；
3. 不让部署限制静默覆盖模型轨迹，也不删除 NaN 和物理域检查；
4. 任何 checkpoint、horizon、modality、GPU 或 TensorRT 变化都必须重新走对应验证链路；
5. 代码和文档进入 Git，数据、权重、checkpoint、ONNX 和 engines 留在受控本地存储。
