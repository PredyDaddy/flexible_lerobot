# GR00T N1.7 SO101 多任务训练技术设计

> 实施状态（2026-07-12 更新）：10-epoch 正式训练和 checkpoint 真机无动作预测已经完成。本文保留
> 设计依据；实际训练数字以执行报告和 `../reports/formal_training_validation.json` 为准，真机安全入口
> 以 `GR00T_N1_7_SO101_ROBOT_DEPLOYMENT.md` 为准。

## 1. 技术结论摘要

本项目不能把现有 LeRobot v3 数据目录直接交给 Isaac GR00T N1.7。N1.7 当前数据加载器要求每个 episode 独立 parquet/video 的 LeRobot v2.1 布局，并额外要求 `meta/modality.json`。正确方案是在 `my_devs/gr00t_17/data/` 内生成一份非破坏性的 GR00T LeRobot v2.1 数据副本。

推荐的首版动作建模是：

- 前五个 SO101 臂关节：原数据继续保存绝对目标，进入 N1.7 processor 后转为相对动作；
- 夹爪：保持绝对动作；
- action horizon：16 帧，对 30 FPS 数据约为 0.533 秒；
- state：当前帧；
- video：当前 top + wrist 双视角；
- language：从 `task_index` 映射到三个英文任务文本；
- 不对度制关节值使用 sin/cos embedding。

训练采用 N1.7 官方默认的 projector + diffusion head 微调，不在单卡上训练 LLM 或 visual backbone。
真实双相机 micro batch 1 已通过训练和恢复验证；正式基线使用该保守 micro batch，通过梯度累积得到
effective batch 8。更大的单步 batch 必须另做显存阶梯后才能启用。

## 2. 输入系统审计

### 2.1 现有 PI 训练链路

`my_devs/train/pi/so101/easy_train.sh` 的核心输入为：

```text
dataset.repo_id = desk_cleanup_v1/eraser_cup_multi_task
dataset.root    = datasets/desk_cleanup_v1/eraser_cup_multi_task
policy.type     = pi05
policy.pretrained_path = assets/modelscope/lerobot/pi05_base
batch_size      = 16
steps           = 33300
dtype           = bfloat16
```

PI checkpoint 显示：

```text
n_obs_steps     = 1
chunk_size      = 50
n_action_steps  = 50
state_dim       = 6
action_dim      = 6
image inputs    = top + wrist, 480 x 640
image resolution in model = 224 x 224
normalization   = action/state mean-std
```

现有真机脚本每个控制周期读取 top、wrist 和 6 维 state，调用 PI policy 后将 6 维绝对关节目标发送给 SO follower。N1.7 的真机适配需要保持相同外部机器人契约，但模型内部的相对动作表示不能直接暴露给 `send_action()`。

### 2.2 原数据 schema

v3 主 parquet 列：

```text
action: list<float32>[6]
observation.state: list<float32>[6]
timestamp: float32
frame_index: int64
episode_index: int64
index: int64
task_index: int64
```

状态和动作维度名称完全一致：

```text
0 shoulder_pan.pos
1 shoulder_lift.pos
2 elbow_flex.pos
3 wrist_flex.pos
4 wrist_roll.pos
5 gripper.pos
```

数值范围显示这是旧 SO follower 位置表示：臂关节约为度制范围，夹爪约为 0-70 的位置值。它们不是弧度，也不是速度命令。

动作与同帧 state 的差值统计进一步支持“绝对目标”判断：

| 维度 | action-state 绝对差中位数 | 绝对差 q99 |
| --- | ---: | ---: |
| shoulder_pan | 0.879 | 10.462 |
| shoulder_lift | 2.286 | 14.330 |
| elbow_flex | 2.374 | 14.330 |
| wrist_flex | 0.396 | 5.758 |
| wrist_roll | 0.440 | 9.319 |

这些差异包含 leader/follower 跟踪和时序滞后，不应在数据转换阶段被“修正”。

### 2.3 数据量和任务分布

| 项目 | 值 |
| --- | ---: |
| episodes | 157 |
| frames | 53,235 |
| fps | 30 |
| top frames | 53,235 |
| wrist frames | 53,235 |
| task 0 episodes/frames | 62 / 18,600 |
| task 1 episodes/frames | 55 / 16,500 |
| task 2 episodes/frames | 40 / 18,135 |

task 0、1 的 episode 多数为 300 帧；组合任务 episode 更长。训练 sampler 必须按 episode/window 正确处理，不能把 task 2 因 episode 少而误判为严重欠采样，也不能仅看 episode 数忽略其总帧数。

### 2.4 原视频

- 编码：AV1；
- pix_fmt：yuv420p；
- 分辨率：640 x 480；
- FPS：30；
- 音频：无；
- v3 中每路相机由多个聚合文件组成，episode 起止时间位于 episodes metadata。

N1.7 release 自带 SO101 demo 的 metadata 同样声明 AV1 和 HWC `[480, 640, 3]`，因此 AV1 不是天然不兼容项。TorchCodec 的实际解码仍必须在目标环境中验证。

## 3. N1.7 代码路径分析

### 3.1 训练入口

主要入口：

`gr00t/experiment/launch_finetune.py`

它会：

1. 通过自定义 Python 文件注册 modality config；
2. 把一个数据集路径和 embodiment tag 写入默认 config；
3. 从 `base_model_path` 加载基础 checkpoint；
4. 默认冻结 LLM 和视觉编码器；
5. 默认训练 projector 和 diffusion model；
6. 使用 Hugging Face Trainer 保存 checkpoint。

首版不需要改 N1.7 模型结构。定制点应限制为本项目的 modality config、数据转换/验证和安全训练包装脚本。

### 3.2 数据加载器的硬要求

`LeRobotEpisodeLoader` 会直接读取：

```text
meta/info.json
meta/episodes.jsonl
meta/tasks.jsonl
meta/modality.json
meta/stats.json
meta/relative_stats.json  # 相对动作时使用
```

它按照 `info.json` 的 v2.1 模板定位：

```text
data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet
videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4
```

因此当前 v3 的聚合 parquet、聚合 video、tasks parquet 和 episodes parquet 都不能直接使用。

语言加载不是要求 parquet 真正增加一个长字符串列。`modality.json` 可以把：

```text
annotation.human.task_description -> original_key: task_index
```

加载器再通过 `tasks.jsonl` 将整数转换为文本。这意味着转换数据无需复制长字符串到每一帧，也不需要改原始 parquet schema。

### 3.3 训练 batch 的语义陷阱

N1.7 当前代码执行：

```text
per_device_train_batch_size = global_batch_size // num_gpus
```

随后 Trainer 仍然接收 `gradient_accumulation_steps`。因此在单卡上：

```text
effective batch = global_batch_size * gradient_accumulation_steps
```

如果误把 `global_batch_size` 当成已经包含累积的有效 batch，会得到比预期更大的训练 batch。包装脚本必须计算并打印实际值，配置审计也要把这个值写入 run manifest。

### 3.4 checkpoint 恢复行为

训练调用 `trainer.train(resume_from_checkpoint=True)`。如果输出目录已有有效 checkpoint，会自动恢复；如果目录复用不当，可能意外继续旧实验。

因此：

- 每个新实验必须使用新的 run 目录；
- resume 必须显式指定已有 run，并校验 config/data/model hash；
- smoke 和正式训练不得共享目录；
- 默认 `save_only_model=false`，否则 checkpoint 不能完整恢复；
- 不允许将基础模型目录作为 output directory。

## 4. v3 到 GR00T v2.1 的映射设计

### 4.1 目标结构

```text
converted_dataset/
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   ├── tasks.jsonl
│   ├── modality.json
│   ├── stats.json
│   └── relative_stats.json
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       └── ... episode_000156.parquet
└── videos/
    └── chunk-000/
        ├── observation.images.top/
        │   ├── episode_000000.mp4
        │   └── ... episode_000156.mp4
        └── observation.images.wrist/
            ├── episode_000000.mp4
            └── ... episode_000156.mp4
```

### 4.2 metadata 映射

| v3 输入 | v2.1 输出 | 处理 |
| --- | --- | --- |
| `meta/info.json` | `meta/info.json` | version 改为 v2.1，路径模板改为 per-episode，增加 total_chunks/total_videos |
| `meta/tasks.parquet` | `meta/tasks.jsonl` | 按 task_index 排序后逐行写 JSONL |
| `meta/episodes/chunk-*/file-*.parquet` | `meta/episodes.jsonl` | 保留 episode_index/tasks/length |
| 同上 stats 字段 | 可选 `episodes_stats.jsonl` | 只用于审计；N1.7 主路径不依赖 |
| `meta/stats.json` | `meta/stats.json` | 数据未变时复制并复算核验 |
| 无 | `meta/modality.json` | 本项目新增 |
| 无 | `meta/relative_stats.json` | 由 N1.7 stats 工具按 horizon 16 生成 |

`info.json` 的视频 feature shape 应保持 N1.7 官方 demo 使用的 HWC：

```json
"shape": [480, 640, 3],
"names": ["height", "width", "channels"]
```

其他目录中的 OpenPI v2.1 副本使用 CHW metadata 是 OpenPI/LeRobot 转换路径的选择，不应直接复制到本 N1.7 数据集。

### 4.3 parquet 切分

v3 episode metadata 提供：

```text
data/chunk_index
data/file_index
dataset_from_index
dataset_to_index
```

转换器读取对应聚合 parquet，并按 `[from:to)` 切出 episode。输出保持原列、dtype、值和列顺序，不做以下操作：

- 不改状态/动作单位；
- 不预计算 relative action；
- 不重排关节；
- 不改变 timestamp；
- 不覆盖 task_index；
- 不添加训练模型特有 padding。

每个输出 parquet 写完后立即检查：

```text
num_rows == episode.length
unique episode_index == metadata episode_index
frame_index == 0..length-1
task_index matches episode task
state/action dimension == 6
```

### 4.4 视频切分

v3 episode metadata 对每个 camera 提供：

```text
videos/<video_key>/chunk_index
videos/<video_key>/file_index
videos/<video_key>/from_timestamp
videos/<video_key>/to_timestamp
```

首选使用 FFmpeg stream copy：

```text
-ss <start> -i <source> -t <duration> -c copy -avoid_negative_ts 1
```

优势：

- 不损失图像质量；
- 不改变 AV1 编码；
- 速度快；
- 与 N1.7 demo codec 一致。

风险：输入 seek 与 keyframe 边界可能让个别文件出现起点或帧数偏差。为此不能仅凭 FFmpeg 返回码判断成功，必须全量运行 FFprobe：

```text
nb_frames == episode.length
avg_frame_rate == 30/1
width == 640
height == 480
codec_name == av1
```

此外对首、中、尾 frame 做 TorchCodec 解码。若 stream copy 不能全量通过，则在目标目录中统一重编码所有视频，而不是只修补个别文件造成 metadata 与 codec 不一致。

### 4.5 原子性和幂等性

转换目标先写到：

`data/converted_v21/<dataset>.partial-<run_id>`

流程：

1. 验证 source 是只读输入且不位于 allowed write root；
2. 验证 partial 不存在；
3. 写 conversion manifest；
4. 生成所有文件；
5. 全量检查；
6. 再次验证 source hash；
7. `rename(partial, final)` 原子发布。

如果 final 已存在，默认报错。禁止 `--force` 直接递归删除 final；重做时应生成新版本目录，验证后由人工选择保留版本。

## 5. GR00T modality 设计

### 5.1 `meta/modality.json`

计划内容：

```json
{
  "state": {
    "single_arm": {"start": 0, "end": 5},
    "gripper": {"start": 5, "end": 6}
  },
  "action": {
    "single_arm": {"start": 0, "end": 5},
    "gripper": {"start": 5, "end": 6}
  },
  "video": {
    "top": {"original_key": "observation.images.top"},
    "wrist": {"original_key": "observation.images.wrist"}
  },
  "annotation": {
    "human.task_description": {"original_key": "task_index"}
  }
}
```

这里保留 `top` 命名，不强行改成官方示例的 `front`。原因是训练数据和现有真机相机都叫 top，直接一致可减少部署映射错误。N1.7 对 view key 没有要求必须叫 front，只要求 Python config 和 modality metadata 一致。

### 5.2 Python modality config

等价设计：

```python
so101_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["top", "wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["single_arm", "gripper"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["single_arm", "gripper"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}
```

该配置注册到 `EmbodimentTag.NEW_EMBODIMENT`，并通过 `--modality-config-path` 加载。配置文件必须位于本项目 `configs/`，不修改 reference 中的 `examples/SO100/so100_config.py`。

### 5.3 为什么臂用 relative、夹爪用 absolute

臂关节：

- 数据保存的是绝对目标；
- N1.7 processor 能用当前 state 作为 reference，在训练时计算未来目标相对当前 state 的变化；
- 相对表示降低不同初始姿态造成的绝对值偏移；
- 官方 SO100 配置使用同一策略。

夹爪：

- 它更接近开合位置/状态；
- 相对变化会累积漂移；
- 绝对表示更容易学习开与关的目标位置；
- 官方 SO100 配置同样使用 absolute。

### 5.4 为什么不使用 sin/cos state embedding

N1.7 文档建议 sin/cos 只用于弧度角。当前 SO follower 数据明显是度制/百分比位置。如果直接对数值 `60` 计算 `sin(60)`，模型会把它当 60 radians，而不是 60 degrees，语义错误。

首版保持默认 percentile/min-max 类型的 state 处理。只有将数据和真机接口共同、显式转换为 radians 并完成全链路验证后，才可以考虑 sin/cos；本项目当前没有这个必要。

### 5.5 relative stats

相对动作统计必须根据 16 步 horizon 重新生成，不能沿用原 v3 的绝对 action stats。预期：

```text
relative_stats.json
└── single_arm
    ├── min  [16, 5]
    ├── max  [16, 5]
    ├── q01  [16, 5]
    ├── q99  [16, 5]
    ├── mean [16, 5]
    └── std  [16, 5]
```

如果未来把 horizon 改为 8、32 或 40，必须重新生成 relative stats。shape 不匹配会在 normalization 阶段失败，或更危险地应用错误的逐 horizon 统计。

## 6. 环境与依赖隔离设计

### 6.1 原则

- `lerobot_flex` 作为仓库规定的开发/引导环境；
- N1.7 的锁定依赖安装到 `my_devs/gr00t_17/env/` 的本地虚拟环境；
- 不修改 base conda 或其他已有 conda env；
- 不复用 `$HOME/.cache/huggingface` 中的可变 Hub 缓存作为正式输入；
- reference 保持只读，实际 editable install 指向 workspace 副本。

### 6.2 参考仓关键依赖

参考 `pyproject.toml`：

```text
Python == 3.10.*
torch == 2.7.1
torchvision == 0.22.1
transformers == 4.57.3
diffusers == 0.35.1
torchcodec == 0.4.0 (x86_64)
flash-attn == 2.7.4.post1
deepspeed == 0.17.6
av == 16.1.0
```

参考 lockfile 使用 PyTorch CUDA 12.8 index。driver 570 理论上满足新 runtime，但环境验收必须以实际 import/kernel test 为准。

### 6.3 缓存隔离

启动任何安装、下载、统计、训练或推理命令前，统一 source 本项目的环境脚本。该脚本负责：

- 导出所有 cache 路径；
- 创建本项目 tmp；
- 禁用 W&B 网络记录；
- 默认 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`；
- 只有模型下载子命令临时允许 online；
- 设置 `PYTHONPYCACHEPREFIX`，避免 workspace/reference 产生 `__pycache__`；
- 打印并校验 cwd、Python 和所有输出根目录。

## 7. 基础模型管理

### 7.1 下载策略

基础模型约 6 GiB，但实际占用必须以 snapshot manifest 为准。下载时：

1. 使用本地 `models/.partial`；
2. 指定 revision；
3. 禁止 symlink 到根目录外缓存；
4. 下载后检查必要 config、processor、safetensors index 和全部 shards；
5. 记录 SHA-256；
6. 训练命令传本地绝对路径。

### 7.2 权重保护

基础模型目录同样只读使用。训练 output 不得位于模型目录内。run manifest 同时记录 base model hash 和训练后 checkpoint hash，二者不能混淆。

## 8. 训练参数设计

### 8.1 首版冻结策略

```text
tune_llm             = false
tune_visual          = false
tune_projector       = true
tune_diffusion_model = true
```

官方说明该策略峰值通常低于约 35 GiB。双相机和 batch 会改变实际占用，所以该数字仅作为可行性依据，不作为容量承诺。

### 8.2 初始优化参数

```text
learning_rate = 1e-4
weight_decay = 1e-5
warmup_ratio = 0.05
state_dropout_prob = 0.2
bf16 = true
tf32 = true
save_only_model = false
use_wandb = false
```

`state_dropout_prob=0.2` 是 finetune CLI 默认值，而模型基础 config 默认值可能不同。正式命令必须显式传入，避免依赖默认值漂移。该任务依赖关节 state 计算 relative action，因此开发实验应至少比较 0.0、0.1、0.2，但不能同时改变多个超参后无法归因。

### 8.3 batch 与累积

双相机条件下按以下矩阵做短测：

```text
per-device batch: 1 -> 2 -> 4 -> 8 -> 16 -> 32
gradient accumulation: 首轮固定 1
```

选择原则：

- 不 OOM；
- 峰值显存不超过可用显存约 90%；
- 数据加载不成为主要瓶颈；
- 20 steps 后 loss 有限；
- checkpoint save 后仍有足够 host/GPU 余量。

只有单步 batch 太小时才引入 gradient accumulation。引入后以实际 effective batch 为准调整学习率，不机械使用线性缩放。

### 8.4 训练步数

horizon 16 且不 padding 时，完整数据的有效 windows 为：

```text
53235 - 157 * 15 = 50880
```

传统意义上的近似样本 epoch 可写为：

```text
approx_epochs = steps * effective_batch / 50880
```

但 N1.7 使用 sharded iterable sampling，`episode_sampling_rate` 和 shard 设置会影响实际重复率，因此必须同时记录 sampler 统计，不能只依据上述近似数。

建议从 2,000 steps 基线开始，在 4,000、8,000 设延长决策点。33,300 是 PI 模型的训练设置，不适合作为 N1.7 默认目标。

### 8.5 保存策略

开发阶段建议：

```text
save_steps = 500
save_total_limit = 5
logging_steps = 10
```

smoke 阶段把 `save_steps` 设为 1，实际完成 step 1 保存、跨进程恢复到 step 2 和再次保存。正式 run
不使用 `last` 符号链接覆盖旧 checkpoint；恢复通过同一 `RUN_ID` 和显式 `RESUME=1` 完成。

### 8.6 已验证的单卡基线

真实 smoke 的 micro batch 为 1，采样峰值总显存 37,852 MiB；可训练 action-head 权重在恢复后的
step 2 发生 `1.001358e-4` 的最大绝对变化，冻结 LLM q-proj 与基础权重逐值一致。每个包含模型、
optimizer、scheduler 和 RNG 的 checkpoint 为 23,314,772,866 bytes。基于该结果，正式入口固定
micro batch 1、gradient accumulation 8，并要求启动前至少有 40,000 MiB 空闲显存。后续经用户授权的
正式 run 实际使用 micro batch 2、gradient accumulation 4；effective batch 仍为 8，并完成 63,600
optimizer steps。该实际结果替代本节早期 baseline 作为部署 checkpoint 的训练事实。

## 9. 数据划分与评估设计

### 9.1 分层划分

开发阶段按 task 分层选择约 10% episode 作为 validation，并保存明确的 episode ID manifest。不能逐帧随机划分，因为同一轨迹的相邻帧高度相关，会造成严重泄漏。

可采用固定 seed 42 随机选择：

- task 0：约 6 episodes；
- task 1：约 6 episodes；
- task 2：约 4 episodes。

转换后的 master v2.1 数据保持全部 157 episodes 不变。train/val 通过派生目录或显式 manifest 实现，任何 split 都不改 master。

### 9.2 open-loop 指标

至少输出：

```text
MAE/MSE by task
MAE/MSE by joint
MAE/MSE by horizon step
relative-arm error
absolute-arm error after inverse transform
absolute-gripper error
q01/q99 exceedance rate
first/second difference smoothness
```

组合任务必须单独显示。总体平均值可能被较容易的单任务掩盖。

### 9.3 checkpoint 选择

优先级：

1. 数据和输出安全检查通过；
2. validation 无 NaN/越界；
3. 三任务指标均衡；
4. 绝对动作轨迹平滑；
5. 真机 dry-run 合理；
6. 才比较总体 loss。

训练 loss 最低不等于可部署 checkpoint。

## 10. 真机推理接口设计要点

### 10.1 observation 映射

从现有 SO follower 读取：

```text
observation.state -> split into single_arm[5], gripper[1]
observation.images.top -> top
observation.images.wrist -> wrist
task string -> annotation.human.task_description equivalent input
```

图像采集保持 640 x 480、30 FPS，由 N1.7 processor 做 crop/resize。相机顺序必须与训练 config 一致，因为 loader 的某些兼容路径会按位置自动映射不同 view key，依赖这种自动映射会隐藏 top/wrist 颠倒问题；真机适配应显式键映射。

### 10.2 action 反变换

模型输出的 `single_arm` 是规范化后的相对动作，必须经过 checkpoint processor：

1. 去 padding/mask；
2. 反归一化 relative action；
3. 使用推理时原始 current state 转为绝对关节目标；
4. gripper 反归一化为绝对值；
5. 合并成训练时 6 维顺序；
6. 应用机器人安全限幅；
7. 才发送。

不能自行手写一套与训练 processor 不同的 normalization。

### 10.3 时序

16 步在 30 Hz 下覆盖约 0.533 秒。N1.7 单次推理可能不能达到 30 Hz，因此需要在后续实现中选择：

- action chunk 缓冲；
- 固定消费前 N 步后重规划；
- 异步推理与 observation 时间戳；
- 过期 chunk 丢弃；
- 推理失败时 hold/stop。

这属于部署层，不应为了适配推理延迟而在首轮训练前擅自改数据 FPS。

### 10.4 安全边界

- 每关节绝对范围检查；
- 每周期最大相对变化；
- NaN/Inf 立即停止；
- observation 超时停止；
- camera 缺帧停止或受控 hold；
- 夹爪单独限幅；
- 急停和 torque disable 路径先验证；
- 先 dry-run，再低速短时运行。

## 11. 测试设计

### 11.1 单元测试

- 路径守卫拒绝根目录外写入；
- v3 metadata 到 v2.1 metadata 映射；
- task parquet 到 JSONL；
- episode parquet 切片边界；
- modality schema；
- state/action split 与 merge；
- relative -> absolute round trip；
- horizon 尾部边界；
- run manifest hash。

### 11.2 集成测试

- 转换 1 个短 episode 到临时目录；
- 两路视频帧数和随机帧；
- N1.7 loader 读取三个任务各一个样本；
- collator/processor shape；
- 基础模型单 batch forward/backward；
- 20 step save + resume；
- checkpoint 离线加载和单次 inference。

### 11.3 全量数据检查

- 157 parquet；
- 314 video；
- 53,235 行；
- 106,470 camera frames；
- 所有 episode/task/index/timestamp；
- 所有 active source/output manifest；
- 源数据前后哈希一致。

## 12. 风险清单和处理

| 风险 | 影响 | 检测 | 处理 |
| --- | --- | --- | --- |
| 官方转换器移动输入 | 破坏原数据路径 | 代码审计已确认 | 不直接使用默认 CLI，写只读 source/destination wrapper |
| AV1 stream-copy seek 偏差 | 图像和 parquet 错位 | 全量逐 episode 帧数、随机解码 | 目标目录统一重编码，不改源 |
| top/wrist 顺序颠倒 | 模型训练/推理语义错误 | 显式带标签可视化样本 | 全链路使用显式 key，不依赖 positional auto-map |
| 度数误用 sin/cos | state 表示错误 | config 审计、数值范围测试 | 不启用 sin/cos |
| relative stats horizon 不匹配 | normalization 错误/崩溃 | shape 断言 `(16,5)` | horizon 改动后强制重算 |
| global batch 语义误解 | OOM 或训练尺度错误 | 启动时打印 effective batch | 累积首轮为 1，显式计算 |
| 双相机显存超限 | OOM | batch 阶梯 | 减 batch，不训练 LLM/visual |
| iterable sampler 重复率未知 | 过拟合/步数估计错误 | 记录 sampled episode/window | 用 validation 决定延长 |
| checkpoint 自动恢复错 run | 污染实验 | output 目录和 hash 检查 | 每 run 新目录，resume 显式授权 |
| 默认缓存写到 HOME | 污染其他环境 | 运行前后缓存审计 | 统一环境变量和 allowed-root guard |
| 真机发送 relative action | 危险动作 | dry-run 和 round-trip test | processor 反变换后才 send |
| 训练只看总 loss | 单任务失败被掩盖 | per-task/per-joint 指标 | 分任务报告和真机分级验证 |

## 13. 已确认事项与待执行验证

### 已确认

- 原数据是 LeRobot v3.0，N1.7 loader 需要 v2.1 per-episode 布局；
- 原数据字段、任务、episode/frame 数和两路视频总帧数一致；
- N1.7 需要额外 `modality.json`；
- 官方 SO100 config 与当前 6 维 SO101 数据结构直接对应；
- 官方转换脚本会移动输入，不能对原路径运行；
- N1.7 demo 使用 AV1 和 HWC metadata；
- 当前 GPU/显存满足默认冻结策略的最低要求；
- 本地没有 N1.7 3B 基础模型快照，后续必须下载到本项目目录；
- `lerobot_flex` 没有 uv，后续需要在本项目目录安装本地工具或采用等价锁定安装。

### 待执行验证

- 目标虚拟环境的 CUDA 12.8、flash-attn、torchcodec 组合；
- TorchCodec 对转换后 AV1 全量随机 seek；
- 双相机真实最大 batch；
- relative stats 的数值与 round-trip；
- N1.7 base model 对自定义 `NEW_EMBODIMENT` config 的端到端加载；
- 2,000 steps 是否足够以及 state dropout 最优值；
- 真机推理延迟和 action chunk 消费策略。

这些待验证项必须通过实验回答，文档不把它们伪装成已确定事实。
