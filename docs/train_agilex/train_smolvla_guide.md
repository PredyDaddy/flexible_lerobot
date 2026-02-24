# Agilex Piper 双臂机器人 SmolVLA 训练指南

本文档指导你使用 SmolVLA (Vision-Language-Action) 模型训练 Agilex Piper 双臂机器人的策略。

---

## 1. SmolVLA 简介

SmolVLA 是 Hugging Face 开发的轻量级视觉-语言-动作模型，基于 SmolVLM2 视觉语言模型，专为机器人操作任务设计。

**核心特点：**
- 基于预训练的 VLM (Vision-Language Model)
- 支持语言指令条件
- 使用 Flow Matching 进行动作预测
- 相比 Pi0 等大模型更加轻量

**论文**: https://arxiv.org/abs/2506.01844

---

## 2. 前置准备

### 2.1 安装 SmolVLA 依赖

```bash
# 在 lerobot 目录下安装 smolvla 额外依赖
pip install -e ".[smolvla]"
```

### 2.2 配置 Hugging Face 镜像（国内用户必须）

SmolVLA 需要从 Hugging Face Hub 下载预训练权重，国内用户需要配置镜像加速：

> 下文命令示例以 `HF_ENDPOINT=https://hf-mirror.com` 前缀写法为例；如果你已在 shell 中 `export HF_ENDPOINT=...`，可省略前缀。

**方法一：临时设置环境变量**
```bash
# 在训练命令前设置
export HF_ENDPOINT=https://hf-mirror.com

# 然后运行训练
lerobot-train ...
```

**方法二：写入 ~/.bashrc 永久生效**
```bash
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

**方法三：在命令中直接指定**
```bash
HF_ENDPOINT=https://hf-mirror.com lerobot-train \
    --policy.path=lerobot/smolvla_base \
    ...
```

**需要下载的模型：**
| 模型 | 用途 | 大小 |
|------|------|------|
| `lerobot/smolvla_base` | SmolVLA 预训练权重（用于 `--policy.path` 微调） | ~2GB |
| `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | VLM backbone（tokenizer/processor 必需；启用 `--policy.load_vlm_weights=true` 时会额外下载权重） | ~1GB |

**手动预下载（可选）：**
```bash
# 使用 huggingface-cli 预下载模型
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download lerobot/smolvla_base
huggingface-cli download HuggingFaceTB/SmolVLM2-500M-Video-Instruct
```

### 2.3 GPU 要求

| 训练方式 | 最小显存 | 推荐显存 | 说明 |
|----------|----------|----------|------|
| 微调预训练模型 | 16GB | 24GB+ | 冻结 vision encoder |
| 不使用 `smolvla_base`（从 SmolVLM2 初始化） | 24GB | 40GB+ | 需要 `--policy.load_vlm_weights=true` |

> 如果要全量微调（训练 VLM/视觉塔），需要设置 `--policy.train_expert_only=false` / `--policy.freeze_vision_encoder=false`，显存需求会明显增加。

### 2.4 数据集要求

SmolVLA 需要带语言指令的数据集（`task` 必填，否则 tokenizer 会报错）：
- `observation.state`: 关节状态
- `observation.images.*`: 相机图像
- `action`: 动作
- `task`: 语言任务描述

### 2.5（可选）使用 Accelerate 开启 bf16 / 多卡

SmolVLA 模型较大，显存紧张或需要多卡时，建议用 `accelerate launch`（单卡也可以）：

```bash
HF_ENDPOINT=https://hf-mirror.com accelerate launch --mixed_precision=bf16 $(which lerobot-train) \
    --policy.path=lerobot/smolvla_base \
    ...
```

更多参数说明可参考：`docs/source/multi_gpu_training.mdx`

---

## 3. 训练方式

### 3.1 方式一：微调预训练的 SmolVLA（推荐）

使用 Hugging Face Hub 上的预训练模型 `lerobot/smolvla_base`：

```bash
HF_ENDPOINT=https://hf-mirror.com lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=your_username/agilex_dataset1 \
    --output_dir=outputs/train/smolvla_agilex \
    --job_name=smolvla_agilex \
    --policy.push_to_hub=false \
    --batch_size=32 \
    --steps=100000 \
    --save_freq=10000 \
    --log_freq=100 \
    --wandb.enable=false
```

### 3.2 方式二：不使用 `smolvla_base`（从 SmolVLM2 初始化）

从 SmolVLM2 视觉语言模型初始化（需显式打开 `--policy.load_vlm_weights=true`，否则默认不加载 VLM 权重）：

```bash
HF_ENDPOINT=https://hf-mirror.com lerobot-train \
    --policy.type=smolvla \
    --dataset.repo_id=your_username/agilex_dataset1 \
    --output_dir=outputs/train/smolvla_agilex_scratch \
    --job_name=smolvla_agilex_scratch \
    --policy.load_vlm_weights=true \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --batch_size=16 \
    --steps=200000 \
    --save_freq=20000 \
    --log_freq=100 \
    --wandb.enable=false
```

> 如需全量微调，可在上述命令中额外加入：`--policy.train_expert_only=false --policy.freeze_vision_encoder=false`（显存需求更高）。

---

## 4. SmolVLA 关键参数

### 4.1 模型结构参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--policy.n_obs_steps` | 1 | 输入的观测步数 |
| `--policy.chunk_size` | 50 | 动作预测长度 |
| `--policy.n_action_steps` | 50 | 实际执行的动作步数 |
| `--policy.num_steps` | 10 | Flow Matching 解码步数 |
| `--policy.max_state_dim` | 32 | 最大状态维度（自动 padding）|
| `--policy.max_action_dim` | 32 | 最大动作维度（自动 padding）|

### 4.2 微调参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--policy.freeze_vision_encoder` | true | 冻结视觉编码器 |
| `--policy.train_expert_only` | true | 只训练动作专家网络 |
| `--policy.train_state_proj` | true | 训练状态投影层 |

### 4.3 VLM Backbone 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--policy.vlm_model_name` | HuggingFaceTB/SmolVLM2-500M-Video-Instruct | VLM 模型 |
| `--policy.num_vlm_layers` | 16 | 使用的 VLM 层数 |
| `--policy.expert_width_multiplier` | 0.75 | 动作专家宽度比例 |

### 4.4 训练超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--policy.optimizer_lr` | 1e-4 | 学习率 |
| `--policy.scheduler_warmup_steps` | 1000 | 预热步数 |
| `--policy.scheduler_decay_steps` | 30000 | 衰减步数 |

---

## 5. 低显存配置（适合 16GB GPU）

```bash
HF_ENDPOINT=https://hf-mirror.com lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=your_username/agilex_dataset1 \
    --output_dir=outputs/train/smolvla_agilex_lite \
    --job_name=smolvla_agilex_lite \
    --policy.push_to_hub=false \
    --batch_size=8 \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=true \
    --policy.chunk_size=30 \
    --policy.n_action_steps=30 \
    --steps=50000 \
    --wandb.enable=false
```

---

## 6. 训练监控

```bash
# 启用 Weights & Biases
HF_ENDPOINT=https://hf-mirror.com lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=your_username/agilex_dataset1 \
    --output_dir=outputs/train/smolvla_agilex \
    --policy.push_to_hub=false \
    --wandb.enable=true \
    --wandb.project=agilex_smolvla
```

---

## 7. 从断点恢复训练

```bash
HF_ENDPOINT=https://hf-mirror.com lerobot-train \
    --config_path=outputs/train/smolvla_agilex/checkpoints/last/pretrained_model/train_config.json \
    --resume=true
```

---

## 8. SmolVLA vs ACT vs Diffusion 对比

| 特性 | SmolVLA | ACT | Diffusion |
|------|---------|-----|-----------|
| 模型类型 | VLA (视觉-语言-动作) | Transformer | 扩散模型 |
| 语言指令支持 | ✅ 原生支持 | ❌ | ❌ |
| 预训练基础 | SmolVLM2 | 无 | 无 |
| 显存需求 | 高 | 中 | 中高 |
| 泛化能力 | 强（VLM 预训练）| 中 | 中 |
| 推理速度 | 中 | 快 | 慢 |

---

## 9. 常见问题

| 问题 | 解决方案 |
|------|----------|
| CUDA out of memory | 减小 batch_size，启用 freeze_vision_encoder |
| 找不到 smolvla 模块 | 确保运行 `pip install -e ".[smolvla]"` |
| 下载预训练模型失败 | 设置 `export HF_ENDPOINT=https://hf-mirror.com` |
| 下载超时/连接失败 | 使用 `huggingface-cli download` 手动预下载 |
| tokenizer 加载失败 | 确保 SmolVLM2 模型已完整下载 |
| `output_dir` 已存在且未 resume | 更换 `--output_dir`，或使用 `--resume=true --config_path=.../train_config.json` |

---

**下一步：完成训练后，使用 `lerobot-eval` 或 `lerobot-record --policy.path=...` 在真实机器人上测试。**
