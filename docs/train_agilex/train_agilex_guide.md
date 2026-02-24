# Agilex Piper 双臂机器人训练指南

本文档指导你使用 ACT 和 Diffusion Policy 训练 Agilex Piper 双臂机器人的策略模型。

---

## 1. 前置准备

### 1.1 数据集要求

确保你已经使用 `lerobot-record` 录制了数据集，数据集应包含：
- `observation.state`: 14 维关节状态（双臂各 7 个关节）
- `observation.images.*`: 相机图像
- `action`: 14 维动作

```bash
# 验证数据集
cat ~/.cache/huggingface/lerobot/your_username/agilex_dataset1/meta/info.json
```

### 1.2 GPU 要求

| 模型 | 最小显存 | 推荐显存 | 训练时间参考 |
|------|----------|----------|--------------|
| ACT | 8GB | 16GB+ | ~3-6 小时 (100k steps) |
| Diffusion | 10GB | 24GB+ | ~6-12 小时 (100k steps) |

---

## 2. ACT (Action Chunking with Transformers) 训练

### 2.1 基础训练命令

```bash
lerobot-train \
    --dataset.repo_id=your_username/agilex_dataset1 \
    --policy.type=act \
    --output_dir=outputs/train/act_agilex \
    --job_name=act_agilex \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --batch_size=8 \
    --steps=100000 \
    --save_freq=10000 \
    --eval_freq=10000 \
    --log_freq=100 \
    --wandb.enable=false
```

### 2.2 ACT 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--policy.chunk_size` | 100 | 动作预测长度（帧数） |
| `--policy.n_action_steps` | 100 | 实际执行的动作步数 |
| `--policy.dim_model` | 512 | Transformer 隐藏层维度 |
| `--policy.n_heads` | 8 | 注意力头数 |
| `--policy.n_encoder_layers` | 4 | 编码器层数 |
| `--policy.use_vae` | true | 是否使用 VAE |
| `--policy.kl_weight` | 10.0 | KL 散度损失权重 |

### 2.3 轻量化 ACT（适合低显存）

```bash
lerobot-train \
    --dataset.repo_id=your_username/agilex_dataset1 \
    --policy.type=act \
    --output_dir=outputs/train/act_agilex_lite \
    --job_name=act_agilex_lite \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.dim_model=256 \
    --policy.chunk_size=50 \
    --policy.n_action_steps=50 \
    --batch_size=4 \
    --steps=50000 \
    --wandb.enable=false
```

---

## 3. Diffusion Policy 训练

### 3.1 基础训练命令

```bash
lerobot-train \
    --dataset.repo_id=your_username/agilex_dataset1 \
    --policy.type=diffusion \
    --output_dir=outputs/train/diffusion_agilex \
    --job_name=diffusion_agilex \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --batch_size=8 \
    --steps=100000 \
    --save_freq=10000 \
    --eval_freq=10000 \
    --log_freq=100 \
    --wandb.enable=false
```

### 3.2 Diffusion Policy 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--policy.n_obs_steps` | 2 | 输入的观测帧数 |
| `--policy.horizon` | 16 | 预测的 action 长度 |
| `--policy.n_action_steps` | 8 | 实际执行的动作步数 |
| `--policy.num_train_timesteps` | 100 | 扩散训练步数 |
| `--policy.num_inference_steps` | null | 推理去噪步数（默认=训练步数）|
| `--policy.down_dims` | (512,1024,2048) | UNet 各层通道数 |

### 3.3 轻量化 Diffusion（适合低显存）

```bash
lerobot-train \
    --dataset.repo_id=your_username/agilex_dataset1 \
    --policy.type=diffusion \
    --output_dir=outputs/train/diffusion_agilex_lite \
    --job_name=diffusion_agilex_lite \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.down_dims='(256, 512, 1024)' \
    --policy.num_train_timesteps=50 \
    --batch_size=4 \
    --steps=50000 \
    --wandb.enable=false
```

---

## 4. 训练监控

### 4.1 使用 Weights & Biases（推荐）

```bash
# 首次使用需要登录
wandb login

# 启用 wandb 监控
lerobot-train \
    --dataset.repo_id=your_username/agilex_dataset1 \
    --policy.type=act \
    --output_dir=outputs/train/act_agilex \
    --policy.push_to_hub=false \
    --wandb.enable=true \
    --wandb.project=agilex_training
```

### 4.2 查看训练日志

```bash
# 查看输出目录结构
ls -la outputs/train/act_agilex/

# 检查 checkpoints
ls outputs/train/act_agilex/checkpoints/
```

---

## 5. 从断点恢复训练

```bash
lerobot-train \
    --config_path=outputs/train/act_agilex/checkpoints/last/pretrained_model/train_config.json \
    --resume=true
```

---

## 6. 训练完成后

### 6.1 模型位置

训练完成后，模型保存在：
```
outputs/train/<job_name>/checkpoints/
├── 010000/          # step 10000 的 checkpoint
├── 020000/          # step 20000 的 checkpoint
├── ...
└── last/            # 最后一个 checkpoint
    └── pretrained_model/
        ├── config.json
        ├── model.safetensors
        └── train_config.json
```

### 6.2 推送到 HuggingFace Hub（可选）

```bash
lerobot-train \
    --dataset.repo_id=your_username/agilex_dataset1 \
    --policy.type=act \
    --output_dir=outputs/train/act_agilex \
    --policy.push_to_hub=true \
    --policy.repo_id=your_username/act_agilex_policy
```

---

## 7. 常见问题

| 问题 | 解决方案 |
|------|----------|
| CUDA out of memory | 减小 `batch_size` 或使用轻量化配置 |
| 训练 loss 不下降 | 检查数据集质量，增加训练步数 |
| 找不到数据集 | 确认 `repo_id` 路径正确 |

---

**下一步：完成训练后，请参考部署指南在真实机器人上评估策略。**
