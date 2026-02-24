# Agilex Piper 双臂机器人 GR00T N1.5 训练指南

本文档以本仓库当前实现为准（`lerobot-train` / `GrootPolicy` / `processor_groot.py`），重点修正：
- GR00T 的 `action_horizon` 在实现中**硬上限为 16**（`min(chunk_size, 16)`），设置更大不会生效。
- `GrootConfig` 里虽然有 `lora_*` 字段，但当前训练路径**没有接入 LoRA**；实际可控的是 `tune_llm/tune_visual/tune_projector/tune_diffusion_model`。
- GR00T 目前依赖 Flash Attention（CUDA 环境），安装不对会 import 报错或 fallback 变慢。

---

## 0. 先确认你的数据（本次将用于训练的两套数据）

- `cqy/agilex_both_side_box`：3 cameras，`observation.state/action`=14（双臂）
- `cqy/agilex_left_box`：2 cameras，`observation.state/action`=7（左臂）

GR00T 预处理会把所有 `observation.images.*` 相机按 key 排序后堆成多视角 video 输入（V 维度），并从样本里的 `task` 作为语言输入。

---

## 1. 安装与下载（GR00T 的坑主要在这里）

### 1.1 依赖（在 my_lerobot 环境）

```bash
pip install -e ".[groot]"
```

### 1.2 Flash Attention（CUDA 必需）

详细版本匹配请以 `docs/source/groot.mdx` 为准。常见安装方式（示例）：
```bash
pip install ninja "packaging>=24.2,<26.0"
pip install "flash-attn>=2.5.9,<3.0.0" --no-build-isolation
python -c "import flash_attn; print(f'flash_attn {flash_attn.__version__} ok')"
```

### 1.3 Hugging Face 镜像（国内常用）

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 1.4 需要的模型资产

- `nvidia/GR00T-N1.5-3B`：基础模型
- `lerobot/eagle2hg-processor-groot-n1p5`：Eagle tokenizer/processor 资源

可选预下载：
```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download nvidia/GR00T-N1.5-3B
huggingface-cli download lerobot/eagle2hg-processor-groot-n1p5
```

---

## 2. 关键配置点（保证“训练是对的”）

### 2.1 action horizon 必须是 16

本仓库实现里：
- `processor_groot.py` 强制 `action_horizon = min(chunk_size, 16)`
- `GrootConfig.action_delta_indices` 也强制 `min(chunk_size, 16)`

所以请直接设置：
```bash
--policy.chunk_size=16 --policy.n_action_steps=16
```

### 2.2 不要把当前实现当作 LoRA 微调

当前 `lora_rank/lora_alpha/lora_dropout` 没有被 `GrootPolicy` 使用。显存与训练范围主要靠：
- 冻结：`--policy.tune_llm=false --policy.tune_visual=false`
- 训练：`--policy.tune_projector=true --policy.tune_diffusion_model=true`

---

## 3. 推荐训练命令（对应你的两套数据）

### 3.1 双臂数据：`cqy/agilex_both_side_box`

```bash
HF_ENDPOINT=https://hf-mirror.com lerobot-train \
  --dataset.repo_id=cqy/agilex_both_side_box \
  --policy.type=groot \
  --policy.device=cuda \
  --policy.base_model_path=nvidia/GR00T-N1.5-3B \
  --policy.tokenizer_assets_repo=lerobot/eagle2hg-processor-groot-n1p5 \
  --policy.chunk_size=16 \
  --policy.n_action_steps=16 \
  --policy.use_bf16=true \
  --policy.tune_llm=false \
  --policy.tune_visual=false \
  --policy.tune_projector=true \
  --policy.tune_diffusion_model=true \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/groot_agilex_both_side_box \
  --job_name=groot_agilex_both_side_box \
  --batch_size=4 \
  --steps=20000 \
  --save_freq=5000 \
  --log_freq=100 \
  --wandb.enable=false
```

### 3.2 单臂数据：`cqy/agilex_left_box`

```bash
HF_ENDPOINT=https://hf-mirror.com lerobot-train \
  --dataset.repo_id=cqy/agilex_left_box \
  --policy.type=groot \
  --policy.device=cuda \
  --policy.base_model_path=nvidia/GR00T-N1.5-3B \
  --policy.tokenizer_assets_repo=lerobot/eagle2hg-processor-groot-n1p5 \
  --policy.chunk_size=16 \
  --policy.n_action_steps=16 \
  --policy.use_bf16=true \
  --policy.tune_llm=false \
  --policy.tune_visual=false \
  --policy.tune_projector=true \
  --policy.tune_diffusion_model=true \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/groot_agilex_left_box \
  --job_name=groot_agilex_left_box \
  --batch_size=4 \
  --steps=20000 \
  --save_freq=5000 \
  --log_freq=100 \
  --wandb.enable=false
```

---

## 4. 重要提醒：单臂数据集与真实双臂机器人

`cqy/agilex_left_box` 的 action 只有左臂 7 维、且是**绝对关节位置**。在真实双臂 AgileX 上直接执行会缺少 `right_*` 目标关节位置（需要额外补齐/hold 逻辑）。

---

## 5. 从断点恢复训练

```bash
lerobot-train \
  --config_path=outputs/train/groot_agilex_both_side_box/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

建议先离线验证（回放/可视化），再上真机。
