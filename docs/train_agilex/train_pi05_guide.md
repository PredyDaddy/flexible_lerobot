# Agilex Piper 双臂机器人 π₀.₅ (Pi0.5) 训练指南

本文档以本仓库当前实现为准（`lerobot-train` / `PI05Policy` / `processor_pi05.py`），重点修正：
- π₀.₅ 的「微调」推荐用法是 `--policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base`（更适配自定义相机键名/动作维度）。
- PaliGemma **只需要 tokenizer 文件**；不必下载整套 `google/paligemma-3b-pt-224` 权重。
- π₀.₅ 默认对 state/action 使用 **quantiles** 归一化，数据集必须有 `q01/q99` 等统计。

---

## 0. 先确认你的数据（本次将用于训练的两套数据）

你给的两套数据（都在 v3.0 格式、fps=30、198 episodes / 47,520 frames，且样本里会自动带 `task` 文本）：

- `cqy/agilex_both_side_box`：3 个相机（`observation.images.camera_front/left/right`），`observation.state`/`action` 维度=14（双臂）
- `cqy/agilex_left_box`：2 个相机（`observation.images.camera_front/left`），`observation.state`/`action` 维度=7（左臂）

快速自检（如遇到无法写 `~/.cache/huggingface/datasets` 的锁文件，可加 `HF_DATASETS_CACHE=/tmp/hf_datasets`）：
```bash
HF_DATASETS_CACHE=/tmp/hf_datasets conda run -n my_lerobot python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset as D; ds=D('cqy/agilex_both_side_box'); x=ds[0]; print(sorted([k for k in x if k.startswith('observation.images.')]), x['observation.state'].shape, x['action'].shape, x['task'])"
```

---

## 1. 前置准备

### 1.1 安装依赖（在 my_lerobot 环境）

```bash
pip install -e ".[pi]"
```

### 1.2 Hugging Face 镜像（国内常用）

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 2. 模型与 tokenizer 下载（重点纠错）

π₀.₅ 微调通常需要：
- `lerobot/pi05_base`：模型权重（~6GB）
- `google/paligemma-3b-pt-224`：**tokenizer 文件**（`transformers.AutoTokenizer` 会自动下载所需文件）

不建议直接执行 `huggingface-cli download google/paligemma-3b-pt-224`（会把 3B 权重也下下来，体积非常大且训练用不到）。

推荐做法：
```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download lerobot/pi05_base
conda run -n my_lerobot python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google/paligemma-3b-pt-224')"
```

---

## 3. 数据集要求（Pi0.5 的关键点）

π₀.₅ 训练时需要：
- `observation.state`
- `observation.images.*`（至少 1 个相机；LeRobot 会从视频解码当前帧并送入模型）
- `action`
- `task`（语言任务文本，LeRobot v3.0 数据集会根据 `tasks.parquet` 自动注入到样本里）

### 3.1 quantile 统计（必须有）

π₀.₅ 默认对 STATE/ACTION 使用 `NormalizationMode.QUANTILES`，因此数据集 `meta/stats.json` 必须含 `q01/q99` 等字段。

如果你的数据缺少 quantiles，可运行（注意：脚本会写回 stats 并 push 到 Hub）：
```bash
conda run -n my_lerobot python src/lerobot/datasets/v30/augment_dataset_quantile_stats.py --repo-id=your_dataset
```

或者绕开 quantiles（不推荐，但可用于先跑通流程）：
```bash
--policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}'
```

---

## 4. 推荐训练命令（对应你的两套数据）

### 4.1 双臂数据：`cqy/agilex_both_side_box`（action/state=14，3 cameras）

```bash
HF_ENDPOINT=https://hf-mirror.com lerobot-train \
  --dataset.repo_id=cqy/agilex_both_side_box \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/pi05_agilex_both_side_box \
  --job_name=pi05_agilex_both_side_box \
  --batch_size=8 \
  --steps=30000 \
  --save_freq=5000 \
  --log_freq=100 \
  --wandb.enable=false
```

### 4.2 单臂数据：`cqy/agilex_left_box`（action/state=7，2 cameras）

```bash
HF_ENDPOINT=https://hf-mirror.com lerobot-train \
  --dataset.repo_id=cqy/agilex_left_box \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/pi05_agilex_left_box \
  --job_name=pi05_agilex_left_box \
  --batch_size=8 \
  --steps=30000 \
  --save_freq=5000 \
  --log_freq=100 \
  --wandb.enable=false
```

---

## 5. 关于 `--policy.path` 与 `--rename_map`（可选，高阶）

如果你用 `--policy.path=lerobot/pi05_base`（会加载预训练 config 并更严格校验输入特征），当相机键名不一致时会抛出 feature mismatch。

此时用 `--rename_map` 把你的数据键名映射到预训练 config 期待的键名即可。映射目标键名以报错信息里 “Missing features/Extra features” 为准。例如（仅示例）：
```bash
--rename_map='{"observation.images.camera_front":"observation.images.camera1","observation.images.camera_left":"observation.images.camera2"}'
```

另外：π₀.₅ 对缺失相机会自动补「空图 + mask=0」，所以你只需要映射你真实存在的相机即可。

---

## 6. 重要提醒：单臂数据集与真实双臂机器人

`cqy/agilex_left_box` 的 `action` 只有 7 维左臂关节（且是**绝对关节位置**）。如果你把该策略直接跑在真实 AgileX 双臂机器人上：
- 如果你的真实机器人实现的 `Robot.send_action()` 需要同时包含 `left_*` 和 `right_*` 的目标关节位置；
- 只输出左臂会导致缺键报错，或你自行补零会让右臂被命令到 0 位姿（不安全）。

如果你的目标是“只训左臂但真实右臂保持不动”，建议：
- 训练/评估时在 action 侧把右臂目标设为“当前右臂状态”（hold），或
- 直接用双臂数据训练（把右臂也记录为 hold/协作），避免部署阶段再补逻辑。

---

## 7. 从断点恢复训练

```bash
lerobot-train \
  --config_path=outputs/train/pi05_agilex_both_side_box/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

训练完成后建议先用 `lerobot-eval` 离线验证（回放/可视化），再上真机。
