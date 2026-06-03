# GROOT 训练 Playbook（可交付版）

## 1. 文档目的

本手册用于把 `GROOT` 在本仓库中的训练流程标准化，目标是：

- 给到任何一位工程师，都能按步骤独立完成训练。
- 过程可复现、可排障、可交接。
- 默认适配本次已验证成功的场景：`admin123/grasp_block_in_bin1`，训练到 `15000` 步。

本手册基于一次完整实操沉淀：已在本机成功跑到 `15000` 步并产出 `015000` checkpoint。

## 2. 适用范围与前提

### 2.1 适用代码仓库

- 仓库根目录：`/data/cqy_workspace/flexible_lerobot`

### 2.2 适用环境

- Linux + NVIDIA GPU（建议 CUDA 可用）
- Python 3.10
- conda 环境名称：`lerobot_flex`

### 2.3 本次推荐数据与模型

- 数据集：
  - `repo_id`: `admin123/grasp_block_in_bin1`
  - `root`: `/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin1`
- GROOT 基模（ModelScope）：
  - `nv-community/GR00T-N1.5-3B`

## 3. 资源映射（一定要先理解）

### 3.1 GROOT 基模权重（大文件，约 5.1G）

- 推荐训练入口路径（本地）：
  - `/data/cqy_workspace/flexible_lerobot/assets/modelscope/GR00T-N1.5-3B`
- 常见真实落盘路径（ModelScope 会把 `N1.5` 转义为 `N1___5`）：
  - `/data/cqy_workspace/flexible_lerobot/assets/modelscope/nv-community/GR00T-N1___5-3B`

### 3.2 Eagle 处理器资产（小文件，约几 MB）

默认配置字段（见 `src/lerobot/policies/groot/configuration_groot.py`）：

- `tokenizer_assets_repo = "lerobot/eagle2hg-processor-groot-n1p5"`

运行时实际缓存路径（默认）：

- `/home/cqy/.cache/huggingface/lerobot/lerobot/eagle2hg-processor-groot-n1p5`

重要说明：

- 训练不是“完全不需要 Eagle 资产”。
- 代码会在模型初始化时自动准备该目录：
  - 先把仓库内 vendored Eagle 文件拷贝到缓存目录。
  - 再补齐缺失的 tokenizer/config 资产。
- 因此即使你不手动准备，训练阶段也会自动用到它。

### 3.3 数据集视频解码后端

本数据集视频为 AV1。建议优先级：

1. `torchcodec`（若系统库兼容）
2. `pyav`（当 `torchcodec` 依赖冲突时）

本次已验证可稳定跑通的是：`--dataset.video_backend=pyav`。

## 4. 一键总览（先看这个）

1. 创建并激活 `lerobot_flex`。  
2. 安装本仓库与 GROOT 依赖。  
3. 从 ModelScope 下载 GROOT 基模到当前仓库。  
4. 验证数据集与模型关键文件。  
5. 启动训练（15000 步，保存频率 2000）。  
6. 监控日志到 10000 和 15000。  
7. 检查 `015000` checkpoint。  

## 5. 详细执行步骤（逐条可复制）

### 步骤 0：进入仓库

```bash
cd /data/cqy_workspace/flexible_lerobot
```

### 步骤 1：创建 conda 环境

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -y -n lerobot_flex python=3.10
conda activate lerobot_flex
python --version
which python
```

预期：

- Python 3.10.x
- python 路径在 `.../miniconda3/envs/lerobot_flex/bin/python`

### 步骤 2：安装依赖

```bash
python -m pip install -U pip setuptools wheel
python -m pip install -e ".[groot]"
```

验证：

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count())"
python -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
python -m pip show lerobot | sed -n '1,40p'
```

若 `flash_attn` 因 GLIBC 报错，执行修复：

```bash
python -m pip uninstall -y flash-attn
FLASH_ATTENTION_FORCE_BUILD=TRUE python -m pip install --no-build-isolation --no-cache-dir flash-attn==2.8.3
python -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
```

### 步骤 3：下载基模到当前仓库（ModelScope）

先安装 modelscope：

```bash
python -m pip install -U modelscope
```

下载并固定训练入口路径：

```bash
python - <<'PY'
from pathlib import Path
from modelscope.hub.snapshot_download import snapshot_download

repo_id = "nv-community/GR00T-N1.5-3B"
cache_dir = Path("/data/cqy_workspace/flexible_lerobot/assets/modelscope")
cache_dir.mkdir(parents=True, exist_ok=True)

out = snapshot_download(model_id=repo_id, cache_dir=str(cache_dir))
print("download_path =", out)

link = cache_dir / "GR00T-N1.5-3B"
real = Path(out)
if link.exists() or link.is_symlink():
    link.unlink()
link.symlink_to(real, target_is_directory=True)
print("symlink =", link, "->", real)
PY
```

校验关键文件：

```bash
MODEL=/data/cqy_workspace/flexible_lerobot/assets/modelscope/GR00T-N1.5-3B
ls -lh \
  "$MODEL/config.json" \
  "$MODEL/model-00001-of-00003.safetensors" \
  "$MODEL/model-00002-of-00003.safetensors" \
  "$MODEL/model-00003-of-00003.safetensors" \
  "$MODEL/model.safetensors.index.json"
du -sh "$MODEL" "$(readlink -f "$MODEL")"
```

### 步骤 4：可选预下载 Eagle 资产（ModelScope）

说明：

- 这一步可选，训练时代码会自动准备 Eagle 缓存。
- 若希望“训练前就确认资产可取到”，执行如下命令。

```bash
python - <<'PY'
from pathlib import Path
from modelscope.hub.snapshot_download import snapshot_download

repo_id = "lerobot/eagle2hg-processor-groot-n1p5"
cache_dir = Path("/data/cqy_workspace/flexible_lerobot/assets/modelscope")
cache_dir.mkdir(parents=True, exist_ok=True)

out = snapshot_download(model_id=repo_id, cache_dir=str(cache_dir))
print("download_path =", out)
PY
```

训练时实际使用的默认缓存目录仍是：

- `/home/cqy/.cache/huggingface/lerobot/lerobot/eagle2hg-processor-groot-n1p5`

可直接查看：

```bash
ls -la /home/cqy/.cache/huggingface/lerobot/lerobot/eagle2hg-processor-groot-n1p5 | sed -n '1,120p'
```

### 步骤 5：验证数据集

```bash
DATA_ROOT=/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin1
test -d "$DATA_ROOT" && echo "[OK] $DATA_ROOT" || echo "[MISSING] $DATA_ROOT"
find "$DATA_ROOT/meta" -maxdepth 2 -type f | sort
find "$DATA_ROOT/data" -maxdepth 3 -type f | sort | sed -n '1,20p'
find "$DATA_ROOT/videos" -maxdepth 4 -type f | sort | sed -n '1,20p'
```

检查 AV1：

```bash
cat "$DATA_ROOT/meta/info.json" | sed -n '1,120p'
```

### 步骤 6：启动训练（推荐可复现命令）

说明：

- 本命令使用 `pyav` 后端（已验证可规避 torchcodec 动态库冲突）。
- 默认关闭 `push_to_hub`，避免训练结束后因权限问题导致非零退出。

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot_flex
cd /data/cqy_workspace/flexible_lerobot

RUN_ID=$(date +%Y%m%d_%H%M%S)
OUT_DIR="/data/cqy_workspace/flexible_lerobot/outputs/train/groot_grasp_block_in_bin1_playbook_${RUN_ID}"
LOG="${OUT_DIR}.log"

PYTHONUNBUFFERED=1 HF_ENDPOINT=https://hf-mirror.com lerobot-train \
  --policy.type=groot \
  --policy.repo_id=robotech/groot \
  --dataset.repo_id=admin123/grasp_block_in_bin1 \
  --dataset.root=/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin1 \
  --dataset.video_backend=pyav \
  --batch_size=32 \
  --steps=15000 \
  --output_dir="$OUT_DIR" \
  --job_name=groot_grasp_block_in_bin1_playbook \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.base_model_path=/data/cqy_workspace/flexible_lerobot/assets/modelscope/GR00T-N1.5-3B \
  --save_freq=2000 \
  --eval_freq=20000 \
  --policy.use_bf16=true \
  --policy.push_to_hub=false \
  2>&1 | tee "$LOG"

echo "OUT_DIR=$OUT_DIR"
echo "LOG=$LOG"
```

### 步骤 7：训练参数说明（关键项）

| 参数 | 推荐值 | 说明 |
|---|---|---|
| `--policy.type` | `groot` | 使用 GROOT policy |
| `--dataset.repo_id` | `admin123/grasp_block_in_bin1` | 训练数据集 |
| `--dataset.root` | 本地缓存路径 | 指定本地数据，避免重复下载 |
| `--dataset.video_backend` | `pyav` | AV1 场景更稳（本机验证） |
| `--policy.base_model_path` | 本地 ModelScope 路径 | 使用本地基模 |
| `--batch_size` | `32` | 本机 4090 48G 实测可跑 |
| `--steps` | `15000` | 本次目标步数 |
| `--save_freq` | `2000` | 每 2k 步存一次 |
| `--policy.use_bf16` | `true` | 训练效率与显存平衡 |
| `--policy.push_to_hub` | `false` | 推荐默认关闭，避免收尾权限报错 |

### 步骤 8：训练监控

实时看 step/loss：

```bash
tail -f "$LOG"
```

提取关键行：

```bash
tr '\r' '\n' < "$LOG" | rg -n "step:|Checkpoint policy|End of training|ERROR|Traceback" -i | tail -n 80
```

看 GPU：

```bash
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv,noheader
```

### 步骤 9：验收标准

通过条件：

1. 日志出现 `step:10K`。  
2. 日志出现 `step:15K`。  
3. 日志出现 `Checkpoint policy after step 15000`。  
4. 输出目录存在 `checkpoints/015000`。  

验收命令：

```bash
find "$OUT_DIR/checkpoints" -maxdepth 1 -mindepth 1 -type d | sort
test -d "$OUT_DIR/checkpoints/015000" && echo "PASS: checkpoint 015000 exists"
```

### 步骤 10：断点续训

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot_flex

lerobot-train \
  --resume=true \
  --config_path="$OUT_DIR/checkpoints/last/pretrained_model/train_config.json"
```

## 6. 常见问题与处理

### 6.1 `FileExistsError`（output_dir 已存在）

现象：

- 启动即失败，提示 output dir exists。

处理：

- 更换新的 `--output_dir`，不要手工提前创建训练输出目录。

### 6.2 torchcodec 动态库错误（FFmpeg/GLIBC）

现象：

- `Could not load libtorchcodec` 或 `libavutil/libffi` 相关报错。

处理：

- 改用 `--dataset.video_backend=pyav`。

### 6.3 训练末尾 403 Forbidden（push hub）

现象：

- 训练已经完成，但最后抛 403，进程非零退出。

原因：

- 无权限在目标 namespace（如 `robotech`）创建仓库。

处理：

1. 默认加 `--policy.push_to_hub=false`。  
2. 若必须推送，换有权限的 token 或 repo_id。  

### 6.4 `flash_attn` 导入失败（GLIBC 不兼容）

处理：

```bash
python -m pip uninstall -y flash-attn
FLASH_ATTENTION_FORCE_BUILD=TRUE python -m pip install --no-build-isolation --no-cache-dir flash-attn==2.8.3
```

### 6.5 OOM（显存不足）

处理顺序：

1. `--batch_size 32 -> 16 -> 8 -> 4`  
2. 关闭不必要进程，确认独占 GPU。  

## 7. 交付物清单（训练完成后）

至少交付以下路径给下游同事：

1. 训练输出目录（含 checkpoints）  
2. 训练日志 `.log`  
3. `checkpoints/015000/pretrained_model/train_config.json`  

推荐附带：

- 一段 step/loss 摘要（10K 与 15K）
- 使用的命令行原文

## 8. 标准复盘模板（可复制）

```text
Run ID:
Start Time:
End Time:
Env:
Dataset:
Base Model:
Video Backend:
Batch Size:
Steps Target:
Steps Reached:
Checkpoint Path:
Any Errors:
Final Status:
```

## 9. 本机已验证基线（供估时）

- 机器：RTX 4090 48GB（单卡）
- 命令：`batch_size=32`, `steps=15000`, `video_backend=pyav`
- 实测：可稳定跑完至 `15000` 步并生成 `015000`
- 粗略耗时：约 4 小时 50 分钟（含 checkpoint 保存）

## 10. 版本建议

为保证可复现，建议每次训练都记录：

1. 当前仓库 commit hash  
2. conda env 导出（`conda env export -n lerobot_flex`）  
3. 训练命令全文  
4. 模型与数据绝对路径  

