# GROOT 复刻训练工作报告（grasp_block_in_bin1）

## 1. 报告概览

- 报告时间：2026-03-03
- 执行人：Codex（本次会话）
- 目标：
  - 在当前仓库重新创建环境 `lerobot_flex`
  - 复用既有数据集 `admin123/grasp_block_in_bin1`
  - 将基模从 ModelScope 下载到当前路径
  - 启动并全程监控训练，至少盯到 10000 步，并完成 15000 步

## 2. 最终结论

- 训练主目标达成：
  - 已到达并超过 10000 步
  - 已完成 15000 步并保存 `015000` checkpoint
- 进程退出码为 `1`，但原因发生在训练结束后：
  - `push_model_to_hub` 阶段触发 Hugging Face 403 权限错误（`robotech` namespace 无创建权限）
  - 本地训练结果与 checkpoint 不受影响

## 3. 关键路径

- 原始参考文档：
  - `/data/cqy_workspace/flexible_lerobot/my_devs/docs/groot_train/groot_grasp_block_in_bin1.md`
- 本次训练根目录：
  - `/data/cqy_workspace/flexible_lerobot/outputs/train/groot_grasp_block_in_bin1_repro_20260302_223413`
- 本次有效训练输出目录：
  - `/data/cqy_workspace/flexible_lerobot/outputs/train/groot_grasp_block_in_bin1_repro_20260302_223413/bs32_20260302_223447`
- 本次训练日志：
  - `/data/cqy_workspace/flexible_lerobot/outputs/train/groot_grasp_block_in_bin1_repro_20260302_223413/bs32_20260302_223447.log`
- 本次状态文件：
  - `/data/cqy_workspace/flexible_lerobot/outputs/train/groot_grasp_block_in_bin1_repro_20260302_223413/monitor_status.txt`

### 3.1 模型权重落盘位置（详细）

- ModelScope 下载目标（训练时实际使用的模型入口路径）：
  - `/data/cqy_workspace/flexible_lerobot/assets/modelscope/GR00T-N1.5-3B`
- 说明：
  - 上述路径是软链接（link），训练时可直接使用该路径
  - 软链接真实指向目录为：
    - `/data/cqy_workspace/flexible_lerobot/assets/modelscope/nv-community/GR00T-N1___5-3B`
- 模型关键文件（已核验存在）：
  - `/data/cqy_workspace/flexible_lerobot/assets/modelscope/GR00T-N1.5-3B/config.json`（1.7K）
  - `/data/cqy_workspace/flexible_lerobot/assets/modelscope/GR00T-N1.5-3B/model-00001-of-00003.safetensors`（1.9G）
  - `/data/cqy_workspace/flexible_lerobot/assets/modelscope/GR00T-N1.5-3B/model-00002-of-00003.safetensors`（1.9G）
  - `/data/cqy_workspace/flexible_lerobot/assets/modelscope/GR00T-N1.5-3B/model-00003-of-00003.safetensors`（1.4G）
  - `/data/cqy_workspace/flexible_lerobot/assets/modelscope/GR00T-N1.5-3B/model.safetensors.index.json`（103K）
- 目录体量说明：
  - `du -sh /data/cqy_workspace/flexible_lerobot/assets/modelscope/GR00T-N1.5-3B` 显示 `4.0K`（仅软链接本身大小）
  - 真实模型目录体量 `5.1G`（真实指向目录）

### 3.2 模型下载执行方式（详细）

- 执行环境：
  - `conda activate lerobot_flex`
- 下载方式：
  - 通过 `modelscope` 的 `snapshot_download` 下载 `nv-community/GR00T-N1.5-3B`
  - `cache_dir` 指向仓库内 `assets/modelscope`
- 参考执行片段：

```python
from modelscope.hub.snapshot_download import snapshot_download

snapshot_download(
    "nv-community/GR00T-N1.5-3B",
    cache_dir="assets/modelscope",
)
```

### 3.3 数据集与视频后端（详细）

- 训练数据集：
  - `repo_id`: `admin123/grasp_block_in_bin1`
  - `root`: `/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin1`
- 视频后端选择：
  - 初始按照报告尝试 `torchcodec`
  - 因系统动态库冲突导致失败后，切换为 `pyav`
  - `pyav` 已实测可正确解码该数据集 AV1 视频并完成全程训练

## 4. 逐步执行记录（每一步 + 结果）

### 步骤 1：读取参考文档并提取关键资源

- 动作：
  - 读取 `groot_grasp_block_in_bin1.md`
  - 提取数据集路径、基模路径、Eagle 资产路径、训练参数
- 结果：
  - 数据集、基模、Eagle 路径均确认存在
  - 基模权重分片齐全（3 个 safetensors + index）

### 步骤 2：复核本机资源可用性

- 动作：
  - 核查数据集目录结构：`data/`、`meta/`、`videos/`
  - 核查 `meta/info.json`，确认视频编码为 AV1
  - 核查基模目录体积与权重分片
- 结果：
  - 数据集可用，结构完整
  - 基模可用，权重完整，体量约 5.1G

### 步骤 3：创建新环境 `lerobot_flex`

- 动作：
  - 新建 conda 环境 `lerobot_flex`（Python 3.10）
  - 安装项目依赖与 GR00T 依赖：`pip install -e ".[groot]"`
  - 验证 `torch/cuda/flash_attn`
- 结果：
  - 环境创建成功
  - `torch` CUDA 可用
  - `flash_attn` 初次导入遇到 GLIBC 兼容问题，已通过源码重编译修复
  - 最终 `flash_attn` 可正常导入

### 步骤 4：从 ModelScope 下载基模到当前路径

- 动作：
  - 在 `lerobot_flex` 安装 `modelscope`
  - 下载 `nv-community/GR00T-N1.5-3B` 到仓库 `assets/modelscope`
  - 建立标准软链接 `assets/modelscope/GR00T-N1.5-3B`
- 结果：
  - 下载来源为 ModelScope 直下（非缓存兜底）
  - 关键文件齐全：
    - `config.json`
    - `model-00001-of-00003.safetensors`
    - `model-00002-of-00003.safetensors`
    - `model-00003-of-00003.safetensors`
    - `model.safetensors.index.json`

### 步骤 5：首次训练尝试（失败，启动策略问题）

- 动作：
  - 以自动容错脚本启动训练（15000 步，bs=32 起）
- 结果：
  - 训练尚未开始即失败
  - 原因：`output_dir` 预先存在且 `resume=false`，触发 `FileExistsError`
  - 处理：修复启动逻辑，不再预创建 `lerobot-train` 输出目录

### 步骤 6：第二次训练尝试（失败，视频后端依赖冲突）

- 动作：
  - 修复目录逻辑后再次启动
- 结果：
  - 进入训练初始化后失败
  - 原因：`torchcodec` 动态库加载失败（FFmpeg/GLIBC 依赖冲突）
  - 处理：验证替代后端可行性

### 步骤 7：后端切换验证与最终训练启动

- 动作：
  - 用 `pyav` 实测读取 AV1 视频，解码成功
  - 训练参数切换 `--dataset.video_backend=pyav`
  - 重新启动并进入长时监控
- 结果：
  - 训练稳定持续推进，GPU 持续高利用
  - 批大小稳定为 `32`，无 OOM 重试

### 步骤 8：全自动监控到 15000 步

- 动作：
  - 周期性巡检日志、GPU、checkpoint 目录
  - 关键里程碑核验：2K/4K/6K/8K/10K/12K/14K/15K
- 结果：
  - `10000` 步达到并保存 checkpoint
  - `15000` 步达到并保存 checkpoint
  - 训练日志明确出现 `End of training`
  - 结束后在 push hub 阶段报 403，导致退出码为 1

## 4.1 最终训练命令与参数（详细）

### 4.1.1 最终执行命令（实际生效版本）

```bash
PYTHONUNBUFFERED=1 HF_ENDPOINT=https://hf-mirror.com lerobot-train \
  --policy.type=groot \
  --policy.repo_id=robotech/groot \
  --dataset.repo_id=admin123/grasp_block_in_bin1 \
  --dataset.root=/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin1 \
  --dataset.video_backend=pyav \
  --batch_size=32 \
  --steps=15000 \
  --output_dir=/data/cqy_workspace/flexible_lerobot/outputs/train/groot_grasp_block_in_bin1_repro_20260302_223413/bs32_20260302_223447 \
  --job_name=groot_grasp_block_in_bin1_repro \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.base_model_path=/data/cqy_workspace/flexible_lerobot/assets/modelscope/GR00T-N1.5-3B \
  --save_freq=2000 \
  --eval_freq=20000 \
  --policy.use_bf16=true
```

### 4.1.2 参数逐项说明

| 参数 | 本次取值 | 用途 |
|---|---|---|
| `--policy.type` | `groot` | 指定 GROOT 策略训练 |
| `--policy.repo_id` | `robotech/groot` | 策略 repo 标识；训练末尾会用于 push hub |
| `--dataset.repo_id` | `admin123/grasp_block_in_bin1` | 数据集标识 |
| `--dataset.root` | `/home/cqy/.cache/huggingface/lerobot/admin123/grasp_block_in_bin1` | 数据本地根目录 |
| `--dataset.video_backend` | `pyav` | 视频解码后端（本次为兼容性修复后的有效设置） |
| `--batch_size` | `32` | 每步 batch 大小 |
| `--steps` | `15000` | 训练总步数目标 |
| `--output_dir` | `.../bs32_20260302_223447` | 训练输出目录 |
| `--job_name` | `groot_grasp_block_in_bin1_repro` | 任务名 |
| `--policy.device` | `cuda` | GPU 训练 |
| `--wandb.enable` | `false` | 禁用 wandb 在线记录 |
| `--policy.base_model_path` | `/data/cqy_workspace/flexible_lerobot/assets/modelscope/GR00T-N1.5-3B` | 指向本地基模入口路径 |
| `--save_freq` | `2000` | 每 2000 步保存 checkpoint |
| `--eval_freq` | `20000` | 评估周期（大于总步数，本次训练中不触发评估） |
| `--policy.use_bf16` | `true` | bf16 混合精度训练 |
| `PYTHONUNBUFFERED=1` | 已设置 | 日志实时刷新 |
| `HF_ENDPOINT=https://hf-mirror.com` | 已设置 | 使用镜像端点 |

### 4.1.3 尝试轮次与参数差异

- 第 1 轮失败：
  - 问题：`output_dir` 预创建导致 `FileExistsError`
  - 参数层面：核心参数基本一致
- 第 2 轮失败：
  - 问题：`torchcodec` 动态库冲突
  - 参数差异：视频后端未切换
- 第 3 轮成功跑到 15000：
  - 关键变更：`--dataset.video_backend=pyav`
  - 其余训练超参保持与复刻目标一致

## 5. 训练里程碑（摘自日志）

- Start offline training：`2026-03-02 22:34:53`
- `step:1K`：`2026-03-02 22:54:05`（loss 0.029）
- `step:2K`：`2026-03-02 23:05:34`（loss 0.026）
- `step:4K`：`2026-03-02 23:44:05`（loss 0.018）
- `step:6K`：`2026-03-03 00:22:35`（loss 0.015）
- `step:8K`：`2026-03-03 01:01:05`（loss 0.010）
- `step:10K`：`2026-03-03 01:39:34`（loss 0.007）
- `Checkpoint step 10000`：`2026-03-03 01:47:14`
- `step:12K`：`2026-03-03 02:18:03`（loss 0.007）
- `step:14K`：`2026-03-03 02:56:31`（loss 0.006）
- `step:15K`：`2026-03-03 03:23:30`（loss 0.006）
- `Checkpoint step 15000`：`2026-03-03 03:23:30`
- `End of training`：`2026-03-03 03:23:41`

## 6. 产出物清单

- checkpoints 目录（2K 间隔 + 15K）：
  - `002000`
  - `004000`
  - `006000`
  - `008000`
  - `010000`
  - `012000`
  - `014000`
  - `015000`
- 训练目录体量：
  - `116G`（含 checkpoints）

## 7. Agent 使用统计

- 本次总共使用 Agent 数量：`6`

| Agent ID | 昵称 | 主要职责 | 结果 |
|---|---|---|---|
| `019caeda-07c4-7b33-910d-67bdb1aad946` | Aristotle | 首次环境创建任务（中途会话被用户打断） | 中断，未完成 |
| `019caedd-b85e-75b1-b0b6-1aae34dc1cf2` | Galileo | 创建 `lerobot_flex`、安装依赖、修复 flash-attn 导入问题 | 完成 |
| `019caef1-da3c-7463-b126-6b88c264f438` | Banach | 安装 modelscope、下载基模到当前仓库、校验关键文件 | 完成 |
| `019caef6-5810-7171-9d5c-0eae4acefaee` | Zeno | 第一轮训练自动容错启动（目录冲突场景） | 失败（`FileExistsError`） |
| `019caef7-4139-75f1-b6ea-9ab16fc142f0` | Bohr | 第二轮训练自动容错启动（torchcodec 场景） | 失败（动态库依赖冲突） |
| `019caef8-e00d-7cb0-b296-a52d34a1ad05` | Mill | 第三轮训练（切换 pyav）+ 长时监控到 15000 步 | 训练达成；末尾 push hub 403 |

## 8. 失败点与影响评估

- 失败点：
  - 训练结束后执行 `push_model_to_hub` 触发 403
  - 报错信息：无权限在 `robotech` namespace 下创建模型仓库
- 影响：
  - 不影响本地训练完成与 checkpoint 产出
  - 影响仅限“自动上传到 hub”这一步

## 9. 建议的后续动作

- 若只需本地复刻完成：
  - 现有结果已满足（已到 15000 步，checkpoint 完整）
- 若要避免非零退出码并可复跑：
  - 下次增加 `--policy.push_to_hub=false`
- 若要推送到 Hub：
  - 使用有目标 namespace 写权限的 token 或改为有权限的 repo_id

## 10. Token 消耗（估算）

- 精确 token 用量需依赖平台 API usage 统计
- 本次会话估算总量（含工具输出、日志、agent 通信）：约 `30万 - 60万 tokens`
