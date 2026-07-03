# OpenPI PI0.5 SO101 LoRA 训练验证报告

日期：2026-07-01

本文档记录如何在官方 `openpi-main` 代码上，使用之前 LeRobot PI0.5 训练流程里的同一份 SO101 桌面整理数据，对 OpenPI PI0.5 JAX LoRA 模型进行训练验证。

本次新增的代码、脚本、生成资产、日志和 checkpoint 都放在：

```bash
/data/cqy_workspace/flexible_lerobot/my_devs/openpi_train
```

原始 LeRobot 数据集只读使用，不在原目录做转换，也不原地修改。

## 源码仓库状态

`openpi-main` 现在只作为普通源码引用目录使用，不再作为一个独立关联 GitHub 的仓库。

检查命令：

```bash
find openpi-main -maxdepth 2 \( -name .git -o -name .github -o -name .gitmodules \) -print
```

结果：无输出。

这说明当前目录下没有 `.git`、`.github`、`.gitmodules`，已经断开和原 GitHub 仓库的关系。

## 旧 LeRobot 训练基线

之前 LeRobot PI0.5 的训练入口是：

```bash
/data/cqy_workspace/flexible_lerobot/my_devs/train/pi/so101/easy_train.sh
```

关键训练配置：

```text
dataset.repo_id = desk_cleanup_v1/eraser_cup_multi_task
dataset.root    = /data/cqy_workspace/flexible_lerobot/datasets/desk_cleanup_v1/eraser_cup_multi_task
policy.type     = pi05
pretrained      = /data/cqy_workspace/flexible_lerobot/assets/modelscope/lerobot/pi05_base
batch_size      = 16
steps           = 33300
```

数据集信息：

```text
codebase_version = v3.0
total_episodes   = 157
total_frames     = 53235
total_tasks      = 3
fps              = 30
state_dim        = 6
action_dim       = 6
top camera       = observation.images.top,  480x640 RGB video
wrist camera     = observation.images.wrist, 480x640 RGB video
```

任务文本：

```text
Put the eraser into the small box
Move the cup back to the upper-right corner
First put the eraser into the small box, then move the cup back to the upper-right corner
```

## OpenPI 适配方案

官方 OpenPI 的 JAX 训练栈支持 LoRA。当前官方 PyTorch 路径不支持 LoRA，因此本验证使用 JAX 训练路径。

当前保留两条方案：

```text
方案 A：直接读取 LeRobot v3 原始数据
  - 使用 openpi_so101/dataset_v3.py 只读适配器
  - 不生成训练数据副本
  - 已完成 smoke 训练和推理验证

方案 B：把前几个 episode 转成 LeRobot v2.1 pilot 副本
  - 输出到 my_devs/openpi_train/data/...
  - 原始数据不动
  - 通过对齐检查后，再使用转换数据跑 OpenPI LoRA smoke
```

官方 OpenPI 的数据加载代码使用的是它固定版本里较旧的 LeRobot 导入路径，而当前数据集是 LeRobot v3 格式。为了不修改原始数据，也不直接改 `openpi-main` 源码，这里在本地 wrapper 里增加了一个运行时数据加载适配层：

```text
openpi_so101/dataset_v3.py   只读 LeRobot v3 parquet/video 适配器
openpi_so101/policy.py       SO101 输入/输出 transform
openpi_so101/config.py       PI0.5 LoRA TrainConfig 注册
openpi_so101/patches.py      运行时 patch OpenPI create_torch_dataset
openpi_so101/train.py        包装官方 scripts/train.py 的训练入口
```

训练样本契约：

```text
observation.images.top   -> OpenPI base_0_rgb
observation.images.wrist -> OpenPI left_wrist_0_rgb
observation.state[6]     -> OpenPI state, padding 到 32 维
action[50, 6]            -> OpenPI actions, padding 到 32 维
prompt                   -> OpenPI tokenizer prompt
```

由于本数据只有 top 和 wrist 两路相机，OpenPI 需要的 right wrist 图像使用零图像补齐，并且在 PI0.5 下 mask 掉。

推理时使用单独的 repack transform，只需要图像、state 和 prompt，不需要 `action` 字段。

## 环境

本轮 OpenPI 验证实际使用 `openpi_train` conda 环境，因为当前 `lerobot_flex` 环境缺少 OpenPI/JAX 训练必需的 `jax` 和 `openpi` 包。仓库通用规范仍要求使用 `lerobot_flex`；但 OpenPI 训练子项目已明确允许使用 `openpi_train` 作为专用训练环境。

```bash
cd /data/cqy_workspace/flexible_lerobot/my_devs/openpi_train
bash scripts/setup_openpi_train_env.sh
source scripts/env.sh
```

`scripts/env.sh` 默认使用：

```text
OPENPI_CONDA_ENV=openpi_train
```

如需切换到其他环境，可以在 source 前显式设置 `OPENPI_CONDA_ENV`。但当前 `lerobot_flex` 是 Python 3.10，而 `openpi-main` 声明 `requires-python >= 3.11`，所以不能直接把 `OPENPI_CONDA_ENV=lerobot_flex` 作为等价替代。

本轮已验证环境：

```text
conda env = /home/cqy/miniconda3/envs/openpi_train
python    = 3.11.15
jax       = 0.5.3
jaxlib    = 0.5.3
torch     = 2.7.1
GPU       = NVIDIA GeForce RTX 4090
JAX sees  = CudaDevice(id=0)
```

`scripts/env.sh` 会把 OpenPI cache、checkpoint 和数据路径固定到 `my_devs/openpi_train` 下面：

```text
OPENPI_DATA_HOME
OPENPI_PI05_BASE_PARAMS
OPENPI_SO101_ASSETS_BASE_DIR
OPENPI_SO101_CHECKPOINT_BASE_DIR
OPENPI_SO101_DATASET_ROOT
OPENPI_SO101_REPO_ID
```

## 数据验证

运行：

```bash
source scripts/env.sh
bash scripts/check_data.sh --max-frames 8 --batch-size 2
```

验证得到的 shape：

```text
raw action                         = [50, 6] float32
raw observation.state              = [6] float32
raw observation.images.top         = [480, 640, 3] uint8
raw observation.images.wrist       = [480, 640, 3] uint8
OpenPI observation.state           = [2, 32] float32
OpenPI images base/left/right      = [2, 224, 224, 3] float32
OpenPI tokenized_prompt            = [2, 200] int32
OpenPI actions                     = [2, 50, 32] float32
```

注意：原始数据视频使用 AV1 编码。OpenCV 直接读取失败，因此本地适配器使用 PyAV/libdav1d 解码。

## 归一化统计

smoke 验证时用 256 帧计算了归一化统计：

```bash
source scripts/env.sh
bash scripts/compute_norm_stats.sh --max-frames 256 --batch-size 32 --no-decode-images
```

输出文件：

```text
assets/openpi_assets/pi05_so101_eraser_cup_lora/desk_cleanup_v1/eraser_cup_multi_task/norm_stats.json
```

该文件顶层保存 `norm_stats.state` 和 `norm_stats.actions`，两者都包含 6 维的 `mean`、`std`、`q01`、`q99` 数组。

正式训练前建议用完整数据集重新计算：

```bash
source scripts/env.sh
bash scripts/compute_norm_stats.sh --max-frames 53235 --batch-size 64 --no-decode-images
```

## PI0.5 基础权重

官方 OpenPI JAX `pi05_base/params` 已从公开 Google Storage checkpoint 下载到本地 OpenPI cache：

```bash
source scripts/env.sh
bash scripts/download_pi05_base_params.sh
```

已验证的本地 checkpoint：

```text
path  = assets/openpi_cache/openpi-assets/checkpoints/pi05_base/params
files = 20
size  = 12G
```

这是 OpenPI LoRA 训练使用的 JAX/Orbax checkpoint，和之前 LeRobot 使用的 `model.safetensors` 权重格式不同。

## LeRobot v2.1 Pilot 转换验证

为了验证“转换数据方式”是否更贴近官方 OpenPI/LeRobot 数据加载路径，新增了一个 pilot 转换脚本：

```bash
source scripts/env.sh
bash scripts/convert_v3_to_v21_pilot.sh --episodes 3 --overwrite
```

转换输出目录：

```text
data/lerobot_v21_pilot/desk_cleanup_v1/eraser_cup_multi_task
```

该目录是 `my_devs/openpi_train` 下面的数据副本，不会修改原始数据集：

```text
/data/cqy_workspace/flexible_lerobot/datasets/desk_cleanup_v1/eraser_cup_multi_task
```

pilot 转换采用官方 `lerobot.common.datasets.lerobot_dataset.LeRobotDataset.create/add_frame/save_episode` 生成 v2.1 数据目录。训练读取 converted v21 数据时，因为当前环境里的 `datasets==5.0.0` 与 LeRobot v2.1 的 `Column` 处理存在兼容差异，代码使用了一个很薄的本地兼容包装；metadata、episode index、video 解码、delta timestamp 检查仍按 LeRobot v2.1 官方数据集逻辑执行。

```text
data/chunk-000/episode_000000.parquet
data/chunk-000/episode_000001.parquet
data/chunk-000/episode_000002.parquet
meta/info.json
meta/tasks.jsonl
meta/episodes.jsonl
meta/episodes_stats.jsonl
videos/chunk-000/observation.images.top/episode_000000.mp4
videos/chunk-000/observation.images.wrist/episode_000000.mp4
...
```

注意：pilot 转换会重新编码 episode 视频，因此图像不能要求逐像素完全相同；对齐检查中对 state/action/prompt 要求严格一致，对图像使用重编码后的像素误差阈值。

脚本默认按 pilot 用法限制为最多 3 个 episode。如果后续要做全量转换，需要显式传入 `--allow-more-episodes`，避免误把试验脚本跑成大规模覆盖式转换。

对齐检查命令：

```bash
source scripts/env.sh
bash scripts/check_converted_v21.sh --num-samples 8
```

对齐结果：

```text
samples = 8
max_state_abs_diff = 0.0
max_action_abs_diff = 0.0
observation.images.top:
  max_mean_abs_diff = 1.3620594618055555
  max_pixel_abs_diff = 40.0
observation.images.wrist:
  max_mean_abs_diff = 1.2811924913194443
  max_pixel_abs_diff = 42.0
```

结论：

```text
state/action chunk/prompt 对齐通过；
图像存在 AV1 重编码误差，但平均像素差约 1.3，最大像素差 42，在 pilot 验证阈值内可接受。
```

v21 pilot 使用独立的 norm stats asset，避免覆盖 v3 方案：

```text
assets/openpi_assets/pi05_so101_eraser_cup_lora/desk_cleanup_v1/eraser_cup_multi_task_v21_pilot/norm_stats.json
```

计算命令：

```bash
source scripts/env.sh
bash scripts/compute_norm_stats.sh \
  --dataset-format v21 \
  --max-frames 128 \
  --batch-size 32 \
  --no-decode-images
```

## LoRA Smoke 训练

已运行命令：

```bash
source scripts/env.sh
OPENPI_SO101_STEPS=3 \
OPENPI_SO101_MAX_FRAMES=128 \
OPENPI_SO101_BATCH_SIZE=1 \
OPENPI_SO101_SAVE_INTERVAL=3 \
bash scripts/train_lora_smoke.sh
```

运行 ID：

```text
EXP_NAME = so101_lora_smoke_20260701_215922
```

训练结果：

```text
Step 0: grad_norm=2.5287, loss=0.0618, param_norm=1803.7708
Step 1: grad_norm=1.7338, loss=0.0669, param_norm=1803.7711
Step 2: grad_norm=0.7211, loss=0.0440, param_norm=1803.7716
```

生成的 checkpoint：

```text
outputs/checkpoints/pi05_so101_eraser_cup_lora/so101_lora_smoke_20260701_215922/2
```

checkpoint 大小：

```text
8.9G
```

日志文件：

```text
logs/so101_lora_smoke_20260701_215922/train.log
```

观察到的显存：

```text
JAX PI0.5 LoRA smoke training used about 33GB GPU memory on RTX 4090.
```

第一次保存 checkpoint 大约用了 3 分钟，因为 OpenPI 会写完整的 Orbax checkpoint。

## 本地推理 Smoke

已运行命令：

```bash
source scripts/env.sh
bash scripts/smoke_infer_client.sh \
  --checkpoint-dir outputs/checkpoints/pi05_so101_eraser_cup_lora/so101_lora_smoke_20260701_215922/2
```

结果：

```text
actions shape=(50, 6) dtype=float32
first action rows:
[[-11.652592  -100.505135    79.46312     61.551342     7.095498    10.0498905]
 [-10.32804   -100.08839     79.6817      61.928917     7.6438675   10.34575  ]]
```

这验证了训练出的 checkpoint 可以重新加载，并能输出 SO101 需要的 6 维动作序列。

在完成度审计时又重新跑了一次相同推理 smoke，结果同样输出 `actions shape=(50, 6) dtype=float32`。

## v21 Pilot LoRA Smoke 训练

converted v21 pilot 路线也完成了 3 step LoRA smoke：

```bash
source scripts/env.sh
RUN_ID=v21_pilot_fixed_$(date +%Y%m%d_%H%M%S) \
OPENPI_SO101_DATASET_FORMAT=v21 \
OPENPI_SO101_STEPS=3 \
OPENPI_SO101_MAX_FRAMES=128 \
OPENPI_SO101_BATCH_SIZE=1 \
OPENPI_SO101_SAVE_INTERVAL=3 \
bash scripts/train_lora_smoke.sh
```

运行 ID：

```text
EXP_NAME = so101_lora_smoke_v21_pilot_fixed_20260702_102405
```

训练结果：

```text
Step 0: grad_norm=0.3984, loss=0.0235, param_norm=1803.7708
Step 1: grad_norm=1.6809, loss=0.0694, param_norm=1803.7709
Step 2: grad_norm=1.0563, loss=0.0442, param_norm=1803.7711
```

生成的 checkpoint：

```text
outputs/checkpoints/pi05_so101_eraser_cup_lora/so101_lora_smoke_v21_pilot_fixed_20260702_102405/2
```

日志确认该训练使用的是 converted v21 pilot 数据和独立 asset：

```text
Using SO101 converted LeRobot v2.1 pilot
Loaded norm stats from .../desk_cleanup_v1/eraser_cup_multi_task_v21_pilot
```

v21 checkpoint 本地推理 smoke：

```bash
source scripts/env.sh
bash scripts/smoke_infer_client.sh \
  --dataset-format v21 \
  --checkpoint-dir outputs/checkpoints/pi05_so101_eraser_cup_lora/so101_lora_smoke_v21_pilot_fixed_20260702_102405/2
```

结果：

```text
actions shape=(50, 6) dtype=float32
first action rows:
[[  -5.3230395 -106.1018      92.883026    62.979385    10.650322    12.824932 ]
 [  -5.247539  -106.77709     92.968025    63.12142     10.822487    12.871187 ]]
```

## 正式训练命令

正式训练前，先用完整数据集重新计算归一化统计：

```bash
source scripts/env.sh
bash scripts/compute_norm_stats.sh --max-frames 53235 --batch-size 64 --no-decode-images
```

如果要对齐之前 LeRobot 的训练步数，可以使用：

```bash
source scripts/env.sh
RUN_ID=full_33300_$(date +%Y%m%d_%H%M%S) \
OPENPI_SO101_STEPS=33300 \
OPENPI_SO101_MAX_FRAMES=53235 \
OPENPI_SO101_BATCH_SIZE=1 \
OPENPI_SO101_SAVE_INTERVAL=1000 \
bash scripts/train_lora_smoke.sh
```

`OPENPI_SO101_BATCH_SIZE=1` 是单张 4090 上比较保守的设置。只有确认显存足够后再增大 batch size。旧 LeRobot 脚本里的 batch size 16 不适合作为单 GPU JAX PI0.5 LoRA 的起始配置。

如果要使用 converted v21 数据方案，将 `OPENPI_SO101_DATASET_FORMAT` 改为 `v21`，并先确保已完成全量转换和对应的 norm stats 计算：

```bash
source scripts/env.sh
OPENPI_SO101_DATASET_FORMAT=v21 \
OPENPI_SO101_STEPS=33300 \
OPENPI_SO101_MAX_FRAMES=53235 \
OPENPI_SO101_BATCH_SIZE=1 \
OPENPI_SO101_SAVE_INTERVAL=1000 \
bash scripts/train_lora_smoke.sh
```

当前只完成了 3 个 episode 的 pilot 转换验证；如果要正式使用 v21 方案，需要把转换脚本从 pilot 扩展到全量 episode，并重新跑全量对齐抽查和全量 norm stats。

如果用现有 pilot 脚本做更大规模转换，需要显式加上 `--allow-more-episodes`。这是一道安全开关，用来区分“2-3 episode pilot”与“有意全量转换”。

## 环境一致性限制

本仓库通用规范要求使用 `lerobot_flex` conda 环境。但当前 `lerobot_flex` 是 Python 3.10.19，而 `openpi-main` 声明 `requires-python >= 3.11`，并且 `lerobot_flex` 里没有安装 OpenPI 训练必需的 `jax`、`openpi`、`flax`、`orbax` 等包。因此 OpenPI 训练子项目允许使用 `scripts/env.sh` 中默认的 `openpi_train` 环境。

这不是数据转换逻辑本身的安全问题。当前口径是：

```text
OpenPI 训练、norm stats、推理 smoke、后续正式 LoRA 训练：使用 openpi_train。
主仓库其他 lerobot_flex 开发任务：继续遵循主仓库规范。
```

## Policy Server 推理

启动一个训练后 checkpoint 的 policy server：

```bash
source scripts/env.sh
bash scripts/serve_policy.sh \
  --checkpoint-dir outputs/checkpoints/pi05_so101_eraser_cup_lora/so101_lora_smoke_20260701_215922/2 \
  --port 8000 \
  --default-prompt "Put the eraser into the small box"
```

如果服务的是 v21 pilot 或后续 v21 converted checkpoint，需要加上：

```bash
--dataset-format v21
```

服务端每次请求需要提供：

```text
observation.images.top    HWC uint8 或 float 图像
observation.images.wrist  HWC uint8 或 float 图像
observation.state         shape [6]
prompt                    字符串；如果设置了 default_prompt，可省略
```

返回：

```text
actions shape [50, 6]
```

机器人实机推理可参考之前的 LeRobot SO101 推理脚本：

```text
/data/cqy_workspace/flexible_lerobot/my_devs/train/pi/so101/run_pi05_infer.py
```

推荐集成方式：

1. 保留原脚本里的相机采集和 SO101 follower 机器人控制逻辑。
2. 将 LeRobot policy 加载替换为 OpenPI websocket policy client，或者在同进程中直接调用 `policy_config.create_trained_policy`。
3. 构造 observation dict，包含 top 图像、wrist 图像、6 维 state 和任务文本 prompt。
4. 按现有控制循环节奏执行第一步动作或一个 action chunk。

## 新增文件

```text
openpi_so101/
  __init__.py
  paths.py
  runtime.py
  dataset_v3.py
  policy.py
  config.py
  patches.py
  check_data.py
  compute_norm_stats.py
  train.py
  serve_policy.py
  smoke_infer_client.py
  convert_v3_to_v21_pilot.py
  check_converted_v21.py

scripts/
  env.sh
  setup_openpi_train_env.sh
  download_pi05_base_params.py
  download_pi05_base_params.sh
  convert_v3_to_v21_pilot.sh
  check_converted_v21.sh
  check_data.sh
  compute_norm_stats.sh
  train_lora_smoke.sh
  serve_policy.sh
  smoke_infer_client.sh

docs/
  openpi_so101_train_report.md
```

生成的本地资产：

```text
assets/openpi_cache/openpi-assets/checkpoints/pi05_base/params
assets/openpi_assets/pi05_so101_eraser_cup_lora/desk_cleanup_v1/eraser_cup_multi_task/norm_stats.json
assets/openpi_assets/pi05_so101_eraser_cup_lora/desk_cleanup_v1/eraser_cup_multi_task_v21_pilot/norm_stats.json
data/lerobot_v21_pilot/desk_cleanup_v1/eraser_cup_multi_task
outputs/checkpoints/pi05_so101_eraser_cup_lora/so101_lora_smoke_20260701_215922/2
outputs/checkpoints/pi05_so101_eraser_cup_lora/so101_lora_smoke_v21_pilot_fixed_20260702_102405/2
logs/so101_lora_smoke_20260701_215922/train.log
logs/so101_lora_smoke_v21_pilot_fixed_20260702_102405/train.log
```

原始数据集目录没有被修改：

```text
/data/cqy_workspace/flexible_lerobot/datasets/desk_cleanup_v1/eraser_cup_multi_task
```
