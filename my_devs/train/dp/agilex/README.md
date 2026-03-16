# Agilex Diffusion Policy 训练脚本说明

目录：

- `my_devs/train/dp/agilex`

本目录当前提供：

1. `train_first_test.sh`
   - 默认训练双臂数据集 `datasets/lerobot_datasets/first_test`
2. `train_first_test_right.sh`
   - 默认训练右臂单臂数据集 `datasets/lerobot_datasets/first_test_right`

训练脚本的调用风格、参数布局和环境变量覆盖方式，对齐 `my_devs/train/act/agilex`。

共同点如下：

1. 使用 `lerobot_flex` conda 环境
2. 通过环境变量覆盖参数
3. 最终调用 `lerobot-train`
4. 固定 `policy.type=diffusion`

## 1. 怎么运行

先进入仓库根目录：

```bash
cd /home/agilex/cqy/flexible_lerobot
```

### 1.1 训练双臂 first_test

```bash
bash my_devs/train/dp/agilex/train_first_test.sh
```

### 1.2 训练右臂 first_test_right

```bash
bash my_devs/train/dp/agilex/train_first_test_right.sh
```

## 2. DATASET_ROOT 约束

`DATASET_ROOT` 必须直接指向具体数据集目录本身，而不是父目录。

正确示例：

```bash
DATASET_ROOT=/home/agilex/cqy/flexible_lerobot/datasets/lerobot_datasets/first_test
```

错误示例：

```bash
DATASET_ROOT=/home/agilex/cqy/flexible_lerobot/datasets/lerobot_datasets
```

简单判断方式：

1. 这个目录下应该能直接看到 `meta/info.json`
2. 不能只是一层数据集父目录

## 3. 默认值

### 3.1 双臂脚本默认值

- `DATASET_REPO_ID=first_test`
- `DATASET_ROOT=/home/agilex/cqy/flexible_lerobot/datasets/lerobot_datasets/first_test`
- `JOB_NAME=dp_agilex_first_test_full`
- `POLICY_DEVICE=auto`
- `BATCH_SIZE=8`
- `STEPS=100000`
- `SAVE_FREQ=10000`
- `EVAL_FREQ=-1`
- `LOG_FREQ=100`
- `NUM_WORKERS=4`
- `SEED=1000`
- `PUSH_TO_HUB=false`
- `WANDB_ENABLE=false`

### 3.2 右臂脚本默认值

- `DATASET_REPO_ID=first_test_right`
- `DATASET_ROOT=/home/agilex/cqy/flexible_lerobot/datasets/lerobot_datasets/first_test_right`
- `JOB_NAME=dp_agilex_first_test_right_full`
- `POLICY_DEVICE=auto`
- `BATCH_SIZE=16`
- `STEPS=100000`
- `SAVE_FREQ=10000`
- `EVAL_FREQ=-1`
- `LOG_FREQ=100`
- `NUM_WORKERS=4`
- `SEED=1000`
- `PUSH_TO_HUB=false`
- `WANDB_ENABLE=false`

说明：

1. 双臂 Diffusion 默认更重，所以 `BATCH_SIZE` 更保守
2. 右臂单臂默认输入更轻，所以 `BATCH_SIZE` 略大

## 4. 最常改的参数

这些参数都可以直接在命令前覆盖。

### 4.1 改训练步数

```bash
STEPS=20000 bash my_devs/train/dp/agilex/train_first_test.sh
```

### 4.2 改 batch size

```bash
BATCH_SIZE=4 bash my_devs/train/dp/agilex/train_first_test.sh
```

### 4.3 指定 GPU

```bash
CUDA_VISIBLE_DEVICES=0 POLICY_DEVICE=cuda \
bash my_devs/train/dp/agilex/train_first_test_right.sh
```

### 4.4 只训练部分 episode

```bash
DATASET_EPISODES='[0,1,2,3]' \
bash my_devs/train/dp/agilex/train_first_test.sh
```

### 4.5 改输出目录

```bash
OUTPUT_ROOT=/home/agilex/cqy/flexible_lerobot/outputs/train_dp \
bash my_devs/train/dp/agilex/train_first_test.sh
```

### 4.6 改任务名

```bash
JOB_NAME=dp_first_test_debug \
bash my_devs/train/dp/agilex/train_first_test.sh
```

## 5. Diffusion 常用参数

脚本里额外预留了以下 Diffusion 环境变量：

- `DIFFUSION_N_OBS_STEPS`
- `DIFFUSION_HORIZON`
- `DIFFUSION_N_ACTION_STEPS`
- `DIFFUSION_NUM_TRAIN_TIMESTEPS`
- `DIFFUSION_NUM_INFERENCE_STEPS`

示例：

```bash
DIFFUSION_N_OBS_STEPS=2 \
DIFFUSION_HORIZON=16 \
DIFFUSION_N_ACTION_STEPS=8 \
DIFFUSION_NUM_TRAIN_TIMESTEPS=50 \
bash my_devs/train/dp/agilex/train_first_test.sh
```

注意：

1. `policy.type` 在脚本里已经固定为 `diffusion`
2. 当前脚本不直接开放 tuple 形态的 `down_dims` / `crop_shape` 参数，避免 CLI 解析兼容性问题
3. 如果你需要改这类结构化参数，建议直接使用完整 `lerobot-train` 命令或单独的配置文件

## 6. 输出到哪里

默认输出位置有两类：

1. 日志
   - `/home/agilex/cqy/flexible_lerobot/logs/`
2. 训练产物
   - `/home/agilex/cqy/flexible_lerobot/outputs/train/`

默认情况下：

1. 日志文件名类似：
   - `logs/train_dp_agilex_first_test_full_时间戳.log`
2. 输出目录类似：
   - `outputs/train/时间戳_dp_agilex_first_test_full`

右臂脚本同理，只是 `JOB_NAME` 不同。

## 7. 推荐先做 smoke test

第一次不要直接长跑，先用很小的配置验证命令、数据集和训练链路。

### 7.1 双臂 smoke test

```bash
POLICY_DEVICE=cpu \
STEPS=5 \
BATCH_SIZE=1 \
NUM_WORKERS=0 \
DATASET_EPISODES='[0]' \
EVAL_FREQ=-1 \
LOG_FREQ=1 \
SAVE_FREQ=5 \
WANDB_ENABLE=false \
bash my_devs/train/dp/agilex/train_first_test.sh
```

### 7.2 右臂 smoke test

```bash
POLICY_DEVICE=cpu \
STEPS=5 \
BATCH_SIZE=1 \
NUM_WORKERS=0 \
DATASET_EPISODES='[0]' \
EVAL_FREQ=-1 \
LOG_FREQ=1 \
SAVE_FREQ=5 \
WANDB_ENABLE=false \
bash my_devs/train/dp/agilex/train_first_test_right.sh
```

如果你只想先看命令有没有拼对，不真正启动训练，可以用：

```bash
DRY_RUN=1 bash my_devs/train/dp/agilex/train_first_test.sh
DRY_RUN=1 bash my_devs/train/dp/agilex/train_first_test_right.sh
```
