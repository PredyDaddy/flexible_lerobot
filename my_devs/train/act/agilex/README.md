# Agilex ACT 训练脚本说明

目录：

- `my_devs/train/act/agilex`

本目录当前提供：

1. `train_first_test.sh`
   - 默认训练双臂数据集 `datasets/lerobot_datasets/first_test`
2. `train_first_test_right.sh`
   - 默认训练右臂单臂数据集 `datasets/lerobot_datasets/first_test_right`
3. `run_act_infer.py`
   - Agilex ACT 实机推理主脚本，支持只读观测检查、影子推理和真实闭环
4. `run_act_infer.sh`
   - Agilex ACT 实机推理包装脚本，负责默认参数、环境和日志落盘

训练脚本的风格、参数布局和环境变量覆盖方式参考 `my_devs/train/act/so101/train_full.sh`，但当前实现并不是直接复用该脚本。

默认训练目标就是仓库里的本地数据集目录：

1. `/home/agilex/cqy/flexible_lerobot/datasets/lerobot_datasets/first_test`
2. `/home/agilex/cqy/flexible_lerobot/datasets/lerobot_datasets/first_test_right`

不是 `my_devs/split_datasets` 目录下的其他产物。

共同点如下：

1. 使用 `lerobot_flex` conda 环境
2. 通过环境变量覆盖参数
3. 最后调用 `lerobot-train`

---

## 1. 怎么运行

先进入仓库根目录：

```bash
cd /home/agilex/cqy/flexible_lerobot
```

### 1.1 训练双臂 first_test

```bash
bash my_devs/train/act/agilex/train_first_test.sh
```

### 1.2 训练右臂 first_test_right

```bash
bash my_devs/train/act/agilex/train_first_test_right.sh
```

补充说明：

1. `DATASET_ROOT` 必须指向具体数据集目录本身，例如 `/home/agilex/cqy/flexible_lerobot/datasets/lerobot_datasets/first_test`
2. 不能只指向父目录 `/home/agilex/cqy/flexible_lerobot/datasets/lerobot_datasets`

---

## 2. 推荐先做 smoke test

这两个脚本默认是偏正式训练配置，不建议第一次就直接长跑。

当前脚本已经通过 2 步轻量检查：

1. `bash -n` 语法检查
2. `DRY_RUN=1` 命令拼装检查

建议先用很小的步数做一次 smoke test：

### 2.1 双臂 smoke

```bash
STEPS=20 \
BATCH_SIZE=2 \
NUM_WORKERS=0 \
SAVE_FREQ=20 \
LOG_FREQ=1 \
WANDB_ENABLE=false \
bash my_devs/train/act/agilex/train_first_test.sh
```

### 2.2 右臂 smoke

```bash
BATCH_SIZE=2 \
NUM_WORKERS=0 \
SAVE_FREQ=20 \
LOG_FREQ=1 \
STEPS=20 \
WANDB_ENABLE=false \
bash my_devs/train/act/agilex/train_first_test_right.sh
```

如果你只想先看命令有没有拼对，不真正启动训练，可以用：

```bash
DRY_RUN=1 bash my_devs/train/act/agilex/train_first_test.sh
DRY_RUN=1 bash my_devs/train/act/agilex/train_first_test_right.sh
```

---

## 3. 默认值是什么

### 3.1 双臂脚本默认值

- `DATASET_REPO_ID=first_test`
- `DATASET_ROOT=/home/agilex/cqy/flexible_lerobot/datasets/lerobot_datasets/first_test`
- `JOB_NAME=act_agilex_first_test_full`
- `POLICY_TYPE=act`
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

说明：双臂默认 `BATCH_SIZE=16`，因为双臂输入、状态和动作都更重，默认设置更保守。

### 3.2 右臂脚本默认值

- `DATASET_REPO_ID=first_test_right`
- `DATASET_ROOT=/home/agilex/cqy/flexible_lerobot/datasets/lerobot_datasets/first_test_right`
- `DATASET_VIDEO_BACKEND=pyav`
- `EPOCHS=15`
- `JOB_NAME=act_agilex_first_test_right_e15`
- `POLICY_TYPE=act`
- `POLICY_DEVICE=auto`
- `BATCH_SIZE=16`
- `STEPS=` 空表示自动按 epoch 换算
- `SAVE_EVERY_EPOCHS=5`
- `SAVE_FREQ=` 空表示自动按 `SAVE_EVERY_EPOCHS` 换算
- `EVAL_FREQ=-1`
- `LOG_FREQ=100`
- `NUM_WORKERS=4`
- `SEED=1000`
- `PUSH_TO_HUB=false`
- `WANDB_ENABLE=false`

说明：右臂脚本不是直接写死 `STEPS`，而是会读取 `meta/info.json` 里的 `total_frames`，按 `ceil(total_frames / batch_size) * EPOCHS` 自动换算总步数。对当前 `first_test_right` 数据集，默认 `BATCH_SIZE=16`、`EPOCHS=15` 时，总步数约为 `59910`，每 5 个 epoch 保存一次时 `save_freq=19970`。

---

## 4. 最常改的参数

这些参数都可以在命令前面直接覆盖。

### 4.1 改训练步数

```bash
STEPS=20000 bash my_devs/train/act/agilex/train_first_test.sh
```

右臂脚本更推荐直接改 epoch 数：

```bash
EPOCHS=15 bash my_devs/train/act/agilex/train_first_test_right.sh
```

如果只是想做一个很小的 smoke test，也可以直接手工覆盖步数：

```bash
STEPS=20 SAVE_FREQ=20 bash my_devs/train/act/agilex/train_first_test_right.sh
```

### 4.2 改 batch size

```bash
BATCH_SIZE=8 bash my_devs/train/act/agilex/train_first_test.sh
```

### 4.3 指定 GPU

```bash
CUDA_VISIBLE_DEVICES=0 POLICY_DEVICE=cuda \
bash my_devs/train/act/agilex/train_first_test_right.sh
```

### 4.4 只训练部分 episode

```bash
DATASET_EPISODES='[0,1,2,3]' \
bash my_devs/train/act/agilex/train_first_test.sh
```

### 4.5 改输出目录

```bash
OUTPUT_ROOT=/home/agilex/cqy/flexible_lerobot/outputs/train_agilex \
bash my_devs/train/act/agilex/train_first_test.sh
```

### 4.6 改任务名

```bash
JOB_NAME=act_first_test_debug \
bash my_devs/train/act/agilex/train_first_test.sh
```

---

## 5. 输出到哪里

默认输出位置有两类：

1. 日志
   - `/home/agilex/cqy/flexible_lerobot/logs/`
2. 训练产物
   - `/home/agilex/cqy/flexible_lerobot/outputs/train/`

默认情况下：

1. 日志文件名类似：
   - `logs/train_act_agilex_first_test_full_时间戳.log`
2. 输出目录类似：
   - `outputs/train/时间戳_act_agilex_first_test_full`

右臂脚本同理，只是 `JOB_NAME` 不同。

---

## 6. 常见改法

### 6.1 双臂正式训练

```bash
CUDA_VISIBLE_DEVICES=0 POLICY_DEVICE=cuda \
BATCH_SIZE=16 \
STEPS=50000 \
NUM_WORKERS=4 \
bash my_devs/train/act/agilex/train_first_test.sh
```

---

## 7. Agilex ACT 上机推理

当前上机推理针对这份 checkpoint：

- `/home/agilex/cqy/flexible_lerobot/outputs/train/20260313_194500_act_agilex_first_test_full/checkpoints/100000/pretrained_model`

脚本分两层：

1. Python 主脚本：
   - `my_devs/train/act/agilex/run_act_infer.py`
2. Shell 包装脚本：
   - `my_devs/train/act/agilex/run_act_infer.sh`

### 7.1 支持的运行模式

第一版支持 3 种使用方式：

1. `dry-run`
   - 只解析 checkpoint、设备、topic 和模式，不连机器人
2. `observation_only + passive_follow`
   - 只读观测检查，不跑 policy，不发动作
3. `policy_inference + passive_follow`
   - 影子推理，跑 policy，但不发动作
4. `policy_inference + command_master`
   - 真实闭环，会向命令 topic 发布动作

补充说明：

1. 默认 `CONTROL_MODE=passive_follow`
2. 第一次上机不要直接进入 `command_master`
3. 在确认 14 维 `action` 与 `/master/joint_*` 控制语义完全一致前，不要做正式闭环

### 7.2 先做 dry-run

直接运行 Python：

```bash
conda run -n lerobot_flex python my_devs/train/act/agilex/run_act_infer.py --dry-run true
```

或者运行 shell 包装：

```bash
DRY_RUN=1 bash my_devs/train/act/agilex/run_act_infer.sh
```

### 7.3 只读观测检查

推荐先检查双臂 state 和三路图像是否都正常刷新：

```bash
EXECUTION_MODE=observation_only \
CONTROL_MODE=passive_follow \
RUN_TIME_S=10 \
bash my_devs/train/act/agilex/run_act_infer.sh
```

### 7.4 影子推理

影子推理会跑完整的 ACT 推理链路，但因为还是 `passive_follow`，不会发布动作：

```bash
EXECUTION_MODE=policy_inference \
CONTROL_MODE=passive_follow \
POLICY_N_ACTION_STEPS=16 \
RUN_TIME_S=20 \
bash my_devs/train/act/agilex/run_act_infer.sh
```

说明：

1. 当前训练 checkpoint 默认 `chunk_size=100`、`n_action_steps=100`
2. 第一次实机验证更建议显式设置 `POLICY_N_ACTION_STEPS=16` 或 `8`

### 7.5 真实闭环

只有在只读观测和影子推理都通过后，再进入闭环：

```bash
EXECUTION_MODE=policy_inference \
CONTROL_MODE=command_master \
POLICY_N_ACTION_STEPS=16 \
RUN_TIME_S=15 \
bash my_devs/train/act/agilex/run_act_infer.sh
```

### 7.6 常用环境变量

`run_act_infer.sh` 常用覆盖项包括：

1. `POLICY_PATH`
2. `EXECUTION_MODE`
3. `CONTROL_MODE`
4. `POLICY_DEVICE_OVERRIDE`
5. `POLICY_N_ACTION_STEPS`
6. `POLICY_TEMPORAL_ENSEMBLE_COEFF`
7. `DATASET_TASK`
8. `RUN_TIME_S`
9. `FPS`
10. `STATE_LEFT_TOPIC`
11. `STATE_RIGHT_TOPIC`
12. `COMMAND_LEFT_TOPIC`
13. `COMMAND_RIGHT_TOPIC`
14. `FRONT_CAMERA_TOPIC`
15. `LEFT_CAMERA_TOPIC`
16. `RIGHT_CAMERA_TOPIC`
17. `OBSERVATION_TIMEOUT_S`

### 7.7 日志输出位置

默认日志目录：

1. `/home/agilex/cqy/flexible_lerobot/logs`

默认日志文件名类似：

1. `infer_act_agilex_first_test_infer_时间戳.log`

### 6.2 右臂单臂训练

```bash
CUDA_VISIBLE_DEVICES=0 POLICY_DEVICE=cuda \
BATCH_SIZE=16 \
EPOCHS=15 \
NUM_WORKERS=4 \
bash my_devs/train/act/agilex/train_first_test_right.sh
```

### 6.3 关闭外网依赖并快速验证

这两个脚本默认已经：

1. `HF_HUB_OFFLINE=1`
2. `HF_DATASETS_OFFLINE=1`
3. `TRANSFORMERS_OFFLINE=1`

如果你只是要先检查命令拼装是否正确，优先用：

```bash
DRY_RUN=1 bash my_devs/train/act/agilex/train_first_test.sh
```

---

## 7. 调参建议

如果是第一次在这两个数据集上训练，建议顺序如下：

1. 先 `DRY_RUN=1`
2. 再做 `STEPS=20~100` 的 smoke test
3. 确认 loss、日志、checkpoint 都正常后，再提高步数

一个比较稳的起点是：

- 双臂：
  - `BATCH_SIZE=8`
  - `STEPS=20000`
  - `NUM_WORKERS=2`
- 右臂：
  - `BATCH_SIZE=16`
  - `EPOCHS=15`
  - 或者 `STEPS=20000`
  - `NUM_WORKERS=2`

如果显存紧张，优先先降：

1. `BATCH_SIZE`
2. `NUM_WORKERS`

---

## 8. 注意事项

1. 这两个脚本都依赖 `conda` 可用。
2. 如果当前 shell 不在 `lerobot_flex` 环境里，脚本会自动用：

```bash
conda run --no-capture-output -n lerobot_flex
```

3. 如果你已经手动 `conda activate lerobot_flex`，脚本不会重复切环境。
4. 如果数据集目录不存在，脚本会直接退出，不会启动训练。
