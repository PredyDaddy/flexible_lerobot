# 单臂数据集 remove_feature_pipeline 测试报告（2026-03-05）

本报告用于验证以下 3 件事：

1. `my_devs/remove_feature_pipeline` 的实现是否符合 `my_devs/docs/remove_feature_pipeline/单臂数据集构建与训练Pipeline技术方案.md` 中的技术方案。
2. `my_devs/docs/remove_feature_pipeline/工作报告.md` 中描述的“测试通过 / 产物正确”等结论是否属实（可复现）。
3. 你给出的数据集产物是否“结构正确 + 数值健康 + 可被 LeRobot 正常加载”，从而可以继续用于训练。

## 1. 测试对象与输入

### 1.1 方案与报告（被核对材料）

1. 技术方案：
   - `/home/agilex/cqy/flexible_lerobot/my_devs/docs/remove_feature_pipeline/单臂数据集构建与训练Pipeline技术方案.md`
2. 工作报告：
   - `/home/agilex/cqy/flexible_lerobot/my_devs/docs/remove_feature_pipeline/工作报告.md`

### 1.2 代码（被测实现）

被测实现路径：

- `/home/agilex/cqy/flexible_lerobot/my_devs/remove_feature_pipeline/core.py`
- `/home/agilex/cqy/flexible_lerobot/my_devs/remove_feature_pipeline/single_arm_pipeline.py`
- `/home/agilex/cqy/flexible_lerobot/my_devs/remove_feature_pipeline/tests/test_single_arm_pipeline.py`
- `/home/agilex/cqy/flexible_lerobot/my_devs/remove_feature_pipeline/test_core.py`

### 1.3 数据集（你提供的输入与产物）

1. 输入数据集（14D 双臂 + 3 相机）：
   - `/home/agilex/cqy/flexible_lerobot/datasets/lerobot_datasets/test_pipeline_clean_12_19_removed_statsfix`
2. 你提供的右臂产物（7D + 2 相机）：
   - `/home/agilex/cqy/flexible_lerobot/datasets/lerobot_datasets/test_pipeline_clean_12_19_removed_right_arm7`
3. 你提供的左臂产物（7D + 2 相机）：
   - `/home/agilex/cqy/flexible_lerobot/datasets/lerobot_datasets/test_pipeline_clean_12_19_removed_left_arm7`

## 2. 测试环境与运行约束

按照仓库 `AGENTS.md` 约定，本次所有测试与可执行验证均在 conda 环境 `lerobot_flex` 内完成。

1. Conda env：`lerobot_flex`
2. Python：`3.10.17`（通过 `conda run -n lerobot_flex python -V` 读取）

重要运行注意事项（训练与加载同样适用）：

- 直接用 `LeRobotDataset` / HuggingFace `datasets` 读取本地 parquet 时，会尝试在 `~/.cache/huggingface/datasets` 写 lock 文件。
- 当前机器对该目录没有写权限，因此如果不设置环境变量，会出现类似错误：
  - `PermissionError: ... ~/.cache/huggingface/datasets/... .lock`
- 解决方式：在执行 pipeline / 训练 / 加载数据集时增加以下环境变量，指向可写目录（例如 `/tmp`）：

```bash
export HF_HOME=/tmp/hf_home
export HF_DATASETS_CACHE=/tmp/hf_datasets
```

## 3. 技术方案符合性核对（Doc vs Code）

技术方案中定义的关键规则（摘取对验收最重要的部分）：

1. 右臂数据集：
   - 保留相机：`cam_high + cam_right_wrist`
   - 删除相机：`cam_left_wrist`
   - `action = action[:, 7:14]`
   - `observation.state = observation.state[:, 7:14]`
2. 左臂数据集：
   - 保留相机：`cam_high + cam_left_wrist`
   - 删除相机：`cam_right_wrist`
   - `action = action[:, 0:7]`
   - `observation.state = observation.state[:, 0:7]`
3. `meta/info.json`：
   - `action.shape=[7]`
   - `observation.state.shape=[7]`
4. `meta/stats.json`：
   - 至少重算 `action` / `observation.state` 的 stats，并应用 `std floor`（避免 `std≈0` 导致归一化爆炸）。
5. 数据健康检查：
   - 相机 key 组合正确
   - action/state 维度正确
   - NaN/Inf 为 0

代码实现核对结论：

1. 规则匹配：
   - `core.py` 的 `ARM_PLANS` 明确写死了 left/right 的 slice 区间、需要删除的相机、以及期望相机列表，和技术方案完全一致。
2. 工作流匹配：
   - `build_single_arm_dataset()` 执行顺序是：
     - 调用 `lerobot.datasets.dataset_tools.remove_feature()` 删除对侧腕相机并落盘到 target；
     - 对 target 里的 parquet 重写 `action`/`observation.state` 列，做 14D->7D 切片；
     - 更新 `meta/info.json` 的 shape；
     - 重算 `meta/stats.json` 的 `action`/`observation.state` 并应用 `std_floor`；
     - 执行 `validate_dataset()` 做相机键、shape、NaN/Inf 校验。
3. 与方案的“形式差异”（不影响结果）：
   - 技术方案建议分两段（先相机裁剪得到中间集，再做 action/state 重写得到最终集），并给出了中间集命名建议；
   - 当前实现把两段合并在一次 pipeline 里，直接写最终 target 目录，不会额外保留中间集目录。
   - 结论：这是可接受的实现选择，最终产物符合方案定义。

## 4. 工作报告可复现性验证（单元测试 + 编译）

### 4.1 单元测试

执行命令：

```bash
conda run --no-capture-output -n lerobot_flex \
pytest -q my_devs/remove_feature_pipeline
```

实际结果：

- `9 passed in 1.82s`

这与工作报告中宣称的 “9 passed” 一致。

### 4.2 语法编译检查

执行命令：

```bash
conda run --no-capture-output -n lerobot_flex \
python -m py_compile \
  my_devs/remove_feature_pipeline/core.py \
  my_devs/remove_feature_pipeline/single_arm_pipeline.py \
  my_devs/remove_feature_pipeline/test_core.py \
  my_devs/remove_feature_pipeline/tests/test_single_arm_pipeline.py
```

实际结果：

- 通过（无输出）

## 5. 你提供的数据集产物验收（结构 + 数值 + 一致性）

### 5.1 元数据结构检查（info.json / 相机键 / shape）

读取 `meta/info.json` 后检查到：

1. `test_pipeline_clean_12_19_removed_statsfix`
   - 相机 keys：`cam_high + cam_left_wrist + cam_right_wrist`（共 3 个）
   - `action.shape=[14]`，`observation.state.shape=[14]`
2. `test_pipeline_clean_12_19_removed_right_arm7`
   - 相机 keys：`cam_high + cam_right_wrist`（共 2 个）
   - `action.shape=[7]`，`observation.state.shape=[7]`
3. `test_pipeline_clean_12_19_removed_left_arm7`
   - 相机 keys：`cam_high + cam_left_wrist`（共 2 个）
   - `action.shape=[7]`，`observation.state.shape=[7]`

结论：相机组合与 shape 均符合技术方案。

### 5.2 数据规模一致性（总帧数）

统计 `data/chunk-*/file-*.parquet` 的 `num_rows`：

- 三个数据集的总行数均为 `33600`

结论：左右臂产物没有丢帧，帧数与输入一致。

### 5.3 14D->7D 切片正确性（强一致性验证）

对输入 `test_pipeline_clean_12_19_removed_statsfix` 与两份产物进行逐元素对齐验证：

1. 右臂：
   - 验证 `right_arm7.action == source.action[:, 7:14]`
   - 验证 `right_arm7.observation.state == source.observation.state[:, 7:14]`
2. 左臂：
   - 验证 `left_arm7.action == source.action[:, 0:7]`
   - 验证 `left_arm7.observation.state == source.observation.state[:, 0:7]`

结果：

- `max_abs_diff` 全部为 `0.0`（完全一致）

结论：左右臂产物确实是从输入数据集中按方案切片得到的，不存在维度乱序或索引错误。

### 5.4 视觉资源检查（videos 目录）

检查 `videos/` 目录结构：

1. 输入 statsfix：包含三个相机视频子目录。
2. right_arm7：仅保留 `cam_high` 与 `cam_right_wrist` 子目录。
3. left_arm7：仅保留 `cam_high` 与 `cam_left_wrist` 子目录。

结论：相机裁剪不仅体现在 `info.json`，对应的视频资源也同步裁剪，符合预期。

### 5.5 数值健康检查（NaN/Inf + std 范围 + 归一化风险）

1. `validate_dataset()` 级别的 NaN/Inf 检查：
   - 右臂与左臂产物 `NaN/Inf = 0/0`
2. `meta/stats.json` 的 `std` 范围（关键结论）：
   - `right_arm7`：
     - `action std min/max = 0.0280 / 0.3413`
     - `state  std min/max = 0.0202 / 0.3360`
   - `left_arm7`：
     - `action std min/max = 0.001 / 0.001`（全部维度触发 floor）
     - `state  std min/max = 0.001 / 0.001`（全部维度触发 floor）
3. 归一化值范围（使用 stats 计算 z-score）：
   - `right_arm7`：`max|z|≈7.21`，`p99|z|≈2.71`
   - `left_arm7`：
     - `action` 全为常量时 `|z|` 全 0；
     - `state` 大部分接近 0，极少数点 `max|z|≈1.96`

结论：

- right_arm7 的数值分布正常，不存在 std 极小导致的归一化爆炸。
- left_arm7 的“全部维度 std 被 floor 到 0.001”说明这份数据里左臂几乎静止；数值上是安全的，但从“训练有效性”角度看，模型可能学不到明显左臂动作策略（这不是 pipeline 的错误，而是数据分布决定的）。

## 6. 端到端可执行 Smoke Test（用当前代码从 statsfix 现场重建）

为了验证“当前代码在本机可端到端运行”，我额外做了两次实际构建（输出到 `/tmp`，避免覆盖你已有产物）。

右臂构建命令：

```bash
mkdir -p /tmp/hf_home /tmp/hf_datasets /tmp/single_arm_pipeline_smoke_20260305
HF_HOME=/tmp/hf_home HF_DATASETS_CACHE=/tmp/hf_datasets \
conda run --no-capture-output -n lerobot_flex \
python my_devs/remove_feature_pipeline/single_arm_pipeline.py \
  --source-root /home/agilex/cqy/flexible_lerobot/datasets/lerobot_datasets \
  --source-repo-id test_pipeline_clean_12_19_removed_statsfix \
  --target-repo-id statsfix_right_arm7_smoketest_20260305 \
  --arm right \
  --work-root /tmp/single_arm_pipeline_smoke_20260305 \
  --std-floor 1e-3
```

左臂构建命令：

```bash
HF_HOME=/tmp/hf_home HF_DATASETS_CACHE=/tmp/hf_datasets \
conda run --no-capture-output -n lerobot_flex \
python my_devs/remove_feature_pipeline/single_arm_pipeline.py \
  --source-root /home/agilex/cqy/flexible_lerobot/datasets/lerobot_datasets \
  --source-repo-id test_pipeline_clean_12_19_removed_statsfix \
  --target-repo-id statsfix_left_arm7_smoketest_20260305 \
  --arm left \
  --work-root /tmp/single_arm_pipeline_smoke_20260305 \
  --std-floor 1e-3
```

实际结果（两次都成功）：

- pipeline 日志显示：
  - remove_feature 正常完成（相机裁剪、视频复制、stats 重算）
  - action/state 切片完成
  - info.json shape 更新完成
  - action/state stats 重算完成（并显示 tiny_std_dims 统计）
  - validate_dataset 通过（NaN/Inf=0，camera keys=2）

进一步一致性对照：

- 将 `/tmp/...smoketest...` 的输出与 `datasets/lerobot_datasets/test_pipeline_clean_12_19_removed_{left,right}_arm7` 对比：
  - action/state parquet 逐元素一致（max_abs_diff=0）
  - action/state 的 stats(mean/std/min/max) 逐元素一致（max_abs_diff=0）

结论：你给出的左右臂产物与“当前代码在本机从 statsfix 输入重建的结果”完全一致。

## 7. 训练可用性验证（能否被 LeRobotDataset 正常加载）

验证方式：

- 使用 `LeRobotDataset(repo_id=..., root=<具体数据集目录>)` 读取；
- 读取 `ds[0]`，检查 `action/state` 维度与相机 key。

注意：需要设置 `HF_HOME/HF_DATASETS_CACHE`（见第 2 节），否则会因为 lock 文件写入失败而报错。

验证结果：

1. `test_pipeline_clean_12_19_removed_right_arm7`
   - `len(ds)=33600`
   - `action.shape=torch.Size([7])`
   - `observation.state.shape=torch.Size([7])`
   - image keys：`cam_high + cam_right_wrist`
2. `test_pipeline_clean_12_19_removed_left_arm7`
   - `len(ds)=33600`
   - `action.shape=torch.Size([7])`
   - `observation.state.shape=torch.Size([7])`
   - image keys：`cam_high + cam_left_wrist`

结论：两份产物均可被 LeRobot 正常加载，具备进入训练管线的基本条件。

## 8. 风险点与建议（面向继续训练）

### 8.1 数据层面的风险

1. `left_arm7` 在本数据里“7 维几乎静止”，std 全部被 floor 到 `0.001`：
   - 数值上是安全的（避免 std=0 的归一化爆炸）。
   - 但从训练有效性看，如果你的目标是训练左臂动作策略，这份数据可能信号不足。
   - 如果你的目标是“右臂动、左臂固定姿态”，这份 left 数据是合理的。

### 8.2 运行/权限风险

1. 默认 `~/.cache/huggingface` 不可写会导致加载/训练失败：
   - 建议在训练命令前统一设置 `HF_HOME` 与 `HF_DATASETS_CACHE` 到可写目录（如 `/tmp`）。

### 8.3 实现层面的安全性提示

1. `--overwrite` 会 `shutil.rmtree(target_dir)` 删除整个目标目录：
   - 这符合预期行为，但要求你在训练前非常确认 `--target-repo-id` 写对了（避免误删）。
   - 若担心误操作，可以先 `--dry-run` 看计划，再执行真实落盘。

## 9. 训练 Smoke 验证（已实际跑通）

目的：验证 `right_arm7` 产物不仅“可加载”，且能被 `lerobot-train` 端到端训练跑通并落盘 checkpoint。

训练命令（使用仓库自带 smoke 脚本，限制 episode 数以加速）：

```bash
cd /home/agilex/cqy/flexible_lerobot/my_devs/train/act/so101
mkdir -p /tmp/hf_home /tmp/hf_datasets

HF_HOME=/tmp/hf_home \
HF_DATASETS_CACHE=/tmp/hf_datasets \
DATASET_REPO_ID=test_pipeline_clean_12_19_removed_right_arm7 \
DATASET_ROOT=/home/agilex/cqy/flexible_lerobot/datasets/lerobot_datasets/test_pipeline_clean_12_19_removed_right_arm7 \
DATASET_EPISODES='[0]' \
JOB_NAME=act_right_arm7_smoke_20260305 \
POLICY_DEVICE=cpu \
STEPS=100 \
SAVE_FREQ=100 \
LOG_FREQ=10 \
NUM_WORKERS=0 \
./train_smoke.sh
```

运行结果：

1. 环境检测：`torch.cuda.is_available() == False`，因此本次在 CPU 上训练（这是本机能力限制，不影响正确性验证）。
2. 数据集读取：成功（脚本输出 `dataset.num_frames=300`，`dataset.num_episodes=1`，因为限定了 `episodes=[0]`）。
3. 训练完成：100 steps 正常结束，exit code `0`。
4. checkpoint 落盘成功：
   - 输出目录：
     - `/home/agilex/cqy/flexible_lerobot/outputs/train/20260305_154635_act_right_arm7_smoke_20260305`
   - checkpoint：
     - `.../checkpoints/000100/pretrained_model/model.safetensors`
5. 日志文件（可追溯）：
   - `/home/agilex/cqy/flexible_lerobot/logs/train_act_right_arm7_smoke_20260305_20260305_154635.log`
   - 关键日志包含：`Start offline training...`、每 10 step 打印一次 loss、以及 `Checkpoint policy after step 100` / `End of training`。

说明：

- 该 smoke run 的目标是“证明训练链路与数据格式正确”，不是为了得到可用的收敛模型（样本少 + 步数少 + CPU）。

## 10. 总结结论

1. 代码实现与技术方案在关键规则上完全一致（相机组合、切片规则、info/stats 更新、健康检查）。
2. 工作报告中最关键的可复现声明（`pytest` 通过、端到端构建可执行、产物验收指标）已被本次复测证实。
3. 你提供的两份产物数据集：
   - 结构正确（相机键/shape/帧数）
   - 切片正确（与 source 对齐逐元素完全一致）
   - 数值健康（无 NaN/Inf，std floor 生效）
   - 可被 `LeRobotDataset` 正常加载（需设置 HF cache 环境变量）
4. 训练链路验证：`right_arm7` 已使用 `lerobot-train` 在本机（CPU）成功跑完 100 steps 并落盘 checkpoint。

综合判断：

- `test_pipeline_clean_12_19_removed_right_arm7` 可以安全用于继续训练右臂策略。
- `test_pipeline_clean_12_19_removed_left_arm7` 数值上安全，但训练是否“有意义”取决于你的左臂是否在数据中存在足够动作变化；若目标是训练左臂运动策略，建议先抽样可视化确认左臂确实在动，或补充包含左臂运动的示教数据。
