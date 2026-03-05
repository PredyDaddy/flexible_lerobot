# 单臂数据集构建与训练 Pipeline 技术方案

## 1. 背景与目标

当前 `test_pipeline`/`test_pipeline_clean_12_19_removed` 数据是双臂 14 维动作（通常左 7 + 右 7），但你的实际采集/示教里出现了“单臂主导运动、另一臂几乎静止”的场景。

在这种场景下，继续直接训练 14 维双臂策略，会出现两个问题：

1. **监督目标里包含大量无信息维度**
   - 例如静止臂 7 维几乎常量。
2. **归一化放大风险**
   - 常量维 `std≈0`，再叠加微小均值偏差，会放大归一化值，导致 loss 被单维异常拉高。

因此，需要构建“单臂专用数据集”：

1. **只保留目标臂对应关节维度（7 维）**
2. **只保留相关相机**
   - 左臂：`cam_high + cam_left_wrist`
   - 右臂：`cam_high + cam_right_wrist`

目标是让训练信号更干净、loss 更可解释、模型收敛更稳定。

---

## 2. 复用仓库现有能力（你提到的功能）

### 2.1 `lerobot_edit_dataset --operation remove_feature`（可直接用）

仓库已有官方编辑脚本，支持删除 feature（包括相机特征）：

- 脚本入口：
  - `src/lerobot/scripts/lerobot_edit_dataset.py`
- 关键能力：
  - `operation.type=remove_feature`
  - 可生成新数据集目录，不破坏原数据

参考实现位置：

- [lerobot_edit_dataset.py](/home/agilex/cqy/flexible_lerobot/src/lerobot/scripts/lerobot_edit_dataset.py)
- [dataset_tools.py remove_feature](/home/agilex/cqy/flexible_lerobot/src/lerobot/datasets/dataset_tools.py:419)

适合用于：

1. 删除 `observation.images.cam_left_wrist` 或 `observation.images.cam_right_wrist`
2. 快速得到“单腕 + 头相机”的视觉子集

### 2.2 现有能力的边界（必须说明）

`remove_feature` 只能“删列”，不能直接把 `action` 从 14 维裁成 7 维并覆盖原键名。

原因：

1. `ACTPolicy` 训练时固定读取键名 `action`
2. 单纯 add/remove 不能无缝把旧 `action` 替换成新的 7 维 `action`

因此单臂 pipeline 需要两段：

1. **段 A：用 `remove_feature` 做相机裁剪（现成）**
2. **段 B：做动作/状态维度重映射（14->7）并重算 stats（补充步骤）**

---

## 3. 单臂组合规则（标准化）

## 3.1 右臂训练组合

1. 相机保留：
   - `observation.images.cam_high`
   - `observation.images.cam_right_wrist`
2. 相机删除：
   - `observation.images.cam_left_wrist`
3. 动作/状态维度：
   - `action = action[:, 7:14]`
   - `observation.state = observation.state[:, 7:14]`

## 3.2 左臂训练组合

1. 相机保留：
   - `observation.images.cam_high`
   - `observation.images.cam_left_wrist`
2. 相机删除：
   - `observation.images.cam_right_wrist`
3. 动作/状态维度：
   - `action = action[:, 0:7]`
   - `observation.state = observation.state[:, 0:7]`

## 3.3 为什么 state 也要同步裁剪

如果只裁剪 `action` 不裁剪 `state`，模型输入仍会包含“静止臂状态”，会引入额外噪声和维度不一致风险。单臂策略建议输入/输出都做单臂闭环（7->7）。

---

## 4. 目录与命名规范（建议）

建议固定使用以下命名，便于复现实验：

1. 原始清洗集（已有）：
   - `datasets/lerobot_datasets/test_pipeline_clean_12_19_removed`
2. 右臂视觉裁剪中间集：
   - `.../test_pipeline_clean_12_19_removed_cam_high_right`
3. 左臂视觉裁剪中间集：
   - `.../test_pipeline_clean_12_19_removed_cam_high_left`
4. 右臂最终训练集（7 维）：
   - `.../test_pipeline_clean_12_19_removed_right_arm7`
5. 左臂最终训练集（7 维）：
   - `.../test_pipeline_clean_12_19_removed_left_arm7`

---

## 5. 详细工作清单（Checklist）

以下 checklist 按“先右臂，再左臂”执行，两个流程完全对称。

## 5.1 阶段 0：准备与冻结

1. 固定 Conda 环境：`lerobot_flex`
2. 冻结输入数据集路径（不要直接改原目录）
3. 创建实验记录目录（记录命令、日志、输出路径）

验收标准：

1. 原数据目录可读，且不会被覆盖
2. 每次处理都写明 `source -> target`

## 5.2 阶段 A：相机裁剪（用现成功能）

### 右臂版本（保留高位+右腕）

```bash
conda run --no-capture-output -n lerobot_flex \
python -m lerobot.scripts.lerobot_edit_dataset \
  --repo_id test_pipeline_clean_12_19_removed \
  --root /home/agilex/cqy/flexible_lerobot/datasets/lerobot_datasets \
  --new_repo_id test_pipeline_clean_12_19_removed_cam_high_right \
  --operation.type remove_feature \
  --operation.feature_names "['observation.images.cam_left_wrist']"
```

### 左臂版本（保留高位+左腕）

```bash
conda run --no-capture-output -n lerobot_flex \
python -m lerobot.scripts.lerobot_edit_dataset \
  --repo_id test_pipeline_clean_12_19_removed \
  --root /home/agilex/cqy/flexible_lerobot/datasets/lerobot_datasets \
  --new_repo_id test_pipeline_clean_12_19_removed_cam_high_left \
  --operation.type remove_feature \
  --operation.feature_names "['observation.images.cam_right_wrist']"
```

验收标准：

1. `meta/info.json` 中相机 key 数量从 3 变成 2
2. 数据帧数、episode 数与输入一致

## 5.3 阶段 B：动作/状态裁剪（14->7）

本阶段建议用一个专用“数据重写脚本”执行以下逻辑（放在 `my_devs/docs/remove_feature_pipeline` 或 `my_devs/train/act` 均可）：

1. 读取中间集 parquet
2. 覆盖写回：
   - 右臂：`action=action[:,7:14]`，`observation.state=state[:,7:14]`
   - 左臂：`action=action[:,0:7]`，`observation.state=state[:,0:7]`
3. 更新 `meta/info.json` 的 feature shape：
   - `action.shape=[7]`
   - `observation.state.shape=[7]`
4. 重算 `meta/stats.json`（至少 action/state）
5. 输出到最终训练集目录（`*_right_arm7` / `*_left_arm7`）

验收标准：

1. parquet 里 `action`、`observation.state` 维度均为 7
2. `stats.json` 中 action/state 的 `std` 不出现异常放大触发（至少避免 `std=0`+偏移导致极端归一化）
3. `lerobot-dataset-viz` 可正常读取与播放

## 5.4 阶段 C：数据健康检查（必须）

每次生成新数据集后，跑以下检查：

1. 形状检查：
   - `action.shape == [7]`
   - `observation.state.shape == [7]`
2. 相机检查：
   - 仅存在目标 2 个相机 key
3. 数值检查：
   - NaN/Inf 为 0
   - action/state 每维 `std` 合理（无异常极小值引发归一化爆炸）
4. 语义抽检：
   - 随机抽 3~5 个 episode 可视化，确认运动臂与保留相机一致

## 5.5 阶段 D：训练（单臂）

推荐先 smoke（500~1000 steps），再 full（20k+）。

右臂训练示例（使用你现有脚本）：

```bash
cd /home/agilex/cqy/flexible_lerobot

DATASET_REPO_ID=test_pipeline_clean_12_19_removed_right_arm7 \
DATASET_ROOT=/home/agilex/cqy/flexible_lerobot/datasets/lerobot_datasets/test_pipeline_clean_12_19_removed_right_arm7 \
JOB_NAME=act_right_arm7_cam_high_right \
STEPS=20000 \
BATCH_SIZE=8 \
NUM_WORKERS=2 \
WANDB_ENABLE=false \
bash my_devs/train/act/train_full.sh
```

左臂训练同理，只替换 `DATASET_REPO_ID/ROOT/JOB_NAME`。

验收标准：

1. 训练日志可见稳定下降
2. 不再出现“单维度占据 98%+ L1”的现象
3. checkpoint 可正常导出与加载

---

## 6. 左/右臂可复用 Pipeline 模板

## 6.1 右臂模板

1. 输入数据：`base_clean`
2. 删除左腕相机：`remove_feature(cam_left_wrist)`
3. 切片动作状态：`[7:14]`
4. 重算 stats
5. smoke train
6. full train

## 6.2 左臂模板

1. 输入数据：`base_clean`
2. 删除右腕相机：`remove_feature(cam_right_wrist)`
3. 切片动作状态：`[0:7]`
4. 重算 stats
5. smoke train
6. full train

---

## 7. 风险与回滚策略

## 7.1 主要风险

1. 只删相机不裁动作维度，仍会保留静止臂噪声
2. 裁动作维度但不重算 stats，loss 可能继续异常
3. 直接覆盖原数据目录，难以回滚

## 7.2 回滚

1. 所有步骤都输出到新目录
2. 每一步都保留 `meta/info.json` 与 `meta/stats.json` 备份
3. 训练前做一次 `--steps=500` smoke，失败不进入 full run

---

## 8. 执行建议（落地顺序）

1. 先完成 **右臂方案**（你的主需求）
2. 右臂方案跑通后，复用同模板生成左臂数据集
3. 两套数据集分别保留独立训练日志与 checkpoint，便于对比

---

## 9. 你下一步可以直接按这个清单做什么

1. 先执行阶段 A 的 `remove_feature` 命令（右臂版）
2. 再执行阶段 B 的 14->7 重写脚本（右臂版）
3. 完成阶段 C 检查
4. 跑 `train_full.sh` 做 1k smoke，再到 20k full

如果需要，我下一步可以把“阶段 B 的重写脚本”按这个文档直接实现成可执行脚本，并附一键命令（左/右臂各一条）。

