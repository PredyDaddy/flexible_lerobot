# remove_feature_pipeline

单臂数据集构建工具（左臂/右臂），用于从 14 维双臂数据生成 7 维单臂训练数据，并裁剪相机输入。

## 功能

1. 复用 LeRobot 原生 `remove_feature` 删除不需要的腕部相机。
2. 将 `action`、`observation.state` 从 14 维切片到 7 维：
   - `left`: `[0:7]`
   - `right`: `[7:14]`
3. 更新 `meta/info.json` 的 shape 与 names。
4. 重算 `meta/stats.json`（`action`、`observation.state`），支持 `std floor`。
5. 基础校验：相机组合、向量维度、NaN/Inf。

## 输入输出约定

1. 输入目录结构（示例）：
   - `datasets/lerobot_datasets/<source_repo_id>/...`
2. 输出目录结构（示例）：
   - `datasets/lerobot_datasets/<target_repo_id>/...`
3. 目标目录：
   - 默认在 `source-root` 下创建
   - 可通过 `--work-root` 指定父目录

## 运行示例

## 右臂训练集（保留 `cam_high + cam_right_wrist`）

```bash
conda run --no-capture-output -n lerobot_flex \
python my_devs/remove_feature_pipeline/single_arm_pipeline.py \
  --source-root /home/agilex/cqy/flexible_lerobot/datasets/lerobot_datasets \
  --source-repo-id test_pipeline_clean_12_19_removed \
  --target-repo-id test_pipeline_clean_12_19_removed_right_arm7 \
  --arm right \
  --std-floor 1e-3 \
  --overwrite
```

## 左臂训练集（保留 `cam_high + cam_left_wrist`）

```bash
conda run --no-capture-output -n lerobot_flex \
python my_devs/remove_feature_pipeline/single_arm_pipeline.py \
  --source-root /home/agilex/cqy/flexible_lerobot/datasets/lerobot_datasets \
  --source-repo-id test_pipeline_clean_12_19_removed \
  --target-repo-id test_pipeline_clean_12_19_removed_left_arm7 \
  --arm left \
  --std-floor 1e-3 \
  --overwrite
```

## Dry-run（先看计划不落盘）

```bash
conda run --no-capture-output -n lerobot_flex \
python my_devs/remove_feature_pipeline/single_arm_pipeline.py \
  --source-root /home/agilex/cqy/flexible_lerobot/datasets/lerobot_datasets \
  --source-repo-id test_pipeline_clean_12_19_removed \
  --target-repo-id tmp_plan_only \
  --arm right \
  --dry-run
```

## 测试

```bash
conda run --no-capture-output -n lerobot_flex \
pytest -q \
  my_devs/remove_feature_pipeline/test_core.py \
  my_devs/remove_feature_pipeline/tests/test_single_arm_pipeline.py
```

## 常见问题

1. `FileExistsError: Target exists`
   - 增加 `--overwrite`，或手动清理目标目录。
2. `missing required columns: action / observation.state`
   - 检查输入数据是否为标准 LeRobot 格式。
3. 相机校验失败
   - 确认输入有 `cam_high` 与对应腕部相机；不要混用 left/right 配置。
4. 训练 loss 异常偏高
   - 优先检查 `meta/stats.json` 中 `action/std`、`observation.state/std` 是否过小。
