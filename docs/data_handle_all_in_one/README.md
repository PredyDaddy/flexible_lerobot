## 数据检查
```bash
# 从仓库根目录执行（脚本在 docs/data_clean_check/ 下）
python3 docs/data_clean_check/check_lerobot_v3_dataset_final.py \
  /home/agilex/.cache/huggingface/lerobot/cqy/agilex_vla_demo_ee_pinocchio_Bowl_on_Plate_Placement_test
```

## 数据可视化
0.4.3 baseline 推荐使用内置的 `lerobot-dataset-viz`（基于 Rerun）逐条查看 episode：

```bash
lerobot-dataset-viz \
    --repo-id cqy/agilex_vla_demo_ee_pinocchio_Dual_Arm_Simultaneous_Execution1 \
    --episode-index 0 \
    --display-compressed-images false
```

如果数据在本地路径（不走 HF repo_id），也可以直接传 `--root`：

```bash
lerobot-dataset-viz \
    --repo-id local \
    --root /home/agilex/.cache/huggingface/lerobot/cqy/agilex_vla_demo_ee_pinocchio_Dual_Arm_Simultaneous_Execution1 \
    --episode-index 0 \
    --display-compressed-images false
```

## 数据清洗
```bash
python -m lerobot.scripts.lerobot_edit_dataset \
    --repo_id cqy/agilex_vla_demo_ee_pinocchio_Dual_Arm_Simultaneous_Execution \
    --new_repo_id cqy/agilex_vla_demo_ee_pinocchio_Dual_Arm_Simultaneous_Execution_clean \
    --operation.type delete_episodes \
    --operation.episode_indices "[6,17,70,77,98,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,201,211,239]"
```

## joint 和 ee分离
```bash
python docs/split_datasets_joint_ee/split_ee_or_joint_dataset.py \
    --input-root /home/agilex/.cache/huggingface/lerobot/cqy/agilex_vla_demo_ee_pinocchio_Dual_Arm_Simultaneous_Execution_clean \
    --output-base outputs/split_datasets_custom \
    --spaces both \
    --video-mode auto \
    --overwrite
```
