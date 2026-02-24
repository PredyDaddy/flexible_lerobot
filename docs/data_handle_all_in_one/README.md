## 数据检查
```bash
python3 check_lerobot_v3_dataset_final.py /home/agilex/.cache/huggingface/lerobot/cqy/agilex_vla_demo_ee_pinocchio_Bowl_on_Plate_Placement_test
```

## 数据可视化
生成各种文件
```bash
python -m vis.scripts.render_episodes \
      --root ~/.cache/huggingface/lerobot \
      --repo-id cqy/agilex_vla_demo_ee_pinocchio_Dual_Arm_Simultaneous_Execution1 \
      --episodes all \
      --out-dir ./lerobot_rendered \
      --overwrite  
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
    --output-base /home/agilex/cqy/my_lerobot/outputs/split_datasets_custom \
    --spaces both \
    --video-mode auto \
    --overwrite
```

