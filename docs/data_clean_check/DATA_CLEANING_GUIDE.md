# LeRobot 数据清洗工具使用指南

## 1. 概述

LeRobot 提供了一套完整的数据集编辑工具，支持以下操作：
- **delete_episodes**: 删除指定的episodes
- **split**: 按比例或索引分割数据集
- **merge**: 合并多个数据集
- **remove_feature**: 移除特定特征

工具入口：`python -m lerobot.scripts.lerobot_edit_dataset`

## 2. 环境准备

```bash
conda activate my_lerobot
# 可选：如果你以 editable 方式安装 lerobot（`pip install -e .`），建议在仓库根目录执行
cd /path/to/flexible_lerobot
```

## 3. 删除Episodes

### 基本用法(复制一份然后这么弄弄)
自动生成 *_old 备份文件
```bash
python -m lerobot.scripts.lerobot_edit_dataset \
    --repo_id cqy/agilex_vla_demo_ee_pinocchio_Dual_Arm_Simultaneous_Execution_clean \
    --operation.type delete_episodes \
    --operation.episode_indices "[6, 17, 70, 77, 98, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 201, 211, 239]"
```
生成新数据
```bash
python -m lerobot.scripts.lerobot_edit_dataset \
    --repo_id cqy/agilex_vla_demo_ee_pinocchio_Dual_Arm_Simultaneous_Execution \
    --new_repo_id cqy/agilex_vla_demo_ee_pinocchio_Dual_Arm_Simultaneous_Execution_clean \
    --operation.type delete_episodes \
    --operation.episode_indices "[6,17,70,77,98,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,201,211,239]"
```


### 自己的删除案例
#### 仓库内路径下数据
```bash
HF_LEROBOT_HOME=/path/to/split_datasets \
python -m lerobot.scripts.lerobot_edit_dataset \
--repo_id agilex_vla_demo_ee_pinocchio_Bowl_on_Plate_Placement_test_joint \
--new_repo_id \
agilex_vla_demo_ee_pinocchio_Bowl_on_Plate_Placement_test_joint_cleaned \
--operation.type delete_episodes \
--operation.episode_indices "[0, 101, 110]"
```

#### huggingface 缓存数据
```bash
HF_HUB_OFFLINE=1 python -m lerobot.scripts.lerobot_edit_dataset \
  --root /home/agilex/.cache/huggingface/lerobot \
  --repo_id cqy/agilex_vla_demo_ee_pinocchio_Dual_Arm_Simultaneous_Execution \
  --new_repo_id cqy/agilex_vla_demo_ee_pinocchio_Dual_Arm_Simultaneous_Execution_cleaned \
  --operation.type delete_episodes \
  --operation.episode_indices "[6, 17, 70, 77, 98, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 201, 211, 239]"
```

### 保存到新数据集
```bash
python -m lerobot.scripts.lerobot_edit_dataset \
    --repo_id cqy/your_dataset \
    --new_repo_id cqy/your_dataset_cleaned \
    --operation.type delete_episodes \
    --operation.episode_indices "[0, 2, 5]"
```

## 4. 分割数据集

### 按比例分割
```bash
python -m lerobot.scripts.lerobot_edit_dataset \
    --repo_id cqy/your_dataset \
    --operation.type split \
    --operation.splits '{"train": 0.8, "val": 0.2}'
```

### 按索引分割
```bash
python -m lerobot.scripts.lerobot_edit_dataset \
    --repo_id cqy/your_dataset \
    --operation.type split \
    --operation.splits '{"train": [0,1,2,3,4,5,6,7], "val": [8,9]}'
```

### 三分割
```bash
python -m lerobot.scripts.lerobot_edit_dataset \
    --repo_id cqy/your_dataset \
    --operation.type split \
    --operation.splits '{"train": 0.6, "val": 0.2, "test": 0.2}'
```

## 5. 合并数据集

```bash
python -m lerobot.scripts.lerobot_edit_dataset \
    --repo_id cqy/merged_dataset \
    --operation.type merge \
    --operation.repo_ids "['cqy/dataset1', 'cqy/dataset2']"
```

**注意**: 合并的数据集必须具有相同的schema（特征维度、字段名、相机配置）。

## 6. 移除特征

```bash
python -m lerobot.scripts.lerobot_edit_dataset \
    --repo_id cqy/your_dataset \
    --new_repo_id cqy/your_dataset_no_front \
    --operation.type remove_feature \
    --operation.feature_names "['observation.images.camera_front']"
```

## 7. 数据质量检查

### 7.1 批量检查脚本

```python
# check_dataset_quality.py
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np

ds = LeRobotDataset('cqy/your_dataset')
bad_episodes = []

for ep_idx in range(ds.meta.total_episodes):
    ep_data = ds.get_episode(ep_idx)
    states = np.array([frame['observation.state'] for frame in ep_data])

    # 检查NaN/Inf
    has_nan = np.any(np.isnan(states)) or np.any(np.isinf(states))

    if has_nan:
        bad_episodes.append(ep_idx)
        print(f"Episode {ep_idx}: has NaN/Inf")

print(f"\nBad episodes: {bad_episodes}")
```

### 7.2 质量检查指标

| 检查项 | 阈值 | 说明 |
|--------|------|------|
| EE单帧位移 | < 5cm | 超过可能是录制抖动 |
| EE速度峰值 | < 0.8m/s | 超过需要限速回放 |
| 关节跳变 | < 0.2rad | 超过可能是数据对齐问题 |
| NaN/Inf检查 | 无 | 数据完整性 |

## 8. EE数据兼容性说明

### 数据格式对比

| 数据集类型 | observation.state维度 | EE字段 |
|-----------|---------------------|--------|
| 原版 | 14 | 无 |
| EE版(Pinocchio) | 26 | left/right_ee.{x,y,z,roll,pitch,yaw} |
| EE版(Placo) | 26 | left/right_ee.{x,y,z,wx,wy,wz} |

### 兼容性

所有清洗操作完全兼容EE数据格式。

**注意事项**:
1. 不要混合不同后端(Pinocchio/Placo)的数据集
2. 合并前检查schema一致性
3. 统计量会自动重算

## 9. 常见问题

**Q: 删除操作会影响原数据集吗？**
A: 如果不指定`--new_repo_id`，原数据集会被重命名为`*_old`，新数据集使用原名称。

**Q: 分割后的数据集在哪里？**
A: 分割后的数据集会保存在`~/.cache/huggingface/lerobot/{repo_id}_{split_name}/`。

**Q: 如何推送到HuggingFace Hub？**
A: 添加`--push_to_hub`参数。
