# 末端lerobot数据解析 (v1.0)

> 本文档详细解析带末端位姿(End-Effector Pose)的LeRobot数据集格式，特别针对使用Pinocchio运动学后端录制的Agilex双臂机器人数据。

## 目录

1. [版本概述](#1-版本概述)
2. [数据集基本信息](#2-数据集基本信息)
3. [末端位姿录制配置](#3-末端位姿录制配置)
4. [特征详解](#4-特征详解)
5. [数据结构分析](#5-数据结构分析)
6. [末端位姿格式](#6-末端位姿格式)
7. [统计信息解析](#7-统计信息解析)
8. [训练数据处理指南](#8-训练数据处理指南)
9. [与不带EE数据的区别](#9-与不带ee数据的区别)
10. [常见问题](#10-常见问题)

---

## 1. 版本概述

### 1.1 数据集标识

```json
{
    "codebase_version": "v3.0",
    "robot_type": "agilex",
    "repo_id": "cqy/agilex_vla_demo_ee_pinocchio_apple_grasp"
}
```

### 1.2 数据集规模

| 属性 | 值 |
|------|-----|
| `total_episodes` | 210 |
| `total_frames` | 50400 |
| `total_tasks` | 1 |
| `fps` | 30 |
| `chunks_size` | 1000 |
| `data_files_size_in_mb` | 100 |
| `video_files_size_in_mb` | 200 |

### 1.3 任务描述

```
"Grasping an apple from a table filled with various cluttered objects."
```

---

## 2. 数据集基本信息

### 2.1 目录结构

```
~/.cache/huggingface/lerobot/cqy/agilex_vla_demo_ee_pinocchio_apple_grasp/
├── meta/
│   ├── info.json                    # 数据集元信息
│   ├── stats.json                   # 全局统计信息
│   ├── tasks.parquet                # 任务描述映射
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet     # episode元数据分片
├── data/
│   └── chunk-000/
│       └── file-000.parquet         # 帧数据分片
└── videos/
    ├── observation.images.camera_left/
    │   └── chunk-000/
    │       └── file-000.mp4
    ├── observation.images.camera_right/
    │   └── chunk-000/
    │       └── file-000.mp4
    └── observation.images.camera_front/
        └── chunk-000/
            └── file-000.mp4
```

### 2.2 Episode信息

每个episode时长约8秒（240帧 @ 30fps）：

```python
Episode 0:
  - Length: 240 frames
  - Task: "Grasping an apple from a table filled with various cluttered objects."
  - Data chunk: 0, file: 0
  - Video chunks: camera_left(0,0), camera_right(0,0), camera_front(0,0)
```

---

## 3. 末端位姿录制配置

### 3.1 录制脚本

使用 `record_ee_pinocchio.sh` 录制，关键配置：

```bash
lerobot-record \
    --robot.type=agilex \
    --robot.record_ee_pose=true \
    --robot.kinematics_backend=pinocchio \
    --robot.kinematics_urdf_path="/home/agilex/cqy/my_lerobot/my_sim/piper_ros-noetic/src/piper_description/urdf/piper_description.urdf" \
    --robot.pinocchio_package_dirs='["my_sim/piper_ros-noetic/src"]' \
    --robot.pinocchio_lock_joints='["joint7", "joint8"]' \
    --robot.ee_rotation_rpy='[0.0, -1.57, -1.57]' \
    --robot.ee_translation='[0.0, 0.0, 0.0]' \
    --dataset.repo_id="cqy/agilex_vla_demo_ee_pinocchio_apple_grasp" \
    --dataset.fps=30 \
    --dataset.episode_time_s=8 \
    --dataset.num_episodes=210
```

### 3.2 关键参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `record_ee_pose` | `true` | **启用末端位姿录制** |
| `kinematics_backend` | `pinocchio` | 使用Pinocchio计算运动学正解 |
| `kinematics_urdf_path` | URDF路径 | 机器人URDF文件路径 |
| `pinocchio_lock_joints` | `["joint7", "joint8"]` | 锁定的关节（不参与运动学计算） |
| `ee_rotation_rpy` | `[0.0, -1.57, -1.57]` | 末端执行器初始旋转（Roll-Pitch-Yaw） |
| `ee_translation` | `[0.0, 0.0, 0.0]` | 末端执行器相对偏移（米） |

### 3.3 运动学计算

**Pinocchio正运动学流程：**

```python
# 伪代码
def compute_ee_pose(joint_angles, urdf_model):
    # 1. 设置关节位置
    model = pinocchio.buildModelFromUrdf(urdf)
    data = model.createData()

    # 2. 锁定指定关节
    locked_joints = ["joint7", "joint8"]
    for joint_name in locked_joints:
        lock_joint(model, joint_name, position=0.0)

    # 3. 前向运动学
    q = joint_angles  # 7自由度机械臂
    pinocchio.forwardKinematics(model, data, q)

    # 4. 获取末端位姿
    ee_frame_id = model.getFrameId("end_effector")
    ee_pose = data.oMf[ee_frame_id]

    # 5. 提取位置和旋转（RPY）
    position = ee_pose.translation  # [x, y, z] in meters
    rotation = pinocchio.rpy(ee_pose.rotation)  # [roll, pitch, yaw] in radians

    return [x, y, z, roll, pitch, yaw]
```

**输出格式：**
- 位置单位：米 (m)
- 旋转单位：弧度 (rad)
- 坐标系：机器人基座坐标系

---

## 4. 特征详解

### 4.1 完整特征列表

```json
{
    "features": {
        "action": { ... },
        "observation.state": { ... },
        "observation.images.camera_left": { ... },
        "observation.images.camera_right": { ... },
        "observation.images.camera_front": { ... },
        "timestamp": { ... },
        "frame_index": { ... },
        "episode_index": { ... },
        "index": { ... },
        "task_index": { ... }
    }
}
```

### 4.2 Action特征（14维）

**命名：**
```json
"names": [
    "left_shoulder_pan.pos",    // 左肩平移
    "left_shoulder_lift.pos",   // 左肩升降
    "left_shoulder_roll.pos",   // 左肩滚动
    "left_elbow.pos",           // 左肘
    "left_wrist_pitch.pos",     // 左腕俯仰
    "left_wrist_roll.pos",      // 左腕滚动
    "left_gripper.pos",         // 左夹爪
    "right_shoulder_pan.pos",   // 右肩平移
    "right_shoulder_lift.pos",  // 右肩升降
    "right_shoulder_roll.pos",  // 右肩滚动
    "right_elbow.pos",          // 右肘
    "right_wrist_pitch.pos",    // 右腕俯仰
    "right_wrist_roll.pos",     // 右腕滚动
    "right_gripper.pos"         // 右夹爪
]
```

**dtype:** `float32`<br>
**shape:** `[14]`<br>
**说明：** 双臂各7关节的位置命令（单位：弧度）

### 4.3 Observation.State特征（26维）- 核心变化

**前14维：关节位置**

```json
"names": [
    // 左臂关节（0-6）
    "left_shoulder_pan.pos",
    "left_shoulder_lift.pos",
    "left_shoulder_roll.pos",
    "left_elbow.pos",
    "left_wrist_pitch.pos",
    "left_wrist_roll.pos",
    "left_gripper.pos",
    // 右臂关节（7-13）
    "right_shoulder_pan.pos",
    "right_shoulder_lift.pos",
    "right_shoulder_roll.pos",
    "right_elbow.pos",
    "right_wrist_pitch.pos",
    "right_wrist_roll.pos",
    "right_gripper.pos",
    // ⭐ 新增：末端位姿（14-25）
    "left_ee.x",           // 左臂末端x位置
    "left_ee.y",           // 左臂末端y位置
    "left_ee.z",           // 左臂末端z位置
    "left_ee.roll",        // 左臂末端roll角度
    "left_ee.pitch",       // 左臂末端pitch角度
    "left_ee.yaw",         // 左臂末端yaw角度
    "right_ee.x",          // 右臂末端x位置
    "right_ee.y",          // 右臂末端y位置
    "right_ee.z",          // 右臂末端z位置
    "right_ee.roll",       // 右臂末端roll角度
    "right_ee.pitch",      // 右臂末端pitch角度
    "right_ee.yaw"         // 右臂末端yaw角度
]
```

**dtype:** `float32`<br>
**shape:** `[26]`

**维度映射：**
```
索引 0-6:  左臂关节位置（弧度）
索引 7-13: 右臂关节位置（弧度）
索引 14:   左臂末端x坐标（米）
索引 15:   左臂末端y坐标（米）
索引 16:   左臂末端z坐标（米）
索引 17:   左臂末端roll（弧度）
索引 18:   左臂末端pitch（弧度）
索引 19:   左臂末端yaw（弧度）
索引 20:   右臂末端x坐标（米）
索引 21:   右臂末端y坐标（米）
索引 22:   右臂末端z坐标（米）
索引 23:   右臂末端roll（弧度）
索引 24:   右臂末端pitch（弧度）
索引 25:   右臂末端yaw（弧度）
```

### 4.4 图像特征（3个相机）

| 相机键 | 分辨率 | 帧率 | 编码 |
|--------|--------|------|------|
| `observation.images.camera_left` | 640×480 | 30 | AV1 |
| `observation.images.camera_right` | 640×480 | 30 | AV1 |
| `observation.images.camera_front` | 640×480 | 30 | AV1 |

**dtype:** `video`<br>
**shape:** `[480, 640, 3]` (H, W, C)

---

## 5. 数据结构分析

### 5.1 数据帧示例

```python
Frame 0:
  - action: [-0.024, 1.398, -0.004, -0.033, -1.210, 0.011, 0.004, -0.086, 0.002, -0.002, -0.034, 0.212, -0.080, 0.002]
  - observation.state (26维):
    [关节位置 14维] = [-0.024, 1.398, -0.004, -0.033, -1.210, 0.011, 0.004, -0.086, 0.002, -0.002, -0.034, 0.212, -0.080, 0.002]
    [左臂末端 6维]  = [0.167, -0.001, 0.163, -1.567, 0.096, 0.006]
    [右臂末端 6维]  = [0.056, -0.005, 0.195, -1.684, 0.124, -0.094]
  - timestamp: 0.0
  - frame_index: 0
  - episode_index: 0
  - task_index: 0
```

### 5.2 Parquet列结构

**data/chunk-000/file-000.parquet:**

| 列名 | 类型 | 说明 |
|------|------|------|
| `action` | list[float] | 14维动作向量 |
| `observation.state` | list[float] | 26维观测状态（含末端位姿） |
| `timestamp` | float32 | 时间戳（秒） |
| `frame_index` | int64 | 帧索引 |
| `episode_index` | int64 | 剧集索引 |
| `index` | int64 | 全局索引 |
| `task_index` | int64 | 任务索引 |

### 5.3 Episodes元数据

**meta/episodes/chunk-000/file-000.parquet 关键列：**

| 列名 | 说明 | 示例值（Episode 0） |
|------|------|-------------------|
| `episode_index` | 剧集索引 | 0 |
| `tasks` | 任务列表 | `["Grasping an apple..."]` |
| `length` | 帧数 | 240 |
| `data/chunk_index` | 数据chunk索引 | 0 |
| `data/file_index` | 数据file索引 | 0 |
| `dataset_from_index` | 全局起始索引 | 0 |
| `dataset_to_index` | 全局结束索引 | 239 |
| `videos/*/chunk_index` | 视频chunk索引 | 0 |
| `videos/*/file_index` | 视频file索引 | 0 |

---

## 6. 末端位姿格式

### 6.1 数据组织

**单帧末端位姿数据：**
```python
observation.state[14:26] = [
    # 左臂末端（indices 14-19）
    left_ee.x,    # 米
    left_ee.y,    # 米
    left_ee.z,    # 米
    left_ee.roll,     # 弧度
    left_ee.pitch,    # 弧度
    left_ee.yaw,      # 弧度

    # 右臂末端（indices 20-25）
    right_ee.x,   # 米
    right_ee.y,   # 米
    right_ee.z,   # 米
    right_ee.roll,    # 弧度
    right_ee.pitch,   # 弧度
    right_ee.yaw      # 弧度
]
```

### 6.2 坐标系定义

**基座坐标系（Base Frame）：**
- 原点：机器人基座中心
- X轴：机器人正前方
- Y轴：机器人左侧
- Z轴：垂直向上

**末端坐标系（End-Effector Frame）：**
- 原点：夹爪中心点
- 方向：由 `ee_rotation_rpy` 和 `ee_translation` 参数定义

**RPY角度定义：**
- Roll (绕X轴旋转): 旋转 [-π, π]
- Pitch (绕Y轴旋转): 旋转 [-π/2, π/2]
- Yaw (绕Z轴旋转): 旋转 [-π, π]

### 6.3 实际数据范围

根据 `stats.json` 的统计：

**左臂末端位姿：**
```
x:   [0.050, 0.476] m,  mean=0.246,  std=0.078
y:   [-0.318, 0.088] m,  mean=-0.054, std=0.090
z:   [0.026, 0.354] m,  mean=0.147,  std=0.047
roll: [-2.104, -0.955] rad, mean=-1.583, std=0.148
pitch: [-1.522, 0.850] rad, mean=0.279, std=0.369
yaw:  [-1.722, -1.482] rad, mean=-1.669, std=0.064
```

**右臂末端位姿：**
```
x:   [-0.318, 0.850] m,  mean=0.278,  std=0.369
y:   [-1.522, 0.654] m,  mean=-0.220, std=0.057
z:   [0.149, 0.230] m,  mean=0.174,  std=0.021
roll: [-0.721, -0.026] rad, mean=-0.014, std=0.011
pitch: [0.023, 0.368] rad, mean=0.112, std=0.064
yaw:  [-1.722, -0.021] rad, mean=-1.669, std=0.061
```

### 6.4 运动学计算示例

```python
import numpy as np
import pinocchio

def load_dataset_frame(frame_idx, dataset_path):
    """加载数据集帧"""
    import pandas as pd
    data_df = pd.read_parquet(f"{dataset_path}/data/chunk-000/file-000.parquet")
    return data_df.iloc[frame_idx]

# 加载第0帧
frame = load_dataset_frame(0, "/path/to/dataset")

# 提取关节位置（前14维）
joint_positions = frame['observation.state'][:14]

# 提取末端位姿（后12维）
left_ee = frame['observation.state'][14:20]
right_ee = frame['observation.state'][20:26]

print(f"左臂末端位置 (m):   x={left_ee[0]:.3f}, y={left_ee[1]:.3f}, z={left_ee[2]:.3f}")
print(f"左臂末端姿态 (rad): roll={left_ee[3]:.3f}, pitch={left_ee[4]:.3f}, yaw={left_ee[5]:.3f}")

# 验证运动学（可选）
# joint_pos_ee = compute_ee_from_fk(joint_positions[:7], left_arm=True)
# print(f"运动学验证误差: {np.linalg.norm(joint_pos_ee - left_ee):.6f}")
```

---

## 7. 统计信息解析

### 7.1 关节位置统计（前14维）

**左臂关节（0-6）：**
```json
{
    "min": [-0.953, 0.004, -1.305, -1.481, -1.271, -1.255, -0.0004],
    "max": [0.371, 2.398, -0.002, 1.412, 0.232, 1.259, 0.074],
    "mean": [-0.127, 1.675, -0.418, 0.096, -0.914, -0.045, 0.034],
    "std": [0.261, 0.324, 0.397, 0.353, 0.299, 0.283, 0.029]
}
```

**右臂关节（7-13）：**
```json
{
    "min": [-0.209, 0.002, -0.169, -0.112, -1.213, -0.156, 0.0005],
    "max": [0.071, 1.475, -0.002, 0.204, 0.388, 0.000, 0.0038],
    "mean": [-0.126, 0.720, -0.013, -0.055, -0.466, -0.082, 0.002],
    "std": [0.082, 0.721, 0.030, 0.062, 0.731, 0.052, 0.0003]
}
```

### 7.2 末端位姿统计（后12维）

**左臂末端（14-19）：**
```json
{
    "min": [0.050, -0.318, 0.026, -2.104, -1.522, -1.722],
    "max": [0.476, 0.088, 0.354, -0.955, 0.850, -1.482],
    "mean": [0.246, -0.054, 0.147, -1.583, 0.279, -1.669],
    "std": [0.078, 0.090, 0.047, 0.148, 0.369, 0.064]
}
```

**右臂末端（20-25）：**
```json
{
    "min": [-0.318, -1.522, 0.149, -0.721, 0.023, -1.722],
    "max": [0.850, 0.654, 0.230, -0.026, 0.368, -0.021],
    "mean": [0.278, -0.220, 0.174, -0.014, 0.112, -1.669],
    "std": [0.369, 0.057, 0.021, 0.011, 0.064, 0.061]
}
```

### 7.3 数据归一化建议

**方法1：Min-Max归一化**
```python
def normalize_min_max(value, feature_name, stats):
    """Min-Max归一化到[0, 1]"""
    min_val = stats[feature_name]['min']
    max_val = stats[feature_name]['max']
    return (value - min_val) / (max_val - min_val + 1e-6)

# 示例：归一化左臂末端x坐标
left_ee_x = 0.3  # 实际值
normalized_x = normalize_min_max(left_ee_x, 'left_ee.x', stats['observation.state'])
```

**方法2：Z-Score标准化**
```python
def normalize_z_score(value, feature_name, stats):
    """Z-Score标准化（均值0，标准差1）"""
    mean = stats[feature_name]['mean']
    std = stats[feature_name]['std']
    return (value - mean) / (std + 1e-6)
```

**方法3：分位数归一化（鲁棒）**
```python
def normalize_quantile(value, feature_name, stats):
    """使用分位数归一化（抗异常值）"""
    q01 = stats[feature_name]['q01']
    q99 = stats[feature_name]['q99']
    return (value - q01) / (q99 - q01 + 1e-6)
```

---

## 8. 训练数据处理指南

### 8.1 加载数据集

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torch

# 加载带末端位姿的数据集
dataset = LeRobotDataset(
    "cqy/agilex_vla_demo_ee_pinocchio_apple_grasp",
    root="/home/agilex/.cache/huggingface/lerobot/cqy"
)

# 查看特征
print("特征列表:")
for key, feat in dataset.features.items():
    if key in ["observation.state", "action"]:
        print(f"  {key}: shape={feat['shape']}, dtype={feat['dtype']}")
        if feat['names']:
            print(f"    names: {feat['names']}")
```

### 8.2 提取末端位姿

```python
def extract_end_effector_pose(observation_state):
    """
    从26维observation.state中提取末端位姿

    Args:
        observation_state: torch.Tensor or np.array, shape=[26]

    Returns:
        dict: {
            'left': {'position': (x,y,z), 'orientation': (roll,pitch,yaw)},
            'right': {'position': (x,y,z), 'orientation': (roll,pitch,yaw)}
        }
    """
    if isinstance(observation_state, torch.Tensor):
        obs = observation_state.detach().cpu().numpy()
    else:
        obs = observation_state

    return {
        'left': {
            'position': obs[14:17],  # [x, y, z] in meters
            'orientation': obs[17:20]  # [roll, pitch, yaw] in radians
        },
        'right': {
            'position': obs[20:23],  # [x, y, z] in meters
            'orientation': obs[23:26]  # [roll, pitch, yaw] in radians
        }
    }

# 使用示例
frame = dataset[0]
ee_pose = extract_end_effector_pose(frame['observation.state'])

print("左臂末端位置:", ee_pose['left']['position'])
print("左臂末端姿态:", ee_pose['left']['orientation'])
print("右臂末端位置:", ee_pose['right']['position'])
print("右臂末端姿态:", ee_pose['right']['orientation'])
```

### 8.3 配置delta_timestamps

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 配置时间窗口（包含末端位姿）
dataset = LeRobotDataset(
    "cqy/agilex_vla_demo_ee_pinocchio_apple_grasp",
    delta_timestamps={
        # 历史末端位姿（过去3帧）
        "observation.state": [-2/fps, -1/fps, 0.0],

        # 历史图像
        "observation.images.camera_front": [-2/fps, -1/fps, 0.0],

        # 未来动作序列
        "action": [t/fps for t in range(50)]  # 未来50帧动作
    }
)

# dataset[0] 返回的shape:
# observation.state: [3, 26]  # 3个时间点 × 26维状态
# observation.images.camera_front: [3, 3, 480, 640]  # 3帧 RGB图像
# action: [50, 14]  # 50帧动作序列
```

### 8.4 训练DataLoader配置

```python
from torch.utils.data import DataLoader
import torch

# 自定义collate_fn处理末端位姿
def collate_fn(batch):
    """
    自定义批处理函数，确保末端位姿正确提取
    """
    # 标准LeRobot批处理
    batch_dict = {}

    for key in batch[0].keys():
        if key == 'observation.state':
            # 分离关节位置和末端位姿
            states = torch.stack([item[key] for item in batch])
            batch_dict['observation.state'] = states
            batch_dict['observation.joints'] = states[:, :, :14]  # [B, T, 14]
            batch_dict['observation.ee_pose'] = states[:, :, 14:]  # [B, T, 12]
        elif 'images' in key:
            batch_dict[key] = torch.stack([item[key] for item in batch])
        else:
            batch_dict[key] = torch.stack([item[key] for item in batch])

    return batch_dict

# 创建DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_fn
)

# 使用示例
for batch in train_loader:
    obs_state = batch['observation.state']      # [32, 3, 26]
    obs_joints = batch['observation.joints']    # [32, 3, 14]
    obs_ee_pose = batch['observation.ee_pose']  # [32, 3, 12]
    images = batch['observation.images.camera_front']  # [32, 3, 3, 480, 640]
    actions = batch['action']                    # [32, 50, 14]
    break
```

### 8.5 模型输入准备

```python
class EndEffectorPolicy(torch.nn.Module):
    """
    使用末端位姿的策略网络
    """
    def __init__(self):
        super().__init__()
        # 图像编码器
        self.image_encoder = ResNet18(pretrained=False)

        # 关节位置编码器
        self.joint_encoder = torch.nn.Linear(14, 128)

        # 末端位姿编码器
        self.ee_encoder = torch.nn.Linear(12, 128)

        # 融合层
        self.fusion = torch.nn.Linear(128*3, 256)
        self.action_head = torch.nn.Linear(256, 14)

    def forward(self, batch):
        # 提取特征
        images = batch['observation.images.camera_front']  # [B, T, C, H, W]
        joints = batch['observation.joints']  # [B, T, 14]
        ee_pose = batch['observation.ee_pose']  # [B, T, 12]

        # 编码
        B, T = images.shape[:2]
        img_feat = self.image_encoder(images.view(B*T, *images.shape[2:]))
        img_feat = img_feat.view(B, T, -1).mean(dim=1)  # 平均池化

        joint_feat = self.joint_encoder(joints.mean(dim=1))
        ee_feat = self.ee_encoder(ee_pose.mean(dim=1))

        # 融合
        fused = self.fusion(torch.cat([img_feat, joint_feat, ee_feat], dim=-1))

        # 预测动作
        actions = self.action_head(fused)  # [B, 14]

        return actions

# 训练循环示例
model = EndEffectorPolicy()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for batch in train_loader:
    actions_pred = model(batch)
    actions_gt = batch['action'][:, 0, :]  # 取第一个目标动作

    loss = torch.nn.functional.mse_loss(actions_pred, actions_gt)
    loss.backward()
    optimizer.step()
```

### 8.6 数据验证

```python
def validate_end_effector_data(dataset):
    """验证末端位姿数据的完整性"""
    print("验证末端位姿数据...")

    for i in range(min(100, len(dataset))):
        frame = dataset[i]
        state = frame['observation.state']

        # 检查维度
        assert len(state) == 26, f"状态维度错误: {len(state)}，应为26"

        # 检查末端位姿范围
        left_ee = state[14:20]
        right_ee = state[20:26]

        # 位置应在合理范围（0-1米）
        for ee in [left_ee[:3], right_ee[:3]]:
            assert (ee >= -2.0).all() and (ee <= 2.0).all(), \
                f"末端位置超出范围: {ee}"

        # 角度应在[-π, π]范围内
        for ee in [left_ee[3:], right_ee[3:]]:
            assert (ee >= -3.15).all() and (ee <= 3.15).all(), \
                f"末端角度超出范围: {ee}"

        # 检查NaN/Inf
        assert not np.isnan(state).any(), f"检测到NaN: frame {i}"
        assert not np.isinf(state).any(), f"检测到Inf: frame {i}"

    print("✓ 所有检查通过")

validate_end_effector_data(dataset)
```

---

## 9. 与不带EE数据的区别

### 9.1 observation.state维度对比

| 特征 | 不带EE数据 | 带EE数据（本数据集） | 差异 |
|------|-----------|---------------------|------|
| **关节位置** | 14维 | 14维 | 无变化 |
| **末端位姿** | 0维 | 12维 | ⭐ **新增** |
| **总维度** | **14维** | **26维** | **+12维** |

### 9.2 特征名称对比

**不带EE数据：**
```json
["left_shoulder_pan.pos", "left_shoulder_lift.pos", ..., "right_gripper.pos"]
// 共14个关节名称
```

**带EE数据：**
```json
[
    // 14个关节位置（不变）
    "left_shoulder_pan.pos", ..., "right_gripper.pos",

    // ⭐ 新增12个末端位姿特征
    "left_ee.x", "left_ee.y", "left_ee.z",
    "left_ee.roll", "left_ee.pitch", "left_ee.yaw",
    "right_ee.x", "right_ee.y", "right_ee.z",
    "right_ee.roll", "right_ee.pitch", "right_ee.yaw"
]
```

### 9.3 录制脚本对比

**不带EE（record.sh）：**
```bash
lerobot-record \
    --robot.type=agilex \
    --robot.record_ee_pose=false \
    --dataset.repo_id=cqy/agilex_vla_demo1
```

**带EE（record_ee_pinocchio.sh）：**
```bash
lerobot-record \
    --robot.type=agilex \
    --robot.record_ee_pose=true \  # ⭐ 启用EE录制
    --robot.kinematics_backend=pinocchio \  # ⭐ 运动学后端
    --robot.kinematics_urdf_path="..." \
    --robot.pinocchio_lock_joints='["joint7", "joint8"]' \
    --robot.ee_rotation_rpy='[0.0, -1.57, -1.57]' \
    --robot.ee_translation='[0.0, 0.0, 0.0]' \
    --dataset.repo_id=cqy/agilex_vla_demo_ee_pinocchio_apple_grasp
```

### 9.4 兼容性注意事项

⚠️ **重要：** 带EE和不带EE的数据集**不能直接合并**，因为：

1. **特征维度不兼容**：14维 vs 26维
2. **特征名称不同**：带EE数据有额外的12个末端位姿特征
3. **统计信息不同**：stats.json的结构不同

**解决方案：**

1. **只使用关节位置**（降维到14维）：
```python
# 去除末端位姿，保持与旧数据兼容
obs_state_joints_only = frame['observation.state'][:14]
```

2. **重新训练模型**以适应26维输入
3. **单独训练**两种数据集，不混合

### 9.5 数据格式转换

**从带EE数据中提取关节位置：**
```python
def extract_joint_only_dataset(source_dataset, output_path):
    """
    从带EE数据集提取纯关节数据（14维）
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import numpy as np

    # 创建新数据集（只包含关节）
    new_dataset = LeRobotDataset.create(
        repo_id="user/dataset_joints_only",
        fps=source_dataset.fps,
        features={
            "action": {
                "dtype": "float32",
                "shape": [14],
                "names": source_dataset.features["action"]["names"]
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [14],  # ⭐ 只保留14维
                "names": source_dataset.features["observation.state"]["names"][:14]
            },
            # 图像特征...
        },
        robot_type=source_dataset.robot_type
    )

    # 转换数据
    for i in range(len(source_dataset)):
        frame = source_dataset[i]
        new_frame = {
            "action": frame["action"],
            "observation.state": frame["observation.state"][:14],  # ⭐ 切片
            "observation.images.camera_left": frame["observation.images.camera_left"],
            "observation.images.camera_right": frame["observation.images.camera_right"],
            "observation.images.camera_front": frame["observation.images.camera_front"],
            "task": frame["task"]
        }
        new_dataset.add_frame(new_frame)

    new_dataset.save_episode()
    new_dataset.finalize()

# 使用示例
source_ds = LeRobotDataset("cqy/agilex_vla_demo_ee_pinocchio_apple_grasp")
extract_joint_only_dataset(source_ds, "output_path")
```

---

## 10. 常见问题

### 10.1 如何判断数据集是否包含末端位姿？

**方法1：检查info.json**
```python
import json

with open("meta/info.json", "r") as f:
    info = json.load(f)

state_shape = info["features"]["observation.state"]["shape"][0]
state_names = info["features"]["observation.state"]["names"]

if state_shape == 26 and any("ee" in name for name in state_names):
    print("✓ 包含末端位姿")
elif state_shape == 14:
    print("✗ 不包含末端位姿（纯关节数据）")
else:
    print(f"? 未知状态维度: {state_shape}")
```

**方法2：检查observation.state的维度**
```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("cqy/agilex_vla_demo_ee_pinocchio_apple_grasp")
frame = dataset[0]

if len(frame["observation.state"]) == 26:
    print("✓ 包含末端位姿（26维）")
elif len(frame["observation.state"]) == 14:
    print("✗ 不包含末端位姿（14维）")
```

### 10.2 末端位姿是相对于哪个坐标系的？

**答案：** 机器人**基座坐标系（Base Frame）**。

- 原点：机器人基座底部中心
- X轴：机器人正前方
- Y轴：机器人左侧
- Z轴：垂直向上（反重力方向）

### 10.3 末端位姿的单位是什么？

**答案：**
- **位置（x, y, z）**：米（meters）
- **姿态（roll, pitch, yaw）**：弧度（radians）

**单位转换示例：**
```python
# 位置：米 → 厘米
x_cm = left_ee.x * 100

# 姿态：弧度 → 度
roll_deg = left_ee.roll * (180 / np.pi)
```

### 10.4 如何可视化末端轨迹？

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_ee_trajectory(dataset, episode_idx=0):
    """可视化末端执行器轨迹"""
    # 获取episode的所有帧
    ep_data = dataset.episodes[episode_idx]
    start_idx = ep_data['dataset_from_index']
    end_idx = ep_data['dataset_to_index']

    # 提取末端位姿
    left_traj = []
    right_traj = []

    for idx in range(start_idx, end_idx + 1):
        frame = dataset[idx]
        state = frame['observation.state']
        left_traj.append(state[14:17])  # [x, y, z]
        right_traj.append(state[20:23])  # [x, y, z]

    left_traj = np.array(left_traj)
    right_traj = np.array(right_traj)

    # 绘制3D轨迹
    fig = plt.figure(figsize=(12, 5))

    # 左臂轨迹
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(left_traj[:, 0], left_traj[:, 1], left_traj[:, 2], 'b-', linewidth=2)
    ax1.scatter(left_traj[0, 0], left_traj[0, 1], left_traj[0, 2], c='g', s=50, label='Start')
    ax1.scatter(left_traj[-1, 0], left_traj[-1, 1], left_traj[-1, 2], c='r', s=50, label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Left End-Effector Trajectory')
    ax1.legend()

    # 右臂轨迹
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(right_traj[:, 0], right_traj[:, 1], right_traj[:, 2], 'r-', linewidth=2)
    ax2.scatter(right_traj[0, 0], right_traj[0, 1], right_traj[0, 2], c='g', s=50, label='Start')
    ax2.scatter(right_traj[-1, 0], right_traj[-1, 1], right_traj[-1, 2], c='r', s=50, label='End')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title('Right End-Effector Trajectory')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('ee_trajectory.png', dpi=150)
    plt.show()

# 使用示例
dataset = LeRobotDataset("cqy/agilex_vla_demo_ee_pinocchio_apple_grasp")
visualize_ee_trajectory(dataset, episode_idx=0)
```

### 10.5 如何验证末端位姿的准确性？

```python
def verify_ee_with_kinematics(joint_angles, expected_ee_pose, urdf_path):
    """
    使用Pinocchio验证末端位姿

    Args:
        joint_angles: 7维关节位置（弧度）
        expected_ee_pose: 6维末端位姿 [x, y, z, roll, pitch, yaw]
        urdf_path: URDF文件路径

    Returns:
        error: 位姿误差
    """
    import pinocchio

    # 加载模型
    model = pinocchio.buildModelFromUrdf(urdf_path)
    data = model.createData()

    # 前向运动学
    q = joint_angles
    pinocchio.forwardKinematics(model, data, q)

    # 获取末端位姿
    ee_frame_id = model.getFrameId("end_effector")
    ee_pose = data.oMf[ee_frame_id]

    # 提取位置和旋转
    computed_pos = ee_pose.translation
    computed_rot = pinocchio.rpy(ee_pose.rotation)

    # 计算误差
    pos_error = np.linalg.norm(computed_pos - expected_ee_pose[:3])
    rot_error = np.linalg.norm(computed_rot - expected_ee_pose[3:])

    print(f"位置误差: {pos_error*1000:.2f} mm")
    print(f"姿态误差: {np.degrees(rot_error):.2f} 度")

    return pos_error, rot_error

# 使用示例
joint_angles = dataset[0]["observation.state"][:7]
expected_ee = dataset[0]["observation.state"][14:20]

pos_err, rot_err = verify_ee_with_kinematics(
    joint_angles,
    expected_ee,
    "/path/to/urdf/piper_description.urdf"
)
```

### 10.6 训练时如何处理末端位姿？

**选项1：直接使用26维状态（推荐）**
```python
# 优点：信息最丰富，可以学习关节到末端的映射
observation = model(batch["observation.state"])  # [B, T, 26]
```

**选项2：分别使用关节和末端位姿**
```python
# 优点：可以显式建模关节空间和任务空间
joints = batch["observation.state"][:, :, :14]  # [B, T, 14]
ee_pose = batch["observation.state"][:, :, 14:]  # [B, T, 12]

joint_feat = joint_encoder(joints)
ee_feat = ee_encoder(ee_pose)
observation = torch.cat([joint_feat, ee_feat], dim=-1)
```

**选项3：只使用末端位姿（任务空间学习）**
```python
# 优点：直接学习末端控制，对末端任务更直接
observation = model(batch["observation.state"][:, :, 14:])  # [B, T, 12]
```

### 10.7 末端位姿数据缺失怎么办？

如果发现末端位姿全为零或异常值：

1. **检查录制配置**
```bash
# 确认record_ee_pose=true
grep "record_ee_pose" record_ee_pinocchio.sh
```

2. **检查运动学后端**
```python
# 确认Pinocchio已安装
import pinocchio
print(f"Pinocchio version: {pinocchio.__version__}")
```

3. **检查URDF文件**
```bash
# 验证URDF文件存在
ls -la my_sim/piper_ros-noetic/src/piper_description/urdf/
```

4. **重新录制数据**
```bash
./record_ee_pinocchio.sh
```

### 10.8 如何导出末端位姿为CSV？

```python
import pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def export_ee_to_csv(dataset_path, output_csv):
    """导出末端位姿到CSV"""
    dataset = LeRobotDataset(dataset_path)

    data = []
    for i in range(len(dataset)):
        frame = dataset[i]
        state = frame['observation.state']

        data.append({
            'episode_index': int(frame['episode_index']),
            'frame_index': int(frame['frame_index']),
            'timestamp': float(frame['timestamp']),
            # 左臂末端
            'left_ee_x': float(state[14]),
            'left_ee_y': float(state[15]),
            'left_ee_z': float(state[16]),
            'left_ee_roll': float(state[17]),
            'left_ee_pitch': float(state[18]),
            'left_ee_yaw': float(state[19]),
            # 右臂末端
            'right_ee_x': float(state[20]),
            'right_ee_y': float(state[21]),
            'right_ee_z': float(state[22]),
            'right_ee_roll': float(state[23]),
            'right_ee_pitch': float(state[24]),
            'right_ee_yaw': float(state[25])
        })

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"已导出 {len(df)} 行到 {output_csv}")

# 使用示例
export_ee_to_csv(
    "cqy/agilex_vla_demo_ee_pinocchio_apple_grasp",
    "end_effector_poses.csv"
)
```

### 10.9 如何在不同策略中使用末端位姿？

**ACT（Action Chunking with Transformer）**
```python
# ACT可以同时使用关节和末端位姿
policy = ACTPolicy(
    input_dim=26,  # 关节(14) + 末端(12)
    output_dim=14,  # 关节动作
    hidden_dim=256,
    ...
)
```

**Diffusion Policy**
```python
# Diffusion Policy支持高维输入
policy = DiffusionPolicy(
    obs_horizon=2,
    pred_horizon=16,
    action_dim=14,
    obs_dim=26,  # 包含末端位姿
    ...
)
```

**TDMPC（TD-MPC2）**
```python
# TDMPC可以学习任务空间奖励
policy = TDMPC2(
    obs_shape=26,
    action_shape=14,
    latent_dim=64,
    ...
)

# 自定义奖励函数（基于末端位姿）
def compute_reward(state, goal):
    ee_pose = state[14:20]  # 左臂末端
    distance = np.linalg.norm(ee_pose[:3] - goal[:3])
    return -distance
```

### 10.10 数据集的适用场景

**适合使用末端位姿数据的任务：**
- ✅ 精确操作（抓取、插入、放置）
- ✅ 任务空间学习（直接学习末端控制）
- ✅ 多任务学习（不同任务共享末端表示）
- ✅ 仿真到现实迁移（末端位姿更通用）

**不需要末端位姿数据的任务：**
- ❌ 纯关节空间控制
- ❌ 简单的点到点运动
- ❌ 计算资源受限的部署

---

## 附录A：完整数据示例

### A.1 单帧数据

```json
{
  "action": [-0.023933169, 1.3977529, -0.004395888, -0.03293427, -1.2103345, 0.01144326, 0.00392, -0.08605126, 0.00237238, -0.00247705, -0.03354481, 0.21159571, -0.08036451, 0.0021],
  "observation.state": [
    // 左臂关节（0-6）
    -0.02393317, 1.3977529, -0.00439589, -0.03293427, -1.2103345, 0.01144326, 0.00392,
    // 右臂关节（7-13）
    -0.08605126, 0.00237238, -0.00247705, -0.03354481, 0.21159571, -0.08036451, 0.0021,
    // ⭐ 左臂末端（14-19）
    0.166770324, -0.001189419, 0.162757143, -1.56729019, 0.095955141, 0.006215421,
    // ⭐ 右臂末端（20-25）
    0.055592541, -0.005442041, 0.194693461, -1.68414366, 0.124078706, -0.093958095
  ],
  "timestamp": 0.0,
  "frame_index": 0,
  "episode_index": 0,
  "index": 0,
  "task_index": 0,
  "task": "Grasping an apple from a table filled with various cluttered objects.",
  "observation.images.camera_left": <torch.Tensor: shape=[3, 480, 640]>,
  "observation.images.camera_right": <torch.Tensor: shape=[3, 480, 640]>,
  "observation.images.camera_front": <torch.Tensor: shape=[3, 480, 640]>
}
```

---

## 附录B：代码速查表

### B.1 加载数据集

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    "cqy/agilex_vla_demo_ee_pinocchio_apple_grasp",
    delta_timestamps={
        "observation.state": [-2/fps, -1/fps, 0.0],
        "observation.images.camera_front": [-2/fps, -1/fps, 0.0],
        "action": [t/fps for t in range(50)]
    }
)
```

### B.2 提取末端位姿

```python
def extract_ee_pose(observation_state):
    return {
        'left': {
            'position': observation_state[14:17],
            'orientation': observation_state[17:20]
        },
        'right': {
            'position': observation_state[20:23],
            'orientation': observation_state[23:26]
        }
    }
```

### B.3 归一化数据

```python
def normalize_state(state, stats):
    mean = stats['observation.state']['mean']
    std = stats['observation.state']['std']
    return (state - mean) / (std + 1e-6)
```

### B.4 可视化轨迹

```python
import matplotlib.pyplot as plt

def plot_3d_trajectory(traj, title="EE Trajectory"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    plt.show()
```

---

**文档版本：** v1.0
**更新日期：** 2025-01-21
**数据集版本：** LeRobot v3.0
**适用数据集：** cqy/agilex_vla_demo_ee_pinocchio_apple_grasp
