# 末端位姿 LeRobot 数据格式技术文档 (v3.0)

> 本文档详细解析使用 Pinocchio 运动学计算末端位姿的 LeRobot 数据集格式，适用于 Agilex 双臂机器人的 VLA (Vision-Language-Action) 模型训练。

## 目录

1. [数据集概述](#1-数据集概述)
2. [与标准数据集的区别](#2-与标准数据集的区别)
3. [目录结构](#3-目录结构)
4. [特征定义详解](#4-特征定义详解)
5. [末端位姿计算原理](#5-末端位姿计算原理)
6. [数据文件格式](#6-数据文件格式)
7. [统计信息](#7-统计信息)
8. [数据加载与使用](#8-数据加载与使用)
9. [训练注意事项](#9-训练注意事项)
10. [常见问题](#10-常见问题)

---

## 1. 数据集概述

### 1.1 基本信息

| 属性 | 值 |
|------|-----|
| 数据集名称 | `cqy/agilex_vla_demo_ee_pinocchio_apple_grasp` |
| 格式版本 | v3.0 |
| 机器人类型 | agilex (双臂机器人) |
| 总 Episode 数 | 210 |
| 总帧数 | 50,400 |
| 采集帧率 | 30 FPS |
| 单 Episode 时长 | 约 8 秒 (240 帧) |
| 任务描述 | "Grasping an apple from a table filled with various cluttered objects." |

### 1.2 数据集特点

本数据集的核心特点是 **包含末端执行器 (End-Effector, EE) 位姿信息**：

- 使用 **Pinocchio** 库进行正向运动学计算
- 末端位姿格式：`[x, y, z, roll, pitch, yaw]`
- 双臂各自独立计算末端位姿
- 末端位姿与关节角度同步录制

### 1.3 录制配置

数据集使用以下关键配置录制：

```bash
--robot.record_ee_pose=true                    # 启用末端位姿录制
--robot.kinematics_backend=pinocchio           # 使用 Pinocchio 运动学后端
--robot.pinocchio_lock_joints='["joint7", "joint8"]'  # 锁定夹爪关节
--robot.ee_rotation_rpy='[0.0, -1.57, -1.57]'  # 末端坐标系旋转补偿
```

---

## 2. 与标准数据集的区别

### 2.1 observation.state 维度对比

| 数据集类型 | observation.state 维度 | 内容 |
|-----------|----------------------|------|
| **标准数据集** | 14 维 | 仅关节角度 (7 × 2 臂) |
| **末端位姿数据集** | 26 维 | 关节角度 (14) + 末端位姿 (6 × 2 臂) |

### 2.2 特征结构对比

**标准数据集 observation.state (14维):**
```
[left_arm_joints(7), right_arm_joints(7)]
```

**末端位姿数据集 observation.state (26维):**
```
[left_arm_joints(7), right_arm_joints(7), left_ee_pose(6), right_ee_pose(6)]
```

### 2.3 action 维度

action 维度保持不变，仍为 **14 维**（仅关节角度），因为机器人控制仍然基于关节空间。

---

## 3. 目录结构

```
agilex_vla_demo_ee_pinocchio_apple_grasp/
├── meta/
│   ├── info.json                              # 数据集元信息
│   ├── stats.json                             # 全局统计信息
│   ├── tasks.parquet                          # 任务描述
│   └── episodes/
│       └── chunk-000/
│           ├── file-000.parquet               # Episode 0-49 元数据
│           ├── file-001.parquet               # Episode 50-99 元数据
│           └── ...
├── data/
│   └── chunk-000/
│       ├── file-000.parquet                   # 帧数据 (约 12000 帧)
│       ├── file-001.parquet
│       └── ...
└── videos/
    ├── observation.images.camera_left/
    │   └── chunk-000/
    │       ├── file-000.mp4                   # 左相机视频
    │       └── ...
    ├── observation.images.camera_right/
    │   └── chunk-000/
    │       └── ...
    └── observation.images.camera_front/
        └── chunk-000/
            └── ...
```

---

## 4. 特征定义详解

### 4.1 完整特征列表

数据集包含以下特征：

| 特征名 | 数据类型 | 形状 | 说明 |
|--------|---------|------|------|
| `observation.state` | float32 | [26] | 关节角度 + 末端位姿 |
| `action` | float32 | [14] | 关节角度动作 |
| `observation.images.camera_left` | video | [480, 640, 3] | 左相机图像 |
| `observation.images.camera_right` | video | [480, 640, 3] | 右相机图像 |
| `observation.images.camera_front` | video | [480, 640, 3] | 前相机图像 |
| `timestamp` | float32 | [1] | 时间戳 (秒) |
| `frame_index` | int64 | [1] | 帧索引 (0-239) |
| `episode_index` | int64 | [1] | Episode 索引 |
| `index` | int64 | [1] | 全局帧索引 |
| `task_index` | int64 | [1] | 任务索引 |

### 4.2 observation.state 详细定义 (26维)

```python
observation.state = [
    # === 左臂关节角度 (索引 0-6) ===
    "left_shoulder_pan.pos",    # [0]  左肩平移关节
    "left_shoulder_lift.pos",   # [1]  左肩抬升关节
    "left_shoulder_roll.pos",   # [2]  左肩旋转关节
    "left_elbow.pos",           # [3]  左肘关节
    "left_wrist_pitch.pos",     # [4]  左腕俯仰关节
    "left_wrist_roll.pos",      # [5]  左腕旋转关节
    "left_gripper.pos",         # [6]  左夹爪开合

    # === 右臂关节角度 (索引 7-13) ===
    "right_shoulder_pan.pos",   # [7]  右肩平移关节
    "right_shoulder_lift.pos",  # [8]  右肩抬升关节
    "right_shoulder_roll.pos",  # [9]  右肩旋转关节
    "right_elbow.pos",          # [10] 右肘关节
    "right_wrist_pitch.pos",    # [11] 右腕俯仰关节
    "right_wrist_roll.pos",     # [12] 右腕旋转关节
    "right_gripper.pos",        # [13] 右夹爪开合

    # === 左臂末端位姿 (索引 14-19) ===
    "left_ee.x",                # [14] 左末端 X 坐标 (米)
    "left_ee.y",                # [15] 左末端 Y 坐标 (米)
    "left_ee.z",                # [16] 左末端 Z 坐标 (米)
    "left_ee.roll",             # [17] 左末端 Roll 角 (弧度)
    "left_ee.pitch",            # [18] 左末端 Pitch 角 (弧度)
    "left_ee.yaw",              # [19] 左末端 Yaw 角 (弧度)

    # === 右臂末端位姿 (索引 20-25) ===
    "right_ee.x",               # [20] 右末端 X 坐标 (米)
    "right_ee.y",               # [21] 右末端 Y 坐标 (米)
    "right_ee.z",               # [22] 右末端 Z 坐标 (米)
    "right_ee.roll",            # [23] 右末端 Roll 角 (弧度)
    "right_ee.pitch",           # [24] 右末端 Pitch 角 (弧度)
    "right_ee.yaw",             # [25] 右末端 Yaw 角 (弧度)
]
```

### 4.3 action 详细定义 (14维)

```python
action = [
    # === 左臂关节角度目标 (索引 0-6) ===
    "left_shoulder_pan.pos",    # [0]
    "left_shoulder_lift.pos",   # [1]
    "left_shoulder_roll.pos",   # [2]
    "left_elbow.pos",           # [3]
    "left_wrist_pitch.pos",     # [4]
    "left_wrist_roll.pos",      # [5]
    "left_gripper.pos",         # [6]

    # === 右臂关节角度目标 (索引 7-13) ===
    "right_shoulder_pan.pos",   # [7]
    "right_shoulder_lift.pos",  # [8]
    "right_shoulder_roll.pos",  # [9]
    "right_elbow.pos",          # [10]
    "right_wrist_pitch.pos",    # [11]
    "right_wrist_roll.pos",     # [12]
    "right_gripper.pos",        # [13]
]
```

### 4.4 视频特征

三个相机的视频配置相同：

```json
{
    "dtype": "video",
    "shape": [480, 640, 3],
    "names": ["height", "width", "channels"],
    "info": {
        "video.height": 480,
        "video.width": 640,
        "video.codec": "av1",
        "video.pix_fmt": "yuv420p",
        "video.fps": 30,
        "video.channels": 3,
        "has_audio": false
    }
}
```

| 相机 | ROS Topic | 位置说明 |
|------|-----------|---------|
| camera_left | /camera_l/color/image_raw | 左侧视角 |
| camera_right | /camera_r/color/image_raw | 右侧视角 |
| camera_front | /camera_f/color/image_raw | 正前方视角 |

---

## 5. 末端位姿计算原理

### 5.1 Pinocchio 正向运动学

末端位姿通过 **Pinocchio** 库的正向运动学 (Forward Kinematics, FK) 计算得到：

```python
import pinocchio as pin

# 加载 URDF 模型
model = pin.buildModelFromUrdf(urdf_path, package_dirs)
data = model.createData()

# 锁定夹爪关节 (joint7, joint8)
# 这些关节不参与末端位姿计算

# 正向运动学计算
pin.forwardKinematics(model, data, joint_positions)
pin.updateFramePlacements(model, data)

# 获取末端执行器位姿
ee_frame_id = model.getFrameId("end_effector_frame")
ee_pose = data.oMf[ee_frame_id]  # SE3 变换矩阵
```

### 5.2 坐标系定义

**末端位姿坐标系：**
- **原点**: 末端执行器参考点（通常在夹爪中心）
- **X 轴**: 指向前方（抓取方向）
- **Y 轴**: 指向左侧
- **Z 轴**: 指向上方

**旋转补偿：**
录制时应用了旋转补偿 `ee_rotation_rpy=[0.0, -1.57, -1.57]`，用于将 URDF 中的末端坐标系对齐到期望的工具坐标系。

### 5.3 欧拉角约定

末端姿态使用 **RPY (Roll-Pitch-Yaw)** 欧拉角表示：

| 角度 | 旋转轴 | 范围 | 说明 |
|------|--------|------|------|
| Roll | X 轴 | [-π, π] | 绕前进方向旋转 |
| Pitch | Y 轴 | [-π/2, π/2] | 俯仰角 |
| Yaw | Z 轴 | [-π, π] | 偏航角 |

**旋转顺序**: ZYX (先 Yaw，再 Pitch，最后 Roll)

### 5.4 URDF 模型配置

```bash
# URDF 路径
urdf_path="/home/agilex/cqy/my_lerobot/my_sim/piper_ros-noetic/src/piper_description/urdf/piper_description.urdf"

# 包目录（用于解析 mesh 文件）
package_dirs='["my_sim/piper_ros-noetic/src"]'

# 锁定的关节（夹爪关节不参与 FK 计算）
lock_joints='["joint7", "joint8"]'
```

---

## 6. 数据文件格式

### 6.1 data/*.parquet 结构

每个 parquet 文件包含以下列：

| 列名 | 类型 | 说明 |
|------|------|------|
| `action` | list[float32] | 14维动作向量 |
| `observation.state` | list[float32] | 26维状态向量 |
| `timestamp` | float32 | 时间戳 (秒) |
| `frame_index` | int64 | Episode 内帧索引 |
| `episode_index` | int64 | Episode 索引 |
| `index` | int64 | 全局帧索引 |
| `task_index` | int64 | 任务索引 |

### 6.2 数据示例

```python
import pandas as pd

# 读取数据
df = pd.read_parquet("data/chunk-000/file-000.parquet")

# 查看第一帧
row = df.iloc[0]

# observation.state (26维)
state = row['observation.state']
print(f"关节角度 (左臂): {state[0:7]}")
print(f"关节角度 (右臂): {state[7:14]}")
print(f"末端位姿 (左臂): {state[14:20]}")
print(f"末端位姿 (右臂): {state[20:26]}")

# action (14维)
action = row['action']
print(f"动作 (左臂): {action[0:7]}")
print(f"动作 (右臂): {action[7:14]}")
```

### 6.3 tasks.parquet 结构

```python
# 任务描述
task_index | task_description
-----------+--------------------------------------------------
0          | "Grasping an apple from a table filled with various cluttered objects."
```

---

## 7. 统计信息

### 7.1 observation.state 统计

末端位姿的典型数值范围（基于实际数据）：

| 特征 | 最小值 | 最大值 | 均值 | 标准差 |
|------|--------|--------|------|--------|
| left_ee.x | 0.050 | 0.476 | 0.246 | 0.078 |
| left_ee.y | -0.318 | 0.088 | -0.054 | 0.090 |
| left_ee.z | 0.026 | 0.354 | 0.147 | 0.047 |
| left_ee.roll | -2.104 | -0.955 | -1.583 | 0.126 |
| left_ee.pitch | -0.318 | 0.850 | 0.279 | 0.148 |
| left_ee.yaw | -1.522 | 0.655 | -0.220 | 0.369 |
| right_ee.x | 0.051 | 0.182 | 0.112 | 0.057 |
| right_ee.y | -0.026 | -0.001 | -0.014 | 0.011 |
| right_ee.z | 0.149 | 0.230 | 0.174 | 0.021 |
| right_ee.roll | -1.722 | -1.482 | -1.669 | 0.065 |
| right_ee.pitch | -0.148 | 0.330 | 0.155 | 0.061 |
| right_ee.yaw | -0.144 | -0.021 | -0.086 | 0.029 |

### 7.2 统计信息用途

统计信息用于训练时的数据归一化：

```python
# 归一化公式
normalized_state = (state - mean) / (std + 1e-6)

# 或使用分位数归一化
normalized_state = (state - q01) / (q99 - q01 + 1e-6)
```

---

## 8. 数据加载与使用

### 8.1 使用 LeRobotDataset 加载

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 加载数据集
dataset = LeRobotDataset(
    repo_id="cqy/agilex_vla_demo_ee_pinocchio_apple_grasp",
    root="~/.cache/huggingface/lerobot"
)

# 获取单帧数据
frame = dataset[0]
print(frame.keys())
# dict_keys(['observation.state', 'action', 'observation.images.camera_left', ...])

# 提取末端位姿
state = frame['observation.state']
left_ee_pose = state[14:20]   # [x, y, z, roll, pitch, yaw]
right_ee_pose = state[20:26]  # [x, y, z, roll, pitch, yaw]
```

### 8.2 使用 delta_timestamps 加载多帧

```python
dataset = LeRobotDataset(
    repo_id="cqy/agilex_vla_demo_ee_pinocchio_apple_grasp",
    delta_timestamps={
        "observation.state": [-0.1, 0.0],  # 历史帧 + 当前帧
        "action": [t / 30 for t in range(16)]  # 未来 16 帧动作
    }
)

frame = dataset[100]
# observation.state: shape (2, 26) - 2个时间点，每个26维
# action: shape (16, 14) - 16个时间点，每个14维
```

### 8.3 提取末端位姿的辅助函数

```python
def extract_ee_pose(observation_state):
    """
    从 observation.state 中提取末端位姿

    Args:
        observation_state: shape (..., 26) 的张量

    Returns:
        left_ee_pose: shape (..., 6) - [x, y, z, roll, pitch, yaw]
        right_ee_pose: shape (..., 6) - [x, y, z, roll, pitch, yaw]
    """
    left_ee_pose = observation_state[..., 14:20]
    right_ee_pose = observation_state[..., 20:26]
    return left_ee_pose, right_ee_pose

def extract_joint_positions(observation_state):
    """
    从 observation.state 中提取关节角度

    Args:
        observation_state: shape (..., 26) 的张量

    Returns:
        left_joints: shape (..., 7)
        right_joints: shape (..., 7)
    """
    left_joints = observation_state[..., 0:7]
    right_joints = observation_state[..., 7:14]
    return left_joints, right_joints
```

---

## 9. 训练注意事项

### 9.1 特征维度变化

使用末端位姿数据集训练时，需要注意 `observation.state` 维度从 14 变为 26：

```python
# 模型配置需要更新
config = {
    "observation_state_dim": 26,  # 不是 14
    "action_dim": 14,             # 保持不变
}
```

### 9.2 归一化策略

建议对不同类型的数据使用不同的归一化策略：

```python
# 关节角度: 使用 mean/std 归一化
joint_normalized = (joints - stats['mean'][:14]) / (stats['std'][:14] + 1e-6)

# 末端位置 (x, y, z): 使用 min/max 归一化到 [-1, 1]
pos_normalized = 2 * (pos - stats['min']) / (stats['max'] - stats['min'] + 1e-6) - 1

# 末端姿态 (roll, pitch, yaw): 使用 mean/std 归一化
rot_normalized = (rot - stats['mean']) / (stats['std'] + 1e-6)
```

### 9.3 末端位姿的使用场景

末端位姿信息可用于：

1. **任务空间控制**: 直接预测末端位姿变化
2. **辅助特征**: 作为额外输入提升模型性能
3. **奖励计算**: 计算末端与目标的距离
4. **可视化**: 在 3D 空间中可视化轨迹

---

## 10. 常见问题

### 10.1 为什么 action 不包含末端位姿？

action 仅包含关节角度（14维），因为：
- 机器人底层控制基于关节空间
- 末端位姿是通过正向运动学从关节角度计算得到的
- 如需末端空间控制，需要额外实现逆运动学

### 10.2 如何从末端位姿恢复关节角度？

需要使用逆运动学 (IK)：

```python
import pinocchio as pin

# 使用 Pinocchio 的 IK 求解器
q_solution = pin.computeFrameJacobian(...)
# 或使用其他 IK 库
```

### 10.3 数据集兼容性

- 本数据集 **不兼容** 标准 14 维数据集的模型
- 需要修改模型输入维度或提取部分特征
- 不要使用 `--resume=true` 在旧数据集上继续录制

### 10.4 末端位姿精度

末端位姿精度取决于：
- URDF 模型的准确性
- 关节编码器精度
- 运动学标定质量

---

## 附录

### A. info.json 完整内容

```json
{
    "codebase_version": "v3.0",
    "robot_type": "agilex",
    "total_episodes": 210,
    "total_frames": 50400,
    "total_tasks": 1,
    "fps": 30,
    "chunks_size": 1000,
    "data_files_size_in_mb": 100,
    "video_files_size_in_mb": 200,
    "splits": {"train": "0:210"},
    "features": {
        "observation.state": {
            "dtype": "float32",
            "shape": [26],
            "names": [
                "left_shoulder_pan.pos", "left_shoulder_lift.pos",
                "left_shoulder_roll.pos", "left_elbow.pos",
                "left_wrist_pitch.pos", "left_wrist_roll.pos",
                "left_gripper.pos",
                "right_shoulder_pan.pos", "right_shoulder_lift.pos",
                "right_shoulder_roll.pos", "right_elbow.pos",
                "right_wrist_pitch.pos", "right_wrist_roll.pos",
                "right_gripper.pos",
                "left_ee.x", "left_ee.y", "left_ee.z",
                "left_ee.roll", "left_ee.pitch", "left_ee.yaw",
                "right_ee.x", "right_ee.y", "right_ee.z",
                "right_ee.roll", "right_ee.pitch", "right_ee.yaw"
            ]
        },
        "action": {
            "dtype": "float32",
            "shape": [14],
            "names": [...]
        }
    }
}
```

### B. 录制命令参考

```bash
#!/usr/bin/env bash
lerobot-record \
    --robot.type=agilex \
    --robot.record_ee_pose=true \
    --robot.kinematics_backend=pinocchio \
    --robot.kinematics_urdf_path="<URDF_PATH>" \
    --robot.pinocchio_package_dirs='["<PACKAGE_DIR>"]' \
    --robot.pinocchio_lock_joints='["joint7", "joint8"]' \
    --robot.ee_rotation_rpy='[0.0, -1.57, -1.57]' \
    --dataset.repo_id="<YOUR_REPO_ID>" \
    --dataset.single_task="<TASK_DESCRIPTION>" \
    --dataset.fps=30
```

### C. 索引快速参考

| 索引范围 | 内容 | 维度 |
|---------|------|------|
| state[0:7] | 左臂关节角度 | 7 |
| state[7:14] | 右臂关节角度 | 7 |
| state[14:17] | 左末端位置 (x,y,z) | 3 |
| state[17:20] | 左末端姿态 (r,p,y) | 3 |
| state[20:23] | 右末端位置 (x,y,z) | 3 |
| state[23:26] | 右末端姿态 (r,p,y) | 3 |

---

**文档版本**: v1.0
**最后更新**: 2025-01
**适用数据集**: agilex_vla_demo_ee_pinocchio_apple_grasp
**LeRobot 版本**: v3.0+

