# 末端 LeRobot 数据解析 2.0 (AgileX + Pinocchio EE Pose)

本文档详细解析由 `record_ee_pinocchio.sh` 脚本录制的、包含末端位姿 (End-Effector Pose) 的 LeRobot 数据集格式。

## 1. 概述

该数据集在标准 LeRobot 格式 (v2.1/v3.0) 的基础上，增强了 `observation.state` 特征。除了包含机器人的关节角度外，还通过 Pinocchio 动力学库的运动学解算 (FK)，实时记录了双臂末端执行器的位姿 (位置 + 姿态)。

**核心差异:**
*   **普通数据**: `observation.state` 仅包含 14 个关节角度。
*   **本数据**: `observation.state` 包含 14 个关节角度 + 12 个末端位姿分量 (共 26 维)。

## 2. 数据结构与特征

数据集遵循 LeRobot 标准目录结构 (`meta/`, `data/`, `videos/`)。关键的特征定义如下：

### 2.1 图像特征 (`observation.images`)
包含来自各个相机的视频帧或图像。
*   **Key**: `observation.images.camera_left`, `observation.images.camera_right`, `observation.images.camera_front` 等。
*   **Shape**: `(H, W, 3)` (HWC 格式) 或 `(3, H, W)` (PyTorch CHW 格式)。

### 2.2 动作特征 (`action`)
记录了发送给机器人的控制指令。
*   **Key**: `action`
*   **Shape**: `(14,)`
*   **内容**: 双臂 14 个自由度的目标关节角度 (弧度)。
*   **顺序**: 左臂 7 个关节 + 右臂 7 个关节。

### 2.3 状态特征 (`observation.state`) - **重点**
记录了机器人的实时观测状态。

*   **Key**: `observation.state`
*   **Dtype**: `float32`
*   **Shape**: `(26,)` (14 关节 + 12 EE Pose)
*   **组成**:
    该向量由 **关节角度** 和 **末端位姿** 拼接而成。

#### 详细索引分布

| 索引范围 | 描述 | 包含变量名 (示例) | 单位 |
| :--- | :--- | :--- | :--- |
| **0 - 13** | **关节角度 (Joint Positions)** | `left_shoulder_pan.pos` ... `right_gripper.pos` | 弧度 (rad) / 米 (m) |
| **14 - 19** | **左臂末端位姿 (Left EE Pose)** | `left_ee.x`, `left_ee.y`, `left_ee.z`, `left_ee.roll`, `left_ee.pitch`, `left_ee.yaw` | 米 (m), 弧度 (rad) |
| **20 - 25** | **右臂末端位姿 (Right EE Pose)** | `right_ee.x`, `right_ee.y`, `right_ee.z`, `right_ee.roll`, `right_ee.pitch`, `right_ee.yaw` | 米 (m), 弧度 (rad) |

> **注意**: 具体的索引顺序完全取决于 `meta/info.json` 中 `observation.state.names` 的列表顺序。虽然通常是上述顺序，但建议编写代码时总是通过 `names` 查找索引。

#### 2.3.1 关节名称列表 (Standard AgileX Piper)
1.  `shoulder_pan`
2.  `shoulder_lift`
3.  `shoulder_roll`
4.  `elbow`
5.  `wrist_pitch`
6.  `wrist_roll`
7.  `gripper`

#### 2.3.2 末端位姿格式 (Pinocchio Backend)
使用 `record_ee_pinocchio.sh` 录制的数据采用 **XYZ + RPY (Roll-Pitch-Yaw)** 格式。

*   **位置 (Translation)**: `x`, `y`, `z` (相对于机器人基座标系)
*   **姿态 (Rotation)**: `roll` (翻滚), `pitch` (俯仰), `yaw` (偏航) (欧拉角，固有旋转或外旋取决于后端实现，Pinocchio 通常输出标准欧拉角)

> **对比说明**: 如果使用的是 `placo` 后端 (旧版)，姿态部分是 `wx, wy, wz` (旋转向量/轴角)。请务必确认 `names` 中是 `roll/pitch/yaw` 还是 `wx/wy/wz`。

## 3. 如何解析数据 (Python 示例)

读取数据时，可以通过 Dataset 对象的元数据自动解析，无需硬编码索引。

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torch

# 1. 加载数据集
dataset = LeRobotDataset("user/your_dataset_repo_id")

# 2. 获取特征名称列表
state_names = dataset.meta.features["observation.state"]["names"]
print("State vector layout:", state_names)

# 3. 读取一帧数据
frame = dataset[0]
state_vector = frame["observation.state"]  # Tensor shape (26,)

# 4. 解析关节角度和末端位姿
# 方法 A: 智能查找 (推荐)
def get_value(name, vector, names_list):
    idx = names_list.index(name)
    return vector[idx]

left_ee_x = get_value("left_ee.x", state_vector, state_names)
left_joint_1 = get_value("left_shoulder_pan.pos", state_vector, state_names)

print(f"Left EE X: {left_ee_x}")

# 方法 B: 切片 (如果确认顺序固定)
# 假设前14个是关节，中间6个是左臂EE，最后6个是右臂EE
joints = state_vector[:14]
left_ee_pose = state_vector[14:20]  # [x, y, z, r, p, y]
right_ee_pose = state_vector[20:26] # [x, y, z, r, p, y]

print(f"Left EE Pose: {left_ee_pose}")
```

## 4. 坐标系说明

*   **基座标系**: 通常定义在机器人底座中心。`x` 前, `y` 左, `z` 上 (需参考具体 URDF)。
*   **末端坐标系**: 定义在 `gripper_base` 或指尖中心。
    *   `record_ee_pinocchio.sh` 中可以通过 `--robot.ee_rotation_rpy` 和 `--robot.ee_translation` 参数对默认的末端坐标系进行偏移修正。
    *   录制时使用的参数:
        *   Rotation Offset: `[0.0, -1.57, -1.57]`
        *   Translation Offset: `[0.0, 0.0, 0.0]`

## 5. 常见问题

**Q1: 为什么我看不到 `observation.ee_pose` 这个 Key?**
A: LeRobot 数据集为了保持通用性，会将所有连续的浮点型观测值合并到 `observation.state` 中。你需要从这个大向量中根据 `names` 切分出你需要的数据。

**Q2: 如何区分是 Pinocchio (RPY) 还是 Placo (RotVec) 格式?**
A: 检查 `meta/info.json` 中的 `names` 字段。
*   出现 `roll`, `pitch`, `yaw` -> **Pinocchio (RPY)**
*   出现 `wx`, `wy`, `wz` -> **Placo (Rotation Vector)**

**Q3: 训练时如何只使用关节角度，忽略 EE Pose?**
A: 在训练配置文件或代码中，通过索引切片只取前 14 维数据作为策略网络的输入；或者在数据处理 pipeline 中根据 name 过滤掉 `ee` 相关的特征。
