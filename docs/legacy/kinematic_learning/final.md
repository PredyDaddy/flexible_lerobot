# 运控从零学习（Final）：本仓库 Pinocchio FK/IK 录制与回放

> **兼容性说明 (lerobot 0.4.3 baseline)**  
> 本文对应的实现存在于旧仓库；当前仓库的 0.4.3 baseline 默认代码树不包含 `record_ee_pinocchio.sh`、`replay_ee_pinocchio.sh`、`my_ik/`、`src/lerobot/scripts/lerobot_replay_ee.py`、`src/lerobot/robots/agilex/` 等内容，因此无法直接按本文复现。  
> 建议仅作为历史记录/移植参考（路线 B），或改写为 0.4.3 现有的 `lerobot-replay` 工作流。  

本文是对 `docs/legacy/kinematic_learning` 中多份文档的整合版。目标读者是零基础运控同学：不要求推导公式，但能看懂“录 EE → 回放 EE”的完整链路，并知道如何迁移到新机器人。

本仓库涉及的核心文件如下（请对照阅读）：

- 录制脚本：`record_ee_pinocchio.sh`
- 回放脚本：`replay_ee_pinocchio.sh`
- 录制端 FK 写入：`src/lerobot/robots/agilex/agilex.py`
- 回放端 IK 解算：`src/lerobot/scripts/lerobot_replay_ee.py`
- Pinocchio FK/IK 实现：`my_ik/piper_pinocchio_kinematics.py`
- 机器人配置：`src/lerobot/robots/agilex/config_agilex.py`

---

## 1. 一句话理解：本方案在干什么？

**录制**：每帧读取关节角 `q` → 用 FK 算 EE 位姿 `T(q)` → 写入数据集 `observation.state` 的 `left_ee.* / right_ee.*`。  
**回放**：每帧读取数据集 EE 位姿 `T*` → 用 IK 算关节角 `q_hat` → 下发到机器人。

注意：**数据集的 `action` 仍是关节空间，不是末端空间**。EE 位姿只体现在 `observation.state`。

---

## 2. 坐标系与姿态表示（本仓库约定）

- EE 位姿格式（Pinocchio 后端）：`[x, y, z, roll, pitch, yaw]`（单位：m + rad）
- 旋转约定：`R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`
- 若用 placo 后端：姿态是 rotvec（`wx, wy, wz`），格式不同

这两个格式在 `lerobot_replay_ee.py` 中会被自动识别：

- rpy：`EE_KEYS_RPY = ["x", "y", "z", "roll", "pitch", "yaw"]`
- rotvec：`EE_KEYS_ROTVEC = ["x", "y", "z", "wx", "wy", "wz"]`

---

## 3. 数据结构：observation/state 与 action

以 Agilex 双臂为例（每臂 6 关节 + 1 gripper）：

- `observation.state`：  
  - 14 个关节位置（`left_*` + `right_*`）  
  - 若 `record_ee_pose=true`，追加 12 个 EE 位姿字段（两臂各 6）
- `action`：  
  - 仍是关节角控制（包含 gripper）

**IK 只控制前 6 个关节，gripper 直接照录制值回放**：  
`ARM_JOINT_NAMES = JOINT_NAMES[:6]`（见 `lerobot_replay_ee.py`）

---

## 4. 录制端（FK）：代码到底怎么做？

核心位置：`AgileXRobot.get_observation()`（`src/lerobot/robots/agilex/agilex.py`）

逻辑简化如下：

```text
读取 ROS 关节角 q_ros (前 6 关节)
  → q_urdf = q_ros * signs + offsets
  → Pinocchio FK: xyzrpy = FK(q_urdf)
  → (可选) TCP offset: T = FK_matrix(q) @ T_tcp
  → 写入 left_ee.* / right_ee.*
```

关键点：

- **只取前 6 个关节参与 FK**（gripper 不参与）
  - `q_left_ros = left_state.position[:6]`
- **关节映射是必须的**（ROS ↔ URDF）
  - `q_urdf = q_ros * signs + offsets`
  - 来自配置：`left_joint_signs/offsets`（长度 6）
- **TCP 叠加**（仅录制端）
  - `left_tcp_offset_xyzrpy` / `right_tcp_offset_xyzrpy`
  - 对 pinocchio：在 FK 后矩阵相乘并回写 rpy

注意：`record_ee_pinocchio.sh` 已显式开启：

```bash
--robot.record_ee_pose=true
--robot.kinematics_backend=pinocchio
```

---

## 5. 回放端（IK）：代码怎么解算并下发？

核心位置：`lerobot_replay_ee.py`（`src/lerobot/scripts/lerobot_replay_ee.py`）

关键流程：

1. 检测数据集 EE 格式（rpy / rotvec）
2. 初始化 IK（pinocchio 或 placo）
3. 用当前机器人姿态做 IK warm start
4. 每帧：
   - 读 `left_ee.* / right_ee.*`
   - IK 得到 `q_urdf`
   - 映射回 ROS：`q_ros = (q_urdf - offsets) / signs`
   - 下发前 6 关节 + gripper

Pinocchio IK 的调用形态（简化）：

```python
xyzrpy = np.concatenate([pos, rot])
q_urdf_rad, success = ik.inverse_kinematics(xyzrpy, q_urdf_rad)
q_ros = (q_urdf_rad - offsets) / signs
```

安全相关开关（建议优先使用）：

- `--dry_run=true`：只解算不下发
- `--ik.arms=left/right`：只控制单臂
- `--dataset.fps=10/15`：放慢回放速度（脚本不会降采样，只会睡眠更久）
- `--preflight=true`：运行前粗略检查 EE 速度

---

## 6. Pinocchio FK/IK 的实现细节（当前代码）

文件：`my_ik/piper_pinocchio_kinematics.py`

### 6.1 FK（PiperPinocchioFK）

- 加载 URDF
  - `pin.RobotWrapper.BuildFromURDF(urdf_path, package_dirs)`
- 锁定夹爪关节
  - `lock_joints=["joint7", "joint8"]`（默认）
- 添加 EE frame（**挂在 joint6 上**）
  - `getJointId("joint6")`（注意写死）
  - `ee_rotation_rpy` / `ee_translation`
- FK 计算：
  - `pin.forwardKinematics + pin.updateFramePlacements`
  - 输出 `xyzrpy`

### 6.2 IK（PiperPinocchioIK）

核心思路：把 IK 写成带关节限位的优化问题（Casadi + IPOPT）：

```
min  pose_weight * ||log6(T_current^-1 * T_target)||^2
   + regularization_weight * ||q||^2
s.t. q_min <= q <= q_max
```

工程增强：

- Warm start：用上一帧解作为初值
- 跳变检测：若关节突变 > 30°，重置初值
- 失败时返回历史解并标记 `success=False`

依赖要求：

- Pinocchio + Casadi（且 pinocchio 需包含 casadi 支持）

---

## 7. 配置与参数速查（与当前实现一致）

配置文件：`src/lerobot/robots/agilex/config_agilex.py`

### 7.1 与 FK/IK 直接相关的配置

- `record_ee_pose`: 是否写 EE 到 observation
- `kinematics_backend`: `"placo"` or `"pinocchio"`
- `kinematics_urdf_path`: URDF 路径（建议绝对或相对 repo root）
- `pinocchio_package_dirs`: 解析 `package://` 用
- `pinocchio_lock_joints`: 夹爪关节锁定
- `ee_rotation_rpy` / `ee_translation`: EE frame 定义
- `left/right_joint_signs` + `left/right_joint_offsets_rad`: ROS ↔ URDF 映射（长度 6）
- `left/right_tcp_offset_xyzrpy`: 录制端 TCP 叠加

### 7.2 IK 参数（pinocchio）

来自 `lerobot_replay_ee.py` 的 `IKConfig`：

- `--ik.max_iter`
- `--ik.tol`
- `--ik.pose_weight`
- `--ik.regularization_weight`

### 7.3 安全参数

- `max_relative_target`: 单步关节变化限幅（发送前裁剪）
- `enforce_joint_limits`: 强制关节限位
- `joint_limit_margin_rad`: 限位安全边界

---

## 8. 重要注意事项（当前实现的真实行为）

1. **本仓库 pinocchio 集成目前只按 6 轴臂做 IK/FK**
   - `q_left_ros = left_state.position[:6]`
   - `ARM_JOINT_NAMES = JOINT_NAMES[:6]`
   - gripper 单独回放

2. **`ee` frame 挂在 `joint6` 上是写死的**
   - `getJointId("joint6")` 在 `PiperPinocchioFK/IK` 中硬编码

3. **Pinocchio 回放端不会应用 TCP offset**
   - 录制端会把 `left/right_tcp_offset_xyzrpy` 叠加进 EE
   - 回放端 pinocchio 路径不会反解 TCP offset  
   - 结论：若使用 pinocchio，请**把 TCP 偏移放进 `ee_translation/rotation`**，并保持 `left/right_tcp_offset_xyzrpy` 为 0（除非你改代码）

---

## 9. 常见误差来源与排查顺序

优先级从高到低：

1. **关节映射错误（signs / offsets）**
2. **URDF 与真实尺寸不一致**
3. **EE/TCP 定义不正确**
4. **IK 数值误差（迭代次数不足/权重不当）**
5. **执行误差（回放频率过高/电机跟踪不足）**

推荐排查顺序（不需要上机）：

1. 用同一 URDF 做 `FK → IK → FK` 闭环
2. 检查输出位姿误差是否很小
3. 再接硬件做 small range 回放

---

## 10. 迁移到新机器人（尤其 7 轴）

如果换机器人，以下部分必须改：

- **URDF 与 package_dirs**
- **lock_joints**
- **ee_rotation_rpy / ee_translation**
- **signs / offsets（长度必须匹配）**

如果改成 7 轴，需要动的代码点（当前实现写死了 6）：

- `src/lerobot/robots/agilex/agilex.py`
  - `position[:6]`、`_validate_len(... expected_len=6)`
- `src/lerobot/scripts/lerobot_replay_ee.py`
  - `ARM_JOINT_NAMES = JOINT_NAMES[:6]`
- `my_ik/piper_pinocchio_kinematics.py`
  - `getJointId("joint6")`（EE frame 依赖最后一轴）
- `config_agilex.py`
  - `kinematics_joint_names`、signs/offsets、关节限位

7 轴是冗余臂：IK 解会有无穷多组。  
当前实现仅用 `||q||^2` 做正则，会偏好“关节角更小”的解。如果你发现肘部抖动或解支跳变，建议在
`my_ik/piper_pinocchio_kinematics.py` 的代价函数中加入 **q_nominal 或 q_prev 偏好项**，
并适当提高 `regularization_weight`。

---

## 11. 快速上手与安全建议

### 11.1 录制（Pinocchio FK）

```bash
./record_ee_pinocchio.sh
```

注意事项：

- 脚本里 `--robot.record_ee_pose=true` 会改变 `observation.state` 结构  
  **不要**在旧数据集上直接 `--resume=true` 叠加（除非 schema 一致）
- 确认 `--robot.kinematics_urdf_path` 和 `--robot.pinocchio_package_dirs` 指向正确 URDF

### 11.2 回放（Pinocchio IK）

建议先安全验证：

```bash
./replay_ee_pinocchio.sh --dry_run=true --ik.arms=right --dataset.fps=10
```

确认 IK 正常后再逐步打开：

1. `--dry_run=false`
2. `--ik.arms=both`
3. `--dataset.fps` 提升到录制 fps

### 11.3 推荐的离线一致性检查

在 Python 中做闭环验证（不接硬件）：

```python
import numpy as np
from my_ik.piper_pinocchio_kinematics import create_piper_fk, create_piper_ik

fk = create_piper_fk()
ik = create_piper_ik()

q = np.array([0.1, 0.2, -0.3, 0.1, -0.2, 0.3])
xyzrpy = fk.forward_kinematics(q)
q_hat, ok = ik.inverse_kinematics(xyzrpy, q_init=q)
xyzrpy_hat = fk.forward_kinematics(q_hat)
print("ok=", ok, "err=", np.linalg.norm(xyzrpy - xyzrpy_hat))
```

---

## 12. 结语

本仓库已经提供了完整的 **“FK 录 EE / IK 回放 EE”** 端到端链路。只要保证：

- URDF 与真实机器人一致
- signs/offsets 正确
- EE/TCP 定义清晰

你就可以稳定复用这套方案。  
后续如需支持 7 轴或更多 DOF，建议先“让代码可配置化”，再逐步引入冗余臂的选解策略。  
