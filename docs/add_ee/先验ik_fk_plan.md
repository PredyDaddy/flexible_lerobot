# 先验 IK/FK 开发计划

> 基于 `my_ik/fk_demo.py` 和 `my_ik/ik_demo.py` 的 pinocchio + casadi 方案

---

## 1. 项目背景

### 1.1 为什么需要这个方案
- 现有的 placo 方案无法满足需求
- 同事提供了基于 pinocchio + casadi 的 FK/IK 实现
- 需要将其集成到 LeRobot 的录制/回放流程中

### 1.2 目标
1. 使用 pinocchio FK 录制末端位姿（EE pose）到 dataset
2. 使用 pinocchio + casadi IK 从 dataset 读取 EE pose 并回放
3. 实现 FK → IK 闭环

---

## 2. 技术决策（已确认）

| 决策项 | 选择 | 说明 |
|--------|------|------|
| FK/IK 后端 | pinocchio + casadi | 替代现有 placo 方案 |
| EE 姿态格式 | rpy (roll, pitch, yaw) | 与 fk_demo.py / ik_demo.py 一致 |
| EE 定义 | 以 ik_demo.py 为准 | 旋转 `(0, -1.57, -1.57)`，无平移 |
| EE 位置 | joint6 坐标系 | 暂不加 TCP 偏移，后续可通过配置添加 |
| 关节映射 | 默认值 | signs=[1,1,1,1,1,1], offsets=[0,0,0,0,0,0] |
| Gripper | 直接透传 | 不经过 IK 处理 |

### 2.1 EE Frame 定义（关键）

```python
# 以 ik_demo.py 为准
ee_rotation_rpy = [0, -1.57, -1.57]  # (roll, pitch, yaw) in radians
ee_translation = [0, 0, 0]           # 无平移

# 等效的 quaternion（ik_demo.py 中的写法）
q = quaternion_from_euler(0, -1.57, -1.57)
```

### 2.2 Dataset 中的 EE 字段

录制时写入以下 observation 字段（标量 float）：
- `left_ee.x`, `left_ee.y`, `left_ee.z`
- `left_ee.roll`, `left_ee.pitch`, `left_ee.yaw`
- `right_ee.x`, `right_ee.y`, `right_ee.z`
- `right_ee.roll`, `right_ee.pitch`, `right_ee.yaw`

---

## 3. 依赖项

### 3.1 需要安装的 Python 包

```bash
# 在 my_lerobot 环境中执行
conda activate my_lerobot

# casadi（IK 优化求解器）
pip install casadi

# tf_transformations（欧拉角/四元数转换）
pip install transforms3d
# 或
pip install tf-transformations

# pinocchio with casadi support
# 需要确认当前版本是否支持，如不支持需重新安装
# 方式1: conda
conda install -c conda-forge pinocchio
# 方式2: 从源码编译（如果需要 casadi 支持）
```

### 3.2 已有依赖（已安装）

- pinocchio 3.8.0 ✓
- placo ✓（保留，不影响新方案）
- numpy ✓

### 3.3 需要验证的依赖

```python
# 验证 pinocchio.casadi 是否可用
from pinocchio import casadi as cpin  # 如果报错需要重新安装
```

---

## 4. 文件结构

### 4.1 新增文件

```
my_ik/
├── docs/
│   ├── 先验_ikfk.md              # 原始方案文档
│   └── 先验ik_fk_plan.md         # 本开发计划
├── fk_demo.py                     # 同事提供的 FK 参考（ROS2）
├── ik_demo.py                     # 同事提供的 IK 参考（ROS1）
└── piper_pinocchio_kinematics.py  # [新增] 纯计算模块

src/lerobot/
├── robots/agilex/
│   ├── config_agilex.py           # [修改] 新增 pinocchio 配置字段
│   ├── agilex.py                  # [修改] 集成 pinocchio FK
│   └── piper_pinocchio_fk.py      # [新增] FK 封装类
├── model/
│   └── piper_pinocchio_ik.py      # [新增] IK 封装类
└── scripts/
    ├── lerobot_replay_ee.py       # [修改] 使用 pinocchio IK
    └── visualize_replay_safety.py # [新增] 可视化安全验证工具
```

### 4.2 核心模块设计

#### piper_pinocchio_kinematics.py（纯计算模块）

```python
class PiperPinocchioFK:
    """Pinocchio FK 计算类（无 ROS 依赖）"""

    def __init__(
        self,
        urdf_path: str,
        package_dirs: list[str],
        lock_joints: list[str] = ["joint7", "joint8"],
        ee_rotation_rpy: list[float] = [0, -1.57, -1.57],
        ee_translation: list[float] = [0, 0, 0],
    ):
        ...

    def forward_kinematics(self, q_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        计算 FK

        Args:
            q_rad: 6 个关节角（radians）

        Returns:
            position: [x, y, z] (meters)
            orientation: [roll, pitch, yaw] (radians)
        """
        ...


class PiperPinocchioIK:
    """Pinocchio + Casadi IK 计算类（无 ROS 依赖）"""

    def __init__(
        self,
        urdf_path: str,
        package_dirs: list[str],
        lock_joints: list[str] = ["joint7", "joint8"],
        ee_rotation_rpy: list[float] = [0, -1.57, -1.57],
        ee_translation: list[float] = [0, 0, 0],
        solver_max_iter: int = 50,
        solver_tol: float = 1e-4,
    ):
        ...

    def inverse_kinematics(
        self,
        target_xyzrpy: np.ndarray,
        q_init: np.ndarray | None = None,
    ) -> tuple[np.ndarray, bool]:
        """
        计算 IK

        Args:
            target_xyzrpy: [x, y, z, roll, pitch, yaw]
            q_init: 初始关节角（用于迭代）

        Returns:
            q_solution: 6 个关节角（radians）
            success: 是否收敛
        """
        ...
```

---

## 5. 开发阶段

### Phase 1: FK 录制（只录制，不改回放）

#### 5.1.1 任务清单

- [ ] 创建 `PiperPinocchioFK` 类
  - 从 fk_demo.py 抽取核心逻辑
  - 使用 ik_demo.py 的 ee 定义
  - 删除 ROS2 依赖
- [ ] 修改 `config_agilex.py`
  - 新增 `kinematics_backend: str = "pinocchio"` 配置
  - 新增 `ee_rotation_rpy` 配置
- [ ] 修改 `agilex.py`
  - 当 `record_ee_pose=True` 且 `kinematics_backend="pinocchio"` 时使用新 FK
  - 输出 `left_ee.x/y/z/roll/pitch/yaw` 字段
- [ ] 创建 `record_ee_pinocchio.sh` 测试脚本

#### 5.1.2 验收标准

1. **FK 输出一致性测试**
   ```python
   # 给定相同的关节角 q
   # PiperPinocchioFK 输出的 ee pose 应与 ik_demo.py 中
   # 使用相同 ee 定义计算出的结果一致
   ```

2. **Dataset 字段检查**
   ```bash
   # 录制后检查 dataset 是否包含 EE 字段
   python -c "
   from lerobot.datasets.lerobot_dataset import LeRobotDataset
   ds = LeRobotDataset('cqy/agilex_vla_demo_ee_pinocchio')
   names = ds.features['observation.state']['names']
   print([n for n in names if 'ee.' in n])
   "
   # 应输出: ['left_ee.x', 'left_ee.y', ..., 'right_ee.yaw']
   ```

---

### Phase 1.5: 可视化安全验证工具

#### 5.2.1 任务清单

- [ ] 创建 `visualize_replay_safety.py`
  - 加载 dataset
  - 绿色影子：关节角直接 FK（Ground Truth）
  - 红色影子：EE pose → IK → FK（Solver Test）
  - 使用 meshcat 可视化

#### 5.2.2 验收标准

1. 红绿影子重合 → IK 正确
2. 红色影子抖动/偏离 → IK 有问题，不能上机

---

### Phase 2: IK 回放

#### 5.3.1 任务清单

- [ ] 创建 `PiperPinocchioIK` 类
  - 从 ik_demo.py 抽取核心逻辑
  - 删除 ROS、meshcat、piper_control 依赖
  - 保留 casadi + IPOPT 优化逻辑
  - 保留初值策略和跳变保护
- [ ] 修改 `lerobot_replay_ee.py`
  - 支持 `--ik.backend=pinocchio_casadi` 选项
  - 使用 `PiperPinocchioIK` 替代 placo
- [ ] 创建 `replay_ee_pinocchio.sh` 测试脚本

#### 5.3.2 验收标准

1. **FK → IK → FK 闭环测试**
   ```python
   # 随机取 10 个关节角 q（在限位内）
   for q in random_joint_configs:
       T = fk.forward_kinematics(q)      # FK
       q_ik = ik.inverse_kinematics(T)   # IK
       T_check = fk.forward_kinematics(q_ik)  # FK again

       position_error = np.linalg.norm(T[:3] - T_check[:3])
       # 要求: position_error < 1cm
   ```

2. **实机回放测试**
   - 先用 `visualize_replay_safety.py` 确认安全
   - 再上机回放，观察动作是否与录制一致

---

## 6. 配置参数汇总

### 6.1 AgileXConfig 新增字段

```python
@dataclass
class AgileXConfig(RobotConfig):
    # ... 现有字段 ...

    # === Pinocchio Kinematics Configuration ===
    kinematics_backend: str = "placo"  # "placo" or "pinocchio"

    # Pinocchio 专用配置
    pinocchio_urdf_path: str = "my_sim/piper_ros-noetic/src/piper_description/urdf/piper_description.urdf"
    pinocchio_package_dirs: list[str] = field(
        default_factory=lambda: ["my_sim/piper_ros-noetic/src"]
    )
    pinocchio_lock_joints: list[str] = field(
        default_factory=lambda: ["joint7", "joint8"]
    )

    # EE 定义（以 ik_demo.py 为准）
    ee_rotation_rpy: list[float] = field(
        default_factory=lambda: [0.0, -1.57, -1.57]
    )
    ee_translation: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0]
    )

    # IK solver 参数
    ik_solver_max_iter: int = 50
    ik_solver_tol: float = 1e-4
    ik_regularization_weight: float = 0.01
```

### 6.2 录制命令模板

```bash
# record_ee_pinocchio.sh
lerobot-record \
    --robot.type=agilex \
    --robot.mock=false \
    --robot.record_ee_pose=true \
    --robot.kinematics_backend=pinocchio \
    --robot.pinocchio_urdf_path="my_sim/piper_ros-noetic/src/piper_description/urdf/piper_description.urdf" \
    --robot.pinocchio_package_dirs='["my_sim/piper_ros-noetic/src"]' \
    --robot.ee_rotation_rpy='[0, -1.57, -1.57]' \
    --robot.ee_translation='[0, 0, 0]' \
    --dataset.repo_id=cqy/agilex_vla_demo_ee_pinocchio \
    # ... 其他参数 ...
```

### 6.3 回放命令模板

```bash
# replay_ee_pinocchio.sh
python -m lerobot.scripts.lerobot_replay_ee \
    --robot.type=agilex \
    --robot.mock=false \
    --robot.kinematics_backend=pinocchio \
    --robot.pinocchio_urdf_path="my_sim/piper_ros-noetic/src/piper_description/urdf/piper_description.urdf" \
    --robot.ee_rotation_rpy='[0, -1.57, -1.57]' \
    --ik.backend=pinocchio_casadi \
    --dataset.repo_id=cqy/agilex_vla_demo_ee_pinocchio \
    --dataset.episode=0 \
    --dataset.fps=30 \
    --ik.arms=both
```

---

## 7. 风险与注意事项

### 7.1 已知风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| pinocchio.casadi 不可用 | IK 无法运行 | 重新安装支持 casadi 的 pinocchio |
| 关节映射不正确 | FK/IK 结果错误 | 通过实验调整 signs/offsets |
| IK 不收敛 | 回放抖动或失败 | 调整 solver 参数，使用可视化工具验证 |
| EE 定义不一致 | 无法闭环 | 严格使用 ik_demo.py 的定义 |

### 7.2 安全注意事项

1. **先可视化，再上机**：Phase 1.5 的可视化验证必须通过
2. **小步迭代**：先用小 fps（如 10Hz）回放，确认安全后再提高
3. **保留关节限位**：`enforce_joint_limits=true`
4. **离线预计算 IK**：推荐使用 `--replay.mode=offline`

---

## 8. 时间规划

| 阶段 | 预计工作量 | 依赖 |
|------|-----------|------|
| Phase 1: FK 录制 | 1-2 天 | 依赖安装完成 |
| Phase 1 验收 | 0.5 天 | Phase 1 完成 |
| Phase 1.5: 可视化工具 | 1 天 | Phase 1 完成 |
| Phase 2: IK 回放 | 1-2 天 | Phase 1.5 验收通过 |
| Phase 2 验收 | 0.5-1 天 | Phase 2 完成 |
| **总计** | **4-6 天** | |

---

## 9. 参考资料

- `my_ik/fk_demo.py` - 同事 FK 实现（ROS2）
- `my_ik/ik_demo.py` - 同事 IK 实现（ROS1）
- `my_ik/docs/先验_ikfk.md` - 原始方案文档
- `src/lerobot/robots/agilex/` - 现有 Agilex 集成代码
- `src/lerobot/model/kinematics.py` - 现有 placo kinematics（参考）
