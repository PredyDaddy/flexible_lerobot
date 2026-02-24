# 先验 IK/FK 方案（基于同事提供的 `my_ik/fk_demo.py` / `my_ik/ik_demo.py`）

> 目标：**严格以你同事给的 pinocchio IK/FK 逻辑为“先验/标准”**，把末端位姿（EE pose）录进 LeRobot dataset，并且在回放时用同一套 IK 把 EE pose 还原成关节角再驱动 Agilex（LeRobot 内的 agilex 机器人），而不是用 piper sdk 重构你的整套框架。
>
> 本文是完整可落地的工程方案说明（偏“怎么做/怎么验收”），不要求你懂运动学数学细节。

---

## 0. 为什么你会迷茫（先把“问题本质”讲清楚）

你同事给的 `fk_demo.py` / `ik_demo.py` 和 LeRobot 内置的 `RobotKinematics`（placo）**本质上在做同一件事**：

- 都是解析同一份 URDF（`piper_description.urdf`），然后：
  - FK：`q(关节角) -> T(末端位姿)`
  - IK：`T(末端位姿) -> q(关节角)`

真正决定“末端位姿对不对”的关键，不是用 pinocchio 还是 placo，而是三件事是否对齐：

1. **关节映射是否对齐**：ROS 读到的 6 个角，对应 URDF 里的 joint1..joint6 的顺序/零位/正方向是否一致（需要 signs/offsets）。
2. **末端 TCP 定义是否对齐**：你同事脚本里不是直接用 URDF 的 `gripper_base`，而是“在 joint6 上自定义了一个 `ee` frame”（带固定旋转/偏置）。
3. **单位/姿态表示是否对齐**：rad vs deg、rpy vs rotvec vs quat。

本文的“先验 IKFK 方案”就是：**把你同事的 `ee` 定义、关节映射、IK 初值策略、约束与优化方式完整搬进 LeRobot 的录制/回放链路**，让“录到的 EE”在“回放 IK”里能闭环。

---

## 1. 同事脚本到底做了什么（你不需要懂细节，但要知道关键点）

### 1.1 `my_ik/fk_demo.py`（pinocchio FK）

核心逻辑（去掉 ROS2/可视化外壳后）：

- 读 URDF：`piper_description(.urdf)`（脚本里通过 `get_package_share_directory('piper_description')` 找 ROS package）
- 锁定 `joint7` / `joint8`（两个手指的伸缩关节），只保留 6-DOF 机械臂
- 定义一个“自定义末端变换”：
  - `first_matrix = create_transformation_matrix(0,0,0, 0, -1.57, 0)`
  - `second_matrix = create_transformation_matrix(gripper_xyzrpy...)`（默认 `x=0.19`）
  - `last_matrix = first_matrix @ second_matrix`
- FK：用 pinocchio 从 base 计算到 joint6 的位姿，再乘 `last_matrix` 得到“ee pose”
- 输出是 `xyzrpy`（roll/pitch/yaw），并发布到 ROS topic（注意脚本用的是 **ROS2 rclpy**）。

### 1.2 `my_ik/ik_demo.py`（pinocchio + casadi IK）

核心逻辑（去掉 ROS/可视化/控制外壳后）：

- 读 URDF（脚本里写死 `/home/agilex/piper_ws/...`，需要替换成你 repo 内 URDF）
- 同样锁定 `joint7` / `joint8`，只对 6 关节求解
- 在 joint6 上 `addFrame("ee", ...)` 作为 IK 目标 frame
- 用 casadi + IPOPT 做优化：
  - 目标：最小化 `log6(T_current^-1 * T_target)` 的 6 维误差（位置 + 姿态）
  - 带关节上下限约束（来自 URDF）
  - 用上一帧解作为初值（`self.init_data`），并做“跳变过大则重置”的保护
- 输出 `sol_q`（关节角），脚本里把它发给 `piper_control` 控硬件（这部分在 LeRobot 集成里不需要）。

### 1.3 两个脚本在“ee 定义”上可能不完全一致（必须明确）

你会注意到：

- `fk_demo.py` 用的是 `first_matrix(0,-1.57,0)` + `gripper_xyzrpy`
- `ik_demo.py` 里 `addFrame('ee', ...)` 的旋转写的是 `(0, -1.57, -1.57)`，且没有平移

这意味着：**如果你直接“拿 FK 录制的 ee pose”去喂“IK 求解器”，但两边 `ee` 定义不一致，就无法闭环回放**。

因此本方案的第一条工程原则是：

- **先验定义 = 你同事 IK 求解器里使用的 `ee` frame 定义**
- FK 录制必须输出同一个 `ee` frame 的位姿（同一套固定旋转/偏置）

落地做法（推荐）：

1. 把 `ee` 的固定变换统一成一个配置：`ee_offset_xyzrpy=[x,y,z,roll,pitch,yaw]`
2. FK 与 IK 都使用同一份 `ee_offset_xyzrpy`
3. 用 1～2 个静态姿态做“FK->IK->FK 误差闭环测试”（后文给步骤）

> 备注：你同事脚本里既有 `gripper_xyzrpy`（默认 0.19m 前向偏置）也有固定旋转（-1.57 等）。
> 工程上你不需要理解原因，只要把它们当作“TCP 定义的一部分”保持一致即可。

---

## 2. 总体架构（录制与回放怎么闭环）

### 2.1 录制（Record）数据流

在 LeRobot 的 `lerobot-record` 主循环中（你现有 `record.sh`）：

1. 从 ROS `/puppet/joint_left`、`/puppet/joint_right` 读到 7 维（6+gripper）的关节状态（rad）
2. 取前 6 维作为机械臂关节角 `q_ros`（rad）
3. 做关节映射修正得到 `q_urdf`（rad）：
   - `q_urdf = q_ros * signs + offsets`
4. 用 pinocchio FK 计算 `T_base_ee(q_urdf)`（这里的 `ee` 必须与 IK 的 `ee` 完全一致）
5. 把 `ee` 位姿拆成标量字段写入 observation（满足 LeRobot dataset 的 feature 规则）

最终 dataset 里会同时包含：

- 原有：关节角（14 维）
- 新增：`left_ee.*`、`right_ee.*`（每臂 6 维或更多，取决于你记录的姿态表示）

### 2.2 回放（Replay）数据流（EE -> IK -> Joint -> ROS）

增加一个“末端回放”的 replay 模式（推荐新脚本，避免破坏原 `lerobot-replay`）：

1. 从 dataset 读每一帧的 `left_ee.*`/`right_ee.*`（以及可选的 gripper）
2. 把 EE pose 还原成 `T_base_ee_target`（4x4）
3. 用 pinocchio IK 求解：
   - 输入：`T_base_ee_target`
   - 初值：上一帧 IK 解（保证连续性）
   - 输出：`q_urdf_solution`（rad）
4. 把 `q_urdf_solution` 反向映射回 ROS 关节角 `q_ros_cmd`（rad）：
   - `q_ros = (q_urdf - offsets) / signs`
5. 拼装成 LeRobot action dict（`left_*.pos/right_*.pos`），调用 `AgileXRobot.send_action()` 发到 `/master/joint_left/right`

这样就实现了闭环：

- 录制时：`q_ros -> FK -> ee`
- 回放时：`ee -> IK -> q_ros`

---

## 3. 关键工程决策（推荐选项）

### 3.1 不直接“跑同事脚本当 ROS 节点”（强烈不推荐）

原因：

- `fk_demo.py` 用的是 **ROS2 rclpy**，而你 Agilex LeRobot bridge 用的是 **ROS1 rospy**
- 直接订阅/发布需要 ROS1<->ROS2 bridge，工程复杂且维护成本高
- IK 脚本里还混了 `meshcat/cv2/piper_control/piper_msgs` 等依赖，不适合作为 LeRobot 回放内核

结论：**不要把同事脚本当黑盒节点跑起来再去订阅 topic**。

### 3.2 推荐做法：抽出“纯计算 IK/FK 内核”，嵌入 LeRobot（推荐）

思路：

- 保留你同事脚本里“pinocchio + casadi 的数学核心”
- 删除/绕过 ROS、可视化、硬件控制相关代码
- 作为一个可复用模块，在 LeRobot 录制与回放中直接调用

这样才能：

- 在同一个 Python 进程里做 FK/IK（无需 ROS bridge）
- 复用你现有 agilex 控制链路（`send_action` -> `/master/joint_*`）
- 保证“先验 ee 定义”一致

---

## 4. 详细落地步骤（把同事 IK/FK 变成 LeRobot 可用的后端）

下面按“你要改哪些代码/新增哪些脚本/怎么验收”给出可执行的工程路径。

> 说明：这里写的是“完整方案”，你可以分阶段做：
> - Phase 1：只录 EE（FK）
> - Phase 2：新增 EE 回放（IK）

### 4.1 Phase 1：只用同事 FK 录制 EE（不改回放）

#### 4.1.1 抽出纯 FK 内核（pinocchio）

建议新增一个“纯计算模块”（位置任选其一）：

- 推荐放进 LeRobot 包里（便于长期维护）：`src/lerobot/robots/agilex/piper_pinocchio_kinematics.py`
- 或者先放在 `my_ik/` 里验证（不推荐长期这样做）

模块职责：

- 输入：URDF 路径、package_dirs（用于解析 `package://piper_description/...` mesh）、锁定关节列表、`ee_offset_xyzrpy`
- 初始化一次：构建 pinocchio model（reduced robot）
- 每帧：`fk(q_urdf_rad) -> T_base_ee (4x4)`

关键点（对齐同事脚本）：

- 锁定 `joint7`、`joint8`（URDF 里两个 prismatic 手指关节），FK 只对 6 关节
- `ee_offset_xyzrpy` 的实现要与同事脚本的 `create_transformation_matrix()` 一致（同事脚本是典型的 ZYX yaw-pitch-roll 组合）
- FK 输出的 `ee` 必须是你 IK 那边用的同一个 `ee`

> 强烈建议：不要把你同事脚本里的 `first_matrix @ second_matrix` 强行“合并成一个 xyzrpy”写死。
> 欧拉角不是可加的，合并后很容易因为约定不一致导致 FK/IK 不闭环。
>
> 最稳妥的做法是：**保持同事脚本的结构**，配置里直接存两段 xyzrpy，然后在代码里做矩阵乘法：
>
> - `ee_first_xyzrpy = [0,0,0, 0, -1.57, 0]`
> - `ee_second_xyzrpy = gripper_xyzrpy（同事脚本默认 [0.19,0,0, 0,0,0]）`
> - `T_offset = T(ee_first_xyzrpy) @ T(ee_second_xyzrpy)`

#### 4.1.2 把 FK 接入 `AgileXRobot.get_observation()`

做法与我们之前 “Path A（placo）”一致，只是 FK 后端换成 pinocchio：

- 在 `AgileXConfig` 增加：
  - `record_ee_pose: bool`
  - `kinematics_backend: str = "pinocchio"`（或单独布尔开关，例如 `use_pinocchio_ikfk: bool`）
  - `pinocchio_urdf_path: str = "my_sim/piper_ros-noetic/src/piper_description/urdf/piper_description.urdf"`
  - `pinocchio_package_dirs: list[str] = ["my_sim/piper_ros-noetic/src"]`
  - `pinocchio_lock_joints: list[str] = ["joint7", "joint8"]`
  - `ee_first_xyzrpy: list[float] = [0,0,0, 0, -1.57, 0]`
  - `ee_second_xyzrpy: list[float] = [0.19,0,0, 0,0,0]`
  - `left/right_joint_signs`、`left/right_joint_offsets_rad`
- 在 `AgileXRobot.connect()` 初始化 FK 模型（只做一次）
- 在 `AgileXRobot.get_observation()`：
  - 读 `q_ros`（rad）
  - 映射到 `q_urdf`（rad）
  - 算 FK 得 `T_base_ee`
  - 写入 `obs["left_ee.*"]`、`obs["right_ee.*"]`

建议涉及文件（便于你按仓库结构落地）：

- `src/lerobot/robots/agilex/config_agilex.py`：新增 pinocchio 配置字段
- `src/lerobot/robots/agilex/agilex.py`：初始化 pinocchio FK；每帧写入 `left_ee.*` / `right_ee.*`

#### 4.1.3 EE 位姿在 dataset 里怎么存（建议存 xyz + rpy，方便直接喂 IK）

因为你同事 IK 的入口就是 `(x,y,z,roll,pitch,yaw)`，最省事的录制字段是：

- `left_ee.x / y / z`
- `left_ee.roll / pitch / yaw`
- 右臂同理

这些都必须是 `float` 标量（不要 tuple），否则 LeRobot dataset 会把它当 image-like 特征。

> 如果你更希望保留 LeRobot 生态常用的 rotvec（`wx/wy/wz`），也可以：
> - 录制 rotvec
> - 回放时把 rotvec 转回 rpy 再喂 IK
> 但这会多一步姿态转换与奇异性处理，工程复杂度更高。

#### 4.1.3.1 推荐的 observation keys（双臂）

建议固定使用如下 key（全部标量 float）：

- 左臂：`left_ee.x/y/z` + `left_ee.roll/pitch/yaw`
- 右臂：`right_ee.x/y/z` + `right_ee.roll/pitch/yaw`

这样做的好处是：回放 IK 不需要再做姿态表示转换，直接用 `(x,y,z,roll,pitch,yaw)` 组装目标位姿。

#### 4.1.4 Phase 1 验收（必须通过）

1. 同一组关节角 `q`：
   - LeRobot 内算出来的 `ee`（pinocchio FK）
   - 与同事 `fk_demo.py` 的 `Arm_FK.get_pose()` 输出的 `xyzrpy`
   - 误差应当非常小（数值级别接近，至少 cm/deg 以内；如果相差很大一定是 ee 定义或关节映射没对齐）
2. 录制一个短 episode 后检查 dataset feature names 包含 `left_ee.*` / `right_ee.*`

到这里，你已经实现“用同事 FK 作为先验，把 EE 录进 dataset”。回放仍可以用关节 action（现有 `lerobot-replay` 不变）。

---

### 4.2 Phase 2：用同事 IK 做“末端回放”（EE -> IK -> joint）

Phase 2 才是你想要的“录制末端数据，回放末端数据”。

#### 4.2.1 抽出纯 IK 内核（pinocchio + casadi）

建议把 `ik_demo.py` 的 IK 求解部分重构为一个纯类：

- 输入：
  - URDF 路径 + package_dirs
  - 锁定关节 `joint7/joint8`
  - `ee_offset_xyzrpy`（与 FK 一致）
  - solver 超参数（迭代次数、tol、权重等）
- API：
  - `solve(T_target, q_init) -> q_solution`

**关键架构调整：**

- **核心计算类**：必须删掉/绕过 Meshcat 可视化、ROS、`piper_control` 等依赖。这是为了让它能在 LeRobot 的训练/推理进程中高效运行。
- **独立可视化工具（新增）**：虽然核心类不带可视化，但为了满足你的**安全上机需求**，我们需要专门写一个**独立的可视化脚本**（详见 Phase 1.5）。这个脚本会调用核心计算类，并用 Meshcat 画出结果。

保留的关键点（对齐同事行为）：

- 关节上下限约束（URDF 自带）
- 初值策略：上一帧解作为下一帧初值
- “跳变过大则重置初值”的保护逻辑（防止解发散）

---

### 4.3 Phase 1.5：可视化安全验证工具（Safety Visualizer）

**这是你最关心的部分：在上机前回放，确保不出问题。**

我们需要创建一个新的独立脚本（例如 `scripts/visualize_replay_safety.py`），它不连接真实机器人，而是连接 **Meshcat 浏览器可视化**。

#### 功能设计
脚本加载你录制好的 dataset，提供两种“影子模式”供你肉眼对比：

1.  **绿色影子（Ground Truth / 关节回放）**：
    - 直接读取 dataset 里的 `joint_position`。
    - 通过 Pinocchio FK 更新 URDF 模型姿态。
    - **含义**：这是你录制时真实的动作。如果这个看起来不对，说明录制或关节映射有问题。

2.  **红色影子（Solver Test / EE 回放）**：
    - 读取 dataset 里的 `ee_pose`。
    - **实时调用你重构的 IK 核心类** 算出关节角。
    - 更新另一个红色 URDF 模型姿态。
    - **含义**：这是 IK 求解器“认为”机器人该怎么走。

#### 验收标准
在上机前，你运行这个脚本：
- 如果**红绿重合**：IK 完美，可以安全上机。
- 如果**红色乱飞/抽搐**：IK 参数（正则化/限位）有问题，或者碰到了奇异点。**绝对不能上机**，必须先调优参数。

---

#### 4.2.2 新增一个 EE 回放脚本（推荐新建，不改原 `lerobot-replay`）

原因：

- 原 `lerobot-replay` 固定从 dataset 的 `action` 读关节角回放
- EE 回放需要读 `observation`（末端位姿）并做 IK

建议新增：`src/lerobot/scripts/lerobot_replay_ee.py`（入口名可叫 `lerobot-replay-ee`）。

脚本要做的事：

1. 加载 dataset episode
2. 每帧从 `observation.state` 里取出 `left_ee.*`/`right_ee.*`（以及 gripper）
3. 组装 `T_target_left/right`
4. 调 IK 得 `q_left/right`（rad）
5. 做反向映射得到 ROS 命令关节角（rad）
6. 调 `robot.send_action(action_dict)`

为了稳定与实时性，推荐两种运行模式（二选一或都实现）：

**模式 A：离线预计算 IK（强烈推荐）**

- 先把整条 episode 的 IK 全部算完，得到 `q[t]`
- 再按固定 fps 发给机器人
- 优点：回放 fps 稳定，不受 IK 计算耗时影响
- 缺点：启动回放前需要等待预计算

**模式 B：在线逐帧 IK（实现简单，但可能卡顿）**

- 每帧现算现发
- 如果 IK 计算时间 > `1/fps`，回放会慢或抖

#### 4.2.3 Phase 2 验收（必须通过）

必须做一个“闭环一致性”验收：

1. 随机取 10 个关节角 `q`（在限位内）
2. 先做 FK 得 `T`
3. 再做 IK 得 `q'`
4. 再做 FK 得 `T'`

要求：

- `T` 与 `T'` 的误差很小（位置误差在 mm~cm 级；姿态误差在小角度内）

如果不通过，优先排查顺序：

1. FK/IK 是否使用同一份 `ee_offset_xyzrpy`
2. FK/IK 是否使用同一份 URDF、同一份 locked joints
3. `q_ros <-> q_urdf` 映射是否一致（sign/offset 的正反向）

---

## 5. 依赖与环境（你需要提前确认）

### 5.1 Python 依赖

要在 LeRobot 里跑你同事的 pinocchio IKFK，通常需要：

- `pinocchio`
- `casadi`
- `ipopt`（casadi 的 solver backend，具体安装方式取决于你的环境）

你同事脚本还引用了：

- `meshcat`、`cv2`（可视化/调试，可删）
- ROS1/ROS2 相关包（在“纯计算模块”里不需要）

建议做法：

- Phase 1（只 FK）先保证 `pinocchio` 能 import 即可
- Phase 2（IK）再补齐 `casadi/ipopt`

### 5.2 URDF 与 package:// 资源解析

你的 URDF 在 repo 内：

- `my_sim/piper_ros-noetic/src/piper_description/urdf/piper_description.urdf`

URDF 内 mesh 路径是 `package://piper_description/...`，pinocchio 需要 `package_dirs` 才能找到 mesh：

- 推荐把 `package_dirs` 指向包含 `piper_description/` 的父目录：
  - `my_sim/piper_ros-noetic/src`

这样 pinocchio 才能解析 `package://piper_description/meshes/*.STL`。

---

## 6. 左右臂问题：URDF 是单臂，双臂怎么用？

`piper_description.urdf` 是单臂模型（没有 left/right 两套链）。

在 LeRobot 中对双臂的做法是：

- 左臂：用一套 FK/IK 实例
- 右臂：用另一套 FK/IK 实例（可以复用同一份 URDF，但**必须有各自的初值状态**）

如果左右臂在硬件上是镜像、或驱动层做了坐标系/零位约定差异：

- 用 `left_*` / `right_*` 的 `signs/offsets` 分别对齐

---

## 7. 常见坑与排查清单（按优先级排序）

1. **ee 定义不一致**：FK 用了 `gripper_base`，IK 用了自定义 `ee`，或两边固定旋转不一致 → 必然无法闭环
2. **关节映射不一致**：sign/offset 写错，或者正向/反向映射没写对
3. **姿态表示混乱**：录了 rpy，回放当 rotvec；或 rpy 顺序/旋转约定不一致
4. **IK 性能不够**：双臂 30Hz 在线求解基本很吃力 → 优先选“离线预计算”
5. **依赖太重**：meshcat/cv2/ROS 混进纯计算模块 → 建议彻底剥离

---

## 8. 最小可行落地建议（你按这个顺序做最稳）

1. 先把 `fk_demo.py` 的 FK 部分抽成“纯函数/纯类”，在 python 里给定 `q` 能返回 `T`/`xyzrpy`
2. 在 LeRobot agilex 里接入 FK 并录进 dataset（Phase 1）
3. 做 FK 输出与 `fk_demo.py` 输出的一致性对比（验收）
4. 再把 `ik_demo.py` 的 IK 求解抽成纯类（去掉 ROS/可视化/硬件控制）
5. **实现 `visualize_replay_safety.py`，加载录好的数据，对比“关节回放”与“IK解算回放”的轨迹（Phase 1.5）。只有当两者在视觉上一致且平滑时，才进入下一步。**
6. 写 `lerobot-replay-ee` 做离线预计算 IK 再回放（Phase 2）
7. 做 FK->IK->FK 闭环验收


---

## 9. 操作命令模板（录制/末端回放）

> 这里给的是“你把方案落地后”的命令模板，便于你把录制/回放串起来跑通并验收。

### 9.1 录制（Record：关节角 + 先验 EE）

你的入口是 `record.sh`，建议复制一份：

```bash
cp record.sh record_ee_pinocchio.sh
```

在 `record_ee_pinocchio.sh` 的 `lerobot-record \` 参数末尾追加（示例）：

```bash
--robot.record_ee_pose=true \
--robot.kinematics_backend=pinocchio \
--robot.pinocchio_urdf_path="my_sim/piper_ros-noetic/src/piper_description/urdf/piper_description.urdf" \
--robot.pinocchio_package_dirs='["my_sim/piper_ros-noetic/src"]' \
--robot.pinocchio_lock_joints='["joint7","joint8"]' \
--robot.ee_first_xyzrpy='[0,0,0,0,-1.57,0]' \
--robot.ee_second_xyzrpy='[0.19,0,0,0,0,0]' \
--robot.left_joint_signs='[1,1,1,1,1,1]' \
--robot.left_joint_offsets_rad='[0,0,0,0,0,0]' \
--robot.right_joint_signs='[1,1,1,1,1,1]' \
--robot.right_joint_offsets_rad='[0,0,0,0,0,0]'
```

然后运行：

```bash
bash record_ee_pinocchio.sh
```

> 注意：新增 EE 字段会改变 dataset schema，**不要对旧数据集使用 `--resume=true`**，建议换一个新的 `--dataset.repo_id`。

### 9.2 快速检查：确认 dataset 里真的有 EE 字段

录完一个 episode 后，执行：

```bash
python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import OBS_STR

repo_id = "cqy/agilex_vla_demo"  # 改成你的
ds = LeRobotDataset(repo_id)     # 如果录制时指定了 --dataset.root，这里也传 root
names = ds.features[f"{OBS_STR}.state"]["names"]
print([n for n in names if "ee." in n])
PY
```

### 9.3 回放（Replay：关节角回放，作为 fallback）

原始关节角回放不变：

```bash
bash replay.sh
```

### 9.4 回放（Replay：末端位姿回放 = EE -> IK -> Joint）

等你实现了 `lerobot-replay-ee`（本方案 Phase 2 的新脚本）后，典型命令形态是：

```bash
lerobot-replay-ee \
  --robot.type=agilex \
  --robot.mock=false \
  --dataset.repo_id=cqy/agilex_vla_demo \
  --dataset.episode=0 \
  --dataset.fps=30 \
  --ik.backend=pinocchio_casadi \
  --ik.urdf_path="my_sim/piper_ros-noetic/src/piper_description/urdf/piper_description.urdf" \
  --ik.package_dirs='["my_sim/piper_ros-noetic/src"]' \
  --ik.lock_joints='["joint7","joint8"]' \
  --ik.ee_first_xyzrpy='[0,0,0,0,-1.57,0]' \
  --ik.ee_second_xyzrpy='[0.19,0,0,0,0,0]' \
  --replay.mode=offline
```

其中：

- `--replay.mode=offline`：先预计算整条 episode 的 IK，再按 fps 发送（推荐）
- `--replay.mode=online`：逐帧在线求解（实现简单，但容易卡顿）

---

## 附录 A：Agilex 关节顺序与 URDF joint1..6 的映射

你当前 Agilex 观测与 action 的关节顺序来自：

- `src/lerobot/robots/agilex/config_agilex.py:JOINT_NAMES`

顺序是：

1. `shoulder_pan`
2. `shoulder_lift`
3. `shoulder_roll`
4. `elbow`
5. `wrist_pitch`
6. `wrist_roll`
7. `gripper`（LeRobot 里是单标量开合/行程）

URDF（`piper_description.urdf`）里机械臂 6 关节通常是：

- `joint1..joint6`

推荐约定（从最常见情况开始）：

- `shoulder_pan -> joint1`
- `shoulder_lift -> joint2`
- `shoulder_roll -> joint3`
- `elbow -> joint4`
- `wrist_pitch -> joint5`
- `wrist_roll -> joint6`

如果 FK/IK 对不上，通过 `left/right_joint_signs` 与 `left/right_joint_offsets_rad` 修正。

### 关于 gripper 与 URDF 的 joint7/joint8

URDF 里：

- `joint7`、`joint8` 是两个手指的 prismatic 关节（通常一正一负）

在本方案里（对齐同事脚本）：

- FK/IK 计算时锁定 `joint7/joint8`，只求 6-DOF 位姿
- 回放时仍然把 dataset 里的 `left_gripper.pos/right_gripper.pos` 原样发给 Agilex（由驱动层处理）

如果你未来要把 LeRobot 的单标量 gripper 映射成 URDF finger joints（例如做碰撞检测），可以参考你同事 IK 里的做法：

- `joint7 = gripper / 2`
- `joint8 = -gripper / 2`
