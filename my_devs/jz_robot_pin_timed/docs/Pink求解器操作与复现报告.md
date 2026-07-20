# Pink 求解器操作与复现报告

更新日期：2026-07-17

适用仓库：`/home/luzhuang/cqy/aaa/flexible_lerobot`

适用组件：`my_devs/my_var_tp/live_vr_replay_bridge` 与其引用的 `light_teleop`

## 1. 背景与结论

当前 VR 遥操中的 Pink 并不直接连接机器人，也不读取 UDP State `39010`、不向 Orin command `39020`
发包。Pink 的职责只有一件事：**在给定机器人运动学模型、当前关节配置和两个末端位姿目标时，求出
下一小步的双臂关节运动。**

完整链路是：

```text
VR pose
  -> 坐标映射与目标限速
  -> 左/右末端 SE3 target
  -> Pink differential IK
  -> 双臂 14 关节 target
  -> target_action UDP
  -> 另一个 receiver/recorder 结合 State 与控制策略
  -> command UDP
  -> Orin executor
  -> 真实机器人
```

因此，复现 Pink 操作要分成两层：

1. **运动学复现**：URDF + Pinocchio + Pink，离线验证末端目标能否转成合理关节轨迹。
2. **机器人系统集成**：输入设备、坐标映射、目标发布、机器人 State、command transport、安全门和
   真实驱动器。

只复制 Pink 求解器并不能让另一台机械臂被 VR 控制；还必须重新建立第二层。

## 2. 本项目中的实际组件边界

| 模块 | 实际职责 | 不负责什么 |
|---|---|---|
| Pinocchio | 读取 URDF、构建运动学模型、正/逆运动学相关计算、关节限制 | VR 网络、真实机器人通信 |
| Pink | 将多个 task 组成速度级 QP/IK 问题，求解关节速度 | 读取 VR UDP、发送 command、驱动电机 |
| `VrCoordinateTransformer` | 将 VR 的相对位移和旋转映射为机器人末端目标 | 求解关节角 |
| `vr_visual_publisher` | 读取 VR、调用 Pink、更新 Meshcat、发布 target action | 直接控制 Orin command port |
| recorder/replay/receiver/policy | 消费 target action 或 policy action，读取机器人 State，生成 command | 求解 VR 的 IK |
| Orin executor | 接收并执行 command | 解释 Pink task、读取 VR 数据 |

活跃集成代码位于：

- [Pink solver](../../my_var_tp/better_version_reference/my_robot/light_teleop/light_teleop/pink_solver.py)
- [VR visual publisher](../../my_var_tp/live_vr_replay_bridge/live_vr_replay_bridge/vr_visual_publisher.py)
- [模型加载器](../../my_var_tp/better_version_reference/my_robot/light_teleop/light_teleop/model_loader.py)
- [VR 坐标变换器](../../my_var_tp/better_version_reference/my_robot/test/teleop_vr_py/coordinate_transformer.py)

## 3. Pink 在数学上做什么

### 3.1 不是“直接反解一个关节角”

Pink 在这里使用的是 **differential inverse kinematics，速度级逆运动学**。每个循环不是直接寻找
“目标位姿对应的唯一关节角”，而是在当前配置 `q` 附近求一个小的关节速度 `dq/dt`：

```text
已知：当前 q、时间步长 dt、左右末端目标 T_left* / T_right*
求解：关节速度 v
积分：q_next = integrate(q, v, dt)
```

末端速度和关节速度由 Jacobian 近似关联：

```text
v_ee = J(q) v_joint
```

单个末端任务希望减少当前位姿与目标位姿之间的误差。双臂、姿态保持、回 home 偏好和关节运动偏好
一起形成一个优化问题。直观上可以写成：

```text
minimize
  left_position_error
  + left_orientation_error
  + right_position_error
  + right_orientation_error
  + posture_regularization
  + optional_joint_motion_regularization

subject to
  robot kinematics
  URDF joint limits
```

Pink 接收 task 列表和当前 `Configuration`，内部构造 QP，由 `qpsolvers` 选择可用的 QP solver 求解。
项目不在活跃路径中手工实现矩阵求逆。

### 3.2 为什么需要多个 task

只有“左右手必须到目标点”会造成冗余关节自由度没有偏好，姿态可能漂移或出现不稳定姿势。因此当前
求解器同时使用：

```text
left FrameTask
right FrameTask
PostureTask
optional motion PostureTask
```

任务含义：

| Task | 目标 | 当前实际默认/配置 |
|---|---|---|
| `left FrameTask` | 左末端跟踪世界坐标系下的 SE3 目标 | position cost `1.0`，orientation cost `0.05` |
| `right FrameTask` | 右末端跟踪世界坐标系下的 SE3 目标 | position cost `1.0`，orientation cost `0.05` |
| `PostureTask` | 在跟踪误差较小时温和偏向 home posture | CLI 默认 cost `0.01` |
| optional `motion_task` | 抑制每一步关节变化 | `joint_motion_cost > 0` 才加入，现场常用 `0.0` |

这里 position 权重显著高于 orientation 权重，表示系统优先让手到达指定位置；手腕姿态有较低但非零的
约束。调整它们会改变机械臂的动作风格，并不是“更高一定更好”。

## 4. 本项目 Pink 求解器的实现

### 4.1 从 URDF 创建 Pinocchio 模型

默认 URDF 为：

```text
my_devs/my_var_tp/better_version_reference/my_robot/
  jz_descripetion-main/robot_urdf/urdf/robot urdf.10.8.SLDASM.urdf
```

模型加载流程：

```text
URDF
  -> pin.buildModelFromUrdf(...)
  -> pin.RobotWrapper(model)
  -> q0 = 当前选择的 home configuration
```

只做运动学时可以不加载 mesh；需要 Meshcat 几何显示时再传入 package directory 加载视觉几何。URDF
至少必须准确表达：关节顺序、旋转轴、父子 link、link 长度、joint limit、末端 frame 名称。URDF 有
任何一个错误，Pink 即使“成功求解”，对应真实机器人也可能走错方向或撞到极限。

### 4.2 构建 14 自由度 reduced model

当前机器人全模型可能包含头、腰或其他非双臂关节。Pink solver 不让这些关节参与双臂 VR IK：

```text
full model
  -> 保留 left_joint1..7 + right_joint1..7
  -> 其余可动关节以 q0 锁定
  -> pin.buildReducedModel(...)
  -> reduced model（14 DoF）
```

这样做的结果：

- VR 不会通过 Pink 意外改变头、腰或其他非手臂关节；
- 双臂 IK 的优化变量只有 14 个关节；
- 每步后会验证非手臂关节偏移接近零；
- 对另一种机器人，必须重新列出“要控制的关节”和“必须锁定的关节”。

### 4.3 末端 frame 与初始姿态

求解器代码的默认末端 frame 是：

```text
left_arm_link9
right_arm_link9
```

实际 VR publisher 可通过 `--left-ee-frame`、`--right-ee-frame` 覆盖。当前现场命令也出现过
`left_arm_link7/right_arm_link7`，因此必须以**当前启动命令和当前 URDF 中存在的 frame**为准，不能
机械照抄文档默认值。

推荐的校验顺序：

```text
1. 在 URDF 中确认 frame 存在。
2. 在 Meshcat 中显示该 frame 的坐标轴。
3. 让关节小幅离线变化，确认它是目标 TCP/末端而不是中间 link。
4. 明确是否需 TCP offset；工具尖端与 link frame 不同则必须建 offset。
```

当前 `PinkTeleopSolver` 初始化会：

```text
Configuration(reduced_model, data, q0_reduced)
FrameTask(left_ee_frame, position_cost=1.0, orientation_cost=0.05, lm_damping=1e-4)
FrameTask(right_ee_frame, position_cost=1.0, orientation_cost=0.05, lm_damping=1e-4)
PostureTask(cost=posture_cost, lm_damping=1e-6)
```

### 4.4 每一轮求解做什么

每个周期 `step(dt)` 的具体步骤是：

```text
1. 检查 dt 为有限正数。
2. 根据当前末端误差动态设置 posture target。
3. 将 motion task 的目标设为当前 configuration.q。
4. pink.solve_ik(configuration, tasks, dt, solver=..., damping=1e-8)。
5. 检查求得 velocity 中没有 NaN/Inf。
6. configuration.integrate_inplace(velocity, dt)。
7. 检查 URDF joint limits 与非手臂关节是否保持锁定。
8. 返回新的 q、左右臂关节位置、当前末端姿态和位置误差。
```

动态 posture 逻辑避免“手离目标还很远时，home task 强行拉回初始姿态”。当前实现中，最大末端位置误差
达到 `0.05 m` 时，home bias 关闭；误差足够小时才逐步恢复回 home 的温和偏好。

### 4.5 当前活跃路径没有什么

以下功能经常在 Pink 示例中出现，但**当前求解器并未启用**：

```text
Pink Barrier task：未使用
Pink RateLimiter：未使用
碰撞几何约束：未作为当前 live solver 的硬约束
直接电机控制：Pink 本身不做
```

当前的保护主要来自 URDF joint limit 检查、目标位姿的上层限速、receiver/action filter、X86 Robot
边界和 Orin executor。对新机械臂，不能因为“Pink 能解 IK”就假设已经具备自碰、环境碰撞或速度安全。

## 5. VR 如何变成 Pink 的末端目标

### 5.1 使用相对坐标，而不是把 VR 世界坐标硬贴到机器人

操作者抓住/激活 VR 控制时，系统锁定参考：

```text
VR reference pose
robot reference EE pose
```

之后每轮计算：

```text
vr_delta = current_vr_position - vr_reference_position
mapped_delta = axis_map(vr_delta)
target_position = robot_reference_position + scale * mapped_delta
```

旋转也以相对四元数处理：

```text
vr_rotation_delta = current_vr_orientation * inverse(vr_reference_orientation)
mapped_rotation_delta = axis_map(rotation_delta)
target_orientation = robot_reference_orientation * tool_offset * mapped_rotation_delta * inverse(tool_offset)
```

这样操作者从任意舒适姿势抓住手柄后，机器人从当前末端姿势相对跟随，而不是被强行拉到 VR 设备的绝对
世界坐标。

### 5.2 当前映射配置

当前 demo 默认：

```text
scale = 1.0
axis_mapping = [-2, 3, 1]
target_max_speed = 0.80 m/s
frequency = 60 Hz
posture_cost = 0.01
```

`axis_mapping=[-2, 3, 1]` 表示 VR 三个坐标分量到机器人坐标的重排与符号翻转。它是具体 VR 坐标系与
具体机器人 base/world frame 的标定结果，不是其他机械臂可直接复用的物理常数。

现场脚本可把频率设为 80 Hz，例如 `--frequency 80 --publish-every 1`。这表示目标动作最多每秒发布
80 包；实际机器人控制速度仍取决于 receiver、State、新 command、Orin executor 和网络。

### 5.3 目标限速、姿态模式和 TCP offset

在调用 Pink 前，VR publisher 还可执行：

```text
VR smoothing
target orientation mode
目标位姿单步限速（target max speed）
TCP local offset
```

`--tcp-control-offset` 用来控制工具尖端而不是单纯控制 URDF link 原点。若工具沿末端本地 `z` 轴前伸
`d` 米，目标应是 TCP 位姿；否则视觉上“手到了目标”，真正工具尖端却可能偏移 `d`。

## 6. 从 Pink 输出到 target_action

Pink 输出的是 reduced model 中的关节配置。VR publisher 从中取出左右各 7 个关节，再合并夹爪目标：

```text
Pink q
  -> left_joint1..7 + right_joint1..7
  -> gripper width/force
  -> target_action JSON
  -> UDP sender
```

默认 target-action receiver 是 `127.0.0.1:39030`；独立遥操 wrapper 通过环境变量改为
`127.0.0.1:39031`，以免抢占录制使用的 target-action 端口。两者都只是 X86 本机的目标动作通道。

`target_action` 被 consumer 转为 LeRobot action 后，才会根据模式处理夹爪、滤波、初始差、步长差和
机器人最新 observation，最终使用 `39020/UDP` 发送 command。

## 7. 在另一台机械臂上复现的分阶段步骤

### 7.1 阶段 A：准备可信的机器人运动学资产

必须准备：

```text
URDF（或由 CAD 转出的已校验 URDF）
joint name / joint order 清单
joint lower/upper limits
机器人 base/world frame
左右或单臂末端/TCP frame
home configuration
末端工具偏移
```

验收标准：

```text
每个可动关节在 Pinocchio 中均存在；
FK 与实物/厂商模型方向一致；
关节上下限合理；
末端 frame 与真正工具点对应；
无 NaN、无长度单位混乱（统一米、弧度、秒）。
```

### 7.2 阶段 B：只离线构建 Pinocchio + Pink

为新机器人创建最小配置：

```python
import pinocchio as pin
import pink
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask

model = pin.buildModelFromUrdf("/absolute/path/to/robot.urdf")
data = model.createData()
q0 = pin.neutral(model)
configuration = pink.Configuration(model, data, q0)

tcp_task = FrameTask(
    "tool0",
    position_cost=1.0,
    orientation_cost=0.05,
    lm_damping=1e-4,
)
posture_task = PostureTask(cost=0.01, lm_damping=1e-6)
tcp_task.set_target(configuration.get_transform_frame_to_world("tool0"))
posture_task.set_target(q0)

dt = 1.0 / 60.0
velocity = solve_ik(configuration, [tcp_task, posture_task], dt)
configuration.integrate_inplace(velocity, dt)
configuration.check_limits()
```

先把目标设置为当前末端 pose，确认零运动稳定；再每次只给 `x/y/z` 方向很小的偏移；最后才加入姿态变化。
离线阶段不需要网络、VR、机器人 State 或 command transport。

### 7.3 阶段 C：定义控制自由度

对于单臂机器人，通常只保留一个 FrameTask；对于双臂机器人，保留两个 FrameTask。对于有头、腰、底盘、
升降柱或冗余手指的机器人，决定：

```text
哪些关节进入 IK optimization variable；
哪些关节在 reduced model 中锁定；
夹爪是独立控制还是进入 IK；
是否需要 torso/base task；
是否允许冗余关节为避限位而参与。
```

不要仅按数组索引复制本项目的“14 关节”逻辑；新机器人必须按 joint name 构建映射。

### 7.4 阶段 D：标定 VR 到机器人坐标映射

在不连接真实执行器的可视化环境中完成：

```text
1. 记录 VR 控制器正前、右、上三个方向的位移。
2. 观察机器人末端应对应哪个 base/world 方向。
3. 建立 axis permutation 与 sign。
4. 设置 scale，使 10 cm VR 移动对应可预期的机器人移动。
5. 设置 tool orientation offset，验证旋转方向。
6. 使用相对 reference lock，不使用绝对 VR 世界坐标直接控制机器人。
7. 在 Meshcat 同时画 VR raw 点、mapped target、FK 末端点。
```

通过标准：每个单轴移动只在预期机器人轴上移动；左右手不会镜像错误；小旋转不触发大范围关节翻转。

### 7.5 阶段 E：加入输入和输出适配器

将新机器人与 Pink 隔离在两个 adapter 中：

```text
Input adapter
  VR/HMD/手柄协议 -> position + quaternion + buttons

Kinematics adapter
  joint names / frame names / URDF / home q / TCP offset

Output adapter
  Pink joint positions -> robot SDK / ROS action / UDP command
```

目标是让 Pink 不知道网络协议、机器人厂商 SDK 和数据集格式。替换机械臂时优先替换 adapter，而不是
修改通用 Pink 优化逻辑。

### 7.6 阶段 F：先干运行，再接真实机器人

推荐验收序列：

```text
URDF/FK 单元测试
  -> 静态目标 IK 测试
  -> 连续轨迹 IK 测试
  -> joint-limit / unreachable-target 测试
  -> Meshcat 可视化 VR 测试
  -> 输出 adapter 的 log-only / dry-run
  -> 低速、空载、受保护的硬件测试
  -> 再讨论现场 armed 测试
```

每一阶段记录：目标末端位姿、FK 实际位姿、position/orientation error、每关节最大步长、关节极限距离、
求解时间 P50/P95/P99、丢帧与网络 packet age。

## 8. 复现时推荐的安全与可靠性边界

Pink 所有输出都应经过独立机器人边界，而不是直接调用底层电机接口：

```text
Pink joint target
  -> finite check
  -> joint name/order conversion
  -> joint limit check
  -> initial pose delta policy
  -> per-step velocity / delta limit
  -> target stale timeout
  -> command sequence + timestamp
  -> transport
  -> robot-side watchdog / executor
```

建议新系统至少具备：

- 物理急停和现场监控；
- 真实 State feedback，不能只信命令 echo；
- command timeout/watchdog；
- 单一控制权；
- 可记录和回放的 command/state 日志；
- 明确的 dry-run；
- 自碰与环境碰撞策略。若 Pink 未配置 collision/barrier，不能称为已有碰撞保护。

## 9. 当前项目的参数调节指南

| 参数 | 调大后的主要效果 | 常见代价 |
|---|---|---|
| `position_cost` | 更积极追手的位置 | 可能使姿态和冗余关节更难兼顾 |
| `orientation_cost` | 更积极跟随手腕方向 | 增大腕部/交叉关节动作，可能影响位置跟踪 |
| `posture_cost` | 更偏向 home/舒适姿态 | 过大时末端可能不愿追目标 |
| `joint_motion_cost` | 每周期关节变化更小 | 动作更慢、跟手性下降 |
| `target_max_speed` | 允许更快末端目标变化 | 更容易造成大关节速度与跳跃 |
| `scale` | VR 位移映射更大 | 操作更灵敏，也更容易越界 |
| `tcp_control_offset` | 控制工具尖端而非 link 原点 | 设错轴或长度会让目标偏移 |
| `frequency` | 更高 IK/发布上限 | 计算、可视化和下游通信可能跟不上 |

修改一个参数后，应优先在离线和 Meshcat 中对比：末端误差、关节速度、关节限位距离和求解耗时，而不是
只凭“看起来更灵敏”判断。

## 10. 复现检查清单

```text
[ ] URDF 与实物 joint axis、单位、limit、TCP 一致。
[ ] 所有 joint/frame 都按 name 验证，而非靠数组位置猜测。
[ ] Pink 在当前末端 target 下零运动稳定。
[ ] 单轴位置和单轴姿态测试通过。
[ ] 未启用的 Barrier/碰撞功能没有被误认为已经存在。
[ ] VR relative reference、axis mapping、scale、tool offset 已标定。
[ ] Pink 输出与机器人 command 的关节顺序、单位完全一致。
[ ] State、command、target 三条链路有独立日志与 sequence/timestamp。
[ ] 同一时刻只有一个控制进程持有机器人 command 权。
[ ] 已先通过 log-only/dry-run，再进行低风险硬件验证。
```

## 11. 最终理解

Pink 是本系统的“运动学求解层”，不是“机器人通信层”也不是“安全层”。本项目真正完成 VR 遥操需要
四个层次一起成立：

```text
输入层：VR UDP 与坐标映射
求解层：Pinocchio + Pink task/QP + joint integration
系统层：target action、State、command、时序与控制权
执行层：Orin executor、机器人驱动、现场安全与急停
```

迁移到其他机械臂时，URDF、frame、joint mapping、home pose、VR frame mapping、TCP offset、State/command
adapter 与安全策略都必须重新验证；只有 Pink task 的基本结构通常可以复用。
