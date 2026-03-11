# AgileX 接入 LeRobot 的架构师核对版最终报告

## 1. 结论先行

基于以下三类证据：

- 你的真实运行说明：[my_devs/docs/add_robot/agilex.md](/home/agilex/cqy/flexible_lerobot/my_devs/docs/add_robot/agilex.md)
- 你们当前的采集/回放脚本：
  - [collect_data.py](/home/agilex/cqy/flexible_lerobot/my_devs/cobot_magic/collect_data/collect_data.py)
  - [replay_data.py](/home/agilex/cqy/flexible_lerobot/my_devs/cobot_magic/collect_data/replay_data.py)
- LeRobot 当前主流程与抽象：
  - [robot.py](/home/agilex/cqy/flexible_lerobot/src/lerobot/robots/robot.py)
  - [teleoperator.py](/home/agilex/cqy/flexible_lerobot/src/lerobot/teleoperators/teleoperator.py)
  - [lerobot_teleoperate.py](/home/agilex/cqy/flexible_lerobot/src/lerobot/scripts/lerobot_teleoperate.py)
  - [lerobot_record.py](/home/agilex/cqy/flexible_lerobot/src/lerobot/scripts/lerobot_record.py)
  - [lerobot_replay.py](/home/agilex/cqy/flexible_lerobot/src/lerobot/scripts/lerobot_replay.py)
  - [lerobot_calibrate.py](/home/agilex/cqy/flexible_lerobot/src/lerobot/scripts/lerobot_calibrate.py)

我给出的最终判断是：

1. 你不需要重写“物理上的”遥操作链路。
   - 现有 AgileX ROS 系统已经有主臂、从臂、相机和回放链路。

2. 但如果你要复用 LeRobot 的标准主流程，而不是继续只用你们自己的脚本，就不能只加一个 `Robot` 类。
   - 你仍然需要一个“逻辑上的” teleoperator 适配层。
   - 这个 teleoperator 不是新硬件，而是把现有主臂 ROS topic 包装成 `teleop.get_action()`。

3. 第一版最小闭环应该是：
   - `AgileXRobot`
   - `AgileXTeleoperator`
   - `AgileXRosBridge`
   - 一个最小 ROS 图像接入方案

4. 第一版只承诺四个入口：
   - `lerobot-calibrate`
   - `lerobot-teleoperate`
   - `lerobot-record`
   - `lerobot-replay`

5. 第一版先不要碰：
   - RL
   - async inference
   - `lerobot-eval`
   - `setup-motors`
   - `find-joint-limits`
   - FK/IK/末端空间 replay
   - 深度图和底盘动作的完整正式化

一句话总结：

这件事不是“零适配”，而是“薄适配”。你不需要重做主从控制，但你需要把现有 ROS 主从链路翻译成 LeRobot 的 `Robot + Teleoperator` 抽象。

## 1.1 实现前必须先确认的 4 件事

这 4 件事不确认，后面的实现都会带着假设：

1. `collect_data.py` 实际采集时，action 到底来自：
   - `/master/*`
   - 还是 `/puppet/*`

2. `mode := 1` 下，真正吃控制命令的入口到底是：
   - `/master/joint_states`
   - 还是 remap 后的 `/master/joint_left` / `/master/joint_right`

3. 左右臂的 14 维拼接顺序是否和现有 HDF5 完全一致。

4. 第一版是否要保留底盘动作。
   - 如果不要，必须明确这只是“先打通 LeRobot 主流程”，不是“1:1 替代旧回放脚本”。

## 2. 当前系统真实在做什么

### 2.1 采集模式

[agilex.md](/home/agilex/cqy/flexible_lerobot/my_devs/docs/add_robot/agilex.md) 给出的采集启动顺序是：

1. `roscore`
2. `roslaunch piper start_ms_piper.launch mode:=0 auto_enable:=false`
3. `roslaunch astra_camera multi_camera.launch`
4. 运行 [collect_data.py](/home/agilex/cqy/flexible_lerobot/my_devs/cobot_magic/collect_data/collect_data.py)

[collect_data.py](/home/agilex/cqy/flexible_lerobot/my_devs/cobot_magic/collect_data/collect_data.py) 的核心语义是：

- 观测：
  - 从臂 `qpos`
  - 从臂 `qvel`
  - 从臂 `effort`
  - 三路相机图像
  - 可选底盘速度
- 动作：
  - 主臂左 7 维 + 主臂右 7 维

也就是说，现有采集链路的真实语义是：

- `observation = puppet/follower state + images + optional base`
- `action = master/leader joint positions`

但这里要特别加一条保留：

- [collect_data.py](/home/agilex/cqy/flexible_lerobot/my_devs/cobot_magic/collect_data/collect_data.py) 的默认参数里，`master_arm_left_topic` / `master_arm_right_topic` 默认值写成了 `/puppet/joint_left` / `/puppet/joint_right`
- 这和“action 来自主臂”这件事在字面上是冲突的
- 因此实现前必须核实你们现场真实运行时是否总是通过外部参数把它改成 `/master/...`

这和 LeRobot 的 `teleop.get_action() -> robot.send_action()` 主流程在语义上是能对齐的。

### 2.2 回放模式

[agilex.md](/home/agilex/cqy/flexible_lerobot/my_devs/docs/add_robot/agilex.md) 给出的回放启动顺序是：

1. `roscore`
2. `roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=true`
3. `roslaunch astra_camera multi_camera.launch`
4. 运行 [replay_data.py](/home/agilex/cqy/flexible_lerobot/my_devs/cobot_magic/collect_data/replay_data.py)

[replay_data.py](/home/agilex/cqy/flexible_lerobot/my_devs/cobot_magic/collect_data/replay_data.py) 会：

- 读取 HDF5 中的 14 维手臂动作
- 把左 7 维和右 7 维发布到：
  - `/master/joint_left`
  - `/master/joint_right`
- 把底盘动作发布到 `/cmd_vel`
- 同时重发布从臂和图像话题用于观察

这说明回放模式下，LeRobot 的 `robot.send_action()` 需要落到 ROS 控制发布，而不是只读状态。

## 3. 需要特别说明的一个矛盾点

Piper README 和节点实现对 mode 0 / mode 1 的文字说明并不完全一致：

- README 一处写的是 mode 1 用 `/master/joint_left`、`/master/joint_right` 做外部控制
- 但节点代码 [piper_start_ms_node.py](/home/agilex/cqy/flexible_lerobot/my_devs/cobot_magic/Piper_ros_private-ros-noetic/src/piper/scripts/piper_start_ms_node.py) 里，`mode == 1` 时明确订阅的是 `/master/joint_states`
- README 又展示了从 `/master/joint_states` 到 `/master/joint_left/right` 的 remap

所以最终实现前，必须把下面这件事作为第一优先级真机核对项：

- 你们实机在 `mode := 1` 下，真正接受控制的最终 ROS 入口到底是：
  - `/master/joint_states`
  - 还是 remap 后的 `/master/joint_left` + `/master/joint_right`

这一点不核实，`send_action()` 的实现就会建立在猜测上。

## 4. 为什么不能只加一个 Robot

LeRobot 当前主流程不是“Robot 自己产生动作”。

从源码看：

- [lerobot_teleoperate.py](/home/agilex/cqy/flexible_lerobot/src/lerobot/scripts/lerobot_teleoperate.py) 要求 `teleop` 和 `robot` 同时存在
- [lerobot_record.py](/home/agilex/cqy/flexible_lerobot/src/lerobot/scripts/lerobot_record.py) 在没有 `policy` 时也要求有 `teleop`
- [teleoperator.py](/home/agilex/cqy/flexible_lerobot/src/lerobot/teleoperators/teleoperator.py) 明确定义了 `get_action()`

所以：

- 物理层面：你不需要新做一个遥操作硬件
- 软件抽象层面：你仍然需要一个 teleoperator 适配器

这个适配器的工作非常薄：

- 订阅主臂 ROS joint 话题
- 输出和 `AgileXRobot.action_features` 完全一致的 action 字典

因此，我建议的对象拆分是：

- `AgileXRobot`
  - 从臂状态
  - 图像
  - 回放命令
- `AgileXTeleoperator`
  - 主臂动作读取

## 5. 最终推荐架构

### 5.1 第一版走 in-tree

直接做 in-tree 接入，不先做第三方插件。

原因：

- 你的目标是尽快把这台真机接进当前仓库
- 当前要调的不是分发机制，而是 ROS topic、schema 和生命周期
- in-tree 的调试面更短

推荐目录：

```text
src/lerobot/robots/agilex/
├── __init__.py
├── config_agilex.py
├── agilex.py
└── agilex_ros_bridge.py

src/lerobot/teleoperators/agilex_teleoperator/
├── __init__.py
├── config_agilex_teleoperator.py
└── agilex_teleoperator.py
```

如果你决定第一版同时补 ROS 相机：

```text
src/lerobot/cameras/ros_camera/
├── __init__.py
├── configuration_ros_camera.py
└── ros_camera.py
```

### 5.2 命名建议

为了降低 factory 歧义，我建议类名遵守当前仓库的自动构造约定：

- `AgileXRobotConfig` -> `AgileXRobot`
- `AgileXTeleoperatorConfig` -> `AgileXTeleoperator`

CLI type 我建议用：

- `--robot.type=agilex`
- `--teleop.type=agilex_teleoperator`

如果你更想贴近仓库现有 leader/follower 命名，也可以改成：

- `agilex_follower`
- `agilex_leader`

但要从第一版开始就保持统一，不要文档一套、代码一套。

### 5.3 `AgileXRosBridge` 的职责

它应该负责：

- 建立 ROS 节点连接或复用现有 ROS 环境
- 订阅：
  - 主臂 joint
  - 从臂 joint
  - 三路图像
  - 可选底盘状态
- 发布：
  - 回放时的主臂控制入口
  - 可选底盘 `/cmd_vel`

不要把 ROS topic 读写逻辑散落在 `Robot` 和 `Teleoperator` 里。

## 6. `AgileXRobot` 应该怎么设计

### 6.1 它的职责

- `get_observation()`
  - 返回从臂状态
  - 返回三路图像
  - 可选返回底盘状态
- `send_action()`
  - 在采集模式下：可以是被动 no-op，或者返回实际收到的 action
  - 在回放模式下：真正向 ROS 控制入口发命令

### 6.2 它必须满足的 LeRobot 约束

根据 [robot.py](/home/agilex/cqy/flexible_lerobot/src/lerobot/robots/robot.py) 和 [lerobot_record.py](/home/agilex/cqy/flexible_lerobot/src/lerobot/scripts/lerobot_record.py)：

- 必须先 `super().__init__(config)`
- `observation_features` 和 `action_features` 在未连接时也能取
- `self.cameras` 最好始终存在
  - 就算第一版不走正式 `Camera` 抽象，也至少要保留这个属性
- `get_observation()` 的键必须和 `observation_features` 完全一致
- `send_action()` 的键必须和 `action_features` 完全一致

### 6.3 关于采集模式和回放模式

我建议在 `AgileXRobotConfig` 里显式放一个控制模式字段，例如：

- `control_mode = "passive_follow"`
- `control_mode = "command_slave"`

含义：

- `passive_follow`
  - 用于 `teleoperate` / `record`
  - 主从联动由现有 ROS/Piper 系统负责
  - `robot.send_action()` 不主动重复下发命令
- `command_slave`
  - 用于 `replay`
  - `robot.send_action()` 真正发布 ROS 控制命令

这是这次接入里最重要的设计点。

## 7. `AgileXTeleoperator` 应该怎么设计

它只做一件事：

- 从主臂 joint 话题取数
- 输出一个 14 维动作字典

它不需要：

- 控制物理硬件
- 做复杂 force feedback
- 做独立相机处理

如果当前主臂并不需要额外标定：

- `is_calibrated = True`
- `calibrate() = pass`

是可以接受的。

## 8. 关于相机，我的最终建议

我不建议在最终报告里继续保持模糊。

我的建议是：

1. 如果你的目标只是尽快 bring-up：
   - 可以先把 ROS 图像订阅直接写进 `AgileXRobot`
   - 先不抽 `ros_camera`

2. 如果你的目标是很快进入正式采集：
   - 第一版就把 `ros_camera` 一起补上

原因：

- 当前真实采集链路里，相机不是可选项
- 当前仓库没有现成 ROS camera backend
- 但第一版真正的主阻塞仍然是 joint/action 语义和 lifecycle

因此：

- 快速 bring-up：图像先内嵌到 `Robot`
- 正式落地：尽快把 `ros_camera` 补出来

还有一个实现层约束需要写清楚：

- [lerobot_record.py](/home/agilex/cqy/flexible_lerobot/src/lerobot/scripts/lerobot_record.py) 会直接用 `len(robot.cameras)` 计算 image writer 线程数
- 所以即使第一版先不走正式 `Camera` 抽象，也要保证：
  - `robot.cameras` 存在
  - 图像 observation 和写盘行为不会互相打架

## 9. 第一版实现范围

### 9.1 必做

1. `AgileXRosBridge`
2. `AgileXRobotConfig`
3. `AgileXRobot`
4. `AgileXTeleoperatorConfig`
5. `AgileXTeleoperator`
6. 主流程脚本导入接入：
   - `lerobot_calibrate.py`
   - `lerobot_teleoperate.py`
   - `lerobot_record.py`
   - `lerobot_replay.py`

### 9.2 建议第一版就做

1. 一个最小的 ROS 图像接入方案
2. 一组 robot/teleop 的单测
3. 本地 `record -> replay` 闭环验证

### 9.3 第二阶段再做

1. 底盘动作正式并入 action schema
2. 深度图
3. `lerobot-eval`
4. RL
5. async inference
6. FK / IK / EE replay
7. 老 HDF5 数据直接转 LeRobot dataset 的 converter

这里再明确一句边界：

- 如果第一版不做底盘动作，它就不是对 [replay_data.py](/home/agilex/cqy/flexible_lerobot/my_devs/cobot_magic/collect_data/replay_data.py) 的 1:1 完整替代
- 它的目标只是先打通 LeRobot 的双臂主流程闭环

## 10. 我建议的实施顺序

1. 先起外部 ROS：
   - 按 [agilex.md](/home/agilex/cqy/flexible_lerobot/my_devs/docs/add_robot/agilex.md) 的方式起 mode 0 或 mode 1

2. 先做 topic 盘点：
   - `rostopic list`
   - `rostopic echo /master/joint_states`
   - `rostopic echo /master/joint_left`
   - `rostopic echo /master/joint_right`
   - `rostopic echo /puppet/joint_left`
   - `rostopic echo /puppet/joint_right`

3. 先写 bridge，再写 robot/teleop

4. 先打通对象级 smoke test：
   - `AgileXRobot(...).observation_features`
   - `AgileXTeleoperator(...).action_features`

5. 再跑四个主流程：
   - `calibrate`
   - `teleoperate`
   - `record`
   - `replay`

6. `record` 和 `replay` 必须共用同一个本地 `--dataset.root`

## 11. 我最终给你的执行建议

如果你现在就准备开始落代码，我建议你按下面的决定执行：

1. 路线选 in-tree。
2. 第一版最小集合选：
   - `AgileXRobot`
   - `AgileXTeleoperator`
   - `AgileXRosBridge`
3. 第一版把采集和回放分成两种控制模式。
4. 第一版只打通：
   - `calibrate`
   - `teleoperate`
   - `record`
   - `replay`
5. 相机先采用“能跑通优先”的策略：
   - bring-up 阶段可先内嵌到 `Robot`
   - 稳定后尽快补 `ros_camera`

## 12. 最后的判断

“因为机器人已经有主从臂，所以只需要把当前机器人加进仓库，不需要再考虑 teleop” 这句话，只在硬件层面成立，不在 LeRobot 软件抽象层面成立。

更准确的表述应该是：

- 不需要重做物理遥操作系统
- 但需要把现有主臂翻译成一个逻辑上的 `Teleoperator`
- 需要把现有从臂和图像翻译成一个 `Robot`

这就是这次 AgileX 接入的正确边界。
