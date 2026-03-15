# Agilex ACT 上机推理脑暴收敛纪要

## 1. 文档目的

本文件用于把第一轮发散式脑暴文档收敛为一组可执行的架构结论，供后续实现团队直接接手。

本纪要综合了以下材料：

1. `my_devs/train/act/agilex/docs/tmp_brainstorm/01_checkpoint_and_runtime_constraints.md`
2. `my_devs/train/act/agilex/docs/tmp_brainstorm/02_so101_script_migration_strategy.md`
3. `my_devs/train/act/agilex/docs/tmp_brainstorm/03_agilex_runtime_control_architecture.md`
4. `my_devs/train/act/agilex/docs/tmp_brainstorm/04_deployment_safety_and_validation_plan.md`
5. `my_devs/docs/add_robot/agilex/工作报告.md`

---

## 2. 发散阶段形成的核心共识

### 2.1 不重写推理主线

四份脑暴文档在这一点上没有分歧：

1. 不能为了 Agilex 再发明一套新的策略推理框架。
2. 应直接复用 `my_devs/train/act/so101/run_act_infer.py` 已经验证过的 LeRobot 推理主线。
3. 应保持以下链路不变：
   - checkpoint config 加载
   - policy 加载
   - checkpoint 自带 pre/post processor 加载
   - `predict_action(...)`
   - `make_robot_action(...)`
   - robot action / observation processor

收敛结论：

Agilex 版本是一次“机器人适配迁移”，不是一次“推理框架重构”。

### 2.2 第一版纯推理只使用 `AgileXRobot`

四份脑暴文档一致认为：

1. 上机推理时应以 `AgileXRobot` 作为唯一实机桥接对象。
2. `AgileXTeleoperator` 属于录制和示教链路，不属于第一版最小推理闭环。
3. 如果后续要做人工接管、人机混控、策略对比，再把 teleoperator 作为第二阶段扩展件引入。

收敛结论：

第一版闭环推理不接 teleoperator，不做混控。

### 2.3 Checkpoint 输入输出契约必须严格服从

脑暴文档对训练产物约束的判断是一致的：

1. 输入固定为 14 维 `observation.state`
2. 输入固定包含三路图像：
   - `observation.images.camera_front`
   - `observation.images.camera_left`
   - `observation.images.camera_right`
3. 图像 shape 固定为 `3 x 480 x 640`
4. 输出固定为 14 维 `action`
5. preprocessor / postprocessor 不应被手工重写或绕过

收敛结论：

第一版方案必须禁止任何“少一路图像也先跑起来”“自己写归一化”“自己拼 state/action 顺序”的实现捷径。

### 2.4 上机必须采用分级模式，而不是单阶段闭环

四份文档都明确反对“一上来就 command_master 闭环发布”的方式。

收敛出的统一阶段划分是：

1. dry-run
2. 只读观测检查
3. 影子推理
4. 受限闭环
5. 正式闭环

收敛结论：

安全分级不是附加项，而是第一版架构的一部分。

### 2.5 ACT 部署参数必须允许运行期覆盖

各份文档都认为，训练时的：

1. `chunk_size=100`
2. `n_action_steps=100`

不应直接等价为首次上线时的默认运行策略。

收敛结论：

第一版必须保留至少以下运行期 override：

1. `policy_device`
2. `policy_n_action_steps`
3. `policy_temporal_ensemble_coeff`

---

## 3. 建议冻结的架构决策

### 3.1 交付形态

建议冻结为双层交付：

1. Python 主脚本
   - 承载所有真实推理逻辑
2. Shell 包装脚本
   - 承载环境、默认 checkpoint、常用 topic、日志落盘入口

同时补一份面向现场执行人员的运行文档。

### 3.2 推理脚本结构

建议冻结如下主流程：

1. 解析 CLI / 环境变量
2. 校验 checkpoint 契约是否符合 Agilex 当前 schema
3. 解析 device 与 ACT runtime override
4. 构造 `AgileXRobotConfig`
5. 加载 policy、preprocessor、postprocessor
6. 构造 robot feature bridge
7. dry-run 输出完整摘要
8. 连接机器人并执行 readiness 检查
9. 进入观测读取 / 推理 / 动作发送循环
10. 记录周期日志与异常日志
11. 退出时断开 robot

### 3.3 运行模式

建议冻结如下策略：

1. `control_mode=passive_follow` 作为默认安全模式
2. 只有显式指定 `command_master` 才允许真实闭环
3. 第一版不做隐式自动切换

### 3.4 日志策略

建议冻结如下日志要求：

1. 启动日志
   - checkpoint、device、topic、control_mode、ACT 参数摘要
2. readiness 日志
   - state / image ready 状态
3. 周期日志
   - step、elapsed、loop_dt、effective_fps、action 摘要
4. 异常日志
   - 区分 checkpoint、device、topic、image、policy、action 非法等错误类型
5. 日志落盘
   - 与训练脚本风格一致，单独输出到仓库根目录 `logs/`

---

## 4. 仍需人工确认的关键假设

这些问题不是实现细节，而是上线前必须确认的前提。

### 4.1 动作语义是否与命令 topic 完全一致

当前最高优先级开放问题：

1. 训练数据里的 `action` 来自主臂 `master`
2. 推理阶段 `AgileXRobot.send_action()` 也会向 `/master/joint_left` 与 `/master/joint_right` 发布

必须确认：

1. 这个 14 维动作是否就是现场控制链路期望接收的绝对关节目标
2. 发布到 `/master/joint_*` 是否会产生与训练时一致的控制语义

如果这个假设不成立，第一版闭环不得上线。

### 4.2 双臂关节顺序是否与训练时完全一致

虽然 shape 已对齐，但仍要确认：

1. 左右臂 joint order
2. `left_joint{i}.pos` / `right_joint{i}.pos` 的实际顺序
3. 训练数据、processor、ROS bridge 三者之间是否没有置换

### 4.3 三路相机是否稳定可用

必须确认：

1. 三路图像 topic 长期在线
2. 编码为 `rgb8` 或 `bgr8`
3. 实际分辨率能稳定对齐到训练期 `640 x 480`

### 4.4 首版是否需要动作保护层

脑暴阶段对这一点的方向一致，但尚未冻结实现策略：

1. 是否增加动作限幅
2. 是否增加首帧保持
3. 是否增加动作变化率保护

这可以进入实现前的专项设计，不建议在未确认控制语义前抢先实现。

---

## 5. 不进入第一版的内容

为避免范围失控，以下内容建议明确排除在第一版外：

1. teleoperator 混控或人工接管
2. 在线可视化面板
3. 自动 rosbag 录制
4. 自动成功率统计
5. 更复杂的性能 profiling
6. 面向多拓扑自动发现 topic 的能力
7. 推理和录制一体化的复合脚本

第一版目标只有一个：

在严格对齐 checkpoint 契约的前提下，把 Agilex ACT 模型稳定、安全地带到实机闭环验证。

---

## 6. 给实现团队的工作包拆解建议

### 6.1 工作包 A：资产与契约校验

产出目标：

1. 明确 checkpoint 契约
2. 明确第一版默认 topic
3. 明确运行模式与设备策略

### 6.2 工作包 B：Python 主脚本

产出目标：

1. 新建 Agilex 专用 ACT 推理主脚本
2. 保持与 `so101/run_act_infer.py` 的主干一致
3. 完成 `AgileXRobotConfig` 适配与 readiness / shadow / closed-loop 逻辑

### 6.3 工作包 C：Shell 包装与日志规范

产出目标：

1. 固定 `lerobot_flex` 环境
2. 固定默认 checkpoint 路径
3. 规范日志命名与输出位置

### 6.4 工作包 D：运行文档

产出目标：

1. dry-run 命令
2. 观测检查命令
3. 影子推理命令
4. 受限闭环命令
5. 故障排查入口

### 6.5 工作包 E：上线前验证

产出目标：

1. 逐阶段通过记录
2. 动作语义确认记录
3. 首次闭环风险评审

---

## 7. 最终收敛结论

本轮脑暴的最终收敛结论如下：

1. Agilex ACT 上机推理应建立在现有 LeRobot 标准推理路径之上，不重写推理框架。
2. 第一版只使用 `AgileXRobot`，不接入 `AgileXTeleoperator`。
3. checkpoint 的 14 维 state、3 路图像、14 维 action 契约必须被严格执行。
4. 运行模式必须显式区分 `passive_follow` 与 `command_master`。
5. 第一版必须内建 dry-run、只读观测、影子推理、受限闭环四级能力。
6. `/master/joint_*` 的动作语义一致性是上线前最高优先级确认项。
7. 推荐最终交付为“Python 主脚本 + Shell 包装脚本 + 运行文档 + 日志规范”。
