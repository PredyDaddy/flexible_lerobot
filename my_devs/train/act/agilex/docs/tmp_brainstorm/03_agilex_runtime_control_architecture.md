# Agilex ACT 上机推理临时脑暴 03：ROS 接入与实机控制架构

## 1. 结论先行

从 Agilex 当前接入方式看，实机推理最合理的主链路不是“再走一遍 teleoperator 录制模式”，而是：

1. 使用 `AgileXRobot` 作为唯一的实机桥接对象。
2. 用 `AgileXRobot.get_observation()` 读取从臂状态和三路相机。
3. 将观测送入 ACT policy 做推理。
4. 将 policy 输出的 14 维动作通过 `AgileXRobot.send_action()` 发布到命令 topic。

在这个闭环里，`AgileXTeleoperator` 不是推理必需件。它属于“读取主臂动作”的遥操作设备，服务于录制链路，不是纯 policy 部署链路的核心依赖。

我认为最终上线方案应该显式区分两种模式：

1. `passive_follow`：只连观测，不发动作，用于 dry-run、readiness 检查、shape 检查、图像和关节验证。
2. `command_master`：连观测也连发布器，真正向 `/master/joint_left` 和 `/master/joint_right` 发动作，用于闭环推理。

这两个模式不应该混在一起。上线流程必须从只读模式逐步进入闭环模式。

## 2. 训练数据链路和推理控制链路的差别

### 2.1 训练数据链路的真实语义

根据现有 Agilex 工作报告，录制时的默认 topic 语义是：

1. `observation.state` 来自 `/puppet/joint_left` 和 `/puppet/joint_right`
2. `action` 来自 `/master/joint_left` 和 `/master/joint_right`
3. 图像来自三路相机 `/camera_f/color/image_raw`、`/camera_l/color/image_raw`、`/camera_r/color/image_raw`

也就是说，训练数据本质上是：

1. 输入：从臂当前状态 + 三路视觉
2. 输出：主臂侧的关节目标

这不是“预测从臂下一时刻状态”，而是“预测应该给控制链路什么动作命令”。

### 2.2 推理控制链路的真实语义

推理时不再有 teleoperator 提供动作标签，policy 自己生成动作。因此链路变成：

1. 从 `AgileXRobot` 读取从臂状态和图像
2. policy 预测 14 维 `action`
3. `AgileXRobot.send_action()` 将动作发到命令 topic

因此，训练链路和推理链路最大的区别是：

1. 训练时 `action` 是外部示教信号
2. 推理时 `action` 是 policy 输出
3. 两者共享同一套 action feature 语义和 topic 终点

这也是为什么推理阶段不能沿用 teleoperator 作为动作来源。teleoperator 会把主臂实时读数重新注入动作通道，破坏 policy 输出在控制链路中的唯一性。

## 3. 上机推理时应如何使用 `AgileXRobot`

### 3.1 `AgileXRobot` 是实机推理的主对象

`AgileXRobot` 当前已经具备推理所需的两个核心能力：

1. `get_observation()`：
   - 读取左右从臂 joint state
   - 读取三路 RGB 图像
2. `send_action()`：
   - 接收 14 维动作字典
   - 在 `command_master` 模式下发布到左右命令 topic

其 feature 语义也和训练数据对齐：

1. 观测状态键名为 `left_joint{i}.pos` / `right_joint{i}.pos`
2. 动作键名也是 `left_joint{i}.pos` / `right_joint{i}.pos`
3. 图像键名为 `camera_front` / `camera_left` / `camera_right`

所以从架构上看，不需要另起一套 Agilex 推理桥。

### 3.2 `AgileXTeleoperator` 在纯推理里不是必需件

`AgileXTeleoperator` 的职责很明确：

1. 订阅 `/master/joint_left`
2. 订阅 `/master/joint_right`
3. 将主臂读数暴露成 action feature

这适用于：

1. 遥操作录制
2. 手动示教
3. 人机切换场景下的参考输入

但不适用于最小闭环推理路径。纯 policy inference 下，如果同时接入 teleoperator，容易造成两个问题：

1. 架构职责不清，动作到底来自 policy 还是来自 teleoperator 会变模糊
2. 后续一旦要加手动接管，接口边界会先天混乱

我的建议是：

1. 第一版纯推理脚本不要依赖 `AgileXTeleoperator`
2. 如果后续要做人工接管或混合控制，再把 teleoperator 作为单独可选模块引入

## 4. `control_mode` 的关键考虑

### 4.1 `passive_follow` 适合只读检查

`AgileXRobotConfig.control_mode` 默认是 `passive_follow`。在这个模式下：

1. 机器人仍会订阅 state topic 和 image topic
2. 不会创建 command publisher
3. `send_action()` 即使被调用，也只会返回动作字典，不会真的发出去

这非常适合以下阶段：

1. dry-run 之外的真机只读联通检查
2. checkpoint / processor / feature 对齐检查
3. 观测刷新率和图像 readiness 检查
4. 推理输出范围观察但不下发

我认为这应该成为 Agilex 上机脚本的默认安全模式。

### 4.2 `command_master` 才是闭环推理模式

当 `control_mode=command_master` 时：

1. `AgileXRobot.connect()` 会要求创建左右 publisher
2. `send_action()` 会通过 bridge 将动作发布到：
   - `command_left_topic`
   - `command_right_topic`

当前默认配置里，这两个 topic 是：

1. `/master/joint_left`
2. `/master/joint_right`

这意味着 policy 输出不是直接发给 `/puppet/*`，而是发给主命令入口。这个设计和录制数据的 action 语义是一致的，但它也引入一个前提：

1. 现场控制系统必须保证“向 `/master/joint_*` 发布关节目标”这件事，会被下游正确消费并驱动从臂

如果这个前提不成立，推理脚本逻辑再正确也不会真实起作用。

### 4.3 `control_mode` 设计上要成为显式 CLI 参数

从架构角度，最终上机脚本最好把 `control_mode` 直接暴露出来，而不是隐藏在配置文件里。原因很简单：

1. 只读联调和真实闭环是两种风险等级完全不同的运行方式
2. 让操作者在命令行明确写出模式，能减少误发动作
3. 也方便日志里明确记录这次运行是不是发过真实控制命令

## 5. Topic 绑定与动作发布路径的关键考虑

### 5.1 当前默认 topic 绑定是可复用的

从代码和工作报告看，当前默认 topic 绑定是：

1. 状态：
   - `/puppet/joint_left`
   - `/puppet/joint_right`
2. 命令：
   - `/master/joint_left`
   - `/master/joint_right`
3. 图像：
   - `/camera_f/color/image_raw`
   - `/camera_l/color/image_raw`
   - `/camera_r/color/image_raw`

对于“训练好的当前模型直接上机”，这套绑定应该优先原样复用，因为它与数据采集链路一致。

### 5.2 真正的风险不在 topic 名，而在 topic 语义闭环是否成立

比起 topic 名字本身，更需要在上线前确认下面三件事：

1. `/puppet/joint_*` 是否稳定反映 follower 真实状态
2. `/master/joint_*` 是否确实是系统接受动作命令的入口
3. 发布到 `/master/joint_*` 后，是否会导致 follower 产生与训练时一致的执行语义

如果第三点没有被现场确认，推理上线风险会很高，因为 policy 学的是“历史数据里的 action 语义”，而不是抽象意义上的 14 维数值。

### 5.3 action 发布路径需要特别强调“14 维完整性”

当前 `publish_action()` 会构造两条 `JointState` 消息：

1. 左臂 7 维位置
2. 右臂 7 维位置

这意味着上机推理不能允许：

1. 只发一半动作
2. 维度错位
3. 左右臂关节顺序和训练时不一致

架构上应该把 joint order 视为硬约束，而不是运行时可随意猜测的参数。

## 6. 三路图像 readiness 的关键考虑

### 6.1 当前实现对图像 readiness 是强依赖

`AgileXRobot.connect()` 调用 `wait_for_ready(..., require_images=True)`。这意味着：

1. 左右 state topic 必须先收到数据
2. 三路 image topic 也必须都收到数据
3. 任一相机无数据，连接阶段就会超时失败

这对稳定性是好事，因为它避免“缺一路图像但仍然盲跑推理”。

### 6.2 这也意味着上线前必须明确三路相机是否都是训练必要输入

既然训练时 checkpoint 使用了三路图像输入，那么推理时强制等待三路 ready 是合理的。  
但这同时要求现场具备：

1. 三个 topic 都在线
2. 编码为 `rgb8` 或 `bgr8`
3. 图像分辨率与训练配置相容

任何一路不满足，推理都不应自动降级为“两路图像”或“单路图像”，因为那会直接破坏 checkpoint 输入契约。

### 6.3 readiness 检查应输出“缺哪个 topic”

当前 bridge 已经会在超时后列出缺失 topic。这个行为应该在最终上线文档和脚本日志里被保留，因为它是现场排障的第一入口。

## 7. 我建议的上线顺序

我认为最合理的上线顺序应该是四段式，而不是直接运行闭环推理。

### 7.1 第一阶段：纯 dry-run

目标：

1. 不连 ROS
2. 不加载真实机器人
3. 只验证参数、checkpoint 路径、配置选择、运行模式

要确认的点：

1. 使用的是正确 checkpoint
2. 推理模式是 `passive_follow` 还是 `command_master`
3. 设备选择、日志路径、任务文本、运行时覆盖项都被正确解析

### 7.2 第二阶段：观测检查

目标：

1. 连接 Agilex
2. 只读取 state 和 images
3. 不发布动作

推荐方式：

1. `control_mode=passive_follow`
2. 循环打印或记录：
   - 14 维状态是否连续刷新
   - 三路图像是否都有数据
   - 图像尺寸和编码是否符合预期

这一阶段本质是在验证“训练输入是否在实机上真实可复现”。

### 7.3 第三阶段：零动作 / 只读推理检查

目标：

1. 连观测
2. 跑 policy 前向
3. 观察输出，但先不发真实控制

建议拆成两个小步骤：

1. 先在 `passive_follow` 下让脚本正常跑推理，打印动作范围、均值、是否存在 NaN/Inf
2. 再测试“假发送”链路，即调用统一的发送分支，但依然保持 `passive_follow`，确保不会真发命令

这样能验证：

1. checkpoint、processor、policy 与 Agilex observation 对齐
2. 推理循环本身没有 runtime 崩溃
3. 发送接口在架构上已经打通，但还没有风险动作

### 7.4 第四阶段：真实闭环

目标：

1. 切到 `command_master`
2. 真正向 `/master/joint_*` 发布 policy 输出
3. 在短时、低风险工况下验证 follower 响应

建议控制策略：

1. 首次闭环只跑很短时间窗口
2. 必须有人在现场可立刻接管或急停
3. 首轮只做低幅度、低速度、易撤销动作
4. 先验证“能动且方向正确”，再验证任务成功率

## 8. 架构建议：最终方案的模块边界

从 ROS 接入和控制链路角度，我建议最终上机方案采用下面的职责划分：

1. 推理主脚本
   - 负责参数解析、checkpoint 加载、policy 推理循环、日志
2. `AgileXRobot`
   - 负责观测采集和动作发布
3. `AgileXTeleoperator`
   - 暂不进入第一版纯推理闭环，只在未来的人机接管或示教场景再纳入
4. 运行模式开关
   - 显式区分 `passive_follow` 与 `command_master`
5. 上线前检查
   - 独立的 readiness / topic / observation 自检阶段

这样做的好处是：

1. 复用当前仓库已存在的 Agilex 接入层，不重新造轮子
2. 保持训练数据语义和推理动作语义一致
3. 将安全风险集中在 `control_mode` 切换点，便于运维和排障
4. 让后续扩展人工接管、录验证数据、回放对比时仍有清晰边界

## 9. 给收敛阶段的建议

我建议后续收敛方案重点确认以下问题：

1. 最终闭环脚本是否默认 `passive_follow`，并要求显式参数才能进入 `command_master`
2. 是否需要在真实发动作前增加“动作裁剪 / 限幅 / 首帧保持”保护层
3. 是否要提供单独的 observation-only 子命令，便于现场排障
4. 是否要为未来人工接管预留 teleoperator 旁路，但不在第一版启用
5. `/master/joint_*` 的控制语义是否经过现场验证，且与训练数据动作标签一致
