# Agilex 单臂 ACT 异步推理技术方案

## 1. 目标与结论

目标是基于 `src/lerobot/async_inference`，为当前 Agilex 单臂 ACT 部署增加一条可维护的异步推理路径，同时不破坏已经可用的同步脚本：

- [run_act_single_arm_infer.py](/home/agilex/cqy/flexible_lerobot/my_devs/train/act/agilex/run_act_single_arm_infer.py)

本方案的最终结论是：

1. `v1` 只交付“ACT chunked async”能力，也就是与你当前第一条同步命令等价的那类运行方式。
2. `v1` 不修改 `src/lerobot/async_inference` 的协议和主干逻辑，不扩展 `RemotePolicyConfig`。
3. `v1` 新增一个 Agilex 单臂包装机器人和一个薄入口脚本，复用现有 `PolicyServer` / `RobotClient`。
4. 你当前第二条命令里的 `--policy-n-action-steps 1 --policy-temporal-ensemble-coeff 0.01` 不在 `v1` 内做“表面兼容”。
5. 如果后续必须兼容第二条命令，优先走“ACT-aware overlap aggregator”这条 `phase 2` 路线，而不是把服务端退化成每步只回 1 个动作的近同步模式。

## 2. 当前系统的真实边界

### 2.1 同步脚本已经解决了什么

当前同步脚本的价值不只是“能推理”，而是已经固化了一套正确的单臂部署契约：

- 观测来源于 `AgileXRosBridge`
- 从双臂实时状态中裁出目标臂 7 维状态
- 只保留 `camera_front` 和目标臂侧相机
- 策略输出为目标臂 7 维动作
- 下发前把另一臂补成“当前实时位姿”
- 支持 `passive_follow` 和 `command_master`

这意味着异步方案的关键不是重新设计 transport，而是把这套单臂语义以一个稳定边界封装起来。

### 2.2 `async_inference` 当前天然支持什么

现有异步主干的职责很清楚：

- [policy_server.py](/home/agilex/cqy/flexible_lerobot/src/lerobot/async_inference/policy_server.py)
  - 加载策略
  - 做 pre/postprocess
  - 调用 `predict_action_chunk()`
  - 返回一段带 timestep 的动作 chunk
- [robot_client.py](/home/agilex/cqy/flexible_lerobot/src/lerobot/async_inference/robot_client.py)
  - 维护动作队列
  - 根据 `chunk_size_threshold` 补发观测
  - 对重叠 timestep 做 chunk overlap aggregation
  - 按固定频率执行动作

它天然适合的是“chunk 级别规划 + client 侧动作队列”的异步模式。

### 2.3 为什么 `AgileXRobot` 不能直接拿来接

现有 [agilex.py](/home/agilex/cqy/flexible_lerobot/src/lerobot/robots/agilex/agilex.py) 暴露的是双臂整机契约：

- `observation_features`: 14 维关节 + 3 路相机
- `action_features`: 14 维动作

而你当前单臂 checkpoint 要求的是：

- `observation.state`: 7 维
- 图像输入: `camera_front + camera_left/right`
- `action`: 7 维

如果直接把 `AgileXRobot` 接给 `RobotClient`，schema 不匹配会污染：

- client 侧的 `lerobot_features`
- server 侧的预处理输入
- 动作维度校验
- inactive arm 保持逻辑

因此，异步方案必须先解决 robot 边界。

### 2.4 两条同步命令的语义并不一样

你现在可运行的两条同步命令，语义并不只是参数不同。

第一条命令对应的是“普通 ACT chunk 模式”：

- checkpoint 当前配置下 `chunk_size = 100`
- `n_action_steps = 100`
- `temporal_ensemble_coeff = null`
- `select_action()` 的实际行为是“内部缓存 100 步动作，然后逐步消费”

第二条命令对应的是“逐步 temporal ensemble 模式”：

- 运行时把 `n_action_steps` 改成 `1`
- 同时启用 `temporal_ensemble_coeff = 0.01`
- `select_action()` 每步都会调用 `predict_action_chunk()`
- 当前步动作来自 `ACTTemporalEnsembler`

而现有 async server 走的是 `predict_action_chunk()`，不是 `select_action()`。这就是为什么第二条命令不能在 `v1` 里宣称“天然兼容”。

## 3. 设计原则

本方案采用以下原则：

1. 不改现有同步脚本，让它继续作为稳定回退路径。
2. 不复制一套 Agilex 专用异步框架，优先复用通用 `PolicyServer` / `RobotClient`。
3. 不在 `v1` 修改 `RemotePolicyConfig` 或 gRPC 协议，只在入口脚本做参数映射和 guardrail。
4. 单臂观测裁剪和 inactive arm hold-current 语义必须下沉到机器人包装层。
5. `v1` 只支持语义清晰、可测试的能力，不做“表面兼容、实际失真”的参数映射。

## 4. 推荐架构

### 4.1 组件划分

推荐采用三层结构：

1. 通用异步主干
   - 直接复用 [policy_server.py](/home/agilex/cqy/flexible_lerobot/src/lerobot/async_inference/policy_server.py)
   - 直接复用 [robot_client.py](/home/agilex/cqy/flexible_lerobot/src/lerobot/async_inference/robot_client.py)
2. Agilex 单臂包装层
   - `my_devs/async_act/config_single_arm_agilex_robot.py`
   - `my_devs/async_act/single_arm_agilex_robot.py`
3. 薄入口与公共校验层
   - `my_devs/async_act/run_async_act_single_arm.py`
   - `my_devs/async_act/common.py`

### 4.2 为什么 `v1` 不改 async 协议

`v1` 的目标只是覆盖 chunked async 模式。对这个目标来说：

- `policy_path` 仍然由现有握手字段传给 server
- `actions_per_chunk` 仍然由现有字段表达
- `policy_n_action_steps` 可以在入口层解释成 `actions_per_chunk`
- `policy_temporal_ensemble_coeff` 在入口层直接拒绝

所以 `v1` 不需要扩展：

- `RemotePolicyConfig`
- `PolicyServer.SendPolicyInstructions()`
- gRPC message schema

这能显著压缩改动面，让 `v1` 更容易稳定。

### 4.3 工厂反射闭环约束

`v1` 选择把包装机器人放在 `my_devs/async_act` 下时，必须同时满足现有工厂链路：

- `RobotClient`
- `make_robot_from_config()`
- `make_device_from_device_class()`

能够成功实例化该机器人。为此需要明确以下实现约束：

1. `my_devs/async_act` 必须成为可导入包，也就是要有 `__init__.py`。
2. 配置类必须继承 `RobotConfig` 并通过 `RobotConfig.register_subclass(...)` 注册。
3. 文件命名要与 `make_device_from_device_class()` 的搜索规则兼容。
   - 推荐使用 `config_single_arm_agilex_robot.py`
   - 推荐使用 `single_arm_agilex_robot.py`
4. `my_devs/async_act/__init__.py` 需要显式导出：
   - `SingleArmAgileXRobotConfig`
   - `SingleArmAgileXRobot`

这样即使未来不改核心 robot factory，`RobotClient` 也能通过现有反射路径正常拉起单臂包装机器人。

## 5. `SingleArmAgileXRobot` 的边界设计

### 5.1 建议位置

初版建议放在实验目录，而不是直接进入核心包：

- `my_devs/async_act/config_single_arm_agilex_robot.py`
- `my_devs/async_act/single_arm_agilex_robot.py`
- `my_devs/async_act/__init__.py`

原因：

- 当前需求是面向单一部署场景的工程化适配
- 先在 `my_devs` 验证，避免把实验接口过早固化进 `src`
- 等 `v1` 真机验证稳定后，再评估是否迁入 `src/lerobot/robots/agilex/`

这里要特别强调：初版虽然放在 `my_devs/async_act`，但不能只是“文件放过去”。它必须是一个能被工厂链路 import 和实例化的包内实现，否则 `RobotClient` 在启动时就会失败。

### 5.2 对外观测契约

包装机器人对 `RobotClient` 暴露的观测，只保留单臂需要的部分。

右臂示例：

- `right_joint0.pos` 到 `right_joint6.pos`
- `camera_front`
- `camera_right`

左臂同理：

- `left_joint0.pos` 到 `left_joint6.pos`
- `camera_front`
- `camera_left`

这样 `map_robot_keys_to_lerobot_features()` 会自然生成与单臂 checkpoint 对齐的 `lerobot_features`，server 侧不需要知道 Agilex 是双臂机器。

### 5.3 对外动作契约

包装机器人对 `RobotClient` 暴露的动作，只保留目标臂 7 维。

`send_action()` 内部执行以下逻辑：

1. 读取当前整机实时状态
2. 将目标臂 7 维动作写入对应臂
3. 将另一臂填充为当前实时位姿
4. 仅在 `command_master` 模式下真实发布
5. 在发送前做 inactive arm 一致性校验

这与当前同步脚本的真实执行语义完全一致，而且职责边界正确：

- 策略只负责目标臂
- 机器人包装层负责整机安全执行语义

## 6. 入口脚本与公共模块

### 6.1 薄入口脚本

推荐新增：

- `my_devs/async_act/run_async_act_single_arm.py`

采用：

- `server` 子命令
- `client` 子命令

其中：

- `server` 只是对 `PolicyServer` 的薄包装
- `client` 负责：
  - 构造 `SingleArmAgileXRobot`
  - 做 checkpoint 契约校验
  - 参数映射
  - 启动 `RobotClient`
  - 支持 `run_time_s`

### 6.2 公共辅助模块

推荐新增：

- `my_devs/async_act/common.py`

这里优先抽出当前同步脚本中真正值得共享的逻辑，例如：

- checkpoint 契约校验
- arm/camera 常量
- 参数 guardrail
- 运行时摘要输出

这样可以避免将来同步脚本与异步脚本各自维护一套相同的校验规则。

## 7. `v1` 参数映射策略

### 7.1 `actions_per_chunk`

在异步模式里，真正控制“每次 server 返回多少动作”的参数应该是：

- `actions_per_chunk`

`v1` 建议规则：

1. 如果用户显式传 `--actions-per-chunk`，直接使用。
2. 如果没传，则默认取 checkpoint 的 `chunk_size`。
3. 对你当前 checkpoint，默认值就是 `100`，这样最贴近现有同步命令 A 的初始语义。

同时建议在调参阶段重点比较：

- `100`
- `50`
- `30`

原因：

- `100` 更接近当前同步行为
- `30` / `50` 往往更有实时性

### 7.2 `policy_n_action_steps`

`v1` 中保留这个参数，但只把它当“迁移兼容 alias”，而不是 ACT 内部语义。

规则：

1. 如果设置了 `policy_temporal_ensemble_coeff`，则这个 alias 失效并直接报错。
2. 如果未显式传 `actions_per_chunk`，则把 `policy_n_action_steps` 映射为 `actions_per_chunk`。
3. 如果两者都传了，则以 `actions_per_chunk` 为准，并在日志中明确打印最终生效值。

换句话说，`v1` 不再把 `policy_n_action_steps` 理解为“策略内部缓存长度”，而是把它解释为“异步 chunk 返回长度”的兼容写法。

### 7.3 `policy_temporal_ensemble_coeff`

`v1` 一旦检测到非空的 `policy_temporal_ensemble_coeff`，直接 fail fast。

错误信息必须明确说明：

1. 当前异步路径使用的是 `predict_action_chunk()`
2. 这不等价于同步脚本的 `select_action() + ACTTemporalEnsembler`
3. 如需该语义，请继续使用同步脚本
4. 该需求被列入 `phase 2` 预研项

### 7.4 `chunk_size_threshold`

推荐默认值：

- `0.5`

原因：

- 与当前 `RobotClientConfig` 默认值一致
- 不会过早放大 server 推理频率
- 足以避免大多数明显断粮

### 7.5 `aggregate_fn_name`

推荐默认：

- `weighted_average`

但要在文档中明确写明：

- 它只是 chunk overlap aggregator
- 它不是 ACT temporal ensemble

## 8. 对你当前两条同步命令的支持结论

### 8.1 第一条命令

当前命令特征：

- 不启用 temporal ensemble
- 实际行为是普通 ACT chunk 缓存

异步映射：

- `client` 走 `SingleArmAgileXRobot`
- `server` 继续走 `PolicyServer.predict_action_chunk()`
- `actions_per_chunk` 默认取 `checkpoint.chunk_size`

支持结论：

- `v1` 正式支持

### 8.2 第二条命令

当前命令特征：

- `policy_n_action_steps=1`
- `policy_temporal_ensemble_coeff=0.01`
- 语义依赖 `select_action()` 内部的 `ACTTemporalEnsembler`

异步映射难点：

- 当前 async server 不走 `select_action()`
- 当前 client 的 `aggregate_fn(old, new)` 只有无状态二元融合能力
- 直接用 `weighted_average` 替代会产生语义失真

支持结论：

- `v1` 不支持 exact 语义

## 9. `phase 2` 的正确技术方向

如果后续必须兼容第二条命令，不建议采用“server 侧每步 `select_action()`，只返回 1 个动作”的路线。那样会把异步系统重新退化成近同步系统，失去 chunked async 的主要收益。

更合理的 `phase 2` 路线是：

1. 仍然保留 server 侧 `predict_action_chunk()`
2. 客户端引入 ACT-aware overlap aggregator
3. 对每个未来 timestep 维护：
   - 当前聚合后的动作
   - `ensemble_count`
   - 对应的权重累计
4. 让 chunk overlap 的在线更新公式与 `ACTTemporalEnsembler` 数学一致
5. 在需要 exact 贴近同步 temporal ensemble 时，把观测刷新频率提高到“每步都发”

这条路线的好处是：

- 仍然保留 chunk transmission 和 queue 机制
- 更接近 ACT 原始 temporal ensemble 数学
- 不必把 server 改成每步只返回 1 个动作

但它明显超出 `v1` 范围，因为它需要：

- 新的 client 侧状态管理
- 专门的单元测试
- 新的真机验证矩阵

## 10. 风险与缓解

### 10.1 Schema 不匹配

风险：

- 单臂 checkpoint 与包装机器人输出的 key / shape 不一致

缓解：

- 启动前做 checkpoint 契约校验
- 为包装机器人补单元测试

### 10.2 Queue underrun

风险：

- server 推理或网络时延导致动作队列见底

缓解：

- 默认 `chunk_size_threshold = 0.5`
- 首轮调参优先比较 `actions_per_chunk = 30 / 50 / 100`
- 保留 shadow 模式先验证

### 10.3 Inactive arm 漂移

风险：

- 非目标臂在真实下发时被错误带动

缓解：

- 仅在包装机器人层集中做 merge
- 发送前强制校验 inactive arm 是否等于当前实时位姿

### 10.4 语义误解

风险：

- 用户误以为 `temporal_ensemble_coeff` 在异步 `v1` 中已生效

缓解：

- 入口脚本直接拒绝该参数
- 文档明确写出 `v1` / `phase 2` 边界

## 11. 最终技术决策

本方案收敛后的正式决策如下：

1. `v1` 采用“`SingleArmAgileXRobot` 包装层 + 通用 `PolicyServer` / `RobotClient` + 薄入口脚本”的结构。
2. `v1` 不修改 `src/lerobot/async_inference` 协议与核心实现。
3. `v1` 只正式支持 ACT chunked async，对应你当前第一条同步命令。
4. `policy_n_action_steps` 在 `v1` 里仅作为 `actions_per_chunk` 的兼容 alias。
5. `policy_temporal_ensemble_coeff` 在 `v1` 中直接拒绝，不做表面兼容。
6. 如果后续必须兼容第二条命令，优先进入 `phase 2` 的 ACT-aware overlap aggregator 预研，而不是把系统改成每步单动作返回的近同步模式。
