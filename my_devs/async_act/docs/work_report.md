# Agilex 单臂 ACT 异步推理三阶段工作报告

## 1. 报告概述

本报告总结了 `my_devs/async_act` 方向的三阶段工作成果，覆盖：

1. 第一阶段：异步单臂 ACT 基础链路建设
2. 第二阶段：ACT temporal ensemble 异步兼容
3. 第三阶段：执行端动作平滑与抗抖优化

本轮工作的总体目标是，在不破坏现有同步推理脚本的前提下，基于 `src/lerobot/async_inference` 为 Agilex 单臂 ACT 增加一条可维护、可验证、可逐步扩展的异步部署路径，并在真实控制中进一步降低抖动。

---

## 2. 总体结论

截至当前版本，三阶段目标已完成，形成如下结果：

1. 已完成基于 `src/lerobot/async_inference` 的 Agilex 单臂 ACT 异步推理链路。
2. 已完成单臂 7 维 observation / action 契约封装，保持 inactive arm 的 hold-current 执行语义。
3. 已完成对 ACT temporal ensemble 异步运行模式的兼容，支持：
   - `--policy-n-action-steps 1`
   - `--policy-temporal-ensemble-coeff 0.01`
4. 已完成执行端的可选动作平滑层，实机表现显示抖动有所改善。
5. 相关回归测试当前稳定通过，最新本地结果为 `33 passed`。

---

## 3. 第一阶段：异步单臂 ACT 基础链路建设

### 3.1 阶段目标

第一阶段的目标是建设一个最小可用的 `v1` 异步单臂 ACT 路径，覆盖与你第一条同步命令等价的 chunked async 推理模式。

核心要求包括：

1. 不修改现有同步推理脚本。
2. 不修改 `src/lerobot/async_inference` 协议和主干逻辑。
3. 保持单臂 7 维输入输出契约。
4. 保持另一只手臂在执行时固定为当前实时位姿。
5. 支持 `passive_follow` 与 `command_master` 两种控制模式。

### 3.2 关键设计工作

第一阶段首先完成了技术分析和方案冻结，形成了以下文档：

1. [technical_solution.md](/home/agilex/cqy/flexible_lerobot/my_devs/async_act/docs/technical_solution.md)
2. [technical_plan.md](/home/agilex/cqy/flexible_lerobot/my_devs/async_act/docs/technical_plan.md)

在设计上明确了几个关键原则：

1. 异步路径优先复用现有 `PolicyServer` / `RobotClient`。
2. 单臂语义必须下沉到机器人包装层，而不是塞进策略服务端。
3. `v1` 先支持 chunked async，不对 temporal ensemble 做“表面兼容”。

### 3.3 第一阶段实现内容

第一阶段新增并落地了以下核心文件：

1. [__init__.py](/home/agilex/cqy/flexible_lerobot/my_devs/async_act/__init__.py)
2. [common.py](/home/agilex/cqy/flexible_lerobot/my_devs/async_act/common.py)
3. [config_single_arm_agilex_robot.py](/home/agilex/cqy/flexible_lerobot/my_devs/async_act/config_single_arm_agilex_robot.py)
4. [single_arm_agilex_robot.py](/home/agilex/cqy/flexible_lerobot/my_devs/async_act/single_arm_agilex_robot.py)
5. [run_async_act_single_arm.py](/home/agilex/cqy/flexible_lerobot/my_devs/async_act/run_async_act_single_arm.py)

核心实现点如下：

1. 新增 `SingleArmAgileXRobotConfig`，完成 `RobotConfig.register_subclass(...)` 注册，确保可以被 `RobotClient` 工厂链路正常实例化。
2. 新增 `SingleArmAgileXRobot`，对外暴露单臂 7 维 observation 和 7 维 action。
3. 在 `send_action()` 中读取当前整机位姿，将目标臂动作合并到整机命令，并将 inactive arm 保持为当前位姿。
4. 在 `run_async_act_single_arm.py` 中新增 `server` / `client` 子命令，形成异步 bring-up 入口。
5. 在 `common.py` 中集中实现 checkpoint schema 校验、参数解析、启动 summary 和 guardrail。

### 3.4 第一阶段测试与验证

第一阶段补充了基础测试：

1. [test_single_arm_agilex_robot.py](/home/agilex/cqy/flexible_lerobot/tests/my_devs/test_single_arm_agilex_robot.py)
2. [test_async_act_common.py](/home/agilex/cqy/flexible_lerobot/tests/my_devs/test_async_act_common.py)
3. [test_async_act_runner.py](/home/agilex/cqy/flexible_lerobot/tests/my_devs/test_async_act_runner.py)
4. [test_async_act_robot_client_path.py](/home/agilex/cqy/flexible_lerobot/tests/my_devs/test_async_act_robot_client_path.py)

阶段性验证结果包括：

1. 基础测试通过，阶段稳定结果曾达到 `20 passed, 1 skipped`。
2. 你在真实环境中验证了以下命令链路可以运行：
   - async server
   - async client `--dry-run`
   - `passive_follow`
   - `command_master`

### 3.5 第一阶段产出结论

第一阶段完成后，系统已具备一个可运行的异步单臂 ACT 基线版本，能够覆盖普通 ACT chunk 模式的异步部署需求。

---

## 4. 第二阶段：ACT Temporal Ensemble 异步兼容

### 4.1 阶段目标

第二阶段的目标是支持与你第二条同步命令等价的运行语义，也就是：

```bash
--policy-n-action-steps 1
--policy-temporal-ensemble-coeff 0.01
```

该阶段的关键不是简单接受参数，而是让异步路径在 queue overlap 场景下，以 ACT-aware 的方式对相同 timestep 的重叠动作做在线加权聚合。

### 4.2 核心技术判断

第二阶段分析后确认：

1. 同步脚本的 temporal ensemble 本质上不是“只执行 1 个动作”这么简单。
2. 真正关键的是对不同 observation 时刻预测出的重叠 action chunk，在同一 timestep 上进行指数加权融合。
3. 因此不能只把 `actions_per_chunk` 改成 `1` 来伪装兼容。
4. 更合理的做法是保留服务端返回完整 chunk，由 client 侧对重叠 timestep 做 ACT-aware overlap aggregation。

### 4.3 第二阶段实现内容

第二阶段的主要改动包括：

1. 新增 [temporal_ensemble_client.py](/home/agilex/cqy/flexible_lerobot/my_devs/async_act/temporal_ensemble_client.py)
2. 扩展 [common.py](/home/agilex/cqy/flexible_lerobot/my_devs/async_act/common.py)
3. 扩展 [run_async_act_single_arm.py](/home/agilex/cqy/flexible_lerobot/my_devs/async_act/run_async_act_single_arm.py)

核心实现点如下：

1. 新增 `TemporalEnsembleAggregator`，实现与 ACT `ACTTemporalEnsembler` 语义一致的在线加权更新。
2. 新增 `TemporalEnsembleRobotClient`，在 client 侧覆写 `_aggregate_action_queues()`，对重叠 timestep 的 action 做 ACT-aware 聚合。
3. temporal 模式下强制运行时采用：
   - `actions_per_chunk = checkpoint chunk_size`
   - `chunk_size_threshold = 1.0`
4. temporal 模式下 `aggregate_fn_name` 不再作为实际聚合策略，而是显式提示“忽略该字段，改用 ACT-aware overlap aggregation”。
5. 通过 lazy import 方式避免在模块顶层触发 `grpc` 相关敏感导入。

### 4.4 第二阶段测试与验证

第二阶段新增并更新了以下测试：

1. [test_async_act_temporal_mode.py](/home/agilex/cqy/flexible_lerobot/tests/my_devs/test_async_act_temporal_mode.py)
2. [test_async_act_runner.py](/home/agilex/cqy/flexible_lerobot/tests/my_devs/test_async_act_runner.py)
3. [test_async_act_common.py](/home/agilex/cqy/flexible_lerobot/tests/my_devs/test_async_act_common.py)

关键验证点包括：

1. temporal 模式下 `policy_n_action_steps` 只能为 `None` 或 `1`
2. temporal 模式强制 `actions_per_chunk=chunk_size`
3. temporal 模式强制 `chunk_size_threshold=1.0`
4. overlap aggregator 的在线加权更新与 ACT 语义一致
5. runner 通过 lazy import 保持导入安全

阶段性验证结果包括：

1. 相关测试逐步扩展后，阶段回归达到 `29 passed`。
2. temporal 模式 dry-run 已确认打印：
   - `runtime_mode: act_temporal_ensemble`
   - `actions_per_chunk: 100`
   - `chunk_size_threshold: 1.0`
3. 你在机器上验证了 temporal 模式 client 可以运行。

### 4.5 第二阶段产出结论

第二阶段完成后，异步路径已支持 ACT temporal ensemble 语义，不再只覆盖 chunked async，而是能够兼容第二条同步命令的关键运行模式。

---

## 5. 第三阶段：执行端动作平滑与抗抖优化

### 5.1 阶段背景

在第二阶段完成后，异步 temporal 模式已可运行，但在真实控制中观察到仍存在轻微抖动。

经分析，抖动更可能来自以下因素：

1. policy 输出存在小幅高频波动
2. async client 的控制链路会把这些波动直接传递给关节位置命令
3. 当前命令发布链路没有二次平滑、限幅或固定节拍的本地命令整形层

因此第三阶段的目标是，在不破坏现有架构的前提下，用最小改动增加一个可选的执行端 `action smoother`。

### 5.2 阶段目标

第三阶段要解决的问题是：

1. 在 `command_master` 模式下减少细碎抖动
2. 默认关闭，确保不改变既有行为
3. 同时适用于 chunked 模式和 temporal 模式
4. 不修改训练脚本，不修改 `src/lerobot/async_inference`

### 5.3 第三阶段实现内容

第三阶段主要改动如下：

1. 扩展 [config_single_arm_agilex_robot.py](/home/agilex/cqy/flexible_lerobot/my_devs/async_act/config_single_arm_agilex_robot.py)
2. 扩展 [single_arm_agilex_robot.py](/home/agilex/cqy/flexible_lerobot/my_devs/async_act/single_arm_agilex_robot.py)
3. 扩展 [run_async_act_single_arm.py](/home/agilex/cqy/flexible_lerobot/my_devs/async_act/run_async_act_single_arm.py)

新增两个可选参数：

1. `--action-smoothing-alpha`
   - 含义：低通平滑系数
   - 合法区间：`(0, 1]`
   - 数值越小，动作越稳但响应越钝
2. `--max-joint-step-rad`
   - 含义：单控制周期内每个关节允许的最大变化量
   - 单位：弧度
   - 数值越小，命令越保守

执行逻辑为：

1. 先读取本次 policy 原始输出
2. 若启用 `action_smoothing_alpha`，则对本次目标与上一拍命令做低通融合
3. 若启用 `max_joint_step_rad`，则对当前拍的关节增量再做限幅
4. 最终再与 inactive arm 的 hold-current 逻辑合并后下发

这意味着第三阶段是在 robot 执行边界上增加了一个最小 command shaping 层。

### 5.4 第三阶段测试与验证

第三阶段新增和增强了以下测试点：

1. 默认关闭 smoother 时，行为保持与旧版本一致
2. `action_smoothing_alpha` 会使关节命令向目标逐步逼近
3. `max_joint_step_rad` 会限制单拍最大关节变化
4. smoothing 参数能正确从 CLI 透传到 robot config
5. 非法 smoothing 参数会被配置校验拦截

相关测试文件包括：

1. [test_single_arm_agilex_robot.py](/home/agilex/cqy/flexible_lerobot/tests/my_devs/test_single_arm_agilex_robot.py)
2. [test_async_act_runner.py](/home/agilex/cqy/flexible_lerobot/tests/my_devs/test_async_act_runner.py)

当前最新本地验证结果：

1. `conda run -n lerobot_flex pytest -q ...` 相关回归为 `33 passed`
2. dry-run 已验证 smoothing 参数能正确打印并进入运行时配置
3. 你在真实控制中反馈：加入平滑后，抖动“好了一些”

### 5.5 第三阶段产出结论

第三阶段完成后，异步单臂 ACT 路径不再只是“能跑”，而是具备了一个可调的执行端抗抖层，可用于进一步收敛真实控制表现。

---

## 6. 当前交付物清单

### 6.1 文档

1. [technical_solution.md](/home/agilex/cqy/flexible_lerobot/my_devs/async_act/docs/technical_solution.md)
2. [technical_plan.md](/home/agilex/cqy/flexible_lerobot/my_devs/async_act/docs/technical_plan.md)
3. [work_report.md](/home/agilex/cqy/flexible_lerobot/my_devs/async_act/docs/work_report.md)

### 6.2 代码

1. [__init__.py](/home/agilex/cqy/flexible_lerobot/my_devs/async_act/__init__.py)
2. [common.py](/home/agilex/cqy/flexible_lerobot/my_devs/async_act/common.py)
3. [config_single_arm_agilex_robot.py](/home/agilex/cqy/flexible_lerobot/my_devs/async_act/config_single_arm_agilex_robot.py)
4. [single_arm_agilex_robot.py](/home/agilex/cqy/flexible_lerobot/my_devs/async_act/single_arm_agilex_robot.py)
5. [temporal_ensemble_client.py](/home/agilex/cqy/flexible_lerobot/my_devs/async_act/temporal_ensemble_client.py)
6. [run_async_act_single_arm.py](/home/agilex/cqy/flexible_lerobot/my_devs/async_act/run_async_act_single_arm.py)

### 6.3 测试

1. [test_single_arm_agilex_robot.py](/home/agilex/cqy/flexible_lerobot/tests/my_devs/test_single_arm_agilex_robot.py)
2. [test_async_act_common.py](/home/agilex/cqy/flexible_lerobot/tests/my_devs/test_async_act_common.py)
3. [test_async_act_runner.py](/home/agilex/cqy/flexible_lerobot/tests/my_devs/test_async_act_runner.py)
4. [test_async_act_robot_client_path.py](/home/agilex/cqy/flexible_lerobot/tests/my_devs/test_async_act_robot_client_path.py)
5. [test_async_act_temporal_mode.py](/home/agilex/cqy/flexible_lerobot/tests/my_devs/test_async_act_temporal_mode.py)
6. [test_agilex_single_arm_infer.py](/home/agilex/cqy/flexible_lerobot/tests/my_devs/test_agilex_single_arm_infer.py)

---

## 7. 当前推荐使用方式

当前建议的 bring-up 流程如下。

### 7.1 启动 server

```bash
python my_devs/async_act/run_async_act_single_arm.py server \
    --host 127.0.0.1 \
    --port 8080 \
    --fps 30
```

### 7.2 启动 temporal ensemble client

```bash
python my_devs/async_act/run_async_act_single_arm.py client \
    --arm right \
    --control-mode command_master \
    --policy-path /home/agilex/cqy/flexible_lerobot/outputs/train/20260314_215531_act_agilex_first_test_right_full/checkpoints/100000/pretrained_model \
    --server-address 127.0.0.1:8080 \
    --policy-n-action-steps 1 \
    --policy-temporal-ensemble-coeff 0.01 \
    --action-smoothing-alpha 0.35 \
    --max-joint-step-rad 0.08 \
    --run-time-s 8
```

### 7.3 当前推荐调参起点

如果目标是继续压低抖动，建议优先试这两组参数：

1. `alpha=0.35, max_step=0.08`
2. `alpha=0.25, max_step=0.05`

调参原则如下：

1. 更稳：减小 `alpha`
2. 更稳：减小 `max_step`
3. 更跟手：增大 `alpha`
4. 若动作发钝明显，可先放宽或移除 `max_step`

---

## 8. 已知风险与后续建议

### 8.1 已知风险

当前仍需关注以下残余风险：

1. 当前抗抖策略属于执行端 command shaping，不是从根本上消除所有控制周期波动。
2. 若 future `async_inference` 协议或服务端 action chunk 组织方式变化，temporal overlap aggregation 需要重新复核。
3. 若真实控制仍有明显抖动，说明问题可能已经上升到“控制节拍与观测节拍未完全解耦”的层面。

### 8.2 后续建议

建议的后续工作优先级如下：

1. 继续基于现有 smoother 收敛 `alpha` 与 `max_step` 的实机参数区间。
2. 增加更长时间的实机稳定性测试，例如 30s 到 60s 连续运行。
3. 如仍存在明显抖动，可进入下一阶段，设计“固定频率本地发布线程”，将动作执行节拍与图像采集 / gRPC 波动彻底解耦。

---

## 9. 结论

本轮三阶段工作已经将 `my_devs/async_act` 从“方案阶段”推进到“可用、可测、可调优”的工程状态：

1. 第一阶段解决了基础异步链路问题。
2. 第二阶段解决了 ACT temporal ensemble 异步兼容问题。
3. 第三阶段解决了真实控制中的初步抗抖问题。

当前版本已经具备继续实机调参和进入下一轮控制稳定性优化的基础。
