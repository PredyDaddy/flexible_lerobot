# Agilex 单臂 ACT 异步推理技术计划

## 1. 计划目标

基于 `src/lerobot/async_inference`，交付一条面向 Agilex 单臂 ACT 的异步推理 `v1` 路径，满足以下目标：

1. 复用现有 `PolicyServer` / `RobotClient` 主干。
2. 通过包装机器人保持单臂 7 维部署契约。
3. 保留 inactive arm hold-current 的真实执行语义。
4. 先稳定支持当前第一条同步命令对应的 chunked async。
5. 不在 `v1` 内承诺 `temporal_ensemble_coeff` 的 exact async 兼容。

## 2. `v1` 交付范围

### 2.1 In scope

`v1` 包含以下内容：

1. `SingleArmAgileXRobot`
2. `run_async_act_single_arm.py`
3. checkpoint 契约校验与参数 guardrail
4. chunked async 真机 shadow 与短时闭环验证
5. 最小可维护测试矩阵
6. 文档与调参建议

### 2.2 Out of scope

`v1` 明确不包含：

1. `policy_temporal_ensemble_coeff` 的 exact async 支持
2. 双臂双模型并发
3. 跨机器网络优化
4. 修改 `src/lerobot/async_inference` 协议

## 3. 实施原则

整个实施过程遵循以下原则：

1. 不改现有同步脚本。
2. 先 mock，后真机。
3. 先 `passive_follow`，后 `command_master`。
4. 所有测试和验证命令都使用 `lerobot_flex` conda 环境。
5. 所有运行期配置都必须在启动前打印 resolved summary。

## 4. 阶段拆分

## 阶段 0：方案冻结

### 目标

先把边界讲清楚，避免编码过程里反复改语义。

### 任务

1. 冻结 `v1` / `phase 2` 功能边界。
2. 冻结单臂 observation / action contract。
3. 冻结参数映射规则：
   - `actions_per_chunk`
   - `policy_n_action_steps` alias
   - `policy_temporal_ensemble_coeff` 拒绝逻辑
4. 冻结默认 bring-up 参数。

### 产出

1. 技术方案文档
2. 技术计划文档
3. 参数边界表

### 验收标准

1. 团队对 `v1` 支持范围没有歧义。
2. 不存在“编码时再决定是否偷偷兼容 temporal ensemble”的模糊空间。

## 阶段 1：抽出公共契约与校验模块

### 目标

把当前同步脚本里真正需要复用的规则先抽成公共模块，避免同步/异步各自维护一套。

### 任务

1. 新增：
   - `my_devs/async_act/common.py`
2. 优先抽取：
   - arm / camera 常量
   - checkpoint 契约校验
   - 启动前 summary
   - 参数 guardrail
3. 保持同步脚本行为不变，只允许做无语义变化的复用抽取。

### 产出

1. 公共常量与校验模块
2. 对应单元测试

### 验收标准

1. 同步脚本功能不回归。
2. 异步入口不再复制同一份契约校验逻辑。

## 阶段 2：实现 `SingleArmAgileXRobot`

### 目标

把 Agilex 双臂硬件封装成一个可直接喂给 `RobotClient` 的单臂 `Robot`。

### 任务

1. 新增：
   - `my_devs/async_act/__init__.py`
   - `my_devs/async_act/config_single_arm_agilex_robot.py`
   - `my_devs/async_act/single_arm_agilex_robot.py`
2. 实现配置与核心方法：
   - `connect()`
   - `disconnect()`
   - `observation_features`
   - `action_features`
   - `get_observation()`
   - `send_action()`
3. 内部继续复用 `AgileXRosBridge`。
4. 在 `send_action()` 内实现 inactive arm hold-current merge。
5. 在发布前做 inactive arm 一致性校验。
6. 保证 `RobotClient -> make_robot_from_config -> make_device_from_device_class` 这条工厂链路可以成功实例化该机器人：
   - 配置类完成 `RobotConfig.register_subclass(...)`
   - 包文件命名与反射规则兼容
   - `my_devs/async_act/__init__.py` 显式导出设备类与配置类

### 建议测试

1. `test_single_arm_observation_contract`
2. `test_single_arm_action_merge`
3. `test_passive_follow_no_publish`
4. `test_inactive_arm_hold_validation`
5. `test_single_arm_robot_factory_instantiation`

### 验收标准

1. 左/右臂都能稳定导出 7 维状态。
2. 7 维动作能稳定补齐成 14 维整机命令。
3. `passive_follow` 下不会真实发布。
4. 通过最小实例化测试证明工厂反射链路可用。

## 阶段 3：实现异步入口脚本

### 目标

提供一个符合当前使用习惯、但内部复用 async 主干的入口脚本。

### 任务

1. 新增：
   - `my_devs/async_act/run_async_act_single_arm.py`
2. 设计：
   - `server` 子命令
   - `client` 子命令
3. `server` 子命令负责：
   - 启动 `PolicyServer`
4. `client` 子命令负责：
   - 读取 checkpoint config
   - 执行契约校验
   - 创建 `SingleArmAgileXRobot`
   - 构造 `RobotClientConfig`
   - 支持 `run_time_s`
5. 参数规则：
   - 如果没传 `actions_per_chunk`，默认取 checkpoint `chunk_size`
   - `policy_n_action_steps` 只作为 alias
   - 一旦设置 `policy_temporal_ensemble_coeff`，直接报错

### 建议测试

1. parser / dry-run 测试
2. invalid argument fail-fast 测试
3. resolved config 输出测试

### 验收标准

1. `dry-run` 能清楚打印最终生效配置。
2. 参数误用可在连接真机前被拦截。

## 阶段 4：接入 async runtime 并完成 mock e2e

### 目标

先在不接真机的前提下证明异步链路语义正确。

### 任务

1. 使用 `SingleArmAgileXRobot` 驱动 `RobotClient`。
2. 保持 server 端直接复用 `PolicyServer`。
3. 构建 mock bridge 或 fake robot。
4. 验证：
   - client 发出的 observation 是单臂契约
   - server 返回的是单臂 action chunk
   - client 能稳定消费 queue
   - inactive arm merge 在执行前发生

### 建议测试

1. `tests/async_act/test_async_act_single_arm_e2e.py`
2. `tests/async_act/test_queue_threshold_behavior.py`
3. `tests/async_act/test_runtime_timeout_exit.py`

### 验收标准

1. mock e2e 稳定通过。
2. 没有 schema / timestep / queue 回退类错误。

## 阶段 5：真机 shadow 验证

### 目标

先证明异步链路和调参策略可运行，再碰真实闭环。

### 任务

1. 启动 server。
2. 以 `passive_follow` 启动 client。
3. 重点观察：
   - queue size
   - queue underrun count
   - observation send count
   - action receive count
   - 首帧 observation summary
   - 首批 action chunk summary
4. 重点调优：
   - `actions_per_chunk = 100 / 50 / 30`
   - `chunk_size_threshold = 0.5 / 0.7`
   - `aggregate_fn_name = weighted_average / latest_only`

### 推荐命令

```bash
conda run -n lerobot_flex python my_devs/async_act/run_async_act_single_arm.py server \
    --host 127.0.0.1 \
    --port 8080 \
    --fps 30
```

```bash
conda run -n lerobot_flex python my_devs/async_act/run_async_act_single_arm.py client \
    --arm right \
    --control-mode passive_follow \
    --policy-path /home/agilex/cqy/flexible_lerobot/outputs/train/20260314_215531_act_agilex_first_test_right_full/checkpoints/100000/pretrained_model \
    --server-address 127.0.0.1:8080 \
    --actions-per-chunk 100 \
    --chunk-size-threshold 0.5 \
    --aggregate-fn-name weighted_average \
    --run-time-s 30
```

### 验收标准

1. shadow 模式连续运行至少 5 分钟稳定。
2. queue 不长期见底。
3. 没有 schema、topic、连接异常。

## 阶段 6：短时真实闭环验证

### 目标

验证 `command_master` 下的真实机器人行为。

### 任务

1. 先以 8 秒短跑开始。
2. 核查：
   - 目标臂动作是否合理
   - 非目标臂是否保持稳定
   - 是否有明显卡顿、爆跳、连续断粮
3. 必要时回退到 shadow 模式继续调参。

### 推荐命令

```bash
conda run -n lerobot_flex python my_devs/async_act/run_async_act_single_arm.py client \
    --arm right \
    --control-mode command_master \
    --policy-path /home/agilex/cqy/flexible_lerobot/outputs/train/20260314_215531_act_agilex_first_test_right_full/checkpoints/100000/pretrained_model \
    --server-address 127.0.0.1:8080 \
    --actions-per-chunk 100 \
    --chunk-size-threshold 0.5 \
    --aggregate-fn-name weighted_average \
    --run-time-s 8
```

### 验收标准

1. 8 秒短跑稳定。
2. inactive arm 无误动。
3. 未出现连续 queue underrun。

## 阶段 7：文档收口与上线准备

### 目标

把限制、用法、调参和回退路径沉淀清楚。

### 任务

1. 形成最终用户文档。
2. 记录已知限制。
3. 记录默认参数建议。
4. 明确写出回退路径仍然是同步脚本。

### 验收标准

1. 新成员可以按文档独立完成 shadow bring-up。
2. 参数误用和回退路径都能从文档直接找到。

## 5. 测试矩阵

### 5.1 单元测试

至少覆盖以下测试点：

1. 单臂 observation contract
2. 单臂 action merge
3. `passive_follow` 不发布
4. checkpoint 契约校验
5. alias / guardrail 校验

### 5.2 集成测试

至少覆盖以下测试点：

1. async client/server mock round-trip
2. queue threshold 行为
3. timeout / shutdown 行为
4. inactive arm merge 在执行前发生

### 5.3 真机验证

至少覆盖：

1. shadow 5 分钟
2. command 8 秒
3. 参数对比：
   - `actions_per_chunk = 100 / 50 / 30`
   - `aggregate_fn_name = weighted_average / latest_only`

## 6. 可观测性清单

实现期至少要打印和记录以下指标：

1. queue size
2. queue underrun count
3. observation send / action receive 频率
4. server preprocess / inference / postprocess 耗时
5. inactive arm hold deviation
6. resolved checkpoint schema
7. resolved runtime config

## 7. 决策门

### Gate A

如果 mock e2e 不稳定：

1. 停止真机验证
2. 先修包装层或 queue 行为

### Gate B

如果 shadow 模式 queue 频繁见底：

1. 先调大 `actions_per_chunk`
2. 再调大 `chunk_size_threshold`
3. 若仍不稳定，暂停 `command_master`

### Gate C

如果 `command_master` 模式出现 inactive arm 异常：

1. 立即回退到同步脚本
2. 把问题定位到 merge 逻辑后再继续

## 8. 回退方案

回退方案必须始终可执行：

1. 不删除同步脚本
2. 新 async 功能独立入口
3. 任何异常优先回退到当前同步命令

回退命令如下：

```bash
conda run -n lerobot_flex python my_devs/train/act/agilex/run_act_single_arm_infer.py \
    --arm right \
    --execution-mode policy_inference \
    --control-mode command_master \
    --policy-path /home/agilex/cqy/flexible_lerobot/outputs/train/20260314_215531_act_agilex_first_test_right_full/checkpoints/100000/pretrained_model \
    --run-time-s 8
```

## 9. `phase 2` 预研计划

只有在 `v1` 稳定后，才进入 `phase 2`。`phase 2` 的首选方向不是“server 每步只回 1 个动作”，而是：

1. 保留 `predict_action_chunk()` 路径
2. 设计 ACT-aware overlap aggregator
3. 为每个 timestep 维护额外聚合状态
4. 用专门测试验证是否足够接近同步 temporal ensemble 语义

只有当这套方案在软件层和真机层都可解释、可测试、可维护时，才重新引入：

1. `policy_n_action_steps=1`
2. `policy_temporal_ensemble_coeff`

## 10. 一句话收口

这项工作的正确推进顺序是：

1. 先把单臂语义封装干净
2. 再复用 async 主干
3. 再用 mock 和 shadow 把行为跑稳
4. 最后才考虑 temporal ensemble 的 `phase 2` 兼容
