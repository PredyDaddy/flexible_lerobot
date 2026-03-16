# Agilex GROOT Async MVP Technical Solution

## 1. Goal

本方案只解决一个问题：

- 让训练产物 `outputs/train/groot_agilex_first_test_right_20260315_221522` 的 `groot` checkpoint
- 通过服务端和客户端的异步链路
- 在另一台 Agilex 机器人设备上完成最小一步推理闭环

这里的“一步推理闭环”定义为：

1. 客户端发送一帧观测
2. 服务端完成一次 `groot` 推理
3. 服务端只返回 1 个动作
4. 客户端只执行这 1 个动作

本阶段不追求完整异步控制功能，不追求最佳吞吐，不追求多种运行模式兼容。

## 2. Current Facts

### 2.1 Checkpoint contract

当前可用训练产物位于：

- `outputs/train/groot_agilex_first_test_right_20260315_221522/bs4_20260315_221522/checkpoints/020000/pretrained_model`

该 checkpoint 的真实输入输出契约已经固定：

- `policy.type = groot`
- `input_features.observation.state.shape = [7]`
- `input_features.observation.images.camera_front.shape = [3, 480, 640]`
- `input_features.observation.images.camera_right.shape = [3, 480, 640]`
- `output_features.action.shape = [7]`
- `chunk_size = 16`
- `n_action_steps = 16`

同时，保存下来的 `policy_preprocessor.json` 和 `policy_postprocessor.json` 说明：

- preprocessor 会走 `groot_pack_inputs_v3`
- `language_key = "task"`
- `action_horizon = 16`
- postprocessor 会把动作裁成 `env_action_dim = 7`

此外，这个 fine-tuned checkpoint 仍依赖外部底座资源：

- `model.safetensors` 或等价权重分片
- `base_model_path`
- `tokenizer_assets_repo` 对应的 Eagle 处理器资产

这意味着当前 MVP 的目标臂应固定为右臂，客户端必须至少提供：

- 7 维右臂状态
- `camera_front`
- `camera_right`
- `task`

### 2.2 Async framework facts

现有 `src/lerobot/async_inference` 对 `groot` 服务端天然可用：

- `SUPPORTED_POLICIES` 已包含 `groot`
- `PolicyServer.SendPolicyInstructions()` 会按 `policy_type` 加载策略
- `make_pre_post_processors()` 已包含 `GrootConfig` 专门分支

同时，服务端的动作返回逻辑已经满足 MVP 需求：

- 服务端总是先调用 `policy.predict_action_chunk()`
- 再用 `actions_per_chunk` 截断返回值

因此，把 `actions_per_chunk` 设为 `1`，就可以把服务端行为收敛为“一次观测，只回 1 个动作”。

### 2.3 Client-side constraint

现有 `python -m lerobot.async_inference.robot_client` 不能直接用于 Agilex MVP。

原因不是 `groot`，而是该 CLI 会校验：

- `cfg.robot.type in SUPPORTED_ROBOTS`

而当前 `SUPPORTED_ROBOTS` 不包含 `agilex`。

所以 MVP 不能依赖内置 CLI，而应采用一个薄客户端包装脚本，内部直接构造：

- `RobotClientConfig`
- `RobotClient`

这样可以绕开 `SUPPORTED_ROBOTS` 的 CLI 限制，同时继续复用通用异步主干。

这里还有一个跨设备部署约束：

- `pretrained_name_or_path` 是客户端发送给服务端的字符串
- 真正解析并加载 checkpoint 的进程是服务端

因此，GROOT MVP 不能默认照搬 `my_devs/async_act` 那种“客户端本地读取 checkpoint config 再校验”的模式。
在机器人和 GPU 服务端分离的前提下，`policy_path` 必须首先是服务端可见路径。

这里还有一个与“一步返回”直接相关的隐藏约束：

- 现有 `RobotClient.control_loop_observation()` 会把 observation timestep 设为 `latest_action`
- 现有服务端会把返回动作的起始 timestep 设为 `observation_timestep`
- 现有客户端会丢弃 `timestep <= latest_action` 的动作

这意味着如果直接把 `actions_per_chunk = 1` 套到原始 `RobotClient`，首个动作执行后，后续返回的单步动作会因为 timestep 不前进而被客户端持续丢弃。

因此，MVP 虽然仍然采用“单步动作返回”，但客户端必须通过一个薄包装层修正 observation timestep 语义，使其在单步模式下递增到 `latest_action + 1`。

## 3. Design Decision

### 3.1 Reuse as much as possible

MVP 只复用已经稳定的四个边界：

1. `src/lerobot/async_inference/policy_server.py`
2. `src/lerobot/async_inference/robot_client.py`
3. `my_devs/async_act/config_single_arm_agilex_robot.py`
4. `my_devs/async_act/single_arm_agilex_robot.py`

这里最关键的判断是：

- `async_act` 的机器人包装层是通用的 Agilex 单臂部署边界
- ACT 专属的是 schema 校验、temporal ensemble 和部分运行时参数

因此，GROOT MVP 不应重写机器人包装层，而应复用单臂 Agilex 包装层，只替换策略校验和入口参数。

同时，GROOT MVP 不应直接复用 `async_act` 的 `build_client_config()` 模式。更合理的边界是：

- 机器人客户端只负责透传服务端可解析的 `policy_path`
- checkpoint schema 校验改成服务端 preflight 或离线部署检查
- 机器人客户端本地只解析机器人参数和网络参数

### 3.2 Keep the protocol unchanged

MVP 不修改 gRPC 协议，不扩 `RemotePolicyConfig`，也不扩 server/client 主干消息格式。

当前握手字段已经足够：

- `policy_type`
- `pretrained_name_or_path`
- `lerobot_features`
- `actions_per_chunk`
- `device`
- `rename_map`

对于当前目标，没有必要为了 MVP 去扩协议。

另外，客户端在读取 `groot` checkpoint 配置前，必须确保 `groot` 配置类已经注册。

最直接的做法是：

- 在薄包装脚本里显式导入 `lerobot.policies`

否则 `PreTrainedConfig.from_pretrained()` 可能在解析 `type = groot` 时失败。

### 3.3 Choose one-step action return

MVP 采用如下运行时策略：

- `actions_per_chunk = 1`
- `chunk_size_threshold = 1.0`
- `aggregate_fn_name = latest_only`
- `observation_timestep = latest_action + 1` for one-step mode

这样做的目的不是优化控制效果，而是最小化变量：

1. 服务端仍然按 `groot` 原始逻辑做 chunk 推理
2. 但只回传第一个动作
3. 客户端不做复杂 overlap aggregation
4. 客户端通过最小 timestep 修正保证单步动作不会被当成旧动作丢弃
5. 控制回路退化为最容易观测和验证的“一发一收一执行”

## 4. MVP Architecture

### 4.1 Components

MVP 组件划分如下：

1. GPU 服务端
   - 启动 `PolicyServer`
   - 加载 `groot` checkpoint
   - 接收客户端观测
   - 返回 1-step action

2. 机器人客户端
   - 启动薄包装脚本
   - 复用 `RobotClient` 的主体逻辑
   - 复用 `SingleArmAgileXRobot`
   - 从 Agilex ROS topic 取实时状态和图像
   - 接收动作并执行
   - 在单步模式下修正 observation timestep

3. 单臂机器人包装层
   - 对外只暴露右臂 7 维状态
   - 对外只暴露 `camera_front` 和 `camera_right`
   - 对外只接受右臂 7 维动作
   - 下发前把未控臂补成当前位姿

### 4.2 Data flow

最小数据流如下：

1. 客户端连接服务端并发送 `RemotePolicyConfig`
2. 客户端从 `SingleArmAgileXRobot` 读取观测
3. 客户端把观测补上 `task`
4. 服务端把原始观测转成 LeRobot observation
5. 服务端执行 `groot` preprocessor -> policy -> postprocessor
6. 服务端把预测动作裁成长度为 `1` 的 action chunk
7. 客户端收到该动作并立即执行
8. 下一帧观测以 `latest_action + 1` 作为 timestep 再次发送

### 4.3 Contract boundaries

服务端不应该知道 Agilex 是双臂机器人。

服务端应只看到和 checkpoint 一致的策略输入：

- `observation.state`
- `observation.images.camera_front`
- `observation.images.camera_right`
- `task`

因此，双臂裁剪和 inactive arm hold-current 必须留在机器人包装层，而不是下沉到服务端或协议层。

## 5. Minimal Interface Contract

### 5.1 Server-side contract

服务端最小配置项：

- `host`
- `port`
- `fps`
- `inference_latency`
- `obs_queue_timeout`

服务端最小策略配置：

- `policy_type = groot`
- `pretrained_name_or_path = <checkpoint>/pretrained_model`
- `device = cuda`
- `actions_per_chunk = 1`

### 5.2 Client-side contract

客户端最小配置项：

- `server_address`
- `policy_type = groot`
- `pretrained_name_or_path = <server-visible checkpoint path>`
- `policy_device = cuda`
- `client_device = cpu`
- `actions_per_chunk = 1`
- `chunk_size_threshold = 1.0`
- `aggregate_fn_name = latest_only`
- `task = <non-empty string>`

### 5.3 Robot-side contract

当前 MVP 只支持右臂 checkpoint，因此机器人对外契约固定为：

- 状态：
  - `right_joint0.pos` 到 `right_joint6.pos`
- 相机：
  - `camera_front`
  - `camera_right`
- 动作：
  - `right_joint0.pos` 到 `right_joint6.pos`

如果未来切到左臂 checkpoint，则切换为：

- `left_joint*.pos`
- `camera_front`
- `camera_left`

## 6. Non-goals

本方案明确不做以下内容：

1. 不修改 `src/lerobot/async_inference` 协议
2. 不修改 `PolicyServer` 主干逻辑
3. 不修改内置 `async_client()` CLI 来适配 Agilex
4. 不引入 ACT temporal ensemble 兼容层
5. 不引入复杂 queue 调优策略
6. 不支持双臂并发部署
7. 不支持多 checkpoint 自动切换
8. 不在本阶段加入额外安全整形策略作为强依赖

其中第 8 条尤其重要：

- `action_smoothing_alpha`
- `max_joint_step_rad`

这些能力可以作为后续增强项保留，但不应阻塞最小闭环打通。

## 7. Risks And Guardrails

### 7.1 Highest-risk items

MVP 的最高风险不在 `groot` 模型本身，而在部署契约：

1. 客户端如果直接使用整机 `AgileXRobot`，会造成 14 维状态和 14 维动作错配
2. 客户端如果漏传 `task`，会和 `groot_pack_inputs_v3` 的语言输入约定不一致
3. 客户端如果错误选择相机键，会和 checkpoint schema 不一致
4. 如果试图复用 `async_act` 的 ACT schema 校验，会因 `policy.type != act` 直接失败
5. 如果试图使用内置 `async_client()` CLI，会被 `SUPPORTED_ROBOTS` 拦住
6. 如果直接复用原始 `RobotClient` 的 observation timestep 逻辑，单步模式会在首个动作后陷入“动作被当成旧 timestep 丢弃”的错误路径
7. 如果机器人客户端传入的是“只在客户端本机存在”的 checkpoint 路径，服务端会在加载策略时直接失败

### 7.2 Required pre-deployment checks

正式编码前，技术方案要求实现以下校验：

1. 服务端可见的 checkpoint 路径下必须存在：
   - `config.json`
   - `model.safetensors` 或等价权重分片
   - `policy_preprocessor.json`
   - `policy_postprocessor.json`
2. `config.json` 必须满足：
   - `type == groot`
   - `input_features` 与右臂 schema 匹配
   - `output_features.action.shape == [7]`
3. `policy_preprocessor.json` 必须确认：
   - `language_key == "task"`
4. 客户端读取 checkpoint config 前必须确保 `lerobot.policies` 已导入，保证 `groot` 配置类已注册
5. `config.json` 中的底座依赖必须真实可访问：
   - `base_model_path`
   - `tokenizer_assets_repo` 对应的 Eagle 资产
6. 如果机器人客户端与服务端不共享文件系统，则客户端运行时不应强依赖本地读取 checkpoint config
7. 客户端运行前必须打印 resolved summary：
   - checkpoint 路径
   - server 地址
   - policy device
   - actions_per_chunk
   - one-step timestep strategy
   - camera keys
   - arm
   - task

## 8. Recommended Minimal Command Shape

### 8.1 Server

服务端命令保持复用现有模块：

```bash
conda run -n lerobot_flex python -m lerobot.async_inference.policy_server \
  --host 0.0.0.0 \
  --port 8080 \
  --fps 30 \
  --inference_latency 0.033 \
  --obs_queue_timeout 2
```

### 8.2 Client

客户端不建议直接用内置 CLI，而应使用一个新的 GROOT 薄包装脚本。

这个脚本的边界应当是：

- `policy-path` 只作为服务端加载路径透传
- 默认不要求机器人设备本地可读取该 checkpoint
- 本地只处理机器人参数、网络参数、一步模式参数和 task

建议命令形态：

```bash
conda run -n lerobot_flex python my_devs/train/groot/agilex/run_async_groot_single_arm.py client \
  --arm right \
  --control-mode passive_follow \
  --policy-path outputs/train/groot_agilex_first_test_right_20260315_221522/bs4_20260315_221522/checkpoints/020000/pretrained_model \
  --server-address <server_ip>:8080 \
  --policy-device cuda \
  --actions-per-chunk 1 \
  --chunk-size-threshold 1.0 \
  --aggregate-fn-name latest_only \
  --task "Execute the trained Agilex right-arm GROOT task"
```

这里的核心不是命令名称，而是运行语义：

- 只返回 1 个动作
- 只在客户端薄包装层上做 timestep 修正
- 先 `passive_follow`
- 先验证 transport 和 schema
- 不在 MVP 阶段引入额外运行模式

## 9. Implementation Decision Summary

技术方案最终决策如下：

1. 复用现有 `PolicyServer`
2. 复用现有 `RobotClient` 类，但不复用内置 `async_client()` CLI
3. 复用 `my_devs/async_act` 的单臂 Agilex 机器人包装层
4. 新增 GROOT 专属薄包装脚本、GROOT schema 校验和单步 timestep 修正
5. MVP 只支持右臂 checkpoint
6. MVP 只支持一步动作返回
7. MVP 先跑通 `passive_follow`，再讨论真实控制

这份方案的核心价值不是“功能多”，而是把最小闭环的边界压缩到最可控的范围内。
