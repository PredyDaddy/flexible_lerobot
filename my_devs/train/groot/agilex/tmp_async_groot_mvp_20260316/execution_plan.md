# Agilex GROOT Async MVP Execution Plan

## 1. Goal

本执行文档只服务于一个交付目标：

- 在不扩功能的前提下
- 完成 `groot` 训练产物的服务端 + 客户端一步异步推理 MVP
- 先跑通 `passive_follow`

最终交付定义为：

1. 服务端可以加载指定 `groot` checkpoint
2. 客户端可以在另一台 Agilex 设备上发送单臂观测
3. 服务端可以返回 1-step action
4. 客户端可以稳定接收并执行该动作

## 2. Deliverables

本次执行阶段只承诺以下产出：

1. 一个 GROOT 薄客户端方案
2. 一个可复用的右臂单步推理运行路径
3. 一套部署前校验
4. 一套 cross-device bring-up 验证步骤
5. 一份运行记录和问题清单

不承诺：

1. 多模式运行参数
2. 双臂部署
3. temporal ensemble
4. queue 调优
5. 性能优化

## 3. Risk Gates

在进入编码或联调前，必须先过以下风险门禁。

### 3.1 Asset gate

输入：

- `outputs/train/groot_agilex_first_test_right_20260315_221522/.../020000/pretrained_model`
- 本地 `base_model_path`
- Eagle tokenizer assets

输出：

- 资产完整性确认结果

验收标准：

1. checkpoint 目录下存在：
   - `config.json`
   - `policy_preprocessor.json`
   - `policy_postprocessor.json`
   - `model.safetensors` 或等价权重分片
2. `base_model_path` 可访问
3. `tokenizer_assets_repo` 对应资产可访问

风险前置项：

1. 如果 checkpoint 权重不完整，后续阶段全部阻塞
2. 如果底座模型不可访问，服务端虽能读取微调配置但无法真正起模

### 3.2 Schema gate

输入：

- checkpoint `config.json`
- checkpoint `policy_preprocessor.json`
- Agilex 右臂部署目标

输出：

- 部署契约确认结果

验收标准：

1. `policy.type == groot`
2. 输入 schema 只包含：
   - `observation.state`
   - `observation.images.camera_front`
   - `observation.images.camera_right`
3. 输出 schema 为 `7` 维 `action`
4. `language_key == task`
5. 本次交付明确标记为“右臂 checkpoint only”

风险前置项：

1. 如果相机键或动作维度理解错误，机器人侧接入一定返工
2. 如果把当前模型误认为左右臂通用，会在真实部署阶段出错

### 3.3 Runtime gate

输入：

- `src/lerobot/async_inference` 现状
- MVP 目标为一步动作返回

输出：

- 运行时边界确认结果

验收标准：

1. 明确不改 gRPC 协议
2. 明确不使用内置 `async_client()` CLI
3. 明确必须做 one-step timestep 修正

风险前置项：

1. 如果直接把 `actions_per_chunk = 1` 套到原始 `RobotClient`，首个动作后会因为 timestep 不前进而被持续丢弃
2. 如果试图直接复用 ACT temporal ensemble 分支，会扩大问题面

## 4. Recommended Phases

## Phase 0: Freeze MVP Scope

输入：

- `technical_solution.md`
- 当前仓库代码事实

输出：

- 冻结后的 MVP 边界

实施内容：

1. 冻结“只做一步动作返回”的目标
2. 冻结“只支持右臂 checkpoint”的边界
3. 冻结“复用 `PolicyServer` + 复用 `SingleArmAgileXRobot` + 新增 GROOT 薄客户端”的实现主线
4. 冻结不做项：
   - 不扩协议
   - 不改 core CLI
   - 不上 temporal ensemble

验收标准：

1. 实施人员对 MVP 范围没有歧义
2. 所有非目标明确列出

风险前置项：

1. 如果范围不先冻结，后续很容易把 ACT 的复杂运行时带进来

## Phase 1: Add Deployment Checks

输入：

- checkpoint 目录
- `groot` 配置加载路径
- Agilex 右臂契约

输出：

- 一套 GROOT 专用部署前校验规则

实施内容：

1. 补 GROOT checkpoint 校验：
   - `type == groot`
   - 输入输出 schema 对齐
   - `task` 语言键存在
2. 补资产校验：
   - 底座模型
   - Eagle 资产
3. 补启动前 resolved summary 要求
4. 明确客户端在读配置前需要导入 `lerobot.policies`

验收标准：

1. 在连接真机前即可发现 schema/资产问题
2. 启动日志能完整打印有效配置

风险前置项：

1. 如果没有这一步，问题会推迟到跨机联调阶段才暴露

## Phase 2: Build GROOT Thin Client Path

输入：

- `RobotClient`
- `SingleArmAgileXRobot`
- GROOT checkpoint 契约

输出：

- 一条 GROOT 专属的客户端启动路径

实施内容：

1. 新增 GROOT 薄客户端入口
2. 内部直接构造 `RobotClientConfig`
3. 复用 `SingleArmAgileXRobotConfig`
4. 固定 MVP 默认值：
   - `actions_per_chunk = 1`
   - `chunk_size_threshold = 1.0`
   - `aggregate_fn_name = latest_only`
5. 在单步模式下修正 observation timestep 为 `latest_action + 1`

验收标准：

1. 不依赖内置 `async_client()` CLI 也能启动客户端
2. 单步动作不会在第二轮开始被客户端丢弃

风险前置项：

1. timestep 修正是 MVP 的必要条件，不是可选优化

## Phase 3: Validate Robot Contract

输入：

- `SingleArmAgileXRobot`
- 右臂 checkpoint 契约
- 当前 Agilex ROS topic 配置

输出：

- 机器人侧部署契约确认

实施内容：

1. 确认机器人只暴露：
   - 右臂 7 维状态
   - `camera_front`
   - `camera_right`
2. 确认客户端只发送这两路图像
3. 确认机器人执行前会把左臂补为当前位姿
4. 确认 `passive_follow` 不真实发命令

验收标准：

1. 机器人包装层对外契约与 checkpoint 完全匹配
2. 未控臂安全语义留在机器人层

风险前置项：

1. 如果直接暴露整机 14 维契约，后续所有 async 调试都没有意义

## Phase 4: Local Dry Run And Service-only Smoke

输入：

- 服务端启动命令
- 客户端启动命令
- 一套不接真机的本地验证条件

输出：

- 本地 dry-run 结果
- 服务端最小 smoke 结果

实施内容：

1. 先验证服务端能加载 checkpoint
2. 再验证客户端参数解析和 summary 打印
3. 验证单步模式下的 timestep 语义
4. 验证服务端最小推理 smoke 能走通

验收标准：

1. 本地 dry-run 不报 schema/资产错误
2. 服务端推理链能走通
3. 单步模式不会在首轮之后卡死

风险前置项：

1. 如果本地 dry-run 没过，不允许进入跨机联调

## Phase 5: Cross-device Passive Follow Bring-up

输入：

- GPU 服务端机器
- Agilex 客户端机器
- 网络地址和 ROS topic

输出：

- 跨机联调 bring-up 结果

实施内容：

1. 在服务端机器启动 `PolicyServer`
2. 在机器人机器启动 GROOT 薄客户端
3. 首先使用 `passive_follow`
4. 观察：
   - Ready 握手
   - policy instructions
   - observation send
   - action receive
   - queue 行为
   - timestep 是否递增

验收标准：

1. 服务端能稳定收到观测
2. 客户端能稳定收到单步动作
3. 至少持续运行一个短时间窗口而不报错

风险前置项：

1. 网络延迟不是本阶段优化目标，只要闭环稳定即可
2. 如果日志里出现旧 timestep 丢弃，需要优先回查 one-step timestep 修正

## Phase 6: Minimal Real Action Test

输入：

- `passive_follow` bring-up 成果
- 真实机器人执行权限

输出：

- 最小真实动作验证结果

实施内容：

1. 切到最保守的真实执行模式
2. 只验证短时、低风险动作
3. 重点记录：
   - 单步动作是否落地
   - 未控臂是否保持当前位姿
   - 相机/状态输入是否稳定

验收标准：

1. 机器人能执行最小单步动作
2. 未控臂不漂移
3. 没有明显的维度、时序或 topic 错配

风险前置项：

1. 如果 `passive_follow` 阶段未稳定，不允许进入真实动作测试

## 5. Acceptance Matrix

执行文档最终只用以下六条判断是否完成：

1. `groot` checkpoint 资产和 schema 校验全部通过
2. 服务端能独立加载 checkpoint 并完成一次推理
3. 客户端能绕开内置 CLI 限制启动
4. 单步 timestep 修正已生效
5. 跨机 `passive_follow` 下可以稳定收发一次以上 observation/action
6. 真实机器人能执行最小单步动作而不破坏未控臂

## 6. Stop Conditions

出现以下任一情况，应立即停止继续扩功能，只回到当前阶段修复：

1. checkpoint 资产不完整
2. 输入输出 schema 与右臂部署不一致
3. 客户端没有完成 one-step timestep 修正
4. 仍在尝试直接使用内置 `async_client()` CLI
5. `passive_follow` 还没稳定，就开始做真实连续控制

## 7. Final Recommendation

推荐执行顺序只有一条主线：

1. 先冻结边界
2. 再补部署校验
3. 再做 GROOT 薄客户端
4. 再做本地 dry-run
5. 再做跨机 `passive_follow`
6. 最后做最小真实动作验证

这条顺序的核心目的是把风险拆开，而不是把所有变量一次性混在真机联调里。
