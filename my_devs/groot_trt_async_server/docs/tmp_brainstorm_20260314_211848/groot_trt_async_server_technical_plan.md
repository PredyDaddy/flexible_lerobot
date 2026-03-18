# GR00T TRT Async Server 技术方案草案

## 1. 目标

把当前 `my_devs/groot_trt_async_server/run_groot_infer_trt.py` 这条“本地机器人 IO + 本地 TensorRT 推理”的同步单机闭环，改造成：

- 机器人侧只负责：
  - 相机采集
  - 关节状态读取
  - 安全动作下发
  - 本地 action queue 控制
- 服务器侧负责：
  - checkpoint 加载
  - `policy_preprocessor.json`
  - `policy_postprocessor.json`
  - 7 个 TensorRT engine 推理
  - action chunk 生成

目标收益：

- 机器人本体不需要显卡。
- 推理耗电和发热集中到服务器。
- 机器人侧控制环和服务器侧推理解耦。
- 后续具备进一步做远端推理和多后端扩展的基础。

## 2. 当前代码现状

### 2.1 已有可复用骨架

仓库已经存在一个可运行的异步 server/client 框架：

- `src/lerobot/async_inference/policy_server.py`
- `src/lerobot/async_inference/robot_client.py`
- `src/lerobot/async_inference/helpers.py`
- `src/lerobot/async_inference/configs.py`

它的工作方式已经非常接近目标系统：

- client 本地连机器人和相机
- client 周期性发送 observation
- server 收 observation 后运行 policy
- server 返回 action chunk
- client 本地维护 action queue 并逐步执行动作

### 2.2 当前 GR00T TRT 入口的限制

`my_devs/groot_trt_async_server/run_groot_infer_trt.py` 当前是一个同步本地闭环脚本，不是 server/client 结构。

它的关键特点：

- 包含本地 robot connect / get_observation / send_action
- 包含 checkpoint / processor / engine 加载
- TensorRT 推理逻辑封装在 `TrtGrootPolicyAdapter`
- 对外主接口更偏本地控制环使用的 `select_action()`

这意味着：

- 不能把这个脚本原样当成异步服务端。
- 但可以把里面的 TRT policy 核心抽出来，变成 server-side backend。

## 3. 架构判断

结论：

- 方向可行。
- 第一阶段不建议另起炉灶自研通信协议。
- 最小可行路线是：
  - 复用 `async_inference` 的通信和控制骨架
  - 新增一个 GR00T TRT 专用服务端 backend
  - 在服务端提供 `predict_action_chunk()` 语义

## 4. 推荐系统边界

### 4.1 Client 侧职责

建议继续基于 `src/lerobot/async_inference/robot_client.py` 实现：

- 管理机器人和相机硬件连接
- 在本地控制周期中读取 observation
- 把 raw observation 发给 server
- 接收 server 返回的 `TimedAction` 列表
- 在本地维护 action queue
- 在本地执行动作
- 当队列低于阈值时触发新的 observation 补货

不要在 client 侧新增：

- 模型前处理
- TensorRT 推理
- 复杂 fallback 推理
- 复杂多会话调度

### 4.2 Server 侧职责

建议新增一个 GR00T TRT 专用 server-side policy backend，持有以下资源：

- checkpoint 目录
- `policy_preprocessor.json`
- `policy_postprocessor.json`
- 7 个 `.engine`
- TensorRT Python runtime
- CUDA device

服务端主流程保持与现有 `PolicyServer` 一致：

1. 收到 raw observation
2. 转成 LeRobot observation
3. 调用 preprocessor
4. 调用 GR00T TRT `predict_action_chunk()`
5. 调用 postprocessor
6. 转成 `TimedAction` 列表回传

## 5. 核心接口设计

### 5.1 统一语义

异步服务端不应使用 `select_action()` 作为内部主接口，而应统一为：

```python
predict_action_chunk(batch: dict[str, torch.Tensor]) -> torch.Tensor
```

原因：

- `async_inference` 当前就是按 chunk 工作
- chunk 更适合抵抗网络抖动
- server 不需要同步本地控制环的 `_action_queue`

### 5.2 适配器边界

建议从 `run_groot_infer_trt.py` 中抽出一个新的 server-side adapter，例如：

- `groot_trt_policy.py`
- 或 `groot_trt_adapter.py`

职责：

- 加载 7 个 TensorRT session
- 保留必要的 torch glue modules
- 暴露公开 `predict_action_chunk()`
- 不包含 robot IO
- 不包含本地 while loop
- 不持有 client 侧 action queue 语义

### 5.3 配置扩展

现有 `RemotePolicyConfig` 不够，需要扩展 TRT 相关字段，例如：

- `backend`
- `engine_dir`
- `tensorrt_py_dir`
- `vit_dtype`
- `llm_dtype`
- `dit_dtype`
- `num_denoising_steps`

这样 client 能告诉 server 当前应加载哪套 TRT 资源。

## 6. 推荐目录规划

建议在 `my_devs/groot_trt_async_server/` 下按职责拆分：

- `groot_trt_policy.py`
  - 纯 server-side TRT policy backend
- `configs.py`
  - 异步服务端/客户端的本模块配置扩展
- `policy_server.py`
  - 基于现有 `async_inference.PolicyServer` 的 GR00T TRT 版服务端入口
- `robot_client.py`
  - 轻量包装或复用现有 `async_inference.RobotClient`
- `run_server.py`
  - server CLI
- `run_client.py`
  - client CLI
- `run_mock_roundtrip.py`
  - 不下发动作的异步链路验证入口
- `docs/`
  - 方案、验证记录、问题排查文档

第一阶段不建议做：

- 过多抽象层
- 多机器人调度器
- 独立数据库
- 自定义传输协议

## 7. MVP 方案

### 7.1 范围

第一阶段只做：

- 单机器人
- 单服务端
- 单会话
- 局域网内
- gRPC
- mock 优先

### 7.2 验证顺序

1. 同机 loopback
   - client 和 server 在同一台机器上跑
   - 先验证协议、pre/post、TRT chunk 输出
2. 局域网 mock
   - client 在机器人主机
   - server 在 GPU 服务器
   - client 不真正发送动作
3. 局域网真实动作
   - 观察 queue 稳定性、时延、断粮次数

### 7.3 指标

第一阶段必须记录：

- observation 序列化耗时
- client -> server 单向时延
- server preprocess 耗时
- server TRT inference 耗时
- server postprocess 耗时
- server -> client 动作回传时延
- action queue 最小深度
- action queue 断粮次数

## 8. 风险和约束

### 8.1 网络

- 当前 observation 通过 `pickle` 传输，图像带宽重。
- 第一阶段只适合局域网。
- 当前策略应接受“最新 observation 优先，旧 observation 可丢”。

### 8.2 服务端状态

服务端不是天然无状态：

- processor 有状态
- observation 去重有状态
- TRT backend 的推理会话也需要固定资源

所以第一阶段应采用：

- 单机器人单服务端
- sticky session

不建议第一阶段做无状态多副本负载均衡。

### 8.3 安全性

当前 `async_inference` 偏研发态：

- `grpc.insecure_channel`
- `pickle.loads`

第一阶段只建议在内网使用。

## 9. 分阶段实施建议

### 阶段 1

目标：

- 抽出纯 TRT policy backend
- 跑通同机异步 mock

交付：

- `groot_trt_policy.py`
- `run_server.py`
- `run_client.py`
- `run_mock_roundtrip.py`

### 阶段 2

目标：

- 跑通局域网 client/server
- 跑通真 observation + mock action

交付：

- 局域网延迟报告
- queue 稳定性报告

### 阶段 3

目标：

- 真动作闭环
- 补日志、异常恢复和基础保护

交付：

- 真机验证报告
- 参数建议

### 阶段 4

按需再做：

- 图像压缩
- 更明确的 observation schema
- 多机器人会话
- 更强的鉴权和网络安全

## 10. 最终建议

现在最合理的技术方向不是：

- 继续扩展 `run_groot_infer_trt.py` 这个同步脚本
- 或者重新造一套 server/client 协议

而是：

- 复用 `src/lerobot/async_inference` 的成熟骨架
- 从 `my_devs/groot_trt_async_server/run_groot_infer_trt.py` 中抽出纯服务端 TRT policy backend
- 在 `my_devs/groot_trt_async_server/` 里完成 GR00T TRT 异步化的专用模块

一句话总结：

**可行方案是“保留现有 async_inference 的 client/server 控制骨架，把 GR00T TensorRT 推理抽成 server-side `predict_action_chunk` backend，先做局域网单机器人 mock MVP，再逐步走向真实闭环”。**
