# 子代理摘要 A: 复用 async_inference 骨架

结论：

- 现有 `src/lerobot/async_inference` 可以作为 GR00T TensorRT 远端推理的骨架复用，且复用比例高。
- 最小改造点主要在服务端的 policy 构建和配置传递，不在客户端控制环。

可直接复用：

- `src/lerobot/async_inference/robot_client.py`
  - 已经负责本地 `robot.connect()`、`get_observation()`、`send_action()`、动作队列、阈值补货和异步收动作块。
- `src/lerobot/async_inference/policy_server.py`
  - 已有 `Ready`、`SendPolicyInstructions`、`SendObservations`、`GetActions` 四个 RPC。
- `src/lerobot/async_inference/helpers.py`
  - 可继续复用 `TimedObservation`、`TimedAction`、观测转 LeRobot observation、时间戳封装逻辑。
- `src/lerobot/async_inference/configs.py`
  - 客户端/服务端配置模型基本可保留。
- `src/lerobot/async_inference/constants.py`
  - `SUPPORTED_POLICIES` 已包含 `groot`。

必须改的地方：

- 当前服务端固定走 PyTorch policy 路径：
  - `get_policy_class(...)`
  - `from_pretrained(...)`
  - `policy.predict_action_chunk(...)`
- 这与 `my_devs/groot_trt_async_server/run_groot_infer_trt.py` 中基于 TensorRT engine 的执行路径不一致。
- 现有 `TrtGrootPolicyAdapter` 暴露的是 `select_action()`，而 async server 期望的是公开的 `predict_action_chunk()`。
- `RemotePolicyConfig` 目前无法表达 TRT 服务端需要的参数：
  - `engine_dir`
  - `tensorrt_py_dir`
  - `vit_dtype`
  - `llm_dtype`
  - `dit_dtype`
  - `num_denoising_steps`

建议的最小改造路径：

1. 客户端优先保持不动，继续复用 `RobotClient`。
2. 新增一个 GR00T TRT 专用服务端，而不是直接把通用 `PolicyServer` 改成多后端大杂烩。
3. 在服务端的 policy 构建阶段引入 TRT-aware factory。
4. 给 GR00T TRT backend 提供公开的 `predict_action_chunk()` 接口。
5. 扩展 `RemotePolicyConfig`，把 TRT 所需运行参数从 client 传到 server。
6. 保持 async server 的整体数据流不变：
   - raw observation
   - preprocessor
   - `policy.predict_action_chunk(...)`
   - postprocessor
   - timed action chunk

一句话总结：

- 能复用的是 `async_inference` 的通信骨架、控制时序和 pre/post 流程。
- 必须新增的是一个公开 `predict_action_chunk()` 的 GR00T TensorRT 服务端 backend。
