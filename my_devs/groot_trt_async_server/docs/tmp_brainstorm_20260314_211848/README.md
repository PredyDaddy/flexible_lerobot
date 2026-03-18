# GR00T TRT Async Brainstorm

本目录用于临时收敛 `my_devs/groot_trt_async_server/` 的异步推理方案。

文件说明：

- `reports/agent_async_framework_reuse.md`
  - 子代理对 `src/lerobot/async_inference` 复用路径的分析摘要。
- `reports/agent_risk_and_mvp.md`
  - 子代理对网络、时延、MVP 边界和风险的分析摘要。
- `groot_trt_async_server_technical_plan.md`
  - 基于当前仓库代码和多路子代理结论收敛出的可行技术方案草案。

当前共识：

- 方向可行。
- 第一阶段应复用现有 `async_inference` 的 gRPC client/server 骨架。
- 第一阶段不要重写通信框架，也不要把 `run_groot_infer_trt.py` 原样变成服务端。
- 核心工作是把 GR00T TensorRT 推理抽成一个 server-side `predict_action_chunk` backend。
