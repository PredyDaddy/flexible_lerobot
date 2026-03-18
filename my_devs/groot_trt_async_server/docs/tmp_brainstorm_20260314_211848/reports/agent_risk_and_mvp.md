# 子代理摘要 B: 风险与 MVP 边界

结论：

- 方案工程上可做，但第一阶段必须把它当成“局域网内、单机器人、单会话、最新观测优先”的实时系统来做。
- 不应从第一天就把目标设成通用分布式推理平台。

主要风险：

- 网络抖动会影响动作连续性。
- 当前 observation/action 通过 `pickle` 传输，图像带宽较重。
- TensorRT 服务端必须运行在有 NVIDIA CUDA 的机器上。
- 服务端天然有状态，不适合无状态负载均衡。
- 当前异步框架偏研发态，适合内网，不适合直接公网部署。
- 当前默认 server backend 是 PyTorch，不是 TensorRT。

推荐的协议和序列化边界：

- 第一阶段继续使用 gRPC。
- 第一阶段的边界放在“机器人原始观测 -> 服务端 preprocessor”之前：
  - client 发 raw observation
  - server 做 preprocessor、TRT 推理、postprocessor
- 第一阶段可以继续容忍 `pickle`，但只限局域网内 MVP。
- 第二阶段若优化，优先考虑图像压缩或明确的 observation proto，而不是重写整个通信框架。

MVP 先做什么：

1. 先做单机器人、单服务端、单会话版本。
2. 复用现有 `async_inference` 的 client/server 架子。
3. 服务端 policy backend 替换为 GR00T TRT adapter。
4. 服务端完整持有 checkpoint、processors、7 个 engine、TensorRT 环境。
5. client 只负责采观测、发 observation、收 action chunk 和下发动作。
6. 先跑 mock 链路，再上真实动作。
7. 第一时间打出时延和 queue 指标。

一开始不要做什么：

- 不要一开始做多机器人多会话调度。
- 不要一开始做公网部署和安全体系。
- 不要一开始重写 transport。
- 不要一开始追求每帧必处理。
- 不要一开始做无状态 server 池化。

建议的落地顺序：

1. 复用现有 gRPC 异步框架。
2. 从 `my_devs/groot_trt_async_server/run_groot_infer_trt.py` 中抽出 server-side TRT policy adapter。
3. 让服务端提供 `predict_action_chunk()` 语义。
4. 先同机 loopback，再局域网，再真机器人动作。
5. 最后才做压缩、协议瘦身和多会话增强。
