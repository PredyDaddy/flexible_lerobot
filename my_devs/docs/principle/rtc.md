# RTC（Real-Time Chunking）原理与本仓库实现解读

## 策略背景
本仓库里的 `RTC` 指 Physical Intelligence 在 2025 年提出的 **Real-Time Chunking**，不是泛指“实时轨迹控制器”。它解决的是 action chunking 策略落地到真实机器人时的时延问题：模型一次生成整段动作，但推理完成时，旧 chunk 往往已执行了一部分，若直接切换到新 chunk，边界容易抖动。因而，RTC 在外部来源上是有明确论文依据的推理算法；但在本仓库中，`src/lerobot/policies/rtc/` 更准确地说是 **该算法的工程封装**，不是独立 policy。依据是 `README.md` 明确写了 “RTC is not a policy itself”，且这里只定义配置、处理器和调试/队列逻辑，没有单独训练网络。

## 核心原理
RTC 把“新 chunk 的开头”视作 **轨迹补全 / inpainting**：前缀尽量贴合上一个 chunk 尚未执行的剩余轨迹，后缀仍允许模型依据新观测重规划。仓库实现里，基础策略先预测去噪速度 `v_t`，RTC 再利用 `prev_chunk_left_over`、`inference_delay`、`execution_horizon` 生成前缀权重，对重叠区做更强引导。`modeling_rtc.py` 中最终速度由“原始速度 - 引导修正项”得到，且修正强度会随时间变化并受 `max_guidance_weight` 截断，所以它不是简单复制旧动作，而是让前缀连续、后缀可更新。

## 在本仓库中的实现解读
`configuration_rtc.py` 只暴露少量核心参数：`enabled`、`prefix_attention_schedule`、`execution_horizon`、`max_guidance_weight` 以及调试开关。`modeling_rtc.py` 的 `RTCProcessor` 是核心实现，它并不替代原策略，而是包裹原始 `denoise_step`；首个 chunk 没有 leftover 时直接走基础去噪，后续 chunk 才根据 leftover 和重叠区权重做修正。权重 schedule 支持 `ZEROS/ONES/LINEAR/EXP` 四种形式，其中 `LINEAR/EXP` 都体现“前强后弱”。`README.md` 进一步限定了边界：RTC 仅服务于 `pi0`、`pi0.5`、`SmolVLA` 等 flow-matching policy。

## 训练与推理机制
从调用链看，RTC **主要作用于推理，不改现有训练主干**。以 `pi0/modeling_pi0.py` 为例，训练 `forward()` 仍直接计算原始动作 loss，没有 RTC 专属 loss；而推理 `sample_actions()` 时，才把 `inference_delay`、`prev_chunk_left_over`、`execution_horizon` 传给 `rtc_processor.denoise_step()`。`action_queue.py` 则负责保存旧 chunk 的未执行部分，并在 RTC 模式下按 `real_delay` 丢弃推理期间已经过时的动作，再用新 chunk 替换队列。因此，本仓库实现的是“执行期纠偏版 RTC”。截至 2026-03-18，虽然外部已有 2025-12 的训练期 RTC 后续工作，但当前仓库代码并未暴露对应训练配置或损失。

## 关键超参数及影响
`execution_horizon` 越大，边界越平滑，但也越可能压制对新观测的快速修正；过小则连续性不足。`prefix_attention_schedule` 决定哪些重叠步受强约束，`ONES` 更激进，`LINEAR` 更均衡，`EXP` 更强调最前面的几步。`max_guidance_weight` 是纠偏上限，过大可能过度黏住旧轨迹，过小则收益有限。运行时的 `inference_delay` 虽不是静态配置，却直接决定重叠区长度，是 RTC 实际效果的关键量。

## 适用场景、优势与局限
RTC 适合真实机器人部署、存在明显推理时延、且策略按 chunk 输出动作的场景。优势是无需改写主模型、可直接作为推理增强接入，并能明显缓解 chunk 边界抖动。局限也很清楚：它不是通用控制器，而是依附于 flow-matching chunk policy 的执行技术；参数敏感；若环境突变很大或延迟接近 chunk 长度，前缀连续性与即时修正之间会产生冲突。总体上，应把 RTC 理解为“有明确论文来源的推理算法 + 本仓库中的统一工程封装”。

## 参考资料
1. Physical Intelligence, Real-Time Chunking 研究页：https://www.physicalintelligence.company/research/real_time_chunking
2. Kevin Black, Manuel Y. Galliker, Sergey Levine, Real-Time Execution of Action Chunking Flow Policies：https://arxiv.org/abs/2506.07339
3. Physical Intelligence, OpenPI 官方仓库：https://github.com/Physical-Intelligence/openpi
4. Physical Intelligence, real-time-chunking-kinetix 官方仓库：https://github.com/Physical-Intelligence/real-time-chunking-kinetix
5. Physical Intelligence 后续训练期 RTC 工作（用于区分本仓库当前未实现的方向）：https://arxiv.org/abs/2512.05964
