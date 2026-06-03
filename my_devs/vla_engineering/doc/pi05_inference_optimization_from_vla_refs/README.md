# PI0.5 推理优化参考仓库分析与落地方案

日期：2026-06-02

分析对象：

- 当前仓库 PI0.5：`/data/cqy_workspace/flexible_lerobot/src/lerobot/policies/pi05/`
- VLASH：`/data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering/vlash-main`
- Realtime-VLA V2：`/data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering/realtime-vla-v2-main`

## 结论

这两个仓库都能帮助优化当前仓库 PI0.5 推理，但适合迁移的层级不同。

VLASH 更适合作为第一阶段参考。它的价值不在于直接替换当前 PI0.5 模型，而在于提供一组低到中风险的 PyTorch 级推理优化和控制循环策略：`torch.compile` 预热、QKV/MLP projection fusion、bfloat16 参数策略、异步 overlap、future-state-aware action chunk 以及 action quantization。当前仓库 PI0.5 已经有 `compile_model` 和 prefix KV cache 的基础，因此更合理的做法是对齐 VLASH 的缺口，而不是搬整套 VLASH policy。

Realtime-VLA V2 更适合作为第二阶段或专项加速参考。它包含 PI0.5 Triton backend、CUDA Graph、固定 buffer、RTC action prefill、远端推理服务、时间轴平滑/MPC 和客户端多线程执行栈。它的上限更高，但迁移风险也明显更高：它依赖固定 shape、CUDA/Triton kernel、JAX/OpenPI 权重转换、离线 norm stats、机器人执行假设和服务端/客户端协议。不能把它当作即插即用的 LeRobot PI0.5 替换件。

建议路线：

1. 先做当前 PI0.5 的基线测量，确认瓶颈分布。
2. 优先验证当前已有能力：prefix cache、`torch.compile`、bf16、RTC/predict_action_chunk。
3. 从 VLASH 迁移运行时策略和轻量模型 fusion 思路。
4. 再把 Realtime-VLA V2 的 Triton/CUDA Graph 作为独立 backend 原型验证。
5. 最后再考虑机器人控制栈的异步执行、时间轴重采样和 MPC。

## 当前 PI0.5 推理链路

当前实现入口在 `src/lerobot/policies/pi05/modeling_pi05.py`。

关键证据：

- `PI05Config` 已有 `dtype`、`num_inference_steps`、`rtc_config`、`compile_model`、`compile_mode` 等字段：`src/lerobot/policies/pi05/configuration_pi05.py`。
- `PI05Pytorch.__init__` 已支持 `torch.compile(self.sample_actions, mode=config.compile_mode)`，同时也会 compile training forward：`src/lerobot/policies/pi05/modeling_pi05.py`。
- `sample_actions` 先 `embed_prefix(...)`，然后用 `use_cache=True` 计算 prefix `past_key_values`，denoise loop 中复用 `past_key_values`：`src/lerobot/policies/pi05/modeling_pi05.py`。
- `denoise_step` 每个 ODE step 仍然会构造 suffix embedding、full attention mask，并调用 `paligemma_with_expert.forward(..., inputs_embeds=[None, suffix_embs], past_key_values=past_key_values, use_cache=False)`：`src/lerobot/policies/pi05/modeling_pi05.py`。
- `select_action` 不支持 RTC，代码直接 assert：`RTC is not supported for select_action, use it with predict_action_chunk`。RTC 必须走 `predict_action_chunk` 并传入 `prev_chunk_left_over`、`inference_delay`、`execution_horizon` 等参数：`src/lerobot/policies/pi05/modeling_pi05.py`。
- `Pi05PrepareStateTokenizerProcessorStep` 会要求 `observation.state`，先 pad 到 `max_state_dim`，再把 normalized state 离散成 256 bins，并拼成 `Task: ..., State: ...;\nAction: ` prompt：`src/lerobot/policies/pi05/processor_pi05.py`。因此当前 PI0.5 虽然模型调用处写着 `no separate state needed for PI05`，state 仍然会通过 language tokens 影响 prefix。
- RTC 处理器已经存在，且 README 明确它是 flow-matching policies 的 inference enhancement，适用于 PI0、PI0.5 和 SmolVLA：`src/lerobot/policies/rtc/README.md`、`src/lerobot/policies/rtc/modeling_rtc.py`。

由此可见，当前仓库不是“完全没有优化基础”。最应该做的是测量并补齐现有实现的 runtime 使用方式、编译预热、dtype 策略、projection fusion 和异步执行策略。

## VLASH 可迁移点

### 1. 异步 action chunk 执行

VLASH 的 `VLASHAsyncManager` 在 `vlash/run.py` 中实现：

- 当前 chunk 执行到 `n_action_steps - overlap_steps` 时启动下一次推理。
- 下一次推理使用当前 chunk 末尾 action 作为 future state。
- `get_action(...)` 管理 current chunk、next chunk、chunk index。
- `run_loop(...)` 只有在需要时抓取 observation，减少相机 I/O 的频繁等待。

对当前仓库的意义：

- 如果 PI0.5 单次 chunk 推理耗时大于控制周期，异步 overlap 比单纯减少模型延迟更直接。
- 当前 PI0.5 `select_action` 是同步队列模式；要支持 RTC 或 future-state-aware，需要从 `select_action` 转向显式 `predict_action_chunk` 调度。
- 可以先做运行时包装，不改模型结构。

风险：

- VLASH 的 future state 逻辑把 `observation.state` 替换成当前 chunk 末尾 action；当前 LeRobot PI0.5 的 processor 会把 state 离散后拼进 prompt。也就是说 future-state-aware 会改变 prefix language tokens，而不只是改变一个独立 state tensor。是否能直接用 action 末尾替换 state，必须按当前 processor/tokenizer 语义验证。
- overlap 太大时，下一 chunk 使用的 future state 误差会放大；需要和 RTC 的 `inference_delay`、`execution_horizon` 一起调。

建议：

- 第一版只做调度层验证：保留当前 PI0.5 `predict_action_chunk`，外层模拟 VLASH 的 overlap timeline。
- 先不要把 VLASH policy 类替换进当前仓库。

### 2. `torch.compile` 预热

VLASH 在 `vlash/run.py` 的 `warmup_compiled_policy(...)` 会构造 dummy observation，连续调用 `policy.predict_action_chunk(...)` 触发 compile。

当前仓库已有 `compile_model` 配置，但缺少一个明确的真实部署前 warmup 流程。没有 warmup 时，第一次或前几次推理可能出现明显延迟尖峰。

建议：

- 在部署流程中加入 PI0.5 warmup 清单：固定 batch size、固定 image feature 数量、固定 token padding 长度、固定 chunk size、固定 num inference steps。
- 指标记录 warmup 前后首帧 latency、p50、p95、p99。

### 3. QKV/MLP fusion

VLASH 的 `PI05Model.init_qkv_fusion_from_existing()` 和 `init_mlp_fusion_from_existing()` 做了：

- 把 `q_proj/k_proj/v_proj` 打包成一个 `QKVLinear`。
- 把 MLP 的 `gate_proj/up_proj` 打包成一个 `MergedColumnLinear`。
- 加载权重后再执行 fusion。

对当前仓库的意义：

- 当前 PI0.5 仍通过 Transformers/Gemma 模块走分离 projection，denoise loop 重复执行 10 次，kernel launch 和 memory bandwidth 都可能是瓶颈。
- 这个方向属于中风险优化：收益可能真实，但需要严格数值对齐验证。

风险：

- 当前仓库使用 `PI05PaliGemmaWithExpert` 和 Hugging Face module 结构；VLASH 是自定义 `PI05ModelLayer` 和自定义 attention。不能直接复制函数。
- fusion 会改变 module 结构，影响 checkpoint loading、LoRA、PEFT、`torch.compile` 图和 debug。

建议：

- 先做只读验证：统计当前 PI0.5 的 q/k/v/gate/up 层名、shape、调用路径。
- 再做单层 prototype，对比 fusion 前后输出误差和 latency。
- LoRA/QLoRA 场景先禁用 fusion，或要求 merge adapter 后再 fusion。

### 4. bf16 策略

VLASH 默认 `dtype: bfloat16`，并在模型内选择性保留部分 embedding/norm 参数为 float32。当前 PI0.5 配置默认 `dtype: float32`，但代码也支持 `bfloat16`。

建议：

- 第一阶段直接对比 `dtype=float32` 和 `dtype=bfloat16`。
- 单独记录 action chunk 数值误差、机器人任务成功率、GPU 显存和 latency。
- 不建议盲目全量 fp16，因为 PI0.5 的 denoise 和 adaRMS/normalization 对精度更敏感，bf16 更稳。

### 5. Action quantization / 降低执行频率

VLASH 的 `run_loop(...)` 支持 `action_quant_ratio`：不是每个推理 step 都 `send_action`，而是按比例发送动作。README 也给出 `--action_quant_ratio=2` 的运行示例。

对当前仓库的意义：

- 这不是模型推理变快，而是降低控制发送负载，提高执行速度或减少 I/O 压力。
- 对高频低层控制器可能有价值，但对需要细腻接触的任务有风险。

建议：

- 只在执行栈瓶颈明显时启用。
- 和轨迹平滑、速度限制一起验证。

## Realtime-VLA V2 可迁移点

### 1. Triton PI0.5 backend + CUDA Graph

Realtime-VLA V2 的 `server/pi05_infer.py` 提供了 PI0.5 推理 backend：

- `pi05_model(...)` 分为 `vision_encoder(...)`、`transformer_encoder(...)`、`transformer_decoder(...)`。
- `Pi05Inference.__init__` 预分配大量 CUDA buffer 和 bf16 weights。
- 初始化时预计算 rope table、time embeddings、adaRMS modulation。
- 使用 `torch.cuda.CUDAGraph()`，`record_infer_graph()` warmup 后 capture，然后 `forward(...)` 中只 copy input buffer 并 replay graph。

这是模型级延迟优化的高上限方案。它绕过通用 PyTorch/Transformers 调度，直接面向固定 shape、固定 num steps、固定 chunk size 和固定 CUDA kernel。

强约束：

- 代码硬编码大量 shape，例如 vision patch、hidden size、decoder hidden、chunk/action max dim、num steps=10。
- 输入期望包括 `observation_images_normalized`、`diffusion_noise`、prompt/state token embedding。
- 权重不是直接 LeRobot `model.safetensors` 形态，而是通过转换脚本整理成 backend 需要的 key。`server/convert_from_jax_pi05.py` 和 `server/convert_from_jax_pi05rtc.py` 从 JAX/OpenPI 权重生成包含 `embedding_weight`、`language_embeds` 等字段的 pickle 权重。
- Triton adapter 还依赖离线 `norm_stats.json`。`server/model.py` 的 `_load_norm_stats(...)` 读取 state/action 的 `q01/q99`，配置文件也要求 `norm_stats_dir`，因此要接当前 LeRobot checkpoint 时必须对齐 normalization 统计来源。

建议：

- 作为独立 backend 验证，不要直接替换当前 `PI05Policy`。
- 先写验证计划：同一 checkpoint、同一 tokenizer、同一 dummy observation、同一 noise，比较 Torch PI0.5 vs Triton PI0.5 action chunk。
- 误差门槛建议按任务分层：单步 velocity/action chunk 数值误差、完整 denoise 输出误差、真实机器人任务成功率。

### 2. RTC / action prefill backend

Realtime-VLA V2 的 `server/pi05rtc_infer.py` 在 Triton backend 上加入：

- `prefill_actions`
- `prefill_mask`
- `prefill_inv_mask`
- `action_prefill_len`
- 根据 prefill length 更新 runtime adaRMS mods

这与当前仓库 `src/lerobot/policies/rtc/modeling_rtc.py` 的思路相关，但实现方式不同：

- 当前仓库 RTC 是 PyTorch denoise wrapper，通过 autograd correction 引导当前 chunk 靠近 previous leftover。
- Realtime-VLA V2 RTC backend 更接近 action prefill / realtime chunking，把已执行或计划动作作为条件塞入 decoder buffer。

建议：

- 不要把两个 RTC 实现混用。
- 第一阶段先用当前仓库已有 RTC，验证 `predict_action_chunk` 外层调度是否正确。
- 第二阶段如果走 Triton backend，再单独验证 action prefill 的等价性和收益。

### 3. 远端服务和客户端多线程执行

Realtime-VLA V2 的 `server/infer_server.py` 是 FastAPI + pickle RPC；`client/local_client.py` 启动 state、image、heartbeat、inference 等线程。

对当前仓库的意义：

- 可以把 GPU 推理和机器人本机 I/O 解耦。
- 可以显式测量 `roundtrip_latency_s` 和 `server_infer_time_s`。
- 能用 pending action queue 做延迟补偿。

风险：

- pickle RPC 不适合作为长期通用接口，安全性和跨语言兼容性都弱。
- 网络延迟会吞掉部分模型加速收益。
- 当前机器人抽象、camera key、action dim/order 可能和 Realtime-VLA V2 的 AIRBOT 双臂假设不同。

建议：

- 若当前部署是同机 GPU，先不要引入远端服务。
- 若机器人本机没有足够 GPU，才考虑 remote inference service。
- 协议应重新设计为受控 schema，不建议直接照搬 pickle payload。

### 4. Time-axis smooth / MPC

Realtime-VLA V2 的 `server/optimizer.py` 提供 `TimeParameterizationMPC`，根据 action waypoints 优化时间轴；client executor 还有 raw action、smooth、on-device MPC 等逻辑。

对当前仓库的意义：

- 这是执行质量优化，不是纯模型推理优化。
- 它能让 chunk 输出在硬件速度限制下更平滑、更快地执行。

建议：

- 在模型推理延迟稳定后再做。
- 先做离线 action log replay，验证速度、加速度、jerk、限位和任务成功率。

## 不建议直接迁移的内容

1. 不建议直接用 VLASH 的 `vlash/policies/pi05/modeling_pi05.py` 替换当前 LeRobot PI0.5。

   原因：VLASH 自定义了 PI0.5 model layer、state conditioning、normalization 默认值、weight mapping 和 fusion 路径；当前仓库 PI0.5 是 OpenPI/LeRobot 风格实现，且已有自己的 from_pretrained key remapping、processor 和 RTC。

2. 不建议直接把 Realtime-VLA V2 的 `server/pi05_infer.py` 接进当前 `PI05Policy.select_action`。

   原因：该 backend 是固定 shape Triton/CUDA Graph 专用实现，需要专门 checkpoint 转换、buffer 约束和数值验证。直接接入会绕开当前 processor、normalizer、tokenizer 和 policy abstraction。

3. 不建议一开始就优化机器人客户端/MPC。

   原因：如果模型端 latency、dtype 和 chunk 调度还没测清，控制栈优化会掩盖真实瓶颈。

## 分阶段方案

### 阶段 0：基线测量

目标：知道当前 PI0.5 推理慢在哪里。

测量项：

- `predict_action_chunk` 端到端 latency。
- `_preprocess_images` latency。
- token/mask 准备 latency。
- `embed_prefix` latency。
- prefix prefill latency。
- 单个 `denoise_step` latency 和 10 step 总 latency。
- action unpad/unnormalize latency。
- GPU 显存峰值。
- 首次调用、warmup 后 p50/p95/p99。

建议固定：

- batch size = 1。
- image features 和真实机器人一致。
- chunk_size = 50。
- num_inference_steps = 10。
- tokenizer max length 与 checkpoint config 一致。
- 同一 noise seed 便于数值对比。

### 阶段 1：低风险启用当前已有优化

目标：不改模型结构，先用好当前仓库已有能力。

动作：

- 启用 `dtype=bfloat16` 对比 `float32`。
- 启用 `compile_model=True`，并增加部署前 warmup。
- 确认 `sample_actions` 的 prefix KV cache 正常命中。
- 用 `predict_action_chunk` 替代高频 `select_action` 外层循环，便于 RTC 和 async 调度。
- 单独验证 `rtc_config` 在 PI0.5 上的行为，不通过 `select_action` 使用 RTC。

验收：

- latency 有可重复下降。
- action chunk 数值误差在可接受范围。
- 无 compile 首帧尖峰进入真实控制阶段。

### 阶段 2：迁移 VLASH 运行时策略

目标：隐藏推理延迟，而不是只缩短模型执行时间。

动作：

- 实现一个不侵入模型的 async chunk scheduler 设计，参考 `VLASHAsyncManager`。
- 支持 `inference_overlap_steps`。
- 支持 future-state-aware 但默认关闭，先用当前 state 验证。
- 支持 action queue、next chunk、latency logging。
- 评估 `action_quant_ratio` 是否适合当前机器人控制频率。

验收：

- 控制循环不再等待每个 chunk 推理完成。
- pending chunk 切换无空洞。
- 记录 inference start/end、chunk switch、sent action timestamp。

### 阶段 3：迁移 VLASH 模型内轻量 fusion 思路

目标：减少 denoise loop 内重复 projection 的 kernel launch 和 memory bandwidth。

动作：

- 先只做设计和验证，不直接改主干。
- 对当前 PI0.5 统计 q/k/v/gate/up 层结构。
- 做单层 fusion 数值等价验证。
- 再做全模型 fusion latency 验证。
- 明确 LoRA/QLoRA、checkpoint loading、compile、RTC 的兼容矩阵。

验收：

- fusion 前后同输入输出误差可控。
- latency 有稳定收益。
- 回退开关明确。

### 阶段 4：Realtime-VLA V2 Triton backend 专项

目标：验证高性能 backend 是否值得投入。

动作：

- 梳理当前 LeRobot PI0.5 `model.safetensors` 到 Realtime-VLA V2 checkpoint dict 的 key mapping。
- 对齐 image normalization、language tokenization、state token/prompt 格式。
- 固定 shape：num_views、prompt max length、chunk_size、action_dim、num_steps。
- 建立 Torch vs Triton 对比脚本设计。
- 先跑 mock/dummy，再跑真实 checkpoint 离线 observation。

验收：

- Torch 与 Triton 输出误差可解释。
- CUDA Graph replay latency 明显低于 PyTorch baseline。
- backend 输入输出能被当前 policy/runtime 包装。

### 阶段 5：控制栈实时化

目标：把模型输出变成更稳定、更快的机器人执行。

动作：

- 按需引入远端 inference service。
- 引入 action timeline logging。
- 评估 time-axis smooth 或 MPC。
- 对接当前机器人的 action order、限位、速度/加速度限制。

验收：

- 日志能对齐 observation timestamp、request timestamp、response timestamp、action send timestamp。
- 执行速度提升不牺牲成功率和安全边界。

## 推荐优先级

P0：基线 profiler、bf16、compile warmup、明确 `predict_action_chunk` 路径。

P1：VLASH async overlap 调度、RTC 正确用法、latency logging。

P2：QKV/MLP fusion 原型验证。

P3：Realtime-VLA V2 Triton/CUDA Graph backend 原型。

P4：remote inference service、time-axis smooth/MPC、完整客户端执行栈。

## 验证清单

每个阶段都要保留以下证据：

- 使用的 conda 环境：`lerobot_flex`。
- git commit 或 worktree 摘要。
- checkpoint 路径和 config。
- 输入 features、camera names、state/action dim。
- latency 表：mean、p50、p95、p99、首帧、warmup 后。
- GPU 信息、CUDA/Torch/Triton 版本。
- 数值对比：同 noise、同 observation、同 prompt 下的 action chunk 差异。
- 真实机器人验证时的任务成功率、失败样例、延迟日志和安全事件。

## 最终建议

短期最值得做的是“VLASH 式运行时 + 当前 LeRobot PI0.5 已有优化能力”的组合：`bf16 + compile warmup + prefix cache确认 + async chunk scheduler + RTC/predict_action_chunk 正确调度`。这条路线工作量适中，能较快提升实时性，也不会破坏当前仓库结构。

中期再做 projection fusion。它可能有稳定收益，但必须用当前模型结构重新实现和验证。

长期如果目标是极限延迟，再立项 Realtime-VLA V2 Triton backend。它值得验证，但应该作为独立 backend，而不是直接侵入当前 `PI05Policy`。
