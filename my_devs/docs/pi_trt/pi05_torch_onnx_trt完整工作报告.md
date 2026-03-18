# PI0.5 Torch / ONNX / TensorRT 完整工作报告

最后更新：2026-03-11  
仓库：`/data/cqy_workspace/flexible_lerobot`  
开发环境：`conda env = lerobot_flex`  
目标模型：`pi05 / PI0.5`  
当前结论范围：**只讨论离线导出、离线一致性验证、离线 mock 验证、离线 benchmark，不控制机器人，不发动作。**

---

## 1. 这份报告是给谁看的

这份报告是写给两类人的：

1. **已经熟悉模型工程的人**  
   希望快速知道：这个模型是怎么从 Torch 拆成 ONNX，再变成 TensorRT 的；脚本在哪；结果怎样；还有哪些 caveat。

2. **不熟悉 Torch / ONNX / TensorRT 的人**  
   希望知道：我到底做了什么、为什么这么做、每一步对应模型的哪一段、最终达到了什么效果。

所以这份报告会同时包含：

- 非常直白的概念解释
- 代码和张量边界解释
- 可以直接复跑的命令
- 已经跑出的真实产物和结果
- 当前脚本的已知局限

---

## 2. 一句话总结

我把 `pi05` 的 Torch 推理链路拆成了 3 个固定 shape 的子图：

1. `vision_encoder`
2. `prefix_cache`
3. `denoise_step`

然后完成了下面这条离线链路：

1. Torch FP16 基线准备
2. ONNX 导出
3. Torch FP16 vs ONNX FP16 一致性验证
4. TensorRT engine build
5. Torch FP16 vs TensorRT FP16 一致性验证
6. 不控制机器人的 mock runtime 对比
7. Torch vs TensorRT 离线 benchmark

**核心结论：**

- ONNX 一致性通过
- TRT 一致性通过
- 不接机器人、不发动作的离线 mock 路径可跑通
- TRT 在当前固定 shape 基线下，模块级和闭环级都已经有明确收益

但同时也有几个重要 caveat：

- 当前 `benchmark_trt.py` 里的 `action_chunk_pipeline` 速度对比不是完全 apples-to-apples
- 当前 `run_pi05_infer_trt_mock.py` 的 action-level compare 混入了 dtype 差异，不能把它当成纯粹的 `Torch FP16 vs TRT FP16` 最终精度证明
- 当前脚本默认只支持**固定 shape、batch=1、2 相机、token 长度 200、chunk=50、无 RTC** 的基线

---

## 3. 先解释三个概念：Torch、ONNX、TensorRT 到底是什么

### 3.1 Torch 是什么

这里的 Torch 指的是：

- 原始 `pi05` 模型在 PyTorch 里的实现
- 也就是训练时、最原始推理时所使用的模型代码

它的优点是：

- 实现最完整
- 最容易改逻辑
- 最接近原始论文和原始代码结构

它的缺点是：

- 推理速度不一定最好
- 部分图结构不够“部署友好”

### 3.2 ONNX 是什么

ONNX 可以理解成：

- 一种“中间表示格式”
- 目标是把 PyTorch 模型里一段稳定的计算图，导出成更标准的图结构

在这条链路里，ONNX 的作用不是最终部署，而是：

- 验证我们拆图拆得对不对
- 验证数值是否还对齐
- 给 TensorRT 提供输入

### 3.3 TensorRT 是什么

TensorRT 是 NVIDIA 的推理优化后端。

在这条链路里，它的作用是：

- 读取 ONNX
- 构建 GPU 上更适合推理的 engine
- 在固定 shape、固定 dtype 下把推理做得更快

所以整个关系是：

```text
Torch 原始模型
    -> 导出成 ONNX 子图
    -> 再由 ONNX build 成 TensorRT engine
```

也就是说：

- `Torch` 是原始参考实现
- `ONNX` 是中间桥梁
- `TensorRT` 是最终高性能推理后端

---

## 4. 为什么 `pi05` 不能直接照抄 GR00T 的 TRT 拆法

仓库里已经有一套比较成熟的 `GR00T TRT` 方案，路径在：

- `my_devs/groot_trt/`

但是 `pi05` 和 GR00T 的模型结构不一样，不能硬照抄。

### 4.1 最关键的差异：`pi05` 的 state 不是单独一个 encoder

`pi05` 的 processor 会先把 `observation.state`：

1. pad 到 32 维
2. 归一化
3. 离散化成 token
4. 拼进 prompt 文本
5. 再交给 tokenizer

也就是说，对 `pi05` 来说：

- state 不是独立的数值输入分支
- state 已经被“文本化”
- 所以不能像 GR00T 那样单独搞一个 `state_encoder.onnx`

### 4.2 `pi05` 的推理天然分成 prefix 和 suffix 两段

从源码看，`pi05` 推理本质上是：

1. 先把图像和语言拼成 prefix
2. 跑一次 prefix language model 得到 `past_key_values`
3. 然后在 suffix 上循环做 10 步 denoise

这决定了最自然的拆法不是 GR00T 的 7 段，而是：

1. `vision_encoder`
2. `prefix_cache`
3. `denoise_step`

### 4.3 所以我的策略是

不是“照抄 GR00T 的模块边界”，而是：

- **沿用 GR00T 的方法论**
- **重做更适合 `pi05` 的拆图边界**

GR00T 给我的主要价值是：

- `Torch -> ONNX -> TensorRT engine -> consistency compare` 这个工程套路
- `trt_utils.py` 这类可复用工具
- mock runtime 的思想

---

## 5. `pi05` 原始 Torch 推理链路到底长什么样

这一段非常重要，因为后面的 ONNX 和 TRT 都是在对应这条 Torch 路径。

在源码里，关键入口是：

- `src/lerobot/policies/pi05/modeling_pi05.py`

高层推理流程可以理解为：

```text
observation
    -> preprocessor
    -> policy.predict_action_chunk(...)
    -> model.sample_actions(...)
    -> prefix 构造
    -> prefix cache
    -> 10-step denoise
    -> action chunk
    -> postprocessor
```

### 5.1 预处理阶段

预处理不是 TRT 做的，也不是 ONNX 做的，而是继续保留在 Python / checkpoint processor 侧。

这部分包括：

- `policy_preprocessor.json`
- tokenizer：`google/paligemma-3b-pt-224`
- 状态归一化
- state 离散化拼 prompt
- 输出后再走 `policy_postprocessor.json`

这一步我明确保留为“原样不动”，原因是：

- checkpoint 保存下来的 processor 才是部署真相
- 代码默认 config 不能替代 checkpoint 里的 processor

### 5.2 Vision 阶段

每个相机输入图像会：

1. resize / pad 到 `224 x 224`
2. 从 `[0, 1]` 归一化到 `[-1, 1]`
3. 进入 PaliGemma vision tower
4. 得到图像 token embedding

在当前双相机配置下：

- 每个相机 256 个视觉 token
- 两个相机合计 512 个视觉 token

### 5.3 Prefix 阶段

这一步把：

- 视觉 embedding
- 语言 embedding

拼成 prefix。

当前固定 shape 基线下：

- 语言 token 长度固定为 200
- prefix 总长度固定为 `512 + 200 = 712`

所以 prefix 关键张量是：

- `prefix_embs`: `[1, 712, 2048]`
- `prefix_pad_masks`: `[1, 712]`
- `prefix_attention_mask_4d`: `[1, 1, 712, 712]`
- `prefix_position_ids`: `[1, 712]`

### 5.4 Prefix Cache 阶段

这一步运行的是 prefix-only 的 language model 前向，目标不是直接出 action，而是得到后续 denoise 要反复复用的 `past_key_values`。

当前 `pi05` 配置下，cache 结构可以整理成：

- 层数：18
- 每层 `(key, value)`
- 每个 cache tensor 形状：`[1, 1, 712, 256]`

为了更适合 ONNX 和 TRT，我把它堆成一个大 tensor：

- `kv_cache`: `[18, 2, 1, 1, 712, 256]`

这样后面：

- ONNX 更好导出
- TRT 更好 build
- Python glue 更简单

### 5.5 Denoise 阶段

然后进入真正的 action chunk 生成。

每次会循环 `num_inference_steps = 10` 次：

1. 当前 `x_t`
2. 当前 `timestep`
3. `prefix_pad_masks`
4. `kv_cache`

一起送进 denoise step，输出：

- `velocity`: `[1, 50, 32]`

随后做 Euler 更新：

```text
x_t = x_t + dt * velocity
```

10 步结束后，得到：

- action chunk

---

## 6. 我是怎么把 Torch 拆成 ONNX 的

这一部分是整条工程链最核心的设计。

### 6.1 为什么拆成 3 段

我最后选择的 ONNX 拆分是：

1. `vision_encoder_fp16.onnx`
2. `prefix_cache_fp16.onnx`
3. `denoise_step_fp16.onnx`

这是因为：

- `vision_encoder` 结构清晰，单独导出最容易
- `prefix_cache` 正好把 prefix pass 和 suffix pass 分开
- `denoise_step` 是最核心、最稳定、最适合重复调用的推理单元

### 6.2 这 3 段分别对应 Torch 里的哪段逻辑

#### `vision_encoder_fp16.onnx`

对应 Torch 里的：

- `policy.model.paligemma_with_expert.embed_image(...)`

输入：

- `image_tensor`: `[2, 3, 224, 224]`

输出：

- `image_embeddings`: `[2, 256, 2048]`

#### `prefix_cache_fp16.onnx`

对应 Torch 里的：

- prefix-only language model 前向
- 从 prefix embedding 产出 KV cache

输入：

- `prefix_embs`: `[1, 712, 2048]`
- `prefix_attention_mask_4d`: `[1, 1, 712, 712]`
- `prefix_position_ids`: `[1, 712]`

输出：

- `kv_cache`: `[18, 2, 1, 1, 712, 256]`

#### `denoise_step_fp16.onnx`

对应 Torch 里的单步 suffix denoise：

- 输入当前 `x_t`
- 利用 prefix cache
- 生成单步 `velocity`

输入：

- `x_t`: `[1, 50, 32]`
- `timestep`: `[1]`
- `prefix_pad_masks`: `[1, 712]`
- `kv_cache`: `[18, 2, 1, 1, 712, 256]`

输出：

- `velocity`: `[1, 50, 32]`

### 6.3 导出 wrapper 在哪里

导出 wrapper 和 Torch glue 主要在：

- `my_devs/pi_trt/common.py`

这里面最关键的内容有：

- `prepare_policy_for_fp16`
- `prepare_batch`
- `build_prefix_from_image_embeddings`
- `compute_prefix_cache`
- `compute_denoise_step`
- `compute_action_chunk`
- `VisionEncoderOnnxWrapper`
- `PrefixCacheOnnxWrapper`
- `DenoiseStepOnnxWrapper`

可以把这个文件理解成：

> **PI0.5 Torch / ONNX / TRT 的公共语义定义层。**

也就是说，Torch reference、ONNX compare、TRT compare、mock runtime、benchmark 都靠它统一张量边界和公共 glue。

---

## 7. ONNX 导出时我实际踩到的坑

这一段不是可有可无，而是后面 TRT 能否顺利接上的关键。

### 7.1 `denoise_step` 里的 cache 不能直接还原成普通 tuple

最开始我尝试：

- 把堆叠好的 `kv_cache` 还原成普通 `(key, value)` tuple
- 再喂给 `gemma_expert.model.forward(...)`

结果失败。

本质原因是：

- 当前 `transformers` 版本需要的是 `DynamicCache`
- 普通 tuple 没有 `get_seq_length()` 等接口

最终修法是：

- 使用 `DynamicCache.from_legacy_cache(...)`

### 7.2 ORT 不接受 `CumSum(bool)`

Torch 对 bool mask 做 `cumsum` 没问题，但 ORT 不接受这种图。

所以后来我强制把所有会走 `cumsum` / `sum` 的 mask 显式转成 `int64`。

这包括：

- `prefix_position_ids`
- `denoise_step` 里的 `position_ids`
- `make_att_2d_masks` 里用到的 `att_masks`

### 7.3 ORT CPU 对 `Cos(float64)` 路线敏感

时间位置编码原始实现里会走 `float64`。

这在 Torch 里没问题，但 ORT CPU 在 `Cos` 节点上会报错。

最终修法是：

- 为导出路径单独实现 `float32` 的时间正余弦 embedding
- 在进 `time_mlp` 前再 cast 到 `fp16`

这一步不是数学上“更对”，而是为了让导出图在 ORT / TRT 上都能稳定运行。

---

## 8. ONNX 导出产物在哪里、长什么样

当前已经跑出的 ONNX 产物目录是：

- `outputs/pi_trt/pi05_onnx_consistency_20260306_190504/pi05_onnx`

主文件大致大小：

- `vision_encoder_fp16.onnx`: `792M`
- `prefix_cache_fp16.onnx`: `390K`
- `denoise_step_fp16.onnx`: `821M`

除此之外还有大量：

- `onnx__MatMul_*`
- `onnx__Mul_*`

这些不是垃圾文件，也不是异常，而是：

- ONNX external data sidecar
- 主 `.onnx` 和 sidecar 文件必须放在同一个目录下

所以如果你要搬运 ONNX：

> **不能只拷贝 3 个 `.onnx` 主文件，必须把整个目录一起保留。**

---

## 9. Torch vs ONNX 一致性是怎么做的

对应脚本：

- `my_devs/pi_trt/compare_torch_onnx.py`

### 9.1 为什么不是从机器人 observation 开始比

因为这一步的目标不是“验证真实机器人链路”，而是：

- 验证 Torch reference 和 ONNX 子图在**相同语义输入**上的一致性

所以我采用了更稳定的方法：

1. 读取 checkpoint 保存的 `policy_preprocessor.json`
2. 构造 synthetic observation
3. 走 checkpoint preprocessor
4. 在模型内部做图像 resize / normalize
5. 分别送入 Torch FP16 和 ONNX

### 9.2 比较的 4 个层级

我没有只比一个输出，而是分 4 层比较：

1. `vision_encoder`
2. `prefix_cache`
3. `denoise_step`
4. `action_chunk_pipeline`

也就是说：

- 既看局部
- 也看完整 10 步闭环

### 9.3 ONNX 一致性结果

第一轮：

- `action_chunk_pipeline` cosine = `0.9999969599`

第二轮：

- `action_chunk_pipeline` cosine = `0.9999988286`

所以 ONNX 阶段我判断为：

> **已经对齐成功。**

---

## 10. 我是怎么从 ONNX build 成 TensorRT engine 的

对应脚本：

- `my_devs/pi_trt/build_engine.py`

### 10.1 这个脚本的作用

它的作用很简单：

- 读取 ONNX 文件
- 调 TensorRT Python API
- build 成 `.engine`
- 写出 `build_report.json`

### 10.2 为什么它不是直接用 `trtexec`

因为仓库里 `GR00T TRT` 方案已经验证过：

- 用 Python API 更容易集成进脚本
- 更容易记录 build metadata
- 更容易控制后续 compare/runtime

### 10.3 TensorRT import 是怎么做的

这个脚本复用了：

- `my_devs/groot_trt/trt_utils.py`

所以：

- 不要求必须把 TensorRT 安装进当前 conda env
- 如果默认 import 不到，会尝试 repo-local / data-local 的 TensorRT 目录

当前机器上实际使用的 TensorRT 版本是：

- `10.13.0.35`

### 10.4 为什么 build 时要按文件路径 parse ONNX

`pi05` 的大 ONNX 使用了 external data sidecar。

如果直接读 bytes 再 parse，有可能读不到 sidecar。

所以在 `build_engine.py` 里我专门按：

- `parser.parse_from_file(str(onnx_path))`

这种方式来 parse。

### 10.5 当前 build 出来的 engine

产物目录：

- `outputs/pi_trt/pi05_onnx_consistency_20260306_190504/pi05_engine_api_trt1013`

当前大小大致是：

- `vision_encoder_fp16.engine`: `794M`
- `prefix_cache_fp16.engine`: `3.5G`
- `denoise_step_fp16.engine`: `825M`

这里最需要注意的是：

- `prefix_cache_fp16.engine` 非常大

这不是 build 失败，而是当前图和 TensorRT 优化后的真实产物体积。

这会带来两个现实影响：

1. 更高的磁盘占用
2. 对较小显卡的可迁移性会更差

---

## 11. Torch vs TRT 一致性是怎么做的

对应脚本：

- `my_devs/pi_trt/compare_torch_trt.py`

### 11.1 它和 ONNX compare 的关系

可以把它理解成：

- `compare_torch_onnx.py` 的 TensorRT 版本

也就是：

- Torch reference 仍然走 `common.py` 的统一公共语义
- 只不过 ONNX Runtime 换成了 TensorRT `TrtSession`

### 11.2 它比较了哪 4 层

和 ONNX compare 一样，比较：

1. `vision_encoder`
2. `prefix_cache`
3. `denoise_step`
4. `action_chunk_pipeline`

### 11.3 TRT 一致性结果

第一轮结果：

- `vision_encoder` cosine = `0.99999446`
- `prefix_cache` cosine = `0.99974460`
- `denoise_step` cosine = `0.99999764`
- `action_chunk_pipeline` cosine = `0.99999268`

第二轮结果：

- `vision_encoder` cosine = `0.99999471`
- `prefix_cache` cosine = `0.99973636`
- `denoise_step` cosine = `0.99999780`
- `action_chunk_pipeline` cosine = `0.99998071`

### 11.4 我如何解释这些结果

如果只从“核心 TRT 推理链路是否对齐”这个角度看：

- 是通过的

原因是：

- 两轮都过了
- 最接近最终行为的 `action_chunk_pipeline` cosine 仍然非常高
- 单步 `denoise_step` 也非常稳定

### 11.5 但这里有一个需要诚实写出来的 caveat

当前 `compare_torch_trt.py` 里的模块级指标，并不是完全 stage-isolated。

比如：

- `prefix_cache` 的比较，是建立在 TRT `vision_encoder` 输出上的
- `denoise_step` 的比较，又建立在 TRT `prefix_cache` 输出上的

所以这些模块级指标：

- 更像“累计链路误差”
- 不应被过度解释成“单模块严格隔离误差”

这不影响“整条 TRT 链路已经基本对齐”的结论，但会影响你如何解读模块级数字。

---

## 12. `run_pi05_infer_trt_mock.py` 是干什么的

这是一个**非常重要但容易被误解**的脚本。

路径：

- `my_devs/pi_trt/run_pi05_infer_trt_mock.py`

### 12.1 它不是机器人控制脚本

它的目标是：

- 不接机器人
- 不发动作
- 在同一批 frame 上同时跑：
  - PyTorch baseline
  - TRT adapter
- 然后比较动作输出

所以这是：

> **一个离线 mock compare 工具，而不是上线推理工具。**

### 12.2 它支持什么输入

当前只支持：

- `--source random`
- `--source frames_dir`

也就是说：

- 可以自己生成随机 frame
- 也可以重放已经保存好的 frame

这正是为了保证：

- 不碰机器人
- 但仍能走完整的 preprocessor -> predict_action -> queue 语义

### 12.3 它内部做了什么

它实现了一个：

- `TrtPi05PolicyAdapter`

这个 adapter 的职责是：

- 模拟 policy-like 对象
- 保留 `select_action()` / `_action_queue` 语义
- 在 chunk 为空时，用 TRT engine 计算一个新的 action chunk
- 后续 step 则像原 policy 一样从 queue 里出动作

### 12.4 当前 mock compare 的结果

随机源和 replay 源两次结果一致：

- action cosine = `0.9999938566`
- rmse = `0.1815061129`
- mean_abs = `0.1332122871`
- max_abs = `0.5232925415`

### 12.5 这个结果该怎么解释

这个结果说明：

- 在完整 `predict_action` 调用语义下
- TRT adapter 和 PyTorch baseline 行为非常接近

但这里还有两个 caveat：

#### caveat A：当前 mock compare 混入了 dtype 差异

当前脚本里：

- Torch baseline 没有先统一转成 FP16
- TRT glue policy 则显式转成了 FP16

这意味着 mock 报告里的 action-level 差异：

- 既包含 backend 差异
- 也包含 dtype 差异

所以它更适合被解释成：

> **完整 runtime glue 的 smoke check**

而不是：

> **严格纯粹的 TensorRT 精度证明**

请把这句话单独记住：

> **当前 mock compare 结果不能被解释为纯净的 Torch FP16 vs TRT FP16 最终精度证明。**

#### caveat B：`fps_effective` 不能当真机 chunk latency

原因是：

- `select_action()` 是 queue-backed
- 很多 step 只是从 queue 里 pop 动作
- 并没有重新算 chunk

所以这个脚本里的：

- `fps_effective`

更像“loop throughput”，不是“真实 chunk 生成时延”。

---

## 13. `benchmark_trt.py` 是干什么的

路径：

- `my_devs/pi_trt/benchmark_trt.py`

### 13.1 它测什么

它离线 benchmark 这 4 段：

1. `vision_encoder`
2. `prefix_cache`
3. `denoise_step`
4. `action_chunk_pipeline`

输出内容包括：

- `avg_ms`
- `p50_ms`
- `p95_ms`
- `min_ms`
- `max_ms`
- `stdev_ms`
- `speedup_vs_torch`

### 13.2 当前 benchmark 结果

当前 20 次采样下：

- `vision_encoder`: `6.34ms -> 3.38ms`, `1.87x`
- `prefix_cache`: `25.19ms -> 15.74ms`, `1.60x`
- `denoise_step`: `6.08ms -> 2.40ms`, `2.53x`
- `action_chunk_pipeline`: `92.05ms -> 24.06ms`, `3.83x`

### 13.3 这里也必须写一个关键 caveat

当前 `action_chunk_pipeline` 的 benchmark **不是完全 apples-to-apples**。

原因是：

- Torch 侧测的是完整 `compute_action_chunk(...)`
- TRT 侧当前只测了 denoise loop，`vision_encoder` 和 `prefix_cache` 是预计算后再计时

所以：

- `vision_encoder` / `prefix_cache` / `denoise_step` 的 benchmark 可直接参考
- `action_chunk_pipeline` 当前这个 `3.83x` 不应被当成严格整链路 speedup

更准确的理解应该是：

> **当前 benchmark 已证明 TRT 的核心热点子阶段有收益，但整链路 benchmark 口径仍需修正。**

请把这句话也单独记住：

> **当前 `action_chunk_pipeline` benchmark 不是严格整链路 apples-to-apples。**

---

## 14. `my_devs/pi_trt` 里每个文件到底有什么用

这一节专门回答：

> 这些代码到底是干嘛的？它们能达到什么效果？

### 14.1 `common.py`

这是整个 `pi05 TRT` 工程的公共底座。

它负责：

- 定位 repo root
- 加载 checkpoint policy
- 加载 checkpoint preprocessor
- 准备 synthetic observation
- 准备 batch
- 构造 prefix
- 计算 prefix cache
- 计算 denoise step
- 计算完整 action chunk
- 提供 ONNX wrapper
- 提供指标计算和 JSON 保存

没有这个文件，后面的：

- 导出
- ONNX compare
- TRT compare
- mock runtime
- benchmark

都会各写各的，很快就会语义飘掉。

所以它的效果是：

> **把所有后续脚本绑定到同一套 Torch 参考语义上。**

### 14.2 `export_onnx.py`

作用：

- 从 Torch policy 导出 3 个 ONNX 子图

效果：

- 生成后续 ORT compare 和 TensorRT build 的输入

### 14.3 `compare_torch_onnx.py`

作用：

- 比较 Torch FP16 和 ONNX FP16

效果：

- 验证 ONNX 导出链路没有明显数值漂移

### 14.4 `build_engine.py`

作用：

- 读取 ONNX，build 成 TRT engine

效果：

- 生成：
  - `vision_encoder_fp16.engine`
  - `prefix_cache_fp16.engine`
  - `denoise_step_fp16.engine`
  - `build_report.json`

### 14.5 `compare_torch_trt.py`

作用：

- 比较 Torch FP16 和 TRT FP16

效果：

- 验证核心 TRT 链路是否数值对齐

### 14.6 `run_pi05_infer_trt_mock.py`

作用：

- 用同一批 frames 跑 PyTorch 和 TRT
- 不接机器人，不发动作

效果：

- 验证完整 `predict_action` 语义下，TRT adapter 是否工作正常

### 14.7 `benchmark_trt.py`

作用：

- 做离线 latency benchmark

效果：

- 给出各阶段的 Torch vs TRT 时间对比
- 当前更适合作为“热点子阶段是否有收益”的参考

---

## 15. 我实际做过的工作，按时间顺序说一遍

这一节专门回答：

> 你到底做了什么？按步骤说清楚。

### 15.1 第一步：理解 `pi05` Torch 推理语义

我先确认：

- `pi05` 的真实推理拓扑
- checkpoint 的真实 processor 配置
- state 是文本化进入 prompt 的
- prefix cache 是天然存在的
- denoise 是天然可以单步封装的

结论是：

- 不能照抄 GR00T 7-engine
- 应该拆成 3 段式

### 15.2 第二步：做 `common.py`

我把导出和 compare 所需要的公共语义整理到一个文件里，避免不同脚本各自理解 `pi05`。

### 15.3 第三步：导出 ONNX

我写了：

- `export_onnx.py`

导出了：

- `vision_encoder_fp16.onnx`
- `prefix_cache_fp16.onnx`
- `denoise_step_fp16.onnx`

### 15.4 第四步：修 ONNX 导出时遇到的问题

我逐个修掉：

- `DynamicCache` 问题
- `CumSum(bool)` 问题
- `Cos(float64)` 问题

### 15.5 第五步：做 Torch vs ONNX compare

我写了：

- `compare_torch_onnx.py`

并完成了两轮复核。

### 15.6 第六步：做 TRT build

我写了：

- `build_engine.py`

并成功 build 出 3 个 engine。

### 15.7 第七步：做 Torch vs TRT compare

我写了：

- `compare_torch_trt.py`

并完成了两轮 Torch vs TRT 一致性验证。

### 15.8 第八步：做不发动作的 mock runtime

我写了：

- `run_pi05_infer_trt_mock.py`

让它：

- 只跑离线 frame
- 不控制机器人
- 验证 runtime glue

### 15.9 第九步：做离线 benchmark

我写了：

- `benchmark_trt.py`

对各阶段做 Torch vs TRT benchmark。

### 15.10 第十步：请独立 reviewer 审核

我还请多个独立 reviewer 对这条链路做了代码审核。

它们的核心共识是：

- 核心一致性链路已经通过
- 但 benchmark 口径、mock compare 口径、RTC 契约保护这几项还应再修正

所以当前最准确的状态是：

> **核心一致性通过，离线验证链路可用，但部分验证脚本的解释口径仍需修正。**

---

## 16. 从零复跑：你应该怎么做

这一节是最重要的操作部分。

下面给出一套**从零复跑**的步骤。

### 16.1 前置条件

必须确认下面这些条件满足：

1. 使用 `lerobot_flex` 环境
2. 有可用 CUDA GPU
3. 本地存在 tokenizer 目录：
   - `google/paligemma-3b-pt-224`
4. 有可用的 checkpoint：
   - 当前默认使用：
     - `outputs/train/pi05_grasp_block_in_bin1_repro_20260303_170727/bs8_20260303_170727/checkpoints/last/pretrained_model`
5. 当前机器可 import TensorRT
   - 若环境里没有，就依赖：
     - `my_devs/groot_trt/trt_utils.py`
   - 当前默认 data-local TensorRT 路径是：
     - `/data/cqy_workspace/third_party/tensorrt_10_13_0_35`

### 16.1.1 关于 shell 的一个现实提醒

如果你的 shell 里：

- `conda activate lerobot_flex` 不稳定
- 或者当前 shell 没有正确初始化 conda

那就不要硬用 `conda activate`，直接用报告里的：

```bash
conda run -n lerobot_flex ...
```

这是当前这份报告默认采用的方式，也是我实际验证过的更稳妥方式。

### 16.1.2 大致资源要求

这条链路不是轻量任务，复跑前建议先有心理预期。

大致资源压力如下：

- ONNX 导出会生成很大的主文件和 external data sidecar
- TRT engine 总量也很大
- 当前正式产物目录里：
  - `vision_encoder_fp16.engine` 约 `794M`
  - `prefix_cache_fp16.engine` 约 `3.5G`
  - `denoise_step_fp16.engine` 约 `825M`

因此建议至少提前确认：

- GPU 可用
- `/data` 下有充足磁盘空间
- 允许较长的导出与 build 等待时间

如果只是 smoke 复跑：

- `compare` 往往比 `export` 和 `build` 更快
- `mock compare` 会比单纯 compare 更慢，因为它会走完整 `predict_action` 语义

### 16.2 推荐先定义环境变量

```bash
export POLICY_PATH=/data/cqy_workspace/flexible_lerobot/outputs/train/pi05_grasp_block_in_bin1_repro_20260303_170727/bs8_20260303_170727/checkpoints/last/pretrained_model
export RUN_ROOT=/data/cqy_workspace/flexible_lerobot/outputs/pi_trt/pi05_report_repro_$(date +%Y%m%d_%H%M%S)
export ONNX_DIR=$RUN_ROOT/pi05_onnx
export ENGINE_DIR=$RUN_ROOT/pi05_engine_api_trt1013
mkdir -p "$RUN_ROOT"
```

如果你不想设置环境变量，也可以直接把路径写进命令里。

### 16.3 第一步：导出 ONNX

```bash
conda run -n lerobot_flex \
  python my_devs/pi_trt/export_onnx.py \
  --policy-path "$POLICY_PATH" \
  --onnx-out-dir "$ONNX_DIR"
```

跑完后你应该看到：

- `vision_encoder_fp16.onnx`
- `prefix_cache_fp16.onnx`
- `denoise_step_fp16.onnx`
- 外加大量 `onnx__MatMul_*` / `onnx__Mul_*` sidecar 文件

补充说明：

- 这一阶段可能会比较慢
- 可能看到一些 `TracerWarning`
- 这些 warning 在当前导出链路里通常是正常现象，不代表导出失败

### 16.4 第二步：做 Torch vs ONNX compare

第一轮：

```bash
conda run -n lerobot_flex \
  python my_devs/pi_trt/compare_torch_onnx.py \
  --policy-path "$POLICY_PATH" \
  --onnx-dir "$ONNX_DIR" \
  --json-out "$ONNX_DIR/compare_metrics_torch_onnx.json"
```

第二轮：

```bash
conda run -n lerobot_flex \
  python my_devs/pi_trt/compare_torch_onnx.py \
  --policy-path "$POLICY_PATH" \
  --onnx-dir "$ONNX_DIR" \
  --seed 17 \
  --json-out "$ONNX_DIR/compare_metrics_torch_onnx_seed17.json"
```

### 16.5 第三步：build TensorRT engine

确保：

- `ONNX_DIR` 里 3 个 ONNX 主文件都在
- 同目录 sidecar 文件也都在
- 不要把“只导出成功了其中一个或两个 ONNX”当成可继续的状态

当前 build 前建议你显式检查：

```bash
ls -lah "$ONNX_DIR"
```

并确认至少存在：

- `vision_encoder_fp16.onnx`
- `prefix_cache_fp16.onnx`
- `denoise_step_fp16.onnx`

然后执行：

```bash
conda run -n lerobot_flex \
  python my_devs/pi_trt/build_engine.py \
  --onnx-dir "$ONNX_DIR" \
  --engine-out-dir "$ENGINE_DIR"
```

跑完后应该看到：

- `vision_encoder_fp16.engine`
- `prefix_cache_fp16.engine`
- `denoise_step_fp16.engine`
- `build_report.json`

补充说明：

- 这一阶段可能会看到：
  - `Make sure input prefix_position_ids has Int64 binding`
- 这是当前成功 build 中也出现过的 warning，不等于 build 失败

### 16.6 第四步：做 Torch vs TRT compare

第一轮：

```bash
conda run -n lerobot_flex \
  python my_devs/pi_trt/compare_torch_trt.py \
  --policy-path "$POLICY_PATH" \
  --engine-dir "$ENGINE_DIR" \
  --json-out "$ENGINE_DIR/compare_metrics_torch_trt.json"
```

第二轮：

```bash
conda run -n lerobot_flex \
  python my_devs/pi_trt/compare_torch_trt.py \
  --policy-path "$POLICY_PATH" \
  --engine-dir "$ENGINE_DIR" \
  --seed 17 \
  --json-out "$ENGINE_DIR/compare_metrics_torch_trt_seed17.json"
```

### 16.7 第五步：做 mock compare（不发动作）

随机 frame：

```bash
conda run -n lerobot_flex \
  python my_devs/pi_trt/run_pi05_infer_trt_mock.py \
  --policy-path "$POLICY_PATH" \
  --engine-dir "$ENGINE_DIR" \
  --source random \
  --num-steps 55 \
  --out-dir "$RUN_ROOT/mock_compare_random"
```

然后 replay 同一批 frame：

```bash
conda run -n lerobot_flex \
  python my_devs/pi_trt/run_pi05_infer_trt_mock.py \
  --policy-path "$POLICY_PATH" \
  --engine-dir "$ENGINE_DIR" \
  --source frames_dir \
  --frames-dir "$RUN_ROOT/mock_compare_random/frames" \
  --out-dir "$RUN_ROOT/mock_compare_replay"
```

补充说明：

- 这一步可能明显慢于单纯 compare
- 因为它会同时加载 PyTorch / TRT 路径，并走完整 `predict_action` 语义
- **请再次记住：mock compare 是 runtime smoke check，不是最终纯净精度 gate**

### 16.8 第六步：做 benchmark

如果你只是 smoke 跑法：

```bash
conda run -n lerobot_flex \
  python my_devs/pi_trt/benchmark_trt.py \
  --policy-path "$POLICY_PATH" \
  --engine-dir "$ENGINE_DIR" \
  --warmup-iters 1 \
  --iters 2 \
  --json-out "$ENGINE_DIR/benchmark_metrics_torch_trt_smoke.json"
```

如果你想拿相对稳定一点的 benchmark：

```bash
conda run -n lerobot_flex \
  python my_devs/pi_trt/benchmark_trt.py \
  --policy-path "$POLICY_PATH" \
  --engine-dir "$ENGINE_DIR" \
  --warmup-iters 5 \
  --iters 20 \
  --json-out "$ENGINE_DIR/benchmark_metrics_torch_trt.json"
```

补充说明：

- `vision_encoder` / `prefix_cache` / `denoise_step` 的 benchmark 更适合直接参考
- **当前 `action_chunk_pipeline` benchmark 不是严格整链路 apples-to-apples**
- 所以不要把它直接当成“最终部署端到端 speedup”

### 16.9 常见正常警告与解释

下面这些 warning 在当前链路里都曾出现过，但并不自动代表失败：

#### A. ONNX export 阶段的 `TracerWarning`

含义：

- 导出图时，PyTorch tracer 在提醒有些 Python 侧逻辑被静态化了

当前解释：

- 在这条固定 shape 导出链路里，这类 warning 常见
- 只要导出产物齐全，后续 compare 能跑，就不必把它直接视为失败

#### B. `Vision embedding key might need handling: ... patch_embedding ...`

含义：

- `PI05Policy.from_pretrained(...)` 在 remap state dict 时会对 vision embedding key 给出提醒

当前解释：

- 这是现有加载逻辑里的 warning
- 在当前 checkpoint 下并没有阻止 compare / mock / benchmark 成功运行

#### C. TRT build 阶段的 `prefix_position_ids has Int64 binding`

含义：

- TensorRT 在提醒这个输入需要 Int64 binding

当前解释：

- 当前成功 build 的 engine 也出现过这个 warning
- 如果 engine 成功产出，并且 compare 能跑通，它本身不代表失败

#### D. TRT run 阶段的 `Using default stream in enqueueV3() ...`

含义：

- TensorRT 在提示默认 CUDA stream 可能带来额外同步开销

当前解释：

- 这更偏 benchmark / 性能口径提醒
- 不代表数值错误
- 但意味着 benchmark 结果应该保守解释

---

## 17. 如何判断“这次复跑成功了”

### 17.1 ONNX 成功的最低标准

你至少应该确认：

- 3 个 ONNX 都导出了
- ORT 能加载
- `action_chunk_pipeline` cosine 接近 `0.99999`

### 17.2 TRT 成功的最低标准

你至少应该确认：

- 3 个 engine 都 build 成功
- `compare_torch_trt.py` 两轮都能跑完
- `action_chunk_pipeline` cosine 仍然接近 `0.99998+`

### 17.3 mock 成功的最低标准

你至少应该确认：

- `random` 和 `frames_dir` 两种 source 都能跑通
- 都会生成：
  - `actions_torch.npy`
  - `actions_trt.npy`
  - `compare_metrics_trt_mock.json`

---

## 18. 当前已经跑出的正式产物

### 18.1 ONNX 产物

- 目录：
  - `outputs/pi_trt/pi05_onnx_consistency_20260306_190504/pi05_onnx`

### 18.2 TRT 产物

- 目录：
  - `outputs/pi_trt/pi05_onnx_consistency_20260306_190504/pi05_engine_api_trt1013`

### 18.3 TRT 一致性结果

- 第一轮：
  - `compare_metrics_torch_trt.json`
- 第二轮：
  - `compare_metrics_torch_trt_seed17.json`

### 18.4 mock compare 结果

- 随机：
  - `outputs/pi_trt/mock_compare_20260310_smoke`
- replay：
  - `outputs/pi_trt/mock_compare_20260310_replay`

### 18.5 benchmark 结果

- smoke：
  - `benchmark_metrics_torch_trt_smoke.json`
- 稳定版：
  - `benchmark_metrics_torch_trt.json`

---

## 19. 当前我对结果的最终判断

### 19.1 我确认已经通过的部分

我确认下面这些已经通过：

1. Torch -> ONNX 的核心链路对齐
2. ONNX -> TRT 的 engine build
3. Torch -> TRT 的核心数值链路对齐
4. 不接机器人、不发动作的 mock runtime 路径可跑

### 19.2 我明确不把它说成“完全收口”的部分

下面这些我不会说已经完全收口：

1. `action_chunk_pipeline` 的 benchmark 口径  
   当前不是严格 apples-to-apples。

2. mock compare 的 action-level 精度证明  
   当前混入 dtype 差异，更适合 smoke check。

3. RTC 语义  
   当前工具链默认面向**无 RTC checkpoint**。

### 19.3 所以最准确的结论

最准确的表述应该是：

> **当前 `pi05` 的 Torch -> ONNX -> TRT 核心离线一致性已经通过，且 3-engine TRT 路径已经可用；但 benchmark 和 mock compare 的解释口径仍有 caveat，需要在最终离线验收前进一步修正。**

---

## 20. 后续最合理的动作

如果只做离线工作，最合理的下一步是：

1. 修 benchmark 的 workload 对齐问题
2. 修 mock compare 的 dtype 对齐问题
3. 给 RTC-enabled checkpoint 增加显式拒绝保护

如果这些修完，再进入：

4. 由人工亲自控制机器人做真机测试

也就是：

- 先把离线验证口径彻底修干净
- 再交给人去接触真实机器人

这也是当前最稳妥的方式。

---

## 21. 附录：一条完整的理解链

如果你完全不懂这套东西，可以记住下面这条链：

```text
原始 Torch 模型
    -> 分析推理拓扑
    -> 切成 3 段
    -> 导出 ONNX
    -> 用 ONNX 验证 Torch 对齐
    -> build 成 TensorRT engine
    -> 用 TRT 验证 Torch 对齐
    -> 做不发动作的 mock compare
    -> 做离线 benchmark
```

而 `my_devs/pi_trt/` 这几个文件，就是这条链上每一环的实现。

---

## 22. 附录：本报告要点清单

如果你只想快速检查这份报告是否覆盖完整，请看这个清单：

- 已解释 Torch / ONNX / TRT 的关系
- 已解释 `pi05` 为什么拆成 3 段
- 已解释每个子图对应的 Torch 语义
- 已解释 `common.py` 在整条链路中的作用
- 已解释每个脚本的职责
- 已给出从零复跑命令
- 已写出真实产物目录
- 已写出真实对齐结果
- 已写出真实 benchmark 结果
- 已写出当前 caveat

如果 reviewer 只按这份报告也能复走整条链路，那说明这份报告已经达到目标。
