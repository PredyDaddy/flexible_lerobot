# PI0.5 TensorRT 工程化方案 1（基于现有 GR00T TRT 方案迁移，当前只做 FP16 推理）

最后更新：2026-03-06  
仓库：`/data/cqy_workspace/flexible_lerobot`  
开发环境要求：`conda env = lerobot_flex`  
目标模型：`PI0.5 / pi05`  
当前范围：**仅讨论 FP16 推理工程化**，暂不展开 INT8 / FP8 / BF16 TensorRT / 服务化部署。

---

## 1. 结论先说

可以做，而且**不建议直接照抄 GR00T 的 7-engine 拆法**；应该沿用 GR00T 的方法论，但按 `pi05` 自己的推理拓扑重拆。

我给出的建议是：

1. **保留 checkpoint 保存的 pre/post processor 完全不动**，包括：
   - 状态归一化
   - state 离散化到 prompt
   - `google/paligemma-3b-pt-224` tokenizer
   - 后处理反归一化
2. **不要一开始就导出整个 pi05 policy**，而是按推理路径拆成更适合 TensorRT 的 3 个阶段。
3. **推荐的 FP16 方案 1**：
   - `vision_encoder_fp16`：图像编码
   - `prefix_cache_fp16`：prefix 语言模型前向，产出 KV cache
   - `denoise_step_fp16`：每一步 diffusion / flow-matching 去噪
4. 先做 **Torch BF16 原始模型 -> Torch FP16 导出副本** 的精度对齐，再做：
   - Torch FP16 vs ONNX
   - Torch FP16 vs TensorRT
5. 当前 `pi05` 的项目配置其实非常适合先做 **固定 shape、batch=1、双相机、token 长度固定 200、chunk=50** 的静态 TRT engine；这比 GR00T 还更适合先落一个稳定版本。

一句话概括：

> **GR00T 给我们的是“方法”，不是“模块边界”。PI0.5 应该做成 3 段式 FP16 TRT，而不是硬拆成 GR00T 那 7 个子模块。**

---

## 2. 这份方案基于哪些现有资产

### 2.1 已跑通的参考资产：GR00T TRT 全链路

你仓库里已经有一套非常成熟的参考：

- 文档：`my_devs/docs/gr00t_trt/`
- 实现：`my_devs/groot_trt/`

其中最值得直接迁移的方法有：

- `Torch -> ONNX -> TensorRT engine -> Torch vs TRT consistency` 的完整闭环
- 导出脚本、构建脚本、对齐脚本分离
- 运行时保留 Python glue，不强求把所有控制流塞进 TRT
- 先做模块级对齐，再做端到端 denoising pipeline 对齐
- 用 TensorRT Python API 构建 engine，而不是强依赖 `trtexec`
- 上机前先有 `mock` / dry-run 模式

### 2.2 PI0.5 当前现状

当前仓库里与 `pi05` 直接相关的关键文件：

- 模型：`src/lerobot/policies/pi05/modeling_pi05.py`
- 配置：`src/lerobot/policies/pi05/configuration_pi05.py`
- processor：`src/lerobot/policies/pi05/processor_pi05.py`
- 实机推理脚本：`my_devs/train/pi/so101/run_pi05_infer.py`
- 训练/复刻记录：`my_devs/docs/pi_train/pi05_modelscope_repro_work_report.md`

当前训练好的 checkpoint：

- `outputs/train/pi05_grasp_block_in_bin1_repro_20260303_170727/bs8_20260303_170727/checkpoints/last/pretrained_model`

---

## 3. 我对当前 PI0.5 推理链路的实际分析

这里不是泛泛而谈，而是根据源码和本地 checkpoint 的真实配置整理出来的。

### 3.1 当前 checkpoint 的实际约束

从当前 `config.json` 和 `policy_preprocessor.json` 看，当前部署条件是：

- policy type：`pi05`
- 设备配置：`cuda`（但本次终端会话没有可用 CUDA，所以下面我做的是静态代码与 CPU 侧结构核验）
- 训练 dtype：`bfloat16`
- 双相机输入：
  - `observation.images.top`
  - `observation.images.wrist`
- 原始图像尺寸：`3 x 480 x 640`
- 模型内部图像尺寸：`224 x 224`
- tokenizer 最大长度：`200`
- `chunk_size = 50`
- `n_action_steps = 50`
- `num_inference_steps = 10`
- `max_state_dim = 32`
- `max_action_dim = 32`
- 当前 action 真实维度：`6`
- 当前 state 真实维度：`6`

**非常重要的一点**：
当前 checkpoint 保存下来的 `normalization_mapping` 实际是：

- `ACTION = MEAN_STD`
- `STATE = MEAN_STD`
- `VISUAL = IDENTITY`

这和 `PI05Config` 代码默认值里的 `QUANTILES` 不一样。

这意味着：

> **所有导出 / 对齐 / TRT runtime 都必须以 checkpoint 保存的 processor 为准，不能在代码里重新“猜”默认 processor。**

### 3.2 PI0.5 的 state 不是单独走一个 state encoder

这一点和 GR00T 很不一样。

`pi05` 的 processor 会先：

1. 取 `observation.state`
2. pad 到 32 维
3. 假定 state 已经被 normalizer 归一化到 `[-1, 1]`
4. 离散化成 256 个 bin
5. 拼成 prompt：

```text
Task: <task text>, State: <discretized_state...>;
Action:
```

然后再交给 `PaliGemma tokenizer`。

所以对 `pi05` 来说：

- **state 已经被“文本化”并揉进 prompt 了**
- runtime 主模型里没有像 GR00T 那样明显独立的 `state_encoder` 子模块

这决定了：

- 不能照搬 GR00T 的 `state_encoder.onnx`
- tokenizer / prompt builder 不能轻易挪进 TRT
- preprocessor 必须原样保留

### 3.3 PI0.5 推理的三个真正计算阶段

从 `PI05Policy.select_action -> predict_action_chunk -> PI05Pytorch.sample_actions` 来看，推理可以拆成三段：

#### 阶段 A：prefix embedding

包含：

- 每个相机图像先 resize + pad 到 `224 x 224`
- 归一化到 `[-1, 1]`
- `paligemma.model.get_image_features(image)` 得到图像 token embedding
- 语言 token 经 `embed_tokens`
- 图像 embedding 与语言 embedding 拼接成 prefix

#### 阶段 B：prefix cache 计算

代码路径：

- `paligemma_with_expert.forward(inputs_embeds=[prefix_embs, None], use_cache=True)`

这一步的作用不是直接出 action，而是：

- 用 prefix 跑一次 `paligemma.language_model`
- 产出后续 denoising 会反复复用的 `past_key_values`

#### 阶段 C：10 步 denoising

每次 chunk 生成时，循环 `num_inference_steps=10` 次：

1. 从当前 `x_t` 和 `timestep` 生成 suffix embedding
2. 构造 attention mask / `position_ids`
3. 带着 prefix cache 调 `gemma_expert.model.forward(...)`
4. `action_out_proj`
5. Euler 更新：`x_t = x_t + dt * v_t`

这一步是当前 `pi05` 推理最核心的“扩散/流匹配”主循环。

---

## 4. 我本轮实际核验到的真实张量形状

我在本地把当前 checkpoint 加载起来，结合 dummy 输入核验过一次真实 shape。由于本终端会话没有可用 CUDA，模型实际退回 CPU，但 shape 与拓扑判断仍有效。

### 4.1 关键模型维度

- `paligemma_variant = gemma_2b`
- `action_expert_variant = gemma_300m`
- vision hidden：`1152`
- text hidden：`2048`
- expert hidden：`1024`
- `num_image_tokens = 256`
- `tokenizer_max_length = 200`
- `action_in_proj`: `(1024, 32)`
- `action_out_proj`: `(32, 1024)`
- `time_mlp_in`: `(1024, 1024)`
- `time_mlp_out`: `(1024, 1024)`

### 4.2 prefix 实际 shape

当前双相机配置下：

- 每个相机图像 token 数：`256`
- 两个相机：`2 x 256 = 512`
- 语言 token 长度固定 pad 到：`200`

所以 prefix 总长度固定为：

- `512 + 200 = 712`

实际核验结果：

- `prefix_embs`: `(1, 712, 2048)`，dtype = `torch.bfloat16`
- `prefix_pad_masks`: `(1, 712)`，dtype = `torch.bool`
- `prefix_att_masks`: `(1, 712)`，dtype = `torch.bool`

### 4.3 prefix cache 实际 shape

当前 prefix pass 产出的 cache：

- 层数：`18`
- 每层是 `(key, value)` 二元组
- 每个 cache tensor 形状：`(1, 1, 712, 256)`
- dtype：`torch.bfloat16`

这说明：

- cache 不是 ragged 的，**每一层 shape 完全一致**
- 很适合在导出 wrapper 里把 18 层 KV 堆叠成一个大 tensor

这是我认为 `pi05` 很适合工程化的关键点之一。

### 4.4 denoise step 实际 shape

单步 denoise 时我核验到：

- `suffix_embs`: `(1, 50, 1024)`，dtype = `torch.float32`
- `full_att_2d_masks_4d`: `(1, 1, 50, 762)`，dtype = `torch.float32`
- `position_ids`: `(1, 50)`，范围大致是 `712 ~ 761`
- `suffix_out`: `(1, 50, 1024)`，dtype = `torch.bfloat16`
- `v_t = action_out_proj(suffix_out)`: `(1, 50, 32)`，dtype = `torch.float32`

这进一步说明：

- 当前 `pi05` 的 TRT 拆分天然应该围绕 **prefix cache** 和 **denoise step** 做
- `denoise_step` 是一个非常清晰的、可以单独封装成 wrapper 的推理单元

---

## 5. 为什么不能直接照抄 GR00T 的 7-engine 方案

GR00T 那套 7-engine 拆法之所以成立，是因为它本身在实现上就有比较清晰的边界：

- backbone（ViT + LLM）
- action head 的多个子模块
  - `vlln_vl_self_attention`
  - `state_encoder`
  - `action_encoder`
  - `DiT`
  - `action_decoder`

而 `pi05` 不一样：

1. **state 没有独立 state encoder**，而是先被 processor 文本化。
2. **prefix 与 suffix 通过 KV cache 连接**，不是简单的“上一个模块输出一个 embedding 给下一个模块”。
3. `paligemma_with_expert.forward(...)` 的 inference 分支本身就是：
   - prefix-only 分支
   - suffix-only 分支
   而不是 GR00T 那种容易独立导出的多个 action 子模块。
4. `denoise_step` 里 attention mask、position_ids、adarms_cond 都是 runtime 构造的，粘合逻辑比 GR00T 更紧。

因此，PI0.5 如果想工程化，正确姿势是：

> **继承 GR00T 的工作流、脚手架、验证方法，但重新设计更适合 `pi05` 的导出边界。**

---

## 6. 推荐的 PI0.5 FP16 TRT 拆分方案

这是我认为当前最稳、最像“工程方案”的拆法。

### 6.1 方案总览

建议最终落成 3 个主要 engine：

1. `vision_encoder_fp16.engine`
2. `prefix_cache_fp16.engine`
3. `denoise_step_fp16.engine`

Python/Torch 侧保留：

- checkpoint preprocessor / postprocessor
- 本地 tokenizer
- prompt 拼接
- token embedding
- image embedding 与 language embedding 的拼接
- queue 逻辑（50 步 action chunk）
- denoising 外层 `for step in range(num_steps)` 控制流

### 6.2 Engine 1：`vision_encoder_fp16`

职责：

- 输入单相机或双相机图像（推荐把 camera 维合并到 batch）
- 输出图像 token embedding

建议 wrapper 输入输出：

- 输入：`images`，形状可设计为：
  - 方案 A：`[B, N_cam, 3, 224, 224]`
  - 方案 B：`[B * N_cam, 3, 224, 224]`
- 输出：`image_embs`
  - 若按 B 方案，可输出 `[B * N_cam, 256, 2048]`
  - 再在 Python 里 reshape 回 `[B, N_cam * 256, 2048]`

为什么单独拆出来：

- 视觉 encoder 是大头计算
- 与 tokenizer / cache 无耦合
- 结构清晰，最容易先导出成功

### 6.3 Engine 2：`prefix_cache_fp16`

职责：

- 输入 prefix `inputs_embeds`
- 跑 `paligemma.language_model.forward(... use_cache=True)`
- 输出 prefix KV cache

建议 wrapper 输入：

- `inputs_embeds`: `[B, 712, 2048]`
- `attention_mask_4d`: `[B, 1, 712, 712]`
- `position_ids`: `[B, 712]`

建议 wrapper 输出：

- `kv_cache`: `[18, 2, B, 1, 712, 256]`

这里我**强烈建议不要把 cache 作为 36 个分散输出**，而是堆叠成一个 tensor。

原因：

1. TensorRT binding 更简单
2. ONNX wrapper 更稳定
3. runtime 代码更干净
4. 后面的 `denoise_step_fp16` 只需要一个 cache 输入

### 6.4 Engine 3：`denoise_step_fp16`

职责：

- 输入当前 `x_t`、当前 `timestep`、prefix cache、prefix pad mask
- 内部完成：
  - `embed_suffix`
  - 构造 attention mask / `position_ids`
  - `gemma_expert.model.forward(...)`
  - `action_out_proj`
- 输出一步预测速度 `v_t`

建议 wrapper 输入：

- `x_t`: `[B, 50, 32]`
- `timestep`: `[B]`，float32
- `prefix_pad_masks`: `[B, 712]`
- `kv_cache`: `[18, 2, B, 1, 712, 256]`

建议 wrapper 输出：

- `v_t`: `[B, 50, 32]`

为什么建议把 `action_in_proj + time_mlp + gemma_expert + action_out_proj` 包成一个 engine：

- 这几个模块一起才构成真正的 denoise step
- `action_in_proj` / `time_mlp` 本身很小，单拆反而增加 engine launch 开销
- 和 GR00T 不同，PI0.5 的 action path更像一个单体 step

---

## 7. 为什么我认为当前应该先做“固定 shape FP16”，而不是动态 shape

这是这份方案的一个核心判断。

### 7.1 当前项目实际上天然接近静态 shape

当前 real deployment 条件基本已经把 shape 固定住了：

- batch：上机通常 `B=1`
- camera 数：固定 `2`
- 每相机 image token：固定 `256`
- tokenizer 长度：固定 pad 到 `200`
- prefix 总长度：固定 `712`
- chunk size：固定 `50`
- denoise step 数：固定 `10`
- state / action pad 维度：固定 `32`

这意味着：

- prefix cache engine 可以做静态 shape
- denoise step engine 也可以做静态 shape
- TensorRT profile 复杂度会远低于 GR00T

### 7.2 这样做的直接好处

1. ONNX 导出更容易成功
2. TensorRT build 更稳定
3. 数值对齐更容易
4. runtime 更简单
5. 一开始就能上机验证，不会一上来掉进 dynamic profile 深坑

所以我建议：

> **PI0.5 的第一版 FP16 TRT，不要追求通用性，先把你当前这个双相机、batch=1、chunk=50 的任务做成静态 engine。**

---

## 8. PI0.5 的 ONNX 对齐，正确顺序应该怎么做

这一节非常关键。

GR00T 当时可以直接做 Torch vs ONNX；但 `pi05` 当前训练 checkpoint 是 **BF16**，而你想要的是 **FP16 TRT 推理**。

所以 `pi05` 必须多做一层精度分解，否则你最后不知道误差是来自：

- BF16 -> FP16 转换
- ONNX 导出
- TensorRT 构建
- 还是 runtime glue 自己写错了

### 8.1 我建议的对齐顺序

#### 第 0 层：Torch BF16 原始模型 vs Torch FP16 导出副本

先做一个 export-only model clone：

- 原模型：checkpoint 原始 `bfloat16` 路线
- 导出副本：把导出相关模块 cast 到 `float16`

先比较：

- prefix image embedding
- prefix cache
- 单步 `v_t`
- 完整 10 步 action chunk

如果这一步误差就已经很大，那么问题不在 ONNX / TRT，而在 **BF16 -> FP16 本身**。

#### 第 1 层：Torch FP16 导出副本 vs ONNX FP16

只比较 ONNX 导出是否正确。

#### 第 2 层：Torch FP16 导出副本 vs TRT FP16

只比较 TensorRT build / runtime 是否正确。

#### 第 3 层：Torch BF16 原始模型 vs TRT FP16 端到端

最后再看“最终可交付行为”是不是能接受。

### 8.2 为什么这个顺序是必须的

因为当前 `pi05` 的原始 checkpoint 不是 FP16，而是 BF16。

如果你跳过第 0 层，最后就会出现这种情况：

- TRT 和 Torch 不一致
- 但你根本不知道是 export 错了，还是 FP16 本身就跟 BF16 有漂移

所以：

> **PI0.5 做 FP16 TRT 的第一性问题，不是 TensorRT，而是先把 BF16 -> FP16 的漂移源单独隔离出来。**

---

## 9. ONNX 对齐时应该比什么，不能比什么

### 9.1 应该固定在“checkpoint processor 之后”的接口对齐

建议比较的输入层级：

1. **不要**直接从原始机器人 observation 开始比
2. **应该**从 checkpoint preprocessor 输出后的 batch 开始比

原因：

- tokenizer 是 runtime 依赖
- state prompt 构造是 processor 逻辑
- normalization 也是 processor 逻辑
- 这些都应该作为“稳定前处理资产”冻结，而不是每次导出时重建

正确做法：

- 用 checkpoint 保存的 `policy_preprocessor.json`
- 取一批真实/合成 observation
- 保存 preprocessor 输出张量
- 以这些张量作为 Torch / ONNX / TRT 共同输入

### 9.2 推荐的比较层级

建议至少比 4 层：

#### 层 1：Vision encoder

比较：

- Torch FP16 `embed_image`
- ONNX `vision_encoder_fp16`
- TRT `vision_encoder_fp16`

#### 层 2：Prefix cache

比较：

- KV cache
- 或至少比较若干层 key/value 的 cosine / rmse

如果 wrapper 做成堆叠 cache tensor，则直接比较：

- `kv_cache`

#### 层 3：单步 denoise

比较：

- 单步输出 `v_t`

这是最重要的模块级指标。

#### 层 4：完整 10 步 action chunk

比较：

- 最终输出 `actions_chunk`

这对应 GR00T 里 `action_denoising_pipeline` 的作用，是最接近真实推理闭环的指标。

### 9.3 推荐指标

沿用你在 `groot_trt` 已经跑通的指标最合适：

- `cosine`
- `rmse`
- `mean_abs`
- `max_abs`

### 9.4 推荐验收阈值（建议值，不是当前已实测值）

由于目前还没跑 PI0.5 的真正 ONNX/TRT，我给的是建议阈值：

#### BF16 Torch vs FP16 Torch clone

- vision / denoise step：`cosine >= 0.999`
- 完整 action chunk：`cosine >= 0.9999`

#### Torch FP16 vs ONNX FP16

- vision / prefix cache / denoise step：`cosine >= 0.999`
- 完整 action chunk：`cosine >= 0.9999`

#### Torch FP16 vs TRT FP16

- vision / prefix cache / denoise step：`cosine >= 0.995`
- 完整 action chunk：`cosine >= 0.999`

如果你要更稳：

- 除了看 cosine，也要看 unnormalize 后 6 维真实 action 的 `max_abs`
- 尤其注意 gripper / wrist 之类高敏感维度

---

## 10. FP16 导出时，权重精度应该怎么处理

### 10.1 当前模型原生不是 FP16，而是 BF16

当前 checkpoint 训练配置里明确写的是：

- `policy.dtype = bfloat16`

所以如果现在要做 FP16 TensorRT，不能假设“模型本来就是 FP16”。

### 10.2 推荐做法：单独构造 export-only model

我建议不要直接改业务模型，而是专门做一个 export-only 路线：

- 从 checkpoint 加载原模型
- clone / wrapper 一份 export model
- 只在 export model 上做 dtype 转换

### 10.3 推荐的初始策略

初始策略建议是：

- heavy modules 尽量转 FP16：
  - vision tower
  - paligemma language model
  - gemma expert
  - action_in_proj
  - action_out_proj
  - time_mlp
- mask / `position_ids` / `timestep` 等按需要保留：
  - `position_ids`：`int64`
  - `attention_mask`：通常保留 `float32`
  - `timestep`：`float32`

### 10.4 如果 BF16 -> FP16 漂移过大怎么办

如果第 0 层对齐不过，可以按顺序回退：

1. 先只把 denoise path 做 FP16
2. prefix cache 仍在 Torch 路线
3. 对敏感层保留 FP32
4. 或者让 first version 变成：
   - TRT：vision + denoise
   - Torch：prefix cache

也就是说：

> **你要把“纯 FP16”当目标，但第一版工程不应该为了纯粹性牺牲可落地性。**

---

## 11. TensorRT runtime 应该怎么接进来

这里依然建议沿用 GR00T 的 runtime 设计思路。

### 11.1 运行时不改 processor

保持下面这些完全沿用现在的 `run_pi05_infer.py`：

- 本地 tokenizer 目录检查
- checkpoint pre/post processors
- robot observation -> dataset frame -> policy 输入的构造
- action queue 逻辑
- 最终 postprocess + robot action send

### 11.2 替换点放在 `policy.model.sample_actions`

最稳的接法是：

- 保留 `PI05Policy.select_action(...)`
- 保留 `predict_action_chunk(...)`
- 在 model 侧做一个 `TrtPi05PolicyAdapter`
- 只替换 chunk 生成逻辑

这和当前 `my_devs/groot_trt/run_groot_infer_trt.py` 的思路是一致的。

### 11.3 建议的运行时流程

1. Torch policy 仅用于：
   - config
   - checkpoint processor
   - 小量 glue 逻辑
2. `vision_encoder_fp16.engine` 出图像 embedding
3. Torch token embedding + 拼接 prefix
4. `prefix_cache_fp16.engine` 出 stacked KV cache
5. Python 循环 10 次：
   - 调 `denoise_step_fp16.engine`
   - `x_t = x_t + dt * v_t`
6. 裁出真实 action 6 维
7. 走现有 postprocessor

也就是说：

- **控制流还在 Python**
- **重计算放进 TRT**

这跟 GR00T 当时保留 diffusion loop 在 Python 是同一种工程哲学。

---

## 12. 具体建议你在仓库里怎么组织代码

### 12.1 `my_devs/pi_trt/`：工具链层

建议新增一个和 `my_devs/groot_trt/` 平行的目录：

```text
my_devs/pi_trt/
  README.md
  trt_utils.py
  export_vision_onnx.py
  export_prefix_cache_onnx.py
  export_denoise_step_onnx.py
  compare_torch_fp16.py
  compare_torch_onnx.py
  compare_torch_trt.py
  build_engine.py
  build_engine.sh
  run_pi05_infer_trt.py
```

职责：

- 导出 ONNX
- build engine
- 做 consistency comparison
- 上机 TRT runtime

### 12.2 `src/lerobot/policies/pi05/trt_runtime/`：轻量 runtime 层（可选第二步再做）

如果后续你希望不是只有 `my_devs` 脚本可用，而是整个仓库更正式支持 TRT，可再补：

```text
src/lerobot/policies/pi05/trt_runtime/
  __init__.py
  engine.py
  adapter.py
  config.py
```

第一阶段可以先不动 `src/`，只在 `my_devs/` 里跑通。

这是我更推荐的节奏，因为：

- 先证明方案可行
- 再决定是否侵入主仓代码

---

## 13. 我建议的实施顺序（非常具体）

### 第 1 步：先写 shape / dtype dump 脚本

目标：

- 固化当前 checkpoint 的实际 shape
- 固化 prefix length / cache shape / suffix shape

产物建议：

- `outputs/pi_trt/<run_id>/shape_report.json`

### 第 2 步：先做 Torch BF16 vs Torch FP16 clone

目标：

- 先回答“FP16 是否本身可接受”

产物建议：

- `compare_metrics_torch_bf16_vs_torch_fp16.json`

### 第 3 步：先导出 `denoise_step.onnx`

原因：

- 边界清晰
- 是最关键的 step
- 最容易先形成闭环

比对：

- 单步 `v_t`
- 完整 10 步 chunk

### 第 4 步：构建 `denoise_step.engine`

目标：

- 跑通第一版 TRT 主循环

这一步成功后，即使 prefix 还在 Torch，你也已经有一个真正能跑的 PI0.5 TRT 版本雏形。

### 第 5 步：导出 `vision_encoder.onnx`

目标：

- 减掉图像 encoder 的大头计算

### 第 6 步：导出 `prefix_cache.onnx`

目标：

- 完成完整 chunk 生成的 TRT 化

### 第 7 步：写 `run_pi05_infer_trt.py`

要求：

- 支持 `--dry-run`
- 支持 `--mock`
- 支持只打印 action 不下发
- 支持加载 checkpoint 保存的 processors

### 第 8 步：上机验证顺序

建议顺序：

1. dry-run
2. mock，不发动作
3. 低 FPS，短时间
4. 真机小动作
5. 再逐步提速

---

## 14. 风险点与规避建议

### 14.1 最大风险：BF16 -> FP16 漂移

规避：

- 先做 Torch 内部 BF16/FP16 对齐
- 必要时局部保留 FP32 敏感层

### 14.2 第二大风险：cache I/O 太复杂

规避：

- 不要 36 个分散 cache 输入输出
- 改成 stacked `kv_cache` tensor

### 14.3 第三大风险：transformers 版本与替换实现

当前 `pi05` 明确依赖特定 transformers 实现，训练报告里已经说明用了特定分支。

规避：

- 导出脚本必须继续使用当前 `lerobot_flex` 环境
- 不要随意升级 transformers

### 14.4 第四大风险：tokenizer 离线依赖

规避：

- 继续复用当前本地路径：`google/paligemma-3b-pt-224`
- TRT runtime 脚本沿用 `run_pi05_infer.py` 里的本地 tokenizer 检查逻辑

### 14.5 第五大风险：第一版就做动态 shape

规避：

- 第一版只做固定 shape
- 把工程问题先压缩到最小

---

## 15. 与 GR00T 方案一一对应后的迁移关系

### GR00T 里可直接继承的东西

- 目录组织方式
- `trt_utils.py` 的 TensorRT import / session 封装
- `build_engine.py` / `build_engine.sh` 的工作流
- `compare_torch_onnx.py` / `compare_torch_trt.py` 的指标和 JSON 报告形式
- runtime adapter 只替换重计算模块，不碰 processor 的思路

### GR00T 里不能直接继承的东西

- 7-engine 拆分边界
- 基于 `state_encoder / action_encoder / DiT / action_decoder` 的 action head 拆法
- 按 GR00T 那种“backbone features -> action head modules”传递接口

### PI0.5 自己必须新做的东西

- prefix cache wrapper
- stacked cache 的 ONNX export
- denoise step wrapper
- BF16 -> FP16 对齐脚本

---

## 16. 这份方案下，我建议定义的产物结构

建议一次 TRT 产物固定为：

```text
outputs/pi_trt/<RUN_ID>/
  pi05_onnx/
    vision_encoder_fp16.onnx
    prefix_cache_fp16.onnx
    denoise_step_fp16.onnx
    compare_metrics_torch_fp16_vs_onnx.json
  pi05_engine_trt1013/
    vision_encoder_fp16.engine
    prefix_cache_fp16.engine
    denoise_step_fp16.engine
    build_report.json
    compare_metrics_torch_fp16_vs_trt.json
  logs/
    export_vision.log
    export_prefix_cache.log
    export_denoise_step.log
    compare_torch_fp16.log
    compare_onnx.log
    build_engine.log
    compare_trt.log
  artifacts/
    shape_report.json
    sample_preprocessor_batch.pt
```

这样后续：

- debug 有统一目录
- 每次 rerun 可以完整留痕
- 很像你现在 GR00T 那套已经跑顺的工作模式

---

## 17. 方案 1 的最终推荐落地结论

如果只考虑“尽快做出一个稳的、能落地的 PI0.5 FP16 TRT 版本”，我建议你这么定：

### 方案 1（推荐）

- 保留 checkpoint pre/post processor 不动
- 保留 tokenizer 与 prompt 构造不动
- 保留 Python 外层 10 步 denoise loop 不动
- 固定 shape：
  - `B = 1`
  - `N_cam = 2`
  - `prefix_len = 712`
  - `chunk = 50`
  - `num_steps = 10`
- engine 拆分：
  - `vision_encoder_fp16`
  - `prefix_cache_fp16`
  - `denoise_step_fp16`
- 对齐顺序：
  - Torch BF16 vs Torch FP16
  - Torch FP16 vs ONNX
  - Torch FP16 vs TRT
  - 最后再和原始 BF16 policy 做端到端闭环对比

### 不推荐的做法

- 一开始就整模型导出
- 一开始就做动态 shape
- 一开始就改 processor
- 一开始就把 tokenizer / prompt builder 硬塞进 TRT
- 直接照抄 GR00T 的 7-engine 方案

---

## 18. 当前我能确认的边界

我这次已经通过：

- `gr00t_trt` 文档
- `my_devs/groot_trt/` 已跑通代码
- `pi05` 源码
- `pi05` 当前 checkpoint 配置
- 本地 shape / cache 结构核验

确认了：

1. **这个方向是对的，能做。**
2. **PI0.5 的 TRT 关键不是 tokenizer，而是 prefix cache + denoise step。**
3. **第一版最值得做的是固定 shape FP16，而不是泛化版。**
4. **真正要先解决的第一问题不是 TRT，而是 BF16 -> FP16 的基线对齐。**

同时也要明确：

- 本次终端会话没有可用 CUDA，所以上述内容是基于代码分析 + CPU 结构核验得出的工程方案
- **还没有在当前会话里实际跑 ONNX 导出 / TensorRT build / TRT 推理 benchmark**
- 因此这是一份**工程实施方案报告**，不是“已经跑通 pi05 TRT”的结项报告

---

## 19. 下一步如果要继续做，我建议直接按这个顺序开工

1. 先新建 `my_devs/pi_trt/`
2. 先写 `compare_torch_fp16.py`
3. 先做 `denoise_step` wrapper/export/compare
4. 再做 `prefix_cache`
5. 最后接 `run_pi05_infer_trt.py`

这是最短路径，也是我认为风险最低的路径。

