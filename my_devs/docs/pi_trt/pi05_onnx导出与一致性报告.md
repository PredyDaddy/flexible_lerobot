# PI0.5 ONNX 导出与一致性报告（仅 ONNX，不上机）

最后更新：2026-03-06  
仓库：`/data/cqy_workspace/flexible_lerobot`  
环境：`conda env = lerobot_flex`  
范围：**只做 PI0.5 的 ONNX 导出与 Torch 一致性验证，不做 TRT，不做上机，不做动作下发**。

---

## 1. 本次完成了什么

本次已经把 `pi05` 按之前方案导出成 3 个 ONNX 子图，并完成了两轮 Torch vs ONNX 一致性验证。

导出的 3 个子图是：

1. `vision_encoder_fp16.onnx`
2. `prefix_cache_fp16.onnx`
3. `denoise_step_fp16.onnx`

对应的脚本已经落在：

- `my_devs/pi_trt/common.py`
- `my_devs/pi_trt/export_onnx.py`
- `my_devs/pi_trt/compare_torch_onnx.py`

本次导出与对齐产物目录：

- `outputs/pi_trt/pi05_onnx_consistency_20260306_190504/`

核心子目录：

- ONNX：`outputs/pi_trt/pi05_onnx_consistency_20260306_190504/pi05_onnx`
- 导出日志：`outputs/pi_trt/pi05_onnx_consistency_20260306_190504/logs/export_onnx.log`
- 对齐日志：`outputs/pi_trt/pi05_onnx_consistency_20260306_190504/logs/compare_torch_onnx.log`
- 第二轮复核日志：`outputs/pi_trt/pi05_onnx_consistency_20260306_190504/logs/compare_torch_onnx_seed17.log`

---

## 2. 导出边界

当前采用的是固定 shape 的第一版 ONNX 基线，边界如下：

- batch size：`1`
- camera 数：`2`（`top` + `wrist`）
- 图像预处理后尺寸：`224 x 224`
- language token 长度：`200`
- prefix 长度：`712`
- chunk size：`50`
- num_inference_steps：`10`
- action/state pad 维度：`32`

这次没有导整个 `pi05` policy，而是拆成 3 段：

### 2.1 `vision_encoder_fp16.onnx`

输入：

- `image_tensor`: `[2, 3, 224, 224]`

输出：

- `image_embeddings`: `[2, 256, 2048]`

### 2.2 `prefix_cache_fp16.onnx`

输入：

- `prefix_embs`: `[1, 712, 2048]`
- `prefix_attention_mask_4d`: `[1, 1, 712, 712]`
- `prefix_position_ids`: `[1, 712]`

输出：

- `kv_cache`: `[18, 2, 1, 1, 712, 256]`

说明：

- 这里把 18 层 KV cache 堆叠成了一个 tensor，而不是 36 个分散输出。

### 2.3 `denoise_step_fp16.onnx`

输入：

- `x_t`: `[1, 50, 32]`
- `timestep`: `[1]`
- `prefix_pad_masks`: `[1, 712]`
- `kv_cache`: `[18, 2, 1, 1, 712, 256]`

输出：

- `velocity`: `[1, 50, 32]`

---

## 3. 这次实际踩到的问题，以及怎么修掉的

这部分很重要，因为这些点决定了后面 TRT 能不能顺利接上。

### 3.1 `denoise_step` 的 cache 不能还原成普通 tuple

最开始我把堆叠后的 `kv_cache` 还原成普通 `(key, value)` tuple，再喂给 `gemma_expert.model.forward(...)`，结果失败。

报错本质是：

- 当前 transformers 版本需要的是 `DynamicCache`
- 普通 tuple 没有 `get_seq_length()` 等接口

最终修法：

- 使用 `DynamicCache.from_legacy_cache(...)` 从 stacked tensor 重建 cache

这一步修完后，`denoise_step` 才能正常导出。

### 3.2 ONNX Runtime 不接受 `CumSum(bool)`

最开始 `position_ids` 的路径里直接对 bool mask 做了 `cumsum`，Torch 没问题，但 ORT 认为这是非法图。

最终修法：

- 所有走 `cumsum` / `sum` 的 mask，先显式转成 `int64`

包括：

- `prefix_position_ids`
- `denoise_step` 里的 `position_ids`
- `make_att_2d_masks` 用到的 `att_masks`

### 3.3 ORT CPU 对 `Cos` 路径敏感

最开始 `timestep` 的正余弦位置编码走了 `float64` 路线，ORT CPU 在 `Cos` 节点上无法执行。

最终修法：

- 为导出路径单独实现了 `float32` 的时间正余弦 embedding
- 只在进入 `time_mlp` 前，再 cast 到 `fp16`

修完后，CPU ORT 可以完整加载并执行 `denoise_step_fp16.onnx`。

---

## 4. 一致性验证方式

本次一致性对比，不是从机器人实时 observation 开始比，而是：

1. 先加载 checkpoint 保存的 `policy_preprocessor.json`
2. 构造 synthetic observation
3. 走 checkpoint preprocessor
4. 在模型内部再做 `pi05` 自身的图像 resize / normalize 到 `224`
5. 用同一批输入分别跑 Torch FP16 reference 与 ONNX

比较层级是：

1. `vision_encoder`
2. `prefix_cache`
3. `denoise_step`
4. `action_chunk_pipeline`（完整 10 步闭环）

指标使用：

- `cosine`
- `rmse`
- `mean_abs`
- `max_abs`

---

## 5. 一致性结果

### 5.1 第一轮（seed=7）

结果文件：

- `outputs/pi_trt/pi05_onnx_consistency_20260306_190504/pi05_onnx/compare_metrics_torch_onnx.json`

结果如下：

- `vision_encoder`
  - cosine = `0.9999995835`
  - rmse = `0.0030733377`
  - mean_abs = `0.0019401394`
  - max_abs = `0.25`
- `prefix_cache`
  - cosine = `0.9999820530`
  - rmse = `0.0106171255`
  - mean_abs = `0.0045922482`
  - max_abs = `0.7265625`
- `denoise_step`
  - cosine = `0.9999995840`
  - rmse = `0.0009269530`
  - mean_abs = `0.0006145045`
  - max_abs = `0.0054931641`
- `action_chunk_pipeline`
  - cosine = `0.9999969599`
  - rmse = `0.0007567382`
  - mean_abs = `0.0003174737`
  - max_abs = `0.0043945312`

本轮最低 cosine：

- `prefix_cache = 0.9999820530`

### 5.2 第二轮复核（seed=17）

结果文件：

- `outputs/pi_trt/pi05_onnx_consistency_20260306_190504/pi05_onnx/compare_metrics_torch_onnx_seed17.json`

结果如下：

- `vision_encoder`
  - cosine = `0.9999995836`
  - rmse = `0.0030658792`
  - mean_abs = `0.0019314863`
  - max_abs = `0.1875`
- `prefix_cache`
  - cosine = `0.9999807321`
  - rmse = `0.0110971078`
  - mean_abs = `0.0045701312`
  - max_abs = `1.0`
- `denoise_step`
  - cosine = `0.9999996801`
  - rmse = `0.0008230689`
  - mean_abs = `0.0005999509`
  - max_abs = `0.00390625`
- `action_chunk_pipeline`
  - cosine = `0.9999988286`
  - rmse = `0.0004897515`
  - mean_abs = `0.0002355099`
  - max_abs = `0.0029296875`

本轮最低 cosine：

- `prefix_cache = 0.9999807321`

---

## 6. 结果判断

这次 ONNX 一致性我认为已经可以认定为**对齐成功**。

理由：

1. 两轮复核都通过了。
2. 最接近真实闭环的 `action_chunk_pipeline` cosine 都非常高：
   - 第一轮：`0.9999969599`
   - 第二轮：`0.9999988286`
3. 单步 `denoise_step` 也非常稳定：
   - 第一轮：`0.9999995840`
   - 第二轮：`0.9999996801`
4. 误差最大的模块是 `prefix_cache`，但最低 cosine 仍然在：
   - `0.99998+`

因此，从“Torch FP16 reference vs ONNX FP16”这个层面看：

> **当前 `pi05` 的 ONNX 导出链路已经可用，且模块级与端到端闭环都达到了很高的一致性。**

---

## 7. 产物说明

导出目录里除了 3 个主 ONNX 外，还有很多 `onnx__MatMul_*`、`onnx__Mul_*` 文件。

这不是异常，而是：

- 大图在导出时采用了 ONNX external data 形式
- 主 `.onnx` 文件和这些外部权重文件需要放在同一个目录下

当前主文件大小大致如下：

- `vision_encoder_fp16.onnx`: `792M`
- `prefix_cache_fp16.onnx`: `392K`
- `denoise_step_fp16.onnx`: `821M`

注意：

- 真正可用的导出产物是“主 `.onnx` + 同目录 external data 文件”的整体
- 后续如果做 TRT build，必须保留这些 sidecar 文件

---

## 8. 本次没有做的事情

这次**明确没有做**：

- TensorRT build
- TensorRT 一致性验证
- 真机推理
- 上机动作下发
- benchmark 延迟测试

这次只完成了：

- ONNX 导出
- Torch vs ONNX 一致性修正
- 两轮复核

---

## 9. 下一步建议

在当前结果基础上，下一步就可以进入 TRT 阶段了，建议顺序仍然是：

1. 先保留当前 3 个 ONNX 子图不动
2. 先做 TensorRT engine build
3. 再做 Torch FP16 vs TRT FP16 一致性
4. 仍然不要立刻上机
5. TRT 对齐通过后，再单独规划 runtime 接入

如果只从 ONNX 阶段看，当前这一步已经完成。

