# openpi_trt 实现链路报告

日期：2026-06-11  
仓库：`/data/cqy_workspace/flexible_lerobot`  
项目目录：`my_devs/openpi_trt`  
目标模型：LeRobot PI0.5 / SO101 top+wrist 双相机 checkpoint

## 1. 报告结论

`my_devs/openpi_trt` 实现的是 **LeRobot PI0.5 的 TensorRT 后端替换链路**。

它没有把整个机器人推理流程都放进 TensorRT，而是只替换 PI0.5 模型里最重的推理部分：

```text
PI05Pytorch.sample_actions(...)
```

最终主线实现是：

```text
prefix_cache TensorRT engine
  + denoise_step TensorRT engine
  + Python 10-step denoise loop
  -> 替换 policy.model.sample_actions(...)
```

实现后，上层调用仍然保持原样：

```text
predict_action(...)
  -> policy preprocessor
  -> PI05Policy.select_action(...)
  -> PI05Policy.predict_action_chunk(...)
  -> policy.model.sample_actions(...)
  -> policy postprocessor
```

区别在于，原来 `policy.model.sample_actions(...)` 是纯 PyTorch，现在可以被 patch 成 split TensorRT runtime。

当前实现过的 TRT 能力可以分成三类：

1. 早期局部子图 TRT：

   ```text
   suffix_embedding TRT
   prefix_embedding TRT
   ```

2. 最终主线 split TRT：

   ```text
   prefix_cache TRT
   denoise_step TRT
   ```

3. 上机和服务化接入：

   ```text
   patch_sample_actions_with_split_trt(...)
   run_pi05_split_trt_infer_so101.py
   vlash_iner server --backend tensorrt_split
   ```

其中当前最重要、最推荐继续维护的是：

```text
prefix_cache + denoise_step split TensorRT
```

## 2. 原始 PyTorch 推理链路

原始 PI0.5 推理入口在 LeRobot policy 侧。

高层流程是：

```text
robot observation
  -> policy preprocessor
  -> PI05Policy.select_action(...)
  -> PI05Policy.predict_action_chunk(...)
  -> PI05Pytorch.sample_actions(...)
  -> policy postprocessor
  -> robot action
```

其中 `sample_actions(...)` 是本项目 TensorRT 化的核心目标。

原始 PyTorch `sample_actions(...)` 可以拆成：

```text
输入：
  images:    top/wrist 两路图像 tensor
  img_masks: 两路图像 mask
  tokens:    language/state prompt tokens
  masks:     language attention mask
  noise:     [B, chunk_size, max_action_dim]

流程：
  1. embed_prefix(images, img_masks, tokens, masks)
  2. prefix transformer forward，得到 past_key_values
  3. x_t = noise
  4. for step in num_inference_steps:
       embed_suffix(x_t, timestep)
       transformer denoise forward
       action_out_proj
       x_t = x_t + dt * v_t
  5. return x_t

输出：
  actions: [B, 50, 32]
```

对 SO101 当前 checkpoint，后续上层会把动作截断成真实动作维度：

```text
[B, 50, 32] -> [B, 50, 6]
```

## 3. 为什么不是全图 TensorRT

最直接的想法是导出完整图：

```text
image_0
image_1
img_mask_0
img_mask_1
tokens
masks
noise
  -> sample_actions
  -> actions
```

也就是一个完整 TensorRT engine 替换整个 `sample_actions(...)`。

但实际 PI0.5 全图包含：

```text
vision encoder
language embedding
prefix attention mask
PaliGemma prefix forward
past_key_values cache
10-step denoise loop
Gemma expert transformer
RoPE
AdaRMS / adarms_cond
action_out_proj
```

完整导出遇到的问题：

```text
legacy ONNX exporter:
  RoPE / ComplexDouble 等图导出问题

dynamo exporter:
  图捕获时间长，完整图难以稳定产出
```

所以最终没有强行做一个单体 TensorRT engine，而是按模型结构拆成更可控的子图。

最终选择：

```text
prefix_cache graph:
  图像 + language tokens -> prefix_pad_masks + past_key_values

denoise_step graph:
  prefix_pad_masks + past_key_values + x_t + timestep -> v_t

Python runtime:
  保留 10-step denoise loop
```

## 4. 具体实现了哪些 TRT

### 4.1 suffix_embedding TRT

这是早期局部替换路线。

对应文件：

```text
ONNX wrapper:
  scripts/pi05_onnx_common.py
  PI05SuffixEmbeddingONNXWrapper

runtime patch:
  runtime/pi05_trt_suffix.py
  patch_embed_suffix_with_trt(...)

验证/运行：
  scripts/verify_lerobot_pi05_torch_trt.py --mode suffix_embedding
  scripts/smoke_pi05_camera_torch_trt.py
  scripts/benchmark_pi05_camera_torch_trt.py
  scripts/run_pi05_trt_infer_so101.py
```

导出边界：

```text
输入：
  noisy_actions: [1, 50, 32]
  timestep:      [1]

输出：
  suffix_embs
  adarms_cond
```

它替换的是：

```python
policy.model.embed_suffix(noisy_actions, timestep)
```

实现方式：

```text
1. 导出 noisy_actions + timestep -> suffix_embs + adarms_cond
2. 构建 suffix_embedding TensorRT engine
3. runtime 中 patch policy.model.embed_suffix(...)
4. masks 仍然在 Python 中按 PI0.5 固定规则重建
```

patch 后流程：

```text
sample_actions(...)
  -> prefix 仍走 PyTorch
  -> denoise loop 仍走 PyTorch
  -> 每次需要 embed_suffix 时调用 TensorRT engine
```

这条路线的价值：

- 验证了最小子图 TRT patch 的可行性。
- 上机脚本和 camera smoke 都比较容易接。

这条路线的局限：

- `embed_suffix` 不是最大瓶颈。
- 真正重的 transformer denoise forward 仍然在 PyTorch。
- 整体加速有限。

所以它是早期实验路线，不是最终主线。

### 4.2 prefix_embedding TRT

这是另一个中间实验子图。

对应文件：

```text
ONNX wrapper:
  scripts/pi05_onnx_common.py
  PI05PrefixEmbeddingONNXWrapper

导出：
  scripts/export_lerobot_pi05_onnx.py --mode prefix_embedding

验证：
  scripts/verify_lerobot_pi05_torch_onnx.py --mode prefix_embedding
  scripts/verify_lerobot_pi05_torch_trt.py --mode prefix_embedding
```

导出边界：

```text
输入：
  image_0:    [1, 3, 224, 224]
  image_1:    [1, 3, 224, 224]
  img_mask_0: [1]
  img_mask_1: [1]
  tokens:     [1, 200]
  masks:      [1, 200]

输出：
  prefix_embs
  prefix_pad_masks
  prefix_att_masks
```

它对应的是：

```python
policy.model.embed_prefix(images, img_masks, tokens, masks)
```

这条路线验证了 prefix embedding 子图可以导出和构建 TensorRT engine。

但最终主线没有停在 `prefix_embedding`，因为只导出 prefix embedding 还不能替换最重的 prefix transformer forward，也不能生成 `past_key_values`。因此后续升级成了 `prefix_cache`。

### 4.3 prefix_cache TRT

这是最终主线的第一块 engine。

对应文件：

```text
ONNX wrapper:
  scripts/pi05_onnx_common.py
  PI05PrefixCacheONNXWrapper

导出：
  scripts/export_lerobot_pi05_onnx.py --mode prefix_cache

构建：
  scripts/build_lerobot_pi05_engine.py

验证：
  scripts/verify_lerobot_pi05_torch_onnx.py --mode prefix_cache
  scripts/verify_lerobot_pi05_torch_trt.py --mode prefix_cache

runtime:
  runtime/pi05_trt_split.py
  PI05TensorRTSplitRuntime
```

导出边界：

```text
输入：
  image_0:    [1, 3, 224, 224] float32
  image_1:    [1, 3, 224, 224] float32
  img_mask_0: [1] bool
  img_mask_1: [1] bool
  tokens:     [1, 200] int64
  masks:      [1, 200] bool

输出：
  prefix_pad_masks: [1, 712] bool
  past_key_values.0.key
  past_key_values.0.value
  ...
  past_key_values.17.key
  past_key_values.17.value
```

每层 `past_key_values` shape：

```text
key:   [1, 1, 712, 256] float32
value: [1, 1, 712, 256] float32
```

它实现的是原 PyTorch 中这段逻辑：

```text
embed_prefix(...)
  -> prefix attention mask
  -> prefix position ids
  -> PaliGemma prefix forward(use_cache=True)
  -> past_key_values
```

关键实现点：

1. `DynamicCache` 不能直接作为 TensorRT I/O。

   所以项目实现了：

   ```text
   flatten_past_key_values(...)
   unflatten_past_key_values(...)
   prefix_cache_tensor_names(...)
   denoise_step_input_names(...)
   ```

   把 cache 展平成：

   ```text
   past_key_values.0.key
   past_key_values.0.value
   ...
   past_key_values.17.key
   past_key_values.17.value
   ```

2. attention mask 做了 ONNX-friendly 处理。

   原始 attention mask 构造里有 bool `cumsum` 风险，因此 wrapper 里使用：

   ```python
   cumsum = torch.cumsum(att_masks.to(dtype=torch.int64), dim=1)
   ```

3. wrapper 中固定使用 eager attention 路径，避免导出时进入不稳定实现。

产物：

```text
my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp32.onnx
my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp32.engine
```

注意：

```text
prefix_cache engine 约 11 GB
```

因为它包含 vision/language prefix 相关权重和 PaliGemma prefix forward。

### 4.4 denoise_step TRT

这是最终主线的第二块 engine，也是实际反复执行的核心 engine。

对应文件：

```text
ONNX wrapper:
  scripts/pi05_onnx_common.py
  PI05DenoiseStepONNXWrapper

导出：
  scripts/export_lerobot_pi05_onnx.py --mode denoise_step

构建：
  scripts/build_lerobot_pi05_engine.py

验证：
  scripts/verify_lerobot_pi05_torch_onnx.py --mode denoise_step
  scripts/verify_lerobot_pi05_torch_trt.py --mode denoise_step

runtime:
  runtime/pi05_trt_split.py
  PI05TensorRTSplitRuntime
```

导出边界：

```text
输入：
  prefix_pad_masks: [1, 712] bool
  past_key_values.0.key/value ... past_key_values.17.key/value
  x_t:      [1, 50, 32] float32
  timestep: [1] float32

输出：
  v_t: [1, 50, 32] float32
```

它实现的是原 PyTorch 中单步 denoise：

```text
embed_suffix(x_t, timestep)
  -> suffix masks
  -> full attention mask(prefix + suffix)
  -> position ids
  -> Gemma expert transformer forward
  -> action_out_proj
  -> v_t
```

runtime 中会反复调用它：

```text
x_t = noise
for step in range(num_inference_steps):
    timestep = 1.0 + step * dt
    v_t = denoise_step_engine(prefix_cache, x_t, timestep)
    x_t = x_t + dt * v_t
actions = x_t
```

产物：

```text
my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp32.onnx
my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp32.engine
```

还实现过实验性 FP16 engine：

```text
my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp16.engine
my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp16_constrained.engine
my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp16_rebuild.engine
```

其中 constrained FP16 版本会在 TensorRT build 阶段对 LayerNorm、Softmax、Reduce 等敏感层使用 FP32 precision constraints，以减少数值漂移。

### 4.5 split sample_actions TRT

严格说它不是一个单独 engine，而是一个组合 runtime。

对应文件：

```text
runtime/pi05_trt_split.py
  PI05TensorRTSplitRuntime
  patch_sample_actions_with_split_trt(...)

验证：
  scripts/verify_lerobot_pi05_split_trt_sample_actions.py

camera smoke:
  scripts/smoke_pi05_camera_split_trt.py

benchmark:
  scripts/benchmark_pi05_camera_split_trt.py

SO101 上机：
  scripts/run_pi05_split_trt_infer_so101.py
```

组合关系：

```text
PI05TensorRTSplitRuntime
  prefix_engine = TorchTensorRTEngine(prefix_cache.engine)
  denoise_engine = TorchTensorRTEngine(denoise_step.engine)
```

执行时：

```text
prefix_outputs = prefix_engine(image_0, image_1, img_mask_0, img_mask_1, tokens, masks)
cache_inputs = prefix_outputs[prefix_pad_masks + past_key_values.*]

x_t = noise
for step in range(num_steps):
    v_t = denoise_engine(cache_inputs, x_t, timestep)
    x_t = x_t + dt * v_t

return x_t
```

patch 方式：

```python
original_sample_actions = policy.model.sample_actions
policy.model.sample_actions = trt_sample_actions
```

patch 后，上层仍然调用：

```python
policy.model.sample_actions(images, img_masks, tokens, masks, noise=noise)
```

但实际执行已经变成 TensorRT split runtime。

## 5. TensorRT engine 是怎么构建的

### 5.1 从 checkpoint 恢复 PyTorch 模型

公共加载逻辑在：

```text
scripts/pi05_onnx_common.py
load_policy(...)
```

它做：

```text
PreTrainedConfig.from_pretrained(policy_path)
get_policy_class(config.type)
policy_class.from_pretrained(...)
policy.to(device)
policy.eval()
disable gradient checkpointing
```

重要点：

```text
TensorRT 不是从 model.safetensors 直接生成；
必须先恢复 PyTorch policy，再通过 torch.onnx.export(...) 导出选定子图。
```

### 5.2 构造固定导出输入

`pi05_onnx_common.py` 会构造和当前 SO101 checkpoint 匹配的输入：

```text
image_0:    [1, 3, 224, 224]
image_1:    [1, 3, 224, 224]
img_mask_0: [1]
img_mask_1: [1]
tokens:     [1, 200]
masks:      [1, 200]
noise:      [1, 50, 32]
```

state 不是单独导出输入，因为 PI0.5 的 state 已经在 preprocessor 中：

```text
state -> normalize -> discretize -> prompt -> tokenizer -> tokens/masks
```

### 5.3 选择导出 wrapper

`export_lerobot_pi05_onnx.py` 根据 `--mode` 选择 wrapper：

```text
suffix_embedding  -> PI05SuffixEmbeddingONNXWrapper
prefix_embedding  -> PI05PrefixEmbeddingONNXWrapper
prefix_cache      -> PI05PrefixCacheONNXWrapper
denoise_step      -> PI05DenoiseStepONNXWrapper
sample_actions    -> PI05SampleActionsONNXWrapper
```

主链路选择：

```text
prefix_cache
denoise_step
```

### 5.4 ONNX 导出

导出调用：

```python
torch.onnx.export(
    wrapper,
    inputs,
    output,
    input_names=input_names,
    output_names=output_names,
    opset_version=19,
    do_constant_folding=True,
)
```

主链路导出产物：

```text
pi05_so101_prefix_cache_b1_fp32.onnx
pi05_so101_denoise_step_b1_fp32.onnx
```

### 5.5 TensorRT build

构建脚本：

```text
scripts/build_lerobot_pi05_engine.py
```

它使用 TensorRT Python API：

```text
trt.Builder
trt.OnnxParser
builder.create_builder_config()
builder.build_serialized_network(...)
```

支持：

```text
--precision fp32
--precision fp16
```

FP16 build 时会启用：

```text
trt.BuilderFlag.FP16
trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS
```

并对数值敏感层设置 FP32：

```text
LayerNorm / norm
Softmax
Reduce
Sqrt / Pow
部分 Elementwise
```

这样得到 `.engine`：

```text
pi05_so101_prefix_cache_b1_fp32.engine
pi05_so101_denoise_step_b1_fp32.engine
pi05_so101_denoise_step_b1_fp16_constrained.engine
```

## 6. TensorRT runtime 怎么执行

### 6.1 底层 engine 执行器

底层执行器：

```text
runtime/trt_engine.py
TorchTensorRTEngine
```

它做：

```text
1. 反序列化 TensorRT engine
2. 创建 execution context
3. 读取 engine input/output tensor 名称
4. 读取每个 tensor 的 dtype
5. 接收 torch CUDA tensor 输入
6. 设置 input shape
7. 用 tensor.data_ptr() 绑定 TensorRT I/O
8. 分配 torch CUDA output tensor
9. execute_async_v3(...)
10. 返回 torch tensor 输出
```

所以 split runtime 可以像调用普通 Python 函数一样调用 engine：

```python
outputs = engine(**inputs)
```

### 6.2 split runtime 执行器

split runtime：

```text
runtime/pi05_trt_split.py
PI05TensorRTSplitRuntime
```

初始化：

```python
self.prefix_engine = TorchTensorRTEngine(prefix_engine_path)
self.denoise_engine = TorchTensorRTEngine(denoise_engine_path)
```

推理：

```python
prefix_outputs = self.prefix_engine(...)
cache_inputs = {name: prefix_outputs[name] for name in self.cache_names}

for step in range(num_steps):
    denoise_inputs = dict(cache_inputs)
    denoise_inputs["x_t"] = x_t
    denoise_inputs["timestep"] = timestep
    denoise_outputs = self.denoise_engine(...)
    v_t = denoise_outputs["v_t"]
    x_t = x_t + dt * v_t
```

dtype 处理：

```text
_cast_inputs_for_engine(...)
```

会根据 engine 的输入 dtype，把传入 tensor 转成 engine 期望 dtype。

## 7. 实现后的推理流程

### 7.1 本地 SO101 推理流程

入口：

```text
scripts/run_pi05_split_trt_infer_so101.py
```

启动后先做：

```text
1. 加载 LeRobot PI0.5 policy
2. 加载 prefix_cache engine
3. 加载 denoise_step engine
4. patch policy.model.sample_actions(...)
5. 加载 checkpoint preprocessor/postprocessor
6. 连接 SO101/SO100 和 top/wrist 相机
```

运行时每一轮：

```text
robot.get_observation()
  -> robot_observation_processor
  -> build_dataset_frame
  -> predict_action(...)
      -> policy preprocessor
      -> PI05Policy.select_action(...)
          如果 action queue 还有动作：
            直接弹出下一步 action
          如果 action queue 空：
            -> PI05Policy.predict_action_chunk(...)
                -> policy._preprocess_images(...)
                -> tokens / masks
                -> policy.model.sample_actions(...)
                    此时已经被 patch 成 split TensorRT
                    -> prefix_cache engine
                    -> denoise_step engine x 10
                    -> action chunk [1, 50, 32]
                -> 截断 action dim 到 6
            -> action queue
      -> policy postprocessor
  -> make_robot_action
  -> robot_action_processor
  -> robot.send_action(...)
```

和原始 PyTorch 流程相比，变化点只有：

```text
policy.model.sample_actions(...)
```

从 PyTorch 变成了：

```text
split TensorRT runtime
```

其他如 preprocessor、postprocessor、robot processor、action queue 都继续复用 LeRobot 原实现。

### 7.2 split TensorRT 内部推理流程

当上层调用：

```python
policy.model.sample_actions(images, img_masks, tokens, masks, noise=noise)
```

实际执行：

```text
Step A: prefix_cache engine

输入：
  image_0
  image_1
  img_mask_0
  img_mask_1
  tokens
  masks

输出：
  prefix_pad_masks
  past_key_values.0.key/value
  ...
  past_key_values.17.key/value
```

然后：

```text
Step B: Python denoise loop

x_t = noise
dt = -1.0 / num_inference_steps

for step in range(num_inference_steps):
    timestep = 1.0 + step * dt
```

每一步调用：

```text
Step C: denoise_step engine

输入：
  prefix_pad_masks
  past_key_values.*
  x_t
  timestep

输出：
  v_t
```

再更新：

```text
x_t = x_t + dt * v_t
```

循环结束：

```text
actions = x_t
return actions
```

完整图：

```text
images/tokens
  -> prefix_cache TRT
       -> prefix_pad_masks
       -> past_key_values

noise
  -> x_t
  -> denoise_step TRT, step 0
  -> denoise_step TRT, step 1
  -> ...
  -> denoise_step TRT, step 9
  -> actions [1, 50, 32]
```

### 7.3 接入 vlash_iner 服务化后的流程

`vlash_iner` 服务端支持：

```text
--backend tensorrt_split
```

服务端启动后：

```text
1. 加载 LeRobot PI0.5 policy
2. 加载 checkpoint preprocessor/postprocessor
3. 加载 prefix_cache engine
4. 加载 denoise_step engine
5. patch policy.model.sample_actions(...)
6. 启动 FastAPI /infer
```

客户端请求：

```text
observation + task + robot_type
  -> HTTP /infer
```

服务端推理：

```text
prepare_observation_for_inference
  -> policy preprocessor
  -> policy.predict_action_chunk(...)
      -> patched sample_actions(...)
          -> prefix_cache TRT
          -> denoise_step TRT x 10
  -> policy postprocessor
  -> action_chunk
```

服务端返回：

```text
action_chunk: [50, 6]
infer_time_s
backend: tensorrt_split
```

机器人客户端再用 `AsyncChunkManager` 执行 action chunk。

## 8. 数值验证链路

整个实现不是直接导出后上机，而是逐层验证。

验证顺序：

```text
1. Torch vs ONNX: denoise_step
2. Torch vs TRT: denoise_step
3. Torch vs ONNX/TRT: prefix_cache
4. Torch sample_actions vs split TRT sample_actions
5. 真实相机 camera smoke
6. benchmark
7. 上机 / 服务化验收
```

关键结果：

### 8.1 denoise_step Torch vs ONNX

报告：

```text
artifacts/verify_torch_onnx_denoise_step_fp32.json
```

结果：

```text
passed_allclose: True
v_t mean_abs_diff: 0.00020244
v_t max_abs_diff:  0.00107265
cosine:            0.99999996
```

### 8.2 denoise_step Torch vs TRT

报告：

```text
artifacts/verify_torch_trt_denoise_step_fp32.json
```

结果：

```text
passed_allclose: True
v_t mean_abs_diff: 0.00020167
v_t max_abs_diff:  0.00107288
cosine:            0.99999996
```

### 8.3 prefix_cache Torch vs ONNX/TRT

prefix cache 是中间态，严格逐元素比较更敏感。

观察到：

```text
prefix_pad_masks 完全一致
past_key_values 整体 cosine 很高
后层 value cache 有少量 outlier
```

使用：

```text
rtol=0.02, atol=1.0
```

复验通过。

这些 outlier 没有在最终 action chunk 上明显放大。

### 8.4 split sample_actions Torch vs TRT

报告：

```text
artifacts/verify_split_trt_sample_actions_fp32.json
```

结果：

```text
passed_allclose: True
actions mean_abs_diff: 0.00008763
actions max_abs_diff:  0.00089842
cosine:                0.99999980
```

### 8.5 camera smoke

报告：

```text
artifacts/camera_smoke_split_trt_fp32.json
```

结果：

```text
passed_allclose: True
action chunk mean_abs_diff: 0.00023921
action chunk max_abs_diff:  0.00103939
cosine:                    0.99999993
robot_connected: False
robot_action_sent: False
```

## 9. 性能结果

### 9.1 FP32 split TRT

报告：

```text
artifacts/benchmark_camera_split_trt_fp32.json
```

真实相机输入下：

```text
PyTorch sample_actions mean:   173.47 ms
split TRT sample_actions mean:  84.81 ms
speedup:                         2.05x
```

一致性：

```text
max_abs_diff:      0.000847
cosine_similarity: 0.99999994
```

### 9.2 constrained FP16 denoise

报告：

```text
artifacts/benchmark_camera_split_trt_fp16_constrained.json
```

真实相机输入下：

```text
PyTorch sample_actions mean:   173.04 ms
split TRT sample_actions mean:  74.69 ms
speedup:                         2.32x
```

一致性：

```text
max_abs_diff:      0.005054
cosine_similarity: 0.99999939
```

这说明 FP16 constrained denoise 有更高速度，但数值误差也比 FP32 稍大，需要结合实机任务成功率继续验证。

## 10. 当前推荐的实现组合

当前最稳的主组合：

```text
prefix_cache:
  pi05_so101_prefix_cache_b1_fp32.engine

denoise_step:
  pi05_so101_denoise_step_b1_fp32.engine

runtime:
  PI05TensorRTSplitRuntime

patch:
  patch_sample_actions_with_split_trt(...)
```

更激进的候选组合：

```text
prefix_cache:
  pi05_so101_prefix_cache_b1_fp32.engine

denoise_step:
  pi05_so101_denoise_step_b1_fp16_constrained.engine
```

推荐继续用 FP32 split TRT 做稳定上机 baseline，再把 constrained FP16 denoise 纳入同样的 L1/L2/L3 验收。

## 11. 关键注意点

### 11.1 TensorRT 没有替换 preprocessor/postprocessor

TensorRT 只替换：

```text
sample_actions(...)
```

或者早期路线里的：

```text
embed_suffix(...)
```

没有替换：

```text
policy_preprocessor
tokenizer
state discretization
policy_postprocessor
robot action processor
```

所以 checkpoint processor 仍然必须加载，不能跳过。

### 11.2 state 已经进入 tokens

当前 PI0.5 的 state 不是 TensorRT engine 的单独输入。

真实流程是：

```text
observation.state
  -> policy preprocessor
  -> normalize / discretize
  -> prompt
  -> tokenizer
  -> tokens / masks
```

TensorRT engine 看到的是：

```text
tokens
masks
```

### 11.3 TensorRT engine 绑定构建环境

`.engine` 不是通用模型文件。它和：

```text
GPU 架构
TensorRT 版本
CUDA 版本
precision
builder tactic
```

强相关。换设备或环境后要重新 build 并重新验证。

### 11.4 prefix_cache 中间误差不能单独下结论

prefix cache 的中间 `past_key_values` 会有少量 outlier，但最终 action chunk 验证和 camera smoke 更能说明实际部署误差。

因此判断是否可用时，至少要看：

```text
split sample_actions 一致性
camera smoke 一致性
真实任务成功率
```

### 11.5 不要直接跳到发动作

推荐 gate：

```text
Torch vs ONNX
  -> Torch vs TRT
  -> split sample_actions
  -> camera smoke
  -> readonly service/client
  -> lowspeed real robot
  -> main real robot
```

## 12. 一句话总结

`openpi_trt` 最终实现的核心不是一个单独的 TensorRT engine，而是一套 **split TensorRT sample_actions 后端**：

```text
prefix_cache TRT
  负责图像/语言 prefix 编码和 past_key_values 生成

denoise_step TRT
  负责每一步 action denoise transformer forward

Python denoise loop
  负责 10 步迭代调度和 x_t 更新

patch_sample_actions_with_split_trt
  负责把这套 runtime 接回原始 LeRobot PI0.5 policy
```

实现后的推理流程保持了 LeRobot 上层接口不变，只把最重的 `PI05Pytorch.sample_actions(...)` 从 PyTorch 后端替换为 TensorRT split 后端。

这使得：

```text
robot / processor / policy / postprocessor / service client
```

都能继续复用原来的代码，而模型重推理部分获得约 2 倍的加速。
