# LeRobot PI0.5 TensorRT 优化完整技术报告

## 1. 摘要

本次工作针对 LeRobot 版 PI0.5 policy 的 `sample_actions(...)` 推理链路做 TensorRT 加速，目标是替换推理过程中最耗时的 transformer denoise 部分，同时保证输出动作与原始 PyTorch 路径保持一致。

最终完成了一版可上机运行的 split TensorRT 后端：

```text
prefix_cache engine
  + denoise_step engine
  + Python 10-step denoise loop
  -> sample_actions
  -> policy postprocessor
  -> robot action
```

当前结果：

```text
原始 PyTorch sample_actions 平均耗时:      173.47 ms
split TensorRT sample_actions 平均耗时:    84.81 ms
端到端 sample_actions 提速:                2.05x
真实 camera smoke action max_abs_diff:     0.001039
真实上机:                                  已验证能抓取
```

本次所有开发都限制在：

```text
my_devs/openpi_trt/
```

没有改动 `src/lerobot/`。

## 2. 背景与问题

原始实时推理入口是：

```text
my_devs/train/pi/so101/run_pi05_infer.py
```

它的主要链路是：

```text
robot.get_observation()
  -> robot_observation_processor
  -> build_dataset_frame
  -> predict_action(...)
  -> policy preprocessor
  -> PI05Policy.select_action(...)
  -> PI05Policy.predict_action_chunk(...)
  -> PI05Pytorch.sample_actions(...)
  -> policy postprocessor
  -> make_robot_action
  -> robot_action_processor
  -> robot.send_action(...)
```

真正重的部分在：

```text
PI05Pytorch.sample_actions(...)
```

特别是 denoise loop 里每一步的 Gemma expert transformer forward。

原始 `sample_actions(...)` 大致逻辑：

```text
1. embed_prefix(images, img_masks, tokens, masks)
2. PaliGemma prefix forward，生成 past_key_values
3. x_t = noise
4. for step in num_inference_steps:
       denoise_step(prefix_pad_masks, past_key_values, x_t, timestep)
       x_t = x_t + dt * v_t
5. return x_t
```

当前 checkpoint 配置：

```text
policy:                 PI0.5
paligemma_variant:      gemma_2b
action_expert_variant:  gemma_300m
input state dim:        6
output action dim:      6
max_state_dim:          32
max_action_dim:         32
chunk_size:             50
n_action_steps:         50
num_inference_steps:    10
image_resolution:       224 x 224
tokenizer_max_length:   200
cameras:                top + wrist
```

## 3. 为什么不直接导出完整 sample_actions

最初尝试过完整导出：

```text
image_0 + image_1 + img_mask_0 + img_mask_1 + tokens + masks + noise
  -> actions
```

但完整图包含：

```text
vision encoder
language embedding
prefix attention mask
prefix past_key_values
10-step denoise loop
Gemma expert transformer forward
RoPE
AdaRMS / adarms_cond
action_out_proj
```

完整单体导出遇到两个实际问题：

```text
legacy ONNX exporter:
  ComplexDouble / RoPE 相关导出失败

dynamo ONNX exporter:
  图捕获时间过长，长时间无稳定产物
```

因此最终没有继续硬导单体 `sample_actions`，而是改成 split graph。

这个选择有几个好处：

```text
1. 每个子图边界更小，更容易导出和定位问题。
2. past_key_values 可以显式展开成稳定 TensorRT I/O。
3. denoise loop 保留在 Python，方便逐步验证每一步数值。
4. 后续可以单独优化 prefix_cache 或 denoise_step，不互相绑死。
```

## 4. 最终拆分方案

最终拆成两个 engine：

```text
engine 1: prefix_cache
engine 2: denoise_step
```

Python 里保留 denoise loop：

```text
x_t = noise
for step in range(num_inference_steps):
    timestep = 1.0 + step * dt
    v_t = denoise_step_engine(prefix_cache, x_t, timestep)
    x_t = x_t + dt * v_t
actions = x_t
```

### 4.1 prefix_cache engine

输入：

```text
image_0:    [1, 3, 224, 224] float32
image_1:    [1, 3, 224, 224] float32
img_mask_0: [1] bool
img_mask_1: [1] bool
tokens:     [1, 200] int64
masks:      [1, 200] bool
```

PyTorch 对应逻辑：

```python
prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
    images, img_masks, tokens, masks
)

prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)

_, past_key_values = model.paligemma_with_expert.forward(
    attention_mask=prefix_att_2d_masks_4d,
    position_ids=prefix_position_ids,
    past_key_values=None,
    inputs_embeds=[prefix_embs, None],
    use_cache=True,
)
```

输出：

```text
prefix_pad_masks: [1, 712] bool
past_key_values.0.key
past_key_values.0.value
...
past_key_values.17.key
past_key_values.17.value
```

`past_key_values` 真实结构：

```text
type: DynamicCache
layers: 18
each key shape:   [1, 1, 712, 256] float32
each value shape: [1, 1, 712, 256] float32
```

### 4.2 denoise_step engine

输入：

```text
prefix_pad_masks: [1, 712] bool
past_key_values.0.key/value ... past_key_values.17.key/value
x_t:      [1, 50, 32] float32
timestep: [1] float32
```

PyTorch 对应逻辑：

```python
v_t = model.denoise_step(
    prefix_pad_masks=prefix_pad_masks,
    past_key_values=past_key_values,
    x_t=x_t,
    timestep=timestep,
)
```

内部包含：

```text
embed_suffix(x_t, timestep)
suffix attention mask
position_ids
Gemma expert transformer forward
adarms_cond
action_out_proj
```

输出：

```text
v_t: [1, 50, 32] float32
```

## 5. 关键工程实现

### 5.1 past_key_values 展平

TensorRT I/O 不能直接绑定 Python `DynamicCache` 对象，所以需要把 cache 展平成稳定 tensor 列表：

```text
past_key_values.0.key
past_key_values.0.value
past_key_values.1.key
past_key_values.1.value
...
past_key_values.17.key
past_key_values.17.value
```

相关实现：

```text
my_devs/openpi_trt/scripts/pi05_onnx_common.py
```

核心函数：

```text
prefix_cache_tensor_names(...)
denoise_step_input_names(...)
flatten_past_key_values(...)
unflatten_past_key_values(...)
```

在 denoise step wrapper 内部会重新构造 `DynamicCache`：

```text
flat tensor list -> DynamicCache -> model.paligemma_with_expert.forward(...)
```

### 5.2 ONNX-friendly attention mask

ONNX Runtime 不接受 bool tensor 直接做 `CumSum`。原始 `make_att_2d_masks(...)` 里会对 `att_masks` 做：

```python
torch.cumsum(att_masks, dim=1)
```

当 `att_masks` 是 bool 时，导出的 ONNX 会出现非法 bool `CumSum`。

因此在 `my_devs/openpi_trt` 内部实现了 ONNX-friendly 版本：

```text
make_att_2d_masks_for_onnx(...)
```

核心变化：

```python
cumsum = torch.cumsum(att_masks.to(dtype=torch.int64), dim=1)
```

这只影响导出 wrapper，不改 `src/lerobot`。

### 5.3 split TensorRT runtime

运行时实现：

```text
my_devs/openpi_trt/runtime/pi05_trt_split.py
```

核心类：

```text
PI05TensorRTSplitRuntime
```

它负责：

```text
1. 加载 prefix_cache engine
2. 加载 denoise_step engine
3. 调 prefix_cache engine 得到 prefix_pad_masks + past_key_values
4. Python 中循环 10 次 denoise_step engine
5. 返回 actions
```

同时提供 patch 函数：

```text
patch_sample_actions_with_split_trt(...)
```

这个函数会把：

```python
policy.model.sample_actions(...)
```

替换成 split TensorRT 版本。这样上层 `predict_action(...)`、preprocessor、postprocessor、机器人 action 发送逻辑都不用改。

## 6. 主要文件与职责

### 导出/验证公共逻辑

```text
my_devs/openpi_trt/scripts/pi05_onnx_common.py
```

职责：

```text
加载 PI0.5 policy
构造合成 batch
生成导出输入
定义 ONNX wrapper
past_key_values flatten/unflatten
数值统计
ONNX exporter 兼容 patch
```

### ONNX 导出

```text
my_devs/openpi_trt/scripts/export_lerobot_pi05_onnx.py
```

支持模式：

```text
sample_actions
suffix_embedding
prefix_embedding
prefix_cache
denoise_step
```

本次主要使用：

```text
--mode prefix_cache
--mode denoise_step
```

### TensorRT engine 构建

```text
my_devs/openpi_trt/scripts/build_lerobot_pi05_engine.py
```

使用 TensorRT Python API 构建 engine，不依赖 `trtexec`。

### Torch vs ONNX 验证

```text
my_devs/openpi_trt/scripts/verify_lerobot_pi05_torch_onnx.py
```

### Torch vs TensorRT 验证

```text
my_devs/openpi_trt/scripts/verify_lerobot_pi05_torch_trt.py
```

### 完整 split sample_actions 验证

```text
my_devs/openpi_trt/scripts/verify_lerobot_pi05_split_trt_sample_actions.py
```

### camera smoke

```text
my_devs/openpi_trt/scripts/smoke_pi05_camera_split_trt.py
```

只读真实相机，不连接机器人，不发送 action。

### benchmark

```text
my_devs/openpi_trt/scripts/benchmark_pi05_camera_split_trt.py
```

对比：

```text
原始 PyTorch sample_actions
vs
split TensorRT sample_actions
```

### 真机上机脚本

```text
my_devs/openpi_trt/scripts/run_pi05_split_trt_infer_so101.py
```

这是当前推荐使用的 TensorRT 推理入口。

## 7. 产物

### ONNX

```text
my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp32.onnx
my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp32.onnx
```

### TensorRT engine

```text
my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp32.engine
my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp32.engine
```

### 产物大小

```text
prefix_cache engine:  11.26 GB
denoise_step engine:  1.72 GB
```

prefix_cache engine 很大，原因是它包含了 vision/language embedding 和 PaliGemma prefix forward 相关权重。

## 8. 数值验证结果

### 8.1 denoise_step: Torch vs ONNX

报告：

```text
my_devs/openpi_trt/artifacts/verify_torch_onnx_denoise_step_fp32.json
```

结果：

```text
passed_allclose: True
rtol: 0.02
atol: 0.1
v_t mean_abs_diff: 0.00020244
v_t max_abs_diff:  0.00107265
v_t cosine:        0.99999996
```

### 8.2 denoise_step: Torch vs TensorRT

报告：

```text
my_devs/openpi_trt/artifacts/verify_torch_trt_denoise_step_fp32.json
```

结果：

```text
passed_allclose: True
rtol: 0.02
atol: 0.1
v_t mean_abs_diff: 0.00020167
v_t max_abs_diff:  0.00107288
v_t cosine:        0.99999996
```

### 8.3 prefix_cache: Torch vs ONNX/TRT

中间 cache tensor 的验证情况比较特殊：

```text
prefix_pad_masks: 完全一致
past_key_values: 整体 cosine 很高，约 0.99999+
```

但在严格：

```text
rtol=0.02
atol=0.1
```

下，后层 value cache 有少量逐元素 outlier，因此 prefix cache 中间 tensor 严格 allclose 未完全通过。

复验使用：

```text
rtol=0.02
atol=1.0
```

通过。

报告：

```text
my_devs/openpi_trt/artifacts/verify_torch_onnx_prefix_cache_fp32_atol1.json
my_devs/openpi_trt/artifacts/verify_torch_trt_prefix_cache_fp32_atol1.json
```

最大 outlier：

```text
Torch vs ONNX prefix_cache max_abs_diff: 0.97361374
Torch vs TRT  prefix_cache max_abs_diff: 0.37520790
```

这个 outlier 没有在下游放大，最终 `denoise_step`、完整 `sample_actions`、camera smoke 都保持高一致性。

### 8.4 完整 split sample_actions: Torch vs TensorRT

报告：

```text
my_devs/openpi_trt/artifacts/verify_split_trt_sample_actions_fp32.json
```

结果：

```text
passed_allclose: True
rtol: 0.02
atol: 0.1
actions mean_abs_diff: 0.00008763
actions max_abs_diff:  0.00089842
actions cosine:        0.99999980
```

这个验证直接比较：

```text
PyTorch sample_actions
vs
prefix_cache engine + denoise_step engine + Python 10-step loop
```

输出形状：

```text
[1, 50, 32]
```

### 8.5 camera smoke: Torch vs split TensorRT

报告：

```text
my_devs/openpi_trt/artifacts/camera_smoke_split_trt_fp32.json
```

结果：

```text
passed_allclose: True
rtol: 0.02
atol: 0.1
action chunk mean_abs_diff: 0.00023921
action chunk max_abs_diff:  0.00103939
action chunk cosine:        0.99999993
robot_connected: False
robot_action_sent: False
```

camera smoke 使用真实相机：

```text
top:   /dev/video4
wrist: /dev/video6
```

只读相机，不连接机器人，不发送 action。

## 9. 性能结果

benchmark 报告：

```text
my_devs/openpi_trt/artifacts/benchmark_camera_split_trt_fp32.json
```

测试条件：

```text
真实 top/wrist camera frame
warmup_runs: 1
runs: 5
model dtype: float32
engine precision: fp32
num_inference_steps: 10
```

结果：

```text
PyTorch full sample_actions:
  mean:   173.47 ms
  median: 173.45 ms
  min:    170.59 ms
  max:    175.88 ms
  std:    1.76 ms

split TensorRT full sample_actions:
  mean:   84.81 ms
  median: 84.90 ms
  min:    84.60 ms
  max:    84.96 ms
  std:    0.14 ms
```

提速：

```text
speedup = 173.47 / 84.81 = 2.0453x
```

即当前 FP32 split TensorRT 版本约 **2.05 倍提速**。

一致性：

```text
action chunk mean_abs_diff: 0.00021875
action chunk max_abs_diff:  0.00084716
action chunk cosine:        0.99999994
```

## 10. 为什么能提速

之前只替换 `embed_suffix(...)` 时，端到端收益很小：

```text
embed_suffix PyTorch: 约 0.15 ms
完整 sample_actions: 约 102 ms
```

所以 suffix embedding 虽然本身能被 TRT 加速，但它只占总耗时极小一部分。

真正重的是：

```text
denoise loop 中每一步的 Gemma expert transformer forward
```

当前 `num_inference_steps=10`，所以 denoise step 会跑 10 次。

本次 `denoise_step engine` 把单步里的重计算放进 TensorRT：

```text
embed_suffix
attention mask
position_ids
past_key_values
Gemma expert transformer forward
adarms_cond
action_out_proj
```

因此端到端 `sample_actions` 从约 173 ms 降到约 85 ms。

## 11. 真机上机脚本

脚本：

```text
my_devs/openpi_trt/scripts/run_pi05_split_trt_infer_so101.py
```

安全检查：

```bash
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/run_pi05_split_trt_infer_so101.py \
  --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
  --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00 \
  --top-cam /dev/video4 \
  --wrist-cam /dev/video6 \
  --task "Put the eraser into the small box" \
  --run-time-s 120 \
  --check-policy-load
```

真实控制：

```bash
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/run_pi05_split_trt_infer_so101.py \
  --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
  --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00 \
  --top-cam /dev/video4 \
  --wrist-cam /dev/video6 \
  --task "Put the eraser into the small box" \
  --run-time-s 120 \
  --confirm-control
```

不加 `--confirm-control` 时，脚本会在机器人连接前退出。

当前用户已反馈：

```text
测试了一下，抓也能抓。
```

说明这版不仅离线数值对齐，而且真机闭环也可用。

## 12. 当前限制

### 12.1 prefix_cache engine 很大

当前：

```text
prefix_cache engine: 11.26 GB
```

这是当前最大工程问题。它会带来：

```text
1. engine 加载慢
2. 占用显存/磁盘空间大
3. 部署复制成本高
4. 后续多模型切换不方便
```

### 12.2 当前是 FP32 engine

当前主链路是 FP32：

```text
pi05_so101_prefix_cache_b1_fp32.engine
pi05_so101_denoise_step_b1_fp32.engine
```

还没有完成 FP16 split engine 的构建和完整验证。

### 12.3 prefix_cache 中间 tensor 有 outlier

虽然最终动作稳定，但 prefix cache 中间 value cache 在严格逐元素阈值下有 outlier。

这不影响当前最终动作验证和真机运行，但后续如果要进一步做：

```text
FP16
INT8
动态 batch
动态 token length
更严格中间状态复用
```

仍建议继续观察。

### 12.4 Python loop 仍有调度开销

当前 10 步 loop 保留在 Python：

```text
for step in range(10):
    denoise_step_engine(...)
```

优点是容易验证和定位问题。

缺点是：

```text
1. 每步都有 Python 调度开销
2. 每步都要调用一次 TensorRT context
3. 没有把 10 步融合成单个 engine
```

## 13. 后续提速方案

### 13.1 构建 FP16 split engines

优先级最高。

目标：

```text
prefix_cache FP16 engine
denoise_step FP16 engine
```

预期收益：

```text
1. 降低 engine size
2. 降低显存占用
3. 提高 transformer matmul 性能
4. 进一步降低 sample_actions latency
```

建议顺序：

```text
1. denoise_step FP16 engine
2. Torch vs TRT denoise_step 验证
3. split sample_actions 验证
4. camera smoke
5. prefix_cache FP16 engine
6. 完整 FP16 split runtime benchmark
```

因为 denoise_step 是 10 次循环调用，优先优化它的收益更直接。

### 13.2 prefix_cache 保留 PyTorch，只 TRT 化 denoise_step

这是一个很实用的折中方案。

原因：

```text
prefix_cache 只执行一次
denoise_step 执行 10 次
prefix_cache engine 很大
denoise_step engine 体积相对可控
```

可以做第三种 runtime：

```text
PyTorch prefix cache
  + TensorRT denoise_step
  + Python loop
```

优点：

```text
1. 不需要加载 11.26 GB prefix engine
2. 保留主要加速收益
3. 更容易部署
4. 避免 prefix_cache 中间 outlier
```

需要实测它的速度。如果 PyTorch prefix cache 只占小头，这个方案可能是部署性价比最高的版本。

### 13.3 预分配 TensorRT I/O buffer

当前 runtime 每次调用会根据输出 shape 创建 tensor。可以优化成：

```text
1. 初始化时预分配 prefix outputs
2. 初始化时预分配 denoise v_t output
3. 循环里复用 buffer
4. 避免重复 torch.empty / address setup 成本
```

预期收益不会像 FP16 那么大，但可以降低 jitter。

### 13.4 使用 CUDA Graph

denoise loop 是固定 shape、固定 engine、固定调用序列，很适合 CUDA Graph。

目标：

```text
capture 10-step denoise engine execution
replay graph
```

潜在收益：

```text
减少 Python/kernel launch overhead
降低 latency 抖动
```

需要注意 TensorRT context、输入地址稳定性、输出 buffer 复用。

### 13.5 把 10 步 denoise loop 导成单个 engine

这是更激进的方案。

目标边界：

```text
prefix_pad_masks + past_key_values + noise
  -> actions
```

内部包含 10 次 denoise_step。

优点：

```text
1. 消除 Python loop
2. TensorRT 可能跨步优化部分算子
3. runtime 更简单
```

风险：

```text
1. ONNX 图更大
2. 导出更容易失败
3. engine build 更慢
4. 定位某一步数值漂移更困难
```

建议在当前 split 版本稳定后再尝试。

### 13.6 INT8 / FP8 / 权重量化

后续可以考虑：

```text
INT8 weight-only
FP8
TensorRT ModelOpt
```

但对 VLA policy 来说，量化要非常谨慎：

```text
1. 不能只看单步 v_t
2. 必须看完整 action chunk
3. 必须做 camera smoke
4. 最终要做真机任务成功率对比
```

建议先完成 FP16，再考虑 INT8/FP8。

## 14. 建议的下一阶段路线

### 阶段 A：部署友好版

目标：

```text
PyTorch prefix cache + TensorRT denoise_step
```

原因：

```text
省掉 11.26 GB prefix engine
保留 denoise 10 次循环的主要加速
降低部署复杂度
```

要做：

```text
1. 新 runtime: pi05_trt_denoise_only.py
2. benchmark: PyTorch full vs denoise-only TRT
3. camera smoke
4. 真机验证
```

### 阶段 B：FP16 denoise_step

目标：

```text
denoise_step FP16 engine
```

要做：

```text
1. build denoise_step fp16 engine
2. Torch vs TRT denoise_step
3. full sample_actions
4. camera smoke
5. benchmark
6. 真机抓取测试
```

### 阶段 C：runtime buffer / CUDA Graph

目标：

```text
降低 10-step loop 的 runtime overhead
```

要做：

```text
1. 预分配输入输出 buffer
2. 固定 tensor address
3. 尝试 CUDA Graph capture/replay
4. benchmark latency 和 jitter
```

### 阶段 D：完整 10-step denoise engine

目标：

```text
prefix_cache + denoise_10step
```

这是高级优化，不建议在当前阶段马上做。

## 15. 结论

本次 TensorRT 优化已经完成了一条可用、可验证、可上机的 PI0.5 split TensorRT 推理链路。

核心成果：

```text
1. 成功导出 prefix_cache ONNX
2. 成功导出 denoise_step ONNX
3. 成功构建 prefix_cache TensorRT engine
4. 成功构建 denoise_step TensorRT engine
5. 完成 Torch vs ONNX / Torch vs TRT 验证
6. 完成完整 split sample_actions 验证
7. 完成真实 camera smoke
8. 完成真机上机脚本
9. 用户实测可以抓取
10. 实测 sample_actions 提速约 2.05x
```

当前版本的主要价值是：证明了 PI0.5 最重的 denoise transformer 部分可以稳定拆出来并由 TensorRT 执行，而且最终动作与 PyTorch 对齐，真机任务也能跑。

当前版本的主要不足是：prefix_cache FP32 engine 太大，且整体仍是 FP32。下一步最值得投入的是：

```text
1. denoise_step FP16 engine
2. PyTorch prefix cache + TRT denoise_step 部署友好版
3. runtime buffer 复用 / CUDA Graph
```

这三条路线预计能进一步提高速度、降低显存/磁盘占用，并让部署更轻。
