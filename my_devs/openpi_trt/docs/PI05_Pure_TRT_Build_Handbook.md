# PI0.5 Pure TensorRT Build Handbook

日期：2026-06-24  
适用仓库：`/data/cqy_workspace/flexible_lerobot`  
适用模块：`my_devs/openpi_trt`  
适用模型：LeRobot PI0.5 SO101 policy  
默认环境：`lerobot_flex`  
主要目标：从训练好的 PyTorch checkpoint 构建 pure TensorRT runtime，并完成 FP32 / FP16 constrained 验证。

这份文档是给后续 Agent 执行的 Handbook。执行者不需要知道前面的开发过程，只要按本文走，就可以从训练好的 PI0.5 checkpoint 重新完成：

1. 加载 Torch policy。
2. 导出 pure TRT 所需的 ONNX。
3. 构建 prefix_cache 和 denoise_step 的 TensorRT engine。
4. 准备 pure TRT runtime assets。
5. 验证 FP32 和 FP16 constrained 精度。
6. 做 pure TRT 加载检查。
7. 给用户上机命令，但不要替用户上机。

## 0. 安全边界

任何 Agent 执行本文时必须遵守：

- 所有命令默认在仓库根目录执行：`/data/cqy_workspace/flexible_lerobot`。
- 所有 Python 命令必须使用 `conda run --no-capture-output -n lerobot_flex ...`，除非只是纯 shell 文件查看命令。
- 可以自动执行：检查文件、导出 ONNX、构建 TensorRT engine、离线精度验证、`--check-policy-load`。
- 不能自动执行真实机器人控制命令。
- `simple_pi05_run_robot.py` 只有带 `--confirm-control` 才会连接机器人和发送 action。这个参数只能给用户命令，不能由 Agent 自己执行。
- 如果用户要求清理 artifacts，删除前必须确认当前需要保留的 engine、ONNX、report 是否已经备份或可重建。

## 1. 当前 pure_trt 结论

当前 pure TRT runtime 已经支持两种精度：

```text
FP32:
  prefix_cache FP32 TensorRT engine
  denoise_step FP32 TensorRT engine

FP16 constrained:
  prefix_cache FP16 constrained TensorRT engine
  denoise_step FP16 constrained TensorRT engine
```

推荐真实上机优先使用：

```text
FP16 constrained pure_trt
```

原因：

- 不加载 PyTorch `model.safetensors`。
- prefix 和 denoise 都由 TensorRT 执行。
- 相比 pure TRT FP32，显存压力明显更低。
- 已通过 Torch vs pure TRT 精度验证。

当前关键产物：

```text
my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp32.onnx
my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp32.onnx
my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp32.engine
my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp16_constrained.engine
my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp32.engine
my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp16_constrained.engine
my_devs/openpi_trt/artifacts/pi05_runtime_assets/
```

当前产物体积参考：

```text
prefix_cache ONNX:                  1.3M
denoise_step ONNX:                  1.7G
prefix_cache FP32 engine:           11G
prefix_cache FP16 constrained:      5.3G
denoise_step FP32 engine:           1.7G
denoise_step FP16 constrained:      825M
runtime assets:                     28K
```

注意：`prefix_cache.onnx` 很小，但 `prefix_cache.engine` 很大。这是正常现象。ONNX 里大量权重可能通过外部数据文件或 TensorRT 编译期常量折叠进入 engine；不要用 ONNX 文件大小推断 engine 显存。

## 2. 背景

训练好的 PI0.5 checkpoint 是 PyTorch 权重，默认推理路径会加载完整 `model.safetensors`。为了降低真实机器人推理延迟，并尝试减少部署时对 PyTorch 模型权重的依赖，本模块把 PI0.5 的 `sample_actions(...)` 拆成两个 TensorRT 可执行边界：

```text
prefix_cache:
  image + language + mask
  -> prefix_pad_masks + past_key_values

denoise_step:
  prefix_pad_masks + past_key_values + x_t + timestep
  -> v_t
```

pure TRT runtime 的执行逻辑是：

```text
preprocessor/tokenizer
  -> policy batch
  -> TensorRT prefix_cache
  -> Python denoise loop
       每一步调用 TensorRT denoise_step
  -> postprocessor
  -> robot action
```

这里的 Python denoise loop 只负责编排时间步和更新 `x_t`，不加载 PyTorch 模型权重。

## 3. 关键代码入口

### 3.1 导出、构建、验证入口

```text
my_devs/openpi_trt/scripts/simple_pi05_pipeline.py
```

职责：

- 加载 Torch policy。
- 构造确定性验证 batch。
- 导出 prefix_cache ONNX。
- 导出 denoise_step ONNX。
- 构建 TensorRT engine。
- 跑 Torch vs TensorRT 精度对比。
- 写 JSON report。

### 3.2 ONNX wrapper 和 Torch baseline

```text
my_devs/openpi_trt/scripts/pi05_onnx_common.py
```

关键类和函数：

```text
PI05PrefixCacheONNXWrapper
PI05DenoiseStepONNXWrapper
load_policy
make_policy_batch
make_prefix_cache_inputs
make_denoise_step_inputs
make_export_inputs
tensor_stats
```

### 3.3 pure TensorRT runtime

```text
my_devs/openpi_trt/runtime/pure_pi05_trt.py
```

关键类：

```text
PurePI05TRTProfile
PurePI05TRTRuntime
PurePI05TRTPolicyAdapter
```

说明：

- `PurePI05TRTRuntime` 加载 prefix 和 denoise 两个 engine。
- `PurePI05TRTPolicyAdapter` 模拟 LeRobot policy 的最小接口。
- pure TRT 上机时不加载 `model.safetensors`。
- pure TRT 仍然需要 `config.json`、preprocessor 和 postprocessor 文件。

### 3.4 TensorRT engine loader

```text
my_devs/openpi_trt/runtime/trt_engine.py
```

职责：

- 反序列化 TensorRT engine。
- 创建 execution context。
- 读取 engine I/O 名称、dtype。
- 使用 torch CUDA tensor 作为输入输出 buffer。

### 3.5 TensorRT I/O 协议

```text
my_devs/openpi_trt/runtime/protocol.py
```

这个文件定义 prefix 和 denoise 的固定 tensor 名称。runtime 会严格校验 engine 的 I/O 名称，名称不一致会直接失败。

## 4. 输入 checkpoint

默认 checkpoint：

```text
outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model
```

实际日志中可能解析到：

```text
outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/020000/pretrained_model
```

这是正常的，因为 `last` 可能指向具体 step。

检查 checkpoint 文件：

```bash
find outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
    -maxdepth 1 -type f -printf '%f\n' | sort
```

至少应包含：

```text
config.json
model.safetensors
policy_preprocessor.json
policy_preprocessor_step_2_normalizer_processor.safetensors
policy_postprocessor.json
policy_postprocessor_step_0_unnormalizer_processor.safetensors
train_config.json
```

## 5. pure_trt runtime assets

pure TRT 真实运行不能依赖 `model.safetensors`，但还需要配置和处理器文件。

runtime assets 目录：

```text
my_devs/openpi_trt/artifacts/pi05_runtime_assets
```

内容必须只有小文件：

```text
config.json
policy_preprocessor.json
policy_preprocessor_step_2_normalizer_processor.safetensors
policy_postprocessor.json
policy_postprocessor_step_0_unnormalizer_processor.safetensors
```

不能包含：

```text
model.safetensors
```

如果需要从 checkpoint 重新生成 runtime assets，执行：

```bash
mkdir -p my_devs/openpi_trt/artifacts/pi05_runtime_assets

cp \
  outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model/config.json \
  outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model/policy_preprocessor.json \
  outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model/policy_preprocessor_step_2_normalizer_processor.safetensors \
  outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model/policy_postprocessor.json \
  outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model/policy_postprocessor_step_0_unnormalizer_processor.safetensors \
  my_devs/openpi_trt/artifacts/pi05_runtime_assets/
```

检查：

```bash
find my_devs/openpi_trt/artifacts/pi05_runtime_assets -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
```

如果发现 `model.safetensors`，必须删除：

```bash
rm -f my_devs/openpi_trt/artifacts/pi05_runtime_assets/model.safetensors
```

## 6. 环境检查

在开始构建前执行：

```bash
pwd
conda run --no-capture-output -n lerobot_flex python - <<'PY'
import torch
print("cuda_available=", torch.cuda.is_available())
print("cuda_device=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
try:
    import tensorrt as trt
    print("tensorrt=", trt.__version__)
except Exception as exc:
    print("tensorrt_import_failed=", repr(exc))
PY
```

期望：

```text
cuda_available= True
tensorrt= ...
```

还要确认本地 tokenizer 目录存在：

```bash
test -d google/paligemma-3b-pt-224 && echo OK
```

如果这个目录不存在，PI0.5 离线 tokenizer 会失败。

## 7. 从 Torch 到 ONNX

### 7.1 一次性导出 pure TRT 所需 ONNX

使用 `simple_pi05_pipeline.py` 导出 ONNX。为了导出 prefix_cache ONNX，必须带：

```text
--export-prefix-onnx
```

命令：

```bash
conda run --no-capture-output -n lerobot_flex python my_devs/openpi_trt/scripts/simple_pi05_pipeline.py \
    --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
    --profile fp32 \
    --runtime-backend pure_trt \
    --export-prefix-onnx \
    --force-export \
    --export-only \
    --report my_devs/openpi_trt/artifacts/pure_trt_export_onnx_report.json
```

预期生成：

```text
my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp32.onnx
my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp32.onnx
my_devs/openpi_trt/artifacts/pure_trt_export_onnx_report.json
```

注意：

- 文件名里带 `fp32.onnx`，表示导出基线是 Torch FP32。
- FP16 TensorRT engine 也是从这个 ONNX 构建出来的。
- 不需要单独导出一个 `fp16.onnx`。

### 7.2 ONNX 导出背后的 Torch 加载

导出 ONNX 时必须加载 PyTorch checkpoint：

```text
model.safetensors -> Torch policy -> ONNX
```

这是构建阶段必须发生的事情，不代表 pure TRT 真实运行时还会加载 Torch 权重。

导出时会调用：

```python
load_policy(policy_path, device="cuda", model_dtype="float32")
```

也就是说，构建和验证阶段会使用 Torch FP32 baseline。

## 8. 从 ONNX 到 TensorRT

### 8.1 构建 pure TRT FP32 engine

FP32 pure TRT 需要两个 engine：

```text
pi05_so101_prefix_cache_b1_fp32.engine
pi05_so101_denoise_step_b1_fp32.engine
```

命令：

```bash
conda run --no-capture-output -n lerobot_flex python my_devs/openpi_trt/scripts/simple_pi05_pipeline.py \
    --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
    --profile fp32 \
    --runtime-backend pure_trt \
    --build-prefix-engine \
    --prefix-precision fp32 \
    --convert-only \
    --force-convert \
    --workspace-gb 8 \
    --report my_devs/openpi_trt/artifacts/pure_trt_fp32_build_report.json
```

预期生成：

```text
my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp32.engine
my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp32.engine
my_devs/openpi_trt/artifacts/pure_trt_fp32_build_report.json
```

### 8.2 构建 pure TRT FP16 constrained engine

FP16 constrained pure TRT 需要两个 engine：

```text
pi05_so101_prefix_cache_b1_fp16_constrained.engine
pi05_so101_denoise_step_b1_fp16_constrained.engine
```

命令：

```bash
conda run --no-capture-output -n lerobot_flex python my_devs/openpi_trt/scripts/simple_pi05_pipeline.py \
    --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
    --profile fp16_constrained \
    --runtime-backend pure_trt \
    --build-prefix-engine \
    --prefix-precision fp16_constrained \
    --convert-only \
    --force-convert \
    --workspace-gb 8 \
    --report my_devs/openpi_trt/artifacts/pure_trt_fp16_constrained_build_report.json
```

预期生成：

```text
my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp16_constrained.engine
my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp16_constrained.engine
my_devs/openpi_trt/artifacts/pure_trt_fp16_constrained_build_report.json
```

### 8.3 FP16 constrained 的含义

`simple_pi05_pipeline.py` 构建 FP16 constrained engine 时会：

```python
config.set_flag(trt.BuilderFlag.FP16)
config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
```

并对敏感层保持 FP32：

```text
layernorm
layer_norm
input_layernorm
post_attention_layernorm
softmax
reduce
sqrt
pow
norm
```

敏感 TensorRT layer 类型包括：

```text
REDUCE
SOFTMAX
UNARY
ELEMENTWISE
```

这样可以保留大部分 FP16 性能和显存收益，同时降低动作输出漂移。

## 9. 精度验证

验证逻辑：

```text
Torch policy.model.sample_actions(...)
vs
pure TensorRT prefix_cache + denoise_step
```

比较对象：

```text
action chunk 有效 action 维度
```

判定阈值：

```python
np.allclose(torch_np, trt_np, rtol=2e-2, atol=1e-1)
```

### 9.1 验证 FP32 pure TRT

命令：

```bash
conda run --no-capture-output -n lerobot_flex python my_devs/openpi_trt/scripts/simple_pi05_pipeline.py \
    --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
    --profile fp32 \
    --runtime-backend pure_trt \
    --prefix-engine-path my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp32.engine \
    --validate-only \
    --report my_devs/openpi_trt/artifacts/simple_pipeline_fp32_pure_trt_validate_report.json
```

当前已验证结果：

```text
passed_allclose: true
torch_latency_ms: 354.36
trt_latency_ms: 116.17
mean_abs_diff: 0.00019093
max_abs_diff: 0.00069308
cosine_similarity: 0.99999991
```

### 9.2 验证 FP16 constrained pure TRT

命令：

```bash
conda run --no-capture-output -n lerobot_flex python my_devs/openpi_trt/scripts/simple_pi05_pipeline.py \
    --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
    --profile fp16_constrained \
    --runtime-backend pure_trt \
    --prefix-engine-path my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp16_constrained.engine \
    --validate-only \
    --report my_devs/openpi_trt/artifacts/simple_pipeline_fp16_pure_trt_prefix_fp16_validate_report.json
```

当前已验证结果：

```text
passed_allclose: true
torch_latency_ms: 357.89
trt_latency_ms: 75.81
mean_abs_diff: 0.00349374
max_abs_diff: 0.01450744
cosine_similarity: 0.99997102
```

### 9.3 读取 report 的辅助命令

```bash
python - <<'PY'
import json
from pathlib import Path

for path in [
    Path("my_devs/openpi_trt/artifacts/simple_pipeline_fp32_pure_trt_validate_report.json"),
    Path("my_devs/openpi_trt/artifacts/simple_pipeline_fp16_pure_trt_prefix_fp16_validate_report.json"),
]:
    data = json.loads(path.read_text())
    inf = data["inference"]
    stats = inf["action_chunk_stats"]
    print(path.name)
    print("  passed_allclose:", inf["passed_allclose"])
    print("  torch_latency_ms:", round(inf["torch_latency_ms"], 2))
    print("  trt_latency_ms:", round(inf["trt_latency_ms"], 2))
    print("  mean_abs_diff:", stats["mean_abs_diff"])
    print("  max_abs_diff:", stats["max_abs_diff"])
    print("  cosine_similarity:", stats["cosine_similarity"])
PY
```

## 10. pure TRT 加载检查

在给用户上机命令前，Agent 必须先跑 `--check-policy-load`。这个命令会加载 pure TensorRT runtime 和 processors，但不会连接机器人。

FP16 constrained 检查命令：

```bash
conda run --no-capture-output -n lerobot_flex python my_devs/openpi_trt/scripts/simple_pi05_run_robot.py \
    --runtime-backend pure_trt \
    --runtime-assets-dir my_devs/openpi_trt/artifacts/pi05_runtime_assets \
    --profile fp16_constrained \
    --prefix-engine-path my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp16_constrained.engine \
    --denoise-engine-path my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp16_constrained.engine \
    --task "Put the eraser into the small box" \
    --check-policy-load
```

期望日志：

```text
[INFO] runtime_backend=pure_trt
[INFO] runtime_assets_dir=...
[INFO] Loading pure TensorRT runtime without PyTorch model weights...
[INFO] Pure TensorRT runtime loaded ...
[INFO] Loading processors...
[INFO] CHECK_POLICY_LOAD=true. Exit before robot connection.
```

不应该出现：

```text
Loading model from: .../model.safetensors
```

如果出现 `Loading model from`，说明走错了 hybrid 或 Torch policy 路径。

## 11. 上机命令

下面命令只给用户执行。Agent 不要自己执行。

FP16 constrained pure TRT 上机命令：

```bash
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/simple_pi05_run_robot.py \
    --runtime-backend pure_trt \
    --runtime-assets-dir my_devs/openpi_trt/artifacts/pi05_runtime_assets \
    --profile fp16_constrained \
    --prefix-engine-path my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp16_constrained.engine \
    --denoise-engine-path my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp16_constrained.engine \
    --robot-id hfy_follower \
    --robot-type so101_follower \
    --calib-dir /home/cqy/.cache/huggingface/lerobot/calibration/robots/so_follower \
    --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00 \
    --top-cam /dev/video4 \
    --wrist-cam /dev/video6 \
    --top-cam-fourcc YUYV \
    --wrist-cam-fourcc MJPG \
    --img-width 640 \
    --img-height 480 \
    --fps 30 \
    --task "Put the eraser into the small box" \
    --run-time-s 120 \
    --log-interval 1 \
    --motor-io-retries 10 \
    --confirm-control
```

说明：

- 这条命令不传 `--policy-path`。
- 这条命令不会加载 `model.safetensors`。
- `--runtime-assets-dir` 只提供 config、preprocessor、postprocessor。
- `--prefix-engine-path` 和 `--denoise-engine-path` 都是 FP16 constrained engine。
- `--motor-io-retries 10` 用于缓解 SO101 电机串口偶发无 status packet 的问题。

## 12. 什么时候使用 FP32

FP32 pure TRT 上机不是默认推荐，但可以作为排查数值问题时的对照。

FP32 pure TRT 上机命令：

```bash
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/simple_pi05_run_robot.py \
    --runtime-backend pure_trt \
    --runtime-assets-dir my_devs/openpi_trt/artifacts/pi05_runtime_assets \
    --profile fp32 \
    --prefix-engine-path my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp32.engine \
    --denoise-engine-path my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp32.engine \
    --robot-id hfy_follower \
    --robot-type so101_follower \
    --calib-dir /home/cqy/.cache/huggingface/lerobot/calibration/robots/so_follower \
    --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00 \
    --top-cam /dev/video4 \
    --wrist-cam /dev/video6 \
    --top-cam-fourcc YUYV \
    --wrist-cam-fourcc MJPG \
    --img-width 640 \
    --img-height 480 \
    --fps 30 \
    --task "Put the eraser into the small box" \
    --run-time-s 120 \
    --log-interval 1 \
    --motor-io-retries 10 \
    --confirm-control
```

注意：FP32 prefix engine 当前约 11G，显存会明显高于 FP16 constrained。

## 13. 显存判断

如果用户问为什么 pure TRT 比 Torch 还占显存，要这样解释：

- Torch 原始推理可能使用 BF16 或半精度加载权重。
- pure TRT 如果使用 FP32 prefix engine，`pi05_so101_prefix_cache_b1_fp32.engine` 本身就约 11G。
- TensorRT 加载 engine 后还会有 execution context、activation buffer、CUDA context、workspace 等额外开销。
- 因此 pure TRT FP32 看到 12G 左右并不奇怪。
- 如果目标是降低显存，应该使用 FP16 constrained prefix engine。

推荐检查：

```bash
ls -lh \
  my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp32.engine \
  my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp16_constrained.engine \
  my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp16_constrained.engine
```

## 14. 常见失败与处理

### 14.1 找不到 tokenizer

错误特征：

```text
Missing local tokenizer directory for offline PI0.5 inference.
Expected: .../google/paligemma-3b-pt-224
```

处理：

- 确认仓库根目录下有 `google/paligemma-3b-pt-224`。
- 不要随意改 tokenizer 名称，因为 preprocessor 和 prompt 构造依赖它。

### 14.2 pure TRT 仍然加载 model.safetensors

错误特征：

```text
Loading model from: .../model.safetensors
```

处理：

- 确认命令带了 `--runtime-backend pure_trt`。
- 确认没有用 hybrid 命令。
- 确认传的是 `--runtime-assets-dir`，不是 `--policy-path`。
- 确认 runtime assets 目录没有 `model.safetensors`。

### 14.3 TensorRT I/O 名称不匹配

错误特征：

```text
TensorRT I/O mismatch
```

处理：

- 检查是否使用了旧 engine。
- 重新按本文导出 ONNX 和构建 engine。
- 不要手动改 ONNX output names。
- 对照 `my_devs/openpi_trt/runtime/protocol.py`。

### 14.4 FP16 精度不通过

处理顺序：

1. 先验证 FP32 pure TRT。
2. 再验证 FP16 constrained pure TRT。
3. 如果 FP32 通过、FP16 不通过，重点检查 `_configure_fp16_precision_constraints(...)` 是否被改过。
4. 如果 FP32 都不通过，重点检查 ONNX 是否过期、checkpoint 是否换了、runtime protocol 是否被改过。

### 14.5 上机时电机无 status packet

错误特征：

```text
There is no status packet!
```

处理：

- 这通常是 SO101 电机串口通信问题，不是 TensorRT 数值问题。
- 上机命令保留 `--motor-io-retries 10`。
- 如果仍然失败，可以让用户尝试 `--motor-io-retries 20`。
- 检查电源、USB、舵机线、具体报错 id 对应的电机。

## 15. Agent 执行清单

后续 Agent 接手 pure_trt 工作时，按这个清单执行：

1. 确认当前目录是仓库根目录。
2. 确认 `lerobot_flex` 能 import torch 和 tensorrt。
3. 确认 checkpoint 文件完整。
4. 确认 runtime assets 存在且没有 `model.safetensors`。
5. 如需重建，先导出 ONNX。
6. 构建 FP32 prefix + denoise engine。
7. 验证 FP32 pure TRT。
8. 构建 FP16 constrained prefix + denoise engine。
9. 验证 FP16 constrained pure TRT。
10. 跑 `--check-policy-load`，确认不加载 PyTorch 权重、不连接机器人。
11. 最后只把上机命令交给用户。

## 16. 最短复现流程

如果 artifacts 已经清空，并且用户要求从零复现 pure TRT，按下面顺序执行。

准备 runtime assets：

```bash
mkdir -p my_devs/openpi_trt/artifacts/pi05_runtime_assets

cp \
  outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model/config.json \
  outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model/policy_preprocessor.json \
  outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model/policy_preprocessor_step_2_normalizer_processor.safetensors \
  outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model/policy_postprocessor.json \
  outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model/policy_postprocessor_step_0_unnormalizer_processor.safetensors \
  my_devs/openpi_trt/artifacts/pi05_runtime_assets/
```

导出 ONNX：

```bash
conda run --no-capture-output -n lerobot_flex python my_devs/openpi_trt/scripts/simple_pi05_pipeline.py \
    --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
    --profile fp32 \
    --runtime-backend pure_trt \
    --export-prefix-onnx \
    --force-export \
    --export-only \
    --report my_devs/openpi_trt/artifacts/pure_trt_export_onnx_report.json
```

构建 FP16 constrained engine：

```bash
conda run --no-capture-output -n lerobot_flex python my_devs/openpi_trt/scripts/simple_pi05_pipeline.py \
    --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
    --profile fp16_constrained \
    --runtime-backend pure_trt \
    --build-prefix-engine \
    --prefix-precision fp16_constrained \
    --convert-only \
    --force-convert \
    --workspace-gb 8 \
    --report my_devs/openpi_trt/artifacts/pure_trt_fp16_constrained_build_report.json
```

验证 FP16 constrained：

```bash
conda run --no-capture-output -n lerobot_flex python my_devs/openpi_trt/scripts/simple_pi05_pipeline.py \
    --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
    --profile fp16_constrained \
    --runtime-backend pure_trt \
    --prefix-engine-path my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp16_constrained.engine \
    --validate-only \
    --report my_devs/openpi_trt/artifacts/simple_pipeline_fp16_pure_trt_prefix_fp16_validate_report.json
```

加载检查：

```bash
conda run --no-capture-output -n lerobot_flex python my_devs/openpi_trt/scripts/simple_pi05_run_robot.py \
    --runtime-backend pure_trt \
    --runtime-assets-dir my_devs/openpi_trt/artifacts/pi05_runtime_assets \
    --profile fp16_constrained \
    --prefix-engine-path my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp16_constrained.engine \
    --denoise-engine-path my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp16_constrained.engine \
    --task "Put the eraser into the small box" \
    --check-policy-load
```

只有以上步骤通过后，才给用户 FP16 上机命令。
