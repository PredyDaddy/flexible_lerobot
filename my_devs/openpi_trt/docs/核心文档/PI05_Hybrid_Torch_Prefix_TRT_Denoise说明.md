# PI05 Hybrid：PyTorch Prefix + TensorRT Denoise 说明

本文记录 `my_devs/openpi_trt` 当前的新主线：prefix_cache 不再使用 TensorRT engine，而是保留 PyTorch；denoise_step 继续使用 TensorRT。

## 背景

旧版本链路是：

```text
PyTorch checkpoint
  -> prefix_cache ONNX
  -> prefix_cache TensorRT engine
  -> denoise_step ONNX
  -> denoise_step TensorRT engine
  -> split TensorRT runtime
```

这个链路可以跑通，但 prefix_cache 部分比较麻烦：

- prefix 侧牵涉视觉编码器、语言 token、PaliGemma prefix forward 和 KV cache。
- prefix_cache engine 体积很大，之前约 11GB。
- 做 TensorRT 加载和验证时，仍然要完整加载 PyTorch policy 的一部分上下文。
- prefix_cache 在一次 `sample_actions` 里只执行一次，而 denoise_step 会在去噪循环里执行多次。

因此当前更实用的优化边界是：

```text
prefix_cache: PyTorch
denoise_step: TensorRT
```

这样可以避免维护巨大的 prefix engine，同时保留 denoise 循环的主要加速收益。

## 当前运行时

核心文件：

```text
my_devs/openpi_trt/runtime/simple_pi05_split.py
```

虽然类名仍然叫：

```python
SimplePI05SplitTRTRuntime
```

但实际行为已经是 hybrid：

```text
1. policy.model.embed_prefix(...) 用 PyTorch 计算 prefix embeddings。
2. PyTorch PaliGemma prefix forward 生成 prefix_pad_masks + past_key_values。
3. denoise_step TensorRT engine 接收 prefix cache、x_t、timestep。
4. Python loop 执行多步去噪。
```

兼容 wrapper：

```text
my_devs/openpi_trt/runtime/pi05_trt_split.py
```

仍然保留：

```python
patch_sample_actions_with_split_trt(policy, prefix_engine_path, denoise_engine_path)
```

其中 `prefix_engine_path` 只是兼容旧命令和旧服务端参数，不再加载。

## 构建产物

当前必须保留的产物：

```text
my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp32.onnx
my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp32.engine
my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp16_constrained.engine
```

旧的 prefix 产物不再是主线必须项：

```text
my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp32.onnx
my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp32.engine
```

这些旧文件如果还在，不影响运行；如果删除，当前 hybrid runtime 也不应该依赖它们。

## 构建命令

FP32：

```bash
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/simple_pi05_pipeline.py \
  --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
  --profile fp32 \
  --force-export \
  --force-convert \
  --report my_devs/openpi_trt/artifacts/simple_pipeline_fp32_hybrid_report.json
```

FP16 constrained：

```bash
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/simple_pi05_pipeline.py \
  --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
  --profile fp16_constrained \
  --force-export \
  --force-convert \
  --report my_devs/openpi_trt/artifacts/simple_pipeline_fp16_constrained_hybrid_report.json
```

当前 pipeline 只导出 denoise_step ONNX，并只构建 denoise_step engine。

## 验证命令

FP32 validate-only：

```bash
conda run --no-capture-output -n lerobot_flex python my_devs/openpi_trt/scripts/simple_pi05_pipeline.py \
  --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
  --profile fp32 \
  --validate-only \
  --report my_devs/openpi_trt/artifacts/simple_pipeline_fp32_hybrid_validate_report.json
```

FP16 constrained validate-only：

```bash
conda run --no-capture-output -n lerobot_flex python my_devs/openpi_trt/scripts/simple_pi05_pipeline.py \
  --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
  --profile fp16_constrained \
  --validate-only \
  --report my_devs/openpi_trt/artifacts/simple_pipeline_fp16_hybrid_validate_report.json
```

当前验证结果：

```text
fp32:
  passed_allclose: true
  torch_latency_ms: 348.42
  trt_latency_ms: 139.53
  mean_abs_diff: 0.00008531
  max_abs_diff: 0.00049794
  cosine_similarity: 0.99999998

fp16_constrained:
  passed_allclose: true
  torch_latency_ms: 343.48
  trt_latency_ms: 131.18
  mean_abs_diff: 0.00135363
  max_abs_diff: 0.00743288
  cosine_similarity: 0.99999564
```

两个报告中的 runtime 都应该显示：

```text
prefix_backend: torch
prefix_engine_path: null
```

## 上机前只加载检查

以下命令只加载 policy、hybrid runtime 和 processors，不连接机器人：

```bash
conda run --no-capture-output -n lerobot_flex python my_devs/openpi_trt/scripts/simple_pi05_run_robot.py \
  --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
  --profile fp16_constrained \
  --denoise-engine-path my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp16_constrained.engine \
  --task "Put the eraser into the small box" \
  --check-policy-load
```

预期日志：

```text
[INFO] prefix_backend=torch
[INFO] prefix_engine=None (deprecated, not loaded)
[INFO] Loading simple hybrid TensorRT runtime...
[INFO] CHECK_POLICY_LOAD=true. Exit before robot connection.
```

## 上机命令

真实上机由操作者手动执行。Agent 不要自动运行带 `--confirm-control` 的命令。

FP32：

```bash
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/simple_pi05_run_robot.py \
  --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
  --profile fp32 \
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
  --motor-write-retries 0 \
  --confirm-control
```

FP16 constrained：

```bash
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/simple_pi05_run_robot.py \
  --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
  --profile fp16_constrained \
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
  --motor-write-retries 0 \
  --confirm-control
```

## vla_engineering 服务端兼容

`my_devs/vla_engineering/vlash_iner/server/run_pi05_async_erver.py` 已经兼容 hybrid。服务端命令可以继续带 `--prefix-engine-path`，但该文件不会再被检查和加载。

推荐新写法仍然可以保留 prefix 参数用于兼容旧脚本：

```bash
PYTHONPATH=/data/cqy_workspace/flexible_lerobot/my_devs/vla_engineering \
  conda run --no-capture-output -n lerobot_flex python -u -m vlash_iner.server.run_pi05_async_server \
      --host 127.0.0.1 \
      --port 8008 \
      --endpoint /infer \
      --backend tensorrt_split \
      --policy-path /data/cqy_workspace/flexible_lerobot/outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
      --task "Put the eraser into the small box" \
      --robot-type so101_follower \
      --denoise-engine-path /data/cqy_workspace/flexible_lerobot/my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp16_constrained.engine \
      --trt-model-dtype float32
```

服务端日志应显示：

```text
Patching PI0.5 sample_actions with Torch-prefix + TensorRT-denoise backend.
TensorRT prefix engine argument ignored: ...
TensorRT denoise engine: ...
```

## 后续优化

1. 评估 hybrid 在真实异步服务端上的延迟收益，重点看 `server_infer_s`。
2. 如果 prefix PyTorch 成为瓶颈，再考虑 prefix cache 跨请求复用，而不是先回到 prefix TensorRT engine。
3. 如果要进一步压低延迟，优先优化 denoise_step engine、减少 denoise loop 开销，或评估 FP16 constrained 的上机稳定性。
4. 如果保留旧 prefix engine 文件占空间，可以在确认所有命令都不再依赖它之后删除。
