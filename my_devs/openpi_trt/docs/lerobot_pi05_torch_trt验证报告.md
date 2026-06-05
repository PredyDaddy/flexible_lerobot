# LeRobot PI0.5 Torch vs TensorRT 验证报告

## 结论

已完成 Step 5 的离线 TensorRT 验证：

- 未连接机器人，未操控机器人。
- 已基于已有 ONNX 子图构建 TensorRT FP32 engine。
- 已基于同一 ONNX 子图构建 TensorRT FP16 engine。
- FP32 和 FP16 两个 engine 均完成 Torch vs TensorRT 数值验证，并通过 allclose。

本次验证边界仍沿用前一步已经完成 Torch vs ONNX 对齐的 PI0.5 split-graph 子图：

```text
suffix_embedding:
  noisy_actions + timestep -> suffix_embs + adarms_cond
```

对应 LeRobot PI0.5 内部：

```text
PI05Pytorch.embed_suffix(noisy_actions, timestep)
```

这个边界属于 denoise step 中 action/time suffix embedding 部分。完整 `sample_actions` 单体 TensorRT 后端仍不是本次完成范围，原因已记录在 `lerobot_pi05_torch_onnx验证报告.md`。

## 产物

脚本：

```text
my_devs/openpi_trt/scripts/build_lerobot_pi05_engine.py
my_devs/openpi_trt/scripts/verify_lerobot_pi05_torch_trt.py
my_devs/openpi_trt/runtime/trt_engine.py
```

输入 ONNX：

```text
my_devs/openpi_trt/artifacts/pi05_so101_suffix_embedding_b1.onnx
```

TensorRT engines：

```text
my_devs/openpi_trt/artifacts/pi05_so101_suffix_embedding_b1_fp32.engine
my_devs/openpi_trt/artifacts/pi05_so101_suffix_embedding_b1_fp16.engine
```

验证报告：

```text
my_devs/openpi_trt/artifacts/verify_torch_trt_suffix_embedding_fp32.json
my_devs/openpi_trt/artifacts/verify_torch_trt_suffix_embedding_fp16.json
```

文件大小：

```text
8.2M pi05_so101_suffix_embedding_b1.onnx
8.4M pi05_so101_suffix_embedding_b1_fp32.engine
8.4M pi05_so101_suffix_embedding_b1_fp16.engine
```

## TensorRT 环境

当前 `lerobot_flex` 环境中 TensorRT Python 包可用：

```text
TensorRT version: 10.13.0.35
BuilderFlag.FP16: available
OnnxParser: available
MemoryPoolType: available
```

本次没有依赖 `trtexec`，而是使用 TensorRT Python API 构建 engine，避免外部命令路径不确定的问题。

## FP32 Engine 构建

命令：

```bash
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/build_lerobot_pi05_engine.py \
  --precision fp32 \
  --onnx-path my_devs/openpi_trt/artifacts/pi05_so101_suffix_embedding_b1.onnx \
  --engine-path my_devs/openpi_trt/artifacts/pi05_so101_suffix_embedding_b1_fp32.engine
```

构建输入：

```text
noisy_actions: shape=(1, 50, 32) dtype=DataType.FLOAT
timestep:      shape=(1,) dtype=DataType.FLOAT
```

构建输出：

```text
suffix_embs: shape=(1, 50, 1024) dtype=DataType.FLOAT
adarms_cond: shape=(1, 1024) dtype=DataType.FLOAT
```

结果：

```text
Engine written: my_devs/openpi_trt/artifacts/pi05_so101_suffix_embedding_b1_fp32.engine
```

## FP16 Engine 构建

命令：

```bash
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/build_lerobot_pi05_engine.py \
  --precision fp16 \
  --onnx-path my_devs/openpi_trt/artifacts/pi05_so101_suffix_embedding_b1.onnx \
  --engine-path my_devs/openpi_trt/artifacts/pi05_so101_suffix_embedding_b1_fp16.engine
```

构建输入：

```text
noisy_actions: shape=(1, 50, 32) dtype=DataType.FLOAT
timestep:      shape=(1,) dtype=DataType.FLOAT
```

构建输出：

```text
suffix_embs: shape=(1, 50, 1024) dtype=DataType.FLOAT
adarms_cond: shape=(1, 1024) dtype=DataType.FLOAT
```

结果：

```text
Engine written: my_devs/openpi_trt/artifacts/pi05_so101_suffix_embedding_b1_fp16.engine
```

说明：FP16 engine 使用 TensorRT `BuilderFlag.FP16` 构建。该 ONNX 子图的外部 I/O 仍保持 `DataType.FLOAT`，这是 TensorRT 常见行为；是否选择 FP16 tactic 由 TensorRT builder 在内部决定。

## FP32 Torch vs TensorRT 验证

命令：

```bash
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/verify_lerobot_pi05_torch_trt.py \
  --precision fp32 \
  --engine-path my_devs/openpi_trt/artifacts/pi05_so101_suffix_embedding_b1_fp32.engine \
  --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
  --rtol 1e-4 \
  --atol 2e-3 \
  --report my_devs/openpi_trt/artifacts/verify_torch_trt_suffix_embedding_fp32.json
```

结果：

```text
allclose(rtol=0.0001, atol=0.002)=True
```

数值：

```text
suffix_embs:
  torch shape: [1, 50, 1024]
  trt shape:   [1, 50, 1024]
  mean_abs_diff: 0.00027049571508541703
  max_abs_diff:  0.0016807913780212402
  cosine_similarity: 0.9999999575057874

adarms_cond:
  torch shape: [1, 1024]
  trt shape:   [1, 1024]
  mean_abs_diff: 2.7778145295087597e-07
  max_abs_diff:  1.5050172805786133e-06
  cosine_similarity: 0.9999999999915474
```

## FP16 Torch vs TensorRT 验证

命令：

```bash
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/verify_lerobot_pi05_torch_trt.py \
  --precision fp16 \
  --engine-path my_devs/openpi_trt/artifacts/pi05_so101_suffix_embedding_b1_fp16.engine \
  --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
  --report my_devs/openpi_trt/artifacts/verify_torch_trt_suffix_embedding_fp16.json
```

结果：

```text
allclose(rtol=0.02, atol=0.02)=True
```

数值：

```text
suffix_embs:
  torch shape: [1, 50, 1024]
  trt shape:   [1, 50, 1024]
  mean_abs_diff: 0.00027049571508541703
  max_abs_diff:  0.0016807913780212402
  cosine_similarity: 0.9999999575057874

adarms_cond:
  torch shape: [1, 1024]
  trt shape:   [1, 1024]
  mean_abs_diff: 2.7778145295087597e-07
  max_abs_diff:  1.5050172805786133e-06
  cosine_similarity: 0.9999999999915474
```

## 安全边界

本次没有执行：

```text
my_devs/train/pi/so101/run_pi05_infer.py
```

也没有连接：

```text
SO101 follower robot
top camera
wrist camera
serial port
```

所有验证都是固定随机输入和 checkpoint 权重下的离线模型数值验证。

## 当前限制

本次 Step 5 是基于已完成 ONNX 导出的 split-graph 子图做 TensorRT 验证。它证明：

- TensorRT Python builder 可用。
- FP32/FP16 engine 都能成功构建。
- TensorRT runtime 可用 torch CUDA tensor 直接绑定输入输出。
- 该子图 Torch vs TRT 数值可对齐。

它尚未证明完整 PI0.5 `sample_actions(...)` 已经可以整体替换为 TensorRT。完整后端还需要继续拆：

1. `embed_prefix` 子图。
2. PaliGemma prefix cache 子图。
3. denoise step 子图。
4. Python runtime 中保留 denoise loop，逐步用 TRT engine 替换 PyTorch 子模块。

