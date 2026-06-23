# PI0.5 TensorRT Export Engine Handbook

日期：2026-06-23  
适用仓库：`/data/cqy_workspace/flexible_lerobot`  
适用模块：`my_devs/openpi_trt`  
适用模型：LeRobot PI0.5 SO101 policy  
默认环境：`lerobot_flex`

这份 Handbook 是给后续 Agent 使用的执行手册。目标是让一个没有参与前面开发过程的 Agent，也可以从训练好的 PI0.5 checkpoint 出发，重新完成：

1. 加载 PyTorch policy。
2. 导出 ONNX。
3. 构建 TensorRT engine。
4. 验证 fp32 和 fp16 constrained 两个精度。
5. 准备真实上机命令，但不自动上机。

本手册只覆盖当前已经上机验证通过的 split TensorRT 路线，不覆盖旧的 suffix 单 engine 实验路线。

## 0. 安全边界

任何 Agent 执行本 Handbook 时必须遵守：

- 所有命令使用 `conda run -n lerobot_flex ...`。
- 导出、转换、验证可以自动执行。
- 真实机器人控制不能自动执行，除非用户明确要求并亲自确认。
- `simple_pi05_run_robot.py` 只有在传入 `--confirm-control` 后才会连接机器人和发送 action。
- 自动化验证阶段只能使用 `--dry-run` 或 `--check-policy-load`，不能替用户操控机器人。

## 1. 当前最终目录结构

主目录：

```text
my_devs/openpi_trt/
```

核心 runtime：

```text
my_devs/openpi_trt/runtime/
  trt_engine.py
  protocol.py
  config.py
  metadata.py
  simple_pi05_split.py
  pi05_trt_split.py
```

说明：

- `trt_engine.py`：TensorRT engine 加载器，负责 engine I/O 和 torch tensor 之间的桥接。
- `protocol.py`：固定 prefix_cache 和 denoise_step 的 TensorRT 输入输出名称。
- `simple_pi05_split.py`：当前生产主 runtime。
- `pi05_trt_split.py`：兼容 wrapper，保留给 `my_devs/vla_engineering` 使用。

核心脚本：

```text
my_devs/openpi_trt/scripts/
  pi05_onnx_common.py
  simple_pi05_pipeline.py
  simple_pi05_validate.py
  simple_pi05_run_robot.py
  run_pi05_split_trt_infer_so101.py
```

说明：

- `pi05_onnx_common.py`：PI0.5 加载、dummy batch、ONNX wrapper 和 tensor stats 公共逻辑。
- `simple_pi05_pipeline.py`：导出 ONNX、构建 engine、验证 Torch vs TensorRT 的统一入口。
- `simple_pi05_validate.py`：验证入口，内部复用 `simple_pi05_pipeline.py`。
- `simple_pi05_run_robot.py`：真实 SO101 上机入口。
- `run_pi05_split_trt_infer_so101.py`：旧命令兼容入口，内部转发到 `simple_pi05_run_robot.py`。

## 2. 输入 checkpoint

当前默认训练权重：

```text
outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model
```

执行前确认该目录至少包含：

```text
config.json
model.safetensors
policy_preprocessor.json
policy_preprocessor_step_2_normalizer_processor.safetensors
policy_postprocessor.json
policy_postprocessor_step_0_unnormalizer_processor.safetensors
train_config.json
```

检查命令：

```bash
find outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
  -maxdepth 1 -type f -printf '%f\n' | sort
```

注意：`checkpoints/last/pretrained_model` 可能是软链接或内部解析到具体 step。日志中常见实际路径：

```text
outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/020000/pretrained_model
```

这是正常现象。

## 3. 为什么不是导出一个完整 monolithic ONNX

PI0.5 的 `sample_actions(...)` 包含：

1. 图像和语言 prefix 编码。
2. prefix attention cache。
3. action denoise loop。

直接导出一个完整 `sample_actions` graph 很容易遇到：

- 控制流和 denoise loop 难以稳定导出。
- ONNX graph 过大且难以调试。
- TensorRT 对部分 LayerNorm / Softmax / Reduce / Elementwise 子图的 fp16 数值漂移明显。

当前稳定路线是 split runtime：

```text
prefix_cache engine + denoise_step engine + Python denoise loop
```

也就是只导出两个 ONNX：

```text
pi05_so101_prefix_cache_b1_fp32.onnx
pi05_so101_denoise_step_b1_fp32.onnx
```

然后构建三个 engine：

```text
pi05_so101_prefix_cache_b1_fp32.engine
pi05_so101_denoise_step_b1_fp32.engine
pi05_so101_denoise_step_b1_fp16_constrained.engine
```

fp16 constrained 的含义是：允许 TensorRT 使用 FP16 tactic，但把 LayerNorm、Softmax、Reduce、Sqrt、Pow、Norm 等敏感层强制保持 FP32，避免纯 fp16 denoise step 动作漂移过大。

## 4. TensorRT I/O 协议

协议文件：

```text
my_devs/openpi_trt/runtime/protocol.py
```

prefix_cache engine 输入：

```text
image_0
image_1
img_mask_0
img_mask_1
tokens
masks
```

prefix_cache engine 输出：

```text
prefix_pad_masks
past_key_values.0.key
past_key_values.0.value
...
past_key_values.17.key
past_key_values.17.value
```

denoise_step engine 输入：

```text
prefix_pad_masks
past_key_values.0.key
past_key_values.0.value
...
past_key_values.17.key
past_key_values.17.value
x_t
timestep
```

denoise_step engine 输出：

```text
v_t
```

runtime 初始化时会检查 engine 的 I/O 名称。如果不匹配，会直接报错，不能跳过。

## 5. 一键 pipeline 的行为

主脚本：

```text
my_devs/openpi_trt/scripts/simple_pi05_pipeline.py
```

它做三件事：

1. `export_onnx(...)`
   - 加载 PI0.5 policy。
   - 构造确定性 dummy batch。
   - 导出 prefix_cache ONNX。
   - 导出 denoise_step ONNX。

2. `convert_engines(...)`
   - prefix_cache 始终构建 fp32 engine。
   - profile 为 `fp32` 时，构建 denoise_step fp32 engine。
   - profile 为 `fp16_constrained` 时，构建 denoise_step fp16 constrained engine。
   - TensorRT parser 使用 ONNX 的绝对路径读取文件，并临时切到 ONNX 所在目录，以便外部数据文件和相对引用可以正确解析。

3. `validate_inference(...)`
   - 用固定 noise 跑 PyTorch `policy.model.sample_actions(...)`。
   - 用 TensorRT split runtime 跑同一输入。
   - 对有效 action 维度做 `np.allclose(rtol=2e-2, atol=1e-1)`。
   - 输出 mean/max diff、cosine similarity、latency 和 JSON report。

## 6. 从空 artifacts 重新生成

如果用户要求完全重建，可以清空：

```bash
rm -rf my_devs/openpi_trt/artifacts/*
mkdir -p my_devs/openpi_trt/artifacts
```

清空后按照下面顺序执行。

### 6.1 生成 fp32 ONNX + fp32 engine + fp32 验证

```bash
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/simple_pi05_pipeline.py \
  --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
  --profile fp32 \
  --force-export \
  --force-convert \
  --report my_devs/openpi_trt/artifacts/simple_pipeline_fp32_report.json
```

成功后应生成：

```text
my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp32.onnx
my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp32.onnx
my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp32.engine
my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp32.engine
my_devs/openpi_trt/artifacts/simple_pipeline_fp32_report.json
```

期望日志包含：

```text
[INFER] passed_allclose=True
```

### 6.2 基于同一 ONNX 生成 fp16 constrained engine + 验证

```bash
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/simple_pi05_pipeline.py \
  --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
  --profile fp16_constrained \
  --report my_devs/openpi_trt/artifacts/simple_pipeline_fp16_constrained_report.json
```

该命令会复用已存在的两个 ONNX 和 prefix fp32 engine，并构建：

```text
my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp16_constrained.engine
```

成功后应生成：

```text
my_devs/openpi_trt/artifacts/simple_pipeline_fp16_constrained_report.json
```

期望日志包含：

```text
[INFER] passed_allclose=True
```

## 7. 只验证已有 artifacts

fp32：

```bash
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/simple_pi05_validate.py \
  --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
  --profile fp32 \
  --validate-only \
  --report my_devs/openpi_trt/artifacts/simple_pipeline_fp32_validate_only_report.json
```

fp16 constrained：

```bash
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/simple_pi05_validate.py \
  --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
  --profile fp16_constrained \
  --validate-only \
  --report my_devs/openpi_trt/artifacts/simple_pipeline_fp16_constrained_validate_only_report.json
```

## 8. 如何判断结果合格

JSON report 位置：

```text
my_devs/openpi_trt/artifacts/simple_pipeline_fp32_report.json
my_devs/openpi_trt/artifacts/simple_pipeline_fp16_constrained_report.json
```

检查字段：

```json
{
  "inference": {
    "passed_allclose": true,
    "action_chunk_stats": {
      "mean_abs_diff": "...",
      "max_abs_diff": "...",
      "cosine_similarity": "..."
    }
  }
}
```

合格标准：

- `passed_allclose` 必须为 `true`。
- `cosine_similarity` 应接近 1。
- `max_abs_diff` 必须在 `atol=1e-1` 容忍范围内。

当前工程默认阈值：

```text
rtol = 2e-2
atol = 1e-1
```

## 9. 常见问题

### 9.1 找不到 TensorRT

现象：

```text
ModuleNotFoundError: No module named 'tensorrt'
```

处理：

- 确认使用 `conda run -n lerobot_flex ...`。
- 确认当前机器安装了 TensorRT Python binding。
- 不要换到普通系统 Python。

### 9.2 CUDA 不可用

现象：

```text
RuntimeError: CUDA is required for simple PI0.5 TensorRT pipeline.
```

处理：

- 检查 `nvidia-smi`。
- 检查 conda 环境里的 PyTorch CUDA 是否可用。

### 9.3 TensorRT ONNX parse failed

处理顺序：

1. 确认 ONNX 是用当前 `simple_pi05_pipeline.py` 重新导出的。
2. 确认没有混用旧 suffix ONNX。
3. 删除 artifacts 后重新执行 fp32 pipeline。
4. 如果仍失败，保留完整 TensorRT parser error。

### 9.4 fp16 数值不过

不要使用纯 fp16 denoise engine。必须使用：

```text
pi05_so101_denoise_step_b1_fp16_constrained.engine
```

该 engine 会对敏感层设置 FP32 precision constraints。

### 9.5 上机失败但离线验证通过

这通常不是 ONNX/engine 数值问题，而是 runtime boundary 问题：

- 相机路径不对。
- 串口路径不对。
- 标定目录不对。
- 电机或 bus 连接失败。
- `--task` 文本与训练任务不匹配。

先用：

```bash
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/simple_pi05_run_robot.py \
  ... \
  --check-policy-load
```

确认 policy、TRT runtime、processor 可以加载，再由用户亲自执行真实上机命令。

## 10. 真实上机命令

以下命令只交给用户执行。Agent 不要自动运行带 `--confirm-control` 的命令。

### 10.1 fp32 上机

```bash
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/simple_pi05_run_robot.py \
  --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
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
  --motor-write-retries 0 \
  --confirm-control
```

### 10.2 fp16 constrained 上机

```bash
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/simple_pi05_run_robot.py \
  --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
  --profile fp16_constrained \
  --prefix-engine-path my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp32.engine \
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

## 11. Agent 执行 Checklist

执行人需要逐项确认：

```text
[ ] 当前目录是 /data/cqy_workspace/flexible_lerobot
[ ] 使用 lerobot_flex conda 环境
[ ] checkpoint 文件齐全
[ ] artifacts 已按需求清空或确认复用
[ ] fp32 pipeline 成功
[ ] fp32 report 中 passed_allclose=true
[ ] fp16 constrained pipeline 成功
[ ] fp16 report 中 passed_allclose=true
[ ] simple_pi05_run_robot.py --check-policy-load 成功
[ ] 未自动执行真实机器人 --confirm-control 命令
[ ] 已把 fp32 和 fp16 constrained 上机命令交给用户
```
