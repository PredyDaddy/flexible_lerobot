# LeRobot PI0.5 prefix_cache + denoise_step TensorRT 报告

## 结论

已在 `my_devs/openpi_trt/` 下完成 PI0.5 split TensorRT 主链路：

```text
PyTorch subgraph
  -> ONNX
  -> TensorRT engine
  -> Torch vs ONNX
  -> Torch vs TensorRT
  -> split runtime
  -> final sample_actions / camera smoke
```

本次没有改动 `src/lerobot/`。新增实现都在 `my_devs/openpi_trt/`。

## 新增/修改的代码

```text
my_devs/openpi_trt/scripts/pi05_onnx_common.py
my_devs/openpi_trt/scripts/export_lerobot_pi05_onnx.py
my_devs/openpi_trt/scripts/verify_lerobot_pi05_torch_onnx.py
my_devs/openpi_trt/scripts/verify_lerobot_pi05_torch_trt.py
my_devs/openpi_trt/scripts/inspect_pi05_cache.py
my_devs/openpi_trt/runtime/pi05_trt_split.py
my_devs/openpi_trt/scripts/verify_lerobot_pi05_split_trt_sample_actions.py
my_devs/openpi_trt/scripts/smoke_pi05_camera_split_trt.py
```

## 导出边界

### prefix_cache

输入：

```text
image_0:    [1, 3, 224, 224] float32
image_1:    [1, 3, 224, 224] float32
img_mask_0: [1] bool
img_mask_1: [1] bool
tokens:     [1, 200] int64
masks:      [1, 200] bool
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

`past_key_values` 的真实结构是 transformers `DynamicCache`，共 18 层，每层 key/value 形状：

```text
[1, 1, 712, 256] float32
```

### denoise_step

输入：

```text
prefix_pad_masks
past_key_values.0.key/value ... past_key_values.17.key/value
x_t:      [1, 50, 32] float32
timestep: [1] float32
```

输出：

```text
v_t: [1, 50, 32] float32
```

Python runtime 保留 10 步 denoise loop：

```text
x_t = noise
for step in range(num_inference_steps):
    v_t = denoise_step_engine(prefix_cache, x_t, timestep)
    x_t = x_t + dt * v_t
actions = x_t
```

## 产物

```text
my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp32.onnx
my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp32.engine
my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp32.onnx
my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp32.engine
```

大小：

```text
prefix_cache ONNX:    1.3 MB main ONNX + external data side files
prefix_cache engine:  11.26 GB
denoise_step ONNX:    1.72 GB
denoise_step engine:  1.72 GB
```

## 验证结果

### denoise_step: Torch vs ONNX

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

### denoise_step: Torch vs TensorRT

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

### prefix_cache: Torch vs ONNX/TRT

严格中间 cache tensor 验证报告：

```text
my_devs/openpi_trt/artifacts/verify_torch_onnx_prefix_cache_fp32_noopt.json
my_devs/openpi_trt/artifacts/verify_torch_trt_prefix_cache_fp32.json
```

说明：

- `prefix_pad_masks` 完全一致。
- cache tensor 整体 cosine 很高，约 `0.99999+`。
- 后层 value cache 存在少量逐元素 outlier，因此 `rtol=0.02, atol=0.1` 下 prefix cache 中间 tensor 严格 allclose 未通过。
- 用 `atol=1.0` 复验通过，报告如下：

```text
my_devs/openpi_trt/artifacts/verify_torch_onnx_prefix_cache_fp32_atol1.json
my_devs/openpi_trt/artifacts/verify_torch_trt_prefix_cache_fp32_atol1.json
```

最大 outlier：

```text
Torch vs ONNX prefix_cache max_abs_diff: 0.97361374
Torch vs TRT  prefix_cache max_abs_diff: 0.37520790
```

这个 outlier 没有在下游放大，见下面最终 `sample_actions` 验证。

### split TensorRT sample_actions: Torch vs TensorRT

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

这一步使用：

```text
prefix_cache engine + denoise_step engine + Python 10-step denoise loop
```

直接对齐完整 `sample_actions` 输出 `[1, 50, 32]`。

### camera smoke: Torch vs split TensorRT

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

camera smoke 使用真实 `/dev/video4` 和 `/dev/video6` 相机帧，只读相机，不连接机器人串口，不发送 action。

## 关键命令

导出：

```bash
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/export_lerobot_pi05_onnx.py \
  --mode prefix_cache \
  --model-dtype float32 \
  --output my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp32.onnx \
  --opset 19

conda run -n lerobot_flex python my_devs/openpi_trt/scripts/export_lerobot_pi05_onnx.py \
  --mode denoise_step \
  --model-dtype float32 \
  --output my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp32.onnx \
  --opset 19
```

构建 engine：

```bash
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/build_lerobot_pi05_engine.py \
  --precision fp32 \
  --workspace-gb 8 \
  --onnx-path my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp32.onnx \
  --engine-path my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp32.engine

conda run -n lerobot_flex python my_devs/openpi_trt/scripts/build_lerobot_pi05_engine.py \
  --precision fp32 \
  --workspace-gb 8 \
  --onnx-path my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp32.onnx \
  --engine-path my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp32.engine
```

最终 split sample_actions 验证：

```bash
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/verify_lerobot_pi05_split_trt_sample_actions.py \
  --model-dtype float32 \
  --prefix-engine-path my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp32.engine \
  --denoise-engine-path my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp32.engine \
  --rtol 2e-2 \
  --atol 1e-1 \
  --report my_devs/openpi_trt/artifacts/verify_split_trt_sample_actions_fp32.json
```

camera smoke：

```bash
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/smoke_pi05_camera_split_trt.py \
  --prefix-engine-path my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp32.engine \
  --denoise-engine-path my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp32.engine \
  --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00 \
  --top-cam /dev/video4 \
  --wrist-cam /dev/video6 \
  --top-cam-fourcc YUYV \
  --wrist-cam-fourcc MJPG \
  --task "Put the eraser into the small box" \
  --rtol 2e-2 \
  --atol 1e-1 \
  --save-frames \
  --report my_devs/openpi_trt/artifacts/camera_smoke_split_trt_fp32.json
```

## 当前限制

- 当前是 FP32 split engines。prefix cache engine 约 11.26 GB，非常大。
- prefix cache 中间 tensor 在严格 `atol=0.1` 下有后层 value cache outlier；不过单步 denoise、完整 10 步 `sample_actions`、真实 camera smoke 最终动作均已通过。
- 后续如果要追求部署速度和体积，需要继续做 FP16 engine、engine size 优化和 benchmark。
