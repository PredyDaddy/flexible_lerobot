# openpi_trt 目录规整与 fp32/fp16 验证说明

日期：2026-06-22  
默认 checkpoint：

```text
outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model
```

说明：`checkpoints/last` 当前解析到实际 step：

```text
outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/020000/pretrained_model
```

## 1. 新目录职责

本轮把 `runtime/` 和 `scripts/` 里混在一起的职责拆开：

```text
my_devs/openpi_trt/runtime/
  只放 TensorRT 运行时、I/O 协议、runtime config、metadata。

my_devs/openpi_trt/exporters/
  放 ONNX 导出和 TensorRT engine 构建逻辑。

my_devs/openpi_trt/validation/
  放 Torch / TensorRT 精度验证逻辑和 profile 定义。

my_devs/openpi_trt/scripts/
  只保留命令行入口。旧脚本仍保留兼容，新脚本作为推荐入口。
```

推荐新入口：

```text
scripts/export_pi05_split_artifacts.py
scripts/validate_pi05_split_profiles.py
scripts/check_split_trt_artifact.py
```

旧入口仍然可用：

```text
scripts/export_lerobot_pi05_onnx.py
scripts/build_lerobot_pi05_engine.py
scripts/verify_lerobot_pi05_torch_onnx.py
scripts/verify_lerobot_pi05_torch_trt.py
scripts/verify_lerobot_pi05_split_trt_sample_actions.py
scripts/smoke_pi05_camera_split_trt.py
scripts/run_pi05_split_trt_infer_so101.py
```

已清理的早期实验入口：

```text
runtime/pi05_trt_suffix.py
scripts/run_pi05_trt_infer_so101.py
scripts/smoke_pi05_camera_torch_trt.py
scripts/benchmark_pi05_camera_torch_trt.py
```

这些入口只服务于早期 `suffix_embedding` 局部替换路线。当前主线已经是
`prefix_cache + denoise_step split TensorRT`，因此不再维护 suffix-only
上机、smoke 和 benchmark 脚本。历史报告仍保留在 `docs/` 中，方便追溯。

## 2. 导出和构建入口

推荐命令：

```bash
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/export_pi05_split_artifacts.py \
  --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
  --build-fp32 \
  --build-fp16
```

默认行为：

- 如果 ONNX 已存在，复用现有 ONNX。
- 如果 engine 已存在，复用现有 engine。
- 不加 `--force` 不会覆盖现有大文件。

当前已复用的 artifact：

```text
prefix_cache ONNX:
  my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp32.onnx

denoise_step ONNX:
  my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp32.onnx

fp32 engines:
  my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp32.engine
  my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp32.engine

fp16 constrained engine:
  my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp16_constrained.engine
```

注意：`fp16_constrained` profile 仍然使用 fp32 的 `prefix_cache` engine，只把 `denoise_step` 换成 constrained fp16 engine。

## 3. 离线精度验证入口

推荐命令：

```bash
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/validate_pi05_split_profiles.py \
  --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
  --profiles fp32 fp16_constrained \
  --report-dir my_devs/openpi_trt/artifacts/reports/refactor_profiles \
  --metadata-dir my_devs/openpi_trt/artifacts/profiles/refactor_profiles
```

这条命令只做 synthetic batch 的 Torch `sample_actions` vs split TensorRT `sample_actions` 对比：

- 不打开相机。
- 不连接机器人。
- 不发送 action。

本轮验证结果：

```text
fp32:
  passed_allclose: True
  mean_abs_diff:   0.00028732
  max_abs_diff:    0.00089842
  cosine:          0.99999983

fp16_constrained:
  passed_allclose: True
  mean_abs_diff:   0.00109058
  max_abs_diff:    0.00623655
  cosine:          0.99999681
```

生成的报告：

```text
my_devs/openpi_trt/artifacts/reports/refactor_profiles/validate_fp32.json
my_devs/openpi_trt/artifacts/reports/refactor_profiles/validate_fp16_constrained.json
```

生成的 metadata：

```text
my_devs/openpi_trt/artifacts/profiles/refactor_profiles/fp32.metadata.json
my_devs/openpi_trt/artifacts/profiles/refactor_profiles/fp16_constrained.metadata.json
```

## 4. 上机前推荐顺序

上机交给操作者执行。代码侧推荐的检查顺序：

1. 离线 artifact I/O 检查：

   ```bash
   conda run -n lerobot_flex python my_devs/openpi_trt/scripts/check_split_trt_artifact.py \
     --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
     --prefix-engine-path my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp32.engine \
     --denoise-engine-path my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp32.engine
   ```

2. 离线 profile 精度验证：

   ```bash
   conda run -n lerobot_flex python my_devs/openpi_trt/scripts/validate_pi05_split_profiles.py \
     --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
     --profiles fp32 fp16_constrained
   ```

3. 可选 camera-only smoke：

   ```bash
   conda run -n lerobot_flex python my_devs/openpi_trt/scripts/smoke_pi05_camera_split_trt.py \
     --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
     --prefix-engine-path my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp32.engine \
     --denoise-engine-path my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp32.engine \
     --top-cam /dev/video4 \
     --wrist-cam /dev/video6 \
     --top-cam-fourcc YUYV \
     --wrist-cam-fourcc MJPG \
     --task "Put the eraser into the small box" \
     --save-frames
   ```

4. 真实上机由操作者显式执行，必须使用 `--confirm-control`。

## 5. 后续继续整理

下一步建议继续拆：

```text
common/policy.py      policy/processors/batch/noise
common/camera.py      camera-only smoke 共享逻辑
common/reports.py     tensor stats/report 写入
common/robot_runtime.py 真实机器人循环，继续保留安全门
```

当前这轮先完成核心边界：

```text
runtime = 运行时
exporters = 导出/构建
validation = 精度验证
scripts = 命令入口
```
