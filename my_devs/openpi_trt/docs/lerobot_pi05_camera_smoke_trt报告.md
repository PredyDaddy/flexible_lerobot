# LeRobot PI0.5 Camera Smoke PyTorch vs TensorRT FP16 报告

## 结论

已完成一次安全的 camera-only smoke 推理对比：

- 使用真实摄像头 `/dev/video4` 和 `/dev/video6` 采集到的画面。
- 先运行 PyTorch PI0.5 `sample_actions(...)`。
- 再运行 TensorRT FP16 suffix engine 加速后的 PI0.5 `sample_actions(...)`。
- 两者 action chunk 数值一致性通过。
- 全程没有连接机器人串口。
- 全程没有发送机器人 action。

本次 TensorRT 替换边界：

```text
PI05Pytorch.embed_suffix(noisy_actions, timestep)
```

也就是说当前脚本验证的是完整 `sample_actions(...)` 流程里，suffix embedding 子图替换成 TensorRT FP16 后，最终 action chunk 是否仍能和 PyTorch 对齐。完整 PaliGemma/Gemma denoise step 尚未全部替换成 TensorRT。

## 新增脚本

camera-only smoke 脚本：

```text
my_devs/openpi_trt/scripts/smoke_pi05_camera_torch_trt.py
```

TensorRT patch 共享 runtime：

```text
my_devs/openpi_trt/runtime/pi05_trt_suffix.py
```

后续上机脚本：

```text
my_devs/openpi_trt/scripts/run_pi05_trt_infer_so101.py
```

注意：上机脚本默认不会连接机器人，必须显式加：

```text
--confirm-control
```

才会连接机器人并发送 action。

## 使用的 TensorRT Engine

```text
my_devs/openpi_trt/artifacts/pi05_so101_suffix_embedding_b1_fp16.engine
```

## 安全验证

执行过 `run_pi05_trt_infer_so101.py --check-policy-load`：

```bash
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/run_pi05_trt_infer_so101.py \
  --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
  --engine-path my_devs/openpi_trt/artifacts/pi05_so101_suffix_embedding_b1_fp16.engine \
  --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00 \
  --top-cam /dev/video4 \
  --wrist-cam /dev/video6 \
  --task "Put the eraser into the small box" \
  --check-policy-load
```

结果：

```text
CHECK_POLICY_LOAD=true, loaded policy/processors/TRT engine; exiting before robot connection.
```

这证明脚本在未加 `--confirm-control` 且 check 模式下会在机器人连接前退出。

## Camera Smoke 命令

第一次 smoke：

```bash
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/smoke_pi05_camera_torch_trt.py \
  --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
  --engine-path my_devs/openpi_trt/artifacts/pi05_so101_suffix_embedding_b1_fp16.engine \
  --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00 \
  --top-cam /dev/video4 \
  --wrist-cam /dev/video6 \
  --top-cam-fourcc YUYV \
  --wrist-cam-fourcc MJPG \
  --task "Put the eraser into the small box" \
  --rtol 2e-2 \
  --atol 2e-2 \
  --save-frames \
  --report my_devs/openpi_trt/artifacts/camera_smoke_torch_trt_fp16.json
```

第二次 smoke 是在把 TensorRT patch 抽成共享 runtime 后重新验证：

```bash
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/smoke_pi05_camera_torch_trt.py \
  --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
  --engine-path my_devs/openpi_trt/artifacts/pi05_so101_suffix_embedding_b1_fp16.engine \
  --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00 \
  --top-cam /dev/video4 \
  --wrist-cam /dev/video6 \
  --top-cam-fourcc YUYV \
  --wrist-cam-fourcc MJPG \
  --task "Put the eraser into the small box" \
  --rtol 2e-2 \
  --atol 2e-2 \
  --save-frames \
  --report my_devs/openpi_trt/artifacts/camera_smoke_torch_trt_fp16_after_runtime_refactor.json
```

## Smoke 结果

第一次：

```text
passed_allclose: True
robot_connected: False
robot_action_sent: False
mean_abs_diff: 0.0006899805157445371
max_abs_diff: 0.0031861066818237305
cosine_similarity: 0.9999993833670018
```

第二次：

```text
passed_allclose: True
robot_connected: False
robot_action_sent: False
mean_abs_diff: 0.0006627458496950567
max_abs_diff: 0.002732396125793457
cosine_similarity: 0.9999994412319524
```

第二次 smoke 对应最终代码状态，因此后续以上面第二份报告为准。

## 保存的相机帧

第一次：

```text
my_devs/openpi_trt/artifacts/camera_smoke_torch_trt_fp16_frames/top_rgb.png
my_devs/openpi_trt/artifacts/camera_smoke_torch_trt_fp16_frames/wrist_rgb.png
```

第二次：

```text
my_devs/openpi_trt/artifacts/camera_smoke_torch_trt_fp16_after_runtime_refactor_frames/top_rgb.png
my_devs/openpi_trt/artifacts/camera_smoke_torch_trt_fp16_after_runtime_refactor_frames/wrist_rgb.png
```

## 后续上机命令

一致性 smoke 已通过后，可以使用新脚本上机。注意必须显式添加 `--confirm-control`：

```bash
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/run_pi05_trt_infer_so101.py \
  --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
  --engine-path my_devs/openpi_trt/artifacts/pi05_so101_suffix_embedding_b1_fp16.engine \
  --robot-port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123192-if00 \
  --top-cam /dev/video4 \
  --wrist-cam /dev/video6 \
  --top-cam-fourcc YUYV \
  --wrist-cam-fourcc MJPG \
  --task "Put the eraser into the small box" \
  --run-time-s 120 \
  --confirm-control
```

不加 `--confirm-control` 时，脚本会在机器人连接前退出。

## 当前限制

当前 TensorRT FP16 只替换了 `embed_suffix(...)` 子图，完整 PI0.5 后端还不是全 TensorRT。它已经能证明：

- 使用真实相机画面时，PyTorch 与 TensorRT suffix engine 的最终 action chunk 可以对齐。
- 新上机脚本可以加载 policy、processor、TensorRT engine。
- 上机脚本具有显式控制确认门。

后续如果要进一步提升速度，需要继续把 `embed_prefix`、prefix cache、denoise step 中的 Gemma expert 前向拆出并导出 TensorRT。

