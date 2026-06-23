# openpi_trt 最终精简说明

日期：2026-06-23  
模块：`my_devs/openpi_trt`

## 1. 背景

`my_devs/openpi_trt` 最初混合了三类内容：

1. Jetson AI Lab / OpenPI on Thor 的上游参考代码。
2. LeRobot PI0.5 的 ONNX/TensorRT 实验脚本。
3. SO101 真实机器人上机入口和 `my_devs/vla_engineering` 异步后端引用的 runtime。

早期探索阶段保留了很多中间脚本是合理的，但在 fp32 和 fp16 constrained 都完成真实上机验证之后，继续保留多套入口会带来几个问题：

- 不知道应该使用哪个导出、验证、上机脚本。
- 旧 suffix 单 engine 路线和最终 split 路线并存，容易误用。
- `runtime/` 里有多份功能相近的实现，后续维护成本高。
- `my_devs/vla_engineering` 依赖旧 import path，不能直接粗暴删除。

本次精简目标是：保留已经上机成功的 split TensorRT 主链路，删除重复脚本，同时不破坏 `vla_engineering`。

## 2. 如何从参考仓库开展工作

上游参考代码仍保留在：

```text
my_devs/openpi_trt/openpi_on_thor/
```

这部分只作为参考，不作为当前 LeRobot 上机路径。真正开展工作时按下面顺序：

1. 从训练好的 LeRobot checkpoint 加载 PI0.5 policy：

   ```text
   outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model
   ```

2. 用 `scripts/pi05_onnx_common.py` 里的 wrapper 和 dummy batch 构造导出边界。

3. 用 `scripts/simple_pi05_pipeline.py` 导出两个 ONNX：

   ```text
   pi05_so101_prefix_cache_b1_fp32.onnx
   pi05_so101_denoise_step_b1_fp32.onnx
   ```

4. 构建三个部署 engine：

   ```text
   pi05_so101_prefix_cache_b1_fp32.engine
   pi05_so101_denoise_step_b1_fp32.engine
   pi05_so101_denoise_step_b1_fp16_constrained.engine
   ```

5. 用同一个 pipeline 做 Torch vs TensorRT 数值验证。

6. 用 `scripts/simple_pi05_run_robot.py` 上机。

这条链路避免直接套用 OpenPI Thor 参考脚本里的固定输入名和固定形状，因为当前模型来自本仓库的 LeRobot PI0.5 训练代码。

## 3. 精简后的代码有什么好处

### 入口更少

当前 `scripts/` 只保留：

```text
pi05_onnx_common.py
simple_pi05_pipeline.py
simple_pi05_validate.py
simple_pi05_run_robot.py
run_pi05_split_trt_infer_so101.py
```

其中：

- `simple_pi05_pipeline.py` 是导出、转换、验证主入口。
- `simple_pi05_validate.py` 是验证-only 便捷入口。
- `simple_pi05_run_robot.py` 是真实上机主入口。
- `run_pi05_split_trt_infer_so101.py` 是旧命令兼容 wrapper。

### runtime 更集中

当前 `runtime/` 只保留：

```text
trt_engine.py
protocol.py
config.py
metadata.py
simple_pi05_split.py
pi05_trt_split.py
```

其中：

- `simple_pi05_split.py` 是唯一真实 split TensorRT runtime。
- `pi05_trt_split.py` 是兼容层，保留 `vla_engineering` 依赖的旧 import path。
- `protocol.py` 固定 prefix_cache / denoise_step 的 TensorRT I/O 名称。
- `trt_engine.py` 负责 TensorRT engine 的 torch tensor I/O。

### 删除了容易误用的旧路线

已删除的主要旧路线：

- suffix 单 engine runtime：`runtime/pi05_trt_suffix.py`
- suffix 上机/相机/benchmark 脚本
- 分散的旧导出、build、verify、smoke、benchmark 脚本
- 中间态 `common/`、`exporters/`、`validation/` 包

这些能力已经由 `simple_pi05_pipeline.py` 和 `simple_pi05_run_robot.py` 覆盖。

### 不影响 vla_engineering

`my_devs/vla_engineering` 仍然可以继续使用：

```python
from runtime.pi05_trt_split import patch_sample_actions_with_split_trt
```

该路径没有被删除，只是内部改成调用 `runtime.simple_pi05_split`。

## 4. 当前推荐命令

fp32 验证：

```bash
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/simple_pi05_pipeline.py \
  --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
  --profile fp32 \
  --report my_devs/openpi_trt/artifacts/simple_pipeline_fp32_report.json
```

fp16 constrained 验证：

```bash
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/simple_pi05_pipeline.py \
  --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
  --profile fp16_constrained \
  --report my_devs/openpi_trt/artifacts/simple_pipeline_fp16_constrained_report.json
```

fp16 constrained 上机：

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

## 5. 后续优化方案

1. **把 prefix_cache 也做 fp16 constrained 评估**  
   当前 prefix_cache 保持 fp32，denoise_step 支持 fp32 和 fp16 constrained。下一步可以评估 prefix_cache fp16 是否仍能保持动作精度。

2. **减少 Python denoise loop 开销**  
   当前 denoise loop 在 Python 中逐 step 调 denoise engine。后续可以探索 CUDA graph 或 TensorRT loop/fused engine，但必须先保持 Torch/TRT 数值可比。

3. **把 runtime metadata 固化到 artifact 旁边**  
   每次导出和转换后写出 engine 输入输出、precision、checkpoint、验证指标，避免 engine 文件和代码版本混淆。

4. **把 `vla_engineering` 迁移到 simple runtime 命名**  
   当前为了不破坏异步后端保留了 `runtime.pi05_trt_split`。后续可以在 `vla_engineering` 稳定后改为显式依赖 `runtime.simple_pi05_split`。

5. **保留真实上机边界**  
   后续仍然遵守当前原则：代码侧只做导出、转换、验证和命令准备，真实机器人控制由人工执行 `--confirm-control` 命令。
