# ONNXRuntime 推理计划（ACT 单臂，替代 PyTorch 前向）

面向现有推理脚本：`agilex_infer_single_cc_vertical.py`

目标：在不改变机器人控制/队列语义的前提下，把 **ACT 核心网络的前向**从 PyTorch 替换为 **ONNXRuntime**，其余预处理/后处理尽量保持一致。

---

## 1. 产物与依赖

### 1.1 单个 checkpoint 需要的文件

以 `<ckpt>=outputs/act_agilex_left_box/checkpoints/last/pretrained_model` 为例：

- ONNX：`<ckpt>/act_single.onnx`
- 导出元数据：`<ckpt>/export_metadata.json`
- 输入归一化 stats：`<ckpt>/policy_preprocessor_step_3_normalizer_processor.safetensors`
- 输出反归一化 stats：`<ckpt>/policy_postprocessor_step_0_unnormalizer_processor.safetensors`

如果你把 ONNX/元数据导出到了其它目录（例如 `trt_act/export/exported_models/`），推理脚本侧通过 CLI 指定：

- `--onnx trt_act/export/exported_models/left_arm.onnx`
- `--export-metadata trt_act/export/exported_models/export_metadata.json`

### 1.2 推理脚本（已实现）

- ORT runner（含队列语义）：`trt_act/inference/ort_policy.py`
- AgileX 单臂推理入口：`trt_act/inference/agilex_infer_ort_single.py`

---

## 2. 关键一致性约束

1. 相机顺序必须与 `config.json` 的 VISUAL 插入顺序一致（优先读 `export_metadata.json`）。
2. 预处理必须严格对齐：`uint8 HWC -> float32 /255 -> CHW -> mean/std (eps=1e-8)`。
3. 后处理必须严格对齐：`action = action_norm * std + mean`。
4. 保持 `ACTPolicy.select_action` 的队列语义：队列为空时跑一次 chunk 前向，其余步只出队。

---

## 3. 使用方式

CPU（默认）：

```bash
python trt_act/inference/agilex_infer_ort_single.py \
  --checkpoint outputs/act_agilex_left_box/checkpoints/last/pretrained_model \
  --arm left --fps 30 --binary-gripper \
  --onnx trt_act/export/exported_models/left_arm.onnx \
  --export-metadata trt_act/export/exported_models/export_metadata.json
```

CUDA Provider（需要安装 `onnxruntime-gpu` 且 provider 可用）：

```bash
python trt_act/inference/agilex_infer_ort_single.py \
  --checkpoint outputs/act_agilex_left_box/checkpoints/last/pretrained_model \
  --arm left --fps 30 --binary-gripper \
  --onnx trt_act/export/exported_models/left_arm.onnx \
  --export-metadata trt_act/export/exported_models/export_metadata.json \
  --ort-provider cuda
```
