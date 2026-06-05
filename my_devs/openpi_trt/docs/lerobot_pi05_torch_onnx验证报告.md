# LeRobot PI0.5 Torch vs ONNX 数值验证报告

## 结论

已完成第一阶段 Torch vs ONNX 数值验证。

本次验证采用拆分后的 PI0.5 后端内部边界：

```text
suffix_embedding:
  noisy_actions + timestep -> suffix_embs + adarms_cond
```

这是 `PI05Pytorch.embed_suffix(...)` 对应的 action/time suffix embedding 子图，属于 PI0.5 denoise step 的一部分。完整 `sample_actions` 单体图导出仍需继续攻克，原因见下文。

## 产物

```text
my_devs/openpi_trt/scripts/pi05_onnx_common.py
my_devs/openpi_trt/scripts/export_lerobot_pi05_onnx.py
my_devs/openpi_trt/scripts/verify_lerobot_pi05_torch_onnx.py
my_devs/openpi_trt/artifacts/pi05_so101_suffix_embedding_b1.onnx
my_devs/openpi_trt/artifacts/verify_torch_onnx_suffix_embedding.json
```

ONNX 文件大小：

```text
8.2M my_devs/openpi_trt/artifacts/pi05_so101_suffix_embedding_b1.onnx
```

## 验证环境

所有命令均使用：

```bash
conda run -n lerobot_flex ...
```

本次为了尝试 PyTorch dynamo ONNX exporter，在 `lerobot_flex` 环境补装了：

```text
onnxscript==0.7.0
onnx_ir==0.2.1
ml_dtypes==0.5.4
```

## 导出命令

```bash
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/export_lerobot_pi05_onnx.py \
  --mode suffix_embedding \
  --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
  --output my_devs/openpi_trt/artifacts/pi05_so101_suffix_embedding_b1.onnx
```

导出输入：

```text
noisy_actions: shape=(1, 50, 32) dtype=torch.float32 device=cuda:0
timestep: shape=(1,) dtype=torch.float32 device=cuda:0
```

导出输出：

```text
suffix_embs: [1, 50, 1024]
adarms_cond: [1, 1024]
```

导出结果：

```text
ONNX export completed: my_devs/openpi_trt/artifacts/pi05_so101_suffix_embedding_b1.onnx
```

## 验证命令

```bash
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/verify_lerobot_pi05_torch_onnx.py \
  --mode suffix_embedding \
  --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
  --onnx-path my_devs/openpi_trt/artifacts/pi05_so101_suffix_embedding_b1.onnx \
  --rtol 1e-4 \
  --atol 2e-3 \
  --report my_devs/openpi_trt/artifacts/verify_torch_onnx_suffix_embedding.json
```

验证结果：

```text
allclose(rtol=0.0001, atol=0.002)=True
```

详细数值：

```text
suffix_embs:
  torch shape: [1, 50, 1024]
  onnx shape:  [1, 50, 1024]
  mean_abs_diff: 0.00027049571508541703
  max_abs_diff:  0.0016807913780212402
  cosine_similarity: 0.9999999575057874

adarms_cond:
  torch shape: [1, 1024]
  onnx shape:  [1, 1024]
  mean_abs_diff: 3.1144256951165517e-08
  max_abs_diff:  2.384185791015625e-07
  cosine_similarity: 0.9999999999997655
```

## 完整 sample_actions 导出尝试

本次也尝试了完整边界：

```text
sample_actions:
  image_0 + image_1 + img_mask_0 + img_mask_1 + tokens + masks + noise -> actions
```

输入形状已经验证正确：

```text
image_0:    [1, 3, 224, 224] float32
image_1:    [1, 3, 224, 224] float32
img_mask_0: [1] bool
img_mask_1: [1] bool
tokens:     [1, 200] int64
masks:      [1, 200] bool
noise:      [1, 50, 32] float32
```

legacy ONNX exporter 失败点：

```text
RuntimeError: ScalarType ComplexDouble is an unexpected tensor scalar type
```

这个错误出现在 Gemma/RoPE 相关导出路径，日志中同时出现：

```text
UserWarning: Casting complex values to real discards the imaginary part
```

dynamo ONNX exporter 初始失败点：

```text
ModuleNotFoundError: No module named 'onnxscript'
```

补装 `onnxscript` 后，dynamo exporter 又遇到 transformers GemmaRMSNorm repr 兼容问题：

```text
AttributeError: 'GemmaRMSNorm' object has no attribute 'weight'
```

已在 `my_devs/openpi_trt/scripts/pi05_onnx_common.py` 中添加 exporter-only monkey patch，只影响 `extra_repr`，不改变模型计算。

补丁后，dynamo 完整图导出可以进入长时间图捕获/转换阶段，但运行数分钟没有生成 ONNX 产物。进程当时仍有 CPU/GPU 占用：

```text
python export_lerobot_pi05_onnx.py ... --dynamo
GPU memory: about 7620 MiB
```

为先完成 Torch vs ONNX 数值验证，已停止该长导出，切换到 split-graph 的 `suffix_embedding` 子图完成验证闭环。

## 当前判断

`suffix_embedding` 子图已证明：

- LeRobot PI0.5 checkpoint 可以从 `my_devs/openpi_trt` 脚本加载。
- 固定输入和固定 seed 可复现 Torch baseline。
- ONNX 导出可成功生成。
- ONNX Runtime 输出与 Torch 输出在合理阈值内一致。

这一步完成的是拆分图路线中的第一个可验证子图，不等于完整 TensorRT 后端已经完成。

下一步建议继续拆：

1. `embed_prefix` 子图：

```text
image_0 + image_1 + img_mask_0 + img_mask_1 + tokens + masks
  -> prefix_embs + prefix_pad_masks + prefix_att_masks
```

2. `denoise_step` 子图：

```text
prefix_pad_masks + past_key_values + x_t + timestep -> v_t
```

3. Python runtime 保留 denoise loop，逐步替换 PyTorch 子模块为 ONNX/TRT。

