# GROOT ONNX 导出与一致性验证详细报告（Rerun: 2026-03-04 10:20:22）

## 1. 目标与边界

本报告对应你要求的“清空 `outputs/trt` 后完整重做”这一轮执行，目标是：

1. 按上游拆分思路，导出多模块 ONNX（backbone + action head）。
2. 对导出的 ONNX 与 PyTorch 参考实现做一致性验证。
3. 输出可追溯、可复现、可交叉校验的详细结果。

边界与约束：

- 开发与脚本仅使用 `my_devs/` 路径。
- 不修改 `src/` 业务代码（本轮 `git status --short src` 为空）。
- 环境固定为 `conda` 环境 `lerobot_flex`。

---

## 2. 上游方案对齐关系（你问的“上游方案”）

本轮是按上游 Isaac-GR00T 的“分模块导出 + 分模块构建 + 分模块校验”思想来执行的。

上游参考位置：

- `my_devs/docs/gr00t_trt/Isaac-GR00T-n1.5-release/deployment_scripts/export_onnx.py`
- `my_devs/docs/gr00t_trt/Isaac-GR00T-n1.5-release/deployment_scripts/build_engine.sh`

本仓库对应落地脚本（全部在 `my_devs/groot_trt/`）：

- `export_backbone_onnx.py`：导出 ViT / LLM
- `export_action_head_onnx.py`：导出 action head 的 5 个子模块
- `compare_torch_onnx.py`：模块级 + 流程级一致性比较

---

## 3. 环境与输入

### 3.1 环境

- Conda: `lerobot_flex`
- Python: `3.10.19`
- Torch: `2.7.1+cu126`
- ONNX: `1.17.0`
- ONNX Runtime: `1.20.1`
- ORT 可用 provider: `['AzureExecutionProvider', 'CPUExecutionProvider']`
- 本次 compare 实际使用 provider（记录在 JSON）：`["CPUExecutionProvider"]`

### 3.2 权重与产物路径

- Policy 权重：
  - `/data/cqy_workspace/flexible_lerobot/outputs/train/groot_grasp_block_in_bin1_repro_20260302_223413/bs32_20260302_223447/checkpoints/last/pretrained_model`
- 本轮产物根目录：
  - `outputs/trt/consistency_rerun_20260304_102022`

---

## 4. 执行过程（命令与日志）

以下为本轮执行使用的命令模板（与本轮日志和产物一致）：

1) 导出 backbone（ViT + LLM）

```bash
conda run -n lerobot_flex python my_devs/groot_trt/export_backbone_onnx.py \
  --policy-path /data/cqy_workspace/flexible_lerobot/outputs/train/groot_grasp_block_in_bin1_repro_20260302_223413/bs32_20260302_223447/checkpoints/last/pretrained_model \
  --onnx-out-dir outputs/trt/consistency_rerun_20260304_102022/gr00t_onnx \
  --seq-len 296 \
  --video-views 1 \
  --vit-dtype fp16 \
  --llm-dtype fp16 \
  --device cuda
```

2) 导出 action head（5 子模块）

```bash
conda run -n lerobot_flex python my_devs/groot_trt/export_action_head_onnx.py \
  --policy-path /data/cqy_workspace/flexible_lerobot/outputs/train/groot_grasp_block_in_bin1_repro_20260302_223413/bs32_20260302_223447/checkpoints/last/pretrained_model \
  --onnx-out-dir outputs/trt/consistency_rerun_20260304_102022/gr00t_onnx \
  --seq-len 296 \
  --device cuda
```

3) 1-view 一致性比较（seq=296）

```bash
conda run -n lerobot_flex python my_devs/groot_trt/compare_torch_onnx.py \
  --policy-path /data/cqy_workspace/flexible_lerobot/outputs/train/groot_grasp_block_in_bin1_repro_20260302_223413/bs32_20260302_223447/checkpoints/last/pretrained_model \
  --onnx-dir outputs/trt/consistency_rerun_20260304_102022/gr00t_onnx \
  --seq-len 296 \
  --video-views 1 \
  --vit-dtype fp16 --llm-dtype fp16 --dit-dtype fp16 \
  --device cuda \
  --json-out outputs/trt/consistency_rerun_20260304_102022/gr00t_onnx/compare_metrics_1view.json
```

4) 2-view 一致性比较（seq=568）

```bash
conda run -n lerobot_flex python my_devs/groot_trt/compare_torch_onnx.py \
  --policy-path /data/cqy_workspace/flexible_lerobot/outputs/train/groot_grasp_block_in_bin1_repro_20260302_223413/bs32_20260302_223447/checkpoints/last/pretrained_model \
  --onnx-dir outputs/trt/consistency_rerun_20260304_102022/gr00t_onnx \
  --seq-len 568 \
  --video-views 2 \
  --vit-dtype fp16 --llm-dtype fp16 --dit-dtype fp16 \
  --device cuda \
  --json-out outputs/trt/consistency_rerun_20260304_102022/gr00t_onnx/compare_metrics_2view.json
```

日志（含时间戳）：

- `outputs/trt/consistency_rerun_20260304_102022/logs/export_backbone.log`（2026-03-04 10:20:36）
- `outputs/trt/consistency_rerun_20260304_102022/logs/export_action_head.log`（2026-03-04 10:20:49）
- `outputs/trt/consistency_rerun_20260304_102022/logs/compare_1view.log`（2026-03-04 10:21:38）
- `outputs/trt/consistency_rerun_20260304_102022/logs/compare_2view.log`（2026-03-04 10:21:55）

---

## 5. 导出脚本实现说明（代码层）

### 5.1 `export_backbone_onnx.py`

关键点：

- 通过 `VisionModelForOnnx` 包装视觉塔，导出 `vit_fp16.onnx`。
- 通过 `LanguageModelForOnnx` 包装 LLM 路径，显式构建 causal mask，导出 `llm_fp16.onnx`。
- 将 Vision/LLM 的 attention 实现切换为 `eager`，规避 FlashAttention 路径在 ONNX 导出阶段的不稳定性。
- dynamic axes：
  - ViT：batch 动态
  - LLM：batch 与 sequence 动态

### 5.2 `export_action_head_onnx.py`

导出 action head 五模块：

1. `vlln_vl_self_attention.onnx`
2. `state_encoder.onnx`
3. `action_encoder.onnx`
4. `DiT_fp16.onnx`
5. `action_decoder.onnx`

特点：

- 按模块分别 trace/export，便于后续 TRT 分模块构建与定位误差。
- 所有模块采用 FP16 trace，opset=19。
- 对 batch/sequence 关键维度设置 dynamic axes。

### 5.3 `compare_torch_onnx.py`

对比分为两层：

1. 模块级：ViT、ViT 后处理、LLM（两种输入链路）、ActionHead 五子模块。
2. 流程级：`action_denoising_pipeline`（多步迭代，最接近真实推理闭环）。

可复现性策略：

- 固定随机种子：`20260303`
- 固定 profile：
  - `video_views=1, seq_len=296`
  - `video_views=2, seq_len=568`
- LLM pipeline 比较中复用同一组 `input_ids`，减少无关输入扰动。

---

## 6. ONNX 导出产物清单（7 模型）

目录：`outputs/trt/consistency_rerun_20260304_102022/gr00t_onnx`

| 模型文件 | 大小（bytes） | 约合 |
|---|---:|---:|
| `eagle2/vit_fp16.onnx` | 825,458,145 | 787.22 MiB |
| `eagle2/llm_fp16.onnx` | 1,208,598,013 | 1.13 GiB |
| `action_head/vlln_vl_self_attention.onnx` | 402,962,061 | 384.29 MiB |
| `action_head/state_encoder.onnx` | 105,022,873 | 100.16 MiB |
| `action_head/action_encoder.onnx` | 456,432,921 | 435.29 MiB |
| `action_head/DiT_fp16.onnx` | 1,101,316,622 | 1.03 GiB |
| `action_head/action_decoder.onnx` | 69,275,044 | 66.07 MiB |

### 6.1 ONNX I/O 签名

#### `eagle2/vit_fp16.onnx`
- Inputs
  - `pixel_values`: `[batch_size, 3, 224, 224]` (FLOAT16)
  - `position_ids`: `[batch_size, 256]` (INT64)
- Outputs
  - `vit_embeds`: `[batch_size, Addvit_embeds_dim_1, Addvit_embeds_dim_2]` (FLOAT16)

#### `eagle2/llm_fp16.onnx`
- Inputs
  - `inputs_embeds`: `[batch_size, sequence_length, 2048]` (FLOAT16)
  - `attention_mask`: `[batch_size, sequence_length]` (INT64)
- Outputs
  - `embeddings`: `[batch_size, sequence_length, 2048]` (FLOAT16)

#### `action_head/vlln_vl_self_attention.onnx`
- Inputs
  - `backbone_features`: `[batch_size, sequence_length, 2048]` (FLOAT16)
- Outputs
  - `output`: `[batch_size, sequence_length, 2048]` (FLOAT16)

#### `action_head/state_encoder.onnx`
- Inputs
  - `state`: `[batch_size, 1, 64]` (FLOAT16)
  - `embodiment_id`: `[batch_size]` (INT64)
- Outputs
  - `output`: `[batch_size, 1, 1536]` (FLOAT16)

#### `action_head/action_encoder.onnx`
- Inputs
  - `actions`: `[batch_size, 16, 32]` (FLOAT16)
  - `timesteps_tensor`: `[batch_size]` (INT64)
  - `embodiment_id`: `[batch_size]` (INT64)
- Outputs
  - `output`: `[batch_size, 16, 1536]` (FLOAT16)

#### `action_head/DiT_fp16.onnx`
- Inputs
  - `sa_embs`: `[batch_size, 49, 1536]` (FLOAT16)
  - `vl_embs`: `[batch_size, sequence_length, 2048]` (FLOAT16)
  - `timesteps_tensor`: `[batch_size]` (INT64)
- Outputs
  - `output`: `[batch_size, Addoutput_dim_1, 1024]` (FLOAT16)

#### `action_head/action_decoder.onnx`
- Inputs
  - `model_output`: `[batch_size, 49, 1024]` (FLOAT16)
  - `embodiment_id`: `[batch_size]` (INT64)
- Outputs
  - `output`: `[batch_size, 49, 32]` (FLOAT16)

---

## 7. 一致性验证方法与指标

指标定义（每个模块都算）：

- `cosine`：整体方向一致性（越接近 1 越好）
- `rmse`：均方根误差（越小越好）
- `mean_abs`：平均绝对误差（越小越好）
- `max_abs`：最大绝对误差（越小越好）

对比范围：

- Profile A：`1 view / seq=296`
- Profile B：`2 views / seq=568`

输出 JSON：

- `outputs/trt/consistency_rerun_20260304_102022/gr00t_onnx/compare_metrics_1view.json`
- `outputs/trt/consistency_rerun_20260304_102022/gr00t_onnx/compare_metrics_2view.json`

---

## 8. 结果明细

### 8.1 Profile A（1 view, seq=296）

| 模块 | cosine | rmse | mean_abs | max_abs |
|---|---:|---:|---:|---:|
| vit | 0.99998450 | 0.00551750 | 0.00330749 | 0.18750000 |
| vit_postprocess | 0.99998839 | 0.00384091 | 0.00236022 | 0.09082031 |
| llm_from_vit_pipeline | 0.99997428 | 0.01507224 | 0.00884274 | 0.52343750 |
| llm_direct | 0.99999430 | 0.00540955 | 0.00330271 | 0.31250000 |
| action_vlln_vl_self_attention | 0.99999945 | 0.00441196 | 0.00312413 | 0.18750000 |
| action_state_encoder | 0.99999995 | 0.00002247 | 0.00001180 | 0.00012207 |
| action_action_encoder | 0.99999987 | 0.00011265 | 0.00007458 | 0.00048828 |
| action_dit | 0.99999975 | 0.00048598 | 0.00032585 | 0.00418472 |
| action_decoder | 0.99999994 | 0.00009337 | 0.00005027 | 0.00048828 |
| action_denoising_pipeline | 0.99999996 | 0.00029651 | 0.00011092 | 0.00195312 |

Profile A 最低 cosine：`llm_from_vit_pipeline = 0.99997428`

### 8.2 Profile B（2 views, seq=568）

| 模块 | cosine | rmse | mean_abs | max_abs |
|---|---:|---:|---:|---:|
| vit | 0.99998702 | 0.00504953 | 0.00310211 | 0.26562500 |
| vit_postprocess | 0.99999031 | 0.00351919 | 0.00222844 | 0.06616211 |
| llm_from_vit_pipeline | 0.99998694 | 0.01047317 | 0.00597902 | 0.60156250 |
| llm_direct | 0.99999394 | 0.00572747 | 0.00349664 | 0.35937500 |
| action_vlln_vl_self_attention | 0.99999945 | 0.00401951 | 0.00288285 | 0.18750000 |
| action_state_encoder | 0.99999994 | 0.00002401 | 0.00001234 | 0.00012207 |
| action_action_encoder | 0.99999987 | 0.00011685 | 0.00007714 | 0.00048828 |
| action_dit | 0.99999990 | 0.00035535 | 0.00020288 | 0.00195312 |
| action_decoder | 0.99999995 | 0.00008978 | 0.00004808 | 0.00048828 |
| action_denoising_pipeline | 0.99999996 | 0.00029330 | 0.00011956 | 0.00195312 |

Profile B 最低 cosine：`llm_from_vit_pipeline = 0.99998694`

---

## 9. 多智能体交叉校验结果（独立复核）

为了避免单路径误判，本轮对结果做了三路独立校验：

1. 校验 A（仅看 JSON）
   - 各 profile 最低 cosine 均定位到 `llm_from_vit_pipeline`
   - `action_denoising_pipeline` 两个 profile 均为 `~0.99999996`
   - `missing=[]`
   - providers = `["CPUExecutionProvider"]`

2. 校验 B（仅看 compare 日志）
   - `compare_1view.log` 与 `compare_metrics_1view.json` 一致（按日志精度四舍五入）
   - `compare_2view.log` 与 `compare_metrics_2view.json` 一致
   - 不一致项：无

3. 校验 C（仅看 ONNX 目录）
   - `.onnx` 文件总数 = 7（完整）
   - 文件大小与 I/O 签名均可解析，且与导出预期一致

结论：三路校验互相一致，无冲突。

---

## 10. 结论与下一步建议

### 10.1 结论

- “多 ONNX 分模块导出”已成功完成（7 模型齐全）。
- “Torch vs ONNX 一致性验证”已成功完成（模块级 + 流程级）。
- 核心闭环指标 `action_denoising_pipeline` 在 1-view 与 2-view 均达到 `0.99999996` 量级，说明导出模型可作为 TRT 构建基线。

### 10.2 注意点（如需继续）

- 当前 ORT 比较走 CPU provider，属于数值一致性验证，不是性能测试。
- 后续建议按同目录继续做：
  - ONNX → TRT engine 构建
  - Torch vs TRT 一致性复核
  - 端到端推理时延与吞吐测试

---

## 11. 快速索引（本次最关键文件）

- 报告文件：
  - `my_devs/docs/gr00t_trt/onnx_consistency_rerun_report_20260304.md`
- 导出脚本：
  - `my_devs/groot_trt/export_backbone_onnx.py`
  - `my_devs/groot_trt/export_action_head_onnx.py`
- 对比脚本：
  - `my_devs/groot_trt/compare_torch_onnx.py`
- 产物与日志：
  - `outputs/trt/consistency_rerun_20260304_102022/gr00t_onnx`
  - `outputs/trt/consistency_rerun_20260304_102022/logs`
