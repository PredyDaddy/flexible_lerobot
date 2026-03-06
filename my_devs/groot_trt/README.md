# GROOT ONNX/TRT 工具（本仓库本地脚本）

这个目录 `my_devs/groot_trt/` 放的是 GR00T N1.5（LeRobot Groot policy）在本仓库里做工程化的脚本，覆盖：

- 导出 ONNX（backbone: ViT + LLM，action head: 5 个子模块）
- Torch vs ONNX 一致性对比（模块级 + 端到端 denoising pipeline）
- 用 TensorRT **Python API** 构建 `.engine`（不依赖 `trtexec`）
- Torch vs TensorRT 一致性对比（模块级 + 端到端 denoising pipeline）

我们已经按这套流程完整跑通过一轮，示例产物在：

- `outputs/trt/consistency_rerun_20260305_102931/`

## 0. 前置条件

- 必须使用 conda 环境：`lerobot_flex`
- 需要 CUDA 可用（Torch 在 GPU 上跑参考输出，TRT 引擎也在 GPU 上跑）
- 本流程是 FP16 baseline（量化不在这里展开）

建议在 repo 根目录执行：

```bash
cd /data/cqy_workspace/flexible_lerobot
```

准备统一的输出目录（推荐）：

```bash
RUN_ID=consistency_rerun_$(date +%Y%m%d_%H%M%S)
RUN_DIR=outputs/trt/$RUN_ID
mkdir -p $RUN_DIR/logs
echo $RUN_DIR
```

准备 policy 权重路径：

```bash
POLICY_PATH=/path/to/pretrained_model
```

## 1. 导出 ONNX（7 个文件）

1) 导出 backbone（ViT + LLM）：

```bash
conda run -n lerobot_flex python my_devs/groot_trt/export_backbone_onnx.py \
  --policy-path $POLICY_PATH \
  --onnx-out-dir $RUN_DIR/gr00t_onnx \
  --seq-len 296 \
  --video-views 1 \
  --vit-dtype fp16 \
  --llm-dtype fp16 \
  --device cuda \
  > $RUN_DIR/logs/export_backbone.log 2>&1
```

2) 导出 action head（5 个子模块）：

```bash
conda run -n lerobot_flex python my_devs/groot_trt/export_action_head_onnx.py \
  --policy-path $POLICY_PATH \
  --onnx-out-dir $RUN_DIR/gr00t_onnx \
  --seq-len 296 \
  --device cuda \
  > $RUN_DIR/logs/export_action_head.log 2>&1
```

导出完成后，目录里应该有 7 个 ONNX：

- `$RUN_DIR/gr00t_onnx/eagle2/vit_fp16.onnx`
- `$RUN_DIR/gr00t_onnx/eagle2/llm_fp16.onnx`
- `$RUN_DIR/gr00t_onnx/action_head/vlln_vl_self_attention.onnx`
- `$RUN_DIR/gr00t_onnx/action_head/state_encoder.onnx`
- `$RUN_DIR/gr00t_onnx/action_head/action_encoder.onnx`
- `$RUN_DIR/gr00t_onnx/action_head/DiT_fp16.onnx`
- `$RUN_DIR/gr00t_onnx/action_head/action_decoder.onnx`

## 2. Torch vs ONNX 一致性对比（推荐做 1-view + 2-view）

说明：

- `compare_torch_onnx.py` 会打印每个模块/流程的 cosine、rmse、mean_abs、max_abs
- 同时把完整结果写入 JSON

1) 1-view（`seq_len=296, video_views=1`）：

```bash
conda run -n lerobot_flex python my_devs/groot_trt/compare_torch_onnx.py \
  --policy-path $POLICY_PATH \
  --onnx-dir $RUN_DIR/gr00t_onnx \
  --seq-len 296 \
  --video-views 1 \
  --vit-dtype fp16 --llm-dtype fp16 --dit-dtype fp16 \
  --device cuda \
  --json-out $RUN_DIR/gr00t_onnx/compare_metrics_1view.json \
  > $RUN_DIR/logs/compare_onnx_1view.log 2>&1
```

2) 2-view（`seq_len=568, video_views=2`）：

```bash
conda run -n lerobot_flex python my_devs/groot_trt/compare_torch_onnx.py \
  --policy-path $POLICY_PATH \
  --onnx-dir $RUN_DIR/gr00t_onnx \
  --seq-len 568 \
  --video-views 2 \
  --vit-dtype fp16 --llm-dtype fp16 --dit-dtype fp16 \
  --device cuda \
  --json-out $RUN_DIR/gr00t_onnx/compare_metrics_2view.json \
  > $RUN_DIR/logs/compare_onnx_2view.log 2>&1
```

## 3. 构建 TensorRT 引擎（Python API）

我们用的是 TensorRT Python API（不是 `trtexec`）。默认 TensorRT 安装目录是：

- `/data/cqy_workspace/third_party/tensorrt_10_13_0_35`

如果你自己装在别的地方，运行时传 `TENSORRT_PY_DIR=/your/path` 即可。

构建引擎（默认 profile 按 2-view 的长度设置，覆盖 1-view/2-view）：

```bash
ONNX_DIR=$RUN_DIR/gr00t_onnx \
ENGINE_DIR=$RUN_DIR/gr00t_engine_api_trt1013 \
VIDEO_VIEWS=2 MAX_BATCH=2 WORKSPACE_GB=8 \
bash my_devs/groot_trt/build_engine.sh \
  > $RUN_DIR/logs/build_trt_engine.log 2>&1
```

构建完成后，目录里应该有 7 个 engine：

- `$RUN_DIR/gr00t_engine_api_trt1013/vit_fp16.engine`
- `$RUN_DIR/gr00t_engine_api_trt1013/llm_fp16.engine`
- `$RUN_DIR/gr00t_engine_api_trt1013/vlln_vl_self_attention.engine`
- `$RUN_DIR/gr00t_engine_api_trt1013/state_encoder.engine`
- `$RUN_DIR/gr00t_engine_api_trt1013/action_encoder.engine`
- `$RUN_DIR/gr00t_engine_api_trt1013/DiT_fp16.engine`
- `$RUN_DIR/gr00t_engine_api_trt1013/action_decoder.engine`

可选：如果你需要自定义动态 shape profile，可以额外设置：

- `MIN_LEN`、`OPT_LEN`、`MAX_LEN`

## 4. Torch vs TensorRT 一致性对比（推荐做 1-view + 2-view）

说明：

- `compare_torch_trt.py` 会加载上一步的 `.engine`，用同一套合成输入对比 Torch 输出
- 结果同样会打印到 stdout，并写入 JSON
- 建议设置 `TMPDIR` 到 `/data`，避免 `/tmp` 空间不足

1) 1-view（`seq_len=296, video_views=1`）：

```bash
TMPDIR=/data/cqy_workspace/tmp \
conda run -n lerobot_flex env TENSORRT_PY_DIR=/data/cqy_workspace/third_party/tensorrt_10_13_0_35 \
  python my_devs/groot_trt/compare_torch_trt.py \
    --policy-path $POLICY_PATH \
    --engine-dir $RUN_DIR/gr00t_engine_api_trt1013 \
    --seq-len 296 \
    --video-views 1 \
    --device cuda \
    --json-out $RUN_DIR/gr00t_engine_api_trt1013/compare_metrics_trt_1view.json \
    > $RUN_DIR/logs/compare_trt_1view.log 2>&1
```

2) 2-view（`seq_len=568, video_views=2`）：

```bash
TMPDIR=/data/cqy_workspace/tmp \
conda run -n lerobot_flex env TENSORRT_PY_DIR=/data/cqy_workspace/third_party/tensorrt_10_13_0_35 \
  python my_devs/groot_trt/compare_torch_trt.py \
    --policy-path $POLICY_PATH \
    --engine-dir $RUN_DIR/gr00t_engine_api_trt1013 \
    --seq-len 568 \
    --video-views 2 \
    --device cuda \
    --json-out $RUN_DIR/gr00t_engine_api_trt1013/compare_metrics_trt_2view.json \
    > $RUN_DIR/logs/compare_trt_2view.log 2>&1
```

## 5. 产物目录总览（你只需要记住这两个）

- ONNX：`$RUN_DIR/gr00t_onnx/`
- TensorRT engine：`$RUN_DIR/gr00t_engine_api_trt1013/`
