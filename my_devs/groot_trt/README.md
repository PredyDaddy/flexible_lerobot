# GROOT TensorRT Tooling (Local Dev)

This folder hosts **repo-local** scripts for exporting ONNX and building TensorRT engines for the
LeRobot GROOT (GR00T N1.5) policy.

Design goals:

- Keep day-to-day export/build work under `my_devs/` (your request).
- Keep `src/lerobot/...` clean and focused on runtime integration.
- Start with **FP16 baseline** and **action-head-only** acceleration, then expand to full-chain
  ViT + LLM + ActionHead.

## Quick Start (Action Head Only, FP16)

1) Export action-head ONNX:

```bash
conda run -n lerobot_flex python my_devs/groot_trt/export_action_head_onnx.py \
  --policy-path /path/to/pretrained_model \
  --onnx-out-dir outputs/trt/my_artifact/gr00t_onnx \
  --seq-len 568 \
  --device cuda
```

2) Build action-head engines (TensorRT Python API):

```bash
cd outputs/trt/my_artifact
ONNX_DIR=gr00t_onnx ENGINE_DIR=gr00t_engine_api_trt1013 \
VIDEO_VIEWS=2 MAX_BATCH=2 \
bash /data/cqy_workspace/flexible_lerobot/my_devs/groot_trt/build_engine.sh
```

3) Run robot inference using TRT action head only:

```bash
conda run -n lerobot_flex python my_devs/train/groot/run_groot_infer.py \
  --policy-path /path/to/pretrained_model \
  --backend tensorrt \
  --trt-engine-path outputs/trt/my_artifact/gr00t_engine_api_trt1013 \
  --trt-action-head-only true \
  --vit-dtype fp16 --llm-dtype fp16 --dit-dtype fp16
```

Notes:

- `--seq-len` should match the typical `eagle_attention_mask.shape[1]` you see at runtime.
- The action head engines still require `vlln_vl_self_attention.engine`, not just `DiT_fp16.engine`.

## Full Chain (ViT + LLM + Action Head)

The full-chain export (ViT + LLM ONNX) is more complex (Qwen3 ONNX export + optional quantization).
Use upstream `Isaac-GR00T` scripts as reference:

- `my_devs/docs/gr00t_trt/Isaac-GR00T-n1.5-release/deployment_scripts/export_onnx.py`
- `my_devs/docs/gr00t_trt/Isaac-GR00T-n1.5-release/deployment_scripts/build_engine.sh`
- `my_devs/docs/gr00t_trt/playbook_vla_trt_deployment.md`
- `my_devs/docs/gr00t_trt/lerobot_groot_tensorrt_integration_solution.md`

For repo-local workflows, you can use:

```bash
conda run -n lerobot_flex python my_devs/groot_trt/export_backbone_onnx.py \
  --policy-path /path/to/pretrained_model \
  --onnx-out-dir outputs/trt/my_artifact/gr00t_onnx \
  --seq-len 296 \
  --video-views 1
```

Then compare Torch vs ONNX consistency:

```bash
conda run -n lerobot_flex python my_devs/groot_trt/compare_torch_onnx.py \
  --policy-path /path/to/pretrained_model \
  --onnx-dir outputs/trt/my_artifact/gr00t_onnx \
  --seq-len 296 \
  --video-views 1
```

Then build TensorRT engines (Python API) and compare Torch vs TRT consistency:

```bash
cd outputs/trt/my_artifact
ONNX_DIR=gr00t_onnx ENGINE_DIR=gr00t_engine_api_trt1013 \
VIDEO_VIEWS=2 MAX_BATCH=2 \
bash /data/cqy_workspace/flexible_lerobot/my_devs/groot_trt/build_engine.sh

conda run -n lerobot_flex env TENSORRT_PY_DIR=/data/cqy_workspace/third_party/tensorrt_10_13_0_35 \
  python /data/cqy_workspace/flexible_lerobot/my_devs/groot_trt/compare_torch_trt.py \
    --policy-path /path/to/pretrained_model \
    --engine-dir outputs/trt/my_artifact/gr00t_engine_api_trt1013 \
    --seq-len 568 \
    --video-views 2
```
