# ACT TRT Workspace

这个目录是当前仓库可复现的 ACT TensorRT 工作区，目的是在不依赖 `my_devs/act_trt` 历史脚本的前提下，完成以下几件事：

- 从 ACT checkpoint 导出 `policy.model` 到 ONNX
- 从 ONNX 构建静态 TensorRT engine
- 做 `torch -> onnx -> trt` 核心一致性验证
- 做 `torch policy -> trt policy adapter` 的完整推理一致性验证
- 给 `my_devs/train/act/so101/run_act_infer.py` 提供可复用的 TRT backend

## 目录

- `scripts/act_trt_paths.py`
  统一解析仓库根目录、默认 checkpoint、默认部署输出目录。
- `scripts/act_model_utils.py`
  checkpoint、processor、mock case、对比指标、公用路径函数。
- `scripts/export_act_onnx.py`
  导出 ACT core ONNX，并写 `export_metadata.json`。
- `scripts/build_act_trt_engine.py`
  构建 TensorRT engine，并输出 engine I/O 报告。
- `scripts/verify_act_torch_onnx_trt.py`
  验证 ACT core 在 Torch / ONNX / TRT 三条路径上的数值一致性。
- `scripts/verify_act_torch_trt_policy.py`
  验证完整 policy 路径在 Torch / TRT 下的 `predict_action_chunk`、`select_action` 和 postprocess 后动作一致性。
- `scripts/trt_act_policy.py`
  TRT policy adapter，可直接接到现有 `predict_action(...)` 流程。
- `scripts/run_trt_act.py`
  TRT 推理专用启动脚本，直接复用 `run_act_infer.py` 的控制循环并强制使用 TRT backend。
- `scripts/export_act_checkpoint_engine.py`
  一条命令完成 export / build / verify-core / verify-policy。
- `reference/act_trt/`
  参考实现，只读保留，不作为当前工作区的运行入口。

## 标准命令

所有命令都必须在 `lerobot_flex` 环境执行。

```bash
conda run -n lerobot_flex python my_devs/new_act_trt/scripts/export_act_checkpoint_engine.py \
  --checkpoint outputs/train/20260305_190147_act_grasp_block_in_bin1_e15/checkpoints/last/pretrained_model \
  --device cpu \
  --trt-device cuda:0 \
  --task "Put the block in the bin" \
  --robot-type so101_follower
```

默认输出目录：

```text
outputs/deploy/act_trt/<run_name>/<checkpoint_step>/
```

对当前默认 checkpoint 来说，是：

```text
outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15/025740/
```

## 与 `run_act_infer.py` 的关系

`my_devs/train/act/so101/run_act_infer.py` 已支持：

- `--policy-backend torch`
- `--policy-backend trt`

当选择 `trt` 时，脚本会：

- 继续使用 checkpoint 里的原始 preprocessor / postprocessor
- 自动解析对应的 `act_single_fp32.plan` 和 `export_metadata.json`
- 用 `TrtActPolicyAdapter` 替换原来的 torch `ACTPolicy`
- 保留现有 `predict_action(...)`、动作队列和机器人控制循环

示例：

```bash
conda run -n lerobot_flex python my_devs/train/act/so101/run_act_infer.py \
  --policy-backend trt \
  --policy-path outputs/train/20260305_190147_act_grasp_block_in_bin1_e15/checkpoints/last/pretrained_model \
  --task "Put the block in the bin"
```

也可以直接用 TRT 专用入口：

```bash
conda run -n lerobot_flex python my_devs/new_act_trt/scripts/run_trt_act.py \
  --policy-path outputs/train/20260305_190147_act_grasp_block_in_bin1_e15/checkpoints/last/pretrained_model \
  --task "Put the block in the bin"
```
