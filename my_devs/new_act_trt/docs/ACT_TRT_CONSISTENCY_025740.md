# ACT TRT Consistency Report For Checkpoint 025740

日期：2026-03-12  
环境：`conda run -n lerobot_flex`  
checkpoint：`outputs/train/20260305_190147_act_grasp_block_in_bin1_e15/checkpoints/last/pretrained_model`  
解析后的 step：`025740`

## 执行命令

```bash
conda run -n lerobot_flex python my_devs/new_act_trt/scripts/export_act_checkpoint_engine.py \
  --checkpoint outputs/train/20260305_190147_act_grasp_block_in_bin1_e15/checkpoints/last/pretrained_model \
  --device cpu \
  --trt-device cuda:0 \
  --task "Put the block in the bin" \
  --robot-type so101_follower
```

## 产物

- ONNX：`outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15/025740/act_single.onnx`
- TensorRT engine：`outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15/025740/act_single_fp32.plan`
- Metadata：`outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15/025740/export_metadata.json`
- Build report：`outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15/025740/trt_build_summary_fp32.json`
- Core consistency report：`outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15/025740/consistency_report_act_single_fp32.json`
- Policy consistency report：`outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15/025740/consistency_report_act_single_fp32_policy.json`
- Suite report：`outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15/025740/consistency_suite_act_single_fp32.json`

## Engine I/O

- `obs_state_norm`: `(1, 6)`
- `img0_norm`: `(1, 3, 480, 640)`
- `img1_norm`: `(1, 3, 480, 640)`
- `actions_norm`: `(1, 100, 6)`

## 核心一致性结果

- `torch vs onnx`: `max_abs_diff = 3.4086406230926514e-06`
- `torch vs trt`: `max_abs_diff = 2.3543834686279297e-06`
- `onnx vs trt`: `max_abs_diff = 2.6188790798187256e-06`

## 完整 policy 一致性结果

- `predict_action_chunk torch vs trt`: `max_abs_diff = 3.62396240234375e-05`
- `select_action_norm torch vs trt`: `max_abs_diff = 3.528594970703125e-05`
- `select_action_real torch vs trt`: `max_abs_diff = 0.000339508056640625`

## 结论

- 本次 `025740` checkpoint 的 ACT core 在 Torch / ONNX / TRT 三条路径上通过一致性验证。
- 在保留原始 preprocessor / postprocessor / queue 逻辑的前提下，TRT adapter 与 torch policy 的完整动作输出也通过一致性验证。
- `my_devs/train/act/so101/run_act_infer.py` 已可通过 `--policy-backend trt` 切换到 TRT backend。
