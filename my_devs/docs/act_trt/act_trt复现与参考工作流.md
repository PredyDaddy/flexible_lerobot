# ACT TRT复现与参考工作流

本文档把当前仓库里 ACT 的导出、校验、构建、mock 验证和真机命令串成一条完整链路。所有命令都以当前仓库 `/data/cqy_workspace/flexible_lerobot` 为准，默认只调用 `my_devs/act_trt` 下的脚本；环境统一使用 `lerobot_flex`。

## 1. 适用范围

- 目标是把一个 ACT `pretrained_model` checkpoint 走完 `Torch -> ONNX -> TensorRT` 对齐流程。
- ONNX 导出边界是 ACT core forward，不包含 policy preprocessor 和 postprocessor。
- 真机上机时：
  - ONNX 路径使用新增入口 `my_devs/act_trt/lerobot_record_act_onnx.py`
  - TensorRT 路径使用现有入口 `my_devs/act_trt/run_act_trt_real.py`

## 2. 统一约定

先在 shell 里设置变量，后续命令都直接复用：

```bash
export REPO=/data/cqy_workspace/flexible_lerobot
export PY="conda run --live-stream -n lerobot_flex python"

# 必须替换成你这次要复现的 ACT checkpoint pretrained_model 目录
export CHECKPOINT=$REPO/outputs/train/<run_name>/checkpoints/last/pretrained_model

# 推荐显式指定导出目录，避免多次实验互相覆盖
export EXPORT_DIR=$REPO/outputs/deploy/act_trt/<run_name>
export ONNX=$EXPORT_DIR/act_single.onnx
export METADATA=$EXPORT_DIR/export_metadata.json
export ENGINE_FP32=$EXPORT_DIR/act_single_fp32.plan
export TRT_REPORT=$EXPORT_DIR/trt_build_summary_fp32.json
export VERIFY_POLICY_REPORT=$EXPORT_DIR/consistency_report_torch_onnx_policy.json
export VERIFY_TRT_REPORT=$EXPORT_DIR/consistency_report_act_single_fp32.json
export ORT_MOCK_REPORT=$EXPORT_DIR/mock_infer_report_ort.json
export TRT_MOCK_REPORT=$EXPORT_DIR/mock_infer_report_trt.json
```

关键路径约束：

- `CHECKPOINT` 必须指向 `pretrained_model`，不能只指到 `checkpoints/last`。
- `ONNX` 和 `METADATA` 建议始终成对使用。
- 这套脚本默认约定 ONNX I/O 名字是：
  - `obs_state_norm`
  - `img0_norm`
  - `img1_norm`
  - `actions_norm`

## 3. 产物约定

一次完整流程通常会生成这些文件：

- `act_single.onnx`
- `export_metadata.json`
- `act_single_fp32.plan`
- `act_single_fp32.tcache`
- `trt_build_summary_fp32.json`
- `consistency_report_torch_onnx_policy.json`
- `consistency_report_act_single_fp32.json`
- `mock_infer_report_ort.json`
- `mock_infer_report_trt.json`

其中：

- `verify_torch_onnx_policy.py` 验证的是完整 policy 运行边界：`preprocess -> ONNX adapter -> postprocess`
- `verify_torch_onnx_trt.py` 验证的是 core tensor 边界：`Torch core -> ONNX core -> TRT core`

## 4. 不上机验证命令

这一节默认不连机器人，不依赖串口和相机，适合先把导出和数值对齐问题收敛。

### 4.1 checkpoint 导出为 ONNX

```bash
$PY $REPO/my_devs/act_trt/export_act_onnx.py \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$EXPORT_DIR" \
  --opset 17 \
  --device cpu \
  --verify \
  --simplify
```

预期输出：

- `act_single.onnx`
- `export_metadata.json`

如果这一步失败，先不要进入 TRT build。先把 ONNX 导出和 ONNXRuntime verify 解决。

### 4.2 Torch vs ONNX policy 边界校验

这一步验证完整 policy 边界，而不是只看 core 输出。

```bash
$PY $REPO/my_devs/act_trt/verify_torch_onnx_policy.py \
  --checkpoint "$CHECKPOINT" \
  --onnx "$ONNX" \
  --export-metadata "$METADATA" \
  --device cpu \
  --ort-provider cpu \
  --random-cases 4 \
  --sequence-length 12 \
  --report "$VERIFY_POLICY_REPORT"
```

关注结论：

- `passed=true`
- `select_action_real_torch_vs_onnx.max_abs_diff` 是否在可接受阈值内

### 4.3 构建 TensorRT engine

```bash
$PY $REPO/my_devs/act_trt/build_act_trt_engine.py \
  --onnx "$ONNX" \
  --metadata "$METADATA" \
  --engine "$ENGINE_FP32" \
  --precision fp32 \
  --workspace-gb 4 \
  --opt-level 3 \
  --report "$TRT_REPORT" \
  --device cuda:0
```

说明：

- 当前 builder 走的是静态 shape 路线，不要拿动态 batch ONNX 直接构建。
- 为了做对齐验证，优先先用 `fp32`。

### 4.4 Torch vs ONNX vs TRT 三方对齐

```bash
$PY $REPO/my_devs/act_trt/verify_torch_onnx_trt.py \
  --checkpoint "$CHECKPOINT" \
  --onnx "$ONNX" \
  --engine "$ENGINE_FP32" \
  --device cuda:0 \
  --threshold-max-abs-diff 1e-4 \
  --report "$VERIFY_TRT_REPORT"
```

关注这三项：

- `summary.torch_vs_onnx.max_abs_diff`
- `summary.torch_vs_trt.max_abs_diff`
- `summary.onnx_vs_trt.max_abs_diff`

只要 `build` 成功不代表数值正确，必须看这份 report。

### 4.5 ONNX mock 验证

这一步不连机器人，只跑合成观测的完整 ONNX policy 流程。

```bash
$PY $REPO/my_devs/act_trt/run_act_ort_mock.py \
  --checkpoint "$CHECKPOINT" \
  --onnx "$ONNX" \
  --export-metadata "$METADATA" \
  --device cpu \
  --ort-provider cpu \
  --steps 20 \
  --case random \
  --compare-torch \
  --report "$ORT_MOCK_REPORT"
```

如果你只想快速 smoke test，可改成：

```bash
$PY $REPO/my_devs/act_trt/run_act_ort_mock.py \
  --checkpoint "$CHECKPOINT" \
  --onnx "$ONNX" \
  --export-metadata "$METADATA" \
  --device cpu \
  --ort-provider cpu \
  --steps 5 \
  --case zeros \
  --compare-torch=false
```

### 4.6 TensorRT mock 验证

```bash
$PY $REPO/my_devs/act_trt/run_act_trt_mock.py \
  --checkpoint "$CHECKPOINT" \
  --engine "$ENGINE_FP32" \
  --export-metadata "$METADATA" \
  --device cuda:0 \
  --steps 20 \
  --case random \
  --compare-torch \
  --report "$TRT_MOCK_REPORT"
```

建议顺序：

1. 先跑 `verify_torch_onnx_policy.py`
2. 再跑 `build_act_trt_engine.py`
3. 再跑 `verify_torch_onnx_trt.py`
4. 最后再看 `run_act_ort_mock.py` 和 `run_act_trt_mock.py` 的行为是否稳定

## 5. 真机上机命令

这一节默认已经确认串口、相机、标定目录和任务描述都可用。这里明确区分 ONNX 上机和 TRT 上机。

### 5.1 ONNX record 入口 dry-run

新增入口脚本：

```text
/data/cqy_workspace/flexible_lerobot/my_devs/act_trt/lerobot_record_act_onnx.py
```

`dry_run` 会加载 checkpoint config 和 ONNXRuntime session，但不会创建 dataset、不会连接 robot、不会进入 record loop。建议每次真机前先跑一次：

```bash
$PY $REPO/my_devs/act_trt/lerobot_record_act_onnx.py \
  --robot.type=so101_follower \
  --robot.id=my_so101 \
  --robot.port=/dev/ttyACM0 \
  --robot.calibration_dir=/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower \
  --robot.cameras='{"top":{"type":"opencv","index_or_path":4,"width":640,"height":480,"fps":30},"wrist":{"type":"opencv","index_or_path":6,"width":640,"height":480,"fps":30}}' \
  --dataset.repo_id=<hf_user>/eval_<task_name> \
  --dataset.root=$REPO/outputs/eval/<task_name> \
  --dataset.push_to_hub=false \
  --dataset.num_episodes=1 \
  --dataset.episode_time_s=15 \
  --dataset.reset_time_s=10 \
  --dataset.single_task="Put the block in the bin" \
  --policy.path="$CHECKPOINT" \
  --policy.device=cpu \
  --onnx.path="$ONNX" \
  --onnx.metadata_path="$METADATA" \
  --onnx.provider=cpu \
  --dry_run=true
```

如果 dry-run 都过不去，先不要上机器人。优先检查：

- `--policy.path` 是否真的是 `pretrained_model`
- `--onnx.path` / `--onnx.metadata_path` 是否与当前 checkpoint 配套
- ORT provider 是否和当前环境一致

### 5.2 ONNX 真机 record_loop 命令

```bash
$PY $REPO/my_devs/act_trt/lerobot_record_act_onnx.py \
  --robot.type=so101_follower \
  --robot.id=my_so101 \
  --robot.port=/dev/ttyACM0 \
  --robot.calibration_dir=/home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower \
  --robot.cameras='{"top":{"type":"opencv","index_or_path":4,"width":640,"height":480,"fps":30},"wrist":{"type":"opencv","index_or_path":6,"width":640,"height":480,"fps":30}}' \
  --dataset.repo_id=<hf_user>/eval_<task_name> \
  --dataset.root=$REPO/outputs/eval/<task_name> \
  --dataset.push_to_hub=false \
  --dataset.num_episodes=5 \
  --dataset.episode_time_s=15 \
  --dataset.reset_time_s=10 \
  --dataset.single_task="Put the block in the bin" \
  --policy.path="$CHECKPOINT" \
  --policy.device=cuda \
  --onnx.path="$ONNX" \
  --onnx.metadata_path="$METADATA" \
  --onnx.provider=auto
```

这条命令的运行边界是：

```text
robot observation
  -> lerobot preprocessor
  -> ONNXRuntime ACT core
  -> lerobot postprocessor
  -> robot action
  -> dataset recording
```

### 5.3 TensorRT 真机 dry-run

现有 TRT 真机入口本身已经带 dry-run：

```bash
$PY $REPO/my_devs/act_trt/run_act_trt_real.py \
  --robot-id my_so101 \
  --robot-type so101_follower \
  --calib-dir /home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower \
  --robot-port /dev/ttyACM0 \
  --top-cam-index 4 \
  --wrist-cam-index 6 \
  --img-width 640 \
  --img-height 480 \
  --fps 30 \
  --policy-path "$CHECKPOINT" \
  --engine "$ENGINE_FP32" \
  --export-metadata "$METADATA" \
  --policy-device cuda:0 \
  --task "Put the block in the bin" \
  --dry-run true
```

### 5.4 TensorRT 真机执行命令

```bash
$PY $REPO/my_devs/act_trt/run_act_trt_real.py \
  --robot-id my_so101 \
  --robot-type so101_follower \
  --calib-dir /home/cqy/.cache/huggingface/lerobot/calibration/robots/so101_follower \
  --robot-port /dev/ttyACM0 \
  --top-cam-index 4 \
  --wrist-cam-index 6 \
  --img-width 640 \
  --img-height 480 \
  --fps 30 \
  --policy-path "$CHECKPOINT" \
  --engine "$ENGINE_FP32" \
  --export-metadata "$METADATA" \
  --policy-device cuda:0 \
  --task "Put the block in the bin" \
  --run-time-s 30 \
  --max-steps 0 \
  --log-interval 30
```

如果想先连 robot 但不发动作，可加：

```text
--mock-send true
```

如果想在真机 loop 中同时对比 Torch，可加：

```text
--compare-torch true --torch-policy-device cpu
```

## 6. 推荐复现顺序

推荐严格按这个顺序来，不要跳步骤：

1. `export_act_onnx.py`
2. `verify_torch_onnx_policy.py`
3. `build_act_trt_engine.py`
4. `verify_torch_onnx_trt.py`
5. `run_act_ort_mock.py`
6. `run_act_trt_mock.py`
7. `lerobot_record_act_onnx.py --dry_run=true`
8. `run_act_trt_real.py --dry-run true`
9. ONNX 或 TRT 真机命令

## 7. 常见问题

### 7.1 `--policy.path` 和 `--policy-path` 不是同一个参数位置

- `lerobot_record_act_onnx.py` 用的是 `--policy.path`
- `run_act_trt_real.py` 用的是 `--policy-path`

两者都必须指到 `pretrained_model`。

### 7.2 dataset root 已经存在但内容不完整

如果之前失败过，`dataset.root` 里可能只留下一个空壳目录。第一次重跑时建议：

- 换一个新的 `dataset.root`
- 或者确认旧目录不是残留 skeleton，再决定是否 `--resume=true`

### 7.3 ONNX 和 TRT 都能加载，但数值还是不对

优先按边界排查：

1. 看 `verify_torch_onnx_policy.py`
2. 再看 `verify_torch_onnx_trt.py`
3. 最后才怀疑真机 runtime 接线问题

不要因为 `.plan` 已经 build 成功，就默认 TRT 是对的。
