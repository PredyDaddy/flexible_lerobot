# ACT TRT 工作收敛报告

> 更新日期：2026-03-11
> 收敛范围：`my_devs/act_trt`、`outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15`
> 关联文档：
> - `my_devs/docs/act_trt/act_trt开发工作报告.md`
> - `my_devs/docs/act_trt/act最终技术方案.md`
> - `my_devs/docs/act_trt/act开发Plan.md`

## 1. 本次收敛结论

当前可以收敛成一句话：

> ACT 的 ONNX/ORT 路线已经完成一致性闭环；ACT 的 TRT 路线已经完成 adapter 修复、独立复跑和独立审计，当前状态从“不能上机”更新为“可以考虑受控短时上机验证”，但还不能定义为“默认可长期上机”或“可替代现有生产 Torch 路线”。

这里的关键词必须区分清楚：

- “可以考虑上机”指的是：短时、可回退、有人值守、固定 checkpoint 和固定 engine 的受控验证。
- “不能直接宣称可长期上机”指的是：还没有做长时间稳定性验证、没有做完整端到端在线性能评估、也没有做 FP16 路线收敛。

## 2. 为什么之前 TRT 不能上机

在上一轮收敛前，TRT 不能上机，不是因为数值一致性不过，而是因为 adapter 语义边界还不够可靠。

当时的阻塞点有 4 个：

1. `ActTrtPolicyAdapter` 过于信任 `batch[OBS_IMAGES]`，可能接受 Torch 基线本来不接受的输入形态。
2. `config.image_features` 和 `export_metadata.camera_order_visual_keys` 的相机顺序没有做强一致性约束，存在静默相机顺序错误风险。
3. `temporal_ensemble_coeff != None` 时，adapter 本地没有再次守住 `n_action_steps == 1` 的不变量。
4. `run_act_trt_infer.py` 虽然是 mock wrapper，但报告里的入口追溯不够清楚，不利于审核。

这里最关键的不是“会不会报错”，而是“会不会静默错”。如果是静默错，即使离线 case 通过，也不适合直接上机。

## 3. 本轮已经完成的代码收敛

### 3.1 ONNX/ORT 路径

ONNX/ORT 路径已经具备完整离线一致性验证和 mock 推理能力，核心文件包括：

- `my_devs/act_trt/ort_policy.py`
- `my_devs/act_trt/verify_torch_onnx_policy.py`
- `my_devs/act_trt/run_act_ort_mock.py`
- `my_devs/act_trt/run_act_ort_infer.py`

这一路线当前仍然是最保守、最容易解释的问题定位基线。

### 3.2 TRT 路径

TRT 路径现在已经具备以下代码交付物：

- `my_devs/act_trt/trt_policy.py`
- `my_devs/act_trt/verify_torch_trt_policy.py`
- `my_devs/act_trt/run_act_trt_mock.py`
- `my_devs/act_trt/run_act_trt_infer.py`
- `my_devs/act_trt/run_act_trt_real.py`

本轮修复后的关键变化如下：

1. `trt_policy.py` 已改成 keyed visual features 为真值来源，不再把 `OBS_IMAGES` 当 source of truth。
2. `trt_policy.py` 会在 `config.image_features`、`export_metadata.camera_order_visual_keys`、checkpoint visual key 顺序不一致时直接报错。
3. `trt_policy.py` 在 adapter 本地重新守住了 temporal ensemble 的 `n_action_steps == 1` 不变量。
4. `run_act_trt_mock.py` / `run_act_trt_infer.py` 的报告新增了 `executed_module` 和 `cli_entrypoint`，wrapper 与实际执行模块已经可追溯。

## 4. 已收敛的关键证据

### 4.1 Torch vs ONNX policy 一致性

独立审计报告：

- `outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15/consistency_report_torch_onnx_policy_auditA.json`

关键结论：

- `predict_action_chunk`：`max_abs_diff=6.3896e-05`，`min_cosine_similarity=0.9999998808`
- `select_action(normalized)`：`max_abs_diff=6.2943e-05`，`min_cosine_similarity=0.9999998808`
- `select_action(real)`：`max_abs_diff=6.0654e-04`，`min_cosine_similarity=0.9999998808`

这说明 Torch 到 ONNX policy 级路径已经稳定闭环。

### 4.2 Torch vs ONNX vs TRT core forward 一致性

独立审计报告：

- `outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15/consistency_report_act_single_fp32_auditB.json`

关键结论：

- `Torch vs ONNX`：`max_abs_diff=3.4086e-06`，`min_cosine_similarity=0.9999998212`
- `Torch vs TRT`：`max_abs_diff=2.5034e-06`，`min_cosine_similarity=0.9999998808`
- `ONNX vs TRT`：`max_abs_diff=2.4699e-06`，`min_cosine_similarity=0.9999998808`

这说明 core model 的离线前向边界已经非常稳定。

### 4.3 Torch vs TRT policy 一致性

独立复跑报告：

- `outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15/consistency_report_torch_trt_policy_auditB.json`
- `outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15/consistency_report_torch_trt_policy_auditD.json`

`auditD` 是在 adapter 修复后重新独立复跑得到的正式结果，结论如下：

- `predict_action_chunk`
  - `max_abs_diff = 3.337860107421875e-05`
  - `max_rel_diff = 9.100837633013725e-05`
  - `min_cosine_similarity = 1.0`
- `select_action(normalized)`
  - `max_abs_diff = 3.24249267578125e-05`
  - `max_rel_diff = 1.387816882925108e-05`
  - `min_cosine_similarity = 0.9999998807907104`
- `select_action(real)`
  - `max_abs_diff = 0.00031280517578125`
  - `max_rel_diff = 0.00019859282474499196`
  - `min_cosine_similarity = 0.9999998807907104`

本次正式阈值为：

- `threshold_max_abs_diff_norm = 1e-4`
- `threshold_max_abs_diff_real = 1e-3`
- `threshold_max_rel_diff = 1e-2`
- `threshold_min_cosine_similarity = 0.999`

最终 `passed = true`，并且独立审核已经确认 `max_rel_diff` 和 `min_cosine_similarity` 确实参与了最终判定，不是只写进报告。

### 4.4 TRT mock 入口与追溯性

独立审计报告：

- `outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15/mock_infer_report_trt_auditD_mock.json`
- `outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15/mock_infer_report_trt_auditD_infer.json`

关键结论：

- `run_act_trt_mock.py` 和 `run_act_trt_infer.py` 当前都还是 mock-only。
- 审计中没有发现机器人、相机、串口或网络连接尝试。
- `executed_module` 可以明确指向真实执行逻辑模块。
- `cli_entrypoint` 可以明确指向用户实际调用的入口。
- 通过 `run_act_trt_infer.py` 调用时，`cli_entrypoint.is_alias_wrapper = true`，wrapper 与实际执行模块已清晰可区分。

### 4.5 新增 fail-fast 约束的负例验证

独立定向负例审核已经实际执行，3 个关键约束都按预期硬失败：

1. `temporal_ensemble_coeff != None` 且 `n_action_steps != 1`
   - 结果：`NotImplementedError`
2. `config.image_features` 与 `export_metadata.camera_order_visual_keys` 不一致
   - 结果：`ValueError`
3. batch 只有 `OBS_IMAGES`、缺少 keyed visual features
   - 结果：`KeyError`

这说明本轮新增的约束不是“写在代码里但跑不出来”，而是在实例化或运行时真的会拦截错误输入。

## 5. 当前对“是否可以上机”的正式判断

### 5.1 当前判断

当前判断已经可以更新为：

> TRT 路线可以考虑进入受控短时上机验证。

但是必须同时写清楚：

> TRT 路线还不能定义为默认生产路径，也不能直接替代现有 Torch 路线做长期无人值守运行。

### 5.2 为什么现在可以考虑上机

因为此前阻塞上机的两个关键条件，现在都已经满足：

1. 数值侧已经有独立复跑的 policy 级一致性结果，且 `abs / rel / cosine` 三类指标都过阈值。
2. 语义侧已经把之前最危险的静默错误点改成了 fail-fast，并且通过独立代码审阅和独立负例测试验证了它们真的生效。

换句话说，当前的主要风险已经不再是“离线看着对，实机偷偷错”，而是更普通的“尚未做足够多在线验证”。

## 6. 当前仍然没有收敛的内容

下面这些内容还没有收敛，因此不应把本轮结果解释成“TRT 全面完成”：

1. 还没有做长时间在线稳定性验证。
2. 还没有做完整端到端实时性能对比和抖动评估。
3. 还没有做 FP16 路线的一致性收敛。
4. 当前验证仍然绑定在这一组固定资产上：
   - 固定 checkpoint
   - 固定 `export_metadata.json`
   - 固定双相机顺序
   - 固定 static batch=1 engine
5. 还没有证明这套 adapter 对其他 checkpoint、其他 camera layout、其他导出批次同样无条件成立。

因此，当前更准确的阶段定义是：

> TRT 已完成“可受控上机验证”的收敛，不等于“可长期生产部署”的收敛。

## 7. 如何复现这项工作

这一节的目标不是解释原理，而是回答一个更实际的问题：

> 如果以后要从零复现这次工作，具体应该按什么顺序执行哪些脚本？

下面给出的是当前仓库下已经验证过的复现路径。

### 7.1 复现前提

复现时先满足下面几个前提：

1. 工作目录位于仓库根目录：`/data/cqy_workspace/flexible_lerobot`
2. 所有命令都使用 `lerobot_flex` 环境
3. TRT 相关步骤要求机器具备可用 CUDA 与 TensorRT Python 依赖
4. 当前已经验证通过的是 `FP32 + static batch=1 + 双相机 + 固定 checkpoint/export_metadata/engine` 路线

推荐统一先设置下面这组变量：

```bash
cd /data/cqy_workspace/flexible_lerobot

CHECKPOINT=outputs/train/20260305_190147_act_grasp_block_in_bin1_e15/checkpoints/025740/pretrained_model
DEPLOY_DIR=outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15
ONNX_PATH=$DEPLOY_DIR/act_single.onnx
EXPORT_METADATA=$DEPLOY_DIR/export_metadata.json
TRT_ENGINE_FP32=$DEPLOY_DIR/act_single_fp32.plan
```

如果不显式传 `--output-dir`，`export_act_onnx.py` 默认会把产物写到：

```text
outputs/deploy/act_trt/<run_name>
```

这个行为来自 `my_devs/act_trt/common.py` 里的 `resolve_deploy_dir()`。

### 7.2 脚本总览

`my_devs/act_trt` 目录里的脚本可以按用途分成 4 类：

1. 导出与构建
   - `export_act_onnx.py`
   - `build_act_trt_engine.py`
2. core forward 一致性对比
   - `verify_torch_onnx_trt.py`
3. policy 级一致性对比
   - `verify_torch_onnx_policy.py`
   - `verify_torch_trt_policy.py`
4. mock 推理与运行时检查
   - `run_act_ort_mock.py`
   - `run_act_ort_infer.py`
   - `run_act_trt_mock.py`
   - `run_act_trt_infer.py`
5. TRT 实机入口
   - `run_act_trt_real.py`

其中要特别强调：

- `run_act_ort_infer.py` 目前只是调用 `run_act_ort_mock.py` 的兼容入口。
- `run_act_trt_infer.py` 目前只是调用 `run_act_trt_mock.py` 的兼容入口。
- 这两个 `*_infer.py` 当前都不是正式机器人实机入口，而是 mock-only wrapper。
- 正式 TRT 实机入口现在是 `run_act_trt_real.py`，但它当前只完成了离线 `dry-run` 和 `mock-observation` 验证，还没有在本轮文档里追加独立实机审计结论。

### 7.3 第一步：导出 ONNX

使用脚本：

- `my_devs/act_trt/export_act_onnx.py`

推荐命令：

```bash
conda run -n lerobot_flex python my_devs/act_trt/export_act_onnx.py \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$DEPLOY_DIR" \
  --opset 17 \
  --device cpu \
  --verify \
  --simplify
```

这一步会调用参考导出脚本：

```text
my_devs/docs/act_trt/reference_docs/onnx_export_reference/export/export_single.py
```

导出完成后，至少应关注下面两个产物：

- `act_single.onnx`
- `export_metadata.json`

当前这条路线里，`--simplify` 是默认开启的，而且它不是“可有可无的优化”，而是当前经过验证的必要步骤。当前 ACT 模型如果不做 simplify，TRT 数值容易明显漂移。

### 7.4 第二步：构建 TRT engine

使用脚本：

- `my_devs/act_trt/build_act_trt_engine.py`

推荐命令：

```bash
conda run -n lerobot_flex python my_devs/act_trt/build_act_trt_engine.py \
  --onnx "$ONNX_PATH" \
  --metadata "$EXPORT_METADATA" \
  --engine "$TRT_ENGINE_FP32" \
  --precision fp32 \
  --workspace-gb 4 \
  --opt-level 3 \
  --timing-cache "$DEPLOY_DIR/act_single_fp32.tcache" \
  --report "$DEPLOY_DIR/trt_build_summary_fp32.json" \
  --device cuda:0
```

当前已经验证通过的是 `fp32` 路线。脚本虽然支持：

```text
--precision fp16
```

但 `FP16` 还没有在本轮工作里完成收敛，不应直接当成已验证路径。

构建成功后，主要产物包括：

- `act_single_fp32.plan`
- `act_single_fp32.tcache`
- `trt_build_summary_fp32.json`

### 7.5 第三步：做 core forward 三方一致性对比

这一步验证的是最底层的 core model forward：

```text
obs_state_norm + img0_norm + img1_norm
    -> core model
    -> actions_norm
```

使用脚本：

- `my_devs/act_trt/verify_torch_onnx_trt.py`

推荐命令：

```bash
conda run -n lerobot_flex python my_devs/act_trt/verify_torch_onnx_trt.py \
  --checkpoint "$CHECKPOINT" \
  --onnx "$ONNX_PATH" \
  --engine "$TRT_ENGINE_FP32" \
  --report "$DEPLOY_DIR/consistency_report_act_single_fp32.json" \
  --device cuda:0 \
  --random-cases 3 \
  --threshold-max-abs-diff 1e-4
```

这一步的报告会回答：

1. `Torch vs ONNX`
2. `Torch vs TRT`
3. `ONNX vs TRT`

是否在 `actions_norm` 层面数值一致。

当前正式通过的报告是：

- `outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15/consistency_report_act_single_fp32_auditB.json`

### 7.6 第四步：做 Torch vs ONNX policy 级一致性对比

这一步比 core forward 更接近真实推理行为，因为它会验证：

1. `predict_action_chunk`
2. `select_action` 的 queue 语义
3. postprocessor 后的 real action

使用脚本：

- `my_devs/act_trt/verify_torch_onnx_policy.py`

推荐命令：

```bash
conda run -n lerobot_flex python my_devs/act_trt/verify_torch_onnx_policy.py \
  --checkpoint "$CHECKPOINT" \
  --onnx "$ONNX_PATH" \
  --export-metadata "$EXPORT_METADATA" \
  --report "$DEPLOY_DIR/consistency_report_torch_onnx_policy.json" \
  --device cpu \
  --ort-provider cpu \
  --random-cases 4 \
  --sequence-length 12 \
  --threshold-max-abs-diff-norm 1e-4 \
  --threshold-max-abs-diff-real 1e-3
```

如果你只是想先把 ONNX 这一层完全对稳，这一步是最重要的正式检查项。

当前正式通过的独立审计报告是：

- `outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15/consistency_report_torch_onnx_policy_auditA.json`

### 7.7 第五步：做 Torch vs TRT policy 级一致性对比

这一步是判断 TRT adapter 是否已经达到“可考虑受控上机验证”的关键步骤。

使用脚本：

- `my_devs/act_trt/verify_torch_trt_policy.py`

推荐命令：

```bash
conda run -n lerobot_flex python my_devs/act_trt/verify_torch_trt_policy.py \
  --checkpoint "$CHECKPOINT" \
  --engine "$TRT_ENGINE_FP32" \
  --export-metadata "$EXPORT_METADATA" \
  --report "$DEPLOY_DIR/consistency_report_torch_trt_policy.json" \
  --device cuda:0 \
  --random-cases 4 \
  --sequence-length 12 \
  --threshold-max-abs-diff-norm 1e-4 \
  --threshold-max-abs-diff-real 1e-3 \
  --threshold-max-rel-diff 1e-2 \
  --threshold-min-cosine-similarity 0.999
```

这一步的报告会检查 3 层结果：

1. `predict_action_chunk_torch_vs_trt`
2. `select_action_norm_torch_vs_trt`
3. `select_action_real_torch_vs_trt`

而且当前脚本的最终 `passed` 判定已经同时使用：

- `max_abs_diff`
- `max_rel_diff`
- `min_cosine_similarity`

当前正式通过的独立复跑报告是：

- `outputs/deploy/act_trt/20260305_190147_act_grasp_block_in_bin1_e15/consistency_report_torch_trt_policy_auditD.json`

### 7.8 第六步：做 ORT mock 推理复跑

使用脚本：

- `my_devs/act_trt/run_act_ort_mock.py`

推荐命令：

```bash
conda run -n lerobot_flex python my_devs/act_trt/run_act_ort_mock.py \
  --checkpoint "$CHECKPOINT" \
  --onnx "$ONNX_PATH" \
  --export-metadata "$EXPORT_METADATA" \
  --device cpu \
  --ort-provider cpu \
  --steps 20 \
  --case random \
  --seed 0 \
  --compare-torch \
  --report "$DEPLOY_DIR/mock_infer_report_ort.json"
```

这一步不会接机器人，而是用 `make_mock_observation()` 生成 synthetic observation，主要用于：

1. 看 ORT policy 跑通没有
2. 看 `prepare / model / post / total` timing
3. 在 `--compare-torch` 打开时，附带记录 Torch vs ORT 的动作差异

`run_act_ort_infer.py` 当前只是兼容入口，等价于调用 `run_act_ort_mock.py`。

### 7.9 第七步：做 TRT mock 推理复跑

使用脚本：

- `my_devs/act_trt/run_act_trt_mock.py`

推荐命令：

```bash
conda run -n lerobot_flex python my_devs/act_trt/run_act_trt_mock.py \
  --checkpoint "$CHECKPOINT" \
  --engine "$TRT_ENGINE_FP32" \
  --export-metadata "$EXPORT_METADATA" \
  --device cuda:0 \
  --steps 20 \
  --case random \
  --seed 0 \
  --compare-torch \
  --report "$DEPLOY_DIR/mock_infer_report_trt.json"
```

如果你还想检查 alias wrapper 的追溯性，也可以跑：

```bash
conda run -n lerobot_flex python my_devs/act_trt/run_act_trt_infer.py \
  --checkpoint "$CHECKPOINT" \
  --engine "$TRT_ENGINE_FP32" \
  --export-metadata "$EXPORT_METADATA" \
  --device cuda:0 \
  --steps 20 \
  --case random \
  --seed 0 \
  --compare-torch \
  --report "$DEPLOY_DIR/mock_infer_report_trt_infer_alias.json"
```

这两条命令当前都不会接机器人，仍然是 mock-only。区别在于：

- `run_act_trt_mock.py` 是真实执行逻辑入口
- `run_act_trt_infer.py` 是兼容 wrapper

修复后的报告里会区分：

- `executed_module`
- `cli_entrypoint`

用来说明“实际逻辑是谁执行的”和“用户是从哪个 CLI 入口触发的”。

### 7.10 第八步：使用 TRT 实机入口

真正准备上机时，应该使用：

- `my_devs/act_trt/run_act_trt_real.py`

建议先走两级安全验证。

第一级：只解析配置，不加载 TRT、不连接机器人

```bash
conda run -n lerobot_flex python my_devs/act_trt/run_act_trt_real.py \
  --policy-path "$CHECKPOINT" \
  --engine "$TRT_ENGINE_FP32" \
  --export-metadata "$EXPORT_METADATA" \
  --policy-device cuda:0 \
  --dry-run true
```

第二级：连接同一套 TRT 实机脚本，但仍然不接机器人，改用 synthetic observation 做离线整链路验证

```bash
conda run -n lerobot_flex python my_devs/act_trt/run_act_trt_real.py \
  --policy-path "$CHECKPOINT" \
  --engine "$TRT_ENGINE_FP32" \
  --export-metadata "$EXPORT_METADATA" \
  --policy-device cuda:0 \
  --mock-observation true \
  --mock-case random \
  --seed 0 \
  --fps 30 \
  --max-steps 20 \
  --report "$DEPLOY_DIR/run_act_trt_real_report_offline.json"
```

真正上机时，再把 `--mock-observation true` 去掉。如果想先连机器人、读观测、跑推理，但仍然不发动作，可以再加：

```text
--mock-send true
```

这样主循环会连接机器人并完成推理，但不会执行 `robot.send_action(...)`。

### 7.11 一次完整复现的最小闭环

如果以后只想最小成本复现“这套工作已经被做通了”，建议按下面顺序执行：

1. `export_act_onnx.py`
2. `build_act_trt_engine.py`
3. `verify_torch_onnx_trt.py`
4. `verify_torch_onnx_policy.py`
5. `verify_torch_trt_policy.py`
6. `run_act_ort_mock.py`
7. `run_act_trt_mock.py`
8. `run_act_trt_real.py --dry-run true`
9. `run_act_trt_real.py --mock-observation true`

其中真正决定“这条链路现在是否可信”的不是 mock 脚本，而是这 3 份一致性报告：

- `consistency_report_act_single_fp32*.json`
- `consistency_report_torch_onnx_policy*.json`
- `consistency_report_torch_trt_policy*.json`

### 7.12 当前复现路径里的已验证边界

这份报告里给出的复现命令，只能支持下面这个结论：

> 在当前固定 checkpoint、固定 metadata、固定双相机顺序、固定 FP32 static-batch TRT engine 的前提下，Torch、ONNX、TRT 三层已经完成对齐，ORT 和 TRT 的 adapter/mock 路径都能跑通。

但它不等于下面这些更强结论：

- 任意 checkpoint 都已经验证
- FP16 已经验证
- 任意 camera layout 都已经验证
- 已经具备正式机器人实机入口
- 已经适合长期无人值守运行

## 8. 建议的上机策略

如果下一步要上机，建议采用下面的收敛顺序：

1. 先保留 ORT 路线作为最稳妥的回退路径。
2. TRT 只使用本次通过审计的这组固定资产，不要临时换 checkpoint、engine 或 metadata。
3. 首轮上机只做短时、低风险、有人值守验证，不做长时间连续运行。
4. 首轮上机必须保留快速切回 Torch 或 ORT 的能力。
5. 首轮上机应记录至少这些指标：
   - 单步 `prepare / model / post / total`
   - 首步冷启动耗时
   - 动作输出范围是否异常
   - 是否出现与离线不一致的相机或输入契约报错

如果上述短时上机验证稳定通过，再决定是否进入更长时间、更高频或更复杂任务的验证。

## 9. 最终收敛表述

本轮之后，最准确的收敛表述应该是：

> ONNX/ORT 路线已经完成一致性闭环并可作为稳定基线；TRT 路线已经完成 adapter 语义修复、mock 入口收敛、独立数值复跑和独立负例审计，当前可以进入受控短时上机验证阶段，但尚未收敛到长期生产部署阶段。
