# ACT TensorRT 开发 Plan

> 文档日期：2026-03-06  
> 适用范围：`my_devs/train/act/so101/run_act_infer.py` 当前 ACT 实机推理链路的 TensorRT 加速开发计划  
> 关联方案：`my_devs/docs/act_trt/act最终技术方案.md`

---

## 1. 文档目的

本文不是技术选型文档，而是**开发执行 Plan**。

它回答的是下面几个问题：

- 基于最终技术方案，接下来到底要开发哪些代码；
- 每个阶段的代码目标、输入输出、依赖关系是什么；
- 哪些验证必须做，尤其是**一致性验证**；
- 哪些产物是“没有完成就不能进入下一阶段”的硬门槛。

本文默认技术路线已经确定：

- 保留现有 LeRobot 推理链路；
- 只替换 ACT 核心网络前向为 ONNX / TensorRT；
- queue / temporal ensemble / preprocessor / postprocessor 保持在 Python 侧；
- 第一版只做静态 `batch=1`；
- 第一版必须优先保证正确性与一致性，再追求性能。

---

## 2. 开发目标

本次开发的最终目标不是“做出一个能跑的 TRT demo”，而是交付一个**可验证、可复现、可逐步上线**的 ACT TensorRT 推理链路。

本次开发必须同时满足三类目标：

### 2.1 导出目标

能够把当前 ACT checkpoint 的核心网络稳定导出为 ONNX，并沉淀 metadata 与导出验证结果。

### 2.2 引擎目标

能够把 ONNX 稳定构建为 TensorRT engine，并有完整的 build summary、timing cache 与精度模式信息。

### 2.3 一致性目标

必须完成以下一致性闭环：

1. Torch vs ONNX
2. ONNX vs TRT-FP32
3. Torch policy vs TRT adapter
4. TRT-FP32 vs TRT-FP16
5. 在线短时实机行为验证

其中 **一致性工作是本计划中的强制项，不是可选项**。

---

## 3. 开发范围

### 3.1 本次会开发的内容

本次开发计划包括：

- ONNX 导出调用与产物规范化；
- TensorRT engine 构建脚本；
- TensorRT runtime 封装；
- ACT TRT policy adapter；
- 独立 TRT 推理脚本；
- 一致性验证脚本与验证流程；
- 性能统计与日志；
- 必要的开发文档与使用说明。

### 3.2 本次不会开发的内容

本次开发明确不包括：

- 将 `ACTPolicy.select_action()` 整体导出到 ONNX / TRT；
- 将 `policy_preprocessor` / `policy_postprocessor` 融合进 ONNX / TRT；
- 动态 batch；
- 零拷贝 Torch CUDA tensor 直接喂 TRT；
- INT8 校准与量化部署；
- 对现有 Torch 推理脚本做高侵入式重构。

---

## 4. 开发原则

### 4.1 一致性优先于性能

如果某个优化会显著提高复杂度或降低验证透明度，那么第一版不做。

### 4.2 分层闭环优先于端到端冒进

必须按“导出 -> 构建 -> runtime -> adapter -> 实机”的顺序推进，不允许跳层。

### 4.3 保留可回退路径

现有 `run_act_infer.py` 作为 Torch 基线保留，不直接替换；TRT 路线采用独立脚本接入。

### 4.4 所有开发与测试使用 `lerobot_flex`

仓库规范要求所有开发代码与测试运行必须使用 conda env：`lerobot_flex`。

---

## 5. 计划交付物

### 5.1 文档

- `my_devs/docs/act_trt/act最终技术方案.md`
- `my_devs/docs/act_trt/act开发Plan.md`
- 后续如需要，可补充运行说明 / 故障排查文档

### 5.2 代码交付物

计划开发以下代码文件：

- `my_devs/train/act/so101/build_act_trt_engine.py`
- `my_devs/train/act/so101/act_trt_runtime.py`
- `my_devs/train/act/so101/act_trt_policy.py`
- `my_devs/train/act/so101/run_act_trt_infer.py`
- `my_devs/train/act/so101/verify_act_torch_onnx.py`
- `my_devs/train/act/so101/verify_act_onnx_trt.py`
- `my_devs/train/act/so101/verify_act_policy_consistency.py`

说明：

- 如果后续发现验证脚本适合合并，也可以收敛为更少文件；
- 但从开发计划角度，**导出验证、engine 验证、policy 语义验证必须有独立入口**。

### 5.3 部署产物

计划生成以下部署产物：

- `outputs/deploy/act_trt/<run_name>/act_single.onnx`
- `outputs/deploy/act_trt/<run_name>/act_single.onnx.data`（可选）
- `outputs/deploy/act_trt/<run_name>/export_metadata.json`
- `outputs/deploy/act_trt/<run_name>/act_single_fp32.plan`
- `outputs/deploy/act_trt/<run_name>/act_single_fp16.plan`
- `outputs/deploy/act_trt/<run_name>/build_cache.tcache`
- `outputs/deploy/act_trt/<run_name>/trt_build_summary.json`
- `outputs/deploy/act_trt/<run_name>/consistency_report_*.json`

---

## 6. 开发阶段与任务拆解

### Phase 0：准备与基线确认

目标：在开始 TRT 开发前，明确基线与环境条件。

任务：

- 确认 `lerobot_flex` 环境可用；
- 确认目标 checkpoint 的 `config.json`、pre/post processor stats、图像 key 顺序；
- 给现有 Torch 基线补充必要的 timing 统计口径；
- 固化当前默认 checkpoint 的输入输出事实。

建议输出：

- 当前 checkpoint 事实表；
- 基线 timing 记录；
- 一份简短的环境检查记录。

进入下一阶段前的门槛：

- 已确认 checkpoint 的 camera order、state/action shape、chunk_size、n_action_steps；
- 已确认后续命令统一在 `lerobot_flex` 下执行。

---

### Phase 1：模型导出层

目标：稳定导出 ONNX，并确认 ONNX 与 Torch 在核心前向上的一致性。

开发任务：

- 复用或包装现有参考导出脚本；
- 规范化导出产物目录；
- 生成 `export_metadata.json`；
- 固化 camera order、input/output names、shape、checkpoint path；
- 增加 Torch vs ONNX 的导出后验证入口。

建议代码：

- 优先复用：`my_devs/docs/act_trt/reference_docs/onnx_export_reference/export/export_single.py`
- 如需项目内统一入口，可新增轻量 wrapper，但不建议重复实现一整套导出逻辑。

必须验证：

- ONNX 文件能加载；
- ORT 可以跑通；
- 与 Torch 前向的 `actions_norm` 对齐。

验收标准：

- `act_single.onnx` 生成成功；
- `export_metadata.json` 信息完整；
- Torch vs ONNX：`max_abs_diff <= 1e-4`；
- 若验证失败，不允许进入 engine 构建阶段。

---

### Phase 2：引擎构建层

目标：将 ONNX 构建为可复用、可审计的 TensorRT engine。

开发任务：

- 先用 `trtexec` 验证 ONNX 是否可被 TensorRT 接受；
- 实现 `build_act_trt_engine.py`；
- 支持输出 FP32 engine；
- 在 FP32 对齐稳定后支持输出 FP16 engine；
- 保存 timing cache；
- 记录 build summary。

建议脚本职责：

- 输入：ONNX 路径、metadata 路径、精度模式、输出路径；
- 检查输入输出 tensor 名称与 shape；
- 静态 `batch=1` build；
- 生成 `.plan`、`.tcache`、`trt_build_summary.json`。

必须验证：

- `trtexec` 可以完成 parse / build；
- Python builder 输出的 engine 可被 runtime 正常加载；
- FP32 engine 先完成正确性对齐。

验收标准：

- `act_single_fp32.plan` 生成成功；
- `trt_build_summary.json` 记录完整；
- engine tensor names / shapes / dtypes 与 metadata 一致；
- FP32 engine 构建通过后，才允许进入 runtime 与 adapter 开发；
- FP16 engine 必须在 FP32 路线验证通过后再构建。

---

### Phase 3：TRT runtime 层

目标：提供稳定、清晰、按名称驱动的 TensorRT 推理封装。

开发任务：

- 实现 `act_trt_runtime.py`；
- 封装 engine load、context 初始化、buffer 管理、推理执行；
- 支持按 tensor name 设置输入输出；
- 做 shape / dtype / device 自检；
- 返回 `actions_norm`。

第一版设计要求：

- 输入路径使用 CPU tensor / numpy；
- TRT runtime 内部负责 H2D / D2H；
- 第一版只支持静态 `batch=1`；
- 不做 dynamic shape 扩容逻辑；
- 不做零拷贝。

必须验证：

- 对同一份 normalized 输入，ONNX vs TRT-FP32 输出对齐；
- engine IO 自检可明确报错常见低级问题。

验收标准：

- runtime 可以稳定执行单次 forward；
- 输出 `actions_norm` shape 为 `(1, chunk_size, action_dim)`；
- ONNX vs TRT-FP32 一致性通过；
- 若该阶段无法稳定对齐，不允许继续做 adapter。

---

### Phase 4：ACT policy adapter 层

目标：在 Python 侧复刻 ACT 的语义边界，把 TRT 输出接回现有 processor 链路。

开发任务：

- 实现 `act_trt_policy.py`；
- 定义 `ActTrtPolicyAdapter`；
- 从 `export_metadata.json` 读取 camera order；
- 将 observation 中的 image features 映射为 `img0_norm` / `img1_norm`；
- 调用 runtime 获取 `actions_norm`；
- 复刻 queue 语义；
- 兼容 temporal ensemble。

必须遵守：

- adapter 只负责 TRT 前向与 ACT 语义复刻；
- 不负责 action 反归一化；
- 不允许在代码里手写相机顺序；
- 若启用 temporal ensemble，必须强制 `n_action_steps == 1`。

必须验证：

- 用同一帧或同一段 observation 序列，对齐 Torch policy 与 TRT adapter；
- 既要验证单次 `predict_action_chunk`，也要验证 `select_action` 语义。

验收标准：

- `ActTrtPolicyAdapter.select_action(...)` 的接口行为可替代 Torch policy；
- Torch policy vs TRT adapter 差异在可接受范围内；
- 对齐报告输出完整。

---

### Phase 5：接入层

目标：在不破坏现有 Torch 推理脚本的前提下，增加独立 TRT 推理入口。

开发任务：

- 实现 `run_act_trt_infer.py`；
- 尽量复用 `run_act_infer.py` 的主循环结构与 CLI 语义；
- 加载 config、pre/post processor、stats、metadata、engine；
- 不加载 Torch policy 权重；
- 增加 timing 统计；
- 保留 `policy_n_action_steps` 与 `policy_temporal_ensemble_coeff` 的 runtime override 能力。

必须验证：

- TRT 推理脚本能跑通短时闭环；
- 与现有 Torch 版主循环结构保持一致；
- timing 口径清晰。

验收标准：

- 能在目标设备上稳定跑通短时推理；
- 日志中能区分 `t_obs`、`t_prepare`、`t_model`、`t_post`、`t_total`；
- 不引入对 Torch policy 权重的隐式依赖。

---

### Phase 6：一致性验证层

目标：把一致性验证做成正式交付物，而不是临时手工操作。

这是本计划中最关键的阶段之一。

开发任务：

- 实现 `verify_act_torch_onnx.py`；
- 实现 `verify_act_onnx_trt.py`；
- 实现 `verify_act_policy_consistency.py`；
- 输出结构化验证结果（建议 JSON + 控制台摘要）；
- 支持固定输入样本或离线 observation 回放。

必须覆盖的一致性项：

#### 6.1 Torch vs ONNX

对齐对象：

- `actions_norm`

要求：

- 输入完全相同；
- 使用同一份 normalized state / images；
- 输出 `max_abs_diff`、`mean_abs_diff`。

#### 6.2 ONNX vs TRT-FP32

对齐对象：

- `actions_norm`

要求：

- engine input/output names 正确；
- shape、dtype 完整检查；
- 输出误差指标。

#### 6.3 TRT-FP32 vs TRT-FP16

对齐对象：

- `actions_norm`

要求：

- FP16 只在 FP32 成功后参与对比；
- 输出误差指标；
- 如果 FP16 误差明显异常，要能快速回退并定位。

#### 6.4 Torch policy vs TRT adapter

对齐对象：

- `predict_action_chunk` 输出；
- `select_action` 逐步输出；
- 必要时加 observation 序列回放。

要求：

- 验证 queue 语义；
- 验证 temporal ensemble 语义；
- 输出整体差异统计。

#### 6.5 短时在线实机验证

对齐对象：

- 行为稳定性；
- 基本动作合理性；
- timing 与控制频率。

要求：

- 先保持 `n_action_steps=100`；
- 之后逐步降低到 `32 / 16 / 8`；
- 每一步都要保留日志和结论。

硬门槛：

- 一致性验证不通过，不允许宣称该版本可上线；
- FP16 不通过，不影响 FP32 版本作为阶段性交付；
- adapter 语义不通过，不允许接实机。

---

### Phase 7：性能评估与参数下调

目标：在一致性成立后，验证 TRT 对重规划频率提升的真实价值。

开发任务：

- 在 TRT 推理脚本中固化 timing 指标统计；
- 对不同 `n_action_steps` 配置做实测；
- 给出 `100 -> 32 -> 16 -> 8` 的逐级验证结论；
- 区分“模型前向快了”和“整条控制链真的变好了”。

必须输出：

- p50 / p95 延迟；
- 不同 `n_action_steps` 下的控制频率和稳定性记录；
- 哪个配置是当前硬件下最实用的部署默认值。

验收标准：

- 至少完成一个合理的 `n_action_steps` 下调结论；
- 证明 TRT 的收益不只是理论上的，而是能支撑更频繁的重规划。

---

## 7. 开发顺序

最终建议按以下顺序编码：

1. 先补齐 Phase 0 的基线与环境确认；
2. 先打通 Phase 1 导出；
3. 再完成 Phase 2 的 FP32 engine build；
4. 再做 Phase 3 的 runtime；
5. 再做 Phase 6 中的 ONNX / TRT 一致性验证；
6. 再做 Phase 4 的 adapter；
7. 再做 Phase 6 中的 policy 语义一致性验证；
8. 再做 Phase 5 的实机推理接入；
9. 最后做 Phase 7 的性能评估与 `n_action_steps` 下调。

这样安排的原因是：

- 越靠前的阶段越接近“纯张量计算”，越容易定位问题；
- adapter 与实机接入必须建立在 engine 正确的前提上；
- 一致性验证要贯穿开发，而不是最后补。

---

## 8. 每阶段的完成定义

### Done 1：导出完成

- 有 ONNX
- 有 metadata
- 有 Torch vs ONNX 验证报告

### Done 2：引擎完成

- 有 FP32 engine
- 有 build summary
- runtime 可跑通
- ONNX vs TRT-FP32 通过

### Done 3：语义适配完成

- adapter 可运行
- queue 语义对齐
- temporal ensemble 语义对齐（如启用）
- Torch policy vs TRT adapter 通过

### Done 4：实机接入完成

- 独立 TRT 推理脚本可用
- 短时闭环稳定
- timing 指标完整

### Done 5：阶段性交付完成

- FP16 可选通过；
- 至少一个降低后的 `n_action_steps` 配置完成验证；
- 可以给出“当前建议部署配置”。

---

## 9. 风险管理

### 9.1 高风险项

- 相机顺序错误；
- pre/post stats 不一致；
- ONNX external data 解析失败；
- FP16 误差异常；
- adapter 语义与 Torch policy 不一致；
- 性能测试口径混乱，误判 TRT 收益。

### 9.2 风险应对策略

- 所有输入输出名、shape、camera order 都由 metadata 固化；
- 先 FP32 后 FP16；
- 先 engine 一致性，再 policy 一致性；
- 所有验证输出结构化报告；
- 不在第一版引入零拷贝与动态 shape。

---

## 10. 执行要求

### 10.1 必做项

以下事项在开发中必须完成：

- 导出模型；
- 构建 TensorRT engine；
- Torch vs ONNX 一致性验证；
- ONNX vs TRT-FP32 一致性验证；
- Torch policy vs TRT adapter 一致性验证；
- TRT-FP32 vs TRT-FP16 对比验证；
- 短时在线实机验证。

### 10.2 禁止项

在第一版开发中，禁止：

- 为了省事跳过一致性验证；
- 直接拿 FP16 结果替代 FP32 对齐；
- 直接上机器人验证而不做离线对齐；
- 在代码中写死相机顺序；
- 一开始就做全链路 TRT。

---

## 11. 最终一句话

本开发 Plan 的核心不是“尽快把 TRT 跑起来”，而是：

> **先用可验证的方式把 ACT 核心前向替换成 TensorRT，再把导出、引擎构建和一致性验证做成正式交付物，最后才进入实机部署与性能优化。**
