# TensorRT 引擎构建与推理实践手册（Python）

## 版本信息（必须精确）

- 文档版本: `v1.1.0`
- 更新日期: `2026-02-24`
- 目标环境: `conda env = cqy_trt`
- TensorRT: `10.13.0.35`
- Torch: `2.10.0+cu128`
- TorchVision: `0.25.0+cu128`
- ONNX Runtime: `1.23.2`

版本采集命令:

```bash
conda run -n cqy_trt python -c "import tensorrt as trt, torch, torchvision, onnxruntime as ort; print(trt.__version__, torch.__version__, torchvision.__version__, ort.__version__)"
```


## 1. 文档目标

这份文档的目的不是“解释 ResNet”，而是沉淀一套以后可直接复用的标准流程:

1. 如何正确构建 TensorRT 引擎（build）
2. 如何正确执行 TensorRT 推理（infer）
3. 以后接入任意模型时，前处理/后处理如何与推理框架解耦并对接

本文以 `my_dev/python_simple/` 的实现作为验证案例。


## 2. 总体原则

1. TRT 主流程只走 name-based tensor API，不走 binding-index 老接口。
2. runtime 层只处理“张量与内存”，不处理模型语义。
3. 预处理和后处理必须作为独立适配层，不与 TRT 执行链硬耦合。
4. build 和 infer 都要可审计:
   - 输入参数可追溯
   - 报错可定位
   - 输出 JSON 可回归比较


## 3. 引擎构建（Build）标准流程

### 3.1 构建前检查

必须先确认两件事:

1. TensorRT Python 能导入
2. CUDA 设备可见

```bash
conda run -n cqy_trt python -c "import tensorrt as trt; print(trt.__version__)"
conda run -n cqy_trt python -c "from cuda import cudart; print(cudart.cudaGetDeviceCount())"
```


### 3.2 必须遵守的 API 调用顺序

1. `logger = trt.Logger(trt.Logger.WARNING)`
2. `trt.init_libnvinfer_plugins(logger, "")`
3. `builder = trt.Builder(logger)`
4. `network = builder.create_network(flags)`
5. `parser = trt.OnnxParser(network, logger)`
6. `parser.parse(onnx_bytes)`（失败时打印全部 parser error）
7. `config = builder.create_builder_config()`
8. `config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)`
9. `config.builder_optimization_level = opt_level`
10. 精度配置:
    - FP16: 仅在 `builder.platform_has_fast_fp16` 为真时设置
    - FP32: 不设置额外精度 flag
11. 可选动态 batch profile:
    - `profile = builder.create_optimization_profile()`
    - `profile.set_shape(input_name, min_shape, opt_shape, max_shape)`
    - `config.add_optimization_profile(profile)`
12. 可选 timing cache:
    - `cache = config.create_timing_cache(cache_bytes)`
    - `config.set_timing_cache(cache, ignore_mismatch=True)`
13. `plan = builder.build_serialized_network(network, config)`
14. 写入 `.plan`
15. 可选写入 `.tcache`
16. 反序列化回读验证:
    - `runtime = trt.Runtime(logger)`
    - `engine = runtime.deserialize_cuda_engine(plan_bytes)`


### 3.3 构建阶段的工程防护

1. `parser.parse` 失败必须把 `parser.get_error(i)` 全量输出。
2. 兼容不同 TRT wrapper 行为:
   - `set_shape` 可能返回 `None` 或 `False`
   - `add_optimization_profile` 返回 `0` 是合法索引
3. timing cache 序列化要兼容 `memoryview` 与 context-manager buffer 两类返回值。
4. 没有可用 CUDA 设备时要提前报错，避免 builder 在某些环境直接崩溃。


## 4. 引擎推理（Infer）标准流程

### 4.1 运行时初始化

1. `logger = trt.Logger(...)`
2. `trt.init_libnvinfer_plugins(logger, "")`
3. `runtime = trt.Runtime(logger)`
4. `engine = runtime.deserialize_cuda_engine(plan_bytes)`
5. `context = engine.create_execution_context()`
6. 需要多 profile 时:
   - `context.set_optimization_profile_async(profile_index, stream)`


### 4.2 IO 枚举与形状推导

只用 name-based API:

1. `for i in range(engine.num_io_tensors)`
2. `name = engine.get_tensor_name(i)`
3. `mode = engine.get_tensor_mode(name)`

每次推理前:

1. `context.set_input_shape(input_name, batch.shape)`
2. `unresolved = context.infer_shapes()`
3. `unresolved` 非空直接失败
4. 通过 `context.get_tensor_shape(output_name)` 获取输出 shape
5. 输出 shape 出现负维度直接失败


### 4.3 内存与流（推荐基线）

1. 每个 tensor 分配 pinned host + device:
   - `cudaMallocHost`
   - `cudaMalloc`
2. 创建 stream:
   - `cudaStreamCreate`
3. H2D:
   - `cudaMemcpyAsync(..., cudaMemcpyHostToDevice, stream)`
4. 所有 IO 设置地址:
   - `context.set_tensor_address(name, ptr)`
5. 执行:
   - `context.execute_async_v3(stream_handle=stream)`
6. D2H:
   - `cudaMemcpyAsync(..., cudaMemcpyDeviceToHost, stream)`
7. 同步:
   - `cudaStreamSynchronize(stream)`

动态 shape 下的缓冲策略:

1. 初始可按 profile max shape 分配
2. 每次推理按真实 `required_nbytes` 检查容量
3. 不足时只重分配对应 tensor buffer


### 4.4 dtype 与输出规范

1. 优先使用 TRT dtype 到 NumPy dtype 的映射。
2. 映射失败时启用 byte-level fallback，再按 itemsize/type 解释。
3. 输出给后处理/对比前统一转 `float32`。


### 4.5 benchmark 规范

1. 只统计推理，不统计前处理。
2. 默认 `warmup=10`、`repeat=50`。
3. 使用 `time.perf_counter()`。
4. GPU 计时窗口要保证同步完成。
5. 输出指标:
   - mean/std/min/max/p50/p95/p99
   - qps


## 5. 未来模型如何接入前处理/后处理

这是你后续迁移模型时最重要的部分。

### 5.1 解耦原则

1. TRT runtime 只负责执行，不负责业务语义。
2. 前处理负责把原始输入转成模型输入张量。
3. 后处理负责把模型输出解码成业务结果。
4. runtime 与 pre/post 通过稳定接口连接，不相互侵入。


### 5.2 推荐接口契约

```python
class Preprocessor:
    def __call__(self, raw_input) -> tuple[dict[str, np.ndarray], dict]:
        # feed_dict: {input_tensor_name: contiguous ndarray}
        # meta: 逆变换信息（scale/pad/crop/token-offset 等）
        ...

class Postprocessor:
    def __call__(self, outputs: dict[str, np.ndarray], meta: dict) -> dict:
        # 将输出张量解码为业务结果
        ...
```

约束:

1. `feed_dict` 的 key 必须与引擎输入 tensor name 精确一致。
2. 输入数组必须 contiguous，dtype 必须与模型契约一致。
3. `meta` 必须足够支持后处理做反变换。


### 5.3 前处理设计检查单（通用）

1. 明确定义输入契约:
   - layout（NCHW/NHWC）
   - color space（RGB/BGR）
   - normalize 方式
   - static/dynamic 维度规则
2. 变换过程可复现:
   - 固定配置
   - 可记录 `preprocess_id` 或 config hash
3. 输出必须可被 runtime 直接消费:
   - contiguous
   - dtype 正确
   - shape 符合引擎要求


### 5.4 后处理设计检查单（通用）

1. 按 tensor name 取输出，不按固定 index 假设。
2. 解码前先校验 shape/dtype。
3. 解码逻辑与业务阈值分离。
4. 结果 JSON schema 稳定，便于回归 diff。
5. 保留 raw output 调试路径。


## 6. 以 ResNet 案例说明脚本是怎么做的

这一节对应你当前 `my_dev/python_simple/` 的实现，目的是把“抽象原则”落到“可运行脚本”。

### 6.1 导出阶段（`torch_export.py`）

1. 构建模型: `resnet18/50/101 + imagenet/random`。
2. ONNX 导出:
   - `input_names=["input"]`
   - `output_names=["logits"]`
   - `dynamo=False`
   - `external_data=False`
3. 可选动态 batch 轴导出。

输出:

- `cache/models/<model>_<weights>.onnx`


### 6.2 基线推理阶段（`torch_infer.py` / `ort_infer.py`）

两者使用同口径 pre/post:

1. 前处理:
   - RGB
   - short side 256
   - center crop 224
   - normalize
   - HWC->CHW
   - `np.ascontiguousarray(float32)`
2. 后处理:
   - logits 转 float32
   - `top1 = argmax`
   - `top5 = argpartition + argsort`
3. benchmark:
   - warmup/repeat
   - 统一统计字段
4. ORT 默认 provider 顺序:
   - `CUDAExecutionProvider`
   - `CPUExecutionProvider`

输出:

- `cache/reports/torch_<model>_<weights>.json`
- `cache/reports/ort_<model>_<weights>.json`


### 6.3 TRT 构建阶段（`trt_build.py`）

1. 初始化插件 + builder + parser。
2. parse ONNX，失败打印全量错误。
3. 设置:
   - workspace
   - optimization level
   - precision（fp16/fp32）
   - 可选 dynamic batch profile
   - 可选 timing cache
4. build serialized network，写 `.plan`。
5. 回读反序列化验证 engine 可用。

输出:

- `cache/models/<model>_<weights>_<precision>.plan`
- `cache/models/<model>_<weights>_<precision>.tcache`（可选）


### 6.4 TRT 推理阶段（`trt_infer.py`）

1. 反序列化 engine + 创建 context。
2. 枚举 IO:
   - `num_io_tensors`
   - `get_tensor_name`
   - `get_tensor_mode`
3. 每轮推理:
   - `set_input_shape`
   - `infer_shapes`
   - 根据 shape/dtype 准备或扩容 buffer
   - `set_tensor_address`
   - `execute_async_v3`
   - D2H + stream synchronize
4. 输出 `float32 logits + top1/top5 + benchmark`。

输出:

- `cache/reports/trt_<model>_<weights>_<precision>.json`


### 6.5 三方对比阶段（`compare_triple.py`）

1. 读取 torch/ort/trt 三份 JSON。
2. 强制校验 meta 一致性:
   - model
   - weights
   - image
   - input_shape
   - logits shape
3. 计算精度指标:
   - max_abs_diff
   - mean_abs_diff
   - cosine_similarity
   - top1_match
4. 计算速度指标:
   - trt_vs_torch
   - ort_vs_torch
   - trt_vs_ort

输出:

- `cache/reports/compare_<model>_<weights>_<precision>.json`


## 7. 新模型接入时，哪些保持不变，哪些需要替换

保持不变（框架骨架）:

1. `trt_build.py` 的构建 API 顺序
2. `trt_infer.py` 的 name-based 推理执行链
3. benchmark 口径和 JSON 报告结构
4. compare 阶段的一致性校验与指标定义

需要替换（模型语义层）:

1. 预处理逻辑（图像/文本/多模态）
2. 输入 tensor 名称与 shape 规则
3. 输出解码逻辑（分类/检测/分割/生成）
4. 后处理与业务阈值


## 8. 验收建议（面向过程正确性）

你这个项目的核心价值是“验证 build/infer 过程正确”，建议每次新模型都保留以下证据:

1. build 命令 + 构建摘要（workspace/precision/profile/cache）
2. infer 命令 + 报告 JSON（含 shape、benchmark）
3. compare JSON（含精度与速度）
4. 异常场景日志（例如 parse 失败、shape unresolved）

只要这些证据完整，就能证明“流程正确且可复现”，而不依赖某个单一模型本身。


## 9. 相关文件

1. `my_dev/docs/TENSORRT_PYTHON_API_DESIGN_FINAL.md`
2. `my_dev/python_simple/docs/PYTHON_SIMPLE_6SCRIPTS_IMPLEMENTATION_SPEC.md`
3. `my_dev/python_simple/trt_build.py`
4. `my_dev/python_simple/trt_infer.py`
5. `my_dev/python_simple/compare_triple.py`
