# openpi_trt 重构说明与后续优化方案

日期：2026-06-22  
仓库：`/data/cqy_workspace/flexible_lerobot`  
模块：`my_devs/openpi_trt`  
相关下游：`my_devs/vla_engineering`

## 1. 背景

`my_devs/openpi_trt` 最早保存的是 Jetson AI Lab / OpenPI on Thor 的 PI0.5 TensorRT 参考实现。上游参考代码主要面向 OpenPI 原生仓库、Jetson Thor Docker、TensorRT、NVIDIA ModelOpt 和固定的 OpenPI 推理接口。

当前仓库里的实际目标已经变成另一件事：

```text
把 LeRobot 版 PI0.5 policy 中最重的 sample_actions(...) 推理路径替换成 TensorRT 后端。
```

现有主线不是完整单体 TensorRT engine，而是 split TensorRT：

```text
prefix_cache TensorRT engine
  + denoise_step TensorRT engine
  + Python denoise loop
  -> 替换 policy.model.sample_actions(...)
```

这个拆分是因为完整 `sample_actions(...)` 图里包含 vision encoder、language embedding、prefix cache、Gemma expert、RoPE、AdaRMS、10-step denoise loop 等复杂结构，直接导出完整 ONNX/TensorRT 图不稳定。split 后每个边界更小，更适合逐步验证。

重构前的问题主要是工程化问题：

- runtime、scripts、验证报告里的 TensorRT I/O 协议重复定义。
- `num_layers=18`、默认 engine 路径、SO101 双相机假设分散在多个脚本里。
- 真实上机脚本、camera smoke、benchmark、Torch/TRT verify 之间有很多重复加载逻辑。
- `my_devs/vla_engineering` 已经引用了 `openpi_trt` 的旧导入路径，不能随意改坏。
- artifacts 目录很大，engine、ONNX external data、debug dump、report 混在一起，后续需要进一步整理。

因此本轮重构优先目标不是追求新的速度结果，而是把已经跑通的 split TensorRT 后端整理成更稳定的库边界。

## 2. 如何从参考仓库开展工作

从参考仓库或教程开始时，不建议直接把上游脚本当成最终工程代码使用，而是按下面顺序迁移：

1. 保留上游参考代码作为 vendor/reference。

   当前保留在：

   ```text
   my_devs/openpi_trt/openpi_on_thor/
   ```

   这部分用于对照 OpenPI/Thor 的导出思路、TensorRT engine 构建方式和推理入口，不作为 LeRobot 主线 runtime。

2. 对照 LeRobot 的 checkpoint 和 policy 加载路径。

   LeRobot PI0.5 需要从 `config.json`、`model.safetensors`、processor/tokenizer 等恢复 policy。不能假设上游 OpenPI 的 checkpoint 目录、接口名称、state/action 维度和当前仓库一致。

3. 先确定导出边界。

   当前稳定边界是：

   ```text
   prefix_cache:
     image_0, image_1, img_mask_0, img_mask_1, tokens, masks
     -> prefix_pad_masks + flattened past_key_values

   denoise_step:
     prefix_pad_masks + flattened past_key_values + x_t + timestep
     -> v_t
   ```

4. 先验证 Torch vs ONNX，再验证 Torch vs TensorRT，最后才接真实相机或机器人。

   推荐顺序：

   ```text
   synthetic batch Torch vs ONNX
   synthetic batch Torch vs TensorRT
   camera-only smoke
   benchmark
   人工确认后再上机
   ```

5. 上机控制必须显式交给操作者。

   本轮重构只做离线结构整理和非上机验证。任何会连接机器人、读取真实控制串口或发送 action 的流程都不在重构验证中执行。

## 3. 重构版本代码的好处

本轮新增了三个 runtime 基础模块：

```text
my_devs/openpi_trt/runtime/protocol.py
my_devs/openpi_trt/runtime/config.py
my_devs/openpi_trt/runtime/metadata.py
```

### 3.1 协议集中

`runtime/protocol.py` 集中定义 split TensorRT 的稳定 I/O 协议：

```text
prefix_cache_tensor_names(...)
denoise_step_input_names(...)
PREFIX_CACHE_INPUT_NAMES
DENOISE_OUTPUT_NAMES
validate_names(...)
```

这样 `runtime` 和 `scripts` 不再各自维护一份 `past_key_values.*.key/value` 名称生成逻辑，减少导出、构建、运行时不一致的风险。

### 3.2 配置集中

`runtime/config.py` 新增 `PI05SplitTRTConfig`，集中管理：

```text
prefix_engine_path
denoise_engine_path
num_layers
batch_size
chunk_size
max_action_dim
camera_count
```

`PI05TensorRTSplitRuntime` 现在既支持旧调用方式：

```python
PI05TensorRTSplitRuntime(prefix_engine_path, denoise_engine_path)
```

也支持新配置方式：

```python
config = PI05SplitTRTConfig.from_paths(prefix_engine_path, denoise_engine_path)
runtime = PI05TensorRTSplitRuntime(config)
```

### 3.3 runtime 启动时校验 engine I/O

`PI05TensorRTSplitRuntime` 初始化时会校验：

```text
prefix_cache engine inputs
prefix_cache engine outputs
denoise_step engine inputs
denoise_step engine outputs
```

如果 engine 文件和当前代码协议不匹配，会在加载阶段报错，而不是等到真实推理或上机时才失败。

### 3.4 兼容 vla_engineering

`my_devs/vla_engineering` 当前主要引用：

```python
from runtime.pi05_trt_split import patch_sample_actions_with_split_trt
from scripts.pi05_onnx_common import load_policy
```

本轮保留了这些旧导入路径和函数签名，因此 `vla_engineering` 的现有 TensorRT split 后端不会因为本轮重构被迫改调用代码。

后续如果要重构 `vla_engineering`，可以逐步改成直接使用：

```python
from runtime.config import PI05SplitTRTConfig
from runtime.pi05_trt_split import PI05TensorRTSplitRuntime
```

但这不是本轮必要条件。

### 3.5 新增离线 artifact 预检入口

新增：

```text
my_devs/openpi_trt/scripts/check_split_trt_artifact.py
```

它只做离线预检：

```text
检查 policy path
检查 prefix/denoise engine 文件
反序列化 TensorRT engine
校验 engine I/O 协议
可选写 metadata.json
可选写 runtime describe report
```

它不会打开相机，不会连接机器人，不会发送 action。

示例：

```bash
conda run -n lerobot_flex python my_devs/openpi_trt/scripts/check_split_trt_artifact.py \
  --policy-path outputs/pi05_eraser_cup_multi_task_runs/20260602_200955/checkpoints/last/pretrained_model \
  --prefix-engine-path my_devs/openpi_trt/artifacts/pi05_so101_prefix_cache_b1_fp32.engine \
  --denoise-engine-path my_devs/openpi_trt/artifacts/pi05_so101_denoise_step_b1_fp32.engine \
  --metadata-out my_devs/openpi_trt/artifacts/pi05_so101_split_trt_fp32_metadata.json
```

## 4. 后续优化方案

后续建议分四层推进。

### 4.1 工程结构继续收敛

建议新增公共模块：

```text
my_devs/openpi_trt/common/paths.py
my_devs/openpi_trt/common/policy.py
my_devs/openpi_trt/common/camera.py
my_devs/openpi_trt/common/reports.py
my_devs/openpi_trt/common/robot_runtime.py
```

把 scripts 中重复的路径处理、policy 加载、camera smoke、report 写入、SO101 robot runtime 逻辑逐步抽出来。这样脚本只保留 CLI 编排，核心逻辑沉到 common/runtime。

### 4.2 artifact 目录治理

当前 `my_devs/openpi_trt/artifacts` 中混有 engine、ONNX、external data、debug dump、camera frames 和 reports。建议整理成：

```text
artifacts/
  engines/
  onnx/
  reports/
  frames/
  debug_dumps/
```

并给每套可部署 engine 生成 metadata：

```text
metadata.json
```

后续真实服务和 `vla_engineering` 可以只接收 `artifact_dir`，由 metadata 找到 prefix/denoise engine 和验证记录。

### 4.3 TensorRT 性能优化

当前稳定 baseline 是 FP32 split TensorRT。已有实验显示：

```text
FP32 split TRT:
  数值稳定，约 2.05x 加速

纯 FP16 denoise:
  更快，但动作漂移明显，不适合作为默认方案

constrained FP16 denoise:
  速度更好，数值基本可接受，需要继续验证
```

后续性能优化建议：

- 保留 FP32 split TRT 作为安全 baseline。
- 将 constrained FP16 作为独立 profile，不覆盖 FP32。
- 在 engine build report 中记录哪些 layer 被强制 FP32。
- 分别 benchmark prefix_cache、denoise_step、Python loop 和同步开销。
- 再考虑异步执行、预分配 output tensor、CUDA Graph。

### 4.4 vla_engineering 联动重构

`my_devs/vla_engineering` 目前也比较混乱，但它已经能引用 `openpi_trt` 的 split TensorRT 后端。后续可以做兼容式重构：

1. 保留现有 server CLI 参数。
2. 把 TensorRT 后端加载逻辑改成读取 `PI05SplitTRTConfig` 或 metadata。
3. 将 `/health` 返回的 backend_info 扩展为 artifact metadata + runtime describe。
4. 将 robot acceptance 和 backend acceptance 分开，保证后端验收不会误触发真实控制。

重构原则仍然是：

```text
后端加载、engine 预检、synthetic/camera-only 验证可以自动跑；
真实机器人连接和 action 发送必须由操作者显式执行。
```

## 5. 安全边界

本轮重构不运行上机控制，不连接机器人串口，不发送任何 action。

允许的验证：

```text
python 编译检查
import 检查
TensorRT engine 离线反序列化与 I/O 校验
synthetic Torch/ONNX/TRT 数值验证
camera-only smoke
```

需要人工执行的验证：

```text
真实机器人连接
真实 action 发送
真实任务成功率评估
```
