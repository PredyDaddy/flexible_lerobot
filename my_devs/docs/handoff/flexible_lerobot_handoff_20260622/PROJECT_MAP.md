# Project Map

## 1. 顶层结构

```text
src/lerobot/
  LeRobot 主 Python 包。包含 configs、datasets、envs、robots、policies、training、utils 等。

tests/
  pytest 测试和 fixtures。部分大测试工件走 Git LFS。

docs/source/
  官方文档源码，Markdown/MDX。

examples/ benchmarks/
  示例脚本和基准工具。

my_devs/
  用户主要开发目录。默认新增功能、实验脚本和交付文档都放这里。

my_devs/docs/
  用户按功能点归档的方案、报告、验收文档。
```

## 2. `my_devs/` 重点目录

### ACT TensorRT

```text
my_devs/new_act_trt/
my_devs/act_trt/
my_devs/docs/act_trt/
```

当前建议：

- 优先看 `my_devs/new_act_trt/README.md`。
- `new_act_trt` 是当前更可复现的 ACT TensorRT 工作区。
- `act_trt` 是历史脚本区，仍有 ONNX / TRT / mock / real robot 脚本。
- 文档 `my_devs/docs/act_trt/act_trt复现与参考工作流.md` 串起了历史脚本的一整条链路。

核心目标：

- ACT checkpoint 导出 ONNX。
- 构建 TensorRT engine。
- 验证 Torch / ONNX / TRT 一致性。
- 将 TRT policy adapter 接入 ACT 上机推理。

### GR00T TensorRT

```text
my_devs/groot_trt/
my_devs/groot_trt_async_server/
my_devs/docs/gr00t_trt/
```

当前建议：

- 先看 `my_devs/groot_trt/README.md`。
- `groot_trt` 覆盖 GR00T N1.5 的 ONNX 导出、7 engine 构建、Torch/ONNX/TRT 一致性对比。
- `groot_trt_async_server` 是异步服务化链路，包含 server/client、协议、TRT backend 自检和多轮整改文档。

典型产物：

```text
vit_fp16.onnx / vit_fp16.engine
llm_fp16.onnx / llm_fp16.engine
vlln_vl_self_attention.onnx / .engine
state_encoder.onnx / .engine
action_encoder.onnx / .engine
DiT_fp16.onnx / .engine
action_decoder.onnx / .engine
```

### PI0.5 / OpenPI TensorRT

```text
my_devs/openpi_trt/
my_devs/pi_trt/
my_devs/pi05_engineering/
my_devs/docs/pi_trt/
my_devs/docs/pi05_engineering/
```

当前主线：

```text
prefix_cache TensorRT engine
  + denoise_step TensorRT engine
  + Python 10-step denoise loop
  -> 替换 PI05Pytorch.sample_actions(...)
```

关键文档：

```text
my_devs/openpi_trt/README.md
my_devs/openpi_trt/docs/openpi_trt实现链路报告.md
my_devs/openpi_trt/docs/lerobot_pi05_trt难点分析.md
my_devs/openpi_trt/docs/lerobot_pi05_prefix_cache_denoise_trt报告.md
my_devs/openpi_trt/docs/lerobot_pi05_torch_onnx验证报告.md
my_devs/openpi_trt/docs/lerobot_pi05_torch_trt验证报告.md
my_devs/openpi_trt/docs/lerobot_pi05_camera_smoke_trt报告.md
```

注意：

- `openpi_trt` 里包含 Jetson AI Lab OpenPI on Thor 参考代码和本仓 PI0.5 适配文档。
- 不要假设 OpenPI 上游脚本能直接跑本仓 LeRobot PI0.5 checkpoint。
- 需要核对 `config.json` 里的 state/action dim、chunk size、token length、camera 数量和 processor。

### VLA Engineering

```text
my_devs/vla_engineering/
my_devs/docs/vla_engineering/
my_devs/flash_RT_pi/
```

核心思想：

- 保留 `vlash-main` 和 `realtime-vla-v2-main` 做参考。
- 本仓新增 `vlash_iner` 作为当前 PI0.5 checkpoint 的隔离推理工程。
- 支持同步推理、异步 action chunk、服务化推理、`torch` / `torch_compile` / `tensorrt_split` 后端。
- `flash_RT_pi` 是未跟踪的 FlashRT 参考源码目录，可能用于后续研究 PI0.5 / GROOT / VLA 小 batch 实时推理替代路线。

关键文档：

```text
my_devs/docs/vla_engineering/工作报告.md
my_devs/docs/vla_engineering/扩散VLA推理抖动缓解方案.md
my_devs/vla_engineering/doc/PI05服务化异步推理优化技术文档.md
my_devs/vla_engineering/doc/PI05_TensorRT异步链路当前验收状态.md
my_devs/vla_engineering/doc/PI05_TensorRT后端接入异步链路验收方案.md
my_devs/flash_RT_pi/reference_source_code/FlashRT-main/README.md
```

### AgileX / Web Collection

```text
my_devs/add_robot/agilex/
my_devs/agilex_web_collection/
my_devs/web_collection/
my_devs/docs/add_robot/agilex/
my_devs/docs/web_collection/
my_devs/docs/agilex_lerobot_web_eollection/
```

两条相关但不同的采集链路：

- `agilex_web_collection`：
  - Web 后端包装 `my_devs/add_robot/agilex/record.sh`。
  - 重点是 AgileX LeRobot 采集 job 管理。

- `web_collection`：
  - ROS topic -> HDF5 -> LeRobot v3 dataset。
  - 有简单 Web UI 和 CLI。

### 数据处理工具

```text
my_devs/data_check/
my_devs/remove_feature_pipeline/
my_devs/split_datasets/
```

用途：

- LeRobot v3 数据检查。
- 从数据集中移除 feature / 单臂 pipeline。
- 左右臂数据集拆分和校验。

### 机器人与硬件参考

```text
my_devs/cobot_magic/
my_devs/agilex_scripts/
my_devs/robot_camera_migration/
```

注意这里包含 ROS、相机、第三方参考工程或硬件脚本。上机或系统级操作前要先确认用户机器状态。

## 3. 文档组织习惯

用户习惯：

- 每个功能点一个目录。
- 方案、执行报告、验收报告都放 `my_devs/docs/<feature>/`。
- 功能代码放 `my_devs/<feature>/`。

新增功能建议沿用：

```text
my_devs/<feature_name>/
my_devs/docs/<feature_name>/技术方案.md
my_devs/docs/<feature_name>/工作报告.md
my_devs/docs/<feature_name>/验收报告.md
```

## 4. 代码风格

仓库主配置：

- Ruff。
- 行宽 110。
- 4 空格缩进。
- 双引号。
- import 排序由 Ruff 处理。

常用命令：

```bash
conda run -n lerobot_flex ruff format my_devs/<feature>
conda run -n lerobot_flex ruff check my_devs/<feature> --fix
```
