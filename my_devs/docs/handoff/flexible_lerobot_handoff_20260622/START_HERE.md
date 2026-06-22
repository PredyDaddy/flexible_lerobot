# Flexible LeRobot Handoff

生成日期：2026-06-22  
仓库路径：`/data/cqy_workspace/flexible_lerobot`  
当前分支：`feature/cqy`  
当前提交：`c25e170 pi05trt后端推理`

## 1. 给下一个 Agent 的第一句话

这是一个基于 Hugging Face LeRobot 改造的机器人开发仓库。用户的主要开发都放在 `my_devs/`，文档按功能点放在 `my_devs/docs/`。除非用户明确授权，默认不要直接改 `src/lerobot/` 主包。

最重要的硬约束：

- 所有开发、测试、格式化、脚本执行都必须使用 `lerobot_flex` conda 环境。
- 命令推荐写法：`conda run -n lerobot_flex python ...`。
- 代码开发默认放在 `my_devs/`。
- 如果确实要改主包 `src/lerobot/`，先向用户确认。
- 不要提交密钥、token、模型权重或大产物。
- 仓库里已有用户未跟踪文件，不要擅自删除或回滚。

## 2. 推荐阅读顺序

1. `AGENTS.md`
   - 仓库级开发规范，特别是 `lerobot_flex` 环境和 `my_devs/` 开发约束。
2. 本 handoff 包：
   - `CURRENT_STATE.md`
   - `PROJECT_MAP.md`
   - `COMMANDS.md`
   - `NEXT_AGENT_PROMPT.md`
3. 结合具体任务阅读对应功能文档：
   - ACT TensorRT：`my_devs/new_act_trt/README.md`，`my_devs/docs/act_trt/`
   - GR00T TensorRT：`my_devs/groot_trt/README.md`，`my_devs/docs/gr00t_trt/`
   - PI0.5 / OpenPI TensorRT：`my_devs/openpi_trt/README.md`，`my_devs/openpi_trt/docs/`
   - VLA 工程化：`my_devs/docs/vla_engineering/工作报告.md`，`my_devs/vla_engineering/doc/`
   - AgileX Web 采集：`my_devs/agilex_web_collection/README.md`
   - ROS HDF5 到 LeRobot v3 采集：`my_devs/web_collection/README.md`

## 3. 当前项目主线

仓库保留 LeRobot 主框架，同时在 `my_devs/` 做了一系列本地工程化工作：

- ACT 模型 ONNX / TensorRT 导出、验证、TRT runtime 和上机入口。
- GR00T N1.5 ONNX / TensorRT 导出、7 engine 构建、Torch/ONNX/TRT 一致性验证和异步服务链路。
- PI0.5 / OpenPI TensorRT 化，当前推荐路线是 `prefix_cache TensorRT engine + denoise_step TensorRT engine + Python denoise loop`。
- VLA 异步 chunk 推理、服务化推理、TensorRT split 后端接入。
- AgileX / SO101 / SO100 机器人相关采集、上机和 Web 工具。
- 数据集处理工具，包括 LeRobot v3 数据检查、左右臂拆分、移除特征 pipeline 等。

## 4. 最小安全启动命令

```bash
cd /data/cqy_workspace/flexible_lerobot
git status --short --branch
conda run -n lerobot_flex python --version
conda run -n lerobot_flex python -c "import lerobot; print('lerobot import ok')"
```

如果 conda 环境不存在或 import 失败，先停下来向用户报告环境问题，不要切到别的环境继续做。

## 5. Handoff 包内容

```text
START_HERE.md
CURRENT_STATE.md
PROJECT_MAP.md
COMMANDS.md
NEXT_AGENT_PROMPT.md
MANIFEST.md
```

这个 zip 不包含模型权重、TensorRT engine、ONNX、`__pycache__` 或大型生成物。它是“给 Agent 的交接说明”，不是源码快照。源码仍在当前仓库内。
