# OpenPI PI0.5 TensorRT Reference

这里保存 Jetson AI Lab 教程中公开的 OpenPI PI0.5 TensorRT 参考代码，来源：

- 教程页面：https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/
- 代码下载入口：https://www.jetson-ai-lab.com/code-samples/openpi_on_thor/download.sh
- 教程源码：https://github.com/NVIDIA-AI-IOT/jetson-ai-lab/blob/main/src/content/tutorials/vla/openpi_on_thor.md

当前代码是上游教程的参考实现，主要面向 NVIDIA Jetson AGX Thor、NVIDIA PyTorch
container、TensorRT、NVIDIA ModelOpt、OpenPI 原生仓库。它还不是本仓库
`src/lerobot/policies/pi05` 的直接集成版。

## 目录

- `openpi_on_thor/`: 从 Jetson AI Lab 下载脚本拉取的原始代码。
- `download_upstream.sh`: 重新拉取上游公开文件的脚本。

## 已拿到的上游文件

- `openpi_on_thor/thor.Dockerfile`
- `openpi_on_thor/pyproject.toml`
- `openpi_on_thor/pi05_inference.py`
- `openpi_on_thor/pytorch_to_onnx.py`
- `openpi_on_thor/build_engine.sh`
- `openpi_on_thor/trt_model_forward.py`
- `openpi_on_thor/trt_torch.py`
- `openpi_on_thor/calibration_data.py`
- `openpi_on_thor/patches/apply_gemma_fixes.py`

## 上游流程概要

教程流程可以拆成四段：

1. 准备 OpenPI 仓库和 Thor Docker 环境。
2. 转换或准备 PI0.5 PyTorch checkpoint。
3. 用 `pytorch_to_onnx.py` 导出 ONNX，并可通过 NVIDIA ModelOpt 做 FP8 / NVFP4 量化。
4. 用 `build_engine.sh` 调 `trtexec` 生成 TensorRT engine，再用 `pi05_inference.py`
   做 PyTorch、TensorRT 或二者对比推理。

关键输入输出大致如下：

- PyTorch checkpoint 默认路径：`/root/converted_pytorch_checkpoint`
- ONNX 默认目录：`/root/converted_pytorch_checkpoint/onnx`
- TensorRT engine 默认跟 ONNX 放在同一目录，后缀为 `.engine`
- TensorRT engine 输入名：
  `images`, `img_masks`, `lang_tokens`, `lang_masks`, `state`, `noise`
- TensorRT engine 输出名：`actions`

## 重要注意点

- 这套脚本依赖 `openpi` 包的原生 API，例如 `openpi.policies.policy_config`、
  `openpi.models_pytorch.pi0_pytorch` 和 `openpi.training.config`。
- `thor.Dockerfile` 会安装 OpenPI 期望的老 LeRobot 结构：
  `lerobot.common.*`。这与本仓库当前源码结构不一定一致。
- `build_engine.sh` 固定了 PI0.5 常见形状假设：
  3 路图像、224 图像尺寸、`STATE_DIM=32`、`ACTION_DIM=32`，默认
  `ACTION_HORIZON=15`。如果接入本仓库训练出的 PI0.5 checkpoint，需要先核对
  `config.json` 里的 state/action 维度和 action horizon。
- 教程强依赖 Jetson Thor 生态中的 TensorRT、`trtexec`、ModelOpt 和对应 CUDA wheel。
  在普通 x86 开发机或非 Thor Jetson 上不一定能直接运行。
- 本仓库开发规范要求使用 `lerobot_flex` conda 环境；实际构建 Thor Docker 或运行
  TensorRT 时，仍需要以目标设备容器环境为准。

## 后续适配建议

如果要把这套参考实现接进本仓库的 PI0.5 路径，建议按下面顺序做：

1. 对照本仓库 `src/lerobot/policies/pi05/modeling_pi05.py` 的加载路径，确认
   `model.safetensors`、`config.json` 和 processor/tokenizer 文件如何恢复 PyTorch 模型。
2. 先做 Torch vs ONNX 的最小数值对齐，不急着上 TensorRT。
3. 明确本仓库 PI0.5 的导出边界：单体 graph，还是 vision encoder / prefix cache /
   denoise step 三段式。
4. 根据真实 checkpoint 配置生成动态 shape，而不是直接使用上游脚本里的
   `STATE_DIM=32`、`ACTION_DIM=32`、`ACTION_HORIZON=15`。
5. ONNX 稳定后再构建 engine，并用固定 noise 比较 Torch / ONNX / TRT 输出。

