# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此仓库中工作时提供指导。

## 概述

本仓库是 HuggingFace LeRobot (v0.4.3) 的 fork，一个用于训练和部署机器人学习策略的 PyTorch 框架。在上游框架基础上，通过 `my_devs/cobot_magic/` 增加了对 Piper 机械臂的自定义硬件支持。

## 安装

```bash
pip install -e .
# 安装可选依赖（如特定硬件或策略）：
pip install -e ".[feetech,smolvla,test]"
```

项目支持用 `uv` 替代 pip。注意：`wallx` 和 `pi` 额外依赖因 `transformers` 版本锁定，与大多数其他策略额外依赖存在冲突。

## 常用命令

**训练：**
```bash
lerobot-train --policy.type=act --dataset.repo_id=<hf_repo_id> --output_dir=outputs/
# 从检查点恢复：
lerobot-train --config_path=outputs/checkpoints/000002/pretrained_model/train_config.json --resume=true
```

**评估：**
```bash
lerobot-eval --policy.path=outputs/checkpoints/000004/pretrained_model --env.type=aloha
```

**数据采集 / 遥操作：**
```bash
lerobot-record
lerobot-teleoperate
lerobot-calibrate
lerobot-find-cameras
lerobot-setup-motors
```

**数据集工具：**
```bash
lerobot-dataset-viz
lerobot-edit-dataset
lerobot-info
```

**测试：**
```bash
pytest tests/
pytest tests/test_specific_file.py::test_function  # 单个测试
```

**Makefile 端到端测试：**
```bash
make test-act-ete-train
make test-act-ete-eval
make test-end-to-end  # 运行所有端到端测试
make DEVICE=cuda test-act-ete-train  # 指定设备
```

**代码检查 / 格式化：**
```bash
ruff check src/ tests/
ruff format src/ tests/
mypy src/lerobot/configs/ src/lerobot/envs/  # 仅对部分模块强制执行
```

## 架构

### 包结构（`src/lerobot/`）

框架采用分层设计：

- **`configs/`** — 基于 `draccus` 的 dataclass 配置系统。所有 CLI 参数均映射到配置 dataclass，是理解各脚本参数的入口。
- **`scripts/`** — CLI 入口点（`lerobot_train.py`、`lerobot_eval.py`、`lerobot_record.py` 等），各脚本的 `main()` 在 `pyproject.toml` 中注册。
- **`policies/`** — 策略实现（ACT、Diffusion、VQ-BeT、SmolVLA、Pi0、GR00T、XVLA、HIL-SERL、TDMPC、WallX）。每个策略有独立子目录，包含配置 dataclass 和模型类。通过 `policies/factory.py` 按类型名实例化。
- **`robots/`** — 硬件抽象层。所有机器人实现统一的 `Robot` 接口，新硬件在此添加。
- **`teleoperators/`** — 输入设备抽象（手柄、键盘、手机、主臂），数据采集时与机器人配对使用。
- **`motors/`** — 底层电机驱动（Dynamixel、Feetech）。
- **`cameras/`** — 相机驱动（OpenCV、RealSense 等）。
- **`datasets/`** — `LeRobotDataset`（v3.0）：Parquet 元数据 + MP4/图像帧，集成 HuggingFace Hub。
- **`envs/`** — 仿真环境的 Gymnasium 封装（ALOHA、PushT、LIBERO、MetaWorld）。
- **`transport/`** — 基于 gRPC 的异步/分布式推理通信层。
- **`async_inference/`** — 异步策略推理支持。
- **`rl/`** — 强化学习工具（HIL-SERL）。

### 配置系统

`draccus`（固定版本 `0.10.0`）驱动所有配置。配置 dataclass 位于 `configs/`。CLI 参数通过点号语法直接映射到嵌套 dataclass 字段（如 `--policy.type=act`、`--dataset.repo_id=...`）。保存的配置为 JSON 文件，路径为 `<output_dir>/checkpoints/<step>/pretrained_model/train_config.json`。

### 自定义扩展（`my_devs/`）

`my_devs/cobot_magic/` 包含 Piper 机械臂的专用代码：
- `Piper_ros_private-ros-noetic/` — 用于 Piper 臂控制的 ROS Noetic 包
- `camera_ws/` — 包含 RealSense 和 Astra 相机驱动的 ROS 工作空间
- `aloha-devel/` — 针对 Piper 适配的 ACT/robomimic 训练代码
- `collect_data/` — 数据采集脚本和 Piper SDK 示例

此部分代码独立于主包 `src/lerobot/`，使用 ROS/catkin 构建系统而非 pip。

## 代码规范

- 行长度：110 字符（ruff 强制）
- 引号风格：双引号
- 导入风格：`isort`，`lerobot` 为第一方包
- Mypy 仅对 `configs/`、`envs/`、`optim/`、`model/`、`cameras/`、`transport/` 严格执行，其他模块设置 `ignore_errors = true`
- `T201`（print 语句）被忽略，脚本中允许使用 print
