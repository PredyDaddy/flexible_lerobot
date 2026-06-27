# lerobot_flex 环境迁移说明

这个目录用于迁移当前项目使用的 Conda Python 环境。

## 文件说明

- `environment.yml`: Conda 环境入口文件，只固定 Python 版本，并从 `requirements-lock.txt` 安装 pip 包。
- `requirements-lock.txt`: 从当前 `lerobot_flex` 环境导出的 pip 依赖锁定列表，已去掉当前仓库的 editable 安装引用。

## 在新机器/新目录恢复环境

假设代码已经拉到新目录，并且当前 shell 位于本仓库根目录：

```bash
cd my_devs/env_transfer
conda env create -f environment.yml
conda activate lerobot_flex
cd ../..
python -m pip install -e ".[dev,test]"
```

如果新机器上已经有同名环境，可以改用：

```bash
cd my_devs/env_transfer
conda env update -n lerobot_flex -f environment.yml --prune
conda activate lerobot_flex
cd ../..
python -m pip install -e ".[dev,test]"
```

## 注意事项

- 原环境里的 `lerobot` 是以 editable 方式安装的旧仓库引用，迁移时应在新代码目录重新执行 `python -m pip install -e ".[dev,test]"`。
- 该环境包含 CUDA、TensorRT、ONNX Runtime GPU、PyTorch、ROS 相关 Python 包。新机器的 NVIDIA 驱动、CUDA 运行时、ROS 系统环境如果不同，个别包可能需要按新机器情况重新安装。
- 如果 `requirements-lock.txt` 中某个包安装失败，优先确认新机器的 Python 版本、系统架构、CUDA/驱动版本是否匹配。
- 当前环境 Python 版本为 `3.10.19`，Conda 文件固定为 `python=3.10`，由 Conda 在新机器上选择可用的 3.10 补丁版本。

