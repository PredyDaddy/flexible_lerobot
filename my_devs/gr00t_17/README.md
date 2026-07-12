# GR00T N1.7 SO101 工程入口

本目录保存 SO101 使用同一套桌面清理数据训练 GR00T N1.7，并完成真机部署、RTC 与
TensorRT 加速所需的代码、配置、测试和文档。

完整工作过程、复现顺序、实际参数、结果证据与故障排查统一记录在：

- [`doc/GR00T_N1_7_SO101_END_TO_END_GUIDE.md`](doc/GR00T_N1_7_SO101_END_TO_END_GUIDE.md)

专题文档包括：

- [`doc/GR00T_N1_7_SO101_TRAINING_WORK_PLAN.md`](doc/GR00T_N1_7_SO101_TRAINING_WORK_PLAN.md)：训练工作计划；
- [`doc/GR00T_N1_7_SO101_TECHNICAL_DESIGN.md`](doc/GR00T_N1_7_SO101_TECHNICAL_DESIGN.md)：数据、模型和训练技术设计；
- [`doc/GR00T_N1_7_SO101_TRAINING_RUNBOOK.md`](doc/GR00T_N1_7_SO101_TRAINING_RUNBOOK.md)：训练操作手册；
- [`doc/GR00T_N1_7_SO101_EXECUTION_REPORT.md`](doc/GR00T_N1_7_SO101_EXECUTION_REPORT.md)：实际执行与验收报告；
- [`doc/GR00T_N1_7_SO101_ROBOT_DEPLOYMENT.md`](doc/GR00T_N1_7_SO101_ROBOT_DEPLOYMENT.md)：基础真机部署；
- [`doc/GR00T_N1_7_SO101_RTC_DEPLOYMENT.md`](doc/GR00T_N1_7_SO101_RTC_DEPLOYMENT.md)：RTC Server/Client 部署；
- [`doc/GR00T_N1_7_SO101_TENSORRT.md`](doc/GR00T_N1_7_SO101_TENSORRT.md)：TensorRT 导出、构建与验证。

## 仓库边界

Git 中只保存可复现流程所需的小文件：源码、Shell 入口、modality 配置、测试、文档、小型
manifest 和验证报告。以下内容由 `.gitignore` 强制排除：

- 原始数据和转换后的 parquet/video；
- 基础模型、正式 checkpoint、optimizer state；
- ONNX、TensorRT engine 和校准产物；
- 虚拟环境、依赖缓存、构建缓存、日志和真机输出；
- 用户提供的 reference 副本。

本任务的所有生成物必须位于 `my_devs/gr00t_17` 内。原始数据、原始权重与 reference 仅作为
只读输入，不允许原地修改。

## 环境要求

仓库级开发、测试和脚本检查统一使用 `lerobot_flex` Conda 环境：

```bash
cd /data/cqy_workspace/flexible_lerobot
/home/cqy/miniconda3/bin/conda run --no-capture-output -n lerobot_flex \
  pytest -q my_devs/gr00t_17/tests
```

训练与 TensorRT 构建会进一步使用本目录中的隔离 runtime；具体准备命令和执行顺序见完整工程记录。
