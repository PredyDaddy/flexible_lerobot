# Prompt For Next Agent

你将接手仓库：

```text
/data/cqy_workspace/flexible_lerobot
```

这是一个基于 Hugging Face LeRobot 改造的机器人框架。用户主要开发在 `my_devs/`，文档按功能点放在 `my_devs/docs/`。请先阅读：

```text
AGENTS.md
my_devs/docs/handoff/flexible_lerobot_handoff_20260622/START_HERE.md
my_devs/docs/handoff/flexible_lerobot_handoff_20260622/CURRENT_STATE.md
my_devs/docs/handoff/flexible_lerobot_handoff_20260622/PROJECT_MAP.md
my_devs/docs/handoff/flexible_lerobot_handoff_20260622/COMMANDS.md
```

硬性规则：

- 必须使用 `lerobot_flex` conda 环境。
- 所有测试、格式化、lint、脚本执行都用 `conda run -n lerobot_flex ...`。
- 默认只在 `my_devs/` 下开发。
- 不要直接改 `src/lerobot/`，除非用户明确授权。
- 不要回滚或删除你没有创建的文件。
- 真实机器人上机前必须先做 dry-run / mock / consistency，并核对串口、相机、标定、动作维度和安全阈值。

当前分支和状态：

```text
branch: feature/cqy
HEAD: c25e170 pi05trt后端推理
```

当前有未提交/未跟踪内容，不要删除：

```text
.gitignore
my_devs/docs/add_robot/agilex/按照_agilex_配置新机器人详解.md
my_devs/docs/vla_engineering/
my_devs/flash_RT_pi/
my_devs/openpi_trt/docs/lerobot_pi05_trt难点分析.md
my_devs/openpi_trt/docs/openpi_trt实现链路报告.md
```

其中 `.gitignore` 的未提交修改是新增 FlashRT 参考源码忽略规则；`my_devs/flash_RT_pi/reference_source_code/FlashRT-main/` 可能显示为 ignored。

如果用户让你继续某条技术线，优先从这些入口读起：

- ACT TRT：`my_devs/new_act_trt/README.md`
- GR00T TRT：`my_devs/groot_trt/README.md`
- PI0.5 TRT：`my_devs/openpi_trt/README.md` 和 `my_devs/openpi_trt/docs/openpi_trt实现链路报告.md`
- VLA 工程：`my_devs/docs/vla_engineering/工作报告.md`
- FlashRT 参考：`my_devs/flash_RT_pi/reference_source_code/FlashRT-main/README.md`
- AgileX 采集：`my_devs/agilex_web_collection/README.md`
- ROS HDF5 到 LeRobot：`my_devs/web_collection/README.md`

开工前建议执行：

```bash
cd /data/cqy_workspace/flexible_lerobot
git status --short --branch
conda run -n lerobot_flex python --version
conda run -n lerobot_flex python -c "import lerobot; print('lerobot import ok')"
conda run -n lerobot_flex python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

当前 PI0.5 TensorRT 主线建议理解为：

```text
prefix_cache TensorRT engine
  + denoise_step TensorRT engine
  + Python 10-step denoise loop
  -> patch policy.model.sample_actions(...)
```

不要默认从完整 `sample_actions(...)` 单体 TensorRT engine 开始做，除非用户明确要求。
