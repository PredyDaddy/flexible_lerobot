# Custom / Copied Docs Index (lerobot 0.4.3 baseline)

本文件是对当前仓库 `docs/`(除 `docs/source/` 之外)里“从旧仓库拷贝过来的内部文档/脚本”的索引与兼容性说明。

目标选择: **路线 A** (不移植旧仓库代码，只让文档对齐 0.4.3 现状，可运行的能直接跑，不可运行的明确标注为历史/方案/设计)。

审计基线:
- 当前仓库版本: `pyproject.toml` -> `0.4.3`
- 旧仓库参考: `/home/agilex/cqy/my_lerobot` (`pyproject.toml` 显示为 `0.4.2`)

配套审计报告:
- 逐文件静态审计明细见: `docs/DOCS_AUDIT_REPORT.md`
- 总结报告（能用/不能用 + 归档说明）: `docs/DOCS_REVIEW_SUMMARY_0.4.3.md`

如何重新生成审计报告（静态检查）：

```bash
python3 docs/tools/audit_custom_docs.py > docs/DOCS_AUDIT_REPORT.md
```

## 你应该怎么用这份索引

我把自定义文档分成三类:

- **可直接用(0.4.3 可跑通)**: 文档描述与 0.4.3 代码树/CLI 对齐，不依赖旧仓库独有模块。
- **需要改文档(不改代码)**: 主要是“重命名/目录结构变化/脚本入口变化”，只要把文档里的路径/命令改成 0.4.3 的就能用。
- **历史/方案/设计(0.4.3 不可直接用)**: 依赖旧仓库缺失的实现(例如 AgileX/ROS camera/末端 IK 工具链/vis 可视化服务等)，在不移植代码的前提下无法跑通。

## 0.4.3 的 CLI 入口速查

0.4.3 的脚本入口在 `pyproject.toml` 的 `[project.scripts]`，核心命令包括:

- `lerobot-record` / `python -m lerobot.scripts.lerobot_record`
- `lerobot-replay` / `python -m lerobot.scripts.lerobot_replay`
- `lerobot-train` / `python -m lerobot.scripts.lerobot_train`
- `lerobot-eval` / `python -m lerobot.scripts.lerobot_eval`
- `lerobot-dataset-viz` / `python -m lerobot.scripts.lerobot_dataset_viz`
- `lerobot-edit-dataset` / `python -m lerobot.scripts.lerobot_edit_dataset`
- `lerobot-find-port` / `python -m lerobot.scripts.lerobot_find_port`
- 以及 `lerobot-info`、`lerobot-calibrate`、`lerobot-teleoperate` 等

`lerobot-dataset-viz` 的一个容易踩坑点：
- 当前版本的 `lerobot-dataset-viz` 需要显式传 `--display-compressed-images {true|false}`（即便你想用默认值，也建议写上）。

旧文档中常见的“旧入口”在 0.4.3 需要替换:

- `python -m lerobot.scripts.find_port` -> `lerobot-find-port`
- `python -m lerobot.scripts.collect_dataset` -> **通常对应** `lerobot-record`
- `python -m lerobot.scripts.run_policy` -> 0.4.3 baseline **没有同名入口**:
  - 环境评测: 用 `lerobot-eval`
  - 对真实机器人跑策略: 需要你写一个小脚本(或后续走路线 B 移植旧实现)
- `python -m lerobot.scripts.robot_check` -> 0.4.3 baseline **没有同名入口**:
  - 可用 `lerobot-info`/`lerobot-calibrate`/`lerobot-find-port` 等替代部分检查项

## ✅ 可直接用(0.4.3 可跑通)

数据集工具链(推荐优先看):
- `docs/data_clean_check/DATA_CLEANING_GUIDE.md` (围绕 `lerobot-edit-dataset`)
- `docs/data_clean_check/README.md`
- `docs/data_clean_check/lerobot_v3.0数据检查最终版本.md`
- `docs/data_handle_all_in_one/README.md` (把“检查/可视化/清洗/切分”串成一条流水线)
- `docs/split_datasets_joint_ee/split_ee_or_joint_dataset.py`
- `docs/split_datasets_joint_ee/使用说明.md`
- `docs/split_datasets_joint_ee/数据切分报告.md`

训练命令(命令本身可用，但示例数据/维度是 AgileX 数据集假设):
- `docs/train_agilex/train_agilex_guide.md`
- `docs/train_agilex/train_groot_guide.md`
- `docs/train_agilex/train_pi05_guide.md`
- `docs/train_agilex/train_smolvla_guide.md`
- `docs/train_agilex/数据检查指南.md`

架构理解(偏“代码导读”，不保证所有细节是 API 承诺):
- `docs/核心模块解析.md`

数据集修复/小工具(与 AgileX 机器人实现无关，只操作数据集文件):
- `docs/agilex_datasets_reverse/temp_docs/swap_arm_data.py`
- `docs/agilex_datasets_reverse/temp_docs/agilex_datasets_reverse.md`

## ✅ 新机器人集成模板 (已对齐 0.4.3)

以下文档已经在本仓库做过一次“0.4.3 对齐修订”，可以直接作为模板使用：

- `docs/add_robots/INTEGRATING_NEW_ROBOTS.md`
- `docs/add_robots/add_robot_final.md`
- `docs/add_robots/add_robots_aug.md`
- `docs/add_robots/add_robots_aug_gpt.md`
- `docs/add_robots/add_robots_codex.md`
- `docs/通用添加机器人.md`

关键注意点（容易踩坑）:
- 0.4.3 中 SO100/SO101 的代码目录是 `src/lerobot/robots/so_follower/`，但 `RobotConfig` 的 `type` 仍然注册为 `so100_follower` / `so101_follower`。
- 0.4.3 中 SO100/SO101 的 teleoperator 目录是 `src/lerobot/teleoperators/so_leader/`，但 `TeleoperatorConfig` 的 `type` 仍为 `so100_leader` / `so101_leader`。
- `find_port` 的 python -m 入口是 `python -m lerobot.scripts.lerobot_find_port`，对应 CLI 为 `lerobot-find-port`。

## ❌ 历史/方案/设计(0.4.3 baseline 不可直接用)

这类文档在不移植旧仓库代码的前提下无法跑通，但可以保留为历史记录/设计参考。

归档位置：`docs/legacy/`（说明见 `docs/legacy/README.md`）。

AgileX / ROS Camera 相关(0.4.3 缺少 `lerobot.robots.agilex` 与 `lerobot.cameras.ros_camera`):
- `docs/legacy/add_agilex/*`
- `docs/legacy/async_inference_docs/*` (依赖 `agilex_scripts/*` 等旧仓库脚本)
- `docs/legacy/move_position/align_position.py`
- `docs/legacy/train_agilex/数据录制.md`

末端 IK / Pinocchio 工具链(0.4.3 缺少 `my_ik/`、`record_ee_pinocchio.sh`、`replay_ee_pinocchio.sh`、`lerobot_replay_ee.py`):
- `docs/legacy/add_ee/*`
- `docs/legacy/kinematic_learning/*`
- `docs/legacy/agilex录制回放末端_placo版本.md`

可视化 Web 方案(0.4.3 baseline 没有 `vis.*` 包与 `vis.sh`):
- `docs/legacy/data_vis_docs/*`

策略设计文档(0.4.3 baseline 缺少 `act_pro`/`act_dinov2` 实现):
- `docs/legacy/act_pro_design/*`

## 推荐工作顺序(路线 A)

如果你要尽快得到“能直接照着跑”的文档集合:

1. 先以 `docs/data_clean_check/*` + `docs/split_datasets_joint_ee/*` 打通数据处理闭环。
2. 再用 `docs/train_agilex/*` 里的命令在你自己的 dataset repo_id 上训练验证。
3. `docs/legacy/async_inference_docs/*` 适合作为部署/工程化参考，但其示例依赖旧仓库脚本，不能在 0.4.3 baseline 直接跑通。
4. `docs/legacy/*` 下的 `add_agilex` / `add_ee` / `data_vis_docs` 等先统一视为“历史/方案”，避免误导新同事按文档直接执行。
