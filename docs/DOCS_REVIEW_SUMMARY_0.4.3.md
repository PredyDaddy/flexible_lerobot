# 自定义 DOCS 审核总结（lerobot 0.4.3 / 路线 A）

- 日期：2026-02-25
- 基线：当前仓库 `pyproject.toml` 版本 = `0.4.3`
- 目标（路线 A）：**不移植旧仓库代码**；`docs/` 根目录只保留 **0.4.3 baseline 下可直接执行/可复用** 的文档，其余全部归档到 `docs/legacy/`，并保留索引与跳转。

相关入口：
- 自定义文档索引：`docs/CUSTOM_DOCS_INDEX_0.4.3.md`
- 逐文件静态审计：`docs/DOCS_AUDIT_REPORT.md`

---

## 1. 当前结论（能用 / 不能用）

以 “0.4.3 baseline 是否能直接照着执行跑通” 为标准：

- **能用（保留在 `docs/` 根目录）**：
  - 数据集检查/清洗/切分工具链（`lerobot-edit-dataset`、自检脚本、切分脚本等）
  - 训练命令与训练注意事项（`lerobot-train` 相关）
  - 新机器人集成模板（对齐了 0.4.3 的目录结构与 CLI 入口）
  - 数据格式解析类文档（偏“知识/导读”，不依赖缺失模块）

- **不能用（归档到 `docs/legacy/`）**：
  - 任何依赖当前仓库缺失实现的内容，例如：
    - `lerobot.robots.agilex`（AgileX 机器人实现）
    - `lerobot.cameras.ros_camera`（ROS 相机后端）
    - `my_ik/`、`record_ee_pinocchio.sh`、`replay_ee_pinocchio.sh`、`lerobot_replay_ee.py`（末端 IK / Pinocchio 工具链）
    - `vis.*`、`vis.sh`（旧的数据网页端可视化方案）
    - `act_pro` / `act_dinov2`（旧策略设计稿对应的实现不在 0.4.3 baseline）
    - `agilex_scripts/*`、`agilex_infer_*`（旧仓库脚本）

> 说明：`docs/legacy/` 里的文档仍然可以作为历史记录/设计参考，但不应被当作“0.4.3 可直接跑通的操作手册”。

---

## 2. 已完成的物理归档（Move 到 `docs/legacy/`）

已将以下历史目录/文件从 `docs/` 根目录迁移到 `docs/legacy/`：

- `docs/act_pro_design/` -> `docs/legacy/act_pro_design/`
- `docs/add_agilex/` -> `docs/legacy/add_agilex/`
- `docs/add_ee/` -> `docs/legacy/add_ee/`
- `docs/async_inference_docs/` -> `docs/legacy/async_inference_docs/`
- `docs/data_vis_docs/` -> `docs/legacy/data_vis_docs/`
- `docs/ee_data_descriptions/` -> `docs/legacy/ee_data_descriptions/`
- `docs/kinematic_learning/` -> `docs/legacy/kinematic_learning/`
- `docs/move_position/` -> `docs/legacy/move_position/`
- `docs/train_agilex/数据录制.md` -> `docs/legacy/train_agilex/数据录制.md`
- `docs/agilex录制回放末端_placo版本.md` -> `docs/legacy/agilex录制回放末端_placo版本.md`

路径映射表见：`docs/legacy/README.md`。

---

## 3. 对齐修订（让留在根目录的文档更“可执行”）

对根目录保留文档做了少量“只改文档、不移植代码”的对齐修订，核心点：

- 修正了部分文档里的 **旧 CLI/脚本入口**（例如 `find_port` 等）。
- 将硬编码的旧仓库路径（如 `/home/agilex/cqy/my_lerobot`）改为更通用的占位路径（`/path/to/...`）。
- 修正 `lerobot-dataset-viz` 的用法示例：当前版本建议显式传 `--display-compressed-images {true|false}`。
- `docs/核心模块解析.md` 的 robots/cameras 结构示意已对齐到 0.4.3 代码树（移除 AgileX/ros_camera 的误导结构）。

---

## 4. 不匹配的根因归类（为什么这些文档在 0.4.3 跑不通）

从 `docs/DOCS_AUDIT_REPORT.md` 的静态结果看，不匹配主要集中在以下几类：

1. **模块缺失**：文档引用了 0.4.3 baseline 中不存在的包/模块（典型如 `lerobot.robots.agilex`、`lerobot.cameras.ros_camera`、`vis.*`）。
2. **脚本缺失**：文档要求执行旧仓库脚本（典型如 `agilex_scripts/*`、`agilex_infer_*`、`vis.sh`）。
3. **代码路径变更**：文档提到的 `src/lerobot/...` 路径在 0.4.3 已不存在（例如旧策略实现目录）。
4. **工程资产缺失**：依赖旧仓库的独立工具链目录（例如 `my_ik/`、Pinocchio/PlacO 相关脚本）。

---

## 5. 如何继续（可选）

如果你后续希望把 `docs/legacy/` 中的某部分“复活”为 0.4.3 可运行文档，有两个方向：

1. 继续路线 A：只抽取“概念/流程/设计”写成 0.4.3 baseline 可复用的新文档，不引入旧代码。
2. 路线 B：把 `~/cqy/my_lerobot` 中缺失的实现按模块逐步移植到本仓库（例如先移植 `agilex` 机器人与 `ros_camera`，再处理 EE/IK 工具链）。

这两条路线的成本差异很大，建议按你的目标（只要文档可跑 vs 要把整套硬件链路跑通）来定。

