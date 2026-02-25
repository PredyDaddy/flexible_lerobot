# Legacy Docs Archive

本目录用于归档 **不适用于 lerobot 0.4.3 baseline** 的历史文档/方案/设计稿。

归档原则（路线 A）：
- **不移植旧仓库代码**；只保证 0.4.3 baseline 下可直接运行的文档留在 `docs/` 根目录。
- 任何依赖当前仓库缺失实现的内容（例如 `robots/agilex`、`cameras/ros_camera`、`my_ik/`、`vis/`、`lerobot_replay_ee.py` 等）统一移动到 `docs/legacy/`，避免误导“照文档执行即可跑通”。

索引入口：
- `docs/CUSTOM_DOCS_INDEX_0.4.3.md`

兼容性审计报告：
- `docs/DOCS_AUDIT_REPORT.md`

重新生成审计报告：

```bash
python3 docs/tools/audit_custom_docs.py > docs/DOCS_AUDIT_REPORT.md
```

## 从 `docs/` 根目录迁移过来的路径映射

为保持“索引与跳转”，这里给出常用的旧路径到新路径映射（等价内容，纯移动归档）：

| 旧路径(0.4.1/0.4.2 时代) | 新路径(本仓库 0.4.3) |
|---|---|
| `docs/act_pro_design/` | `docs/legacy/act_pro_design/` |
| `docs/add_agilex/` | `docs/legacy/add_agilex/` |
| `docs/add_ee/` | `docs/legacy/add_ee/` |
| `docs/async_inference_docs/` | `docs/legacy/async_inference_docs/` |
| `docs/data_vis_docs/` | `docs/legacy/data_vis_docs/` |
| `docs/ee_data_descriptions/` | `docs/legacy/ee_data_descriptions/` |
| `docs/kinematic_learning/` | `docs/legacy/kinematic_learning/` |
| `docs/move_position/` | `docs/legacy/move_position/` |
| `docs/train_agilex/数据录制.md` | `docs/legacy/train_agilex/数据录制.md` |
| `docs/agilex录制回放末端_placo版本.md` | `docs/legacy/agilex录制回放末端_placo版本.md` |
