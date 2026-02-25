# 自定义文档兼容性审计报告 (lerobot 0.4.3)

- 审计日期: 2026-02-25
- 审计对象: `docs/` 下除 `docs/source/` 之外的文档/脚本/说明文件
- 基线代码: 当前仓库 `pyproject.toml` 显示版本 `0.4.3`

## 结论概览

- 扫描文件数: 67
- docs/ 根目录可直接使用 (OK): 30
- docs/ 根目录不匹配 (Not compatible): 0
- docs/legacy/ 归档文件数: 37
  - 其中静态发现不匹配点: 32

> 说明: 这是静态审计(模块/路径/CLI 入口存在性 + 明显缺失检查)，没有运行硬件/训练/回放流程。

## 逐文件初判 (按路径)

| 文件 | 状态 | 静态不匹配点(截断) |
|---|---|---|
| `docs/DOCS_REVIEW_SUMMARY_0.4.3.md` | OK | — |
| `docs/README.md` | OK | — |
| `docs/add_robots/INTEGRATING_NEW_ROBOTS.md` | OK | — |
| `docs/add_robots/add_robot_final.md` | OK | — |
| `docs/add_robots/add_robots_aug.md` | OK | — |
| `docs/add_robots/add_robots_aug_gpt.md` | OK | — |
| `docs/add_robots/add_robots_codex.md` | OK | — |
| `docs/agilex_datasets_reverse/temp_docs/agilex_datasets_reverse.md` | OK (draft) | — |
| `docs/agilex_datasets_reverse/temp_docs/swap_arm_data.py` | OK (draft) | — |
| `docs/data_clean_check/DATA_CLEANING_GUIDE.md` | OK | — |
| `docs/data_clean_check/README.md` | OK | — |
| `docs/data_clean_check/check_lerobot_v3_dataset_final.py` | OK | — |
| `docs/data_clean_check/lerobot_v3.0数据检查最终版本.md` | OK | — |
| `docs/data_handle_all_in_one/README.md` | OK | — |
| `docs/legacy/README.md` | Legacy (archived) | — |
| `docs/legacy/act_pro_design/ACT_PRO_DESIGN.md` | Legacy (archived, incompatible) | missing_src_path: src/lerobot/policies/act_pro/ |
| `docs/legacy/act_pro_design/ACT_PRO_IMPLEMENTATION_PLAN.md` | Legacy (archived, incompatible) | missing_src_path: src/lerobot/policies/act_pro; missing_src_path: src/lerobot/policies/act_pro/__init__.py; missing_src_path: src/lerobot/policies/act_pro/backbone_dinov2.py; (+3) |
| `docs/legacy/act_pro_design/ACT_PRO_REVIEW_REPORT.md` | Legacy (archived, incompatible) | missing_src_path: src/lerobot/policies/act_dinov2/; missing_src_path: src/lerobot/policies/act_dinov2/configuration_act_dinov2.py; missing_src_path: src/lerobot/policies/act_dinov2/modeling_act_dinov2.py |
| `docs/legacy/add_agilex/add_agilex_aug.md` | Legacy (archived, incompatible) | missing_import: lerobot.robots.agilex; missing_import: lerobot.robots.agilex.ros_bridge; missing_import: lerobot.scripts.record; (+12) |
| `docs/legacy/add_agilex/add_agilex_cc.md` | Legacy (archived, incompatible) | missing_import: lerobot.cameras.ros_camera.configuration_ros_camera; missing_import: lerobot.policies.act; missing_import: lerobot.robots.agilex; (+19) |
| `docs/legacy/add_agilex/add_agilex_codex.md` | Legacy (archived, incompatible) | missing_python_module: lerobot.scripts.collect_dataset; missing_python_module: lerobot.scripts.robot_check; missing_python_module: lerobot.scripts.run_policy; (+6) |
| `docs/legacy/add_agilex/add_agilex_final.md` | Legacy (archived, incompatible) | missing_import: lerobot.cameras.ros_camera.configuration_ros_camera; missing_import: lerobot.cameras.ros_camera.ros_camera; missing_import: lerobot.robots.agilex.agilex; (+15) |
| `docs/legacy/add_agilex/my_doc/agilex_test_guide.md` | Legacy (archived, incompatible) | missing_import: lerobot.robots.agilex; missing_import: lerobot.teleoperators.agilex; unknown_camera_type: ros_camera; (+2) |
| `docs/legacy/add_ee/先验_ikfk.md` | Legacy (archived, incompatible) | missing_src_path: src/lerobot/robots/agilex/agilex.py; missing_src_path: src/lerobot/robots/agilex/config_agilex.py; missing_src_path: src/lerobot/robots/agilex/piper_pinocchio_kinematics.py; (+2) |
| `docs/legacy/add_ee/先验ik_fk_plan.md` | Legacy (archived, incompatible) | missing_python_module: lerobot.scripts.lerobot_replay_ee; missing_src_path: src/lerobot/robots/agilex/; unknown_robot_type: agilex |
| `docs/legacy/agilex录制回放末端_placo版本.md` | Legacy (archived, incompatible) | missing_python_module: lerobot.scripts.lerobot_replay_ee; missing_src_path: src/lerobot/robots/agilex/; missing_src_path: src/lerobot/robots/agilex/agilex.py; (+2) |
| `docs/legacy/async_inference_docs/temp_docs/异步推理方案1.md` | Legacy (archived, incompatible) | missing_script: agilex_infer_async_client.py |
| `docs/legacy/async_inference_docs/temp_docs/异步推理方案2.md` | Legacy (archived, draft) | — |
| `docs/legacy/async_inference_docs/temp_docs/异步推理方案3.md` | Legacy (archived, incompatible) | missing_script: agilex_infer_single_cc_vertical.py |
| `docs/legacy/async_inference_docs/temp_docs/异步推理方案4.md` | Legacy (archived, incompatible) | missing_script: agilex_infer_async_client.py |
| `docs/legacy/async_inference_docs/使用记录.md` | Legacy (archived, incompatible) | missing_script: agilex_scripts/agilex_async_infer_single_cc_vertical.py |
| `docs/legacy/async_inference_docs/异步推理方案3.md` | Legacy (archived, incompatible) | missing_script: agilex_infer_single_cc_vertical.py |
| `docs/legacy/async_inference_docs/异步推理方案_plan.md` | Legacy (archived, incompatible) | missing_script: agilex_infer_single_cc_vertical.py; missing_script: agilex_scripts/agilex_async_infer_single_cc_horizal.py; missing_script: agilex_scripts/agilex_async_infer_single_cc_vertical.py; (+1) |
| `docs/legacy/async_inference_docs/异步推理最终报告.md` | Legacy (archived, incompatible) | missing_script: agilex_infer_single_cc_vertical.py; missing_script: agilex_scripts/agilex_async_infer_single_cc_horizal.py; missing_script: agilex_scripts/agilex_async_infer_single_cc_vertical.py |
| `docs/legacy/async_inference_docs/服务端异步推理文档2.md` | Legacy (archived, incompatible) | missing_script: agilex_scripts/agilex_async_infer_single_cc_horizal.py; missing_script: agilex_scripts/agilex_async_infer_single_cc_vertical.py |
| `docs/legacy/data_vis_docs/数据网页端可视化_plan.md` | Legacy (archived, incompatible) | missing_python_module: vis.scripts.precompute; missing_python_module: vis.web.main |
| `docs/legacy/data_vis_docs/数据网页端可视化方案1.md` | Legacy (archived, incompatible) | missing_python_module: vis.run |
| `docs/legacy/data_vis_docs/数据网页端可视化方案2.md` | Legacy (archived, incompatible) | missing_python_module: vis.web.main |
| `docs/legacy/data_vis_docs/数据网页端可视化最终方案.md` | Legacy (archived, incompatible) | missing_python_module: vis.web.main |
| `docs/legacy/ee_data_descriptions/EE_DATASET_SPEC.md` | Legacy (archived, incompatible) | missing_src_path: src/lerobot/robots/agilex/agilex.py |
| `docs/legacy/ee_data_descriptions/temp_docs/末端lerobot数据解析1.md` | Legacy (archived, incompatible) | unknown_robot_type: agilex |
| `docs/legacy/ee_data_descriptions/temp_docs/末端lerobot数据解析2.md` | Legacy (archived, draft) | — |
| `docs/legacy/ee_data_descriptions/temp_docs/末端lerobot数据解析3.md` | Legacy (archived, incompatible) | unknown_robot_type: agilex |
| `docs/legacy/ee_data_descriptions/temp_docs/末端lerobot数据解析4.md` | Legacy (archived, incompatible) | missing_src_path: src/lerobot/robots/agilex/agilex.py |
| `docs/legacy/kinematic_learning/final.md` | Legacy (archived, incompatible) | missing_src_path: src/lerobot/robots/agilex/; missing_src_path: src/lerobot/robots/agilex/agilex.py; missing_src_path: src/lerobot/robots/agilex/config_agilex.py; (+1) |
| `docs/legacy/kinematic_learning/temp_docs/零基础学运控1.md` | Legacy (archived, incompatible) | missing_python_module: lerobot.scripts.lerobot_replay_ee; missing_src_path: src/lerobot/robots/agilex/agilex.py; missing_src_path: src/lerobot/robots/agilex/config_agilex.py; (+2) |
| `docs/legacy/kinematic_learning/temp_docs/零基础学运控2.md` | Legacy (archived, draft) | — |
| `docs/legacy/kinematic_learning/temp_docs/零基础学运控3.md` | Legacy (archived, draft) | — |
| `docs/legacy/kinematic_learning/temp_docs/零基础学运控4.md` | Legacy (archived, incompatible) | missing_python_module: lerobot.scripts.lerobot_replay_ee; missing_src_path: src/lerobot/robots/agilex/agilex.py; missing_src_path: src/lerobot/robots/agilex/config_agilex.py; (+1) |
| `docs/legacy/move_position/align_position.py` | Legacy (archived, incompatible) | missing_import: lerobot.cameras.ros_camera; missing_import: lerobot.cameras.ros_camera.configuration_ros_camera; unknown_camera_type: ros_camera |
| `docs/legacy/train_agilex/数据录制.md` | Legacy (archived, incompatible) | unknown_camera_type: ros_camera; unknown_robot_type: agilex; unknown_teleop_type: agilex_teleop |
| `docs/lerobot_21数据格式解析` | OK | — |
| `docs/lerobot_data_temp/lerobot数据格式_aug.md` | OK | — |
| `docs/lerobot_data_temp/lerobot数据格式_cc.md` | OK | — |
| `docs/lerobot_data_temp/lerobot数据格式_codex.md` | OK | — |
| `docs/lerobot_v3数据格式解析.md` | OK | — |
| `docs/lerobot数据格式解析.md` | OK | — |
| `docs/split_datasets_joint_ee/split_ee_or_joint_dataset.py` | OK | — |
| `docs/split_datasets_joint_ee/使用说明.md` | OK | — |
| `docs/split_datasets_joint_ee/数据切分报告.md` | OK | — |
| `docs/train_agilex/train_agilex_guide.md` | OK | — |
| `docs/train_agilex/train_groot_guide.md` | OK | — |
| `docs/train_agilex/train_pi05_guide.md` | OK | — |
| `docs/train_agilex/train_smolvla_guide.md` | OK | — |
| `docs/train_agilex/数据检查指南.md` | OK | — |
| `docs/核心模块解析.md` | OK | — |
| `docs/通用添加机器人.md` | OK | — |

## 缺失引用 Top 列表 (用于集中修复)

- `unknown_robot_type: agilex` (出现 11 次)
- `missing_src_path: src/lerobot/robots/agilex/agilex.py` (出现 9 次)
- `missing_src_path: src/lerobot/robots/agilex/config_agilex.py` (出现 8 次)
- `unknown_camera_type: ros_camera` (出现 6 次)
- `missing_src_path: src/lerobot/robots/agilex/` (出现 5 次)
- `missing_src_path: src/lerobot/scripts/lerobot_replay_ee.py` (出现 5 次)
- `unknown_teleop_type: agilex_teleop` (出现 4 次)
- `missing_python_module: lerobot.scripts.lerobot_replay_ee` (出现 4 次)
- `missing_script: agilex_infer_single_cc_vertical.py` (出现 4 次)
- `missing_script: agilex_scripts/agilex_async_infer_single_cc_vertical.py` (出现 4 次)
- `missing_import: lerobot.robots.agilex` (出现 3 次)
- `missing_import: lerobot.cameras.ros_camera.configuration_ros_camera` (出现 3 次)
- `missing_script: agilex_scripts/agilex_async_infer_single_cc_horizal.py` (出现 3 次)
- `missing_python_module: vis.web.main` (出现 3 次)
- `missing_import: lerobot.teleoperators.agilex` (出现 2 次)
- `missing_script: collect_data.py` (出现 2 次)
- `missing_src_path: src/lerobot/cameras/ros_camera/configuration_ros_camera.py` (出现 2 次)
- `missing_src_path: src/lerobot/cameras/ros_camera/ros_camera.py` (出现 2 次)
- `missing_src_path: src/lerobot/teleoperators/agilex_leader/` (出现 2 次)
- `unknown_teleop_type: agilex_leader` (出现 2 次)
- `missing_script: agilex_infer_async_client.py` (出现 2 次)
- `missing_src_path: src/lerobot/policies/act_pro/` (出现 1 次)
- `missing_src_path: src/lerobot/policies/act_pro` (出现 1 次)
- `missing_src_path: src/lerobot/policies/act_pro/__init__.py` (出现 1 次)
- `missing_src_path: src/lerobot/policies/act_pro/backbone_dinov2.py` (出现 1 次)
- `missing_src_path: src/lerobot/policies/act_pro/configuration_act_pro.py` (出现 1 次)
- `missing_src_path: src/lerobot/policies/act_pro/modeling_act_pro.py` (出现 1 次)
- `missing_src_path: src/lerobot/policies/act_pro/temporal_fusion.py` (出现 1 次)
- `missing_src_path: src/lerobot/policies/act_dinov2/` (出现 1 次)
- `missing_src_path: src/lerobot/policies/act_dinov2/configuration_act_dinov2.py` (出现 1 次)
- `missing_src_path: src/lerobot/policies/act_dinov2/modeling_act_dinov2.py` (出现 1 次)
- `missing_import: lerobot.robots.agilex.ros_bridge` (出现 1 次)
- `missing_import: lerobot.scripts.record` (出现 1 次)
- `missing_src_path: src/lerobot/robots/agilex/__init__.py` (出现 1 次)
- `missing_src_path: src/lerobot/robots/agilex/constants.py` (出现 1 次)
- `missing_src_path: src/lerobot/robots/agilex/robot_agilex.py` (出现 1 次)
- `missing_src_path: src/lerobot/robots/agilex/ros_bridge.py` (出现 1 次)
- `missing_src_path: src/lerobot/robots/bi_so100_follower/` (出现 1 次)
- `missing_src_path: src/lerobot/teleoperators/agilex/__init__.py` (出现 1 次)
- `missing_src_path: src/lerobot/teleoperators/agilex/teleop_agilex.py` (出现 1 次)
