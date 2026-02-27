# Cobot Magic 代码阅读与并入 LeRobot 前理解

更新时间：2026-02-25  
目标：为后续把 `my_devs/cobot_magic` 并入当前 `flexible_lerobot` 仓库提供技术基线。

## 0. 先回答你的第一个问题：当前仓库 LeRobot 数据格式版本

当前这个仓库使用的 LeRobot 数据格式（codebase 版本）是 **`v3.0`**。

证据：
- `src/lerobot/datasets/lerobot_dataset.py:80` 定义了 `CODEBASE_VERSION = "v3.0"`。
- `docs/source/lerobot-dataset-v3.mdx:1` 文档标题就是 `LeRobotDataset v3.0`。
- 仓库内提供了从 `v2.1` 迁移到 `v3.0` 的脚本：`src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py`。

结论：如果要把 Cobot Magic 的数据接入当前仓库，目标格式应直接对齐到 **LeRobot v3.0**。

---

## 1. 本次阅读范围与边界

### 1.1 已重点逐文件阅读的模块（自研核心）
- `collect_data/`（采集、回放、可视化、docs 变体、piper sdk demo）
- `Piper_ros_private-ros-noetic/`（CAN 脚本、机械臂 ROS 节点、launch、msg、仿真桥接）
- `tools/`（构建、串口/CAN、本地 lerobot 安装切换）
- `aloha-devel/act/`（train、inference、policy、utils、detr）

### 1.2 已按入口/模式阅读的模块（第三方为主）
- `camera_ws/src/realsense-ros` 与 `camera_ws/src/ros_astra_camera`
  - `scripts/*.py`、`launch/*.launch` 已逐项过一遍
  - 结论：以上游驱动代码为主，项目自定义主要体现在多相机 launch 参数与序列号配置
- `aloha-devel/robomimic`
  - 重点读了训练入口、ACT 对接点、模板与 config 生成脚本
  - 大量核心训练框架文件为上游 robomimic，建议按“边界接入”而不是重构

### 1.3 关键现实约束
- `my_devs/cobot_magic` 总文件数大（约 2000+），其中大量为 catkin build 产物和第三方代码。
- “后续并入 lerobot”真正高价值的是自研管线与接口边界，不建议把相机驱动/robomimic 底层整包迁入主仓。

---

## 2. 我理解的 Cobot Magic 总体架构

这套项目是一个“**双臂主从遥操作 + 多相机 + 数据采集回放 + 模仿学习训练推理**”链路，核心是：

1. `Piper ROS` 节点把机械臂 CAN 数据转 ROS 话题（主臂、从臂、状态、末端位姿）。
2. `camera_ws` 提供 RGB / 深度话题（RealSense 或 Astra）。
3. `collect_data.py` 做多源时间对齐并落盘为自定义 HDF5 episode。
4. `aloha-devel/act` 读取该 HDF5，训练 ACT/CNNMLP/Diffusion，并在 ROS 中在线推理发布关节控制。
5. `replay_data.py` / `visualize_episodes.py` 做离线回放和可视化检查。

可以概括为：

`硬件 -> ROS话题 -> 自定义HDF5 -> 训练 -> ROS推理控制`

---

## 3. 端到端数据/控制流（按运行顺序）

### 3.1 机械臂控制层（Piper）

核心节点：`piper_start_ms_node.py`

- 文件头已写明模式语义：`my_devs/cobot_magic/Piper_ros_private-ros-noetic/src/piper/scripts/piper_start_ms_node.py:3`。
  - `mode=0`：读取并发布主/从臂状态
  - `mode=1`：接收控制命令驱动从臂
- 双臂启动由 `start_ms_piper.launch` 完成：
  - 左臂 remap 到 `/master/joint_left` 与 `/puppet/joint_left`
  - 右臂 remap 到 `/master/joint_right` 与 `/puppet/joint_right`
  - 见：`my_devs/cobot_magic/Piper_ros_private-ros-noetic/src/piper/launch/start_ms_piper.launch`

单位转换在代码里是硬编码的：
- 关节：`deg*1000 -> rad` 近似用 `*0.017444`
- 反向控制：`rad -> SDK` 用 `57324.840764`
- 夹爪尺度：`*1e6` 并限幅到 `[0, 80000]`
- 相关行可见：`.../piper_start_ms_node.py:197`、`...:291`、`...:308`。

### 3.2 采集层（collect_data）

主脚本：`my_devs/cobot_magic/collect_data/collect_data.py`

做了这些事：
- 订阅三路 RGB、可选三路深度、主臂 joint、从臂 joint、底盘 odom。
- 用 `deque` 做近似时间同步：取各话题最新消息最小时间戳作为 `frame_time`，弹出更旧消息。
- 每帧构造：
  - `obs.images` = RGB
  - `obs.images_depth`（可选）
  - `obs.qpos/qvel/effort` = 从臂状态
  - `action` = 主臂位置（示教动作）
  - `base_vel` = 底盘速度（可选）
- 最终保存为 `episode_x.hdf5`。

### 3.3 回放/可视化层

- `replay_data.py`：
  - 读取 HDF5
  - 发布图像、主臂/从臂 joint、底盘速度
  - 支持仅发布主臂轨迹模式
- `visualize_episodes.py`：
  - 生成拼接视频
  - 绘制 `qpos` 与 `base_action` 曲线图

### 3.4 训练/推理层（aloha-devel/act）

训练入口：`my_devs/cobot_magic/aloha-devel/act/train.py`
- 调 `utils.load_data` 读取 HDF5 episode
- 支持 `ACT / CNNMLP / Diffusion`
- 保存 `policy_best.ckpt` 与 `dataset_stats.pkl`

推理入口：`my_devs/cobot_magic/aloha-devel/act/inference.py`
- 订阅图像、从臂状态、底盘状态
- 模型输出动作后发布到关节命令话题
- 支持 chunk + temporal aggregation + 插值发布

---

## 4. 数据格式细节（当前 Cobot Magic）

### 4.1 主版本采集脚本的 HDF5 结构

`collect_data/collect_data.py` 产物：

```text
episode_x.hdf5
├── attrs:
│   ├── sim: False
│   └── compress: False
├── /observations/qpos            [T, 14]
├── /observations/qvel            [T, 14]
├── /observations/effort          [T, 14]
├── /observations/images/<cam>    [T, 480, 640, 3] uint8
├── /observations/images_depth/<cam>  [T, 480, 640] uint16 (可选)
├── /action                       [T, 14]   (主臂位置)
└── /base_action                  [T, 2]    (底盘线速度/角速度)
```

语义重点：
- `qpos/qvel/effort` 来自 **从臂反馈**。
- `action` 来自 **主臂状态**（示教输入）。

### 4.2 docs 变体采集脚本的结构差异

`collect_data/docs/collect_data.py` 与主版本有关键不一致：

- 深度键名使用 `/observations/depths/<cam>`（不是 `images_depth`）。
- 可选 JPEG 压缩，并有 `compress_len`（仅 RGB）机制。
- 这会与 `act/utils.py` 默认读取 `/observations/images_depth/...` 产生不兼容。

---

## 5. 目录与脚本职责清单（按后续并入价值排序）

## 5.1 `collect_data/`

- `collect_data.py`：主采集脚本，时间同步 + HDF5 落盘。
- `replay_data.py`：HDF5 回放到 ROS 话题。
- `visualize_episodes.py`：视频与曲线可视化工具。
- `requirements.txt`：采集环境依赖。

`collect_data/docs/`：
- `collect_data.py`：早期/文档版采集脚本（含压缩、不同 depth 键）。
- `env.sh`：系统/ROS 安装辅助脚本。
- `env2.sh`：依赖安装与 workspace 初始化辅助脚本。
- `test.sh`：ROS 安装检测与交互脚本。

`collect_data/piper_sdk_demo/`：
- `piper_disable.py`：机械臂失能 demo。
- `piper_joint_ctrl.py`：关节控制 demo。
- `piper_master_config.py`：设置主臂模式。
- `piper_slave_config.py`：设置从臂模式。
- `piper_read_arm_motor_max_angle_spd.py`：读取电机角速度上限。
- `piper_set_mit.py`：切 MIT 控制模式。
- `piper_status.py`：状态读取 demo。
- `can_activate.sh` / `can_config.MD` / `README*.MD`：CAN 与 SDK 使用说明。

## 5.2 `Piper_ros_private-ros-noetic/`

CAN/系统脚本：
- `can_activate.sh`：单模块或指定 USB 地址激活 CAN。
- `can_config.sh`：多模块按 USB bus-info 映射重命名并激活。
- `find_all_can_port.sh`：列出系统 CAN 接口与 USB 端口映射。

机械臂 ROS 节点：
- `src/piper/scripts/piper_ctrl_single_node.py`：单臂控制与状态发布。
- `src/piper/scripts/piper_start_ms_node.py`：双臂主从节点（mode 0/1）。
- `src/piper/scripts/piper_read_master_node.py`：只读主臂控制消息。

launch：
- `start_ms_piper.launch`：双臂节点启动 + topic remap。
- `start_single_piper.launch`：单臂控制启动。
- `start_single_piper_rviz.launch`：单臂 + RViz 控制。

描述与仿真：
- `piper_description/launch/*.launch`：RViz/Gazebo 组合启动。
- `piper_description/scripts/rviz_ctrl_gazebo.py`：`/joint_states` -> Gazebo controller relay。
- `piper_description/config/*.yaml`：关节控制器参数与关节名。

消息：
- `piper_msgs/msg/PiperStatusMsg.msg`：机械臂状态。
- `piper_msgs/msg/PosCmd.msg`：末端位姿控制命令。

## 5.3 `camera_ws/`

结论：本目录绝大多数是上游相机驱动工程（RealSense + Astra）。

关键可执行脚本：
- `src/delete_build_file.sh`：清理 catkin build 产物。
- `realsense2_camera/scripts/*.py`：监听、测试、TF 调参、深度中心点 demo。
- `ros_astra_camera/scripts/*.py`：同步测试、点云距离、视频模式查询等小工具。

关键 launch 形态：
- `realsense2_camera/launch/rs_camera.launch`：单机参数模板。
- `realsense2_camera/launch/rs_multiple_devices.launch` / `multi_camera.launch`：多机序列号编排。
- `ros_astra_camera/launch/*.launch`：不同型号预设（`astra`、`dabai*`、`gemini*`、`embedded*`）。
- `ros_astra_camera/launch/multi_*.launch`：多机封装（通过 include + serial）。

## 5.4 `aloha-devel/act/`

- `train.py`：训练入口，组装 policy config、训练循环、ckpt 输出。
- `inference.py`：在线推理 ROS 节点（取帧、模型推理、关节发布）。
- `policy.py`：ACT/CNNMLP/Diffusion 的统一封装。
- `utils.py`：HDF5 dataset 读取、统计量计算、DataLoader。
- `train.sh`：训练命令样例。

`act/detr/`：
- `models/detr_vae.py`：ACT 核心（视觉 backbone + transformer + latent）。
- `models/backbone.py`：ResNet backbone + `DepthNet`。
- `models/transformer.py`：Transformer 编解码层。
- `main.py`：模型与优化器构建入口。
- `util/*`：DETR 工具模块（主要为上游代码）。

## 5.5 `aloha-devel/robomimic/`

此目录主要是上游 robomimic 框架镜像，和 Cobot Magic 强绑定点集中在：

- `scripts/train.py`：训练主入口。
- `algo/act.py`：ACT 算法适配层（调用 `act.detr.main`）。
- `config/act_config.py` 与 `exps/templates/act.json`：ACT 配置。
- `scripts/config_gen/act_gen.py`：ACT 配置批量生成。

其余 `algo/*`、`models/*`、`utils/*`、`conversion/*` 基本是通用 robomimic 组件。

## 5.6 `tools/`

- `build.sh`：编译 camera_ws（并尝试 remote_control）。
- `can.sh`：通过固定 `/dev/canable*` 启多个 slcand 接口。
- `arm_serial.sh`：检查机械臂 USB serial 与 udev 规则。
- `camera_serial.sh` / `rs_camera_serial.sh`：Astra/RealSense 序列号查询。
- `requirements.sh`：系统依赖安装。
- `cobot_magic_env.sh`：环境变量模板（序列号等）。
- `arx_can.rules`：udev 规则示例。
- `install_local_lerobot.sh`：安装本地 editable lerobot。
- `use_local_lerobot.sh`：当前 shell 注入本地 lerobot `PYTHONPATH`。

---

## 6. 已识别的问题与风险（建议集成前先处理）

以下问题都已在源码中定位到明确位置。

### 6.1 采集主臂 topic 默认值疑似错误（高优先级）

在 `collect_data.py` 参数默认值中，`master_arm_*_topic` 与 `puppet_arm_*_topic` 都指向 `/puppet/...`：
- `my_devs/cobot_magic/collect_data/collect_data.py:410`
- `my_devs/cobot_magic/collect_data/collect_data.py:412`

影响：
- 若不手动传参，会导致 `action`（主臂）和 `qpos`（从臂）来源混淆，示教数据语义错误。

### 6.2 深度图 padding 逻辑在 `use_depth_image=False` 时会崩溃（高优先级）

`img_right_depth` 和 `img_front_depth` 的 `copyMakeBorder` 在 if 外执行：
- `my_devs/cobot_magic/collect_data/collect_data.py:206`
- `my_devs/cobot_magic/collect_data/collect_data.py:214`

当 `use_depth_image=False` 时，变量为 `None`，会触发 OpenCV 异常。

### 6.3 docs 版采集脚本存在明显取队列错误（高优先级）

- 变量定义笔误：`imgl, imgr, imgl, depthl, depthr, depthl`  
  见 `my_devs/cobot_magic/collect_data/docs/collect_data.py:262`
- 右臂队列取值错误：`puppetr = self.puppetl_queue.get()`  
  见 `my_devs/cobot_magic/collect_data/docs/collect_data.py:318`

### 6.4 `argparse` 中大量 `type=bool`，命令行行为不可靠（高优先级）

例如：
- `my_devs/cobot_magic/collect_data/collect_data.py:423`
- `my_devs/cobot_magic/aloha-devel/act/train.py:321`
- `my_devs/cobot_magic/aloha-devel/act/inference.py:732`

风险：
- 传入字符串 `"False"` 常被解析成 `True`（Python bool("False") 为真）。
- 建议统一改为 `action="store_true"/"store_false"`。

### 6.5 训练数据归一化疑似错误：action 用了 qpos 统计量（高优先级）

`act/utils.py` 中：
- `action_data = (action_data - qpos_mean) / qpos_std`
- 见 `my_devs/cobot_magic/aloha-devel/act/utils.py:101`

而同文件已计算 `action_mean/action_std`。这会导致训练/推理量纲混用。

### 6.6 推理后处理同样使用 qpos 统计量（中高优先级）

`inference.py`：
- `post_process = a * qpos_std + qpos_mean`
- 见 `my_devs/cobot_magic/aloha-devel/act/inference.py:296`

若模型训练目标是 action 归一化，此处应与训练一致使用 action 统计量。

### 6.7 `actions_interpolation` 与 state_dim 扩展不一致（中优先级）

`actions_interpolation` 固定把 `arm_steps_length` 左右臂拼成 14 维：
- `my_devs/cobot_magic/aloha-devel/act/inference.py:43`

若开启 `use_robot_base` 使状态变成 16 维，会发生长度不匹配风险。

### 6.8 docs 版与主版 depth 键名不一致（中优先级）

- 主版：`/observations/images_depth/<cam>`
- docs 版：`/observations/depths/<cam>`

`act/utils.py` 只读取 `images_depth`：
- `my_devs/cobot_magic/aloha-devel/act/utils.py:61`

导致 docs 版采集数据不能直接用于当前训练脚本。

### 6.9 docs 版压缩仅记录 RGB 的 `compress_len`（中优先级）

depth 压缩长度没有单独保存：
- 写入 `compress_len` 只在 `my_devs/cobot_magic/collect_data/docs/collect_data.py:181`~`183`

这会让 depth 压缩流反解码难以可靠还原。

### 6.10 robomimic ACT 接口与当前 act/detr 模型签名可能不一致（中优先级）

- `robomimic/algo/act.py` 期望 `self.nets["policy"]` 返回三元组且参数顺序为 `(qpos, images, env_state, ...)`：
  - `my_devs/cobot_magic/aloha-devel/robomimic/algo/act.py:137`
- 但 `detr_vae.py` 的 `forward` 是 `(image, depth_image, robot_state, ...)`，返回 `a_hat, [mu, logvar]`：
  - `my_devs/cobot_magic/aloha-devel/act/detr/models/detr_vae.py:98`
  - `my_devs/cobot_magic/aloha-devel/act/detr/models/detr_vae.py:168`

这说明 robomimic ACT 路径很可能不是当前主工作路径，或存在代码漂移。

### 6.11 相机 launch 文件存在语法残留字符（中优先级）

- `my_devs/cobot_magic/camera_ws/src/ros_astra_camera/launch/multi_dabai_dcw.launch:10`
- 行内容末尾有 `/>7`，疑似误编辑。

### 6.12 文档与文件清单漂移（低优先级）

`piper_sdk_demo/README.MD` 提到的若干脚本不存在（如 `piper_enable.py`、`piper_reset.py`）：
- `my_devs/cobot_magic/collect_data/piper_sdk_demo/README.MD:32`

---

## 7. 与当前 LeRobot v3.0 的差距分析

当前 Cobot Magic 数据是“每 episode 一个 HDF5”，而 LeRobot v3.0 是：
- `meta/` + `data/chunk-xxx/file-yyy.parquet` + `videos/.../file-yyy.mp4`
- 以 `codebase_version=v3.0` 元数据为中心
- 官方 API 是 `LeRobotDataset` / `LeRobotDataset.create`

主要差距：
- 存储形态不同（HDF5 vs parquet+mp4）
- 元数据体系不同（episode/task/stats）
- feature naming 与 dtype 规范不同
- 视频对齐机制不同（LeRobot 用 timestamp + from_timestamp）

---

## 8. 并入 flexible_lerobot 的建议落地路径

建议按四阶段推进：

### 阶段 A：先冻结接口，不直接迁移第三方大包

- 保留 `camera_ws` 和 `Piper_ros` 在独立 workspace。
- 在 `flexible_lerobot` 内只新增“ROS 采集/推理适配层”。
- 避免把 `realsense-ros`、`ros_astra_camera`、`robomimic` 大量源码直接并入主包。

### 阶段 B：先做数据转换器（最高优先级）

新建一个 converter：`cobot_magic_hdf5 -> lerobot v3.0`

推荐字段映射：
- `/observations/qpos` -> `observation.state`（14 维或 + base 16 维）
- `/observations/qvel` -> `observation.qvel`
- `/observations/effort` -> `observation.effort`
- `/action` -> `action`
- `/base_action` -> `action.base` 或 `observation.base_vel`（需统一语义）
- RGB 图像 -> `observation.images.<camera_key>`
- 深度图 -> `observation.images_depth.<camera_key>`（需统一命名与 dtype）

### 阶段 C：统一训练入口到 lerobot policy 栈

- 先复用现有 ACT 配置思想，不直接复用 `aloha-devel/act` 全代码。
- 将必要逻辑提炼为：
  - dataset adapter
  - model config
  - ros deploy adapter

### 阶段 D：上线前清理上述高优先级 bug

最低必须先修：
- `collect_data.py` master topic 默认值
- depth padding 空指针
- `type=bool` 参数
- `act/utils.py` action 归一化统计量
- docs 版队列取值错误

---

## 9. 建议的“并入后目标形态”

建议最终在本仓库形成三层：

1. `lerobot_ros_bridge/`  
   只处理 ROS 订阅、时间同步、发布控制。
2. `lerobot_dataset_adapters/cobot_magic.py`  
   只做 Cobot Magic HDF5 与 LeRobot v3 feature 的映射。
3. `training configs`  
   完全走本仓库已有训练/评估链路，减少双栈维护。

---

## 10. 这份阅读结论可直接用于下一步的任务

后续如果你同意，我建议先做两件事（最短路径）：

1. 先写 `collect_data` 最小修复补丁（只改高优先级 bug，不动架构）。
2. 直接实现一个 `HDF5 -> LeRobot v3.0` 转换脚本，并用 1~2 个 episode 验证可被 `LeRobotDataset` 正常读取。

这两步完成后，再决定是否要把 `aloha-devel/act` 训练能力迁入主仓，还是仅保留其模型思路。

