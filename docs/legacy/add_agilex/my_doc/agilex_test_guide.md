# Agilex Piper 集成测试指南

> **兼容性说明 (lerobot 0.4.3 baseline)**  
> 本文档依赖旧仓库中的 AgileX/ROS Camera 实现（例如 `lerobot.robots.agilex`、`lerobot.cameras.ros_camera`）。  
> 当前仓库的 0.4.3 baseline 默认代码树不包含这些模块，因此本文不能直接照着跑；建议仅作为历史记录/移植参考（路线 B）。  

本文档指导你测试 Agilex Piper 双臂机器人与 LeRobot 的集成代码。

---

## 1. 环境准备

### 1.1 安装依赖

```bash
# 进入项目目录
cd /home/agilex/cqy/lerobot_dev/lerobot_4_2/lerobot_my_fork

# 安装测试依赖
pip install pytest pyserial

# 安装项目（开发模式）
pip install -e .
```

### 1.2 验证模块导入

```bash
# 测试基础导入
python -c "
from lerobot.robots.agilex import AgileXRobot, AgileXConfig, JOINT_NAMES
from lerobot.teleoperators.agilex import AgileXTeleoperator, AgileXTeleoperatorConfig
print('✅ 所有模块导入成功')
print(f'关节名称: {JOINT_NAMES}')
"
```

---

## 2. Mock 模式测试（无需硬件）

### 2.1 测试 Robot 类

```bash
python -c "
from lerobot.robots.agilex import AgileXRobot, AgileXConfig

# 创建 Mock 模式机器人（无相机）
config = AgileXConfig(mock=True, cameras={})
robot = AgileXRobot(config)

print('=== Robot Mock 测试 ===')
print(f'is_connected (before): {robot.is_connected}')

robot.connect()
print(f'is_connected (after): {robot.is_connected}')

# 读取观测
obs = robot.get_observation()
print(f'observation keys: {list(obs.keys())[:5]}...')
print(f'left_shoulder_pan.pos: {obs[\"left_shoulder_pan.pos\"]}')

# 发送动作
action = {k: 0.1 for k in robot.action_features}
result = robot.send_action(action)
print(f'action sent: {list(result.keys())[:3]}...')

robot.disconnect()
print(f'is_connected (after disconnect): {robot.is_connected}')
print('✅ Robot Mock 测试通过')
"
```

### 2.2 测试 Teleoperator 类

```bash
python -c "
from lerobot.teleoperators.agilex import AgileXTeleoperator, AgileXTeleoperatorConfig

config = AgileXTeleoperatorConfig(mock=True)
teleop = AgileXTeleoperator(config)

print('=== Teleoperator Mock 测试 ===')
teleop.connect()
print(f'is_connected: {teleop.is_connected}')

action = teleop.get_action()
print(f'action keys: {list(action.keys())[:5]}...')
print(f'left_shoulder_pan.pos: {action[\"left_shoulder_pan.pos\"]}')

teleop.disconnect()
print('✅ Teleoperator Mock 测试通过')
"
```

### 2.3 运行单元测试

```bash
# 运行 Agilex 相关测试
python -m pytest tests/robots/test_agilex.py tests/teleoperators/test_agilex_teleoperator.py -v
```

---

## 3. 真实硬件测试

### 3.1 前置条件

1. **启动 ROS Master**:
   ```bash
   roscore
   ```

2. **启动 Agilex 驱动**（在另一个终端）:
   ```bash
   # 根据你的 Agilex 驱动启动方式
   roslaunch agilex_piper piper_bringup.launch
   ```

3. **验证 ROS Topic**:
   ```bash
   # 查看可用 topic
   rostopic list | grep -E "(puppet|master)"
   
   # 应该看到类似:
   # /puppet/joint_left
   # /puppet/joint_right
   # /master/joint_left
   # /master/joint_right
   ```

### 3.2 测试真实机器人连接

```bash
python -c "
from lerobot.robots.agilex import AgileXRobot, AgileXConfig

# 真实模式（注意 mock=False）
config = AgileXConfig(
    mock=False,
    puppet_left_topic='/puppet/joint_left',
    puppet_right_topic='/puppet/joint_right',
    cameras={},
)
robot = AgileXRobot(config)

print('正在连接真实机器人...')
robot.connect()

obs = robot.get_observation()
print('当前关节位置:')
for key, val in obs.items():
    print(f'  {key}: {val:.4f}')

robot.disconnect()
print('✅ 真实机器人测试通过')
"
```

### 3.3 测试遥操作

```bash
python -c "
from lerobot.robots.agilex import AgileXRobot, AgileXConfig
from lerobot.teleoperators.agilex import AgileXTeleoperator, AgileXTeleoperatorConfig
import time

# 配置
robot_cfg = AgileXConfig(mock=False, cameras={})
teleop_cfg = AgileXTeleoperatorConfig(mock=False)

robot = AgileXRobot(robot_cfg)
teleop = AgileXTeleoperator(teleop_cfg)

robot.connect()
teleop.connect()

print('开始遥操作测试（按 Ctrl+C 退出）...')
try:
    for i in range(100):
        action = teleop.get_action()
        robot.send_action(action)
        print(f'Step {i}: left_shoulder_pan={action[\"left_shoulder_pan.pos\"]:.3f}', end='\r')
        time.sleep(0.02)  # 50Hz
except KeyboardInterrupt:
    print('\n遥操作测试结束')

robot.disconnect()
teleop.disconnect()
"
```

---

## 4. 数据采集测试（完整流程）

### 4.1 使用 lerobot-record 采集数据

```bash
# Mock 模式测试数据采集流程
export DISPLAY=:0

  lerobot-record \
      --robot.type=agilex \
      --robot.mock=false \
      --robot.cameras='{}' \
      --teleop.type=agilex_teleop \
      --teleop.mock=false \
      --dataset.repo_id=your_username/agilex_dataset1 \
      --dataset.single_task="Your task description" \
      --dataset.num_episodes=2 \
      --dataset.fps=30 \
      --dataset.push_to_hub=false \
      --dataset.episode_time_s=5 \
      --dataset.reset_time_s=5 
```

### 4.2 带相机的数据采集（需要 ROS）

```bash
# 真实硬件 + 双目 ROS 相机
# 如果有图形界面并希望看到相机画面，可以先：export DISPLAY=:0
rm -rf /home/agilex/.cache/huggingface/lerobot/your_username/agilex_dataset1


export DISPLAY=:0

lerobot-record \
    --robot.type=agilex \
    --robot.mock=false \
    --robot.cameras='{
      camera_left:  {"type": "ros_camera", "topic_name": "/camera_l/color/image_raw",     "width": 640, "height":
  480, "fps": 30},
      camera_right: {"type": "ros_camera", "topic_name": "/camera_r/color/image_raw",     "width": 640, "height":
  480, "fps": 30},
      camera_front: {"type": "ros_camera", "topic_name": "/camera_f/color/image_raw", "width": 640, "height":
  480, "fps": 30}
    }' \
    --teleop.type=agilex_teleop \
    --teleop.mock=false \
    --dataset.repo_id=your_username/agilex_dataset1 \
    --dataset.single_task="Your task description" \
    --dataset.num_episodes=5 \
    --dataset.fps=30 \
    --dataset.push_to_hub=false \
    --dataset.episode_time_s=15 \
    --dataset.reset_time_s=15
```

> **提示**：如果将来修好前视相机 `/camera_f/color/image_raw`，只需在 `--robot.cameras` 里多加一行：
> ```
> camera_front: {"type": "ros_camera", "topic_name": "/camera_f/color/image_raw", "width": 640, "height": 480, "fps": 30}
> ```

### 4.3 数据可视化

使用 `lerobot-dataset-viz` 可视化已录制的数据集。该工具基于 [Rerun](https://rerun.io/) 实现。

#### 本地可视化（需要图形界面）

```bash
# 确保有图形环境
export DISPLAY=:0

# 可视化第 0 个 episode
lerobot-dataset-viz \
    --repo-id your_username/agilex_dataset1 \
    --episode-index 0

# 可视化root路径数据
lerobot-dataset-viz \
    --repo-id so101_test_data2 \
    --root /home/agilex/cqy/lerobot_dev/lerobot_4_2/lerobot_my_fork/so101_test_data2 \
    --episode-index 0
```


#### 远程/SSH 环境下可视化

如果在 SSH 环境下，可以先保存为 `.rrd` 文件，然后在本地机器上查看：

```bash
# 在远程机器上保存 .rrd 文件
lerobot-dataset-viz \
    --repo-id your_username/agilex_dataset1 \
    --episode-index 0 \
    --save 1 \
    --output-dir ./viz_output

# 将文件传到本地
# scp agilex@<robot_ip>:~/cqy/.../viz_output/*.rrd .

# 在本地机器上用 rerun 打开
# rerun your_username_agilex_dataset_episode_0.rrd
```

#### 通过 WebSocket 远程流式查看

```bash
# 在远程机器上启动服务（需要端口转发）
# 本地执行: ssh -L 9087:localhost:9087 agilex@<robot_ip>

lerobot-dataset-viz \
    --repo-id your_username/agilex_dataset1 \
    --episode-index 0 \
    --mode distant \
    --ws-port 9087
```

### 4.4 动作回放（Replay）

使用 `lerobot-replay` 在真实机器人上回放已录制的动作序列。

> ⚠️ **安全警告**：回放会让机器人实际运动，请确保：
> - 机器人周围无障碍物
> - 随时准备按下急停按钮
> - 先用短的 episode 测试

```bash
# 回放第 0 个 episode 的动作
lerobot-replay \
    --robot.type=agilex \
    --robot.mock=false \
    --dataset.repo_id=your_username/agilex_dataset1 \
    --dataset.episode=0 \
    --dataset.fps=30
```

#### Mock 模式测试回放逻辑

```bash
# 先用 mock 模式验证流程（不会真正控制机器人）
lerobot-replay \
    --robot.type=agilex \
    --robot.mock=false \
    --dataset.repo_id=your_username/agilex_dataset1 \
    --dataset.episode=0 \
    --dataset.fps=30
```

---

## 5. 问题排查

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| `ModuleNotFoundError: rospy` | ROS 未安装或未 source | `source /opt/ros/noetic/setup.bash` |
| `TimeoutError: Timeout waiting for puppet joint states` | ROS topic 无数据 | 检查 `rostopic echo /puppet/joint_left` |
| `DeviceAlreadyConnectedError` | 重复调用 connect() | 先调用 disconnect() |
| 关节数据全为 0 | Mock 模式或驱动问题 | 确认 mock=False 且驱动正常 |

---

## 6. 已创建的文件清单

```
src/lerobot/
├── robots/agilex/
│   ├── __init__.py
│   ├── config_agilex.py          # 配置类
│   ├── agilex_ros_bridge.py      # ROS 通信桥
│   └── agilex.py                 # Robot 实现
├── teleoperators/agilex/
│   ├── __init__.py
│   ├── config_agilex_teleop.py   # Teleoperator 配置
│   └── agilex_teleoperator.py    # Teleoperator 实现
└── cameras/ros_camera/
    ├── __init__.py
    ├── configuration_ros_camera.py
    └── ros_camera.py             # ROS Image 适配器

tests/
├── robots/test_agilex.py
└── teleoperators/test_agilex_teleoperator.py
```

---

**测试完成后请告诉我遇到的任何问题！**
