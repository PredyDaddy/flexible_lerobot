# jz_robot_pin 修复报告

日期：2026-07-10

## 本次完成

### Git 交付

- 保留仓库通用 `lib/` 忽略规则。
- 为 `my_devs/jz_robot_pin/lib/common.sh` 增加精确白名单。
- 已验证 `common.sh` 不再被 Git 忽略。

### Robot 安全门

- 第一帧关节差值和连续 action 步长默认统一为 `0.02 rad`。
- armed 模式禁止将上述任一限制设为 `0`。
- armed 模式禁止关闭 `JZ_ROBOT_PIN_ARMED` 环境安全门。
- 拒绝 bool、负数、`NaN`、`inf` 等无效安全配置。
- 第一帧没有 state、state stale、sender 不符或 robot 名不符时拒绝发送。
- 每次 armed 发送前重新检查 state 新鲜度和来源。
- state 新包的 seq 倒退或重复时拒绝发送；超过 reset timeout 后允许发送端重启并重置 seq。
- 安全 baseline 缺少任一配置关节时 fail-closed。
- `_last_sent_action` 仅在底层发送成功后更新。

### Target action

- 实时遥操作继续使用 observation-aware `hold_current`。
- target-action 包增加本机时间戳 age、未来偏移、seq 顺序和 seq reset timeout 检查。
- pin joystick 持续发布当前保持目标，默认运行 24 小时，避免 recorder 启动时收不到新包。
- Meshcat 默认以 30 Hz 显示整机模型，target action 以 90 Hz 发布。
- Meshcat 缓存未变化的 transform、骨架和 visibility，减少重复浏览器消息；运行日志报告实测
  `meshcat_hz`、平均/最大显示耗时和 loop overrun。
- 严格录制入口默认等待第一帧 target action 5 秒。
- 录制过程中 target action stale 默认抛错终止，避免保存 hold-current 污染帧。

### 录制 action 语义

- `lerobot_record` 改为保存 `Robot.send_action()` 返回的实际发送 action。
- Rerun 显示和 dataset 使用同一份实际发送 action。
- 对 JZRobotPin，这会正确反映 float 转换和夹爪限幅后的 18 维命令。
- 修复后不得 resume 到旧的 pin 验证数据集；三条检查包装器强制使用新目录和 `RESUME=false`。
- 每条 episode 的第一帧实际写入后，终端记录并语音播报对应的 Start recording 事件。
- reset 阶段继续运行遥操控制循环，但不保存 dataset 帧，并在终端明确记录该状态。

### Conda 约束

- `jz_robot_pin` 公共 Python 启动逻辑不再接受 `PYTHON` 环境变量绕过 conda。
- joystick 包装器会移除继承的 `PYTHON` 覆盖，并使用配置的 `light_tp` conda 环境。

## 三条连续录制与检查

新增：

```text
my_devs/jz_robot_pin/data_check/record_and_check_3.sh
my_devs/jz_robot_pin/data_check/check_3_episodes.py
my_devs/jz_robot_pin/data_check/README.md
```

`record_and_check_3.sh` 在同一个 `lerobot-record` 进程中连续录制 3 个 episode：

- `NUM_EPISODES=3`
- `RESUME=false`
- 默认每条 10 秒、30 FPS、两条之间 reset 5 秒
- 18 维 action/state
- 三路 RTSP 视频
- 三条现场录制包装器的第一帧和连续 action 阈值均为 `10 rad`，对正常关节范围等效于放开
- 核心 robot、普通录制和回放默认第一帧与连续 action 上限仍为 `0.02 rad`
- 包装器把实际第一帧阈值传给自动检查器，避免录制与验收标准不一致
- target action 首包等待 5 秒，stale 时失败
- 数据目录默认带时间戳，已存在时拒绝复用
- 录制成功后自动运行检查器并生成 `data_check_report.json`

检查器硬检查：

- 恰好 3 个 episode，编号为 0、1、2。
- info、stats、tasks、episode metadata、data parquet 完整。
- metadata 帧数、episode length 和实际 parquet 行数一致。
- 全局 index、每条 frame_index 和 timestamp 连续。
- action/state metadata 与实际数组均为 float32 `[18]`。
- 字段顺序为 14 个关节加 4 个夹爪字段，action/state 完全一致。
- 三路视频 metadata、分辨率和对应视频文件存在。
- 所有 action/state 数值有限，夹爪字段位于 `[0, 100]`。
- 每条第一帧 14 个关节最大 action/state 差仍会记录，包装器默认阈值为 `10 rad`。
- 每条相邻 action 最大关节步长仍会记录，包装器默认阈值为 `10 rad`。
- 在未来 1～6 帧内搜索最佳机械跟随延迟。
- 默认最佳 lag MAE 不超过 `0.01 rad`，P95 不超过 `0.03 rad`。
- 不要求不合理的精确 `action_t == observation.state_(t+1)`。

## 验证结果

所有代码测试均使用 `lerobot_flex`：

```text
135 passed
```

覆盖范围：

- JZRobotPin robot、安全配置、armed gate、state freshness/seq。
- pin target-action hold/raise、stamp/seq 校验。
- JZRobotUDP 相关回归。
- 实际发送 action 入库和 Rerun 显示。
- 三 episode 18 维 synthetic dataset、三路视频索引、JSON 报告和失败检测。
- 控制脚本相关测试。

其他验证：

- 所有 `jz_robot_pin` shell 脚本通过 `bash -n`。
- `git diff --check` 通过。
- 连续录制包装器已验证会拒绝复用已有数据目录。
- `common.sh` 已验证不再被 Git 忽略。

未完成的运行时验证：

- 未启动机器人、边缘端、遥操作或真实录制。
- 尚无新的真实 jz_robot_pin 三 episode 数据，因此真实数据报告需由操作者后续运行包装器生成。
- ROS2 专用旧测试在 `lerobot_flex` 中缺少 `rclpy`，无法收集；与本次 pin UDP 修改无关。
- `lerobot_flex` 当前未安装 Ruff，因此未运行 Ruff；已用测试、行宽检查和 `git diff --check` 代替基础静态验证。

## 使用

先由操作者启动边缘端服务和 joystick publisher，然后执行：

```bash
cd /home/luzhuang/cqy/aaa/flexible_lerobot

JZ_ROBOT_PIN_ARMED=1 \
I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT=1 \
bash my_devs/jz_robot_pin/data_check/record_and_check_3.sh
```

包装器不会自动启动或 armed 机器人/边缘端服务。

## 明确保留或延期

- 按使用约束保留 `stop_pin_teleop.sh` 的宽范围 pin 进程清理。
- `my_devs/my_var_tp` 仍为当前机器上的外部本地依赖，后续再纳入 Git 或迁移。
- 暂不进行 `jz_robot_udp`/`jz_robot_pin` 公共模块大重构，先完成真实三条数据验收。
