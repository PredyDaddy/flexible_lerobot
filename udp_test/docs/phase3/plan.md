# Phase 3 UDP Command Executor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 按任务逐项实现。本计划只授权实现代码和测试；任何真实机器人 publish/armed 实机命令必须等待用户明确允许并由用户在现场手动执行。

**Goal:** 在保持默认 `dry_run` 的前提下，把 x86 `JZRobotUDP.send_action()` 发送的 UDP command packet 安全映射到 Orin 侧双臂 JointState 与左右夹爪 Float64MultiArray ROS command topic。

**Architecture:** x86 侧只负责校验 action、生成 command packet v1 并发送 UDP packet，本身不控制机器人。Orin 侧新增 Phase 3 executor，所有 packet 先经过 sender/mode/seq/stamp/action/limit/rate/timeout safety gate；只有 executor `armed`、packet `armed`、CLI `--execution armed` 和环境变量 `JZ_UDP_EXECUTOR_ARMED=1` 同时满足且所有 gate 通过时，才 publish ROS command topic。

**Tech Stack:** Python 3.10+、pytest、Ruff、UDP socket、ROS2 `rclpy`、`sensor_msgs/msg/JointState`、`std_msgs/msg/Float64MultiArray`、本仓库约定 conda 环境（现有脚本默认 `conda run --no-capture-output -n lerobot python`）。

---

## 全局安全边界

- [ ] 默认执行模式必须是 `dry_run`；没有显式配置时，x86 sender 和 Orin executor 都不能真实 publish。
- [ ] 真实 publish 必须双确认：CLI `--execution armed` 与环境变量 `JZ_UDP_EXECUTOR_ARMED=1` 同时满足；缺任一项时必须拒绝启动 armed 或降级为 dry_run，且日志必须清楚说明未 armed。
- [ ] packet 自身也必须是 `mode == "armed"` 才允许真实 publish；executor armed 但 packet dry_run 时不能 publish。
- [ ] 用户急停只是物理兜底，不能替代软件安全门；实现不得把“人在急停旁”当成跳过软件 gate 的理由。
- [ ] Phase 3 第一版只允许 publish：
  - [ ] `/robot1/telecon/arm_left/joint_commands_input`，`sensor_msgs/msg/JointState`
  - [ ] `/robot1/telecon/arm_right/joint_commands_input`，`sensor_msgs/msg/JointState`
  - [ ] `/robot1/left_gripper/gripper_commands`，`std_msgs/msg/Float64MultiArray`
  - [ ] `/robot1/right_gripper/gripper_commands`，`std_msgs/msg/Float64MultiArray`
- [ ] Phase 3 第一版明确不做：
  - [ ] 不做底盘 `cmd_vel`
  - [ ] 不做 body/head/waist
  - [ ] 不做 policy 高速闭环控制
  - [ ] 不修改 VR teleop 让它承担 executor 职责
  - [ ] 不把 Phase 2 dry-run receiver 无门禁改成默认执行器
  - [ ] 不在 timeout/异常/停止时发送零关节位置或未知 home position
- [ ] `JointState` 第一版只填 `name` 和 `position`；不要填 `velocity` 或 `effort`。
- [ ] 夹爪 `Float64MultiArray.data` 顺序必须是 `[width, force]`，与 `src/lerobot/robots/jz_robot/jz_robot.py` 保持一致。
- [ ] 所有测试命令必须在仓库约定 conda 环境运行；本计划中的本机命令默认使用 `conda run --no-capture-output -n lerobot ...`。

## 命令执行分类

### 可由实现子代理在本机运行

- [ ] `conda run --no-capture-output -n lerobot python -m pytest ...`
- [ ] `conda run --no-capture-output -n lerobot python -m ruff check ...`
- [ ] `conda run --no-capture-output -n lerobot python -m ruff format --check ...`
- [ ] 不依赖 ROS 实机、不 publish ROS topic 的 unit/static/mock/dry-run local tests。

### 必须等用户在 Orin/x86 实机手动运行

- [ ] 任何 `udp_test/server_bash/orin_arm/*.sh` 启停脚本。
- [ ] 任何绑定 Orin IP `192.168.1.81`、访问 x86 IP `192.168.1.106` 的实机 UDP 链路命令。
- [ ] 任何 `ros2 topic echo/info/hz`、executor armed、x86 armed packet、`lerobot-record` active episode。
- [ ] 任何可能导致 ROS command topic publish 或机器人运动的命令。

---

## 文件结构与职责

- [ ] Modify: `src/lerobot/robots/jz_robot_udp/protocol.py`
  - 允许 command packet mode 从 Phase 2 的仅 `dry_run` 扩展为 `dry_run | armed`。
  - 继续严格校验 JSON schema、数值类型、bool/NaN/inf 拒绝。

- [ ] Modify: `src/lerobot/robots/jz_robot_udp/config_jz_robot_udp.py`
  - 允许 x86 `send_action_execution` 显式配置为 `dry_run | armed`，默认仍为 `dry_run`。
  - armed 只表示 x86 packet mode 可为 `armed`；x86 不创建 ROS publisher、不控制机器人。

- [ ] Modify: `src/lerobot/robots/jz_robot_udp/jz_robot_udp.py`
  - `send_action()` 根据配置设置 packet mode。
  - 默认仍 dry-run；armed 必须显式配置，日志必须显示 target、mode、seq。
  - 不引入 ROS import，不 publish。

- [ ] Create: `udp_test/test_scripts/arm_side/orin_phase3_command_executor.py`
  - 新建 Orin Phase 3 executor，不能把 `orin_udp_command_receiver.py` 变成默认执行器。
  - 默认 `dry_run`，armed 双确认后才创建/使用 publish 路径。
  - 提供 config、packet decode、safety gate、ROS publisher adapter、summary、shutdown 行为。

- [ ] Create: `udp_test/test_scripts/arm_side/orin_phase3_executor_config.yaml`
  - 默认 dry-run 配置示例。
  - 包含 topic、joint names、network、timeout、rate limit、joint/gripper limits 字段。
  - 允许 TODO limits 出现在文档或 dry-run 配置中；armed 模式遇到 TODO/缺失/非有限/过宽绕过值必须拒绝启动。

- [ ] Modify: `udp_test/server_bash/orin_arm/start_command_receiver.sh`
  - 保持 Phase 2 receiver dry-run 语义，不把它改成 armed executor。
  - 如需要，可标注这是 Phase 2 dry-run receiver。

- [ ] Create: `udp_test/server_bash/orin_arm/start_phase3_executor.sh`
  - 启动 Phase 3 executor，默认 dry-run。
  - armed 启动必须要求 `EXECUTION=armed` 和 `JZ_UDP_EXECUTOR_ARMED=1` 双确认，并把确认透传为 CLI `--execution armed`。

- [ ] Create: `udp_test/server_bash/orin_arm/stop_phase3_executor.sh`
  - 停止 Phase 3 executor 进程。

- [ ] Modify: `udp_test/server_bash/orin_arm/status.sh`
  - 同时显示 Phase 2 dry-run receiver、Phase 3 executor、ROS state bridge 进程。

- [ ] Modify/Create tests: `tests/robots/test_jz_robot_udp.py`
  - 扩展 x86 protocol/config/send_action 测试。

- [ ] Create: `tests/robots/test_jz_robot_udp_phase3_executor.py`
  - Orin executor unit tests、mock publish tests、safety gate tests、static safety tests。

- [ ] Optional Modify: `udp_test/test_scripts/x86_side/x86_jz_robot_udp_send_action_check.py`
  - 支持 `--execution dry_run|armed`，默认 `dry_run`。
  - armed 时只发送 packet mode `armed`；仍不控制机器人，真实执行只发生在 Orin executor gate 通过后。

---

## Task 0: 读取技术文档与 VR probe 证据，确认范围

**Files:**
- Read: `udp_test/docs/phase3/技术文档.md`
- Read: `udp_test/docs/phase2/第二阶段工作报告.md`
- Read: `udp_test/vr_test/logs/vr_command_probe_20260628_114441.log`
- Read: `src/lerobot/robots/jz_robot/jz_robot.py`
- Modify: none
- Test: none

- [ ] **Step 0.1: 复核 Phase 3 技术文档**
  - 确认目标链路是 `UDP command packet -> Orin executor -> safety gate -> ROS command publishers`。
  - 确认默认 `dry_run`、armed 双确认、急停不能替代软件 safety gate。

- [ ] **Step 0.2: 复核 Phase 2 报告**
  - 确认 Phase 2 receiver 只做 dry-run decode/log/stat，不 publish ROS command topic。
  - 确认 x86 -> Orin command dry-run 已验证 `10s * 5fps = 50 command`。

- [ ] **Step 0.3: 复核 VR probe 证据**
  - 确认 arm command topic：
    - `/robot1/telecon/arm_left/joint_commands_input`
    - `/robot1/telecon/arm_right/joint_commands_input`
  - 确认 gripper command topic：
    - `/robot1/left_gripper/gripper_commands`
    - `/robot1/right_gripper/gripper_commands`
  - 确认本次 Phase 3 第一版不使用任何 `cmd_vel` 候选 topic。

- [ ] **Step 0.4: 复核现有 ROS publisher mapping**
  - 在 `src/lerobot/robots/jz_robot/jz_robot.py` 中确认：
    - left/right `JointState.name = config.*_joint_names`
    - left/right `JointState.position = goal positions`
    - gripper `Float64MultiArray.data = [width, force]`
  - 不要填 `JointState.velocity`，除非后续另有文档。

**本机测试命令:** 无。

**实机命令:** 无；Task 0 只读文件，不触发机器人。

**验收标准:**
- [ ] 实现子代理能在任务记录中列出 Phase 3 第一版只做双臂和夹爪。
- [ ] 实现子代理能明确说出 Phase 2 receiver 不能被改成默认 armed executor。
- [ ] 实现子代理能明确说出真实 publish 必须 `--execution armed` 和 `JZ_UDP_EXECUTOR_ARMED=1` 双确认。

**安全注意事项:**
- [ ] Task 0 不运行任何 ROS、UDP 实机或机器人命令。

---

## Task 1: 扩展 command protocol mode，支持 `armed` 但保持严格 validation

**Files:**
- Modify: `src/lerobot/robots/jz_robot_udp/protocol.py`
- Test: `tests/robots/test_jz_robot_udp.py`

- [ ] **Step 1.1: 写失败测试，证明协议接受 `dry_run | armed`**
  - 在 `tests/robots/test_jz_robot_udp.py` 增加/修改测试：
    - `test_command_packet_accepts_dry_run_and_armed_modes`
    - `test_command_packet_rejects_unknown_modes`
    - `test_command_packet_still_rejects_bool_nan_inf`
  - 期望当前代码对 `armed` 失败，因为 Phase 2 只接受 `dry_run`。

本机运行：

```bash
conda run --no-capture-output -n lerobot python -m pytest tests/robots/test_jz_robot_udp.py -k "command_packet and mode" -vv
```

Expected:

```text
FAIL for armed mode before implementation
PASS for existing dry_run validation
```

- [ ] **Step 1.2: 实现最小协议扩展**
  - 在 `protocol.py` 中新增常量：
    - `COMMAND_MODE_DRY_RUN = "dry_run"` 保持不变
    - `COMMAND_MODE_ARMED = "armed"`
    - `COMMAND_MODES = ("dry_run", "armed")`
  - `validate_jz_robot_udp_command_packet()` 改为只接受 `COMMAND_MODES`。
  - 不放宽其他 schema：
    - packet keys 必须精确匹配
    - `seq` 和 `stamp_ns` 必须 int 且 bool 拒绝
    - actions 必须包含 `left/right/grippers`
    - grippers 必须只包含 `width/force`
    - 所有数值必须 finite

- [ ] **Step 1.3: 跑协议测试**

本机运行：

```bash
conda run --no-capture-output -n lerobot python -m pytest tests/robots/test_jz_robot_udp.py -k "command_packet" -vv
```

Expected:

```text
all selected tests PASS
```

**实机命令:** 无。

**验收标准:**
- [ ] `mode="dry_run"` 和 `mode="armed"` 都能通过协议 round-trip。
- [ ] `mode="active"`, `mode="execute"`, `mode="publish"`, 空字符串等仍被拒绝。
- [ ] bool、NaN、inf、missing/extra key 仍被拒绝。

**安全注意事项:**
- [ ] 协议接受 `armed` 不等于真实执行；真实 publish 还必须经过 Orin executor mode gate、CLI/env 双确认和全部 safety gates。

---

## Task 2: x86 JZRobotUDP config/send_action 支持 armed packet，但默认 dry_run

**Files:**
- Modify: `src/lerobot/robots/jz_robot_udp/config_jz_robot_udp.py`
- Modify: `src/lerobot/robots/jz_robot_udp/jz_robot_udp.py`
- Optional Modify: `udp_test/test_scripts/x86_side/x86_jz_robot_udp_send_action_check.py`
- Test: `tests/robots/test_jz_robot_udp.py`

- [ ] **Step 2.1: 写失败测试，锁定 x86 安全默认值**
  - 覆盖：
    - `JZRobotUDPConfig().send_action_execution == "dry_run"`
    - `send_action_execution="armed"` 可以构造 config，但必须显式传入
    - unknown execution 仍拒绝
    - `send_action()` 默认 packet mode 是 `dry_run`
    - `send_action_execution="armed"` 时 packet mode 是 `armed`
    - x86 不 import ROS、不创建 publisher、不调用 publish

本机运行：

```bash
conda run --no-capture-output -n lerobot python -m pytest tests/robots/test_jz_robot_udp.py -k "jz_robot_udp_command_config or send_action" -vv
```

Expected:

```text
FAIL for armed config before implementation
```

- [ ] **Step 2.2: 配置层允许显式 armed**
  - `send_action_execution` 允许值改为 `("dry_run", "armed")`。
  - 默认值保持 `"dry_run"`。
  - 错误信息要明确 unknown execution 不被允许。
  - 不新增任何默认 armed 行为。

- [ ] **Step 2.3: `send_action()` 使用配置生成 packet mode**
  - 当前写死 `COMMAND_MODE_DRY_RUN`，改为 `self.config.send_action_execution`。
  - log 文案不要写死 `DRY_RUN`；应包含 `mode=<mode>`、`transport=<transport>`、`target=<ip:port>`。
  - x86 仍只发送 UDP packet 或 local log，不创建 ROS publisher。

- [ ] **Step 2.4: 可选扩展 x86 check 脚本**
  - `udp_test/test_scripts/x86_side/x86_jz_robot_udp_send_action_check.py` 的 `--execution` choices 改为 `dry_run|armed`。
  - 默认仍 `dry_run`。
  - startup banner 必须显示：
    - x86 只发送 command packet
    - x86 本身不控制机器人
    - armed packet 只有在 Orin executor 也 armed 且 gates 通过时才会 publish

- [ ] **Step 2.5: 跑 x86 单测**

本机运行：

```bash
conda run --no-capture-output -n lerobot python -m pytest tests/robots/test_jz_robot_udp.py -vv
```

Expected:

```text
all tests PASS
```

**实机命令:** 无。

**验收标准:**
- [ ] 默认 config 和默认 check 脚本仍为 dry-run。
- [ ] 只有显式 `send_action_execution="armed"` 才生成 packet `mode="armed"`。
- [ ] x86 侧实现不包含 ROS publisher、不 publish、不控制机器人。
- [ ] Phase 2 dry-run 行为保持兼容。

**安全注意事项:**
- [ ] 不要在 x86 加“真实执行”概念；x86 armed 只表示 packet mode。
- [ ] 不要在 x86 脚本默认启用 armed。

---

## Task 3: 新建 Orin Phase 3 executor 脚本，默认 dry_run，armed 后才 publish

**Files:**
- Create: `udp_test/test_scripts/arm_side/orin_phase3_command_executor.py`
- Create: `udp_test/test_scripts/arm_side/orin_phase3_executor_config.yaml`
- Test: `tests/robots/test_jz_robot_udp_phase3_executor.py`

- [ ] **Step 3.1: 写 executor 默认 dry-run 测试**
  - 使用 fake socket packet 或直接调用内部函数，不需要真实 Orin/ROS。
  - 覆盖：
    - 默认 config `execution == "dry_run"`
    - `dry_run` 下不创建或不使用 real ROS publisher
    - `dry_run + packet armed` 仍不 publish，日志/decision 记录 executor 未 armed
    - startup banner 包含 `PHASE3 COMMAND EXECUTOR DRY-RUN` 和 `NOT publishing ROS command topics`

本机运行：

```bash
conda run --no-capture-output -n lerobot python -m pytest tests/robots/test_jz_robot_udp_phase3_executor.py -k "dry_run or startup" -vv
```

Expected:

```text
FAIL because executor does not exist yet
```

- [ ] **Step 3.2: 新建 executor 脚本骨架**
  - argparse 参数至少包含：
    - `--bind-ip`
    - `--port`
    - `--allowed-sender-ip`
    - `--execution`，choices `dry_run|armed`，default `dry_run`
    - `--config`
    - `--count`
    - `--print-every`
    - `--buffer-size`
    - `--socket-timeout-s`
  - armed ack 参数：
    - 可显式增加 `--i-understand-this-publishes-robot-commands`
    - 但真实 publish 的硬要求仍必须包含用户要求的 `--execution armed` 与 `JZ_UDP_EXECUTOR_ARMED=1`
  - signal handler：
    - SIGINT/SIGTERM 设置 shutdown flag
    - 停止接收 loop
    - 不再 publish 新 command
    - 打印 summary

- [ ] **Step 3.3: 配置文件**
  - 默认配置写入：
    - bind/allowed sender/port
    - topic names
    - left/right joint names
    - timeout/rate defaults
    - joint/gripper limits 字段
  - `dry_run` 允许配置含 TODO limits 但必须打印 would-reject 或 limits 未完成提示。
  - `armed` 模式必须拒绝 TODO/缺失/非有限/过宽绕过 limits。

- [ ] **Step 3.4: ROS publisher adapter**
  - 把 ROS 相关代码隔离在 adapter 类/函数中，便于单测 mock。
  - `dry_run` 不调用 publisher `.publish()`。
  - `armed` 且 gates 全通过才 publish：
    - left JointState `name` 和 `position`
    - right JointState `name` 和 `position`
    - left gripper Float64MultiArray `[width, force]`
    - right gripper Float64MultiArray `[width, force]`
  - 不创建任何 `cmd_vel` publisher。

- [ ] **Step 3.5: 跑 executor dry-run 测试**

本机运行：

```bash
conda run --no-capture-output -n lerobot python -m pytest tests/robots/test_jz_robot_udp_phase3_executor.py -k "dry_run or startup or mapping" -vv
```

Expected:

```text
all selected tests PASS
```

**实机命令:** 无。

**验收标准:**
- [ ] 新 executor 默认 dry-run。
- [ ] `dry_run` 下 packet 合法也不 publish。
- [ ] `armed` 下缺少 env 或 CLI ack 时不 publish。
- [ ] ROS message mapping 与 `JZRobot.send_action()` 一致。
- [ ] Phase 2 `orin_udp_command_receiver.py` 仍是 dry-run receiver，不承担 Phase 3 armed executor。

**安全注意事项:**
- [ ] 不要为了方便把 Phase 2 receiver 改成可 publish 的脚本。
- [ ] 不要在 executor import/load 期间自动启动 ROS 或 publish。

---

## Task 4: 实现 safety gate

**Files:**
- Modify: `udp_test/test_scripts/arm_side/orin_phase3_command_executor.py`
- Modify: `udp_test/test_scripts/arm_side/orin_phase3_executor_config.yaml`
- Test: `tests/robots/test_jz_robot_udp_phase3_executor.py`

- [ ] **Step 4.1: 写 sender IP gate 测试**
  - allowed sender `192.168.1.106` 通过。
  - unexpected sender 被丢弃，不进入 publish 路径，`unexpected_sender` counter 增加。

- [ ] **Step 4.2: 写 mode gate 测试**
  - `executor=dry_run, packet=dry_run`：validate/log，不 publish。
  - `executor=dry_run, packet=armed`：validate/log，不 publish，reason `executor_not_armed`。
  - `executor=armed, packet=dry_run`：不 publish，reason `packet_not_armed`。
  - `executor=armed, packet=armed`：只有其他 gate 全通过才 publish。
  - 缺 `JZ_UDP_EXECUTOR_ARMED=1` 或缺 CLI ack 时，armed 不可 publish。

- [ ] **Step 4.3: 写 seq monotonic gate 测试**
  - 第一条 valid seq 接受。
  - 后续 `seq > last_seq` 接受。
  - `seq <= last_seq` 拒绝。
  - `seq > last_seq + 1` 接受但统计 `seq_gap_count`。

- [ ] **Step 4.4: 写 stamp age gate 测试**
  - 默认 `max_command_age_s=0.25`。
  - `age_s > max_command_age_s` 拒绝。
  - `age_s < -max_clock_skew_s` 拒绝。
  - dry-run 也执行同样判断并打印 would reject。
  - 如使用 wall time `time.time_ns()`，测试要注入 fake clock；如果改用 monotonic 相对时间，必须清楚记录字段语义，armed 不能完全关闭 stale check。

- [ ] **Step 4.5: 写 finite numeric gate 测试**
  - NaN、inf、bool、非数值都拒绝。
  - 覆盖 joint、gripper width、gripper force。

- [ ] **Step 4.6: 写 joint name/count gate 测试**
  - left/right 各 7 个 joint。
  - 名字必须完全匹配配置。
  - missing、extra、拼错都拒绝。
  - packet dict 按配置顺序提取 position，不依赖 JSON key 顺序。

- [ ] **Step 4.7: 写 joint absolute range gate 测试**
  - armed 模式缺任一 joint limit 拒绝启动或拒绝进入 armed。
  - 任一 joint 超出 `[min, max]` 拒绝 publish。
  - `None`、`inf`、`nan`、字符串 TODO、极大绕过值拒绝 armed。
  - dry-run 下执行检查并记录 `would_reject`。

- [ ] **Step 4.8: 写 per-step delta gate 测试**
  - `abs(current - last_accepted) <= max_delta` 通过。
  - 大于 max delta 拒绝。
  - 第一条 armed command：
    - 如果实现读取当前 state，则必须与当前 state 比较。
    - 如果读取不到当前 state，armed 模式必须拒绝第一条大幅 command，不能直接接受未知大跳变。
  - 第一版可要求 first command 使用 `--allow-first-command-within-delta-from-state` 或类似明确机制，但不能默认绕过。

- [ ] **Step 4.9: 写 gripper limits 测试**
  - width/force 范围限制。
  - width/force per-step delta 限制。
  - left/right 分别覆盖。
  - armed 模式缺 gripper limit 拒绝启动。

- [ ] **Step 4.10: 写 publish rate limit 测试**
  - 默认 `max_publish_hz=10.0`。
  - 过快 packet 不 burst publish。
  - 被限速 packet drop 或 reject，并统计 `rate_limited_count`。
  - 日志/summary 输出 accepted_hz、published_hz 或等价统计。

- [ ] **Step 4.11: 写 command timeout 和 shutdown 测试**
  - 默认 `command_timeout_s=0.3`。
  - 超时后 `active=false`，不重复 publish last command。
  - timeout 只停止发布新 command，不发送零关节或未知 stop pose。
  - SIGINT/SIGTERM 后停止 loop、关闭 ROS node/publishers、打印 summary。

- [ ] **Step 4.12: 实现 gate 与 counters**
  - 至少统计：
    - received
    - invalid
    - unexpected_sender
    - rejected_by_gate
    - published
    - dry_run
    - rate_limited
    - timeout
    - seq_gap_count
    - last_seq
    - last_published_seq
    - reject reasons

- [ ] **Step 4.13: 跑 safety gate 测试**

本机运行：

```bash
conda run --no-capture-output -n lerobot python -m pytest tests/robots/test_jz_robot_udp_phase3_executor.py -k "gate or limit or timeout or shutdown" -vv
```

Expected:

```text
all selected tests PASS
```

**实机命令:** 无。

**验收标准:**
- [ ] sender IP、mode、seq、stamp、finite numeric、joint name/count、joint absolute range、per-step delta、gripper limits、rate limit、timeout、shutdown 全有测试。
- [ ] 任一 gate fail 都不会调用 ROS publish。
- [ ] armed 模式缺 joint/gripper limits 不能启动或不能进入 publish-capable state。
- [ ] summary 能定位拒绝原因。

**安全注意事项:**
- [ ] 不允许用配置 `None`、`inf`、巨大数值绕过 limits。
- [ ] 不允许在异常或 timeout 时发送零关节位置。

---

## Task 5: server_bash start/stop/status 脚本，默认 dry_run，armed 必须环境变量双确认

**Files:**
- Create: `udp_test/server_bash/orin_arm/start_phase3_executor.sh`
- Create: `udp_test/server_bash/orin_arm/stop_phase3_executor.sh`
- Modify: `udp_test/server_bash/orin_arm/status.sh`
- Optional Modify: `udp_test/server_bash/orin_arm/start.sh`
- Optional Modify: `udp_test/server_bash/orin_arm/stop.sh`
- Test: `tests/robots/test_jz_robot_udp_phase3_executor.py`

- [ ] **Step 5.1: 写 static/AST 测试**
  - 检查 `start_phase3_executor.sh` 默认 `EXECUTION=dry_run`。
  - 检查 armed 分支同时要求：
    - `EXECUTION=armed`
    - `JZ_UDP_EXECUTOR_ARMED=1`
    - CLI 透传 `--execution armed`
  - 检查脚本输出 `PHASE3 COMMAND EXECUTOR DRY-RUN` 或 `PHASE3 COMMAND EXECUTOR ARMED`。
  - 检查 stop/status pattern 指向 `orin_phase3_command_executor.py`。
  - 检查脚本不包含 `cmd_vel`。

本机运行：

```bash
conda run --no-capture-output -n lerobot python -m pytest tests/robots/test_jz_robot_udp_phase3_executor.py -k "script or static" -vv
```

Expected:

```text
FAIL before scripts are created
```

- [ ] **Step 5.2: 实现 `start_phase3_executor.sh`**
  - 默认变量：
    - `EXECUTION="${EXECUTION:-dry_run}"`
    - `ORIN_IP="${ORIN_IP:-192.168.1.81}"`
    - `X86_IP="${X86_IP:-192.168.1.106}"`
    - `COMMAND_PORT="${COMMAND_PORT:-39020}"`
    - `PYTHON_CMD="${PYTHON_CMD:-conda run --no-capture-output -n lerobot python}"`
  - dry-run 启动：
    - 不要求 `JZ_UDP_EXECUTOR_ARMED`
    - 显示“不 publish ROS command topics”
  - armed 启动：
    - 如果 `JZ_UDP_EXECUTOR_ARMED != 1`，脚本必须退出非零，不启动 executor。
    - 命令行必须包含 `--execution armed`。
    - 可追加 `--i-understand-this-publishes-robot-commands`。
    - banner 必须说明将 publish ROS command topic，且急停只是物理兜底。

- [ ] **Step 5.3: 实现 `stop_phase3_executor.sh`**
  - 先按 pid file 停止。
  - 再按 `pgrep -af "udp_test/test_scripts/arm_side/orin_phase3_command_executor.py"` 清理。
  - 先 TERM，短等后仍存在再 KILL。
  - 不影响 Phase 2 receiver 和 ROS state bridge。

- [ ] **Step 5.4: 更新 `status.sh`**
  - 显示：
    - ROS state UDP bridge
    - Phase 2 command dry-run receiver
    - Phase 3 command executor

- [ ] **Step 5.5: 跑脚本 static 测试**

本机运行：

```bash
conda run --no-capture-output -n lerobot python -m pytest tests/robots/test_jz_robot_udp_phase3_executor.py -k "script or static" -vv
```

Expected:

```text
all selected tests PASS
```

**实机命令:** 不由实现子代理运行。用户在 Orin 上手动运行：

Dry-run only:

```bash
cd /home/data/test/workspace/flexible_lerobot
bash udp_test/server_bash/orin_arm/start_phase3_executor.sh
```

Expected log:

```text
PHASE3 COMMAND EXECUTOR DRY-RUN
NOT publishing ROS command topics
robot should not move
```

Stop:

```bash
cd /home/data/test/workspace/flexible_lerobot
bash udp_test/server_bash/orin_arm/stop_phase3_executor.sh
```

Armed startup, only after用户明确允许真实控制:

```bash
cd /home/data/test/workspace/flexible_lerobot
JZ_UDP_EXECUTOR_ARMED=1 EXECUTION=armed bash udp_test/server_bash/orin_arm/start_phase3_executor.sh
```

Expected log:

```text
PHASE3 COMMAND EXECUTOR ARMED
WILL publish ROS command topics
required acknowledgements present
```

**验收标准:**
- [ ] 默认脚本启动 dry-run。
- [ ] `EXECUTION=armed` 但没有 `JZ_UDP_EXECUTOR_ARMED=1` 时拒绝启动。
- [ ] status 能看到 Phase 3 executor。
- [ ] stop 能停止 Phase 3 executor。

**安全注意事项:**
- [ ] 实现子代理不得运行 armed startup。
- [ ] start/stop/status 脚本必须清楚区分 Phase 2 receiver 与 Phase 3 executor。

---

## Task 6: 测试矩阵与 dry-run/local/mocked armed 验收

**Files:**
- Modify: `tests/robots/test_jz_robot_udp.py`
- Create/Modify: `tests/robots/test_jz_robot_udp_phase3_executor.py`
- Modify: `udp_test/test_scripts/arm_side/orin_phase3_command_executor.py`
- Modify: `udp_test/test_scripts/x86_side/x86_jz_robot_udp_send_action_check.py`
- Modify: `udp_test/server_bash/orin_arm/*.sh`

- [ ] **Step 6.1: Unit tests 全量运行**

本机运行：

```bash
conda run --no-capture-output -n lerobot python -m pytest tests/robots/test_jz_robot_udp.py tests/robots/test_jz_robot_udp_phase3_executor.py -vv
```

Expected:

```text
all tests PASS
```

- [ ] **Step 6.2: Static safety tests**
  - 覆盖：
    - 默认 execution_mode 是 dry_run
    - 默认不会 armed
    - 缺 env var 不 armed
    - 缺 CLI acknowledgement 不 armed
    - armed 缺 joint limits 拒绝启动
    - armed 缺 gripper limits 拒绝启动
    - Phase 3 executor publish path 不含 `cmd_vel`
    - Phase 2 receiver 不被改成默认执行器

本机运行：

```bash
conda run --no-capture-output -n lerobot python -m pytest tests/robots/test_jz_robot_udp_phase3_executor.py -k "static or default or armed" -vv
```

Expected:

```text
all selected tests PASS
```

- [ ] **Step 6.3: Dry-run local executor test**
  - 不绑定 Orin IP，不使用 ROS 实机。
  - 使用 loopback 或直接调用 executor function。
  - 验证 dry-run 收到 packet 后只 log，不 publish。

本机运行示例：

```bash
conda run --no-capture-output -n lerobot python -m pytest tests/robots/test_jz_robot_udp_phase3_executor.py -k "dry_run_local" -vv
```

Expected:

```text
received packet accepted in dry_run
publish mock call count == 0
```

- [ ] **Step 6.4: Armed publish mocked tests**
  - 使用 fake ROS publishers，不能连接真实 ROS graph。
  - 环境变量在测试内用 monkeypatch 设置 `JZ_UDP_EXECUTOR_ARMED=1`。
  - CLI/config 显式设置 `execution=armed`。
  - 合法 packet publish mock 计数：
    - left arm 1
    - right arm 1
    - left gripper 1
    - right gripper 1
  - 任一 gate fail 时 publish mock 计数不增加。

本机运行：

```bash
conda run --no-capture-output -n lerobot python -m pytest tests/robots/test_jz_robot_udp_phase3_executor.py -k "mocked or publish" -vv
```

Expected:

```text
all mocked publish tests PASS
no real ROS publish
```

- [ ] **Step 6.5: Ruff check**

本机运行：

```bash
conda run --no-capture-output -n lerobot python -m ruff check src/lerobot/robots/jz_robot_udp tests/robots/test_jz_robot_udp.py tests/robots/test_jz_robot_udp_phase3_executor.py udp_test/test_scripts/arm_side/orin_phase3_command_executor.py udp_test/test_scripts/x86_side/x86_jz_robot_udp_send_action_check.py
```

Expected:

```text
All checks passed!
```

- [ ] **Step 6.6: Format check**

本机运行：

```bash
conda run --no-capture-output -n lerobot python -m ruff format --check src/lerobot/robots/jz_robot_udp tests/robots/test_jz_robot_udp.py tests/robots/test_jz_robot_udp_phase3_executor.py udp_test/test_scripts/arm_side/orin_phase3_command_executor.py udp_test/test_scripts/x86_side/x86_jz_robot_udp_send_action_check.py
```

Expected:

```text
would reformat 0 files
```

**实机命令:** 无。Task 6 只做本机 unit/static/local/mock 验收。

**验收标准:**
- [ ] Unit tests PASS。
- [ ] Static safety tests PASS。
- [ ] Dry-run local tests PASS。
- [ ] Armed publish mocked tests PASS，且没有真实 ROS publish。
- [ ] Ruff check/format PASS。

**安全注意事项:**
- [ ] 不要为了测试方便连接真实 ROS graph。
- [ ] mocked armed 只验证调用路径，不代表可以实机 armed。

---

## Task 7: 实机验收步骤（只由用户现场手动执行）

**Files:**
- Read: `udp_test/docs/phase3/技术文档.md`
- Run manually on Orin/x86 only after user approval:
  - `udp_test/server_bash/orin_arm/start_phase3_executor.sh`
  - `udp_test/server_bash/orin_arm/stop_phase3_executor.sh`
  - `udp_test/test_scripts/x86_side/x86_jz_robot_udp_send_action_check.py`
  - existing `lerobot-record` command selected by user

> 实现子代理不得自行执行本节命令。本节是交给用户和现场操作人员的验收 runbook。

### Task 7A: Dry-run regression

- [ ] **Step 7A.1: Orin 现场启动 Phase 3 executor dry-run**

用户在 Orin 手动运行：

```bash
cd /home/data/test/workspace/flexible_lerobot
bash udp_test/server_bash/orin_arm/stop_phase3_executor.sh
EXECUTION=dry_run bash udp_test/server_bash/orin_arm/start_phase3_executor.sh
```

Expected log:

```text
PHASE3 COMMAND EXECUTOR DRY-RUN
NOT publishing ROS command topics
robot should not move
bind_ip=192.168.1.81
allowed_sender_ip=192.168.1.106
configured topics include only arm/gripper command topics
```

停止命令：

```bash
cd /home/data/test/workspace/flexible_lerobot
bash udp_test/server_bash/orin_arm/stop_phase3_executor.sh
```

失败时停止标准：
- [ ] banner 没有明确 dry-run。
- [ ] 日志出现 `WILL publish`。
- [ ] 日志出现 `cmd_vel`。
- [ ] 机器人有任何运动。

- [ ] **Step 7A.2: x86 发送 50 条 dry-run command**

用户在 x86 手动运行：

```bash
cd /home/data/test/workspace/flexible_lerobot
conda run --no-capture-output -n lerobot python udp_test/test_scripts/x86_side/x86_jz_robot_udp_send_action_check.py \
  --transport udp \
  --execution dry_run \
  --command-target-ip 192.168.1.81 \
  --command-target-port 39020 \
  --count 50 \
  --hz 5 \
  --command-only \
  --skip-cameras
```

Expected x86 log:

```text
execution=dry_run
SUMMARY: PASS commands=50
```

Expected Orin log:

```text
received=50
published=0
dry_run=50
unexpected_sender=0
invalid=0
rejected_by_gate=0
```

失败时停止标准：
- [ ] Orin published 计数不为 0。
- [ ] reject reason 非测试预期。
- [ ] unexpected sender 非 0。
- [ ] robot moves。

### Task 7B: Armed single small command

前置条件：
- [ ] 用户明确允许真实控制。
- [ ] 人在机器人旁，手在急停附近。
- [ ] 软件 joint/gripper limits 已填真实安全范围。
- [ ] `max_publish_hz` 设为 5 到 10。
- [ ] `command_timeout_s` 默认或更保守。
- [ ] 不启用 policy inference。
- [ ] 不启用底盘。
- [ ] 不运行任何 `cmd_vel` publisher。

- [ ] **Step 7B.1: Orin armed 启动 executor**

用户在 Orin 手动运行：

```bash
cd /home/data/test/workspace/flexible_lerobot
bash udp_test/server_bash/orin_arm/stop_phase3_executor.sh
JZ_UDP_EXECUTOR_ARMED=1 EXECUTION=armed bash udp_test/server_bash/orin_arm/start_phase3_executor.sh
```

Expected log:

```text
PHASE3 COMMAND EXECUTOR ARMED
WILL publish ROS command topics
required acknowledgements present
emergency stop is physical fallback, not a replacement for software limits
configured topics:
  /robot1/telecon/arm_left/joint_commands_input
  /robot1/telecon/arm_right/joint_commands_input
  /robot1/left_gripper/gripper_commands
  /robot1/right_gripper/gripper_commands
no cmd_vel publishers
limits loaded
```

停止命令：

```bash
cd /home/data/test/workspace/flexible_lerobot
bash udp_test/server_bash/orin_arm/stop_phase3_executor.sh
```

失败时停止标准：
- [ ] armed banner 不完整。
- [ ] limits 未加载或存在 TODO。
- [ ] 日志出现 `cmd_vel`。
- [ ] 缺 env/ack 时仍进入 armed。

- [ ] **Step 7B.2: x86 发送单条小幅 armed command**

要求：
- [ ] 先读取当前 observation。
- [ ] 构造相对当前 joint position 的极小 delta command。
- [ ] 只发 1 条或极少数 command。
- [ ] 不发送 policy inference 输出。
- [ ] 不使用 `--command-only` 发送 armed 小幅动作；`--command-only` 只适合 Phase 2/Phase 3 dry-run packet 链路验证。

用户在 x86 手动运行前，Orin state bridge 必须已经运行，x86 必须能通过 `get_observation()` 读到当前状态。
具体命令应使用基于 observation 的小 delta action，例如只偏移一个关节 `0.001`：

```bash
cd /home/data/test/workspace/flexible_lerobot
conda run --no-capture-output -n lerobot python udp_test/test_scripts/x86_side/x86_jz_robot_udp_send_action_check.py \
  --transport udp \
  --execution armed \
  --command-target-ip 192.168.1.81 \
  --command-target-port 39020 \
  --count 1 \
  --hz 1 \
  --no-command-only \
  --skip-cameras \
  --action-source observation_delta \
  --delta-key left_left_joint1.pos \
  --delta-value 0.001
```

如果该脚本没有打印 `observation_delta action built`，或者 observation 读取失败，则不能继续 armed
single-small-command 验收。不要用全零 action 代替当前姿态附近的小 delta；零位不一定是安全姿态。

Expected Orin log:

```text
executor=armed
packet_mode=armed
gate_result=accept
published=1
published topics are only left/right arm and left/right gripper
```

Expected physical behavior:
- [ ] 动作幅度极小。
- [ ] 方向符合预期。
- [ ] timeout 后不继续重复 publish。

停止命令：

```bash
cd /home/data/test/workspace/flexible_lerobot
bash udp_test/server_bash/orin_arm/stop_phase3_executor.sh
```

失败时停止标准：
- [ ] 任一 gate reject 但仍 publish。
- [ ] 实际动作幅度超过预期。
- [ ] timeout 后仍持续 publish。
- [ ] 出现任何底盘、body、head、waist 动作。
- [ ] 现场人员认为有风险；立即 stop executor，必要时按急停。

### Task 7C: Record active short episode

只有 Task 7A dry-run regression 与 Task 7B armed single small command 通过后才能执行。

前置条件：
- [ ] 用户再次明确允许短 episode active record。
- [ ] 人在机器人旁，手在急停附近。
- [ ] executor armed，limits/rate/timeout 全部开启。
- [ ] `duration_s` 5 到 10。
- [ ] `fps` 5。
- [ ] `max_publish_hz` 5 到 10。
- [ ] policy inference disabled。
- [ ] base/cmd_vel disabled。

- [ ] **Step 7C.1: Orin 保持 armed executor**

Expected log before record:

```text
PHASE3 COMMAND EXECUTOR ARMED
published_hz limit active
command_timeout active
no cmd_vel publishers
```

- [ ] **Step 7C.2: x86 执行短 active record**

用户在 x86 根据项目当前 `lerobot-record` 参数手动运行，要求：
- [ ] duration 5 到 10 秒。
- [ ] fps 5。
- [ ] 不启用 policy。
- [ ] 不启用底盘。
- [ ] action command 小幅、低频、可人工确认。

Expected:

```text
record completed
command seq continuous or gaps explained
published_hz within configured limit
no stale command
no limit violation
timeout behavior normal
```

停止命令：

```bash
cd /home/data/test/workspace/flexible_lerobot
bash udp_test/server_bash/orin_arm/stop_phase3_executor.sh
```

失败时停止标准：
- [ ] publish rate 超过限制。
- [ ] 出现 stale command。
- [ ] 出现 limit violation。
- [ ] 机器人动作不符合预期。
- [ ] 任何底盘、body、head、waist 动作。
- [ ] 现场人员认为有风险；立即 stop executor，必要时按急停。

**Task 7 验收标准:**
- [ ] Dry-run regression：50 commands，published=0，robot does not move。
- [ ] Armed single small command：只 publish 双臂和夹爪 topic，动作小且方向正确，timeout 后不重复 publish。
- [ ] Record short episode：5 到 10 秒、5 fps、publish rate 受限、无 stale/limit violation。

**Task 7 安全注意事项:**
- [ ] 实机步骤必须由用户或现场人员手动执行。
- [ ] 任何异常先停止 executor；急停只是物理兜底，不替代软件门禁。
- [ ] 不要直接开启真实执行；每一步都必须先 dry-run，再 armed 小幅单条，再短 episode。

---

## 最终完成定义

- [ ] Phase 3 第一版代码只覆盖 left/right arm JointState publishers 和 left/right gripper Float64MultiArray publishers。
- [ ] 默认 dry-run；所有默认脚本和配置不会真实 publish。
- [ ] `armed` 真实 publish 需要 executor mode、packet mode、CLI `--execution armed`、`JZ_UDP_EXECUTOR_ARMED=1` 四项同时满足。
- [ ] 所有 safety gates 已实现并有测试：sender IP、seq monotonic、stamp age、finite numeric、joint names/count、joint absolute range、per-step delta、gripper limits、rate limit、command timeout、shutdown behavior。
- [ ] x86 侧只发送 UDP packet，不 import ROS、不 publish、不控制机器人。
- [ ] Orin Phase 3 executor 是新脚本，Phase 2 dry-run receiver 保持 dry-run。
- [ ] Unit/static/dry-run local/mocked armed tests 全部通过。
- [ ] Ruff check 和 format check 通过。
- [ ] 实机验收 runbook 清楚区分本机可跑命令与用户现场手动命令。
- [ ] 文档和日志明确：用户急停是物理兜底，不能替代软件 safety gate。
