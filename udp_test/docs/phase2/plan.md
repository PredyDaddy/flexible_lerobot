# Phase 2 JZRobotUDP Command Dry-run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the JZRobotUDP action side up to UDP command dry-run so `lerobot-record` can exercise `send_action()` without moving the robot.

**Architecture:** x86 owns LeRobot record/action generation and sends command packets through `JZRobotUDP.send_action()`. Orin owns robot-side ROS/hardware and runs a command receiver that validates and logs command packets but does not create ROS publishers or execute commands in Phase 2.

**Tech Stack:** Python 3, LeRobot Robot interface, JSON over UDP sockets, ROS2/rclpy only for existing state bridge, pytest, bash start/stop/status scripts, conda env `lerobot_flex`.

---

## Required Reading

Before implementing this plan, read:

```text
udp_test/docs/phase2/技术文档.md
udp_test/docs/phase1/background.md
udp_test/docs/phase1/暂定通信结论.md
udp_test/docs/phase1/udp_readonly_architecture.md
udp_test/docs/phase1/udp_robot_interface_tech_stack.md
udp_test/docs/phase1/第一阶段工作报告.md
```

## File Structure

Expected write scope:

```text
src/lerobot/robots/jz_robot_udp/config_jz_robot_udp.py
  Add command target IP/port, send_action transport/execution dry-run config, timeout/rate fields.

src/lerobot/robots/jz_robot_udp/protocol.py
  Add command packet dataclass/helpers, encode/decode, validation.

src/lerobot/robots/jz_robot_udp/udp_client.py
  Add a UDPCommandSender or equivalent helper separate from the state receive path.

src/lerobot/robots/jz_robot_udp/jz_robot_udp.py
  Add action_features and send_action implementation with local dry-run / UDP dry-run modes.

src/lerobot/configs/robot/jz_robot_udp_three_rtsp.yaml
  Add explicit Phase 2 command dry-run defaults.

udp_test/test_scripts/arm_side/orin_udp_command_receiver.py
  New Orin dry-run command receiver. It must not create ROS publishers.

udp_test/test_scripts/x86_side/x86_jz_robot_udp_send_action_check.py
  New x86 check script that calls send_action with generated action payloads.

udp_test/server_bash/orin_arm/start_command_receiver.sh
udp_test/server_bash/orin_arm/stop_command_receiver.sh
udp_test/server_bash/orin_arm/status.sh
  Start/stop/status integration for command receiver.

udp_test/server_bash/x86/start_send_action_check.sh
udp_test/server_bash/x86/stop_send_action_check.sh
udp_test/server_bash/x86/status.sh
  Optional x86 command sender check wrappers if direct script commands become too long.

tests/robots/test_jz_robot_udp.py
  Extend unit tests for command protocol, action_features, send_action modes.

udp_test/docs/phase2/技术文档.md
udp_test/docs/phase2/plan.md
udp_test/server_bash/README.md
  Keep usage and safety docs current.
```

Do not add ROS command publishers in Phase 2.

## Task 1: Add Command Config Fields

**Files:**
- Modify: `src/lerobot/robots/jz_robot_udp/config_jz_robot_udp.py`
- Test: `tests/robots/test_jz_robot_udp.py`

- [ ] **Step 1: Write failing tests for command config defaults**

Add tests asserting that default config is dry-run and not armed:

```python
def test_jz_robot_udp_command_config_defaults_are_safe():
    cfg = JZRobotUDPConfig()
    assert cfg.command_target_ip == "192.168.1.81"
    assert cfg.command_target_port == 39020
    assert cfg.send_action_transport == "local"
    assert cfg.send_action_execution == "dry_run"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run --no-capture-output -n lerobot_flex python -m pytest -q tests/robots/test_jz_robot_udp.py -k command_config
```

Expected: FAIL because fields do not exist.

- [ ] **Step 3: Implement config fields**

Add conservative fields:

```python
command_target_ip: str = "192.168.1.81"
command_target_port: int = 39020
send_action_transport: str = "local"  # local | udp
send_action_execution: str = "dry_run"  # Phase 2 only supports dry_run
command_robot: str = "robot1"
command_timeout_s: float = 0.2
```

Do not add active-control, armed, execute, or publish modes yet.

- [ ] **Step 4: Run test to verify pass**

Run the same pytest command. Expected: PASS.

- [ ] **Step 5: Update yaml config**

Modify `src/lerobot/configs/robot/jz_robot_udp_three_rtsp.yaml` with explicit safe command defaults:

```yaml
command_target_ip: 192.168.1.81
command_target_port: 39020
send_action_transport: local
send_action_execution: dry_run
command_robot: robot1
command_timeout_s: 0.2
```

## Task 2: Implement Command Packet Protocol

**Files:**
- Modify: `src/lerobot/robots/jz_robot_udp/protocol.py`
- Test: `tests/robots/test_jz_robot_udp.py`

- [ ] **Step 1: Write failing tests for command encode/decode**

Add tests for:

```text
valid command packet round trip
wrong type rejected
missing actions rejected
missing stamp_ns rejected
invalid gripper fields rejected
```

Example:

```python
def test_command_packet_round_trip():
    packet = make_jz_robot_udp_command_packet(
        robot="robot1",
        seq=1,
        stamp_ns=123,
        mode="dry_run",
        actions={
            "left": {f"left_joint{i}": float(i) for i in range(1, 8)},
            "right": {f"right_joint{i}": float(i) for i in range(1, 8)},
            "grippers": {
                "left": {"width": 0.1, "force": 1.0},
                "right": {"width": 0.2, "force": 1.0},
            },
        },
    )
    decoded = decode_jz_robot_udp_command_packet(encode_jz_robot_udp_command_packet(packet))
    assert decoded["type"] == "command"
    assert decoded["seq"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run --no-capture-output -n lerobot_flex python -m pytest -q tests/robots/test_jz_robot_udp.py -k command_packet
```

Expected: FAIL because helpers do not exist.

- [ ] **Step 3: Implement command protocol helpers**

Add helpers in `protocol.py`:

```text
make_jz_robot_udp_command_packet(...)
encode_jz_robot_udp_command_packet(...)
decode_jz_robot_udp_command_packet(...)
validate_jz_robot_udp_command_packet(...)
```

Keep JSON UTF-8 encoding. Keep validation strict and explicit.

- [ ] **Step 4: Run command protocol tests**

Run the same pytest command. Expected: PASS.

## Task 3: Add action_features and local dry-run send_action

**Files:**
- Modify: `src/lerobot/robots/jz_robot_udp/jz_robot_udp.py`
- Modify: `src/lerobot/robots/jz_robot_udp/udp_client.py` only if needed for shared seq helpers
- Test: `tests/robots/test_jz_robot_udp.py`

- [ ] **Step 1: Write failing tests for action_features**

Assert `action_features` contains:

```text
left_left_joint1.pos ... left_left_joint7.pos
right_right_joint1.pos ... right_right_joint7.pos
left_gripper.width
left_gripper.force
right_gripper.width
right_gripper.force
```

Use the same key style as current observation output unless the technical document is updated.

- [ ] **Step 2: Write failing test for local dry-run send_action**

The test should instantiate `JZRobotUDP` with local dry-run and call `send_action()` with a full action dict. It should assert:

```text
no exception
returned action equals/safely mirrors input
no UDP sender socket is required
internal command seq increments
```

- [ ] **Step 3: Run tests to verify failure**

Run:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run --no-capture-output -n lerobot_flex python -m pytest -q tests/robots/test_jz_robot_udp.py -k "action_features or local_dry_run"
```

Expected: FAIL because `send_action()` raises `NotImplementedError`.

- [ ] **Step 4: Implement action_features and local dry-run**

Implement:

```text
action_features property returns non-empty action schema
send_action validates keys
send_action builds command packet
send_action local_dry_run logs and returns action
send_action does not send UDP in local_dry_run
```

Do not implement active control.

- [ ] **Step 5: Run tests**

Run the same pytest command. Expected: PASS.

## Task 4: Implement UDP dry-run send path on x86

**Files:**
- Modify: `src/lerobot/robots/jz_robot_udp/udp_client.py`
- Modify: `src/lerobot/robots/jz_robot_udp/jz_robot_udp.py`
- Test: `tests/robots/test_jz_robot_udp.py`

- [ ] **Step 1: Write failing UDP send test**

Create a local UDP socket receiver in the test. Configure robot:

```text
command_target_ip=127.0.0.1
command_target_port=<test port>
send_action_transport=udp
send_action_execution=dry_run
```

Call `send_action()` and assert receiver gets one valid command packet with `mode=dry_run`.

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run --no-capture-output -n lerobot_flex python -m pytest -q tests/robots/test_jz_robot_udp.py -k udp_dry_run
```

Expected: FAIL because UDP send path is not implemented.

- [ ] **Step 3: Implement UDP command sender**

Keep command send code separate from state receiver:

```text
state socket: bind/recvfrom on x86 state_port
command socket: sendto command_target_ip:command_target_port
```

`send_action_execution=dry_run` must be the only supported Phase 2 execution mode. Do not add active robot control.

- [ ] **Step 4: Run UDP dry-run tests**

Run same pytest command. Expected: PASS.

## Task 5: Add Orin command receiver dry-run script

**Files:**
- Create: `udp_test/test_scripts/arm_side/orin_udp_command_receiver.py`
- Test: `tests/robots/test_jz_robot_udp.py` or a focused script compile test

- [ ] **Step 1: Write script behavior checklist**

Script must:

```text
bind --bind-ip 192.168.1.81 --port 39020
accept only --allowed-sender-ip 192.168.1.106
decode command packet
reject invalid packet
print received seq, bytes, action counts
print DRY_RUN only / NOT publishing ROS command topics on startup
handle SIGINT/SIGTERM cleanly
use standard library socket; do not import rclpy
never import ROS control publishers
never call create_publisher
```

- [ ] **Step 2: Add an AST/source safety test**

Add a test that parses `orin_udp_command_receiver.py` with `ast` and rejects executable code that:

```text
imports rclpy
imports known ROS command message modules
accesses an attribute named create_publisher
calls an attribute named publish
references cmd_vel outside string literals
```

Do not ban the string `NOT publishing ROS command topics`; safety warnings in log strings are allowed and required.

- [ ] **Step 3: Implement receiver**

Use standard library `socket`, `signal`, and protocol decoder from `src/lerobot/robots/jz_robot_udp/protocol.py`.

- [ ] **Step 4: Run py_compile**

Run:

```bash
PYTHONPATH=src conda run --no-capture-output -n lerobot_flex python -m py_compile udp_test/test_scripts/arm_side/orin_udp_command_receiver.py
```

Expected: PASS.

## Task 6: Add x86 send_action check script

**Files:**
- Create: `udp_test/test_scripts/x86_side/x86_jz_robot_udp_send_action_check.py`

- [ ] **Step 1: Implement CLI**

Arguments:

```text
--robot-config
--count
--hz
--transport local|udp
--execution dry_run
--command-only
--skip-cameras
--command-target-ip
--command-target-port
--print-every
```

- [ ] **Step 2: Ensure it is command-only safe**

The default check mode should be `--command-only`. In that mode, it must test the command protocol and `send_action()` path without requiring RTSP cameras or a fresh UDP state packet. This is necessary because current `JZRobotUDP.connect()` waits for state and connects cameras.

If the script supports full Robot connection mode, it may call:

```text
robot.connect()
robot.get_observation()
robot.send_action()
robot.disconnect()
```

Full Robot connection mode must document that a fake state sender or real Orin state bridge must already be running, and camera checks must be explicitly skippable.

It must print clear startup text:

```text
PHASE2 COMMAND DRY-RUN ONLY
This script does not enable robot execution.
```

- [ ] **Step 3: Run py_compile**

Run:

```bash
PYTHONPATH=src conda run --no-capture-output -n lerobot_flex python -m py_compile udp_test/test_scripts/x86_side/x86_jz_robot_udp_send_action_check.py
```

Expected: PASS.

## Task 7: Add server_bash wrappers for command dry-run

**Files:**
- Create: `udp_test/server_bash/orin_arm/start_command_receiver.sh`
- Create: `udp_test/server_bash/orin_arm/stop_command_receiver.sh`
- Modify: `udp_test/server_bash/orin_arm/status.sh`
- Optionally create x86 wrappers if needed
- Modify: `udp_test/server_bash/README.md`

- [ ] **Step 1: Add Orin start script**

Defaults:

```bash
ORIN_IP="${ORIN_IP:-192.168.1.81}"
X86_IP="${X86_IP:-192.168.1.106}"
COMMAND_PORT="${COMMAND_PORT:-39020}"
PYTHON_CMD="${PYTHON_CMD:-conda run --no-capture-output -n lerobot python}"
AUTO_TAIL="${AUTO_TAIL:-1}"
```

For x86-side scripts keep using `lerobot_flex`. For Orin command receiver, default to the ARM-side `lerobot` env because this receiver does not import ROS2/rclpy and the current ARM machine does not have `lerobot_flex`.

It must print:

```text
PHASE2 COMMAND DRY-RUN ONLY
NOT publishing ROS command topics
```

- [ ] **Step 2: Add Orin stop script**

Stop by pid file, then fallback `pgrep -af orin_udp_command_receiver.py`.

- [ ] **Step 3: Update status script**

Show both:

```text
ros_state_udp_bridge.py
orin_udp_command_receiver.py
```

- [ ] **Step 4: Run bash syntax check**

Run:

```bash
bash -n udp_test/server_bash/orin_arm/start_command_receiver.sh udp_test/server_bash/orin_arm/stop_command_receiver.sh udp_test/server_bash/orin_arm/status.sh
```

Expected: PASS.

## Task 8: Run local integration test

**Files:**
- No required file edits unless tests reveal issues.

- [ ] **Step 1: Start local command receiver**

On one terminal:

```bash
PYTHONPATH=src conda run --no-capture-output -n lerobot_flex python udp_test/test_scripts/arm_side/orin_udp_command_receiver.py \
  --bind-ip 127.0.0.1 \
  --port 39020 \
  --allowed-sender-ip 127.0.0.1 \
  --count 20
```

- [ ] **Step 2: Run x86 send_action check**

On another terminal:

```bash
PYTHONPATH=src conda run --no-capture-output -n lerobot_flex python udp_test/test_scripts/x86_side/x86_jz_robot_udp_send_action_check.py \
  --robot-config src/lerobot/configs/robot/jz_robot_udp_three_rtsp.yaml \
  --transport udp \
  --execution dry_run \
  --command-only \
  --command-target-ip 127.0.0.1 \
  --command-target-port 39020 \
  --count 20 \
  --hz 5
```

Expected:

```text
x86: SUMMARY: PASS commands=20
receiver: received=20 invalid=0 unexpected_sender=0
```

- [ ] **Step 3: Optional full Robot connection local test**

Only run this if the x86 check script implements a non-command-only mode. Start a fake state sender first and skip cameras:

```bash
PYTHONPATH=src conda run --no-capture-output -n lerobot_flex python udp_test/test_scripts/arm_side/orin_udp_state_sender.py \
  --bind-ip 127.0.0.1 \
  --target-ip 127.0.0.1 \
  --target-port 39010 \
  --hz 20 \
  --count 200 \
  --schema jz_robot_udp
```

Then run the x86 check with:

```bash
PYTHONPATH=src conda run --no-capture-output -n lerobot_flex python udp_test/test_scripts/x86_side/x86_jz_robot_udp_send_action_check.py \
  --robot-config src/lerobot/configs/robot/jz_robot_udp_three_rtsp.yaml \
  --transport udp \
  --execution dry_run \
  --command-target-ip 127.0.0.1 \
  --command-target-port 39020 \
  --allowed-sender-ip 127.0.0.1 \
  --skip-cameras \
  --count 20 \
  --hz 5
```

## Task 9: Test lerobot-record dry-run path

**Files:**
- Create: `udp_test/test_scripts/x86_side/x86_constant_action_teleop.py` only if no existing safe teleop/policy can generate fixed actions without hardware.
- Possibly modify docs with the exact command used.

- [ ] **Step 1: Provide a safe action source**

Do not assume record can run without teleop/policy. `lerobot-record` requires one of them. Phase 2 must use one of these safe choices:

```text
Preferred: a constant-action teleop/policy that generates fixed zero/small actions and does not connect to hardware.
Fallback: a new test-only x86_constant_action_teleop.py registered only for dry-run record tests.
```

The action source must not open robot hardware, ROS command topics, serial ports, or joystick devices.

- [ ] **Step 2: Run a short dry-run record**

Command template:

```bash
PYTHONPATH=src conda run --no-capture-output -n lerobot_flex lerobot-record \
  --robot.type=jz_robot_udp \
  --robot.send_action_transport=udp \
  --robot.send_action_execution=dry_run \
  --robot.command_target_ip=192.168.1.81 \
  --robot.command_target_port=39020 \
  --dataset.repo_id=local/jz_robot_udp_phase2_dry_run \
  --dataset.root=tests/outputs/jz_robot_udp_phase2_dry_run \
  --dataset.num_episodes=1 \
  --dataset.episode_time_s=10 \
  --dataset.reset_time_s=0 \
  --dataset.fps=5 \
  --dataset.single_task="phase2 dry-run record chain test" \
  --dataset.push_to_hub=false \
  --teleop.type=<safe_constant_action_teleop>
```

Expected:

```text
record does not fail with NotImplementedError
dataset saves one episode
Orin receiver logs command seq
robot does not move
```

This dataset is only a flow artifact. It is not a final trainable demonstration dataset and must not be pushed to the Hub.

- [ ] **Step 3: Inspect dataset features**

Confirm dataset includes:

```text
observation state fields
three image fields
action fields
```

## Task 10: Run Phase 1 Regression

**Files:**
- No required file edits unless regression fails.

- [ ] **Step 1: Run observation regression on x86 + Orin**

x86:

```bash
OBS_COUNT=600 bash udp_test/server_bash/x86/start.sh
```

Orin:

```bash
bash udp_test/server_bash/orin_arm/start.sh
```

Expected:

```text
SUMMARY: PASS observations=600
keys=21 numeric=18
no stale_count / TimeoutError / traceback
```

- [ ] **Step 2: Stop processes**

Run:

```bash
bash udp_test/server_bash/x86/stop.sh
bash udp_test/server_bash/orin_arm/stop.sh
bash udp_test/server_bash/orin_arm/stop_command_receiver.sh
```

- [ ] **Step 3: Check status**

Run:

```bash
bash udp_test/server_bash/x86/status.sh
bash udp_test/server_bash/orin_arm/status.sh
```

Expected: no residual relevant processes.

## Task 11: Update Documentation

**Files:**
- Modify: `udp_test/docs/phase2/技术文档.md`
- Modify: `udp_test/server_bash/README.md`
- Optionally modify: `udp_test/test_scripts/README.md`

- [ ] **Step 1: Document command dry-run startup**

Add commands for:

```text
Orin command receiver dry-run
x86 send_action check
local 127.0.0.1 dry-run test
lerobot-record dry-run caveat
```

- [ ] **Step 2: Document safety boundary**

State clearly:

```text
Phase 2 does not publish ROS command topics.
Phase 2 does not move the robot.
Human emergency stop is a final fallback, not the primary software safety mechanism.
```

- [ ] **Step 3: Document acceptance result after testing**

After tests are run, append exact summaries:

```text
commands=N invalid=0
lerobot-record dry-run PASS
OBS_COUNT=600 regression PASS
```

## Task 12: Final Verification

### Local verification

Run:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run --no-capture-output -n lerobot_flex python -m pytest -q tests/robots/test_jz_robot_udp.py

PYTHONPATH=src conda run --no-capture-output -n lerobot_flex python -m py_compile \
  src/lerobot/robots/jz_robot_udp/*.py \
  udp_test/test_scripts/arm_side/orin_udp_command_receiver.py \
  udp_test/test_scripts/x86_side/x86_jz_robot_udp_send_action_check.py

bash -n \
  udp_test/server_bash/orin_arm/start_command_receiver.sh \
  udp_test/server_bash/orin_arm/stop_command_receiver.sh \
  udp_test/server_bash/orin_arm/status.sh
```

Expected:

```text
pytest PASS
py_compile PASS
bash -n PASS
```

### Hardware/network acceptance

Run only on the correct machines and only with Phase 2 dry-run defaults. This still must not move the robot.

x86:

```bash
OBS_COUNT=600 bash udp_test/server_bash/x86/start.sh
```

Orin:

```bash
bash udp_test/server_bash/orin_arm/start.sh
AUTO_TAIL=1 bash udp_test/server_bash/orin_arm/start_command_receiver.sh
```

Expected:

```text
OBS_COUNT=600 regression PASS
command dry-run receiver logs DRY_RUN / NOT publishing
no robot motion
```

Then run the `lerobot-record` dry-run command from Task 9 and verify:

```text
record dry-run completes one short episode
dataset is local only and not pushed to Hub
Orin receiver logs command seq
robot does not move
```

## Completion Criteria

Phase 2 is complete only when:

```text
1. Command protocol unit tests pass
2. send_action local dry-run tests pass
3. send_action UDP dry-run tests pass
4. Orin command receiver dry-run receives commands and does not publish ROS commands
5. lerobot-record dry-run short episode completes
6. Phase 1 OBS_COUNT=600 regression passes
7. Docs are updated with exact tested commands and outputs
8. Code/document review findings are handled, especially safety boundary and dry-run semantics
```
