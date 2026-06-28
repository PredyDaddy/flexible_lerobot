# VR Topic Probe

This directory contains read-only tools for Phase 3 planning.

The goal is to observe the existing VR control program on Orin and identify which ROS topics it publishes when the robot moves.

## Safety Boundary

`probe_vr_topics.sh` only runs:

```text
ros2 topic list
ros2 topic info
ros2 topic hz
ros2 topic echo --once
```

It does not:

```text
publish ROS topics
send UDP commands
call send_action
create ROS publishers
control arm / gripper / base
```

## Usage

Terminal 1: run the VR program normally.

Terminal 2: run the read-only probe:

```bash
cd ~/workspace/flexible_lerobot
bash udp_test/vr_test/probe_vr_topics.sh
```

The script writes a timestamped log under:

```text
udp_test/vr_test/logs/
```

Optional environment variables:

```bash
HZ_DURATION_S=5 ECHO_TIMEOUT_S=3 bash udp_test/vr_test/probe_vr_topics.sh
```

If important topics are missed, override the keyword filter:

```bash
KEYWORDS='command|cmd|joint|gripper|telecon|arm|hand|left|right|vel|twist|control|target|goal|your_keyword' \
  bash udp_test/vr_test/probe_vr_topics.sh
```

After running, share the generated log so Phase 3 can map UDP command packets to the existing VR ROS command topics safely.
