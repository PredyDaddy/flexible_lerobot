# UDP Readonly Bridge Architecture

本阶段目标：先验证 Orin 和 x86 之间的 UDP 通信链路，只做状态读取，不发送机器人控制命令。

## Current Network

```text
+-------------------------------+                  +-------------------------------+
| Orin AGX / ARM                |                  | x86 Laptop / Ubuntu 24.04     |
|                               |                  |                               |
| lan2                          |   Ethernet       | wired NIC                     |
| 192.168.1.81/24               +------------------+ 192.168.1.x/24                |
| MTU 1500                      |                  |                               |
+-------------------------------+                  +-------------------------------+
```

```text
Orin selected interface:

  lan2
    ip:        192.168.1.81
    netmask:   255.255.255.0
    broadcast: 192.168.1.255
```

## System Boundary

```text
                         Orin AGX
                  keeps ROS and robot I/O

        +--------------------------------------+
        |                                      |
        |  Robot drivers / hardware access     |
        |                  |                   |
        |                  v                   |
        |        +-------------------+         |
        |        | ROS2 local graph   |         |
        |        | /robot1/...        |         |
        |        +---------+---------+         |
        |                  |                   |
        |                  v                   |
        |        +-------------------+         |
        |        | UDP bridge         |         |
        |        | readonly phase     |         |
        |        +---------+---------+         |
        |                  |                   |
        +------------------+-------------------+
                           |
                           | UDP over Ethernet
                           v
        +------------------+-------------------+
        |                  |                   |
        |        +---------+---------+         |
        |        | UDP receiver       |         |
        |        | no ROS required    |         |
        |        +---------+---------+         |
        |                  |                   |
        |                  v                   |
        |        +-------------------+         |
        |        | lerobot / logs /   |         |
        |        | future inference   |         |
        |        +-------------------+         |
        |                                      |
        +--------------------------------------+

                         x86 Laptop
                  owns heavy computation later
```

## Phase 1: UDP Link Test

```text
Purpose:
  prove UDP can travel both ways between x86 and Orin.


        x86                                           Orin
  +---------------+                             +---------------+
  | udp client    |                             | udp server    |
  +-------+-------+                             +-------+-------+
          |                                             |
          |  ping(seq, timestamp)                       |
          +-------------------------------------------->|
          |                                             |
          |  pong(seq, timestamp)                       |
          |<--------------------------------------------+
          |                                             |
  +-------v-------+                             +-------v-------+
  | print latency |                             | print sender  |
  +---------------+                             +---------------+
```

No ROS dependency in this phase.

## Phase 2: Readonly State Stream

```text
Purpose:
  stream state-like data from Orin to x86 without controlling the robot.


        Orin                                           x86
  +---------------+                             +----------------+
  | state sender  |                             | state receiver |
  +-------+-------+                             +--------+-------+
          |                                              |
          |  state(seq, timestamp, payload)              |
          +--------------------------------------------->|
          |                                              |
          |  state(seq, timestamp, payload)              |
          +--------------------------------------------->|
          |                                              |
          |  state(seq, timestamp, payload)              |
          +--------------------------------------------->|
          |                                              |
  +-------v-------+                             +--------v-------+
  | low CPU load  |                             | packet stats   |
  +---------------+                             +----------------+
```

In the first readonly test, the payload can be fake state. After the UDP path is stable, the payload can be replaced with selected ROS topic data.

## Phase 3: ROS Readonly Bridge

```text
Purpose:
  read selected ROS topics on Orin and publish snapshots to x86 over UDP.


        Orin ROS2 graph
  +---------------------------+
  | /robot1/arm_left/...      |
  | /robot1/arm_right/...     |
  | /robot1/body/...          |
  | /robot1/gripper/...       |
  +-------------+-------------+
                |
                | subscribe only
                v
  +-------------+-------------+
  | ROS -> UDP snapshot node  |
  | no control publishers     |
  +-------------+-------------+
                |
                | UDP state packets
                v
        x86 UDP receiver
  +-------------+-------------+
  | print / log / feed future |
  | external runtime          |
  +---------------------------+
```

Initial candidate topics:

```text
/robot1/arm_left/joint_states
/robot1/arm_right/joint_states
/robot1/body/joint_states
/robot1/left_gripper/gripper_status
/robot1/right_gripper/gripper_status
/robot1/agv/connected
/robot1/agv/has_control
/robot1/agv/battery
```

## Later: Control Closed Loop

This is not part of the first test.

```text
        Orin                                           x86
  +---------------+                             +----------------+
  | ROS topics    |                             | lerobot        |
  | robot I/O     |                             | policy         |
  +-------+-------+                             +--------+-------+
          |                                              ^
          | state                                        |
          +--------------------------------------------->|
          |                                              |
          | action command                               |
          |<---------------------------------------------+
          |                                              |
  +-------v-------+                             +--------v-------+
  | safety gate   |                             | compute side   |
  | timeout       |                             | no ROS needed  |
  | limits        |                             |                |
  +---------------+                             +----------------+
```

Before this phase, Orin must keep the final safety decision:

```text
x86 timeout        -> stop accepting commands
invalid command    -> reject
robot disconnected -> reject
read only mode     -> never publish command topics
```

## Readonly State Machine

```text
  +-------------+
  | START       |
  +------+------+ 
         |
         v
  +------+------+
  | BIND UDP    |
  | 192.168.1.81|
  +------+------+
         |
         v
  +------+------+
  | WAIT X86    |
  | optional    |
  +------+------+
         |
         v
  +------+------+
  | SEND STATE  |
  | readonly    |
  +------+------+
         |
         v
  +------+------+
  | LOG STATS   |
  +------+------+
         |
         v
  +------+------+
  | STOP        |
  +-------------+
```

## Packet Direction

```text
Phase 1:
  x86  -> Orin    UDP ping
  Orin -> x86     UDP pong

Phase 2:
  Orin -> x86     UDP state stream

Phase 3:
  ROS topic -> Orin bridge -> UDP -> x86 receiver

Not in current phase:
  x86 -> Orin -> ROS command topic
```

## First Success Criteria

```text
[ ] x86 can reach 192.168.1.81 by ping
[ ] x86 receives UDP packets from Orin
[ ] Orin receives UDP packets from x86
[ ] receiver prints seq and packet rate
[ ] packet loss can be estimated by seq gap
[ ] no robot control command is sent
```
