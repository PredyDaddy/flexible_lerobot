# Local ARM Readiness Checks

conda activate lerobot

这些脚本在 Orin / ARM 本机运行，用来确认本机 ROS 状态和录制输入是否可用。

当前只做只读检测：

```text
不发送 action
不 publish command topic
不控制机器人
```

## 一键只读检测

在 Orin 上运行：

```bash
cd /home/data/test/workspace/flexible_lerobot
python udp_test/local_test/local_robot_readiness_check.py
```

默认检测：

```text
1. 当前 record 使用的 JZRobot 配置能否解析
2. 双臂 joint_states topic 是否存在并有消息
3. 夹爪状态 topic 是否存在并有消息
4. 默认按 RTSP 检测三路相机
```

## 常用参数

只检测 ROS 状态，不检测相机：

```bash
python udp_test/local_test/local_robot_readiness_check.py --skip-cameras
```

相机只读 1 帧：

```bash
python udp_test/local_test/local_robot_readiness_check.py --camera-frames 1
```

当前相机已经改成 RTSP 流，默认直接检测 RTSP。显式写法：

```bash
python udp_test/local_test/local_robot_readiness_check.py --camera-source rtsp --camera-frames 1 --camera-timeout-ms 5000
```

默认 RTSP 地址：

```text
camera_head:  rtsp://192.168.1.81:8554/robot_camera/camera_head
camera_left:  rtsp://192.168.1.81:8554/robot_camera/camera_left
camera_right: rtsp://192.168.1.81:8554/robot_camera/camera_right
```

`192.168.1.81` 是当前 Orin 网线口 `lan2` 的 IP。x86 访问同一个 RTSP 服务时，也使用这三个地址。

## 抓三张相机图片

在 Orin 上运行：

```bash
cd /home/data/test/workspace/flexible_lerobot
python udp_test/local_test/get_pic.py
```

脚本会自动覆盖：

```text
udp_test/local_test/output_media/camera_head.jpg
udp_test/local_test/output_media/camera_left.jpg
udp_test/local_test/output_media/camera_right.jpg
```

同时检测配置相机和 RTSP：

```bash
python udp_test/local_test/local_robot_readiness_check.py --camera-source both --camera-frames 1
```

旧 ROS2 image topic 相机配置检测：

```bash
python udp_test/local_test/local_robot_readiness_check.py --camera-source config --camera-frames 1
```

如果当前系统已切到 RTSP，旧 ROS2 image topic 检测失败是预期现象。

增加等待时间：

```bash
python udp_test/local_test/local_robot_readiness_check.py --timeout-s 10
```

使用其他 robot config：

```bash
python udp_test/local_test/local_robot_readiness_check.py --robot-config <path/to/robot.yaml>
```

## 输出怎么看

```text
[PASS] 表示该项可用
[WARN] 表示不是硬失败，但需要注意
[FAIL] 表示该项不可用
```

最终看到：

```text
SUMMARY: PASS
```

说明 ARM 本机基础输入可用，后续再考虑 x86 UDP 通信。
