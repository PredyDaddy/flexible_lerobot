# orin端重要指令
重启整个节点
```bash
# 重启整个ros2的节点
sudo systemctl restart  robot_bringup.service

ros2 topic list
```

如果restart还是不行，就使用下面的指令
```bash
sudo systemctl stop  robot_bringup.service
sudo systemctl start  robot_bringup.service
ros2 topic list
```

启动VR遥操
```bash
ros2 service call /robot1/enable_udp_receive std_srvs/srv/SetBool "{data: true}"
```

恢复预抓取姿态
```bash
ros2 service call /robot1/choreographer/execute multi_robot_choreographer_interfaces/srv/ExecuteChoreography "{choreography_name: 'VR_inital_no_waist'}"
```

## 数据采集流程

### orin端启动
1. 检查节点
```bash
ros2 topic list
```

如果没有，重启节点


2. 恢复位置启动遥操作
```bash
ros2 service call /robot1/choreographer/execute multi_robot_choreographer_interfaces/srv/ExecuteChoreography "{choreography_name: 'VR_inital_no_waist'}"

ros2 service call /robot1/enable_udp_receive std_srvs/srv/SetBool "{data: true}"
```

3. 启动udp服务(lerobot环境)
```bash
bash udp_test/server_bash/orin_arm/start.sh

PYTHONPATH=$PWD/src:$PYTHONPATH python3 \
    udp_test/test_scripts/arm_side/orin_ros_target_action_udp_bridge.py \
    --target-ip 192.168.1.106 \
    --target-port 39030 \
    --bind-ip 192.168.1.81
```


关闭全部东西
```bash
bash udp_test/server_bash/orin_arm/stop_all.sh
```

### x86端
```bash
bash record.sh
```


## 数据回放
### orin端
```bash
bash udp_test/server_bash/orin_arm/start.sh


CONFIG=udp_test/test_scripts/arm_side/orin_phase3_executor_config_armed_hold.yaml \
  EXECUTION=armed \
  JZ_UDP_EXECUTOR_ARMED=1 \
  AUTO_TAIL=1 \
  bash udp_test/server_bash/orin_arm/start_phase3_executor.sh
```

### x86
```bash
REPLAY_FPS=30   EXECUTION=armed   I_UNDERSTAND_REPLAY_MOVES_ROBOT=1   bash replay.sh
```