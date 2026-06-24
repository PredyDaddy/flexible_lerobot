# 录制
打开机器人
```bash
ros2 service call /robot1/enable_udp_receive std_srvs/srv/SetBool "{data: true}"
```

# 我的录制
```bash
python -u -m my_devs.jz_robot.run_record_jz_three_realsense \
    --dataset-repo-id local/jz_pick_place_three_rs_cqy \
    --dataset-root /home/test/workspace/flexible_lerobot/data_collection/jz_pick_place_bottle \
    --dataset-task "place the bottle in the orange area" \
    --robot-id jz_dual_arm_rs \
    --use-gripper true \
    --init-state-timeout-s 0 \
    --state-timeout-s 0.2 \
    --use-external-commands true \
    --teleop-connect-timeout-s 0 \
    --num-episodes 98 \
    --episode-time-s 15 \
    --reset-time-s 5 \
    --fps 20 \
    --display-data false \
    --play-sounds true
```

# 我的回放
```bash
  python -u -m my_devs.jz_robot.run_replay_jz_three_realsense \
    --dataset-repo-id local/jz_pick_place_three_rs_cqy \
    --dataset-root /home/test/workspace/flexible_lerobot/data_collection/jz_pick_place_three_rs_run10 \
    --episode 0 \
    --robot-id jz_dual_arm_rs \
    --use-gripper true \
    --init-state-timeout-s 0 \
    --state-timeout-s 0.2 \
    --use-external-commands false \
    --connect-cameras false \
    --fps 20 \
    --play-sounds true
```