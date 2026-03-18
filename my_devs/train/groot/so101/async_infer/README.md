# 启动服务
```bash
python my_devs/train/groot/so101/async_infer/run_groot_async_policy_server.py     --host 127.0.0.1     --port 8080     --fps 30
```

# 推理服务
```bash
python my_devs/train/groot/so101/async_infer/run_groot_async_robot_client.py \
      --server-address 127.0.0.1:8080 \
      --robot-port /dev/ttyACM0 \
      --top-cam-index 4 \
      --wrist-cam-index 6 \
      --task "Put the block in the bin" \
      --policy-path /data/cqy_workspace/flexible_lerobot/outputs/train/groot_grasp_block_in_bin1_repro_20260302_223413/bs32_20260302_223447/checkpoints/last/pretrained_model \
      --actions-per-chunk 16 \
      --run-time-s 120 \
      --robot-connect-retries 5 \
      --robot-connect-retry-delay-s 2.0
```